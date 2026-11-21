#!/usr/bin/env python3
"""Offline GPU jobs over SSH; only containers owned by this tool may be stopped.

Never stop unrelated containers or use broad process-name matching in SSH shells.
The host needs Python 3, Docker/NVIDIA, rsync and the prepared project .venv.
"""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import uuid

IMAGE = "pytorch/pytorch:2.11.0-cuda13.0-cudnn9-devel"
OWNER = "rawdenoise.managed=gpu_jobs"
ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}\Z")
ROOT = Path(globals().get("__file__", "scripts/gpu_jobs.py")).resolve().parents[1]


def checked_id(value):
    if not ID_RE.fullmatch(value):
        raise ValueError("Invalid job ID/name")
    return value


def run_dir(value):
    path = PurePosixPath(value)
    if not re.fullmatch(r"[A-Za-z0-9_./-]+", value) or path.is_absolute() or ".." in path.parts or len(path.parts) < 2 or path.parts[0] != "runs" or path.parts[1] == "_jobs":
        raise ValueError("Run directory must be below runs/, outside runs/_jobs")
    return str(path)


def safe_path(root, relative, top_symlink_ok=False):
    """Resolve a project-relative path, refusing traversal and symlinks.

    top_symlink_ok permits only the first component (the local runs/ folder,
    which may live on another volume) to be a symlink; everything below it must
    still be real and stay inside that folder.
    """
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError("Unsafe path/traversal")
    current, anchor = root, root
    for index, part in enumerate(path.parts):
        current = current / part
        if current.is_symlink():
            if not (top_symlink_ok and index == 0):
                raise ValueError("Symlink refused: " + str(relative))
            anchor = current
    if not current.resolve().is_relative_to(anchor.resolve()):
        raise ValueError("Path escapes root")
    return current


def file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"size": path.stat().st_size, "sha256": digest.hexdigest()}


def atomic_json(path, value):
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=path.name + ".", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(json.dumps(value, indent=2) + "\n")
            stream.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def manifest(root, directories, include_steps=False, hash_files=True):
    result = {}
    for directory in directories:
        base = safe_path(root, directory)
        if not base.is_dir():
            raise ValueError("Missing directory: " + directory)
        for path in sorted(base.rglob("*")):
            relative = path.relative_to(root).as_posix()
            safe_path(root, relative)
            if path.is_dir():
                continue
            if not path.is_file():
                raise ValueError("Not a regular file: " + relative)
            if path.name == ".collected" or (not include_steps and re.fullmatch(r"step_.*\.pt", path.name)):
                continue
            result[relative] = file_digest(path) if hash_files else {"size": path.stat().st_size}
    return result


def command_option(command, key):
    for index, value in enumerate(command):
        if value == key:
            if index + 1 == len(command):
                raise ValueError("Missing value for " + key)
            return command[index + 1]
        if value.startswith(key + "="):
            return value.split("=", 1)[1]
    return None


def checkpoint_budget(root, payload):
    # Read only block-style training scalars, without importing YAML on the host.
    if payload.get("no_checkpoints", False):
        # Evaluation/utility jobs write no checkpoints; only min_free_gb applies.
        return {"checkpoint_bytes": 0, "retained_intermediates": 0, "storage_bytes": 0,
                "estimate_source": "no checkpoints declared"}
    command = payload["command"]
    config_path = command_option(command, "--config")
    if not config_path:
        raise ValueError("Storage budgeting requires --config in the training command")
    text = safe_path(root, config_path).read_text()
    block = re.search(r"^training:\s*\n((?:[ \t]+[^\n]*\n|\n)*)", text + "\n", re.M)
    if not block:
        raise ValueError("Storage budgeting requires a block-style training config")
    lines = [line for line in block[1].splitlines() if line.strip() and not line.lstrip().startswith("#")]
    indent = min((len(line) - len(line.lstrip()) for line in lines), default=2)
    values = {}
    for line in lines:
        if len(line) - len(line.lstrip()) != indent:
            continue
        key, _, value = line.strip().partition(":")
        if key == "<<":
            raise ValueError("Storage budgeting does not support merged training mappings")
        if key in {"steps", "checkpoint_every", "keep_checkpoints"}:
            value = value.partition("#")[0].strip()
            if not re.fullmatch(r"[0-9]+", value) or key in values:
                raise ValueError("Storage budgeting requires unique nonnegative integer training scalars")
            values[key] = value
    steps = int(command_option(command, "--steps") or values.get("steps", "0"))
    every = int(values.get("checkpoint_every", "0"))
    keep = int(values.get("keep_checkpoints", "0"))
    if steps <= 0:
        raise ValueError("Storage budgeting requires positive training steps")
    # Include the final save, last.pt and one temporary/copy-in-progress checkpoint.
    saves = steps // every + bool(steps % every) if every else 1
    retained = min(keep, saves)
    size_mb = payload.get("checkpoint_size_mb")
    if size_mb is not None:
        if not math.isfinite(size_mb) or size_mb <= 0:
            raise ValueError("Checkpoint size estimate must be positive")
        size = math.ceil(size_mb * 1024**2)
        source = "explicit"
    else:
        initial = command_option(command, "--resume") or command_option(command, "--init-checkpoint")
        if not initial:
            raise ValueError("Supply --checkpoint-size-mb or a training init/resume checkpoint")
        # Conservative weights-only to model + optimizer + EMA allowance.
        size = safe_path(root, initial).stat().st_size * 6
        source = "init/resume bytes x 6"
        if size <= 0:
            raise ValueError("Empty init/resume checkpoint")
    return {"checkpoint_bytes": size, "retained_intermediates": retained,
            "storage_bytes": (retained + 2) * size, "estimate_source": source}


def overlaps(left, right):
    a, b = PurePosixPath(left), PurePosixPath(right)
    return a == b or a in b.parents or b in a.parents


def execute(command, **kwargs):
    return subprocess.run(command, check=True, text=True, **kwargs)


def config(root=ROOT):
    # Deliberately support only flat YAML scalar keys, without a YAML dependency.
    values = {"project": "rawdenoise-lab", "min_free_gb": "10"}
    path = root / "configs/gpu_host.yaml"
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, sep, value = line.partition(":")
            if not sep or key not in {"host", "user", "project", "min_free_gb"}:
                raise ValueError("gpu_host.yaml accepts flat host/user/project/min_free_gb scalars only")
            values[key] = value.strip().strip("\"'")
    for key in ("host", "user", "project", "min_free_gb"):
        values[key] = os.environ.get("RAWDENOISE_GPU_" + key.upper(), values.get(key, ""))
    if not values["host"] or values["host"].startswith("-") or not re.fullmatch(r"[\w.@-]+", values["host"]):
        raise ValueError("Set RAWDENOISE_GPU_HOST or configs/gpu_host.yaml host")
    if values["user"] and not re.fullmatch(r"[\w.-]+", values["user"]):
        raise ValueError("Invalid SSH user")
    # A home-relative or absolute path, kept free of rsync remote-shell syntax.
    if not re.fullmatch(r"/?[\w./-]+", values["project"]) or ".." in PurePosixPath(values["project"]).parts:
        raise ValueError("project must be a simple absolute or home-relative path")
    values["min_free_gb"] = float(values["min_free_gb"])
    if not math.isfinite(values["min_free_gb"]) or values["min_free_gb"] <= 0:
        raise ValueError("min_free_gb must be positive")
    return values


def target(cfg):
    return (cfg["user"] + "@" if cfg["user"] else "") + cfg["host"]


def remote(cfg, action, **args):
    payload = dict(cfg, action=action, **args)
    command = ["python3", "-c", Path(__file__).read_text(), "--_remote", json.dumps(payload)]
    result = execute(["ssh", "-o", "BatchMode=yes", target(cfg), shlex.join(command)], capture_output=True)
    return json.loads(result.stdout)


def remote_main(payload):
    """Runs on the Linux host. No shell commands or process-name matching."""
    root = Path(payload["project"]).expanduser()
    if not root.is_absolute():
        root = Path.home() / root
    root = root.resolve()
    jobs = safe_path(root, "runs/_jobs")

    def docker(*args):
        return execute(["docker", *args], capture_output=True).stdout

    def containers():
        ids = docker("ps", "-aq", "--filter", "label=" + OWNER).split()
        return json.loads(docker("inspect", *ids)) if ids else []

    def owned(container, job_id):
        labels = container["Config"].get("Labels") or {}
        return labels.get("rawdenoise.managed") == "gpu_jobs" and labels.get("rawdenoise.job") == job_id

    def snapshot():
        found = containers()
        records = []
        for path in sorted(jobs.glob("*/cmd.json")):
            safe_path(root, path.relative_to(root).as_posix())
            record = json.loads(path.read_text())
            checked_id(record["id"])
            matches = [c for c in found if owned(c, record["id"])]
            record["collected"] = safe_path(root, (path.parent / ".collected").relative_to(root).as_posix()).is_file()
            record["state"] = matches[0]["State"]["Status"] if matches else record.get("state", "unknown")
            if matches:
                record["exit_code"] = matches[0]["State"]["ExitCode"]
            log = safe_path(root, (path.parent / "stdout.log").relative_to(root).as_posix())
            if log.exists():
                with log.open("rb") as stream:
                    stream.seek(max(0, log.stat().st_size - 8192))
                    record["tail"] = stream.read().decode(errors="replace").splitlines()[-5:]
            records.append(record)
        known = {r["id"] for r in records}
        for container in found:
            job_id = (container["Config"].get("Labels") or {}).get("rawdenoise.job", "")
            if job_id not in known:
                records.append({"id": job_id, "state": container["State"]["Status"], "metadata_missing": True})
        return records

    action = payload["action"]
    if action == "submit":
        import fcntl
        jobs.mkdir(parents=True, exist_ok=True)
        with safe_path(root, "runs/_jobs/.submit.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            budget = checkpoint_budget(root, payload)
            records = snapshot()
            reserved_bytes = sum(r.get("storage_bytes", 0) for r in records if not finished(r))
            free = shutil.disk_usage(root).free
            if not payload.get("allow_low_space", False) and free - reserved_bytes - budget["storage_bytes"] < payload["min_free_gb"] * 1024**3:
                raise ValueError("Insufficient free disk space after checkpoint reservations")
            gpu = payload["gpu"]
            rows = execute(["nvidia-smi", "--query-gpu=index,uuid,utilization.gpu,memory.used", "--format=csv,noheader,nounits"], capture_output=True).stdout.splitlines()
            selected = [r.split(",") for r in rows if r.split(",")[0].strip() == str(gpu)]
            if len(selected) != 1:
                raise ValueError("GPU index not found")
            _, gpu_uuid, util, memory = [v.strip() for v in selected[0]]
            apps = execute(["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader,nounits"], capture_output=True).stdout.splitlines()
            reserved = any(c["State"].get("Running") and (c["Config"].get("Labels") or {}).get("rawdenoise.gpu") == str(gpu) for c in containers())
            if not payload["force"] and (float(util) > 0 or float(memory) > 256 or gpu_uuid in [v.strip() for v in apps] or reserved):
                raise ValueError("GPU is busy; use --force only after checking status")
            job_id = checked_id(payload["id"])
            output = run_dir(payload["run_dir"])
            output_path = safe_path(root, output)
            related = [r for r in records if r.get("run_dir") and overlaps(r["run_dir"], output)]
            if any(not finished(r) for r in related):
                raise ValueError("Run directory is reserved by an active or unknown job")
            reused = any(not r.get("collected") for r in related) or (output_path.exists() and any(output_path.iterdir()))
            if reused and not payload.get("resume", False):
                raise ValueError("Run directory already used; validated --resume required")
            if payload.get("resume", False):
                initial = command_option(payload["command"], "--resume")
                if not initial or not overlaps(output, initial) or not PurePosixPath(initial).is_relative_to(PurePosixPath(output)) or not safe_path(root, initial).is_file():
                    raise ValueError("--resume requires a training --resume checkpoint inside the run directory")
            if not (root / ".venv/pyvenv.cfg").is_file():
                raise ValueError("Prepare the host project .venv first")
            output_path.mkdir(parents=True, exist_ok=True)
            directory = safe_path(root, "runs/_jobs/" + job_id)
            directory.mkdir()
            record = {key: payload[key] for key in ("id", "name", "gpu", "command", "run_dir")}
            record.update(budget)
            record["created"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            record["state"] = "launch_failed"
            (directory / "cmd.json").write_text(json.dumps(record, indent=2) + "\n")
            (directory / "stdout.log").touch()
            # Redirect inside the container, then exec Python so Docker stop reaches it.
            wrapper = "import os,sys; f=os.open(sys.argv[1],os.O_WRONLY|os.O_APPEND); os.dup2(f,1); os.dup2(f,2); os.execv(sys.executable,[sys.executable,'-u',*sys.argv[2:]])"
            command = ["run", "-d", "--name", "rawdenoise-" + job_id, "--label", OWNER,
                       "--label", "rawdenoise.job=" + job_id, "--label", "rawdenoise.gpu=" + str(gpu),
                       "--pull", "never", "--network", "none", "--gpus", "device=" + str(gpu),
                       "--user", f"{os.getuid()}:{os.getgid()}", "--shm-size", "1g",
                       "-e", "HOME=/workspace", "-e", "PYTHONPATH=/workspace/src",
                       "-e", "RAWDENOISE_RUN_DIR=" + output, "-v", str(root) + ":/workspace",
                       "-w", "/workspace", "--entrypoint", "/workspace/.venv/bin/python", IMAGE,
                       "-u", "-c", wrapper, "/workspace/runs/_jobs/" + job_id + "/stdout.log", *payload["command"]]
            container_id = docker(*command).strip()
            record["state"] = "unknown"
            record["container"] = container_id
            (directory / "cmd.json").write_text(json.dumps(record, indent=2) + "\n")
            return record
    if action in {"prune-checkpoints", "manifest", "collected"}:
        import fcntl
        jobs.mkdir(parents=True, exist_ok=True)
        with safe_path(root, "runs/_jobs/.submit.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            records = snapshot()
            if action == "prune-checkpoints":
                outputs = [run_dir(value) for value in payload["run_dirs"]]
            else:
                job_id = checked_id(payload["id"])
                matches = [r for r in records if r["id"] == job_id]
                if len(matches) != 1 or not finished(matches[0]):
                    raise ValueError("Only finished jobs with metadata can be collected")
                outputs = [run_dir(matches[0]["run_dir"])]
            for output in outputs:
                related = [r for r in records if r.get("run_dir") and overlaps(r["run_dir"], output)]
                if not any(r.get("run_dir") == output for r in related) or any(not finished(r) for r in related):
                    raise ValueError("Run directory must belong to finished jobs only")
                safe_path(root, output)
            if action == "manifest":
                return {"files": manifest(root, [*outputs, "runs/_jobs/" + job_id], payload.get("include_steps", False))}
            if action == "collected":
                marker = safe_path(root, "runs/_jobs/" + job_id + "/.collected")
                temporary_marker = safe_path(root, marker.with_suffix(".tmp").relative_to(root).as_posix())
                temporary_marker.write_text("verified collection\n")
                temporary_marker.replace(marker)
                return {"collected": job_id}
            keep_latest = payload.get("keep_latest", 0)
            if keep_latest < 0:
                raise ValueError("--keep-latest must be nonnegative")
            protected = set()
            for value in payload.get("keep", []):
                # Basenames apply to each selected run; full paths must be inside one.
                if len(PurePosixPath(value).parts) == 1 and value not in {".", ".."}:
                    protected.update(str(PurePosixPath(output) / value) for output in outputs)
                else:
                    checked = run_dir(value)
                    if not any(PurePosixPath(checked).is_relative_to(PurePosixPath(o)) for o in outputs):
                        raise ValueError("--keep must be inside a selected run")
                    protected.add(checked)
            candidates = {}
            for output in outputs:
                base = safe_path(root, output)
                # Validate the whole tree before deleting anything, including kept symlinks.
                manifest(root, [output], True, hash_files=False)
                steps = [p for p in base.glob("step_*.pt") if p.is_file() and re.fullmatch(r"step_[0-9]+\.pt", p.name)]
                steps.sort(key=lambda p: (int(p.stem[5:]), p.name), reverse=True)
                for path in steps[keep_latest:]:
                    relative = path.relative_to(root).as_posix()
                    if relative not in protected:
                        candidates[relative] = path.stat().st_size
            if payload.get("apply", False):
                for relative in candidates:
                    safe_path(root, relative).unlink()
            return {"applied": payload.get("apply", False), "bytes_freed": sum(candidates.values()), "files": sorted(candidates)}
    if action == "stop":
        job_id = checked_id(payload["id"])
        matches = [c for c in containers() if owned(c, job_id)]
        if len(matches) != 1:
            raise ValueError("Expected exactly one container with both ownership labels")
        docker("stop", matches[0]["Id"])
        return {"stopped": job_id}
    if action == "snapshot":
        return {"jobs": snapshot()}
    if action == "status":
        gpu = execute(["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total", "--format=csv,noheader"], capture_output=True).stdout
        return {"jobs": snapshot(), "gpus": gpu, "disk_free_gb": round(shutil.disk_usage(root).free / 1024**3, 2)}
    raise ValueError("Unknown remote action")


def ledger_key(cfg, job_id):
    return target(cfg) + ":" + cfg["project"] + ":" + job_id


def finished(job):
    return job["state"] in {"exited", "dead", "launch_failed"}


def collect(cfg, job, root=ROOT, include_steps=False):
    if not finished(job):
        raise ValueError("Only finished jobs can be collected")
    job_id = checked_id(job["id"])
    if "run_dir" not in job:
        raise ValueError("Job metadata is missing; restore cmd.json before collection")
    output = run_dir(job["run_dir"])
    safe_path(root, output, True)
    metadata = safe_path(root, "runs/_jobs", True)
    metadata.mkdir(parents=True, exist_ok=True)
    import fcntl
    with safe_path(root, "runs/_jobs/ledger.lock", True).open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ledger = safe_path(root, "runs/_jobs/ledger.json", True)
        entries = json.loads(ledger.read_text()) if ledger.exists() else {}
        if ledger_key(cfg, job_id) in entries:
            # A failed explicit re-collection must become pending again.
            entries.pop(ledger_key(cfg, job_id))
            atomic_json(ledger, entries)
        expected = remote(cfg, "manifest", id=job_id, include_steps=include_steps)["files"]
        directories = [output, "runs/_jobs/" + job_id]
        # Reject unexpected paths from a remote manifest before using them locally.
        for relative in expected:
            if not any(PurePosixPath(relative).is_relative_to(PurePosixPath(d)) for d in directories):
                raise ValueError("Manifest path outside selected job: " + relative)
            safe_path(root, relative, True)
        with tempfile.TemporaryDirectory(prefix=job_id + "-", dir=metadata) as temporary:
            staging = Path(temporary)
            for directory in directories:
                destination = staging / directory
                destination.mkdir(parents=True)
                excludes = ["--exclude=.collected"]
                if not include_steps:
                    excludes.append("--exclude=step_*.pt")
                execute(["rsync", "-az", *excludes, "--", f"{target(cfg)}:{cfg['project']}/{directory}/", str(destination) + "/"])
            actual = manifest(staging, directories, include_steps)
            conflicts = [p for p in sorted(expected.keys() | actual.keys()) if expected.get(p) != actual.get(p)]
            if conflicts:
                raise ValueError("Collection manifest mismatch: " + ", ".join(conflicts))
            # Recheck the source after transfer to catch concurrent changes/pruning.
            latest = remote(cfg, "manifest", id=job_id, include_steps=include_steps)["files"]
            if latest != expected:
                changed = [p for p in sorted(expected.keys() | latest.keys()) if expected.get(p) != latest.get(p)]
                raise ValueError("Remote manifest changed during collection: " + ", ".join(changed))
            conflicts = []
            for relative, digest in expected.items():
                path = safe_path(root, relative, True)
                if path.exists() and (not path.is_file() or file_digest(path) != digest):
                    conflicts.append(relative)
            if conflicts:
                raise ValueError("Existing local files conflict: " + ", ".join(conflicts))
            for relative in expected:
                path = safe_path(root, relative, True)
                path.parent.mkdir(parents=True, exist_ok=True)
                if not path.exists():
                    # Same-filesystem staging makes each file promotion atomic.
                    (staging / relative).replace(path)
            remote(cfg, "collected", id=job_id)
            entries[ledger_key(cfg, job_id)] = {"collected": datetime.datetime.now(datetime.timezone.utc).isoformat(), "run_dir": output, "include_steps": include_steps}
            atomic_json(ledger, entries)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    sync = commands.add_parser("sync")
    sync.add_argument("--yes", action="store_true")
    sync.add_argument("--data", type=Path)
    submit = commands.add_parser("submit")
    submit.add_argument("--gpu", type=int, required=True)
    submit.add_argument("--name", required=True)
    submit.add_argument("--run-dir", help="Script's output directory (default runs/NAME); configure the script to match")
    submit.add_argument("--force", action="store_true")
    submit.add_argument("--checkpoint-size-mb", type=float, help="Estimated full checkpoint MiB; otherwise init/resume size x 6")
    submit.add_argument("--allow-low-space", action="store_true", help="Override storage admission only")
    submit.add_argument("--no-checkpoints", action="store_true",
                        help="Evaluation/utility job that writes no checkpoints; skips the checkpoint estimate")
    submit.add_argument("--resume", action="store_true", help="Allow finished output reuse with a training --resume checkpoint")
    submit.add_argument("command", nargs=argparse.REMAINDER)
    commands.add_parser("status")
    commands.add_parser("pending")
    logs = commands.add_parser("logs")
    logs.add_argument("id")
    logs.add_argument("-f", action="store_true")
    stop = commands.add_parser("stop")
    stop.add_argument("id")
    get = commands.add_parser("collect")
    selection = get.add_mutually_exclusive_group(required=True)
    selection.add_argument("id", nargs="?")
    selection.add_argument("--all-finished", action="store_true")
    get.add_argument("--include-step-checkpoints", action="store_true")
    prune = commands.add_parser("prune-checkpoints")
    prune.add_argument("run_dirs", nargs="+")
    prune.add_argument("--apply", action="store_true")
    prune.add_argument("--keep", action="append", default=[])
    prune.add_argument("--keep-latest", type=int, default=0)
    args = parser.parse_args(argv)
    cfg = config()
    if args.action == "sync":
        # One transfer per entry: macOS openrsync ignores the -R "/./" anchor.
        excludes = ["--exclude=data/", "--exclude=runs/", "--exclude=checkpoints/", "--exclude=.venv/", "--exclude=__pycache__/", "--exclude=gpu_host.yaml"]
        batch = [["rsync", "-az", *excludes, "--", str(ROOT / p) + ("/" if p.endswith("/") else ""), f"{target(cfg)}:{cfg['project']}/{p}"]
                 for p in ("src/", "scripts/", "configs/", "calib/", "tests/", "pyproject.toml")]
        if args.data:
            source = args.data.resolve(strict=True)
            if not source.is_dir() or not ID_RE.fullmatch(source.name):
                raise ValueError("--data must name a directory with a simple basename")
            batch.append(["rsync", "-az", "--", str(source), f"{target(cfg)}:{cfg['project']}/data/"])
        for command in batch:
            print(shlex.join(command))
            if args.yes:
                execute(command)
    elif args.action == "submit":
        checked_id(args.name)
        command = args.command[1:] if args.command[:1] == ["--"] else args.command
        if args.gpu < 0 or not command:
            raise ValueError("Provide a nonnegative GPU index and a Python script after --")
        job_id = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + args.name[:48] + "-" + uuid.uuid4().hex[:8]
        result = remote(cfg, "submit", id=job_id, name=args.name, gpu=args.gpu, command=command, run_dir=run_dir(args.run_dir or "runs/" + args.name), force=args.force, checkpoint_size_mb=args.checkpoint_size_mb, allow_low_space=args.allow_low_space, resume=args.resume, no_checkpoints=args.no_checkpoints)
        print(json.dumps(result, indent=2))
    elif args.action == "prune-checkpoints":
        print(json.dumps(remote(cfg, "prune-checkpoints", run_dirs=[run_dir(p) for p in args.run_dirs], apply=args.apply, keep=args.keep, keep_latest=args.keep_latest), indent=2))
    elif args.action == "stop":
        print(json.dumps(remote(cfg, "stop", id=checked_id(args.id))))
    elif args.action == "logs":
        job_id = checked_id(args.id)
        path = cfg["project"] + "/runs/_jobs/" + job_id + "/stdout.log"
        execute(["ssh", "-o", "BatchMode=yes", target(cfg), shlex.join(["tail", "-n", "50", *(["-f"] if args.f else []), "--", path])])
    elif args.action == "status":
        print(json.dumps(remote(cfg, "status"), indent=2))
    else:
        jobs = remote(cfg, "snapshot")["jobs"]
        ledger = ROOT / "runs/_jobs/ledger.json"
        entries = json.loads(ledger.read_text()) if ledger.exists() else {}
        pending = [j for j in jobs if finished(j) and ledger_key(cfg, j["id"]) not in entries]
        if args.action == "pending":
            print(json.dumps(pending, indent=2))
        else:
            selected = pending if args.all_finished else [j for j in jobs if j["id"] == checked_id(args.id)]
            if not selected and not args.all_finished:
                raise ValueError("Job not found")
            for job in selected:
                if args.include_step_checkpoints:
                    collect(cfg, job, include_steps=True)
                else:
                    collect(cfg, job)
                print("Collected " + job["id"])


if __name__ == "__main__":
    try:
        if sys.argv[1:2] == ["--_remote"]:
            print(json.dumps(remote_main(json.loads(sys.argv[2]))))
        else:
            main()
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print(str(error), file=sys.stderr)
        if isinstance(error, subprocess.CalledProcessError) and error.stderr:
            print(error.stderr, file=sys.stderr)
        raise SystemExit(1)
