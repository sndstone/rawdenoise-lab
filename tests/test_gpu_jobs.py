"""Host orchestration uses fake subprocesses only; no SSH or GPU required."""
import importlib.util
import json
from pathlib import Path
import shlex
import shutil
import subprocess
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location("gpu_jobs", Path(__file__).parents[1] / "scripts/gpu_jobs.py")
jobs = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(jobs)


@pytest.fixture
def cfg():
    return {"host": "test-host", "user": "tester", "project": "rawdenoise-lab", "min_free_gb": 10}


@pytest.fixture
def host(tmp_path, monkeypatch, cfg):
    cfg = dict(cfg, project=str(tmp_path))
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv/pyvenv.cfg").touch()
    (tmp_path / "budget.yaml").write_text("training:\n  steps: 24000\n  checkpoint_every: 6000\n  keep_checkpoints: 4\n")
    calls = []
    state = {"util": 0, "memory": 0, "apps": "", "containers": [], "free": 20 * 1024**3}

    def fake(command, **kwargs):
        calls.append(command)
        assert command[0] != "ssh"
        if command[:2] == ["docker", "ps"]:
            result = " ".join(c["Id"] for c in state["containers"])
        elif command[:2] == ["docker", "inspect"]:
            result = json.dumps(state["containers"])
        elif command[:2] == ["docker", "run"]:
            result = "container-id\n"
        elif command[0] == "nvidia-smi":
            result = state["apps"] if "compute-apps" in command[1] else f"0, GPU-zero, {state['util']}, {state['memory']}\n"
        else:
            result = ""
        return SimpleNamespace(stdout=result)

    monkeypatch.setattr(jobs.subprocess, "run", fake)
    monkeypatch.setattr(shutil, "disk_usage", lambda path: SimpleNamespace(free=state["free"]))
    return cfg, state, calls


def submit(cfg, **changes):
    return dict(cfg, action="submit", id="test-job", name="test", gpu=0, command=["scripts/train_arch_study.py", "--config", "budget.yaml", "--run-dir", "runs/test"], run_dir="runs/test", force=False, checkpoint_size_mb=100, **changes)


def container(job_id="test-job", managed=True, running=True):
    return {"Id": "exact-container", "Config": {"Labels": {"rawdenoise.managed": "gpu_jobs" if managed else "other", "rawdenoise.job": job_id, "rawdenoise.gpu": "0"}}, "State": {"Status": "running" if running else "exited", "Running": running, "ExitCode": 0}}


def test_submit_safety_and_logging(host):
    cfg, state, calls = host
    record = jobs.remote_main(submit(cfg))
    command = next(c for c in calls if c[:2] == ["docker", "run"])
    for option, value in [("--network", "none"), ("--pull", "never"), ("--gpus", "device=0"), ("--entrypoint", "/workspace/.venv/bin/python")]:
        assert command[command.index(option) + 1] == value
    assert jobs.OWNER in command and "rawdenoise.job=test-job" in command
    assert "--user" in command and "--rm" not in command
    assert f"{cfg['project']}:/workspace" in command
    assert "/workspace/runs/_jobs/test-job/stdout.log" in command
    assert "PYTHONPATH=/workspace/src" in command
    assert jobs.IMAGE in command
    directory = Path(cfg["project"]) / "runs/_jobs/test-job"
    assert json.loads((directory / "cmd.json").read_text())["command"] == record["command"]
    assert (directory / "stdout.log").exists()
    assert (Path(cfg["project"]) / "runs/test").is_dir()


@pytest.mark.parametrize("field,value", [("util", 1), ("memory", 300), ("apps", "GPU-zero\n"), ("containers", [container()])])
def test_busy_refusal_and_force(host, field, value):
    cfg, state, calls = host
    state[field] = value
    with pytest.raises(ValueError, match="busy"):
        jobs.remote_main(submit(cfg))
    assert not any(c[:2] == ["docker", "run"] for c in calls)
    payload = submit(cfg)
    payload["force"] = True
    jobs.remote_main(payload)


def test_disk_refusal_even_with_force(host):
    cfg, state, calls = host
    state["free"] = 1
    payload = submit(cfg)
    payload["force"] = True
    with pytest.raises(ValueError, match="disk"):
        jobs.remote_main(payload)
    assert not any(c[:2] == ["docker", "run"] for c in calls)


def test_missing_gpu_refused(host):
    cfg, _, calls = host
    payload = submit(cfg)
    payload["gpu"] = 9
    with pytest.raises(ValueError, match="not found"):
        jobs.remote_main(payload)
    assert not any(c[:2] == ["docker", "run"] for c in calls)


@pytest.mark.parametrize("managed,job_id", [(False, "test-job"), (True, "different-job")])
def test_stop_rechecks_both_labels(host, managed, job_id):
    cfg, state, calls = host
    state["containers"] = [container(job_id, managed)]
    with pytest.raises(ValueError, match="ownership"):
        jobs.remote_main(dict(cfg, action="stop", id="test-job"))
    assert not any(c[:2] == ["docker", "stop"] for c in calls)
    assert ["docker", "ps", "-aq", "--filter", "label=" + jobs.OWNER] in calls


def test_stop_uses_exact_container_id(host):
    cfg, state, calls = host
    state["containers"] = [container()]
    jobs.remote_main(dict(cfg, action="stop", id="test-job"))
    assert calls[-1] == ["docker", "stop", "exact-container"]


def test_snapshot_reconciles_job_dirs_and_containers(host):
    cfg, state, _ = host
    jobs.remote_main(submit(cfg))
    state["containers"] = [container(running=False), container("orphan", running=False)]
    log = Path(cfg["project"]) / "runs/_jobs/test-job/stdout.log"
    log.write_text("\n".join(str(i) for i in range(20)))
    result = jobs.remote_main(dict(cfg, action="snapshot"))["jobs"]
    assert result[0]["state"] == "exited" and result[0]["exit_code"] == 0
    assert result[0]["tail"] == [str(i) for i in range(15, 20)]
    assert result[1]["metadata_missing"]


def test_ssh_payload_is_shell_quoted(cfg, monkeypatch):
    calls = []
    monkeypatch.setattr(jobs.subprocess, "run", lambda command, **kw: calls.append(command) or SimpleNamespace(stdout="{}"))
    jobs.remote(cfg, "submit", command=["script.py", "$(touch bad); 'quote'"])
    command = calls[0]
    assert command[:4] == ["ssh", "-o", "BatchMode=yes", "tester@test-host"]
    remote = shlex.split(command[-1])
    assert remote[:2] == ["python3", "-c"]
    assert json.loads(remote[-1])["command"][1] == "$(touch bad); 'quote'"
    assert "globals().get" in remote[2]  # Source also runs under python -c, without __file__.


def test_sync_dry_run_exclusions(cfg, monkeypatch, tmp_path, capsys):
    calls = []
    monkeypatch.setattr(jobs, "config", lambda: cfg)
    monkeypatch.setattr(jobs.subprocess, "run", lambda command, **kw: calls.append(command))
    data = tmp_path / "sample"
    data.mkdir()
    jobs.main(["sync", "--data", str(data)])
    assert not calls
    output = capsys.readouterr().out
    assert "--exclude=gpu_host.yaml" in output and "--exclude=checkpoints/" in output
    assert "test-host:rawdenoise-lab/data/" in output
    jobs.main(["sync", "--yes"])
    assert calls[0][0] == "rsync" and "-az" in calls[0] and len(calls) == 6
    assert calls[0][-2].endswith("/src/") and calls[0][-1].endswith(":rawdenoise-lab/src/")
    assert not any("--delete" in c for c in calls)


@pytest.fixture
def collection(cfg, tmp_path, monkeypatch):
    remote_root = tmp_path / "remote"
    local_root = tmp_path / "local"
    output = remote_root / "runs/result"
    metadata = remote_root / "runs/_jobs/job"
    output.mkdir(parents=True)
    metadata.mkdir(parents=True)
    (output / "last.pt").write_bytes(b"checkpoint")
    (output / "step_0001.pt").write_bytes(b"intermediate")
    (output / "history.json").write_text("[]")
    (metadata / "cmd.json").write_text("{}")
    (metadata / "stdout.log").write_text("done")
    state = {"calls": [], "acknowledged": False, "corrupt": False, "interrupt": False}

    def remote(cfg, action, **kwargs):
        if action == "manifest":
            return {"files": jobs.manifest(remote_root, ["runs/result", "runs/_jobs/job"], kwargs.get("include_steps", False))}
        assert action == "collected"
        state["acknowledged"] = True
        return {}

    def transfer(command, **kwargs):
        state["calls"].append(command)
        relative = command[-2].split(":", 1)[1].removeprefix(cfg["project"] + "/").rstrip("/")
        destination = Path(command[-1])
        for source in (remote_root / relative).iterdir():
            if source.name.startswith("step_") and "--exclude=step_*.pt" in command:
                continue
            shutil.copyfile(source, destination / source.name)
        if state["interrupt"]:
            raise subprocess.CalledProcessError(1, command)
        if state["corrupt"] and relative == "runs/result":
            (destination / "last.pt").write_bytes(b"truncated")
        return SimpleNamespace(stdout="")

    monkeypatch.setattr(jobs, "remote", remote)
    monkeypatch.setattr(jobs.subprocess, "run", transfer)
    return cfg, local_root, remote_root, state


def test_collect_ledger_only_after_success(collection):
    cfg, root, _, state = collection
    record = {"id": "job", "state": "exited", "run_dir": "runs/result"}
    existing = root / "runs/result/last.pt"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"checkpoint")
    jobs.collect(cfg, record, root)
    assert len(state["calls"]) == 2
    assert all("--ignore-existing" not in c for c in state["calls"])
    assert existing.read_bytes() == b"checkpoint"
    assert not (existing.parent / "step_0001.pt").exists()
    assert state["acknowledged"]
    entries = json.loads((root / "runs/_jobs/ledger.json").read_text())
    assert jobs.ledger_key(cfg, "job") in entries
    jobs.collect(cfg, record, root)  # Identical existing files are accepted on retry.
    with pytest.raises(ValueError, match="finished"):
        jobs.collect(cfg, dict(record, state="running"), root)


@pytest.mark.parametrize("failure", ["conflict", "corrupt", "interrupt"])
def test_collection_failures_remain_pending_and_retry(collection, failure):
    cfg, root, _, state = collection
    record = {"id": "job", "state": "exited", "run_dir": "runs/result"}
    existing = root / "runs/result/last.pt"
    if failure == "conflict":
        existing.parent.mkdir(parents=True)
        existing.write_bytes(b"stale")
    else:
        state[failure] = True
    with pytest.raises((ValueError, subprocess.CalledProcessError)) as error:
        jobs.collect(cfg, record, root)
    if failure != "interrupt":
        assert "last.pt" in str(error.value)
    assert not (root / "runs/_jobs/ledger.json").exists()
    assert not state["acknowledged"]
    assert sorted(p.name for p in (root / "runs/_jobs").iterdir()) == ["ledger.lock"]
    if failure == "conflict":
        assert existing.read_bytes() == b"stale"
        existing.unlink()
    else:
        state[failure] = False
    jobs.collect(cfg, record, root, include_steps=True)
    assert (root / "runs/result/step_0001.pt").read_bytes() == b"intermediate"
    assert state["acknowledged"]


def test_pending_uses_host_scoped_ledger(cfg, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(jobs, "ROOT", tmp_path)
    monkeypatch.setattr(jobs, "config", lambda: cfg)
    records = [{"id": name, "state": state, "run_dir": "runs/" + name} for name, state in [("collected", "exited"), ("pending", "exited"), ("active", "running"), ("unknown", "unknown")]]
    monkeypatch.setattr(jobs, "remote", lambda *a, **kw: {"jobs": records})
    ledger = tmp_path / "runs/_jobs/ledger.json"
    ledger.parent.mkdir(parents=True)
    ledger.write_text(json.dumps({jobs.ledger_key(cfg, "collected"): {}}))
    jobs.main(["pending"])
    assert [r["id"] for r in json.loads(capsys.readouterr().out)] == ["pending"]
    collected = []
    monkeypatch.setattr(jobs, "collect", lambda cfg, job: collected.append(job["id"]))
    jobs.main(["collect", "--all-finished"])
    assert collected == ["pending"]


@pytest.mark.parametrize("value", ["../bad", "/runs/bad", "runs/_jobs/bad", "runs/../bad", "runs", "runs/bad;command", "runs/bad name"])
def test_output_traversal_rejected(value):
    with pytest.raises(ValueError):
        jobs.run_dir(value)


def test_config_flat_yaml_and_environment(tmp_path, monkeypatch):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/gpu_host.yaml").write_text("host: example\nuser: researcher\nproject: /srv/project\nmin_free_gb: 12\n")
    monkeypatch.setenv("RAWDENOISE_GPU_HOST", "override")
    cfg = jobs.config(tmp_path)
    assert cfg["host"] == "override" and cfg["min_free_gb"] == 12
    assert cfg["project"] == "/srv/project"


def test_logs_validates_id_and_quotes_path(cfg, monkeypatch):
    calls = []
    monkeypatch.setattr(jobs, "config", lambda: cfg)
    monkeypatch.setattr(jobs.subprocess, "run", lambda command, **kw: calls.append(command))
    jobs.main(["logs", "job", "-f"])
    assert shlex.split(calls[-1][-1]) == ["tail", "-n", "50", "-f", "--", "rawdenoise-lab/runs/_jobs/job/stdout.log"]
    with pytest.raises(ValueError):
        jobs.main(["stop", "bad;kill"])


def test_budget_counts_retention_final_and_transient(host):
    cfg, _, _ = host
    payload = submit(cfg)
    budget = jobs.checkpoint_budget(Path(cfg["project"]), payload)
    assert budget["storage_bytes"] == 6 * 100 * 1024**2
    assert budget["retained_intermediates"] == 4
    initial = Path(cfg["project"]) / "init.pt"
    initial.write_bytes(b"x" * 10)
    payload["checkpoint_size_mb"] = None
    payload["command"] += ["--init-checkpoint", "init.pt", "--steps", "1"]
    budget = jobs.checkpoint_budget(Path(cfg["project"]), payload)
    assert budget["checkpoint_bytes"] == 60
    assert budget["storage_bytes"] == 180
    initial.unlink()
    with pytest.raises(FileNotFoundError):
        jobs.checkpoint_budget(Path(cfg["project"]), payload)


@pytest.mark.parametrize("every,keep,steps,retained", [(0, 0, 5, 0), (0, 4, 5, 1), (3, 4, 5, 2), (1, 2, 5, 2)])
def test_budget_save_schedule(host, every, keep, steps, retained):
    cfg, _, _ = host
    root = Path(cfg["project"])
    (root / "budget.yaml").write_text(f"training:\n  steps: {steps}\n  checkpoint_every: {every}\n  keep_checkpoints: {keep}\n")
    assert jobs.checkpoint_budget(root, submit(cfg))["storage_bytes"] == (retained + 2) * 100 * 1024**2


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan")])
def test_invalid_checkpoint_size(host, value):
    cfg, _, _ = host
    payload = submit(cfg)
    payload["checkpoint_size_mb"] = value
    with pytest.raises(ValueError, match="positive"):
        jobs.remote_main(payload)


def test_storage_reservations_and_explicit_override(host):
    cfg, state, calls = host
    first = jobs.remote_main(submit(cfg))
    state["containers"] = [container()]
    state["free"] = 10 * 1024**3 + first["storage_bytes"] + 1
    second = submit(cfg)
    second.update(id="second", run_dir="runs/second", force=True)
    with pytest.raises(ValueError, match="reservations"):
        jobs.remote_main(second)
    second["allow_low_space"] = True
    result = jobs.remote_main(second)
    assert result["storage_bytes"] == first["storage_bytes"]
    assert len([c for c in calls if c[:2] == ["docker", "run"]]) == 2
    status = jobs.remote_main(dict(cfg, action="status"))
    assert all(r["storage_bytes"] > 0 for r in status["jobs"])


@pytest.mark.parametrize("gpu", [0, 1])
def test_run_reservation_across_gpus(host, gpu):
    cfg, state, calls = host
    jobs.remote_main(submit(cfg))
    state["containers"] = [container()]
    # Mock another available physical GPU.
    original = jobs.execute

    def execute(command, **kwargs):
        if command[0] == "nvidia-smi" and "query-gpu=index,uuid" in command[1]:
            return SimpleNamespace(stdout=f"{gpu}, GPU-selected, 0, 0")
        return original(command, **kwargs)

    from unittest.mock import patch
    with patch.object(jobs, "execute", execute):
        payload = submit(cfg)
        payload.update(id="second", gpu=gpu, force=True, resume=True)
        with pytest.raises(ValueError, match="reserved"):
            jobs.remote_main(payload)
    assert len([c for c in calls if c[:2] == ["docker", "run"]]) == 1


def test_uncollected_and_nonempty_run_need_validated_resume(host):
    cfg, state, _ = host
    jobs.remote_main(submit(cfg))
    state["containers"] = [container(running=False)]
    payload = submit(cfg)
    payload["id"] = "second"
    with pytest.raises(ValueError, match="already used"):
        jobs.remote_main(payload)
    payload["resume"] = True
    with pytest.raises(ValueError, match="training --resume"):
        jobs.remote_main(payload)
    checkpoint = Path(cfg["project"]) / "runs/test/last.pt"
    checkpoint.write_bytes(b"checkpoint")
    payload["command"] += ["--resume", "runs/test/last.pt"]
    assert jobs.remote_main(payload)["id"] == "second"


def test_existing_untracked_output_refused(host):
    cfg, _, _ = host
    output = Path(cfg["project"]) / "runs/test"
    output.mkdir(parents=True)
    (output / "history.json").write_text("[]")
    with pytest.raises(ValueError, match="already used"):
        jobs.remote_main(submit(cfg))


def completed_tree(host):
    cfg, state, _ = host
    jobs.remote_main(submit(cfg))
    state["containers"] = [container(running=False)]
    output = Path(cfg["project"]) / "runs/test"
    for name, content in {"step_2.pt": b"22", "step_10.pt": b"ten", "step_100.pt": b"hundred", "last.pt": b"last", "config.yaml": b"config", "history.json": b"[]", "run_manifest.json": b"{}", "step_notes.pt": b"notes"}.items():
        (output / name).write_bytes(content)
    return cfg, state, output


def test_prune_dry_run_retention_bytes_and_protection(host):
    cfg, _, output = completed_tree(host)
    payload = dict(cfg, action="prune-checkpoints", run_dirs=["runs/test"], keep=["step_2.pt"], keep_latest=1)
    result = jobs.remote_main(payload)
    assert result == {"applied": False, "bytes_freed": 3, "files": ["runs/test/step_10.pt"]}
    assert (output / "step_10.pt").exists()
    payload["apply"] = True
    assert jobs.remote_main(payload)["bytes_freed"] == 3
    assert not (output / "step_10.pt").exists()
    assert {p.name for p in output.iterdir()} == {"step_2.pt", "step_100.pt", "last.pt", "config.yaml", "history.json", "run_manifest.json", "step_notes.pt"}


@pytest.mark.parametrize("state_name", ["running", "unknown"])
def test_prune_refuses_unfinished(host, state_name):
    cfg, state, output = completed_tree(host)
    state["containers"] = [container()] if state_name == "running" else []
    with pytest.raises(ValueError, match="finished"):
        jobs.remote_main(dict(cfg, action="prune-checkpoints", run_dirs=["runs/test"], apply=True))
    assert (output / "step_2.pt").exists()


@pytest.mark.parametrize("value", ["runs/../test", "../runs/test", "/runs/test", "runs/_jobs/test-job", "runs/untracked"])
def test_prune_refuses_unsafe_or_untracked_paths(host, value):
    cfg, _, _ = completed_tree(host)
    with pytest.raises(ValueError):
        jobs.remote_main(dict(cfg, action="prune-checkpoints", run_dirs=[value], apply=True))


@pytest.mark.parametrize("kind", ["run", "file", "kept-file", "nested-directory"])
def test_prune_and_submit_refuse_symlinks(host, kind):
    cfg, _, output = completed_tree(host)
    if kind == "run":
        output.rename(output.with_name("original"))
        output.symlink_to(output.with_name("original"), target_is_directory=True)
    elif kind == "nested-directory":
        (output / "nested").symlink_to(output.parent, target_is_directory=True)
    else:
        (output / "step_2.pt").unlink()
        (output / "step_2.pt").symlink_to(output / "last.pt")
    with pytest.raises(ValueError, match="Symlink"):
        jobs.remote_main(dict(cfg, action="prune-checkpoints", run_dirs=["runs/test"], keep=["step_2.pt"] if kind == "kept-file" else [], apply=True))
    assert (output / "step_10.pt").exists()
    if kind == "run":
        payload = submit(cfg)
        payload.update(id="second", resume=True)
        with pytest.raises(ValueError, match="Symlink"):
            jobs.remote_main(payload)


def test_remote_manifest_and_collection_ack(host):
    cfg, _, output = completed_tree(host)
    result = jobs.remote_main(dict(cfg, action="manifest", id="test-job"))["files"]
    assert "runs/test/last.pt" in result and "runs/test/step_2.pt" not in result
    assert result["runs/test/last.pt"] == jobs.file_digest(output / "last.pt")
    jobs.remote_main(dict(cfg, action="collected", id="test-job"))
    records = jobs.remote_main(dict(cfg, action="snapshot"))["jobs"]
    assert records[0]["collected"]


def test_failed_recollection_invalidates_old_success(collection):
    cfg, root, _, state = collection
    record = {"id": "job", "state": "exited", "run_dir": "runs/result"}
    jobs.collect(cfg, record, root)
    state["corrupt"] = True
    with pytest.raises(ValueError, match="last.pt"):
        jobs.collect(cfg, record, root)
    assert jobs.ledger_key(cfg, "job") not in json.loads((root / "runs/_jobs/ledger.json").read_text())


def test_collection_detects_remote_change_before_promotion(collection, monkeypatch):
    cfg, root, remote_root, _ = collection
    original = jobs.remote
    calls = []

    def changing(cfg, action, **kwargs):
        calls.append(action)
        if calls.count("manifest") == 2:
            (remote_root / "runs/result/last.pt").write_bytes(b"changed")
        return original(cfg, action, **kwargs)

    monkeypatch.setattr(jobs, "remote", changing)
    with pytest.raises(ValueError, match="changed.*last.pt"):
        jobs.collect(cfg, {"id": "job", "state": "exited", "run_dir": "runs/result"}, root)
    assert not (root / "runs/result/last.pt").exists()
    assert not (root / "runs/_jobs/ledger.json").exists()


@pytest.mark.parametrize("location", ["remote", "local"])
def test_collection_refuses_symlinked_files(collection, location):
    cfg, root, remote_root, _ = collection
    if location == "remote":
        target = remote_root / "runs/result/last.pt"
        target.unlink()
        target.symlink_to("history.json")
    else:
        target = root / "runs/result/last.pt"
        target.parent.mkdir(parents=True)
        target.symlink_to(remote_root / "runs/result/last.pt")
    with pytest.raises(ValueError, match="Symlink"):
        jobs.collect(cfg, {"id": "job", "state": "exited", "run_dir": "runs/result"}, root)
    assert not (root / "runs/_jobs/ledger.json").exists()


def test_prune_checks_all_selected_trees_before_deleting(host):
    cfg, state, output = completed_tree(host)
    extra = Path(cfg["project"]) / "runs/other"
    extra.mkdir()
    (extra / "step_1.pt").symlink_to(output / "last.pt")
    metadata = Path(cfg["project"]) / "runs/_jobs/other"
    metadata.mkdir()
    (metadata / "cmd.json").write_text(json.dumps({"id": "other", "state": "launch_failed", "run_dir": "runs/other"}))
    state["containers"] = [container(running=False)]
    with pytest.raises(ValueError, match="Symlink"):
        jobs.remote_main(dict(cfg, action="prune-checkpoints", run_dirs=["runs/test", "runs/other"], apply=True))
    assert (output / "step_2.pt").exists()


@pytest.mark.parametrize("key,value", [("keep_checkpoints", "-1"), ("checkpoint_every", "schedule"), ("steps", "&count 5"), ("<<", "*defaults")])
def test_budget_refuses_unsupported_scalars(host, key, value):
    cfg, _, _ = host
    root = Path(cfg["project"])
    (root / "budget.yaml").write_text(f"training:\n  steps: 5\n  {key}: {value}\n")
    with pytest.raises(ValueError, match="Storage budgeting"):
        jobs.checkpoint_budget(root, submit(cfg))


def test_cli_forwards_new_flags(cfg, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(jobs, "config", lambda: cfg)
    monkeypatch.setattr(jobs, "remote", lambda cfg, action, **kwargs: calls.append((action, kwargs)) or {})
    jobs.main(["submit", "--gpu", "0", "--name", "demo", "--checkpoint-size-mb", "42", "--allow-low-space", "--resume", "--", "script.py", "--config", "test.yaml"])
    action, payload = calls[-1]
    assert action == "submit" and payload["checkpoint_size_mb"] == 42
    assert payload["allow_low_space"] and payload["resume"]
    jobs.main(["prune-checkpoints", "runs/demo", "--keep", "step_1.pt", "--keep-latest", "2", "--apply"])
    assert calls[-1] == ("prune-checkpoints", {"run_dirs": ["runs/demo"], "keep": ["step_1.pt"], "keep_latest": 2, "apply": True})
    assert capsys.readouterr().out


def test_collect_accepts_symlinked_local_runs_folder_only_at_top(collection, tmp_path):
    cfg, root, _, state = collection
    store = tmp_path / "share/runs"
    store.mkdir(parents=True)
    root.mkdir(exist_ok=True)
    (root / "runs").symlink_to(store, target_is_directory=True)
    record = {"id": "job", "state": "exited", "run_dir": "runs/result"}
    jobs.collect(cfg, record, root)
    assert (store / "result/last.pt").read_bytes() == b"checkpoint"
    assert jobs.ledger_key(cfg, "job") in json.loads((store / "_jobs/ledger.json").read_text())
    # A symlink below the top-level folder is still refused.
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    shutil.rmtree(store / "result")
    (store / "result").symlink_to(elsewhere, target_is_directory=True)
    with pytest.raises(ValueError, match="Symlink refused"):
        jobs.collect(cfg, record, root)
    with pytest.raises(ValueError, match="Symlink refused"):
        jobs.safe_path(root, "runs/_jobs")


def test_no_checkpoint_jobs_skip_the_training_config_estimate(tmp_path):
    payload = {"command": ["scripts/scoreboard.py", "--config", "configs/scoreboard/default.yaml"], "no_checkpoints": True}
    assert jobs.checkpoint_budget(tmp_path, payload)["storage_bytes"] == 0
    with pytest.raises(ValueError):
        jobs.checkpoint_budget(tmp_path, {"command": ["scripts/scoreboard.py"]})
