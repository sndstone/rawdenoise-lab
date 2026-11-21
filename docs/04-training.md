Historical (October 2026). Superseded by [docs/09](09-clean-targets-and-defects.md) for the current model; kept for the record.

# 4. Training, compute and export

## Recipe (starting point)

| Item | Setting | Source |
|---|---|---|
| Model | NAFNet-w32, [2,2,4,8] / 12 / [2,2,2,2], gamma head | Published NAFNet paper and SIDD configuration (see references.md) |
| Crops | Packed 256² early → 512² late (progressive, as MR-CAS did at AIM 2025) | AIM 2025 report |
| Batch | 16–32 effective (gradient accumulation on one GPU) | NAFNet official: 32 |
| Optimiser | AdamW, lr 1e-3 → 1e-7 cosine, betas (0.9, 0.9), wd 0 | NAFNet official |
| Iterations | 200k–400k (synthetic stage) + 20k–50k (real-pair fine-tune, lr 1e-4 → 1e-6) | NAFNet; AIM entries |
| Loss | L1 or Charbonnier in the gamma domain + 0.1 × L1 in the linear domain; optional 0.05 × FFT-L1 | AIM 2025 entries; our contract |
| Precision | bf16 autocast on NVIDIA; LayerNorm stats and the loss in fp32 | — |
| Augmentation | BayerAug flips and transposes, even crops, brightness scaling ×[0.5, 2] before the noise is added, mirrored borders | Liu et al. 2019 |
| Tiers (Phase A) | Sample K from each tier's ISO range, slightly overlapping the neighbouring tiers | Planned camera calibration and held-out validation |

**Model selection.** Choose by PSNR and SSIM on the real held-out pairs, not
on synthetic validation data.

## Resume reproducibility (stream version 1)

**Fresh training streams changed on 9 November 2026.** Both trainers now key
sample selection and augmentation to the run seed and absolute sample position.
The four training arms launched earlier that day used the previous stream;
comparisons across this change must record the stream version and cannot claim
identical training draws. Noise and augmentation formulas are unchanged, as are
direct dataset callers and deterministic evaluation datasets.

New checkpoints retain every existing field and add `data_stream` (version,
seed, consumed supervised batches and consumed self-supervised batches), Python
and NumPy RNG states, and all initialized CUDA RNG states. The existing CPU
torch RNG field remains. Only consumed batches advance saved progress; worker
prefetch does not. Shuffle epochs retain partial final batches, while the
self-supervised replacement stream drops partial batches. Mixed replacement
sampling and real-noisy crop selection have independent deterministic streams.
Self-supervised neighbour choices use the restored training torch RNG.

On CPU, uninterrupted versus resumed execution is expected to match exactly
for batches, model weights, optimizer, scheduler and EMA, including gradient
accumulation, with zero or multiple loader workers. Keep the same configuration,
data contents/order, total step budget, software versions, hardware and CPU
thread count. The stream itself is independent of worker count. Resume from an
optimizer-step checkpoint with `--resume`; `--init-checkpoint` starts a new
stream and optimizer schedule. For a controlled interruption test, resume a
periodic checkpoint from the original fixed schedule. Shortening the first run
with `--steps` changes the learning-rate schedule and is not equivalent.

Checkpoints without `data_stream` print a legacy notice and retain the previous
loader-restart behavior, including on subsequent saves/resumes. Their consumed
sample positions and missing historical RNG states cannot be recovered; exact
continuation is unavailable. No existing config key, CLI option or checkpoint
field was removed or renamed.

CUDA RNG restoration supports continuation but does **not** establish bitwise
CUDA reproducibility. Both trainers retain `cudnn.benchmark = True`; algorithm
selection, nondeterministic kernels, mixed precision and hardware/software
differences can change results. CUDA continuation still needs validation on a
training host. No bitwise guarantee is made for other accelerators, changed
training schedules/data, or legacy checkpoints.

## Current archive prototype and shadow experiment

The archive prototypes are smaller than the full recipe above: width 32,
three encoder stages `[1,1,2]`, four middle blocks, decoder `[1,1,1]`, and
2,650,404 parameters. Their initial 7,500 updates per camera used the owner's
ISO 100 single frames plus an uncalibrated shot/read/row prior. These remain
imperfect references and provisional models, not accepted releases.

The 27 October 2026 controlled experiment in `configs/refinement/` tests one
data change: exposure coverage. The control continues with the legacy linear
scale in `[0.6,1]`; the shadow arm draws log-uniform scales in `[1/64,1]`,
covering zero to six stops of attenuation. Attenuation happens before new
noise is drawn, so read noise is not attenuated with the reference. The same
initial checkpoint, manifest, seed, architecture, loss, batch/crop size and
learning-rate schedule are used within each camera comparison. NVIDIA GPU
differences may still introduce small numerical differences.

The initial X1D II shadow arm improved the attenuated development slices but
regressed on bright low-noise crops. The follow-up `*_balanced.yaml` configs
sample the legacy bright range with probability 0.5 and the broad log range
otherwise. X1D II uses the same 5,000-update budget and initial weights as the
control; the Sony follow-up is a shorter 2,500-update development run against
its original checkpoint, not an equal-budget causal ablation.

`scripts/archive/evaluate_shadow_refinement.py` compares the original model, extra-
training control and shadow arm on the same held-out sources/noise draws at
reference scales 1, 1/16 and 1/64. It reports RAW PSNR, gamma error, colour-
difference error, bias and a zero-output baseline. These are development
diagnostics against artificially attenuated single frames; the release gate
still requires independent real references and owner review.

The primary literature reinforces the next data step: Li et al.,
[Noise Modeling in One Hour](https://arxiv.org/abs/2505.00045), CVPR 2025,
samples signal-independent noise from sensor dark frames and subtracts
averaged dark shading. Its results motivate collecting our own dark frames;
they do not validate our present generic prior.
The experiment uses independently authored code and existing owner photos.

## Compute options

CUDA training can run on a Linux NVIDIA host using
`scripts/run_cuda.py` (expects a prepared container; adapt to your setup).

| Option | Speed for one NAFNet-w32 model | Practical use |
|---|---|---|
| Apple Silicon (MPS), owner's Mac | Several times slower than a 4090 (no published benchmark; measure it) | Debugging, small fine-tunes, export and ANE checks |
| 1× RTX 4090 (24 GB) | About 1–3 days per model | Full training for one sensor and tier |
| 1× A100 / H100 (cloud) | About 1.5–3× faster than a 4090 | Batch runs: 3 tiers × 3 bodies |

For reference, AIM 2025 teams reported training times from about 28 h on 2×
4090 up to about 3 weeks on 4–8× 4090 for the winning NAFNet.

Phase A covers 8 models: X2D II ×3, X1D II ×2, Sony ×3. Plan on about 8–20
GPU-days in total, or about 3 conditioned models in Phase B. Rented
H100/A100 time at a few dollars an hour puts a full Phase A sweep in the
**low hundreds of dollars**. Check current prices before booking.

## LAN GPU jobs

`scripts/gpu_jobs.py` uses only the Python standard library locally. Copy
`configs/gpu_host.example.yaml` to the ignored `configs/gpu_host.yaml`, replacing
its placeholder host/user. Alternatively set `RAWDENOISE_GPU_HOST` (optionally
`user@host`), `RAWDENOISE_GPU_USER`, `RAWDENOISE_GPU_PROJECT` and
`RAWDENOISE_GPU_MIN_FREE_GB`. The project path defaults to `rawdenoise-lab`
relative to the remote home; absolute paths also work. Configuration accepts
flat scalar keys only. Keep private host addresses out of committed files.
The host needs Python 3, rsync, Docker/NVIDIA, the cached CUDA image used by
`run_cuda.py`, and the prepared Linux project `.venv`. No packages are installed
and no image is pulled by this tool. SSH uses existing authentication in batch
mode; it never changes credentials.

```bash
python scripts/gpu_jobs.py sync                 # prints rsync commands only
python scripts/gpu_jobs.py sync --yes
python scripts/gpu_jobs.py sync --data data/my-targets --yes
python scripts/gpu_jobs.py submit --gpu 0 --name demo --run-dir runs/demo --checkpoint-size-mb 400 -- scripts/train_arch_study.py --config configs/smoke.yaml --run-dir runs/demo --device cuda:0
python scripts/gpu_jobs.py status
python scripts/gpu_jobs.py logs JOB_ID -f
python scripts/gpu_jobs.py stop JOB_ID
python scripts/gpu_jobs.py pending
python scripts/gpu_jobs.py collect JOB_ID
python scripts/gpu_jobs.py collect --all-finished
python scripts/gpu_jobs.py collect JOB_ID --include-step-checkpoints
python scripts/gpu_jobs.py prune-checkpoints runs/demo --keep-latest 1 --keep step_0006000.pt
python scripts/gpu_jobs.py prune-checkpoints runs/demo --keep-latest 1 --apply
```

Sync copies only code/configuration/calibration/tests and `pyproject.toml`,
excluding private host configuration, data, runs, checkpoints and environments;
there is no deletion pass. `--data` additionally copies that directory under
remote `data/` using its basename. Existing code files on the host can be
updated by `sync --yes`; review the dry-run command first.

Submit runs a Python script, detached and unbuffered, in an offline container
with only the project mounted and the host user's UID/GID. `--gpu` selects the
physical GPU index from `status`; the selected device is `cuda:0` **inside**
the container. Submit refuses active compute processes, nonzero utilization,
more than 256 MiB allocated memory, or another running tool job on that GPU;
`--force` bypasses this GPU check only. Under a shared submission lock, storage
admission requires the configurable free-space floor (10 GiB by default), plus
the new job's checkpoint budget and full budgets for unfinished managed jobs.
`--no-checkpoints` submits an evaluation or utility job that writes no checkpoints
(for example `scripts/scoreboard.py`), skipping the training-config estimate.
`--allow-low-space` explicitly overrides storage admission only. Reservations
are conservative start-time accounting, not disk quotas; other tools and legacy
jobs without recorded budgets are not covered.

Budgeting reads the remote training command's `--config` and optional `--steps`
override. The config must use a block-style `training` mapping with plain
nonnegative integer `steps`, `checkpoint_every` and `keep_checkpoints` scalars
(the latter two default to zero); merged mappings are refused. It counts periodic
and final saves, caps intermediate count by retention, and adds `last.pt` plus
one transient checkpoint for temporary writes/copies before pruning. Supply the
estimated size of one full training checkpoint with `--checkpoint-size-mb`
(MiB), or the tool measures the command's remote `--init-checkpoint`/`--resume`
file and multiplies its bytes by six to allow for model, optimizer and averaged
weights. This factor can overestimate full training checkpoints; use an explicit
measured size when known. Status exposes `checkpoint_bytes`, `storage_bytes`,
retained count and estimate source per job. No trainer or historical config
is changed.

The recorded output path defaults to `runs/NAME`; `--run-dir` changes it.
**Configure the script's own output argument/config to match this path.** The
tool exports `RAWDENOISE_RUN_DIR`, but existing training scripts do not read it
automatically. Outputs elsewhere are not collected. Run paths and their ancestors
must not be symlinks. Submission reserves the run directory across all GPUs,
including overlapping parent/child paths: active or unknown jobs always block
reuse. Finished uncollected jobs and nonempty existing output directories require
submit's explicit `--resume` flag plus the training command's `--resume` pointing
to an existing checkpoint inside that run. Collection acknowledges completion on
the remote host as well as recording the local ledger. Each job records its exact
arguments and GPU in remote `runs/_jobs/JOB_ID/cmd.json` and writes
`stdout.log` there. Exited containers are retained so exit state remains
queryable. Status includes GPU utilization/memory, free disk, job states/exit
codes and the last five log lines. SSH disconnects do not terminate jobs.

Collect transfers the recorded run directory and metadata/logs into temporary
job-specific staging on the destination filesystem. It compares every collected
file's size and SHA-256 with a remote manifest, then rechecks that manifest before
promoting files. Existing identical local files are accepted; differing files
are reported as conflicts and require explicit manual resolution. Intermediate
`step_*.pt` files are excluded by default; `--include-step-checkpoints` includes
them. After verification and promotion, collection acknowledges the remote job
and atomically updates local `runs/_jobs/ledger.json`, keyed by host/project/job.
Staging is removed on success or failure. `pending` and `--all-finished` use this
ledger, including failed/stopped jobs. Failed transfers or conflicts leave new
collections pending; a failed explicit re-collection invalidates its old local
success entry too. Missing containers have unknown state and are not
automatically treated as finished. Jobs started outside this tool have no managed
metadata and need manual discovery/collection.

`prune-checkpoints` requires explicitly selected remote `runs/...` directories
belonging only to finished jobs. It is a dry run unless `--apply` is supplied,
and reports candidate paths and bytes that would be/were freed. It deletes only
direct numeric `step_*.pt` intermediates, preserving `last.pt`, configs, history,
manifests and all other files. `--keep-latest N` retains the highest numeric step
numbers; repeat `--keep` with a basename (for every selected run) or a full
project-relative path inside a selected run to protect specific checkpoints.
Traversal, paths outside `runs/`, `runs/_jobs`, and symlinks anywhere in selected
trees are refused before any deletion. The frozen milestone outside `runs/` is
ineligible. Pruning shares the submission lock so managed launches cannot race
its active-job check.

**Host safety:** unrelated long-lived containers (including
`nvidia-workbench-web` and `nvidia-workbench-api`) and other projects must remain
untouched. Stop filters and verifies both `rawdenoise.managed=gpu_jobs` and
`rawdenoise.job=JOB_ID`, then stops only the exact container ID. Never use
`docker kill $(docker ps -q)` or broad container cleanup. Never use
`pkill -f <pattern>` inside an SSH shell: it can match the SSH-spawned shell
and kill the session. Use verified container IDs/labels or explicit PIDs.
Collection never deletes remote outputs or containers; review disk use and
arrange explicit, job-specific cleanup separately.

## Export to ShopPhoto

1. `scripts/export_coreml.py`: trace with a fixed `1×4×512×512` input named
   `inp_1`, fp16 compute, ML program. Use a **fixed shape**, because dynamic
   shapes fall off the ANE.
2. Record the exported model path, input contract and camera support in an
   owner-controlled deployment manifest.
3. `scripts/export_onnx.py`: opset ≥ 17, so LayerNorm stays a native op. Used
   by tract on Windows and Linux.
4. Before each release:
   - Compare CPU fp32 against ANE fp16 on high-ISO tiles (max abs diff
     < 1e-3 in the linear domain).
   - Check op placement with `MLComputePlan` or the Xcode performance report.
   - Record full-frame latency and exported artifact size for each checkpoint.
5. Optional: weight palettisation or int8, if quality holds.

## ShopPhoto integration

- **Phase A:** configure an owner-controlled deployment manifest for our models,
  then validate ShopPhoto selection and inference against the exported contract.
- **Phase B:** add a conditioned model type to `sp-ai`:
  - 6 input channels.
  - The runtime computes the noise planes from ISO and the calibration YAML.
  - The selection path is keyed on make + sensor.
  - This also lets the "Generic Sony" path ship publicly.
