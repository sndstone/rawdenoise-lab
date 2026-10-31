# Local RAW import/export demo

The repository includes a small browser app for importing one RAW photograph, running the existing Bayer denoising CLI locally, and downloading the processed result. It requires no web framework or extra Python dependencies. It is a Python demo, not a native macOS app or hosted service. Models are provided separately; choose a trusted local unconditioned Bayer `.pt` checkpoint and verify its published SHA256 before use; see [model files and checksums](../README.md#model-files).

## Start in CPU reference mode

Launch from the repository root with the project environment prepared (the default work directory is relative to your current directory):

```sh
PYTHONPATH=src .venv/bin/python -m denoise_lab.demo --model models/r4_plain_weights.pt
```

For a new installation, the existing project dependencies are sufficient:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/rawdenoise-demo --model models/r4_plain_weights.pt
```

On Windows, use `.venv\Scripts\python.exe` and the corresponding executable directory. The module command with `PYTHONPATH=src` does not require reinstalling an existing environment to register the new console entry point.

The terminal prints an address such as `http://127.0.0.1:54321`. Open the exact printed address in your browser; the random port can differ each time. The app binds only to the local loopback interface. Do not expose or proxy it onto a network. No photographs are uploaded to an external service. Only `127.0.0.1` is supported; use the printed address rather than `localhost` or a LAN address.

1. Select **Import RAW photo** and choose a rawpy-readable RGB Bayer file, such as `.DNG`, `.3FR`, `.NEF` or `.ARW`.
2. Choose **DNG** for a processed sensor mosaic, or **16-bit TIFF** for a rendered sRGB image.
3. Click **Process photo** and wait. The app imports a copy, processes one photo at a time, and displays completion or an error. Expand **Processing log** to see the actual producing backend, timings and any fallback.
4. Click **Export processed result**. Your browser downloads a uniquely named result based on the original filename, job ID and selected format. The local session also retains the source copy, original-name result and processing log.

The standalone CLI normally enables the defect prepass and uses the runtime thread count capped at eight, with tile batch 1. The demo deliberately differs. The CPU configuration is explicit: eight threads, tile batch 2, strength 1, overlap 60, and the defect prepass disabled. This avoids the legacy `auto` accelerator selection. CPU8/batch2 parity evidence is host/runtime-specific, not a universal guarantee for all environments or checkpoints. The app supports Bayer only; RGB TIFF input, X-Trans, Foveon and conditioned models are unsupported. Successfully decoding another camera does not validate its denoising quality.

## Experimental Mac GPU mode

The tested experimental configuration used M5 Pro, macOS 27, PyTorch 2.14.1 and the frozen r4 checkpoint SHA256 `cb0ab5ada18adb26a2893ee4e72ccf5fe334b98ccbe5691dc800f7804b3ab52a`. Other hosts, versions, weights and settings require their own validation; numerical canaries are diagnostics, not photographic acceptance. Export both controls before Python startup:

```sh
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0 PYTHONPATH=src \
  .venv/bin/python -m denoise_lab.demo --model models/r4_plain_weights.pt \
  --experimental-mps-body
```

This selects the explicit MPS FP32 neural body with CPU transforms and CFA-aware host tiling. It invokes the existing guarded CLI, including numeric canaries and whole-image CPU fallback for accelerator errors raised to Python. The log, rather than the requested-mode label, identifies the actual producing backend. A GPU hang or process abort is not repaired by that fallback. Each photo uses a fresh isolated worker and reloads the model/canaries, so this demo does **not** achieve the warm persistent-model timing quoted for the API. Startup can dominate small photos. It does not claim quality acceptance on arbitrary cameras, OS versions, weights or settings.

The GPU passed prospective engineering equivalence on the tested sources, but is not bit-identical and rare rendered differences remain. Owner blind acceptance is separate. NPU/FP16 prototypes failed fidelity checks, so this demo offers no NPU mode. Ordinary CLI `.pt --device auto` remains a different legacy path; it is not the explicit guarded GPU mode above.

## Optional persistent worker

Add `--persistent-worker` to either startup command to keep the model in a serial subprocess across photographs. This is opt-in; the default remains a fresh CLI subprocess per photo. The HTTP server remains responsive while the separate worker owns all model and device operations. The persistent worker calls the same CLI processing code, including geometry, output metadata and overwrite checks, with an injected already-loaded Denoiser; it does not change model arithmetic or tiling.

The local qualification checked two different 1024-pixel demo photos in both DNG and TIFF, on CPU and guarded GPU: all 16 exports matched fresh-worker bytes within their respective backend. This is implementation regression evidence, not a CPU-versus-GPU identity or photographic-quality claim. Later GPU jobs in that small-photo session took about 0.45–0.50 seconds, versus about 4 seconds with fresh workers. Those observations demonstrate startup amortization; they are not a new full-resolution throughput benchmark or a guarantee on other hardware. Do not infer a gain from the flag alone on other settings.

There is no automatic retry of a failed, dead or timed-out job. A later photo starts a new subprocess after process death. If an accelerator image falls back to CPU, that image keeps its CPU result and that worker is retired after returning its result, and the next photo starts a new subprocess/model/device context. Worker timeout is 30 minutes. Stopping the persistent server terminates its own worker; stop only after the current photo finishes if you want its output retained.

## Outputs, resource use and stopping

The original file selected in the browser is never modified. Each import gets a unique directory beneath `internal/demo_runs/`; existing files are not overwritten. Use `--work-dir PATH` to choose another session root and `--port 8765` to request a fixed local port. The session root itself is a new UUID directory. Choose custom work directories outside tracked source folders. Do not put private sources or results in tracked repository folders.

Uploads are streamed in bounded chunks, with a maximum of 1 GiB per photo and eight imported jobs per session. The app checks disk headroom before accepting an upload, rejects concurrent processing, and gives each worker a 30 minute timeout. These limits do not guarantee a RAM reserve for arbitrary source sizes. Keep other heavy jobs idle, retain ample RAM/disk headroom, and process one large image at a time. Importing a photograph makes a local copy and processing can require several times its on-disk size. A second simultaneous request is rejected instead of starting another inference.

Wait for processing to finish before pressing **Ctrl+C** in the terminal. This stops the local HTTP server. In the default fresh-worker mode, a worker already running may continue until completion or its timeout; persistent mode terminates its worker. Session files are intentionally retained; after closing the app and confirming no processing remains, remove only session folders you no longer need. Restarting the app creates a fresh session and job limit.

The DNG writer retains essential visible-sensor CFA, black/white levels, color matrix, camera WB and orientation. It does not preserve full EXIF, lens profiles, masked pixels, embedded previews or signatures. Keep the original RAW. TIFF output is 16-bit sRGB rendered with camera white balance and no automatic brightening, not a RAW mosaic or a clean reference. No eight-bit processing target or training data is created by the demo.

The app uses a per-session capability token and same-origin checks, accepts only fixed processing options, has no directory listing/file-browser endpoint, and launches the CLI with an argument list rather than a shell. This protects the intended single-user local workflow; it is not a production multi-user server. The printed page URL contains no secret token. Any local user or process can fetch that page and obtain the session token, so this is not local-user authentication. The token, strict Host check and Origin check protect against cross-origin requests and DNS rebinding, not hostile local software. Do not share download URLs containing tokens or use untrusted checkpoint files. Other models need their own validation.
