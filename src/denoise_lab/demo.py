"""Local RAW import/process/export demo; uses the existing CLI in an isolated worker."""
import argparse
from contextlib import redirect_stdout, redirect_stderr
import json
import multiprocessing
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import secrets
import shutil
import subprocess
import sys
import threading
from urllib.parse import parse_qs, quote, urlencode, urlsplit
import uuid


PAGE = '''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>RAW denoising demo</title>
<style>body{font:16px system-ui,sans-serif;max-width:760px;margin:40px auto;padding:0 20px;color:#202832;background:#f5f7fa}main{background:#fff;padding:28px;border-radius:12px}h1{font-size:26px;margin-top:0}label{display:block;margin:24px 0 8px}input,select,button{font:inherit;max-width:100%;padding:10px}button,a{margin-top:20px}button{background:#174da8;color:white;border:0;border-radius:6px;cursor:pointer}button:disabled{opacity:.5}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#eef1f6;padding:16px;font-size:13px}#download{display:block}#download[hidden]{display:none}small{display:block;line-height:1.5;color:#465361}p{line-height:1.5}</style></head>
<body><main><h1>RAW denoising demo</h1><p>Import a Bayer RAW photograph, process it locally, then export the result.</p>
<small>__MODE__ · Keep your original RAW. Other cameras and denoising quality remain unvalidated. This is a research demo, not a native app.</small>
<form id="form"><label for="file">1. Import RAW photo</label><input id="file" type="file" required accept=".dng,.3fr,.fff,.nef,.arw,.cr2,.cr3,.rw2,.orf,.pef,.raf,.srw,.raw">
<label for="format">2. Export format</label><select id="format"><option value="dng">DNG — processed sensor mosaic</option><option value="tiff">16-bit TIFF — rendered sRGB</option></select>
<small>DNG retains essential sensor/color metadata, not full EXIF, lens profiles or original previews. TIFF is a rendered image, not RAW.</small>
<button id="process" type="submit">Process photo</button></form><p id="status" role="status" aria-live="polite">Choose a file to begin. Maximum upload: __MAX__ MiB.</p><a id="download" hidden>3. Export processed result</a>
<details><summary>Processing log</summary><pre id="log"></pre></details></main>
<script>const token=__TOKEN__;const form=document.getElementById('form'),status=document.getElementById('status'),log=document.getElementById('log'),button=document.getElementById('process'),download=document.getElementById('download');
form.addEventListener('submit',async e=>{e.preventDefault();const file=document.getElementById('file').files[0];if(!file)return;button.disabled=true;download.hidden=true;log.textContent='';status.textContent='Importing photo…';
try{const check=await fetch('/session',{headers:{'X-Demo-Token':token}});const session=await check.json();if(!check.ok)throw Error(session.error);if(session.busy)throw Error('Another photo is processing; wait for it to finish.');if(session.job_count>=session.limit)throw Error('Session job limit reached; restart for a new session.');const response=await fetch('/jobs?'+new URLSearchParams({filename:file.name,format:document.getElementById('format').value}),{method:'POST',headers:{'X-Demo-Token':token,'Content-Type':'application/octet-stream'},body:file});const job=await response.json();if(!response.ok)throw Error(job.error);status.textContent='Processing locally…';
let complete=false;while(!complete){await new Promise(r=>setTimeout(r,750));const r=await fetch('/jobs/'+job.id,{headers:{'X-Demo-Token':token}});const j=await r.json();if(!r.ok)throw Error(j.error);log.textContent=j.log||'';if(j.state==='complete'){complete=true;status.textContent='Finished using '+(j.backend||'backend in log')+'. Export the result below.';download.href=j.download;download.hidden=false;}else if(j.state==='failed'){throw Error(j.error||'Processing failed; see log.');}}
}catch(error){status.textContent=error.message;}finally{button.disabled=false;}});</script></body></html>'''


RAW_SUFFIXES = {'.dng', '.3fr', '.fff', '.nef', '.arw', '.cr2', '.cr3', '.rw2', '.orf', '.pef', '.raf', '.srw', '.raw'}


def cli_worker(command, log_path):
    with log_path.open('xb') as log:
        return subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False, timeout=1800).returncode


def _persistent_loop(pipe, model, gpu):
    """One serialized model owner; imports and tensors stay outside HTTP threads."""
    import torch
    from .cli import main as process_cli
    from .inference import Denoiser
    torch.set_num_threads(8)
    denoiser = None
    while True:
        try:
            argv, log_path = pipe.recv()
        except EOFError:
            return
        code, retire = 0, False
        with Path(log_path).open('x') as log, redirect_stdout(log), redirect_stderr(log):
            try:
                if denoiser is None:
                    denoiser = Denoiser.load(model, device='mps' if gpu else 'cpu', experimental_mps_body=gpu)
                process_cli(argv, denoiser=denoiser)
            except SystemExit as exc:
                code = exc.code if isinstance(exc.code, int) else 1
            except Exception as exc:
                print(f'Worker failed: {type(exc).__name__}: {exc}', flush=True)
                code = 1
            finally:
                provenance = getattr(denoiser, 'provenance', None)
                # Preserve this image's CPU result, then retire its device context.
                if provenance and provenance.get('fallback_reason'):
                    retire = True
        pipe.send((code, retire))
        if retire:
            return


class PersistentWorker:
    def __init__(self, model, gpu, *, timeout=1800, target=_persistent_loop):
        self.model, self.gpu, self.timeout, self.target = str(model), gpu, timeout, target
        self.process = self.pipe = None

    def close(self):
        if self.pipe is not None:
            self.pipe.close()
        if self.process is not None:
            if self.process.is_alive():
                self.process.terminate()
            self.process.join(timeout=5)
        self.process = self.pipe = None

    def __call__(self, command, log_path):
        if self.process is None or not self.process.is_alive():
            self.close()
            context = multiprocessing.get_context('spawn')
            self.pipe, child = context.Pipe()
            self.process = context.Process(target=self.target, args=(child, self.model, self.gpu), daemon=True)
            self.process.start()
            child.close()
        try:
            self.pipe.send((command[3:], str(log_path)))
            if not self.pipe.poll(self.timeout):
                raise TimeoutError('Persistent worker timed out; this image was not retried')
            code, retire = self.pipe.recv()
            if retire:
                self.close()
            return code
        except (EOFError, BrokenPipeError, OSError, TimeoutError):
            self.close()
            raise


class DemoServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, model, work_dir, *, port=0, gpu=False, max_upload=1024*1024*1024,
                 max_jobs=8, worker=cli_worker, persistent=False):
        self.model = Path(model).resolve()
        if not self.model.is_file() or self.model.suffix.lower() != '.pt':
            raise ValueError('The demo requires a trusted local Bayer .pt checkpoint')
        if gpu:
            # Validate without importing Torch; controls must precede worker startup.
            for key in ('PYTORCH_ENABLE_MPS_FALLBACK', 'PYTORCH_MPS_FAST_MATH'):
                if os.environ.get(key) != '0':
                    raise ValueError(f'Experimental GPU demo requires {key}=0 before startup')
        self.gpu, self.max_upload, self.max_jobs = gpu, max_upload, max_jobs
        self.work_dir = Path(work_dir).resolve()
        self.work_dir.mkdir(parents=True, exist_ok=False)
        self.worker = PersistentWorker(self.model, gpu) if persistent else worker
        self.token = secrets.token_urlsafe(32)
        self.jobs = {}
        self.busy = threading.Lock()
        super().__init__(('127.0.0.1', port), DemoHandler)
        self.origin = f'http://127.0.0.1:{self.server_port}'

    def server_close(self):
        if isinstance(self.worker, PersistentWorker):
            self.worker.close()
        super().server_close()

    def command(self, source, output_format, output_dir):
        return [sys.executable, '-m', 'denoise_lab.cli', str(source), '--model', str(self.model),
                '--device', 'mps' if self.gpu else 'cpu', '--cpu-threads', '8',
                '--tile-batch-size', '2', '--no-prepass', '--format', output_format,
                '--output-dir', str(output_dir)] + (['--experimental-mps-body'] if self.gpu else [])

    def process(self, job):
        try:
            code = self.worker(self.command(job['source'], job['format'], job['directory']), job['log_path'])
            if code != 0 or not job['output'].is_file():
                job.update(state='failed', error='Processing failed; see processing log.')
            else:
                job['state'] = 'complete'
        except Exception as exc:
            job.update(state='failed', error=f'Worker failed: {type(exc).__name__}: {exc}')
        finally:
            self.busy.release()


class DemoHandler(BaseHTTPRequestHandler):
    server: DemoServer

    def setup(self):
        super().setup()
        self.connection.settimeout(30)

    def log_message(self, *_args):
        pass  # Do not log capability tokens or filenames from URLs.

    def respond(self, code, value):
        data = json.dumps(value).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(data)

    def valid_host(self):
        return self.headers.get('Host') == self.server.origin.removeprefix('http://')

    def authorized(self, query=False):
        token = parse_qs(urlsplit(self.path).query).get('token', [''])[0] if query else self.headers.get('X-Demo-Token', '')
        return self.valid_host() and secrets.compare_digest(token, self.server.token)

    def do_GET(self):
        path = urlsplit(self.path).path
        if not self.valid_host():
            return self.respond(403, {'error': 'Invalid local host'})
        if path == '/session':
            if not self.authorized():
                return self.respond(403, {'error': 'Invalid session token'})
            return self.respond(200, dict(busy=self.server.busy.locked(), job_count=len(self.server.jobs), limit=self.server.max_jobs))
        if path == '/':
            mode = ('Requested: experimental GPU body + CPU transforms; actual backend is in the log; not quality-accepted' if self.server.gpu
                    else 'CPU FP32 reference mode')
            data = PAGE.replace('__MODE__', mode).replace('__TOKEN__', json.dumps(self.server.token)).replace('__MAX__', str(self.server.max_upload//1024**2)).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Referrer-Policy', 'no-referrer')
            self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'")
            self.end_headers()
            return self.wfile.write(data)
        parts = path.strip('/').split('/')
        is_download = len(parts) == 3 and parts[2] == 'download'
        if len(parts) not in {2, 3} or parts[0] != 'jobs' or (len(parts) == 3 and not is_download):
            return self.respond(404, {'error': 'Not found'})
        if not self.authorized(query=is_download):
            return self.respond(403, {'error': 'Invalid session token'})
        job = self.server.jobs.get(parts[1])
        if not job:
            return self.respond(404, {'error': 'Unknown job'})
        if is_download:
            if job['state'] != 'complete':
                return self.respond(409, {'error': 'Result not ready'})
            output = job['output']
            self.send_response(200)
            self.send_header('Content-Type', 'application/octet-stream')
            name = f'{job["source"].stem}_denoised_{parts[1][:8]}.{job["format"]}'
            self.send_header('Content-Disposition', f'attachment; filename="processed_{parts[1][:8]}.{job["format"]}"; filename*=UTF-8\'\'{quote(name, safe="")}')
            self.send_header('Content-Length', str(output.stat().st_size))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Referrer-Policy', 'no-referrer')
            self.end_headers()
            with output.open('rb') as f:
                shutil.copyfileobj(f, self.wfile, length=1024*1024)
            return
        log = ''
        if job['log_path'].exists():
            with job['log_path'].open('rb') as f:
                f.seek(max(0, job['log_path'].stat().st_size-65536))
                log = f.read(65536).decode(errors='replace')
        # CLI's final Done line names the producing backend after any fallback.
        done = [line.strip() for line in log.splitlines() if line.strip().startswith('Done in ')]
        backend = done[-1].split(' (', 1)[1].rsplit(')', 1)[0] if done and ' (' in done[-1] else None
        return self.respond(200, {'id': parts[1], 'state': job['state'], 'log': log, 'backend': backend,
                                 'error': job.get('error'), 'download': f'/jobs/{parts[1]}/download?'+urlencode({'token': self.server.token})})

    def do_POST(self):
        if not self.authorized() or self.headers.get('Origin') != self.server.origin:
            return self.respond(403, {'error': 'Invalid local session/origin'})
        u = urlsplit(self.path)
        if u.path != '/jobs':
            return self.respond(404, {'error': 'Not found'})
        q = parse_qs(u.query)
        name, fmt = q.get('filename', [''])[0], q.get('format', [''])[0]
        if (not name or len(name) > 200 or any(c in name for c in '/\\:\r\n\x00')
                or Path(name).suffix.lower() not in RAW_SUFFIXES or fmt not in {'dng', 'tiff'}):
            return self.respond(400, {'error': 'Choose a supported RAW filename and DNG or TIFF output'})
        try:
            length = int(self.headers.get('Content-Length', ''))
        except ValueError:
            return self.respond(411, {'error': 'Content-Length required'})
        if self.headers.get('Transfer-Encoding') or not 0 < length <= self.server.max_upload:
            return self.respond(413, {'error': 'Upload is empty or exceeds the configured limit'})
        if len(self.server.jobs) >= self.server.max_jobs:
            return self.respond(409, {'error': 'Session job limit reached; restart for a new session'})
        if not self.server.busy.acquire(blocking=False):
            return self.respond(409, {'error': 'Another photo is processing; wait for it to finish'})
        directory = None
        try:
            if len(self.server.jobs) >= self.server.max_jobs:
                raise ValueError('Session job limit reached; restart for a new session')
            if shutil.disk_usage(self.server.work_dir).free < length*3 + 2*1024**3:
                raise ValueError('Insufficient disk space for source and outputs')
            ident = uuid.uuid4().hex
            directory = self.server.work_dir/ident
            directory.mkdir()
            source = directory/name
            remaining = length
            with source.open('xb') as f:
                while remaining:
                    chunk = self.rfile.read(min(remaining, 1024*1024))
                    if not chunk:
                        raise ValueError('Incomplete upload')
                    f.write(chunk)
                    remaining -= len(chunk)
            job = dict(source=source, directory=directory, format=fmt, output=directory/f'{source.stem}_denoised.{fmt}',
                       log_path=directory/'processing.log', state='processing')
            self.server.jobs[ident] = job
            threading.Thread(target=self.server.process, args=(job,), daemon=True).start()
        except Exception as exc:
            if directory is not None:
                shutil.rmtree(directory)
            self.server.busy.release()
            return self.respond(400, {'error': str(exc)})
        return self.respond(202, {'id': ident})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--work-dir', type=Path, default=Path('internal/demo_runs'))
    parser.add_argument('--port', type=int, default=0)
    parser.add_argument('--experimental-mps-body', action='store_true')
    parser.add_argument('--persistent-worker', action='store_true', help='Experimental serial subprocess reuses the model across photos')
    args = parser.parse_args()
    session = args.work_dir/uuid.uuid4().hex
    try:
        server = DemoServer(args.model, session, port=args.port, gpu=args.experimental_mps_body, persistent=args.persistent_worker)
    except (OSError, ValueError) as exc:
        parser.exit(1, f'Demo startup failed: {exc}\n')
    print(f'Open {server.origin} in your browser. Local files: {session.resolve()}', flush=True)
    print('One photo at a time. Ctrl+C stops the server; session files are retained.', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
