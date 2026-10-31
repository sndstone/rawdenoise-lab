"""HTTP demo integration: real loopback requests, bounded uploads, isolated fake CLI worker."""
import http.client
import json
from pathlib import Path
import threading
import time
from urllib.parse import urlencode

import pytest

from denoise_lab.demo import DemoServer


@pytest.fixture
def demo(tmp_path):
    model = tmp_path/'model.pt'
    model.write_bytes(b'trusted test fixture')
    commands = []
    def worker(command, log):
        commands.append(command)
        source = Path(command[3])
        output_format = command[command.index('--format')+1]
        out = Path(command[command.index('--output-dir')+1])
        (out/f'{source.stem}_denoised.{output_format}').write_bytes(b'processed:'+source.read_bytes())
        log.write_text('producing_backend=cpu test worker\n')
        return 0
    server = DemoServer(model, tmp_path/'session', worker=worker, max_upload=100)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server, commands
    server.shutdown();server.server_close();thread.join()


def request(s, method, path, body=None, headers=None):
    c = http.client.HTTPConnection('127.0.0.1', s.server_port, timeout=10)
    h = {'X-Demo-Token': s.token, 'Origin': s.origin}
    h.update(headers or {})
    c.request(method, path, body=body, headers=h)
    response = c.getresponse()
    code, content = response.status, response.read()
    c.close()
    return code, content


def submit(s, name='photo.dng', fmt='dng', body=b'raw'):
    return request(s, 'POST', '/jobs?'+urlencode(dict(filename=name, format=fmt)), body)


@pytest.mark.parametrize('fmt', ['dng', 'tiff'])
def test_upload_process_export_preserves_source(demo, fmt):
    s, commands = demo
    code, page = request(s, 'GET', '/')
    assert code == 200 and b'Import RAW photo' in page and s.token.encode() in page
    code, data = submit(s, name='a photo.DNG', fmt=fmt)
    assert code == 202
    ident = json.loads(data)['id']
    for _ in range(100):
        code, data = request(s, 'GET', '/jobs/'+ident)
        state = json.loads(data)
        if state['state'] != 'processing':break
        time.sleep(.01)
    assert state['state'] == 'complete'
    assert 'producing_backend=cpu' in state['log']
    code, exported = request(s, 'GET', state['download'])
    assert code == 200 and exported == b'processed:raw'
    assert (s.work_dir/ident/'a photo.DNG').read_bytes() == b'raw'
    assert '--no-prepass' in commands[0]
    assert commands[0][commands[0].index('--device')+1] == 'cpu'
    assert commands[0][commands[0].index('--cpu-threads')+1] == '8'
    assert '--experimental-mps-body' not in commands[0]


@pytest.mark.parametrize('name', ['../x.dng', 'a\\x.nef', 'C:x.nef', 'a\nx.nef', 'photo.jpg'])
def test_unsafe_filename_rejected(demo, name):
    s, _ = demo
    assert submit(s, name)[0] == 400
    assert not s.jobs


def test_origin_token_host_and_upload_bounds(demo):
    s, _ = demo
    url='/jobs?filename=x.dng&format=dng'
    assert request(s, 'POST', url, b'raw', {'Origin':'https://elsewhere.invalid'})[0] == 403
    assert request(s, 'POST', url, b'raw', {'X-Demo-Token':'wrong'})[0] == 403
    assert request(s, 'GET', '/', headers={'Host':'attacker.invalid'})[0] == 403
    assert submit(s, body=b'x'*101)[0] == 413
    assert submit(s, body=b'')[0] == 413
    assert submit(s, fmt='npy')[0] == 400
    assert request(s, 'GET', '/jobs/notreal')[0] == 404
    assert request(s, 'GET', '/jobs/notreal', headers={'X-Demo-Token':'wrong'})[0] == 403
    assert request(s, 'GET', '/../../model.pt')[0] == 404


def test_busy_and_failure(demo):
    s, _ = demo
    event=threading.Event()
    def fail(command, log):
        event.wait(5);log.write_text('failure fixture');return 1
    s.worker=fail
    code,data=submit(s)
    ident=json.loads(data)['id']
    assert request(s, 'GET', f'/jobs/{ident}/download?token={s.token}')[0] == 409
    assert submit(s)[0] == 409
    event.set()
    for _ in range(100):
        state=json.loads(request(s,'GET','/jobs/'+ident)[1])
        if state['state']=='failed':break
        time.sleep(.01)
    assert state['state']=='failed'
    assert request(s,'GET',f'/jobs/{ident}/download?token={s.token}')[0]==409
    s.max_jobs=1
    assert submit(s)[0]==409


def test_gpu_explicit_environment_and_command(tmp_path, monkeypatch):
    p=tmp_path/'model.pt';p.write_bytes(b'fixture')
    monkeypatch.delenv('PYTORCH_ENABLE_MPS_FALLBACK', raising=False)
    with pytest.raises(ValueError, match='before startup'):
        DemoServer(p,tmp_path/'unused',gpu=True)
    monkeypatch.setenv('PYTORCH_ENABLE_MPS_FALLBACK','0')
    monkeypatch.setenv('PYTORCH_MPS_FAST_MATH','0')
    s=DemoServer(p,tmp_path/'gpu',gpu=True)
    try:
        cmd=s.command(tmp_path/'x.dng','dng',tmp_path)
        assert '--experimental-mps-body' in cmd
        assert cmd[cmd.index('--device')+1]=='mps'
    finally:s.server_close()


def persistent_fixture(pipe, model, gpu):
    """Run the real serial worker loop with model/CLI mocks, no inference."""
    from unittest.mock import patch
    from types import SimpleNamespace
    from denoise_lab.demo import _persistent_loop
    loads = 0
    def load(*args, **kwargs):
        nonlocal loads
        loads += 1
        return SimpleNamespace(provenance={}, description='mock')
    def cli(argv, *, denoiser):
        import os
        source=Path(argv[0])
        if source.stem=='crash':os._exit(7)
        out=Path(argv[argv.index('--output-dir')+1])
        fmt=argv[argv.index('--format')+1]
        (out/f'{source.stem}_denoised.{fmt}').write_bytes(source.read_bytes())
        print(f'loads={loads}')
        print('  Done in 0.00s (cpu mock)')
        if source.stem=='fallback':denoiser.provenance={'fallback_reason':'fixture'}
    with patch('denoise_lab.inference.Denoiser.load', side_effect=load), patch('denoise_lab.cli.main', side_effect=cli):
        _persistent_loop(pipe, model, gpu)


def test_persistent_reuse_fallback_next_job_and_process_death(tmp_path):
    from denoise_lab.demo import PersistentWorker
    worker=PersistentWorker(tmp_path/'model.pt',False,target=persistent_fixture,timeout=30)
    def run(name):
        root=tmp_path/name;root.mkdir();source=root/f'{name}.dng';source.write_bytes(name.encode())
        command=['python','-m','denoise_lab.cli',str(source),'--format','dng','--output-dir',str(root)]
        log=root/'processing.log'
        result=worker(command, log)
        assert (root/f'{name}_denoised.dng').read_bytes()==name.encode()
        return result,log.read_text()
    try:
        assert 'loads=1' in run('first')[1]
        pid=worker.process.pid
        assert 'loads=1' in run('second')[1]
        assert worker.process.pid==pid
        assert 'loads=1' in run('fallback')[1]
        assert 'loads=1' in run('after_fallback')[1]
        assert worker.process.pid!=pid
        with pytest.raises((EOFError, OSError)):
            run('crash')
        assert worker.process is None
        assert 'loads=1' in run('after_crash')[1]
    finally:worker.close()


def test_session_preflight_prevents_large_busy_upload(demo):
    s,_=demo
    s.max_upload=10*1024**2
    event=threading.Event()
    def wait(command,log):event.wait(5);return 1
    s.worker=wait
    try:
        assert submit(s)[0]==202
        code,data=request(s,'GET','/session')
        state=json.loads(data)
        assert code==200 and state['busy'] and state['job_count']==1 and state['limit']==8
        assert request(s,'GET','/session',headers={'X-Demo-Token':'wrong'})[0]==403
        # A browser does this GET before sending its multi-MB File body.
        assert b"fetch('/session'" in request(s,'GET','/')[1]
        # For direct clients bypassing preflight, receive 409 before sending body.
        c=http.client.HTTPConnection('127.0.0.1',s.server_port,timeout=10)
        c.putrequest('POST','/jobs?filename=large.dng&format=dng')
        for k,v in {'X-Demo-Token':s.token,'Origin':s.origin,'Content-Length':str(3*1024**2)}.items():c.putheader(k,v)
        c.endheaders();r=c.getresponse();assert r.status==409;r.read();c.close()
    finally:event.set()


def test_truncated_upload_cleans_partial_directory_and_unlocks(demo):
    import socket
    s,_=demo
    conn=socket.create_connection(('127.0.0.1',s.server_port))
    data=(f'POST /jobs?filename=broken.dng&format=dng HTTP/1.0\r\nHost: 127.0.0.1:{s.server_port}\r\nOrigin: {s.origin}\r\nX-Demo-Token: {s.token}\r\nContent-Length: 20\r\n\r\nraw').encode()
    conn.sendall(data);conn.shutdown(socket.SHUT_WR)
    response=conn.recv(65536);conn.close()
    assert b'400' in response
    for _ in range(100):
        if not s.busy.locked():break
        time.sleep(.01)
    assert not s.jobs and not s.busy.locked()
    assert not list(s.work_dir.iterdir())


def test_real_subprocess_cli_worker_and_backend_download(demo,tmp_path):
    import sys
    from denoise_lab.demo import cli_worker
    s,_=demo
    log=tmp_path/'subprocess.log'
    assert cli_worker([sys.executable,'-c',"print('fixture subprocess')"],log)==0
    assert 'fixture subprocess' in log.read_text()
    assert cli_worker([sys.executable,'-c','raise SystemExit(2)'],tmp_path/'failure.log')==2
    def worker(command,log):
        source=Path(command[3]);(source.parent/f'{source.stem}_denoised.dng').write_bytes(b'output')
        log.write_text('  Done in 1.00s (torch: cpu fallback (experimental MPS body; not quality-accepted))\n')
        return 0
    s.worker=worker
    code,data=submit(s,name='café.dng');ident=json.loads(data)['id']
    for _ in range(100):
        state=json.loads(request(s,'GET','/jobs/'+ident)[1])
        if state['state']=='complete':break
        time.sleep(.01)
    assert 'cpu fallback' in state['backend']
    assert request(s,'GET',f'/jobs/{ident}/download?token=wrong')[0]==403
    c=http.client.HTTPConnection('127.0.0.1',s.server_port)
    c.request('GET',state['download']);r=c.getresponse()
    assert "filename*=UTF-8''caf%C3%A9_denoised_" in r.getheader('Content-Disposition')
    assert ident[:8] in r.getheader('Content-Disposition')
    r.read();c.close()


def timeout_fixture(pipe, model, gpu):
    pipe.recv()
    time.sleep(60)


def test_persistent_timeout_is_not_retried(tmp_path):
    from denoise_lab.demo import PersistentWorker
    worker=PersistentWorker(tmp_path/'model.pt',False,target=timeout_fixture,timeout=.05)
    with pytest.raises(TimeoutError,match='not retried'):
        worker(['python','-m','module','photo.dng'],tmp_path/'unused.log')
    assert worker.process is None and not (tmp_path/'unused.log').exists()


def test_injected_cli_keeps_imagekind_validation(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from denoise_lab import cli
    monkeypatch.setattr(cli.Denoiser,'load',lambda *a,**k:pytest.fail('must reuse injected model'))
    # Wrong image kind is rejected before rawpy opens the nonexistent RAW.
    with pytest.raises(SystemExit) as exc:
        cli.main([str(tmp_path/'photo.dng'),'--model','unused.pt','--device','cpu'],
                 denoiser=SimpleNamespace(description='fixtureRGB', image_kind='rgb'))
    assert exc.value.code==1
