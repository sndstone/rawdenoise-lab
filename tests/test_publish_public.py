"""Exercise private publication control flow with offline synthetic commands."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if not (ROOT/'tools/publish_public.sh').is_file():
    pytest.skip('private publication tools are not exported', allow_module_level=True)

FAKE_COMMAND = '''import io
import json
import os
from pathlib import Path
import sys
import tarfile

name = Path(sys.argv[0]).name
args = sys.argv[1:]
with open(os.environ['COMMAND_LOG'], 'a') as log:
    log.write(json.dumps([name, *args]) + '\\n')
failure = os.environ.get('FAIL_AT')
if name == 'python':
    if args[0].endswith('leak_check.py'):
        os.execv(sys.executable, [sys.executable, *args])
    assert args[:2] == ['-m', 'pytest']
    sys.exit(1 if failure == 'tests' else 0)
if name == 'rsync':
    sys.exit(1 if failure == 'rsync' else 0)
assert name == 'git'
command = args[0]
if command == 'rev-parse':
    print(os.environ['FAKE_ROOT'])
elif command in ('status', 'remote', 'add'):
    pass
elif command == 'archive':
    if failure == 'archive':
        sys.exit(1)
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w'):
        pass
    sys.stdout.buffer.write(buffer.getvalue())
elif command == 'clone':
    if failure == 'clone':
        sys.exit(1)
    Path(args[-1]).mkdir()
elif command == 'diff':
    if '--quiet' in args:
        sys.exit(0 if os.environ.get('NO_CHANGE') else 1)
elif command == '-c':
    assert 'commit' in args
    sys.exit(1 if failure == 'commit' else 0)
elif command == 'push':
    sys.exit(1 if failure == 'push' else 0)
else:
    raise AssertionError(args)
'''


@pytest.fixture
def publisher(tmp_path):
    root = tmp_path/'repo'
    tools = root/'tools'
    tools.mkdir(parents=True)
    for name in ('publish_public.sh', 'leak_check.py'):
        shutil.copy2(ROOT/'tools'/name, tools/name)
    policy = dict(deny_terms=['blocked-token'], deny_regexes=['secret[ -]host'],
                  author_email_allowlist=json.loads((ROOT/'tools/leak_check.json').read_text())[
                      'author_email_allowlist'])
    (tools/'leak_check.json').write_text(json.dumps(policy))
    binary = tmp_path/'bin'
    binary.mkdir()
    for name in ('git', 'rsync', 'python'):
        command = binary/name
        command.write_text(f'#!{sys.executable}\n' + FAKE_COMMAND)
        command.chmod(0o755)
    workspaces = tmp_path/'work spaces'
    workspaces.mkdir()
    log = tmp_path/'commands.jsonl'
    env = dict(os.environ, PATH=str(binary) + os.pathsep + os.environ['PATH'],
               PYTHON=str(binary/'python'), TMPDIR=str(workspaces),
               FAKE_ROOT=str(root), COMMAND_LOG=str(log))

    def run(message='Public update', push=False, **settings):
        result = subprocess.run(['bash', str(tools/'publish_public.sh'), message,
                                 *(['--push'] if push else [])],
                                env=dict(env, **settings), text=True, capture_output=True)
        calls = [json.loads(line) for line in log.read_text().splitlines()]
        return result, calls, list(workspaces.iterdir())
    return run


@pytest.mark.parametrize('message', ['BLOCKED-TOKEN', 'Safe title\n\nSecret Host'])
def test_denied_message_stops_before_tests_or_clone(publisher, message):
    result, calls, retained = publisher(message)
    assert result.returncode == 1
    assert '<message>:' in result.stdout
    assert not any(call[:2] in (['git', 'clone'], ['python', '-m']) for call in calls)
    assert not retained


@pytest.mark.parametrize('failure', ['archive', 'tests', 'clone', 'rsync', 'commit', 'push'])
def test_failure_removes_workspace(publisher, failure):
    result, calls, retained = publisher(push=True, FAIL_AT=failure)
    assert result.returncode != 0
    assert not retained


def test_no_change_removes_workspace(publisher):
    result, calls, retained = publisher(NO_CHANGE='1')
    assert result.returncode == 0
    assert 'nothing to publish' in result.stdout
    assert not any('commit' in call or call[:2] == ['git', 'push'] for call in calls)
    assert not retained


def test_successful_push_removes_workspace(publisher):
    result, calls, retained = publisher(push=True)
    assert result.returncode == 0
    assert any(call[:2] == ['git', 'push'] for call in calls)
    assert not retained


def test_dry_run_retains_reviewable_workspace_and_cleanup_command(publisher):
    message = 'Public update\n\nPreserve "quotes" and $literal text'
    result, calls, retained = publisher(message)
    assert result.returncode == 0
    assert len(retained) == 1 and (retained[0]/'public').is_dir()
    assert str(retained[0]/'public') in result.stdout
    assert not any(call[:2] == ['git', 'push'] for call in calls)
    commit = next(call for call in calls if 'commit' in call)
    assert commit[commit.index('-m') + 1] == message
    cleanup = result.stdout.split('After review, clean up with: ', 1)[1].strip()
    subprocess.run(['bash', '-c', cleanup], check=True)
    assert not retained[0].exists()
