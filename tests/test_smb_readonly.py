"""The NAS inspection must never request write access or create a file."""
import importlib.util
import io
from pathlib import Path

import pytest

pytest.importorskip("impacket")
spec = importlib.util.spec_from_file_location("sample_smb_metadata", Path(__file__).parents[1] / "scripts/sample_smb_metadata.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class FakeSMB:
    def __init__(self):
        self.content = bytes(range(256)) * 512
        self.options = None
        self.closed = False

    def openFile(self, tree, path, **options):
        self.options = options
        return "handle"

    def readFile(self, tree, handle, offset, bytesToRead, singleCall):
        return self.content[offset:offset+bytesToRead]

    def closeFile(self, tree, handle):
        self.closed = True


def test_smb_handle_is_existing_file_read_only():
    connection = FakeSMB()
    with module.ReadOnlySMBFile(connection, "tree", "photo.ARW", len(connection.content)) as source:
        assert connection.options["desiredAccess"] == module.FILE_READ_DATA | module.FILE_READ_ATTRIBUTES
        assert connection.options["creationDisposition"] == module.FILE_OPEN
        assert source.writable() is False
        with pytest.raises(io.UnsupportedOperation):
            source.write(b"forbidden")
        with pytest.raises(io.UnsupportedOperation):
            source.truncate(0)
        source.seek(65530)
        assert source.read(20) == connection.content[65530:65550]
        source.seek(-4, 2)
        assert source.read() == connection.content[-4:]
    assert connection.closed


def test_metadata_budget_limits_remote_reads():
    connection = FakeSMB()
    with module.ReadOnlySMBFile(connection, "tree", "photo.ARW", len(connection.content), budget=65536) as source:
        assert source.read(1) == b"\x00"
        source.seek(65536)
        with pytest.raises(ValueError, match="budget"):
            source.read(1)
