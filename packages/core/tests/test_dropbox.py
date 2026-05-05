import sys
import types
from unittest.mock import MagicMock

import pytest


def _install_fake_dropbox(monkeypatch):
    """Tạo module dropbox giả vì SDK thật có thể chưa cài trong CI."""
    if "dropbox" in sys.modules:
        return sys.modules["dropbox"]

    fake = types.ModuleType("dropbox")
    files_mod = types.ModuleType("dropbox.files")
    exc_mod = types.ModuleType("dropbox.exceptions")

    class FileMetadata:
        def __init__(self, name, path_display, size):
            self.name = name
            self.path_display = path_display
            self.size = size

    class ApiError(Exception):
        pass

    class AuthError(Exception):
        pass

    files_mod.FileMetadata = FileMetadata
    exc_mod.ApiError = ApiError
    exc_mod.AuthError = AuthError

    class Dropbox:
        def __init__(self, *a, **kw):
            pass

    fake.Dropbox = Dropbox
    fake.files = files_mod
    fake.exceptions = exc_mod

    monkeypatch.setitem(sys.modules, "dropbox", fake)
    monkeypatch.setitem(sys.modules, "dropbox.files", files_mod)
    monkeypatch.setitem(sys.modules, "dropbox.exceptions", exc_mod)
    return fake


def test_missing_token_raises(monkeypatch):
    monkeypatch.delenv("DROPBOX_ACCESS_TOKEN", raising=False)
    _install_fake_dropbox(monkeypatch)
    from pps_core.dropbox_client import DropboxClient, DropboxError

    with pytest.raises(DropboxError, match="DROPBOX_ACCESS_TOKEN"):
        DropboxClient()


def test_list_folder_extracts_files(monkeypatch):
    fake = _install_fake_dropbox(monkeypatch)
    from pps_core.dropbox_client import DropboxClient

    result = MagicMock()
    result.entries = [
        fake.files.FileMetadata("a.png", "/a.png", 1024 * 1024),
        fake.files.FileMetadata("b.jpg", "/b.jpg", 2 * 1024 * 1024),
        object(),  # FolderMetadata or other — phải bị bỏ qua
    ]
    result.has_more = False

    client = DropboxClient(access_token="fake-token")
    client._dbx = MagicMock()
    client._dbx.files_list_folder.return_value = result

    files = client.list_folder("")
    assert len(files) == 2
    assert files[0].name == "a.png"
    assert files[0].size_mb == pytest.approx(1.0)


def test_list_folder_paginates(monkeypatch):
    fake = _install_fake_dropbox(monkeypatch)
    from pps_core.dropbox_client import DropboxClient

    page1 = MagicMock()
    page1.entries = [fake.files.FileMetadata("a", "/a", 1)]
    page1.has_more = True
    page1.cursor = "C1"

    page2 = MagicMock()
    page2.entries = [fake.files.FileMetadata("b", "/b", 2)]
    page2.has_more = False
    page2.cursor = "C2"

    client = DropboxClient(access_token="fake-token")
    client._dbx = MagicMock()
    client._dbx.files_list_folder.return_value = page1
    client._dbx.files_list_folder_continue.return_value = page2

    files = client.list_folder("")
    names = [f.name for f in files]
    assert names == ["a", "b"]
    client._dbx.files_list_folder_continue.assert_called_once_with("C1")


def test_download_writes_file(monkeypatch, tmp_path):
    _install_fake_dropbox(monkeypatch)
    from pps_core.dropbox_client import DropboxClient

    metadata = MagicMock(size=4)
    response = MagicMock()
    response.iter_content.return_value = [b"DATA"]

    client = DropboxClient(access_token="fake-token")
    client._dbx = MagicMock()
    client._dbx.files_download.return_value = (metadata, response)

    target = client.download("/foo.png", out_dir=tmp_path, show_progress=False)
    assert target.exists()
    assert target.read_bytes() == b"DATA"
    response.close.assert_called_once()
