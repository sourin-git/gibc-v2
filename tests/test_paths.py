import pytest

from gibc.paths import REPO_ROOT, work_dir


@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    for var in ("GIBC_WORK_DIR", "OneDrive", "OneDriveConsumer", "OneDriveCommercial"):
        monkeypatch.delenv(var, raising=False)


def test_unset_is_an_error():
    with pytest.raises(RuntimeError, match="not set"):
        work_dir()


def test_inside_repo_rejected(monkeypatch):
    monkeypatch.setenv("GIBC_WORK_DIR", str(REPO_ROOT / "work"))
    with pytest.raises(RuntimeError, match="inside the repository"):
        work_dir()


def test_inside_onedrive_rejected(monkeypatch, tmp_path):
    monkeypatch.setenv("OneDrive", str(tmp_path / "OneDrive"))
    monkeypatch.setenv("GIBC_WORK_DIR", str(tmp_path / "OneDrive" / "work"))
    with pytest.raises(RuntimeError, match="inside OneDrive"):
        work_dir()


def test_external_dir_accepted(monkeypatch, tmp_path):
    monkeypatch.setenv("OneDrive", str(tmp_path / "OneDrive"))
    monkeypatch.setenv("GIBC_WORK_DIR", str(tmp_path / "gibc-work"))
    assert work_dir() == (tmp_path / "gibc-work").resolve()
