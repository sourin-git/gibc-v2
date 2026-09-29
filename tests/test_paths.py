import pytest

from gibc.paths import REPO_ROOT, require_hf_home, work_dir


@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    for var in ("GIBC_WORK_DIR", "HF_HOME", "OneDrive", "OneDriveConsumer", "OneDriveCommercial"):
        monkeypatch.delenv(var, raising=False)


def test_unset_is_an_error():
    with pytest.raises(RuntimeError, match="GIBC_WORK_DIR is not set"):
        work_dir()
    with pytest.raises(RuntimeError, match="HF_HOME is not set"):
        require_hf_home()


def test_inside_repo_rejected(monkeypatch):
    monkeypatch.setenv("GIBC_WORK_DIR", str(REPO_ROOT / "work"))
    with pytest.raises(RuntimeError, match="inside the repository"):
        work_dir()


def test_inside_onedrive_rejected(monkeypatch, tmp_path):
    monkeypatch.setenv("OneDrive", str(tmp_path / "OneDrive"))
    monkeypatch.setenv("GIBC_WORK_DIR", str(tmp_path / "OneDrive" / "work"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "OneDrive" / "hf"))
    with pytest.raises(RuntimeError, match="inside OneDrive"):
        work_dir()
    with pytest.raises(RuntimeError, match="inside OneDrive"):
        require_hf_home()


def test_external_dir_accepted(monkeypatch, tmp_path):
    monkeypatch.setenv("OneDrive", str(tmp_path / "OneDrive"))
    monkeypatch.setenv("GIBC_WORK_DIR", str(tmp_path / "gibc-work"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "gibc-work" / "hf_home"))
    assert work_dir() == (tmp_path / "gibc-work").resolve()
    assert require_hf_home() == (tmp_path / "gibc-work" / "hf_home").resolve()
