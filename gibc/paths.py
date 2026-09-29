"""Location of large local artifacts: raw data, tokenized data, checkpoints, exports.

GIBC_WORK_DIR must be set explicitly and must point outside both the repository and OneDrive.
There is deliberately no fallback, so multi-GB artifacts never land in the synced repo by accident.
The Hugging Face cache is routed via HF_HOME, which HF libraries read at import time, so it must
be set in the environment; `require_hf_home` checks it before any HF import.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Windows sets these for personal and work/school OneDrive accounts.
_ONEDRIVE_ENV_VARS = ("OneDrive", "OneDriveConsumer", "OneDriveCommercial")


def _external_dir_from_env(var: str) -> Path:
    raw = os.environ.get(var, "").strip()
    if not raw:
        raise RuntimeError(
            f"{var} is not set. Point it at a directory outside the repo and outside "
            "OneDrive (see CLAUDE.md, 'Work directory')."
        )
    path = Path(raw).expanduser().resolve()
    if path.is_relative_to(REPO_ROOT):
        raise RuntimeError(f"{var} ({path}) is inside the repository ({REPO_ROOT}).")
    for onedrive_var in _ONEDRIVE_ENV_VARS:
        onedrive = os.environ.get(onedrive_var, "").strip()
        if onedrive and path.is_relative_to(Path(onedrive).resolve()):
            raise RuntimeError(f"{var} ({path}) is inside OneDrive ({onedrive}).")
    return path


def work_dir() -> Path:
    return _external_dir_from_env("GIBC_WORK_DIR")


def require_hf_home() -> Path:
    """Call before importing any Hugging Face library, so caches never default into the profile/repo."""
    return _external_dir_from_env("HF_HOME")
