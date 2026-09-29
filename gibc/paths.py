"""Location of large local artifacts: raw data, tokenized data, checkpoints, exports.

GIBC_WORK_DIR must be set explicitly and must point outside both the repository and OneDrive.
There is deliberately no fallback, so multi-GB artifacts never land in the synced repo by accident.
The Hugging Face cache is routed separately via the HF_HOME environment variable (see CLAUDE.md),
because it must be set before any HF library is imported.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Windows sets these for personal and work/school OneDrive accounts.
_ONEDRIVE_ENV_VARS = ("OneDrive", "OneDriveConsumer", "OneDriveCommercial")


def work_dir() -> Path:
    raw = os.environ.get("GIBC_WORK_DIR", "").strip()
    if not raw:
        raise RuntimeError(
            "GIBC_WORK_DIR is not set. Point it at a directory outside the repo and outside "
            "OneDrive, e.g. C:\\gibc-work."
        )
    path = Path(raw).expanduser().resolve()
    if path.is_relative_to(REPO_ROOT):
        raise RuntimeError(f"GIBC_WORK_DIR ({path}) is inside the repository ({REPO_ROOT}).")
    for var in _ONEDRIVE_ENV_VARS:
        onedrive = os.environ.get(var, "").strip()
        if onedrive and path.is_relative_to(Path(onedrive).resolve()):
            raise RuntimeError(f"GIBC_WORK_DIR ({path}) is inside OneDrive ({onedrive}).")
    return path
