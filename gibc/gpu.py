"""Read-only GPU telemetry via nvidia-smi (temperature, SM clock, power). Never changes settings."""

from __future__ import annotations

import subprocess
from typing import Any


def gpu_sample() -> dict[str, Any]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu,clocks.sm,power.draw", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=10).stdout.strip().split(", ")
        return {"temp_c": float(out[0]), "sm_clock_mhz": float(out[1]), "power_w": float(out[2])}
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return {}
