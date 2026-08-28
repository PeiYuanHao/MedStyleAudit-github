"""Shared high-throughput DataLoader settings for local GPU execution."""

from __future__ import annotations

from typing import Any, Mapping


def loader_kwargs(settings: Mapping[str, Any], device: str) -> dict[str, Any]:
    """Return safe DataLoader performance options from a training config."""
    workers = int(settings.get("num_workers", 0))
    cuda = str(device).startswith("cuda")
    kwargs: dict[str, Any] = {
        "batch_size": int(settings["batch_size"]),
        "num_workers": workers,
        "pin_memory": bool(settings.get("pin_memory", cuda)) and cuda,
    }
    if workers > 0:
        kwargs["persistent_workers"] = bool(settings.get("persistent_workers", True))
        kwargs["prefetch_factor"] = int(settings.get("prefetch_factor", 4))
    return kwargs
