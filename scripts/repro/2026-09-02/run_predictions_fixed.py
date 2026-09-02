from __future__ import annotations

import runpy
import sys

import torch
import medstyleaudit.models.trainer as trainer


ORIGINAL_BUILD_MODEL = trainer.build_model


def get_arg(name: str, default: str) -> str:
    if name in sys.argv:
        i = sys.argv.index(name)
        if i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return default


DEVICE = get_arg("--device", "cuda:0")


def fixed_build_model(config):
    model = ORIGINAL_BUILD_MODEL(config)

    model = model.to(DEVICE)

    if (
        str(DEVICE).startswith("cuda")
        and bool(config["training"].get("channels_last", False))
    ):
        model = model.to(
            memory_format=torch.channels_last
        )

    p = next(model.parameters())

    print(
        "[prediction-fix] model device =",
        p.device,
        flush=True,
    )

    print(
        "[prediction-fix] model dtype  =",
        p.dtype,
        flush=True,
    )

    return model


trainer.build_model = fixed_build_model

runpy.run_path(
    "/root/MedStyleAudit-github/scripts/05_train_erm.py",
    run_name="__main__",
)
