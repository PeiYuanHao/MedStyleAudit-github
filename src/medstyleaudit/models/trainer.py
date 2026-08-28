"""Single-device BCE trainer with checkpoints and persistent epoch records."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from medstyleaudit.utils.io import save_table


def build_model(config: Mapping[str, Any]):
    architecture = config["model"]["architecture"]
    pretrained = bool(config["model"].get("pretrained", False))
    if pretrained:
        raise ValueError("Primary paper protocol requires random initialization; pretrained=true is not permitted")
    if architecture == "resnet50":
        from .resnet import build_resnet50
        return build_resnet50(False)
    if architecture == "densenet121":
        from .densenet import build_densenet121
        return build_densenet121(False)
    raise ValueError(f"Unknown architecture: {architecture}")


def evaluate(model, loader, device, show_progress: bool = False, description: str = "evaluation"):
    import torch
    model.eval()
    logits, labels, identifiers = [], [], []
    with torch.no_grad():
        iterator = loader
        if show_progress:
            from tqdm.auto import tqdm
            iterator = tqdm(loader, total=len(loader), desc=description, unit="batch", leave=False)
        for batch in iterator:
            images, targets = batch[0].to(device), batch[1].float().to(device)
            output = model(images).reshape(-1)
            logits.extend(output.cpu().numpy().tolist())
            labels.extend(targets.cpu().numpy().tolist())
            if len(batch) > 2:
                ids = batch[2].detach().cpu().numpy().tolist() if hasattr(batch[2], "detach") else list(batch[2])
                identifiers.extend(ids)
            else:
                identifiers.extend(range(len(output)))
    probabilities = 1 / (1 + np.exp(-np.asarray(logits)))
    metrics = {"loss": float("nan"), "accuracy": accuracy_score(labels, probabilities >= 0.5), "auroc": roc_auc_score(labels, probabilities) if len(set(labels)) > 1 else float("nan")}
    return metrics, pd.DataFrame({"source_id": identifiers, "label": labels, "logit": logits, "probability": probabilities})


def train(model, train_loader, validation_loader, config: Mapping[str, Any], output_dir: str | Path, device: str = "cpu", dry_run: bool = False) -> pd.DataFrame:
    import torch
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    settings = config["training"]
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(settings["learning_rate"]), weight_decay=float(settings.get("weight_decay", 0)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(int(settings["epochs"]), 1))
    criterion = torch.nn.BCEWithLogitsLoss()
    start_epoch, best = 0, -float("inf") if settings.get("maximize_metric", True) else float("inf")
    resume = settings.get("resume")
    if resume:
        state = torch.load(resume, map_location=device)
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
        start_epoch, best = int(state["epoch"]) + 1, float(state["best_metric"])
    history = []
    max_batches = int(settings.get("dry_run_batches", 2)) if dry_run else None
    from tqdm.auto import tqdm
    epochs = range(start_epoch, 1 if dry_run else int(settings["epochs"]))
    for epoch in tqdm(epochs, desc="ERM epochs", unit="epoch"):
        model.train(); losses = []
        batches = tqdm(train_loader, total=len(train_loader), desc=f"train epoch {epoch + 1}", unit="batch", leave=False)
        for batch_index, batch in enumerate(batches):
            images, targets = batch[0].to(device), batch[1].float().to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images).reshape(-1), targets.reshape(-1)); loss.backward(); optimizer.step()
            losses.append(float(loss.detach().cpu()))
            if max_batches is not None and batch_index + 1 >= max_batches:
                break
        metrics, predictions = evaluate(model, validation_loader, device, True, f"validate epoch {epoch + 1}")
        record = {"epoch": epoch, "train_loss": np.mean(losses), "learning_rate": optimizer.param_groups[0]["lr"], **{f"val_{key}": value for key, value in metrics.items()}}
        history.append(record); save_table(pd.DataFrame(history), directory / "metrics.csv")
        selection = float(record[settings.get("selection_metric", "val_auroc")])
        improved = selection > best if settings.get("maximize_metric", True) else selection < best
        checkpoint = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "best_metric": selection if improved else best, "config": dict(config)}
        torch.save(checkpoint, directory / "last.ckpt")
        if improved:
            best = selection; torch.save(checkpoint, directory / "best.ckpt")
            if settings.get("save_predictions", True): save_table(predictions, directory / "best_validation_predictions.csv")
        scheduler.step()
    return pd.DataFrame(history)
