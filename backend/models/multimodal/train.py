from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from backend.config import MODELS_DIR, load_yaml_config
from backend.models.multimodal.architecture import (
    NepseMultimodalNet,
    phase_aware_bce_loss,
    torch_available,
)
from backend.models.multimodal.data_builder import build_training_batch

CHECKPOINT = MODELS_DIR / "multimodal.pt"
META_PATH = MODELS_DIR / "multimodal_meta.json"


def train_multimodal(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    broker_panel: pd.DataFrame | None = None,
) -> dict:
    if not torch_available():
        return {"status": "skipped", "reason": "PyTorch not installed. Run: pip install torch"}

    import torch
    from torch.utils.data import DataLoader, TensorDataset

    cfg = load_yaml_config("settings.yaml").get("multimodal", {})
    epochs = int(cfg.get("epochs", 40))
    lr = float(cfg.get("learning_rate", 0.001))
    batch_size = int(cfg.get("batch_size", 64))
    max_samples = cfg.get("max_training_samples")

    batch = build_training_batch(features, labels, broker_panel, max_samples=max_samples)
    if not batch or len(batch["y"]) < 30:
        return {"status": "skipped", "reason": f"Need >=30 labeled rows, got {len(batch.get('y', []))}"}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NepseMultimodalNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    T = torch.tensor(batch["temporal"], dtype=torch.float32)
    S = torch.tensor(batch["semantic"], dtype=torch.float32)
    Y = torch.tensor(batch["y"], dtype=torch.float32)
    W = torch.tensor(batch["weights"], dtype=torch.float32)
    ADJ = torch.tensor(batch["adjacency"], dtype=torch.float32)
    NIDX = torch.tensor(batch["node_idx"], dtype=torch.long)

    n = len(Y)
    split = max(int(n * 0.8), 1)
    ds_train = TensorDataset(T[:split], S[:split], Y[:split], W[:split], NIDX[:split])
    ds_val = TensorDataset(T[split:], S[split:], Y[split:], W[split:], NIDX[split:])
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=batch_size)

    best_val = float("inf")
    history = []

    for ep in range(epochs):
        model.train()
        train_loss = 0.0
        for tb, sb, yb, wb, nib in dl_train:
            tb, sb, yb, wb, nib = tb.to(device), sb.to(device), yb.to(device), wb.to(device), nib.to(device)
            adj_b = ADJ.to(device)
            opt.zero_grad()
            logits, _ = model(tb, sb, adj_b, nib)
            loss = phase_aware_bce_loss(logits, yb, wb)
            loss.backward()
            opt.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for tb, sb, yb, wb, nib in dl_val:
                tb, sb, yb, wb, nib = tb.to(device), sb.to(device), yb.to(device), wb.to(device), nib.to(device)
                logits, _ = model(tb, sb, ADJ.to(device), nib)
                val_loss += phase_aware_bce_loss(logits, yb, wb).item()
                pred = (torch.sigmoid(logits) > 0.5).float()
                correct += (pred == yb).sum().item()
                total += len(yb)

        train_loss /= max(len(dl_train), 1)
        val_loss /= max(len(dl_val), 1)
        acc = correct / max(total, 1)
        history.append({"epoch": ep + 1, "train_loss": train_loss, "val_loss": val_loss, "val_acc": acc})

        if val_loss < best_val:
            best_val = val_loss
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "adj_template_shape": list(ADJ.shape),
                    "sem_dim": S.shape[1],
                    "temporal_shape": list(T.shape[1:]),
                },
                CHECKPOINT,
            )

    meta = {
        "status": "ok",
        "epochs": epochs,
        "samples": int(n),
        "best_val_loss": best_val,
        "device": str(device),
        "history_last": history[-3:] if history else [],
    }
    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta
