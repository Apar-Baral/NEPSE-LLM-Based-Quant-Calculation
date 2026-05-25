from __future__ import annotations

import json

import numpy as np
import pandas as pd

from backend.config import MODELS_DIR
from backend.models.multimodal.architecture import torch_available
from backend.models.multimodal.data_builder import (
    broker_sets,
    build_graph_adjacency,
    build_semantic_vector,
    build_temporal_tensor,
)

CHECKPOINT = MODELS_DIR / "multimodal.pt"
META_PATH = MODELS_DIR / "multimodal_meta.json"
INTERP_PATH = MODELS_DIR / "multimodal_interpret.json"


def predict_multimodal(
    features: pd.DataFrame,
    broker_panel: pd.DataFrame | None = None,
) -> pd.DataFrame:
    out = features.copy()
    out["mm_p_long"] = np.nan
    out["mm_confidence"] = np.nan

    if not torch_available() or not CHECKPOINT.exists():
        return out

    import torch
    from backend.models.multimodal.architecture import NepseMultimodalNet

    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model = NepseMultimodalNet(sem_dim=ckpt.get("sem_dim", 16))
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    bp = broker_panel if broker_panel is not None and not broker_panel.empty else pd.DataFrame()
    broker_map = broker_sets(bp)
    symbols = out["symbol"].astype(str).tolist()
    adj = build_graph_adjacency(symbols, broker_map)
    sym_to_idx = {s: i for i, s in enumerate(dict.fromkeys(symbols))}

    temporal = np.stack([build_temporal_tensor(row) for _, row in out.iterrows()])
    semantic = np.stack([build_semantic_vector(row) for _, row in out.iterrows()])
    node_idx = np.array([sym_to_idx[s] for s in symbols], dtype=np.int64)

    with torch.no_grad():
        T = torch.tensor(temporal, dtype=torch.float32)
        S = torch.tensor(semantic, dtype=torch.float32)
        A = torch.tensor(adj, dtype=torch.float32)
        N = torch.tensor(node_idx, dtype=torch.long)
        logits, interpret = model(T, S, A, N)
        probs = torch.sigmoid(logits).numpy()

    out["mm_p_long"] = probs
    out["mm_confidence"] = np.abs(probs - 0.5) * 2

    _save_interpretation(interpret, out)
    return out


def _save_interpretation(interpret: dict, df: pd.DataFrame) -> None:
    try:
        horizons = ["1D", "2D", "3D", "4D", "1W"]
        attn = interpret.get("temporal_attn")
        if attn is not None:
            attn_np = attn.cpu().numpy()
            rows = []
            for i, sym in enumerate(df["symbol"].tolist()):
                if i >= len(attn_np):
                    break
                for j, h in enumerate(horizons):
                    rows.append({"symbol": sym, "horizon": h, "temporal_weight": float(attn_np[i, j])})
            payload = {
                "decay_lambda": interpret.get("decay_lambda"),
                "temporal_attention": rows[:50],
            }
            INTERP_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception:
        pass


def load_interpretation() -> dict:
    if not INTERP_PATH.exists():
        return {}
    try:
        return json.loads(INTERP_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
