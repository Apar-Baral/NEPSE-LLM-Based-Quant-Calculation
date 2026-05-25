from __future__ import annotations

import pandas as pd

from backend.models.multimodal.predict import load_interpretation


def attention_dataframe() -> pd.DataFrame:
    data = load_interpretation()
    rows = data.get("temporal_attention", [])
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)
