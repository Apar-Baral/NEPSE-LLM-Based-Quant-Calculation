from __future__ import annotations

import pandas as pd


def coerce_numeric(df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    """Force numeric columns (fixes object dtype from None / LLM merge)."""
    out = df.copy()
    targets = cols or [
        "p_long_momentum",
        "llm_p_long",
        "expected_return_10d",
        "early_momentum_score",
        "early_rank_score",
        "broker_pressure",
        "distribution_risk_score",
        "daily_volume",
        "daily_turnover_lac",
        "ltp",
        "float_turnover_1d_abs",
        "volume_rank",
        "turnover_rank",
    ]
    for col in targets:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out
