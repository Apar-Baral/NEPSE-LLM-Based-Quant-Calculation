from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

from backend.config import DB_PATH, PROCESSED_DIR, ensure_dirs


class DataStore:
    def __init__(self, db_path: Path | None = None):
        ensure_dirs()
        self.db_path = db_path or DB_PATH
        self.engine = create_engine(f"sqlite:///{self.db_path}")

    def save_panel(self, df: pd.DataFrame, table: str = "symbol_panel") -> int:
        if df.empty:
            return 0
        df = df.copy()
        if "report_date" in df.columns:
            df["report_date"] = pd.to_datetime(df["report_date"]).dt.normalize()
        path = PROCESSED_DIR / f"{table}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        try:
            df.to_sql(table, self.engine, if_exists="replace", index=False, method="multi", chunksize=500)
        except Exception:
            pass  # Parquet is source of truth
        return len(df)

    def append_panel(self, df: pd.DataFrame, table: str = "symbol_panel") -> int:
        if df.empty:
            return 0
        existing = self.load_panel(table)
        df = df.copy()
        df["report_date"] = pd.to_datetime(df["report_date"]).dt.normalize()
        if not existing.empty:
            keys = ["report_date", "symbol", "horizon", "side"]
            combined = pd.concat([existing, df], ignore_index=True)
            combined = combined.drop_duplicates(subset=keys, keep="last")
        else:
            combined = df
        return self.save_panel(combined, table)

    def load_panel(self, table: str = "symbol_panel") -> pd.DataFrame:
        parquet = PROCESSED_DIR / f"{table}.parquet"
        if parquet.exists():
            return pd.read_parquet(parquet)
        try:
            return pd.read_sql_table(table, self.engine)
        except Exception:
            return pd.DataFrame()

    def save_features(self, df: pd.DataFrame) -> int:
        return self.save_panel(df, "features")

    def load_features(self) -> pd.DataFrame:
        return self.load_panel("features")

    def save_predictions(self, df: pd.DataFrame) -> int:
        return self.save_panel(df, "predictions")

    def load_predictions(self) -> pd.DataFrame:
        return self.load_panel("predictions")

    def save_ohlcv(self, df: pd.DataFrame) -> int:
        if df.empty:
            return 0
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df.to_sql("ohlcv", self.engine, if_exists="replace", index=False)
        df.to_parquet(PROCESSED_DIR / "ohlcv.parquet", index=False)
        return len(df)

    def load_ohlcv(self) -> pd.DataFrame:
        p = PROCESSED_DIR / "ohlcv.parquet"
        if p.exists():
            return pd.read_parquet(p)
        try:
            return pd.read_sql_table("ohlcv", self.engine)
        except Exception:
            return pd.DataFrame()

    def latest_report_date(self) -> date | None:
        panel = self.load_panel()
        if panel.empty:
            return None
        return pd.to_datetime(panel["report_date"]).max().date()

    def execute(self, sql: str, params: dict | None = None):
        with self.engine.begin() as conn:
            conn.execute(text(sql), params or {})
