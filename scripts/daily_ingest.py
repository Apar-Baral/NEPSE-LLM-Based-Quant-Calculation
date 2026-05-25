#!/usr/bin/env python3
"""Daily ingest from inbox folder."""
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.pipeline import run_pipeline

INBOX = ROOT / "data" / "inbox"

if __name__ == "__main__":
    acc = next(INBOX.glob("*ccum*.xlsx"), None) or next(INBOX.glob("*ccum*.xls"), None)
    dist = next(INBOX.glob("*ist*.xlsx"), None) or next(INBOX.glob("*ist*.xls"), None)
    ohlcv = next(INBOX.glob("*.csv"), None)
    result = run_pipeline(
        report_date=date.today(),
        acc_path=acc,
        dist_path=dist,
        ohlcv_path=ohlcv if ohlcv and "ohlcv" in ohlcv.name.lower() else None,
    )
    print(result)
