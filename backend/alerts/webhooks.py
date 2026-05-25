from __future__ import annotations

import httpx

from backend.config import get_settings


def send_alert(message: str, symbols: list[str] | None = None) -> bool:
    settings = get_settings()
    url = settings.alert_webhook_url
    if not url:
        return False
    payload = {"text": message, "symbols": symbols or []}
    try:
        with httpx.Client(timeout=15) as client:
            resp = client.post(url, json=payload)
            return resp.status_code < 400
    except Exception:
        return False


def check_trigger_alerts(scanner_df) -> list[str]:
    if scanner_df.empty:
        return []
    triggered = scanner_df[scanner_df["signal_tier"].isin(["Trigger", "Confirmed"])]
    alerts = []
    for _, row in triggered.iterrows():
        msg = (
            f"NEPSE Alert: {row['symbol']} — {row['signal_tier']} "
            f"P={row.get('p_long_momentum', 0):.0%} EMS={row.get('early_momentum_score', 0):.0f}"
        )
        if send_alert(msg, [row["symbol"]]):
            alerts.append(row["symbol"])
    return alerts
