"""Generate React page components."""
import re
from pathlib import Path

PAGES = Path(__file__).resolve().parent.parent / "frontend_react" / "src" / "pages"
PAGES.mkdir(parents=True, exist_ok=True)


def clean_js(text: str) -> str:
    text = re.sub(r"<motion[^>]*>", "", text)
    return text.replace("</motion>", "")


FILES = {
    "Scanner.jsx": '''import { useEffect, useState } from "react"
import { fetchScan, runPipeline } from "../api"

function pct(v) {
  if (v == null) return "-"
  return (Number(v) * 100).toFixed(0) + "%"
}

export default function Scanner() {
  const [data, setData] = useState([])
  const [meta, setMeta] = useState({})
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState("")

  const load = async () => {
    setLoading(true)
    setError("")
    try {
      const res = await fetchScan()
      setData(res.data || [])
      setMeta({ date: res.report_date, count: res.count })
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { load() }, [])

  const triggers = data.filter((r) => ["Trigger", "Confirmed"].includes(r.signal_tier))
  const avgP = data.length ? data.reduce((s, r) => s + (r.p_long_momentum || 0), 0) / data.length : 0

  return (
    <motion>
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <h2>Momentum Scanner</h2>
        <button onClick={async () => { await runPipeline(); await load() }}>Run Pipeline</button>
      </div>
      {error && <p style={{ color: "#f87171" }}>{error}</p>}
      {loading ? <p>Loading...</p> : (
        <>
          <div className="metrics">
            <div className="metric"><div className="label">Report Date</div><div className="value">{meta.date || "-"}</div></div>
            <div className="metric"><div className="label">Symbols</div><motion className="value">{meta.count || 0}</motion><div className="value">{meta.count || 0}</div></div>
            <div className="metric"><div className="label">Trigger+</div><div className="value">{triggers.length}</div></div>
            <div className="metric"><div className="label">Avg P(Momentum)</div><div className="value">{pct(avgP)}</div></div>
          </div>
          <div className="card">
            <table>
              <thead>
                <svg>
                <tr>
                  <th>Symbol</th><th>LTP</th><th>Tier</th><th>P(Long)</th><th>EMS</th><th>SMS</th><th>Exp Ret</th>
                </tr>
              </thead>
              <tbody>
                {data.map((r) => (
                  <tr key={r.symbol}>
                    <td>{r.symbol}</td>
                    <td>{r.ltp ?? "-"}</td>
                    <td className={"tier-" + r.signal_tier}>{r.signal_tier}</td>
                    <td>{pct(r.p_long_momentum)}</td>
                    <td>{Math.round(r.early_momentum_score || 0)}</td>
                    <td>{Math.round(r.smart_money_score || 0)}</td>
                    <td>{(r.expected_return_10d || 0).toFixed(1)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  )
}
''',
}

# Write only Scanner for now with manual clean content
(PAGES / "Scanner.jsx").write_text(
    clean_js(
        FILES["Scanner.jsx"]
        .replace("<motion>\n    ", "")
        .replace("<motion className=\"value\">", "")
        .replace("<thead>\n                svg>", "<thead>")
    ),
    encoding="utf-8",
)

(PAGES / "SymbolDetail.jsx").write_text(
    '''import { useEffect, useState } from "react"
import { fetchScan, fetchSymbol } from "../api"

export default function SymbolDetail() {
  const [symbols, setSymbols] = useState([])
  const [selected, setSelected] = useState("")
  const [detail, setDetail] = useState(null)

  useEffect(() => {
    fetchScan().then((r) => {
      const syms = (r.data || []).map((x) => x.symbol)
      setSymbols(syms)
      if (syms.length) setSelected(syms[0])
    })
  }, [])

  useEffect(() => {
    if (!selected) return
    fetchSymbol(selected).then(setDetail)
  }, [selected])

  const latest = detail?.latest?.[0] || {}

  return (
    <div>
      <h2>Symbol Deep Dive</h2>
      <select value={selected} onChange={(e) => setSelected(e.target.value)} style={{ marginBottom: "1rem", padding: "0.5rem" }}>
        {symbols.map((s) => <option key={s} value={s}>{s}</option>)}
      </select>
      {latest.symbol && (
        <div className="metrics">
          <div className="metric"><motion className="label">Tier</motion><div className="label">Tier</div><div className="value">{latest.signal_tier}</div></div>
          <div className="metric"><div className="label">P(Long)</div><div className="value">{((latest.p_long_momentum || 0) * 100).toFixed(0)}%</div></div>
          <div className="metric"><div className="label">EMS</div><div className="value">{Math.round(latest.early_momentum_score || 0)}</div></div>
          <motion className="metric"><div className="label">LTP</div><div className="value">{latest.ltp ?? "-"}</div></div>
        </div>
      )}
      {detail?.shap && Object.keys(detail.shap).length > 0 && (
        <div className="card">
          <h3>Top SHAP Drivers</h3>
          <ul>
            {Object.entries(detail.shap)
              .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
              .slice(0, 10)
              .map(([k, v]) => <li key={k}>{k}: {v.toFixed(4)}</li>)}
          </ul>
        </div>
      )}
    </div>
  )
}
''',
    encoding="utf-8",
)

for f in PAGES.glob("*.jsx"):
    f.write_text(clean_js(f.read_text(encoding="utf-8")), encoding="utf-8")

(PAGES / "Upload.jsx").write_text(
    '''import { useState } from "react"
import { uploadFiles } from "../api"

export default function Upload() {
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)

  const onSubmit = async (e) => {
    e.preventDefault()
    const fd = new FormData(e.target)
    setLoading(true)
    try {
      setResult(await uploadFiles(fd))
    } catch (err) {
      setResult({ status: "error", message: err.message })
    } finally {
      setLoading(false)
    }
  }

  return (
    <div>
      <h2>Daily Upload</h2>
      <form onSubmit={onSubmit} className="card">
        <label>Accumulation Excel</label>
        <input type="file" name="accumulation" accept=".xlsx,.xls" />
        <label>Distribution Excel</label>
        <input type="file" name="distribution" accept=".xlsx,.xls" />
        <label>OHLCV CSV (optional)</label>
        <input type="file" name="ohlcv" accept=".csv" />
        <button type="submit" disabled={loading}>{loading ? "Processing..." : "Ingest & Analyze"}</button>
      </form>
      {result && <pre className="card">{JSON.stringify(result, null, 2)}</pre>}
    </div>
  )
}
''',
    encoding="utf-8",
)

(PAGES / "Backtest.jsx").write_text(
    '''import { useEffect, useState } from "react"
import { fetchBacktest } from "../api"

export default function Backtest() {
  const [result, setResult] = useState(null)
  const [tier, setTier] = useState("Trigger")
  const [hold, setHold] = useState(10)

  const run = () => fetchBacktest(tier, hold).then(setResult)

  useEffect(() => { run() }, [])

  return (
    <div>
      <h2>Backtest</h2>
      <div style={{ display: "flex", gap: "1rem", marginBottom: "1rem" }}>
        <select value={tier} onChange={(e) => setTier(e.target.value)}>
          <option>Trigger</option><option>Confirmed</option><option>Setup</option>
        </select>
        <input type="number" value={hold} onChange={(e) => setHold(Number(e.target.value))} min={3} max={30} />
        <button onClick={run}>Run</button>
      </div>
      {result && (
        <>
          <div className="metrics">
            <div className="metric"><div className="label">Trades</div><div className="value">{result.trades}</div></motion></div>
            <div className="metric"><div className="label">Win Rate</div><div className="value">{((result.win_rate || 0) * 100).toFixed(0)}%</div></div>
            <div className="metric"><div className="label">Avg Return</div><div className="value">{(result.avg_return || 0).toFixed(2)}%</div></div>
            <div className="metric"><div className="label">CAGR Proxy</div><div className="value">{((result.cagr_proxy || 0) * 100).toFixed(1)}%</div></div>
          </div>
          {result.details?.length > 0 && (
            <div className="card">
              <table>
                <thead><tr><th>Symbol</th><th>Entry</th><th>Return</th><th>Tier</th></tr></thead>
                <tbody>
                  {result.details.map((t, i) => (
                    <tr key={i}><td>{t.symbol}</td><td>{t.entry_date}</td><td>{t.return_pct?.toFixed(2)}%</td><td>{t.tier}</td></tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </>
      )}
    </div>
  )
}
''',
    encoding="utf-8",
)

(PAGES / "Brief.jsx").write_text(
    '''import { useState } from "react"
import { fetchBrief } from "../api"

export default function Brief() {
  const [text, setText] = useState("")
  const [loading, setLoading] = useState(false)

  return (
    <div>
      <h2>LLM Daily Brief</h2>
      <button onClick={async () => {
        setLoading(true)
        try { const r = await fetchBrief(); setText(r.brief) } finally { setLoading(false) }
      }} disabled={loading}>{loading ? "Generating..." : "Generate Brief"}</button>
      <div className="card report">{text}</div>
    </div>
  )
}
''',
    encoding="utf-8",
)

(PAGES / "Chat.jsx").write_text(
    '''import { useState } from "react"
import { chat } from "../api"

export default function Chat() {
  const [q, setQ] = useState("")
  const [a, setA] = useState("")

  return (
    <div>
      <h2>Quant Chat</h2>
      <input type="text" value={q} onChange={(e) => setQ(e.target.value)} placeholder="Which stocks show early accumulation?" />
      <button onClick={async () => { const r = await chat(q); setA(r.answer) }}>Send</button>
      <div className="card report">{a}</div>
    </div>
  )
}
''',
    encoding="utf-8",
)

for f in PAGES.glob("*.jsx"):
    f.write_text(clean_js(f.read_text(encoding="utf-8")), encoding="utf-8")

print("React pages generated")
