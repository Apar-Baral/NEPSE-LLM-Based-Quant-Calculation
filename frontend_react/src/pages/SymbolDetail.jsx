import { useEffect, useState } from "react"
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
          <div className="metric"><div className="label">Tier</div><div className="value">{latest.signal_tier}</div></div>
          <div className="metric"><div className="label">P(Long)</div><div className="value">{((latest.p_long_momentum || 0) * 100).toFixed(0)}%</div></div>
          <div className="metric"><div className="label">EMS</div><div className="value">{Math.round(latest.early_momentum_score || 0)}</div></div>
          <div className="metric"><div className="label">LTP</div><div className="value">{latest.ltp ?? "-"}</div></div>
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