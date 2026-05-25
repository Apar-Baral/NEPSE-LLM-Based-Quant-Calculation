import { useEffect, useState } from "react"
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
            <div className="metric"><div className="label">Symbols</div>{meta.count || 0}<div className="value">{meta.count || 0}</div></div>
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
