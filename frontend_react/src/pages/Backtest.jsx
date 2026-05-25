import { useEffect, useState } from "react"
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
            <div className="metric"><div className="label">Trades</div><div className="value">{result.trades}</div></div>
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
