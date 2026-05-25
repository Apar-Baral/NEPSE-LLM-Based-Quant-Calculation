import { useState } from "react"
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
