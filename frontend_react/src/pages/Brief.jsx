import { useState } from "react"
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
