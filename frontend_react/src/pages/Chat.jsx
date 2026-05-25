import { useState } from "react"
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
