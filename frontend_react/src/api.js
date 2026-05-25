const API = '/api'

export async function fetchScan() {
  const r = await fetch(`${API}/scan`)
  if (!r.ok) throw new Error('Failed to fetch scanner')
  return r.json()
}

export async function fetchSymbol(symbol) {
  const r = await fetch(`${API}/symbol/${symbol}`)
  if (!r.ok) throw new Error('Failed to fetch symbol')
  return r.json()
}

export async function runPipeline() {
  const r = await fetch(`${API}/pipeline/run`, { method: 'POST' })
  if (!r.ok) throw new Error('Pipeline failed')
  return r.json()
}

export async function fetchBrief() {
  const r = await fetch(`${API}/llm/brief`)
  if (!r.ok) throw new Error('Brief failed')
  return r.json()
}

export async function fetchBacktest(tier = 'Trigger', hold = 10) {
  const r = await fetch(`${API}/backtest?entry_tier=${tier}&hold_days=${hold}`)
  if (!r.ok) throw new Error('Backtest failed')
  return r.json()
}

export async function chat(question) {
  const r = await fetch(`${API}/llm/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question }),
  })
  if (!r.ok) throw new Error('Chat failed')
  return r.json()
}

export async function uploadFiles(formData) {
  const r = await fetch(`${API}/upload`, { method: 'POST', body: formData })
  if (!r.ok) throw new Error('Upload failed')
  return r.json()
}
