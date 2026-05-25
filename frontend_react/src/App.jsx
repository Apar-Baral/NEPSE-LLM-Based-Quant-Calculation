import { BrowserRouter, NavLink, Route, Routes } from 'react-router-dom'
import Scanner from './pages/Scanner'
import SymbolDetail from './pages/SymbolDetail'
import Upload from './pages/Upload'
import Backtest from './pages/Backtest'
import Brief from './pages/Brief'
import Chat from './pages/Chat'

export default function App() {
  return (
    <BrowserRouter>
      <div className="layout">
        <aside className="sidebar">
          <h1>NEPSE Quant</h1>
          <nav>
            <NavLink to="/" end>Scanner</NavLink>
            <NavLink to="/symbol">Symbol</NavLink>
            <NavLink to="/upload">Upload</NavLink>
            <NavLink to="/backtest">Backtest</NavLink>
            <NavLink to="/brief">LLM Brief</NavLink>
            <NavLink to="/chat">Chat</NavLink>
          </nav>
        </aside>
        <main className="main">
          <Routes>
            <Route path="/" element={<Scanner />} />
            <Route path="/symbol" element={<SymbolDetail />} />
            <Route path="/upload" element={<Upload />} />
            <Route path="/backtest" element={<Backtest />} />
            <Route path="/brief" element={<Brief />} />
            <Route path="/chat" element={<Chat />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  )
}
