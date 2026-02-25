import { useEffect, useRef, useState } from 'react'

const WS_URL = 'ws://localhost:8000/ws'

function App() {
  const [segments, setSegments] = useState([])
  const [status, setStatus] = useState('connecting') // 'connecting' | 'open' | 'closed' | 'error'
  const listRef = useRef(null)
  const wsRef = useRef(null)

  useEffect(() => {
    const ws = new WebSocket(WS_URL)
    wsRef.current = ws

    ws.onopen = () => setStatus('open')
    ws.onclose = () => setStatus('closed')
    ws.onerror = () => setStatus('error')

    ws.onmessage = (event) => {
      try {
        const seg = JSON.parse(event.data)
        if (seg && typeof seg.seg_id === 'number' && seg.text != null) {
          setSegments((prev) => [...prev, seg])
        }
      } catch (_) {}
    }

    return () => {
      ws.close()
      wsRef.current = null
    }
  }, [])

  // Auto-scroll to bottom when new segments arrive
  useEffect(() => {
    const el = listRef.current
    if (el) el.scrollTop = el.scrollHeight
  }, [segments])

  return (
    <div style={styles.container}>
      <header style={styles.header}>
        <h1 style={styles.title}>ArgueViz transcript</h1>
        <span style={{ ...styles.badge, ...styles[status] }}>{status}</span>
      </header>
      <div ref={listRef} style={styles.scrollPanel}>
        {segments.length === 0 && status === 'open' && (
          <p style={styles.placeholder}>Waiting for speech…</p>
        )}
        {segments.length === 0 && status === 'connecting' && (
          <p style={styles.placeholder}>Connecting to stream…</p>
        )}
        {status === 'closed' && (
          <p style={styles.placeholder}>Disconnected. Start the pipeline to stream again.</p>
        )}
        {status === 'error' && (
          <p style={styles.placeholder}>Connection error. Is the backend running on port 8000?</p>
        )}
        {segments.map((seg) => (
          <div key={seg.seg_id} style={styles.segment}>
            <span style={styles.speaker}>[{seg.speaker}]</span>
            <span style={styles.text}>{seg.text}</span>
            <span style={styles.time}>
              {seg.start != null && seg.end != null
                ? `${Number(seg.start).toFixed(1)}s – ${Number(seg.end).toFixed(1)}s`
                : ''}
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}

const styles = {
  container: {
    display: 'flex',
    flexDirection: 'column',
    height: '100vh',
  },
  header: {
    flexShrink: 0,
    display: 'flex',
    alignItems: 'center',
    gap: '12px',
    padding: '12px 20px',
    borderBottom: '1px solid #27272a',
    background: '#18181b',
  },
  title: {
    margin: 0,
    fontSize: '1.25rem',
    fontWeight: 600,
  },
  badge: {
    fontSize: '0.75rem',
    padding: '4px 8px',
    borderRadius: '6px',
    textTransform: 'lowercase',
  },
  connecting: { background: '#fef08a', color: '#713f12' },
  open: { background: '#bbf7d0', color: '#14532d' },
  closed: { background: '#e4e4e7', color: '#3f3f46' },
  error: { background: '#fecaca', color: '#991b1b' },
  scrollPanel: {
    flex: 1,
    overflow: 'auto',
    padding: '16px 20px',
    display: 'flex',
    flexDirection: 'column',
    gap: '10px',
  },
  placeholder: {
    color: '#71717a',
    margin: 0,
  },
  segment: {
    display: 'flex',
    alignItems: 'baseline',
    gap: '8px',
    flexWrap: 'wrap',
    padding: '10px 12px',
    background: '#27272a',
    borderRadius: '8px',
    borderLeft: '3px solid #3b82f6',
  },
  speaker: {
    flexShrink: 0,
    fontWeight: 600,
    color: '#a78bfa',
  },
  text: {
    flex: '1 1 auto',
  },
  time: {
    flexShrink: 0,
    fontSize: '0.8rem',
    color: '#71717a',
  },
}

export default App
