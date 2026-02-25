import React, { useEffect, useMemo, useRef, useState } from "react";
import "./App.css";

/**
 * ARgueVis Debug UI (React)
 * - Connects to ws://localhost:8000/ws
 * - Handles:
 *    1) segment records: {seg_id,start,end,speaker,text, ...optional sim}
 *    2) updates/patch:  {type:"update", seg_id, speaker}
 *    3) optional status: {type:"status", enrolled, progress, seconds_left, rms}
 */

const WS_URL_DEFAULT = "ws://localhost:8000/ws";

function fmtSec(s) {
  if (s == null || Number.isNaN(s)) return "—";
  return `${s.toFixed(2)}s`;
}

function clamp01(x) {
  if (x == null || Number.isNaN(x)) return 0;
  return Math.max(0, Math.min(1, x));
}

function durationOf(seg) {
  if (!seg) return 0;
  const a = Number(seg.start);
  const b = Number(seg.end);
  if (Number.isNaN(a) || Number.isNaN(b)) return 0;
  return Math.max(0, b - a);
}

export default function App() {
  const [wsUrl, setWsUrl] = useState(WS_URL_DEFAULT);
  const [connected, setConnected] = useState(false);
  const [lastMsgAt, setLastMsgAt] = useState(null);

  const [status, setStatus] = useState({
    enrolled: null,
    progress: null, // 0..1
    seconds_left: null,
    rms: null,
  });

  const [segmentsById, setSegmentsById] = useState(() => new Map());
  const [order, setOrder] = useState([]); // seg_id in arrival order

  // For small timeline window
  const [windowSec, setWindowSec] = useState(45);

  const wsRef = useRef(null);

  // Derived list
  const segments = useMemo(() => {
    const arr = [];
    for (const id of order) {
      const seg = segmentsById.get(id);
      if (seg) arr.push(seg);
    }
    return arr;
  }, [order, segmentsById]);

  const latestEnd = useMemo(() => {
    let mx = 0;
    for (const seg of segments) {
      const e = Number(seg.end);
      if (!Number.isNaN(e)) mx = Math.max(mx, e);
    }
    return mx;
  }, [segments]);

  const timelineStart = Math.max(0, latestEnd - windowSec);
  const timelineEnd = Math.max(windowSec, latestEnd);

  function connect() {
    // close existing
    try {
      if (wsRef.current) wsRef.current.close();
    } catch {}

    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => setConnected(true);
    ws.onclose = () => setConnected(false);
    ws.onerror = () => setConnected(false);

    ws.onmessage = (evt) => {
      setLastMsgAt(Date.now());
      let msg = null;
      try {
        msg = JSON.parse(evt.data);
      } catch {
        return;
      }

      // 1) optional status event
      if (msg && msg.type === "status") {
        setStatus((prev) => ({
          ...prev,
          enrolled: typeof msg.enrolled === "boolean" ? msg.enrolled : prev.enrolled,
          progress: typeof msg.progress === "number" ? msg.progress : prev.progress,
          seconds_left: typeof msg.seconds_left === "number" ? msg.seconds_left : prev.seconds_left,
          rms: typeof msg.rms === "number" ? msg.rms : prev.rms,
        }));
        return;
      }

      // 2) patch/update event
      if (msg && msg.type === "update" && typeof msg.seg_id === "number") {
        setSegmentsById((prev) => {
          const next = new Map(prev);
          const existing = next.get(msg.seg_id);
          if (existing) {
            next.set(msg.seg_id, { ...existing, speaker: msg.speaker ?? existing.speaker, _patched: true });
          }
          return next;
        });
        return;
      }

      // 3) full segment record
      if (msg && typeof msg.seg_id === "number" && msg.start != null && msg.end != null) {
        setSegmentsById((prev) => {
          const next = new Map(prev);
          const existing = next.get(msg.seg_id);
          const merged = existing ? { ...existing, ...msg } : msg;
          next.set(msg.seg_id, merged);
          return next;
        });
        setOrder((prev) => {
          if (prev.includes(msg.seg_id)) return prev;
          return [...prev, msg.seg_id];
        });
      }
    };
  }

  useEffect(() => {
    connect();
    return () => {
      try {
        wsRef.current?.close();
      } catch {}
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const progress = clamp01(status.progress);
  const enrolledLabel =
    status.enrolled == null ? "Enrollment: (backend not sending status yet)" : status.enrolled ? "Enrolled" : "Enrolling";

  return (
    <div className="page">
      <header className="header">
        <div>
          <div className="title">ARgueVis Speech Debug UI</div>
          <div className="subtitle">VAD segments • Enrollment • Speaker patches • Live transcript</div>
        </div>

        <div className="conn">
          <span className={`dot ${connected ? "ok" : "bad"}`} />
          <span>{connected ? "Connected" : "Disconnected"}</span>
          <span className="muted">{lastMsgAt ? `last msg: ${Math.round((Date.now() - lastMsgAt) / 1000)}s ago` : ""}</span>
        </div>
      </header>

      <div className="grid">
        {/* Left: controls + status */}
        <div className="card">
          <div className="cardTitle">Connection</div>
          <div className="row">
            <input value={wsUrl} onChange={(e) => setWsUrl(e.target.value)} className="input" />
            <button onClick={connect} className="btn">Reconnect</button>
          </div>
          <div className="hint">Backend must be running (your Python) with WebSocket at /ws.</div>

          <div className="sep" />

          <div className="cardTitle">Enrollment</div>
          <div className="pillRow">
            <span className={`pill ${status.enrolled ? "pillOk" : "pillWarn"}`}>{enrolledLabel}</span>
            <span className="pill">RMS: {status.rms == null ? "—" : status.rms.toFixed(4)}</span>
            <span className="pill">Seconds left: {status.seconds_left == null ? "—" : status.seconds_left.toFixed(1)}</span>
          </div>

          <div className="progressWrap">
            <div className="progressBar" style={{ width: `${Math.round(progress * 100)}%` }} />
          </div>
          <div className="hint">
            If this shows “backend not sending status”, Live progress, seconds left, and RMS from backend (updates ~5×/s).
          </div>

          <div className="sep" />

          <div className="cardTitle">Timeline Window</div>
          <div className="row">
            <input
              type="range"
              min="15"
              max="180"
              value={windowSec}
              onChange={(e) => setWindowSec(Number(e.target.value))}
              className="slider"
            />
            <div className="mono">{windowSec}s</div>
          </div>
          <div className="hint">Shows last N seconds of segments as bars (simple segmentation view).</div>
        </div>

        {/* Right: timeline + segment table */}
        <div className="card">
          <div className="cardTitle">Segmentation Timeline (last {windowSec}s)</div>
          <div className="timeline">
            <div className="timelineAxis">
              <span className="mono">{fmtSec(timelineStart)}</span>
              <span className="mono">{fmtSec(timelineEnd)}</span>
            </div>

            <div className="timelineBars">
              {segments
                .filter((s) => Number(s.end) >= timelineStart)
                .map((s) => {
                  const st = Number(s.start);
                  const en = Number(s.end);
                  const left = ((st - timelineStart) / (timelineEnd - timelineStart)) * 100;
                  const width = ((en - st) / (timelineEnd - timelineStart)) * 100;
                  const speaker = s.speaker ?? "?";
                  return (
                    <div
                      key={s.seg_id}
                      className={`bar bar${speaker}`}
                      style={{ left: `${left}%`, width: `${Math.max(0.5, width)}%` }}
                      title={`seg ${s.seg_id} • ${speaker} • ${fmtSec(st)}-${fmtSec(en)} • ${s.text ?? ""}`}
                    />
                  );
                })}
            </div>
          </div>

          <div className="sep" />

          <div className="cardTitle">Segments</div>
          <div className="table">
            <div className="thead">
              <div>ID</div>
              <div>Start</div>
              <div>End</div>
              <div>Dur</div>
              <div>Spk</div>
              <div>Text</div>
            </div>

            {segments.slice().reverse().map((s) => (
              <div className="trow" key={s.seg_id}>
                <div className="mono">{s.seg_id}</div>
                <div className="mono">{fmtSec(Number(s.start))}</div>
                <div className="mono">{fmtSec(Number(s.end))}</div>
                <div className="mono">{fmtSec(durationOf(s))}</div>
                <div className={`speaker ${s.speaker === "A" ? "A" : s.speaker === "B" ? "B" : "U"}`}>
                  {s.speaker ?? "?"}{s._patched ? " (patched)" : ""}
                  {typeof s.sim === "number" ? <span className="muted"> • sim {s.sim.toFixed(3)}</span> : null}
                </div>
                <div className="textCell">{s.text ?? ""}</div>
              </div>
            ))}

            {segments.length === 0 && <div className="empty">No segments yet. Speak into the mic.</div>}
          </div>
        </div>
      </div>
    </div>
  );
}