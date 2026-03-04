import React, { useEffect, useMemo, useRef, useState } from "react";
import "./App.css";

/**
 * ARgueVis Debug UI (React) — v14-compatible
 *
 * Backend (audio_v14.py) sends WebSocket events with:
 *   - type: "status" | "segment" | "turn" | "link"
 *   - event_id: string (primary key; last-write-wins)
 *   - patches: same event_id, plus "_patch": true
 *   - link tombstones: {type:"link", event_id:..., deleted:true}
 *
 * This UI:
 *   - Stores all events by event_id (last-write-wins)
 *   - Renders segments, turns (with raw vs cleaned, confidence), and links
 *   - Handles link tombstones by filtering deleted=true
 */

const WS_URL_DEFAULT = "ws://localhost:8000/ws";

function fmtSec(s) {
  if (s == null || Number.isNaN(s)) return "—";
  return `${Number(s).toFixed(2)}s`;
}

function clamp01(x) {
  if (x == null || Number.isNaN(x)) return 0;
  return Math.max(0, Math.min(1, x));
}

function safeNum(x, fallback = 0) {
  const n = Number(x);
  return Number.isNaN(n) ? fallback : n;
}

function durationOf(obj) {
  if (!obj) return 0;
  const a = safeNum(obj.start, NaN);
  const b = safeNum(obj.end, NaN);
  if (Number.isNaN(a) || Number.isNaN(b)) return 0;
  return Math.max(0, b - a);
}

function shortId(id, n = 10) {
  if (!id) return "—";
  const s = String(id);
  if (s.length <= n) return s;
  return `${s.slice(0, n)}…`;
}

export default function App() {
  const [wsUrl, setWsUrl] = useState(WS_URL_DEFAULT);
  const [connected, setConnected] = useState(false);
  const [lastMsgAt, setLastMsgAt] = useState(null);

  const [status, setStatus] = useState({
    enrolled: null,
    progress: null, // 0..1
    seconds_left: null,
    enroll_sec: null,
    enroll_sec_target: null,
    rms: null,
    mode: null, // "listening" | "recording" | "processing"
  });

  // Store: last-write-wins by event_id
  const [eventsById, setEventsById] = useState(() => new Map());
  const [eventOrder, setEventOrder] = useState([]); // event_id in first-seen order

  // Simple timeline window for segments
  const [windowSec, setWindowSec] = useState(45);

  const wsRef = useRef(null);

  function upsertEvent(msg) {
    if (!msg || typeof msg.event_id !== "string") return;

    setEventsById((prev) => {
      const next = new Map(prev);
      const existing = next.get(msg.event_id);
      // last-write-wins merge
      const merged = existing ? { ...existing, ...msg } : msg;
      next.set(msg.event_id, merged);
      return next;
    });

    setEventOrder((prev) => (prev.includes(msg.event_id) ? prev : [...prev, msg.event_id]));
  }

  function connect() {
    try {
      wsRef.current?.close();
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

      // Status is special (no event_id changes needed)
      if (msg?.type === "status") {
        setStatus((prev) => ({
          ...prev,
          enrolled: typeof msg.enrolled === "boolean" ? msg.enrolled : prev.enrolled,
          progress: typeof msg.progress === "number" ? msg.progress : prev.progress,
          seconds_left: typeof msg.seconds_left === "number" ? msg.seconds_left : prev.seconds_left,
          enroll_sec: typeof msg.enroll_sec === "number" ? msg.enroll_sec : prev.enroll_sec,
          enroll_sec_target:
            typeof msg.enroll_sec_target === "number" ? msg.enroll_sec_target : prev.enroll_sec_target,
          rms: typeof msg.rms === "number" ? msg.rms : prev.rms,
          mode: typeof msg.mode === "string" ? msg.mode : prev.mode,
        }));
        return;
      }

      // Everything else: segment/turn/link (including patches & tombstones)
      if (msg?.event_id && typeof msg.event_id === "string") {
        upsertEvent(msg);
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

  // ---- Derived lists ----
  const allEvents = useMemo(() => {
    const arr = [];
    for (const id of eventOrder) {
      const e = eventsById.get(id);
      if (e) arr.push(e);
    }
    return arr;
  }, [eventOrder, eventsById]);

  const segments = useMemo(
    () => allEvents.filter((e) => e.type === "segment" && typeof e.seg_id === "string"),
    [allEvents]
  );

  const turns = useMemo(
    () => allEvents.filter((e) => e.type === "turn" && typeof e.turn_id === "number"),
    [allEvents]
  );

  const links = useMemo(
    () =>
      allEvents.filter((e) => e.type === "link" && typeof e.link_id === "string" && !e.deleted),
    [allEvents]
  );

  // Segment timeline range
  const latestEnd = useMemo(() => {
    let mx = 0;
    for (const seg of segments) {
      mx = Math.max(mx, safeNum(seg.end, 0));
    }
    return mx;
  }, [segments]);

  const timelineStart = Math.max(0, latestEnd - windowSec);
  const timelineEnd = Math.max(windowSec, latestEnd);

  // ---- UI labels ----
  const progress = clamp01(status.progress);
  const enrolledLabel =
    status.enrolled == null
      ? "Enrollment: (backend not sending status yet)"
      : status.enrolled
      ? "Enrolled"
      : "Enrolling";

  const modeLabel =
    status.mode === "processing"
      ? "Processing"
      : status.mode === "recording"
      ? "Recording"
      : status.mode === "listening"
      ? "Listening"
      : "—";

  const lastMsgLabel = useMemo(() => {
    if (!lastMsgAt) return "—";
    const sec = (Date.now() - lastMsgAt) / 1000;
    return `${sec.toFixed(1)}s ago`;
  }, [lastMsgAt]);

  return (
    <div className="page">
      <header className="header">
        <div>
          <div className="title">ARgueVis Speech Debug UI</div>
          <div className="subtitle">
            Status • VAD segments • Turns (LLM cleaned_text + confidence) • Links (tombstones supported)
          </div>
        </div>
      </header>

      {/* Live status strip */}
      <div className="liveStatusStrip">
        <div className="liveStatusItem">
          <span className="liveStatusLabel">Connection</span>
          <span className={`dot ${connected ? "ok" : "bad"}`} />
          <span>{connected ? "Connected" : "Disconnected"}</span>
        </div>

        <div className="liveStatusItem">
          <span className="liveStatusLabel">Enrollment</span>
          <div className="liveStatusProgressWrap">
            <div className="liveStatusProgressBar" style={{ width: `${Math.round(progress * 100)}%` }} />
          </div>
          <span className="liveStatusValue">
            {status.enrolled
              ? "0.0s left"
              : status.seconds_left != null
              ? `${status.seconds_left.toFixed(1)}s left`
              : "—"}
          </span>
        </div>

        <div className="liveStatusItem">
          <span className="liveStatusLabel">Enroll secs</span>
          <span className="liveStatusValue">
            {status.enroll_sec != null && status.enroll_sec_target != null
              ? `${status.enroll_sec.toFixed(1)}s/${status.enroll_sec_target.toFixed(1)}s`
              : "—"}
          </span>
        </div>

        <div className="liveStatusItem">
          <span className="liveStatusLabel">Live RMS</span>
          <span className="liveStatusValue">{status.rms != null ? status.rms.toFixed(4) : "—"}</span>
        </div>

        <div className="liveStatusItem">
          <span className="liveStatusLabel">Mode</span>
          <span className={`liveStatusMode liveStatusMode-${(status.mode || "").toLowerCase()}`}>{modeLabel}</span>
        </div>

        <div className="liveStatusItem">
          <span className="liveStatusLabel">Last msg</span>
          <span className="liveStatusValue">{lastMsgLabel}</span>
        </div>

        <div className="liveStatusItem">
          <span className="liveStatusLabel">Counts</span>
          <span className="liveStatusValue">
            seg {segments.length} • turn {turns.length} • link {links.length}
          </span>
        </div>
      </div>

      <div className="grid">
        {/* Left card: Connection + controls */}
        <div className="card">
          <div className="cardTitle">Connection</div>
          <div className="row">
            <input value={wsUrl} onChange={(e) => setWsUrl(e.target.value)} className="input" />
            <button onClick={connect} className="btn">
              Reconnect
            </button>
          </div>
          <div className="hint">Backend must be running (audio_v14.py) with WebSocket at /ws.</div>

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

          <div className="sep" />

          <div className="cardTitle">Notes</div>
          <div className="hint">
            <div className="mono" style={{ opacity: 0.9 }}>
              • Segments are raw Whisper outputs.
              <br />
              • Turns get patched by LLM: text → cleaned_text, label + confidence.
              <br />
              • Links may be tombstoned (deleted=true) and replaced after LLM relabel.
            </div>
          </div>
        </div>

        {/* Right card: Timeline + Segments */}
        <div className="card">
          <div className="cardTitle">Segmentation Timeline (last {windowSec}s)</div>
          <div className="timeline">
            <div className="timelineAxis">
              <span className="mono">{fmtSec(timelineStart)}</span>
              <span className="mono">{fmtSec(timelineEnd)}</span>
            </div>

            <div className="timelineBars">
              {segments
                .filter((s) => safeNum(s.end, 0) >= timelineStart)
                .map((s) => {
                  const st = safeNum(s.start, 0);
                  const en = safeNum(s.end, 0);
                  const denom = Math.max(1e-6, timelineEnd - timelineStart);
                  const left = ((st - timelineStart) / denom) * 100;
                  const width = ((en - st) / denom) * 100;
                  const speaker = s.speaker ?? "?";
                  return (
                    <div
                      key={s.event_id}
                      className={`bar bar${speaker}`}
                      style={{ left: `${left}%`, width: `${Math.max(0.5, width)}%` }}
                      title={`seg ${s.seg_id} • ${speaker} • ${fmtSec(st)}-${fmtSec(en)} • ${s.text ?? ""}`}
                    />
                  );
                })}
            </div>
          </div>

          <div className="sep" />

          <div className="cardTitle">Segments (raw ASR)</div>
          <div className="table">
            <div className="thead">
              <div>ID</div>
              <div>Start</div>
              <div>End</div>
              <div>Dur</div>
              <div>Spk</div>
              <div>Text</div>
            </div>

            {segments
              .slice()
              .reverse()
              .map((s) => (
                <div className="trow" key={s.event_id}>
                  <div className="mono" title={s.seg_id}>
                    {shortId(s.seg_id, 14)}
                  </div>
                  <div className="mono">{fmtSec(safeNum(s.start, NaN))}</div>
                  <div className="mono">{fmtSec(safeNum(s.end, NaN))}</div>
                  <div className="mono">{fmtSec(durationOf(s))}</div>
                  <div className={`speaker ${s.speaker === "A" ? "A" : s.speaker === "B" ? "B" : "U"}`}>
                    {s.speaker ?? "?"}
                    {typeof s.sim === "number" ? ` (${s.sim.toFixed(3)})` : ""}
                    {s._patch ? " (patched)" : ""}
                    {typeof s.asr_ms === "number" ? `, ${s.asr_ms}ms` : ""}
                  </div>
                  <div className="textCell">{s.text ?? ""}</div>
                </div>
              ))}

            {segments.length === 0 && <div className="empty">No segments yet. Speak into the mic.</div>}
          </div>
        </div>
      </div>

      {/* Bottom grid: Turns + Links */}
      <div className="grid" style={{ marginTop: 16 }}>
        <div className="card">
          <div className="cardTitle">Turns (LLM patched: label + cleaned_text + per-model labels)</div>
          <div className="table">
            <div className="thead">
              <div>ID</div>
              <div>Spk</div>
              <div>Start</div>
              <div>End</div>
              <div>Final</div>
              <div>GPT</div>
              <div>Llama</div>
              <div>Rule</div>
              <div>Conf</div>
              <div>Raw</div>
              <div>Cleaned</div>
            </div>

            {turns
              .slice()
              .reverse()
              .map((t) => {
                const label = t.label ?? "—";
                const gptLabel = t.gpt_label ?? "—";
                const llamaLabel = t.llama_label ?? "—";
                const ruleLabel = t.rule_label ?? "—";
                const conf = t.confidence ?? "—";
                const raw = t.raw_text ?? t.text ?? "";
                const cleaned = t.cleaned_text ?? t.text ?? "";
                const textChanged =
                  raw && cleaned && String(raw).trim() !== String(cleaned).trim() && t._patch;
                return (
                  <div className="trow" key={t.event_id}>
                    <div className="mono">{t.turn_id}</div>
                    <div className={`speaker ${t.speaker === "A" ? "A" : t.speaker === "B" ? "B" : "U"}`}>
                      {t.speaker ?? "?"}
                      {typeof t.turn_conf === "number" ? ` (${t.turn_conf.toFixed(3)})` : ""}
                      {t._patch ? " (patched)" : ""}
                    </div>
                    <div className="mono">{fmtSec(safeNum(t.start, NaN))}</div>
                    <div className="mono">{fmtSec(safeNum(t.end, NaN))}</div>
                    <div className="mono">{label}</div>
                    <div className="mono">{gptLabel}</div>
                    <div className="mono">{llamaLabel}</div>
                    <div className="mono">{ruleLabel}</div>
                    <div className="mono">{conf}</div>
                    <div className="textCell" title={raw}>
                      {raw}
                    </div>
                    <div className="textCell" title={cleaned}>
                      {cleaned}
                      {textChanged ? <span className="mono" style={{ opacity: 0.7 }}> {" "}★</span> : null}
                    </div>
                  </div>
                );
              })}

            {turns.length === 0 && <div className="empty">No turns yet. Turns flush on speaker A/B (not U).</div>}
          </div>

          <div className="hint" style={{ marginTop: 8 }}>
            ★ appears when LLM patched + cleaned_text differs from raw_text. Confidence is the ensemble result (high/medium/low/rule).
          </div>
        </div>

        <div className="card">
          <div className="cardTitle">Links (AU relations)</div>
          <div className="table">
            <div className="thead">
              <div>Link</div>
              <div>Relation</div>
              <div>Src</div>
              <div>Dst</div>
              <div>Conf</div>
              <div>Start</div>
              <div>End</div>
            </div>

            {links
              .slice()
              .reverse()
              .map((l) => (
                <div className="trow" key={l.event_id}>
                  <div className="mono" title={l.link_id}>{shortId(l.link_id, 18)}</div>
                  <div className="mono">{l.relation ?? "—"}</div>
                  <div className="mono">{l.src_au ?? "—"}</div>
                  <div className="mono">{l.dst_au ?? "—"}</div>
                  <div className="mono">{typeof l.confidence === "number" ? l.confidence.toFixed(2) : "—"}</div>
                  <div className="mono">{fmtSec(safeNum(l.start, NaN))}</div>
                  <div className="mono">{fmtSec(safeNum(l.end, NaN))}</div>
                </div>
              ))}

            {links.length === 0 && <div className="empty">No links yet (premise/support, rebuttal/attack, counterclaim/attack).</div>}
          </div>

          <div className="hint" style={{ marginTop: 8 }}>
            Tombstones (deleted=true) are not shown; they remove previous links after LLM relabel.
          </div>
        </div>
      </div>
    </div>
  );
}