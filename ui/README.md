# ArgueViz transcript UI

Live transcript panel that connects to the pipeline WebSocket.

1. Install: `npm install`
2. Run backend: `python audio_v2.py` (from project root; requires `fastapi`, `uvicorn`). For reliable WebSocket: `pip install "uvicorn[standard]"` or `pip install websockets`.
3. Run UI: `npm run dev` → open http://localhost:5173

Connects to `ws://localhost:8000/ws` and renders segments in a scrolling panel.
