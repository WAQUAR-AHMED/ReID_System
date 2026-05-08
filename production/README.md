# CrossCamReid — production runners

This folder ships **four** ways to run the same multi-camera ReID pipeline
(YOLOv8 pose + keypoint gating + ReID embeddings + Qdrant SID store):

| Runner | File | Purpose |
|---|---|---|
| **CLI** | `app.py` | Local desktop run with cv2 windows. Good for development. |
| **WebSocket (ReID)** | `app_server.py` | One client per WebSocket; per-frame JSON detections streamed back. |
| **HLS HTTP** | `app_hls.py` | Multi-session HTTP API that exposes annotated **HLS playlists** per camera (browser/VLC playback). Mirrors the `ffmpeg_test` API shape. |
| **People Counting WS** | `app_people_counting.py` | WebSocket API for real-time **people counting + occupancy + alerts** built on top of ReID. See [§4](#4-people-counting--occupancy-runner--app_people_countingpy). |

The three runners share the same code under `src/crosscamreid/` — switching
between them does not change pipeline behavior.

---

## Prerequisites

- **Python 3.10+**
- **ffmpeg** on `PATH` (only for `app_hls.py`)
- Optional: **NVIDIA GPU + CUDA** (faster YOLO; required for the
  `tensorrt`/`fastreid` backends configured in `config/config.yaml`)

Install Python deps from the repo root:

```bash
pip install -r ../requirements.txt
```

The repo `requirements.txt` already includes `fastapi` and `uvicorn[standard]`,
which both `app_server.py` and `app_hls.py` need.

---

## Configuration

All three runners load the same YAML schema (`production/config/config.yaml`).
Key sections:

- `cameras` — list of `{id, role: MASTER|SLAVE, source}`. Source can be RTSP,
  HTTP, a local file path, or a webcam index.
- `models.pose_path` / `models.reid_*` — model weights (paths are resolved
  relative to the YAML file).
- `gating` / `enrollment` — ReID thresholds and enrollment voting parameters.
- `database.qdrant` — local or cloud vector store (`keep_db: true` preserves
  the SID gallery between runs).
- `runtime.reid_backend` — `onnxruntime`, `tensorrt`, or `fastreid`.
- `runtime.roi_based_master` — interactive ROI selection on the master frame.
  **Disabled automatically when running under `app_hls.py`** because there is
  no GUI in a server context.
- `runtime.sid_persist_on_kp_loss` — when a tracker ID is already locked to an
  SID, keep returning that SID across transient keypoint dropouts (occlusion).
  Default `false`; enabled only in `localtest/config/local_video.yaml`.

---

## 1. CLI runner — `app.py`

```bash
cd production
python app.py --config config/config.yaml
```

Opens an OpenCV window grid showing every camera with detection overlays.
Press `q` to quit.

---

## 2. WebSocket runner — `app_server.py`

```bash
cd production
uvicorn app_server:app --host 0.0.0.0 --port 8000
```

Connect from a WebSocket client (Postman, `wscat`, etc.):

```
ws://localhost:8000/ws/reid/<any-client-id>
```

Send to start:

```json
{
  "action": "start_stream",
  "cameras": [
    {"cam_id": "cam1", "role": "MASTER", "rtsp_url": "rtsp://..."},
    {"cam_id": "cam2", "role": "SLAVE",  "rtsp_url": "rtsp://..."}
  ]
}
```

Send to stop: `{"action": "stop_stream"}`.

You'll receive one JSON message per processed frame containing per-track SID,
phase (`NEW` / `QUALIFY` / `ENROLL` / `LOCK`), bbox, and similarity score.

`GET /health` returns server status.

---

## 3. HLS HTTP runner — `app_hls.py` (recommended for browser playback)

```bash
cd production
uvicorn app_hls:app --host 0.0.0.0 --port 9000
```

The pipeline runs in a **dedicated worker thread per session**; each camera
in a session has its own `RTSPCapture` reader thread (existing) and its own
`ffmpeg` subprocess that turns annotated frames into an HLS playlist.

### Architecture

```
POST /streams/start
        │
        ▼
HLSStreamSession  (1 worker thread)
        │
        ├── RTSPCapture(cam1)    ── frames ──┐
        ├── RTSPCapture(cam2)    ── frames ──┤
        │                                    │
        │   YOLO + ReID + draw_overlay ◄─────┘
        │                                    │
        ├── FFmpegHLSWriter(cam1) ── stdin ──► ffmpeg ─► hls_out/<id>/cam1/stream.m3u8
        └── FFmpegHLSWriter(cam2) ── stdin ──► ffmpeg ─► hls_out/<id>/cam2/stream.m3u8
```

Multiple sessions can run in parallel; each one gets a fresh `stream_id`
(UUID), its own folder under `hls_out/`, and its own ffmpeg processes. All
cameras inside a single session share one `SIDStore`, so cross-camera ReID
still works.

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/streams/start` | Start a new session. |
| `POST` | `/streams/stop/{stream_id}` | Stop a session and remove its HLS folder. |
| `GET` | `/streams` | List all active sessions. |
| `GET` | `/streams/{stream_id}` | Status of one session (fps, frames written, hls urls). |
| `GET` | `/health` | Liveness + active session count. |
| `GET` | `/hls/{stream_id}/{cam_id}/stream.m3u8` | HLS playlist (auto-served from `hls_out/`). |
| `GET` | `/hls/{stream_id}/{cam_id}/seg_*.ts` | HLS segments. |

### Start a session — using the YAML config as-is

```bash
curl -X POST http://localhost:9000/streams/start \
  -H "Content-Type: application/json" \
  -d '{
    "config_path": "config/config.yaml",
    "use_nvenc": false,
    "output_fps": 25
  }'
```

If `config_path` is omitted, the server falls back to
`production/config/config.yaml`.

### Start a session — overriding the camera list

```bash
curl -X POST http://localhost:9000/streams/start \
  -H "Content-Type: application/json" \
  -d '{
    "cameras": [
      {"id": "cam1", "role": "MASTER", "source": "rtsp://user:pass@192.168.1.10:554/Streaming/Channels/101"},
      {"id": "cam2", "role": "SLAVE",  "source": "rtsp://user:pass@192.168.1.11:554/Streaming/Channels/101"}
    ],
    "use_nvenc": false
  }'
```

Response:

```json
{
  "stream_id": "550e8400-e29b-41d4-a716-446655440000",
  "thread_name": "hls-550e8400",
  "hls_urls": {
    "cam1": "/hls/550e8400-e29b-41d4-a716-446655440000/cam1/stream.m3u8",
    "cam2": "/hls/550e8400-e29b-41d4-a716-446655440000/cam2/stream.m3u8"
  },
  "cameras": [
    {"id": "cam1", "role": "MASTER", "source": "rtsp://..."},
    {"id": "cam2", "role": "SLAVE",  "source": "rtsp://..."}
  ]
}
```

### Watch the stream

Open the playlist URL in any HLS-capable player:

- **VLC**: *Media → Open Network Stream* →
  `http://localhost:9000/hls/<stream_id>/cam1/stream.m3u8`
- **ffplay**: `ffplay http://localhost:9000/hls/<stream_id>/cam1/stream.m3u8`
- **Safari**: paste the URL into the address bar.
- **Chrome / Firefox**: use a tiny [hls.js](https://github.com/video-dev/hls.js)
  page — Chrome and Firefox don't play HLS natively.

It usually takes ~3–5 seconds before the first segment lands on disk and the
playlist becomes valid.

### Stop a session

```bash
curl -X POST http://localhost:9000/streams/stop/<stream_id>
```

This signals the worker thread, drains and closes ffmpeg, stops the capture
threads, and deletes `hls_out/<stream_id>/`.

### Inspect sessions

```bash
curl http://localhost:9000/streams                         # list all
curl http://localhost:9000/streams/<stream_id>             # one session
curl http://localhost:9000/health
```

### `use_nvenc` flag

Mirrors the `ffmpeg_test` API. The writer accepts it and, if the host has
NVIDIA hardware + ffmpeg compiled with `--enable-nvenc`, swaps libx264 for
`h264_nvenc`. Set `false` to force software encoding, which is portable.

---

## 4. People-Counting / Occupancy runner — `app_people_counting.py`

Real-time people counting + occupancy tracking + threshold alerting over a
WebSocket. Runs on top of the **same ReID pipeline** so global IDs (`G1`,
`G2`, …) are stable across cameras of the same `org_id`.

```bash
cd production
uvicorn app_people_counting:app --host 0.0.0.0 --port 8002
```

### Endpoint

```
ws://<host>:8002/ws/people_counting/{client_id}?token=<JWT>
```

- `client_id` — any unique string per client. A second connection using the
  same `client_id` is rejected while the first is still active.
- `token` — required. Validated as **HS256 JWT** when env `PC_JWT_SECRET`
  is set; otherwise any non-empty token is accepted (dev mode; a one-time
  warning is logged).

### Auth

```bash
# Production
export PC_JWT_SECRET=<your-strong-secret>

# Dev — leave PC_JWT_SECRET unset; any non-empty token works
```

Optional dependencies (lazy-imported):

```bash
pip install PyJWT      # JWT validation
pip install boto3      # required for kvs:// stream URLs
pip install pynvml     # GPU utilisation in the response payload
```

### Architecture

```
WebSocket  (one per client)
     │
     ▼
PeopleCountingSession   (max 5 cameras per session)
     │
     ├── _CameraWorker (thread per camera)
     │     • YOLO + RTSPCapture + TIDStateManager
     │     • process_master(...) reused unchanged from processor.py
     │     • EntryExitTracker  (new id ⇒ entry; timeout ⇒ exit)
     │     • OccupancyTracker  (edge-triggered alerts)
     │     • emits one JSON payload per processed frame
     │
     └── OrgRegistry  (process-wide, ref-counted)
            • per-org_id Qdrant collection: <base>__org<id>
            • shared SIDStore + ReID backend across cameras of the org
            • strict cross-org isolation
```

A single connection can stream up to **5 cameras**; each runs in its own
thread so a slow stream never stalls the others.

### Inbound messages

**Start streams** (1–5 cameras):

```json
{
  "action": "start_stream",
  "org_id": 10,
  "user_id": 42,

  "threshold": 50,
  "alert_rate": 80,

  "fps": 5,
  "frame_skip": 2,

  "cameras": [
    { "camera_id": 1, "stream_url": "kvs://CamEntrance01", "region": "ap-south-1" },
    { "camera_id": 2, "stream_url": "rtsp://192.168.1.10/stream" },
    { "camera_id": 3, "stream_url": "https://example.com/sample.mp4" }
  ]
}
```

`stream_url` schemes:

| Scheme | Resolved by |
|---|---|
| `kvs://<StreamName>` | AWS Kinesis Video Streams → HLS via boto3 (uses `region`) |
| `rtsp://...` | OpenCV |
| `http(s)://...` | OpenCV |
| local path | OpenCV |

**Stop one camera**:

```json
{ "action": "stop_stream", "org_id": 10, "user_id": 42, "camera_id": 1 }
```

**Stop all cameras for this connection**:

```json
{ "action": "stop_all", "org_id": 10, "user_id": 42 }
```

A connection is bound to the first `org_id` it successfully started. A
later `start_stream` with a different `org_id` is rejected — reconnect to
switch tenants.

### Outbound events

**Stream lifecycle**:

```json
{ "event": "stream_started", "camera_id": 1, "status": "ok",
  "message": "Stream initialized successfully" }

{ "event": "error", "camera_id": 1, "status": "failed",
  "error_message": "Unable to connect to stream" }

{ "event": "stream_stopped", "camera_id": 1, "status": "ok",
  "message": "Camera stopped" }

{ "event": "all_stopped", "status": "ok", "stopped": 3 }
```

**Per-frame payload** (one per processed frame, per camera; flat schema):

```json
{
  "detections": {
    "camid": 1,
    "org_id": 10,
    "userid": 42,

    "Frame_Id": "FR_1_1734456789123",
    "Time_stamp": "2024-12-17T15:39:49.123456Z",
    "Frame_Count": 42,

    "Total_people_detected": 3,
    "Current_occupancy": 3,

    "People_ids":         ["G1", "G3", "G7"],
    "Entry_time":         ["2024-12-17T15:39:45Z", "...", "..."],
    "People_dwell_time":  ["00:00:04", "00:00:02", "00:00:00"],
    "Confidence_scores":  [0.91, 0.87, 0.83],
    "accuracy":           [0.910, 0.870, 0.830],

    "Exit_time": ["2024-12-17T15:39:47Z"],
    "exitid":    ["G5"],

    "Total_entries": 15,
    "Total_exits": 12,
    "Net_count": 3,

    "Occupancy_percentage": 30.0,
    "Over_capacity_count": 0,
    "Max_occupancy": 50,
    "Average_dwell_time": "00:00:02",

    "Status": "",
    "is_alert_triggered": false,

    "Processing_Status": 1,
    "processing_time_ms": 74.5,
    "gpu_memory_percent": 42.3,
    "gpu_utilization_percent": 68.0,
    "reid_gallery_size": 8,

    "annotated_frame": null
  }
}
```

Notes on the schema:

- `People_ids` / `Entry_time` / `People_dwell_time` / `Confidence_scores` /
  `accuracy` are **parallel arrays** — index `i` describes the same person
  across all five.
- `Exit_time` / `exitid` are **parallel arrays** holding the **last 10**
  exits seen on this camera.
- `Status` is `""` while normal/approaching, `"High Occupancy"` once
  occupancy reaches the threshold, and `"Error"` on error frames.
- `is_alert_triggered` is **edge-triggered** — `true` only on the rising
  edge into a new alert level.
- `annotated_frame` is the raw base64 JPEG (no `data:` prefix). It is
  populated on **every Nth processed frame** (default `N = 20`) and `null`
  in between, to keep bandwidth bounded.

Frame controls (all optional on `start_stream`):

```json
{
  "include_annotated_frame": true,
  "frame_jpeg_quality":      70,
  "frame_send_interval":     20
}
```

- `include_annotated_frame` — set `false` to disable frame upload entirely.
- `frame_jpeg_quality` — clamped 30–95.
- `frame_send_interval` — N. The server emits `annotated_frame` only when
  `Frame_Count % N == 0`; otherwise the field is `null`.

### Counting / occupancy semantics

- **Entry**: a global ID seen for the first time on a camera since startup
  (or after a previous exit).
- **Exit**: an active ID is absent for `exit_timeout_sec` (default `2.0`).
  Tunable per `start_stream` request via `"exit_timeout_sec": <float>`.
- **`current_occupancy`**: number of distinct global IDs currently active
  on the camera.
- **`net_count`**: `total_entries - total_exits`.
- **Alert**: `alert_threshold = threshold * alert_rate / 100`. Edge-triggered
  — the alert fires **once** when occupancy crosses up into a higher level
  (`approaching_threshold` → `at_threshold` → `over_capacity`) and only re-
  fires after dropping below that level and crossing it again.
- **Average dwell**: mean dwell of currently visible people; if nobody is
  visible, falls back to the mean dwell of the **last 10 exits**.
- **Annotated frame**: the worker draws bounding boxes + ID labels onto
  the frame and emits it as `annotated_frame` (base64 JPEG, no `data:`
  prefix) every `frame_send_interval` processed frames. Disable
  per-request with `"include_annotated_frame": false`, tune compression
  with `"frame_jpeg_quality": <30-95>`, or change the cadence with
  `"frame_send_interval": <int>` (default 20). The field is `null` on
  intervening frames.

### Performance controls

| Field | Effect |
|---|---|
| `fps` | Target processing FPS per camera. Loop sleeps to honor the cap. `0` = no cap. |
| `frame_skip` | Process 1 frame, skip K. Default `0`. |
| `threshold` | Capacity used by occupancy/alerts. `0` disables alerts. |
| `alert_rate` | Percent of `threshold` that triggers `approaching_threshold`. |
| `exit_timeout_sec` | Seconds an ID can be missing before counted as exited. Default `2.0`. |

### Cross-org isolation

- Each `org_id` owns its own Qdrant collection: `<base>__org<id>`.
- A `PeopleCountingSession` is bound to one `org_id` for its lifetime.
- The shared `OrgRegistry` ref-counts org resources but does **not**
  dispose them when the last session disconnects (a brief reconnect must
  not wipe the SID gallery).

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| `WS` | `/ws/people_counting/{client_id}?token=…` | Main WebSocket. |
| `GET` | `/health` | Liveness + active session count. |

### Quick smoke test

`wscat` works well for one-off testing:

```bash
wscat -c "ws://localhost:8002/ws/people_counting/test-1?token=devtoken"
> {"action":"start_stream","org_id":10,"user_id":42,"threshold":50,"alert_rate":80,"fps":5,"frame_skip":0,"cameras":[{"camera_id":1,"stream_url":"path/to/sample.mp4"}]}
```

You should immediately receive `stream_started`, then a steady stream of
`detections` payloads.

### File layout (people-counting only)

```
production/
├── app_people_counting.py                          ◄── new
└── src/crosscamreid/
    ├── counting/                                   ◄── new
    │   ├── __init__.py
    │   ├── entry_exit.py        # per-camera entry/exit tracker
    │   ├── occupancy.py         # threshold + edge-triggered alerts
    │   ├── dwell.py             # avg dwell with last-10 fallback
    │   ├── kvs_resolver.py      # kvs:// → HLS via boto3
    │   ├── gpu_stats.py         # pynvml → torch → 0.0 fallback
    │   ├── auth.py              # JWT validator (HS256)
    │   └── org_registry.py      # per-org SIDStore + ReID backend pool
    └── websocket/
        ├── people_counting_runner.py               ◄── new
        └── people_counting_handler.py              ◄── new
```

The existing ReID modules (`processor.py`, `state.py`, `store.py`,
`capture.py`, `reid/*`) are reused **unchanged**.

---

## Local testing without RTSP cameras

For quick smoke tests, point `source` at a video file or HTTP MP4:

```bash
curl -X POST http://localhost:9000/streams/start \
  -H "Content-Type: application/json" \
  -d '{
    "cameras": [
      {"id": "cam1", "role": "MASTER", "source": "/path/to/sample.mp4"}
    ],
    "use_nvenc": false
  }'
```

The same source field accepts a webcam index as a string (e.g. `"0"`).

---

## Troubleshooting

- **`ffmpeg not found on PATH`** — install ffmpeg and reopen the shell.
  On Windows, `winget install Gyan.FFmpeg`. On macOS, `brew install ffmpeg`.
- **HLS playlist 404 for a few seconds after start** — normal: ffmpeg needs
  to write at least one segment (`hls_time: 2s`) before the playlist is
  valid. Reload after ~5 seconds.
- **Worker exits immediately, session shows `error`** — usually one of:
  RTSP URL unreachable, codec OpenCV can't open, or model file missing.
  `GET /streams/{id}` shows the error string.
- **Cross-camera SIDs don't match between sessions** — expected. Each
  session is independent unless `database.qdrant.keep_db: true` is set in
  the YAML; with `keep_db: true`, sessions share the persisted gallery.
- **`runtime.roi_based_master` warning in logs** — ROI selection requires a
  GUI window. Server mode disables it automatically; the warning is just FYI.
- **High GPU memory** — every session loads its own YOLO instance per camera.
  Cap concurrent sessions or reduce camera count per session.

---

## File layout (post-change)

```
production/
├── app.py                                  # CLI runner (cv2 window)
├── app_server.py                           # ReID WebSocket server
├── app_hls.py                              # HLS HTTP server
├── app_people_counting.py                  # People-counting WS server  ◄── new
├── README.md                               # this file
├── config/
│   └── config.yaml
├── hls_out/                                # created at runtime by app_hls.py
└── src/crosscamreid/
    ├── capture.py                          # RTSPCapture (thread per camera)
    ├── config.py
    ├── keypoints.py
    ├── overlay.py
    ├── pipeline.py                         # used by app.py
    ├── processor.py                        # process_master / _slave / _master_roi
    ├── reid/                               # onnx / tensorrt / fastreid backends
    ├── state.py
    ├── store.py
    ├── database/                           # PostgreSQL persistence
    ├── counting/                           # people-counting layer       ◄── new
    │   ├── entry_exit.py
    │   ├── occupancy.py
    │   ├── dwell.py
    │   ├── kvs_resolver.py
    │   ├── gpu_stats.py
    │   ├── auth.py
    │   └── org_registry.py
    ├── websocket/                          # ReID + people-counting handlers
    │   ├── reid_handler.py
    │   ├── reid_runner.py
    │   ├── people_counting_handler.py      ◄── new
    │   └── people_counting_runner.py       ◄── new
    └── server/                             # used by app_hls.py
        ├── __init__.py
        ├── ffmpeg_hls.py                   # FFmpegHLSWriter
        └── hls_session.py                  # HLSStreamSession (worker thread)
```
