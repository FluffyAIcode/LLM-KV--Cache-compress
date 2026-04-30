# tauri-plugin-kakeyalattice

Tauri 2 plugin that:

1. **Supervises** the Python `kakeya-sidecar` process (spawn / health-check / restart).
2. **Proxies** OpenAI-compatible HTTP calls from the Atomic-Chat extension
   (`@atomic-chat/kakeyalattice-extension`) to the sidecar at `localhost:1338`.
3. **Forwards streaming deltas** from the sidecar's SSE stream as Tauri
   events on the channel `kakeyalattice:<stream_id>`.

Registered Tauri commands (used by the TS extension):

| Command | Purpose |
|:--|:--|
| `plugin:kakeyalattice\|list_models` | Proxy `GET /v1/models` |
| `plugin:kakeyalattice\|chat_completion` | Proxy `POST /v1/chat/completions` (non-stream) |
| `plugin:kakeyalattice\|chat_completion_stream_start` | Start a stream, return a stream id |
| `plugin:kakeyalattice\|chat_completion_stream_wait` | Await stream completion |
| `plugin:kakeyalattice\|health` | `GET /health` |
| `plugin:kakeyalattice\|stats` | `GET /v1/kakeya/stats` |

## Sidecar lifecycle

- **Path resolution order**:
  1. `ATOMIC_CHAT_KAKEYA_SIDECAR_PATH` env override.
  2. Bundled binary at `resources/kakeya-sidecar[.exe]` (product builds).
  3. `$PATH` lookup (dev builds — assumes `pip install -e ./kakeya_sidecar`).
- **Arguments**: `--host 127.0.0.1 --port 1338 --device auto --log-level info`.
- **Supervision**: Tauri plugin spawns the process at `setup()` time,
  tails its stdout/stderr into the app log, and restarts on crash
  with exponential backoff (aligned with the existing llamacpp plugin).
- **Port**: pinned to 1338 to sit next to Atomic-Chat's 1337 front door.
  Passed via CLI flag so concurrent instances are possible in tests.

## Why a separate Rust plugin (instead of calling sidecar HTTP from JS)

- Sidecar lifecycle is a native concern (fork / waitpid / SIGTERM on app
  quit). Keeping it in Rust mirrors the `llamacpp` plugin and avoids
  duplicating supervision logic in two languages.
- Tauri's permission model lets us whitelist the sidecar HTTP origin
  without opening CORS holes for the whole web-app.
- Streaming SSE → Tauri events gives the web-app a normal event-bus
  interface instead of fiddling with EventSource behind CSP.

## Minimal example (host app wiring)

In `src-tauri/src/main.rs`:

```rust
fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_kakeyalattice::init())
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
```

In `tauri.conf.json` under `plugins`:

```json
{
  "plugins": {
    "kakeyalattice": {
      "sidecarPort": 1338,
      "autoStart": true,
      "device": "auto"
    }
  }
}
```

## Status

This directory ships the **skeleton** only: lib + commands declared,
no fleshed-out sidecar supervisor yet. `cargo check` passes. Full
implementation (process spawn, SSE bridge, restart policy) will follow
in a dedicated PR inside the Atomic-Chat main repo; this repo keeps
the plugin in a shape the Atomic-Chat maintainers can drop in.
