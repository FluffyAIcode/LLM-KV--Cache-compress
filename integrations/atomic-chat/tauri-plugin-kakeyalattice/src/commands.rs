//! Tauri command handlers — thin HTTP proxies to the sidecar.
//!
//! Streaming uses a `(start, wait)` pair:
//!   - `chat_completion_stream_start` spawns a task reading SSE from
//!     the sidecar, emits each delta as a Tauri event on
//!     `kakeyalattice:<stream_id>`, and returns the id.
//!   - `chat_completion_stream_wait` awaits a oneshot channel so the
//!     JS caller can `await` final completion instead of polling.

use futures_util::StreamExt;
use serde_json::Value as Json;
use std::collections::HashMap;
use std::sync::Mutex;
use tauri::{AppHandle, Emitter, Runtime, State};
use tokio::sync::oneshot;
use uuid::Uuid;

use crate::{Error, PluginState, Result};

lazy_static::lazy_static! {
    // stream_id -> oneshot receiver, waiting for the SSE loop to finish.
    static ref STREAM_WAITERS: Mutex<HashMap<String, oneshot::Receiver<()>>> = Mutex::new(HashMap::new());
}

// We use `lazy_static` to avoid pulling `once_cell` just for this one
// static. If the crate disallows lazy_static, swap to
// `tokio::sync::OnceCell` in a follow-up.

#[tauri::command]
pub async fn list_models(state: State<'_, PluginState>) -> Result<Json> {
    let url = format!("{}/v1/models", state.base_url());
    let resp = state.http.get(url).send().await?;
    Ok(resp.json::<Json>().await?)
}

#[tauri::command]
pub async fn chat_completion(
    state: State<'_, PluginState>,
    request: Json,
) -> Result<Json> {
    let url = format!("{}/v1/chat/completions", state.base_url());
    let resp = state.http.post(url).json(&request).send().await?;
    Ok(resp.json::<Json>().await?)
}

#[tauri::command]
pub async fn chat_completion_stream_start<R: Runtime>(
    app: AppHandle<R>,
    state: State<'_, PluginState>,
    mut request: Json,
) -> Result<String> {
    // Force stream=true on the request we send to the sidecar.
    if let Some(obj) = request.as_object_mut() {
        obj.insert("stream".into(), Json::Bool(true));
    }

    let url = format!("{}/v1/chat/completions", state.base_url());
    let resp = state.http.post(url).json(&request).send().await?;
    if !resp.status().is_success() {
        let code = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err(Error::Protocol(format!("sidecar returned {code}: {body}")));
    }

    let stream_id = Uuid::new_v4().simple().to_string();
    let event_name = format!("kakeyalattice:{stream_id}");

    let (done_tx, done_rx) = oneshot::channel();
    {
        let mut waiters = STREAM_WAITERS.lock().expect("stream waiters lock");
        waiters.insert(stream_id.clone(), done_rx);
    }

    let app_clone = app.clone();
    tauri::async_runtime::spawn(async move {
        let mut stream = resp.bytes_stream();
        let mut buf = Vec::<u8>::new();
        while let Some(chunk) = stream.next().await {
            let chunk = match chunk {
                Ok(c) => c,
                Err(e) => {
                    log::warn!("sidecar stream error: {e}");
                    break;
                }
            };
            buf.extend_from_slice(&chunk);
            while let Some(idx) = buf.windows(2).position(|w| w == b"\n\n") {
                let frame: Vec<u8> = buf.drain(..idx + 2).collect();
                let text = String::from_utf8_lossy(&frame);
                for line in text.lines() {
                    let line = line.trim();
                    if !line.starts_with("data:") { continue; }
                    let payload = line.trim_start_matches("data:").trim();
                    if payload == "[DONE]" {
                        let _ = app_clone.emit(&event_name, "__DONE__");
                        continue;
                    }
                    if let Ok(json) = serde_json::from_str::<Json>(payload) {
                        if let Some(delta) = json.pointer("/choices/0/delta/content")
                            .and_then(Json::as_str)
                        {
                            let _ = app_clone.emit(&event_name, delta.to_string());
                        }
                    }
                }
            }
        }
        let _ = done_tx.send(());
    });

    Ok(stream_id)
}

#[tauri::command]
pub async fn chat_completion_stream_wait(stream_id: String) -> Result<()> {
    let rx = {
        let mut waiters = STREAM_WAITERS.lock().expect("stream waiters lock");
        waiters.remove(&stream_id)
    };
    if let Some(rx) = rx {
        let _ = rx.await;
    }
    Ok(())
}

#[tauri::command]
pub async fn health(state: State<'_, PluginState>) -> Result<Json> {
    let url = format!("{}/health", state.base_url());
    let resp = state.http.get(url).send().await?;
    Ok(resp.json::<Json>().await?)
}

#[tauri::command]
pub async fn stats(state: State<'_, PluginState>) -> Result<Json> {
    let url = format!("{}/v1/kakeya/stats", state.base_url());
    let resp = state.http.get(url).send().await?;
    Ok(resp.json::<Json>().await?)
}
