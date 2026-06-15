//! tauri-plugin-kakeyalattice
//!
//! Tauri 2 plugin that supervises the `kakeya-sidecar` Python process
//! and proxies OpenAI-compatible calls from the Atomic-Chat extension
//! to the sidecar HTTP server at `localhost:1338`.
//!
//! This file is the skeleton entry point. The full supervisor logic
//! will land in `sidecar.rs` / `commands.rs` in a dedicated PR inside
//! the Atomic-Chat main repo.

use tauri::{plugin::TauriPlugin, Manager, Runtime};

mod commands;
mod error;
mod sidecar;

pub use error::{Error, Result};

/// Configuration block read from `tauri.conf.json`:
///
/// ```json
/// { "plugins": { "kakeyalattice": { "sidecarPort": 1338, "autoStart": true } } }
/// ```
#[derive(Clone, Debug, serde::Deserialize)]
pub struct PluginConfig {
    #[serde(default = "default_port")]
    pub sidecar_port: u16,
    #[serde(default = "default_host")]
    pub sidecar_host: String,
    #[serde(default = "default_auto_start")]
    pub auto_start: bool,
    #[serde(default = "default_device")]
    pub device: String,
}

fn default_port() -> u16 { 1338 }
fn default_host() -> String { "127.0.0.1".to_string() }
fn default_auto_start() -> bool { true }
fn default_device() -> String { "auto".to_string() }

impl Default for PluginConfig {
    fn default() -> Self {
        Self {
            sidecar_port: default_port(),
            sidecar_host: default_host(),
            auto_start: default_auto_start(),
            device: default_device(),
        }
    }
}

/// Shared per-app plugin state.
#[derive(Default)]
pub struct PluginState {
    pub config: PluginConfig,
    pub http: reqwest::Client,
    pub supervisor: tokio::sync::Mutex<Option<sidecar::SidecarHandle>>,
}

impl PluginState {
    pub fn base_url(&self) -> String {
        format!("http://{}:{}", self.config.sidecar_host, self.config.sidecar_port)
    }
}

/// Plugin factory.
pub fn init<R: Runtime>() -> TauriPlugin<R> {
    tauri::plugin::Builder::new("kakeyalattice")
        .invoke_handler(tauri::generate_handler![
            commands::list_models,
            commands::chat_completion,
            commands::chat_completion_stream_start,
            commands::chat_completion_stream_wait,
            commands::health,
            commands::stats,
        ])
        .setup(|app, api| {
            let cfg: PluginConfig = api
                .config()
                .cloned()
                .and_then(|v| serde_json::from_value(v).ok())
                .unwrap_or_default();

            let state = PluginState {
                config: cfg,
                http: reqwest::Client::new(),
                supervisor: Default::default(),
            };
            app.manage(state);

            if app.state::<PluginState>().config.auto_start {
                // Kick off the sidecar supervisor. Errors are logged but
                // not fatal — the UI will show an offline state and the
                // user can retry.
                let handle = app.clone();
                tauri::async_runtime::spawn(async move {
                    if let Err(e) = sidecar::start_supervisor(&handle).await {
                        log::error!("kakeyalattice sidecar supervisor failed: {e}");
                    }
                });
            }
            Ok(())
        })
        .build()
}
