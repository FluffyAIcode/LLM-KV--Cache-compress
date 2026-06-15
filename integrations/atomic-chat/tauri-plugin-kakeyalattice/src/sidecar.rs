//! Sidecar process supervisor.
//!
//! Responsibilities:
//! 1. Resolve the sidecar binary path.
//! 2. Spawn it with the configured host/port/device.
//! 3. Tail stdout/stderr into the application log.
//! 4. Health-check on an interval; restart with exponential backoff
//!    on crash. (Identical policy to the existing llama.cpp plugin.)
//!
//! This file is the skeleton; the heavy lifting (signal handling on
//! app quit, Windows-specific job objects, etc.) will follow in a
//! dedicated PR inside the Atomic-Chat main repo.

use std::path::PathBuf;
use std::process::Stdio;
use std::time::Duration;

use tauri::{AppHandle, Manager, Runtime};
use tokio::io::{AsyncBufReadExt, BufReader};

use crate::{Error, PluginState, Result};

pub struct SidecarHandle {
    pub child: tokio::process::Child,
}

fn resolve_sidecar_binary() -> PathBuf {
    if let Ok(p) = std::env::var("ATOMIC_CHAT_KAKEYA_SIDECAR_PATH") {
        return PathBuf::from(p);
    }
    // In product builds we bundle the sidecar next to the Tauri resource
    // root. For dev builds we fall back to the `$PATH` lookup so
    // `pip install -e kakeya_sidecar` (which installs the
    // `kakeya-sidecar` console script) works out of the box.
    PathBuf::from("kakeya-sidecar")
}

pub async fn start_supervisor<R: Runtime>(app: &AppHandle<R>) -> Result<()> {
    let state = app.state::<PluginState>();
    let cfg = state.config.clone();

    let bin = resolve_sidecar_binary();
    log::info!(
        "spawning kakeya-sidecar: {} --host {} --port {} --device {}",
        bin.display(), cfg.sidecar_host, cfg.sidecar_port, cfg.device,
    );

    let mut child = tokio::process::Command::new(&bin)
        .arg("--host").arg(&cfg.sidecar_host)
        .arg("--port").arg(cfg.sidecar_port.to_string())
        .arg("--device").arg(&cfg.device)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| Error::SidecarSpawn(e.to_string()))?;

    if let Some(out) = child.stdout.take() {
        let mut reader = BufReader::new(out).lines();
        tokio::spawn(async move {
            while let Ok(Some(line)) = reader.next_line().await {
                log::info!("[sidecar stdout] {line}");
            }
        });
    }
    if let Some(err) = child.stderr.take() {
        let mut reader = BufReader::new(err).lines();
        tokio::spawn(async move {
            while let Ok(Some(line)) = reader.next_line().await {
                log::warn!("[sidecar stderr] {line}");
            }
        });
    }

    // Health-check loop: wait up to 30s for the sidecar to respond.
    let base_url = state.base_url();
    let http = state.http.clone();
    for attempt in 0..30 {
        tokio::time::sleep(Duration::from_secs(1)).await;
        if let Ok(resp) = http.get(format!("{}/health", base_url)).send().await {
            if resp.status().is_success() {
                log::info!("kakeya-sidecar online after {}s", attempt + 1);
                let mut guard = state.supervisor.lock().await;
                *guard = Some(SidecarHandle { child });
                return Ok(());
            }
        }
    }

    // Sidecar didn't come up; kill the child so we don't leak zombies.
    let _ = child.kill().await;
    Err(Error::SidecarNotReady { tries: 30 })
}
