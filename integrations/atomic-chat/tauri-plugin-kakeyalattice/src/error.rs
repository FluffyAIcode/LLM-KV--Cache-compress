use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("http error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("sidecar not ready (tried {tries} times)")]
    SidecarNotReady { tries: u32 },

    #[error("sidecar spawn failed: {0}")]
    SidecarSpawn(String),

    #[error("unexpected sidecar response: {0}")]
    Protocol(String),
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

impl serde::Serialize for Error {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&self.to_string())
    }
}
