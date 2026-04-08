//! Logging utilities for LightShip

use tracing::Level;
use tracing_subscriber::{
    fmt::{self, format::FmtSpan},
    layer::SubscriberExt,
    util::SubscriberInitExt,
    EnvFilter,
};

/// Log level enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogLevel {
    /// Error level
    Error,
    /// Warning level
    Warn,
    /// Info level
    Info,
    /// Debug level
    Debug,
    /// Trace level
    Trace,
}

impl From<LogLevel> for Level {
    fn from(level: LogLevel) -> Self {
        match level {
            LogLevel::Error => Level::ERROR,
            LogLevel::Warn => Level::WARN,
            LogLevel::Info => Level::INFO,
            LogLevel::Debug => Level::DEBUG,
            LogLevel::Trace => Level::TRACE,
        }
    }
}

impl Default for LogLevel {
    fn default() -> Self {
        LogLevel::Info
    }
}

/// Initialize the logger with the specified log level
pub fn init_logger(level: LogLevel) {
    let level_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info"));

    let fmt_layer = fmt::layer()
        .with_target(true)
        .with_thread_ids(false)
        .with_file(true)
        .with_line_number(true)
        .with_span_events(FmtSpan::CLOSE)
        .compact();

    tracing_subscriber::registry()
        .with(level_filter)
        .with(fmt_layer)
        .init();

    tracing::info!("LightShip logger initialized with level: {:?}", level);
}

/// Set the global log level
pub fn set_log_level(level: LogLevel) {
    let level: Level = level.into();
    tracing::subscriber::set_global_default(
        tracing_subscriber::fmt()
            .with_max_level(level)
            .finish()
    )
    .ok();
}
