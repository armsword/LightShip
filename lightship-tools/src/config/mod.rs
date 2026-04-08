//! Configuration management for LightShip
//!
//! Supports TOML and JSON configuration files.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Engine configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EngineConfig {
    /// Log level
    #[serde(default = "default_log_level")]
    pub log_level: String,

    /// Number of threads (0 = auto)
    #[serde(default)]
    pub num_threads: usize,

    /// Backend selection
    #[serde(default)]
    pub backend: BackendConfig,

    /// Memory settings
    #[serde(default)]
    pub memory: MemoryConfig,
}

fn default_log_level() -> String {
    "info".to_string()
}

fn default_true() -> bool {
    true
}

/// Backend configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendConfig {
    Auto,
    Cpu,
    Gpu,
    Vulkan,
    Metal,
    Npu,
    Preferred { preferred: String, fallback: String },
}

impl Default for BackendConfig {
    fn default() -> Self {
        Self::Auto
    }
}

/// Memory configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    /// Maximum memory usage in bytes (0 = unlimited)
    #[serde(default)]
    pub max_memory: usize,

    /// Enable memory pooling
    #[serde(default = "default_true")]
    pub enable_pooling: bool,

    /// Memory reuse strategy
    #[serde(default)]
    pub reuse_strategy: MemoryReuseStrategy,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            max_memory: 0,
            enable_pooling: true,
            reuse_strategy: MemoryReuseStrategy::Default,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryReuseStrategy {
    Default,
    Aggressive,
    Conservative,
}

impl Default for MemoryReuseStrategy {
    fn default() -> Self {
        Self::Default
    }
}

/// Session configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionConfig {
    /// Backend to use
    #[serde(default)]
    pub backend: BackendConfig,

    /// Enable profiling
    #[serde(default)]
    pub enable_profiling: bool,

    /// Profiling level
    #[serde(default = "default_profiling_level")]
    pub profiling_level: String,

    /// Execution mode
    #[serde(default)]
    pub execution_mode: ExecutionMode,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            backend: BackendConfig::Auto,
            enable_profiling: false,
            profiling_level: "basic".to_string(),
            execution_mode: ExecutionMode::Synchronous,
        }
    }
}

fn default_profiling_level() -> String {
    "basic".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionMode {
    Synchronous,
    Asynchronous,
}

impl Default for ExecutionMode {
    fn default() -> Self {
        Self::Synchronous
    }
}

/// Benchmark configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    /// Number of warmup runs
    #[serde(default = "default_warmup")]
    pub warmup_runs: usize,

    /// Number of benchmark runs
    #[serde(default = "default_runs")]
    pub runs: usize,

    /// Input shape for benchmarking
    #[serde(default)]
    pub input_shape: Vec<usize>,

    /// Batch size
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,

    /// Number of threads for CPU backend
    #[serde(default)]
    pub num_threads: usize,

    /// Enable detailed timing
    #[serde(default)]
    pub detailed_timing: bool,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            warmup_runs: default_warmup(),
            runs: default_runs(),
            input_shape: vec![1, 3, 224, 224],
            batch_size: default_batch_size(),
            num_threads: 0,
            detailed_timing: false,
        }
    }
}

fn default_warmup() -> usize {
    10
}

fn default_runs() -> usize {
    100
}

fn default_batch_size() -> usize {
    1
}

/// Model conversion configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversionConfig {
    /// Input model format
    pub input_format: ModelFormat,

    /// Output model format
    #[serde(default)]
    pub output_format: ModelFormat,

    /// Quantization settings
    #[serde(default)]
    pub quantization: Option<QuantizationSettings>,

    /// Optimization level
    #[serde(default)]
    pub optimization_level: u32,

    /// Target backend
    #[serde(default)]
    pub target_backend: Option<String>,
}

impl Default for ConversionConfig {
    fn default() -> Self {
        Self {
            input_format: ModelFormat::Onnx,
            output_format: ModelFormat::Native,
            quantization: None,
            optimization_level: 3,
            target_backend: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelFormat {
    Native,
    Onnx,
    TFLite,
    Caffe,
}

impl Default for ModelFormat {
    fn default() -> Self {
        Self::Native
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct QuantizationSettings {
    /// Quantization type
    #[serde(default)]
    pub quant_type: QuantizationType,

    /// Enable per-channel quantization
    #[serde(default)]
    pub per_channel: bool,

    /// Calibration data path (optional)
    #[serde(default)]
    pub calibration_data: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QuantizationType {
    #[serde(rename = "int8")]
    Int8,
    #[serde(rename = "fp16")]
    Fp16,
    #[serde(rename = "mixed")]
    Mixed,
}

impl Default for QuantizationType {
    fn default() -> Self {
        Self::Int8
    }
}

impl Config {
    /// Load configuration from a file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let extension = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("");

        let content = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read config file: {}", path.display()))?;

        match extension {
            "toml" => toml::from_str(&content)
                .context("Failed to parse TOML config"),
            "json" => serde_json::from_str(&content)
                .context("Failed to parse JSON config"),
            _ => anyhow::bail!("Unsupported config format: {}", extension),
        }
    }

    /// Save configuration to a file
    pub fn to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let path = path.as_ref();
        let extension = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("");

        let content = match extension {
            "toml" => toml::to_string_pretty(self)
                .context("Failed to serialize TOML")?,
            "json" => serde_json::to_string_pretty(self)
                .context("Failed to serialize JSON")?,
            _ => anyhow::bail!("Unsupported config format: {}", extension),
        };

        std::fs::write(path, content)
            .with_context(|| format!("Failed to write config file: {}", path.display()))?;

        Ok(())
    }
}

/// LightShip configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    #[serde(default)]
    pub engine: EngineConfig,

    #[serde(default)]
    pub session: SessionConfig,

    #[serde(default)]
    pub benchmark: BenchmarkConfig,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            engine: EngineConfig::default(),
            session: SessionConfig::default(),
            benchmark: BenchmarkConfig::default(),
        }
    }
}

impl Config {
    /// Create a new config with sensible defaults
    pub fn new() -> Self {
        Self::default()
    }

    /// Load config from a file (auto-detect format)
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        Self::from_file(path)
    }

    /// Save config to a file
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        self.to_file(path)
    }
}
