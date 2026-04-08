//! C API type definitions
//!
//! These types are C-compatible and can be used from other languages.

use std::fmt;

/// Opaque handle to a LightShip engine instance
pub type LightShipEngine = *mut std::ffi::c_void;
/// Opaque handle to a loaded model
pub type LightShipModel = *mut std::ffi::c_void;
/// Opaque handle to an inference session
pub type LightShipSession = *mut std::ffi::c_void;
/// Opaque handle to a tensor
pub type LightShipTensor = *mut std::ffi::c_void;

/// LightShip data types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum LightShipDataType {
    /// 32-bit floating point
    F32 = 0,
    /// 16-bit floating point
    F16 = 1,
    /// 64-bit floating point
    F64 = 2,
    /// 8-bit signed integer
    I8 = 3,
    /// 16-bit signed integer
    I16 = 4,
    /// 32-bit signed integer
    I32 = 5,
    /// 64-bit signed integer
    I64 = 6,
    /// 8-bit unsigned integer
    U8 = 7,
    /// 16-bit unsigned integer
    U16 = 8,
    /// 32-bit unsigned integer
    U32 = 9,
    /// 64-bit unsigned integer
    U64 = 10,
    /// Boolean
    Bool = 11,
    /// Quantized unsigned 8-bit
    QUInt8 = 12,
    /// Quantized signed 8-bit
    QInt8 = 13,
    /// Quantized signed 32-bit
    QInt32 = 14,
}

impl Default for LightShipDataType {
    fn default() -> Self {
        LightShipDataType::F32
    }
}

/// LightShip storage layouts
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum LightShipLayout {
    /// Channels first (NCHW)
    NCHW = 0,
    /// Channels last (NHWC)
    NHWC = 1,
    /// OIHW for conv weights
    OIHW = 2,
    /// Default layout
    Default = 3,
}

impl Default for LightShipLayout {
    fn default() -> Self {
        LightShipLayout::Default
    }
}

/// LightShip backend types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum LightShipBackend {
    /// CPU backend
    CPU = 0,
    /// GPU backend
    GPU = 1,
    /// NPU backend
    NPU = 2,
    /// Vulkan backend
    Vulkan = 3,
    /// Metal backend (Apple)
    Metal = 4,
    /// DSP backend
    DSP = 5,
}

impl Default for LightShipBackend {
    fn default() -> Self {
        LightShipBackend::CPU
    }
}

/// LightShip inference mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum LightShipInferenceMode {
    /// Synchronous inference
    Synchronous = 0,
    /// Asynchronous inference
    Asynchronous = 1,
}

impl Default for LightShipInferenceMode {
    fn default() -> Self {
        LightShipInferenceMode::Synchronous
    }
}

/// LightShip log level
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum LightShipLogLevel {
    /// Trace level
    Trace = 0,
    /// Debug level
    Debug = 1,
    /// Info level
    Info = 2,
    /// Warning level
    Warn = 3,
    /// Error level
    Error = 4,
    /// No logging
    Off = 5,
}

impl Default for LightShipLogLevel {
    fn default() -> Self {
        LightShipLogLevel::Info
    }
}

impl fmt::Display for LightShipLogLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LightShipLogLevel::Trace => write!(f, "TRACE"),
            LightShipLogLevel::Debug => write!(f, "DEBUG"),
            LightShipLogLevel::Info => write!(f, "INFO"),
            LightShipLogLevel::Warn => write!(f, "WARN"),
            LightShipLogLevel::Error => write!(f, "ERROR"),
            LightShipLogLevel::Off => write!(f, "OFF"),
        }
    }
}

/// Tensor shape
#[derive(Debug, Clone)]
pub struct LightShipShape {
    /// Shape dimensions
    pub dims: Vec<usize>,
}

impl LightShipShape {
    /// Create a new shape
    pub fn new(dims: Vec<usize>) -> Self {
        Self { dims }
    }

    /// Get the number of dimensions
    pub fn num_dims(&self) -> usize {
        self.dims.len()
    }

    /// Get total element count
    pub fn element_count(&self) -> usize {
        self.dims.iter().product()
    }
}

/// Session configuration
#[derive(Debug, Clone)]
pub struct LightShipSessionConfig {
    /// Backend to use
    pub backend: LightShipBackend,
    /// Number of threads
    pub num_threads: usize,
    /// Enable SIMD
    pub use_simd: bool,
    /// Inference mode
    pub mode: LightShipInferenceMode,
    /// Enable profiling
    pub enable_profiling: bool,
}

impl Default for LightShipSessionConfig {
    fn default() -> Self {
        Self {
            backend: LightShipBackend::CPU,
            num_threads: 0,
            use_simd: true,
            mode: LightShipInferenceMode::Synchronous,
            enable_profiling: false,
        }
    }
}

/// Timing information (C-compatible)
#[derive(Debug, Clone)]
pub struct LightShipTiming {
    /// Total inference time in microseconds
    pub total_time_us: u64,
    /// Load time in microseconds
    pub load_time_us: u64,
    /// Compile time in microseconds
    pub compile_time_us: u64,
    /// Execution time in microseconds
    pub execution_time_us: u64,
}

impl Default for LightShipTiming {
    fn default() -> Self {
        Self {
            total_time_us: 0,
            load_time_us: 0,
            compile_time_us: 0,
            execution_time_us: 0,
        }
    }
}

/// Model metadata (C-compatible)
#[derive(Debug, Clone)]
pub struct LightShipModelMetadata {
    /// Model name
    pub name: *const std::ffi::c_char,
    /// Model version
    pub version: *const std::ffi::c_char,
    /// Input tensor count
    pub num_inputs: u32,
    /// Output tensor count
    pub num_outputs: u32,
    /// Whether the model is quantized
    pub is_quantized: bool,
}

impl Default for LightShipModelMetadata {
    fn default() -> Self {
        Self {
            name: std::ptr::null(),
            version: std::ptr::null(),
            num_inputs: 0,
            num_outputs: 0,
            is_quantized: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_type_default() {
        assert_eq!(LightShipDataType::default(), LightShipDataType::F32);
    }

    #[test]
    fn test_backend_default() {
        assert_eq!(LightShipBackend::default(), LightShipBackend::CPU);
    }

    #[test]
    fn test_shape() {
        let shape = LightShipShape::new(vec![1, 3, 224, 224]);
        assert_eq!(shape.num_dims(), 4);
        assert_eq!(shape.element_count(), 150528);
    }

    #[test]
    fn test_session_config_default() {
        let config = LightShipSessionConfig::default();
        assert_eq!(config.backend, LightShipBackend::CPU);
        assert_eq!(config.num_threads, 0);
    }

    #[test]
    fn test_timing_default() {
        let timing = LightShipTiming::default();
        assert_eq!(timing.total_time_us, 0);
    }
}
