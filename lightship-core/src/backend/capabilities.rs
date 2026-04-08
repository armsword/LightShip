//! Backend capabilities definitions

use crate::common::{BackendType, DataType, StorageLayout};
use crate::ir::OperatorType;

/// SIMD flags for CPU backends
#[derive(Debug, Clone, Default)]
pub struct SimdFlags {
    /// AVX support
    pub avx: bool,
    /// AVX2 support
    pub avx2: bool,
    /// AVX512 support
    pub avx512: bool,
    /// NEON support (ARM)
    pub neon: bool,
    /// SVE support (ARM)
    pub sve: bool,
    /// Apple NEON support
    pub apple_neon: bool,
}

/// Backend feature flags
#[derive(Debug, Clone, Default)]
pub struct BackendFeatureFlags {
    /// FP16 support
    pub fp16: bool,
    /// BF16 support
    pub bf16: bool,
    /// INT8 support
    pub int8: bool,
    /// INT16 support
    pub int16: bool,
    /// Dot product support
    pub dot_product: bool,
    /// Matrix multiply accumulate support
    pub matrix_multiply_accumulate: bool,
    /// Unified memory support
    pub unified_memory: bool,
    /// Virtual memory support
    pub virtual_memory: bool,
}

/// Backend capabilities
#[derive(Debug, Clone)]
pub struct BackendCapabilities {
    /// Backend type
    pub backend_type: BackendType,
    /// Supported operators
    pub supported_operators: Vec<OperatorType>,
    /// Supported data types
    pub supported_data_types: Vec<DataType>,
    /// Supported storage layouts
    pub supported_layouts: Vec<StorageLayout>,
    /// Maximum threads supported
    pub max_threads: usize,
    /// Whether SIMD is available
    pub has_simd: bool,
    /// SIMD flags
    pub simd_flags: SimdFlags,
    /// Memory alignment requirement
    pub memory_alignment: usize,
    /// Register size in bits
    pub register_size: usize,
    /// Feature flags
    pub feature_flags: BackendFeatureFlags,
}

impl Default for BackendCapabilities {
    fn default() -> Self {
        Self {
            backend_type: BackendType::CPU,
            supported_operators: Vec::new(),
            supported_data_types: vec![DataType::F32, DataType::F16],
            supported_layouts: vec![StorageLayout::NCHW, StorageLayout::NHWC],
            max_threads: 1,
            has_simd: false,
            simd_flags: SimdFlags::default(),
            memory_alignment: 64,
            register_size: 64,
            feature_flags: BackendFeatureFlags::default(),
        }
    }
}

impl BackendCapabilities {
    /// Check if an operator is supported
    pub fn supports_operator(&self, op: OperatorType) -> bool {
        self.supported_operators.contains(&op)
    }

    /// Check if a data type is supported
    pub fn supports_data_type(&self, dt: DataType) -> bool {
        self.supported_data_types.contains(&dt)
    }

    /// Check if a storage layout is supported
    pub fn supports_layout(&self, layout: StorageLayout) -> bool {
        self.supported_layouts.contains(&layout)
    }
}
