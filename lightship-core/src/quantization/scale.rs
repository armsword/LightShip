//! Scale encoding for quantized tensors

use std::fmt;

/// Scale encoding method
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScaleEncoding {
    /// Float32 scale
    Float32,
    /// Float16 scale
    Float16,
    /// Block-wise quantization with shared scale
    BlockWise {
        /// Block size
        block_size: usize,
    },
    /// Non-linear quantization using look-up table
    LookupTable {
        /// Number of entries in the LUT
        num_entries: usize,
    },
}

impl Default for ScaleEncoding {
    fn default() -> Self {
        ScaleEncoding::Float32
    }
}

impl fmt::Display for ScaleEncoding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ScaleEncoding::Float32 => write!(f, "Float32"),
            ScaleEncoding::Float16 => write!(f, "Float16"),
            ScaleEncoding::BlockWise { block_size } => {
                write!(f, "BlockWise[{}]", block_size)
            }
            ScaleEncoding::LookupTable { num_entries } => {
                write!(f, "LookupTable[{}]", num_entries)
            }
        }
    }
}
