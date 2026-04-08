//! Error types for LightShip

use thiserror::Error;

/// Main error type for LightShip operations
#[derive(Debug, Error)]
pub enum LightShipError {
    /// Model related errors
    #[error("Model error: {0}")]
    Model(#[from] ModelError),

    /// Backend related errors
    #[error("Backend error: {0}")]
    Backend(#[from] BackendError),

    /// Memory related errors
    #[error("Memory error: {0}")]
    Memory(#[from] MemoryError),

    /// Operator related errors
    #[error("Operator error: {0}")]
    Operator(#[from] OperatorError),

    /// Invalid parameter
    #[error("Invalid parameter: {0}")]
    InvalidParam(String),

    /// Unsupported operation
    #[error("Unsupported operation: {0}")]
    Unsupported(String),

    /// Timeout
    #[error("Timeout: {0}")]
    Timeout(String),

    /// Internal error
    #[error("Internal error: {0}")]
    Internal(String),
}

/// Result type alias for LightShip operations
pub type Result<T> = std::result::Result<T, LightShipError>;

/// Model related errors
#[derive(Debug, Error)]
pub enum ModelError {
    /// File not found
    #[error("File not found: {0}")]
    FileNotFound(String),

    /// Invalid format
    #[error("Invalid format: {0}")]
    InvalidFormat(String),

    /// Parse error with location
    #[error("Parse error at {location}: {message}")]
    ParseError {
        /// Location of the error
        location: String,
        /// Error message
        message: String,
    },

    /// Unsupported operator
    #[error("Unsupported operator: {0}")]
    UnsupportedOperator(String),

    /// Shape inference failed
    #[error("Shape inference failed: {0}")]
    ShapeInferenceFailed(String),

    /// Validation failed
    #[error("Validation failed: {0}")]
    ValidationFailed(String),
}

/// Backend related errors
#[derive(Debug, Error)]
pub enum BackendError {
    /// Backend not available
    #[error("Backend not available: {0}")]
    NotAvailable(String),

    /// Compilation failed
    #[error("Compilation failed: {0}")]
    CompilationFailed(String),

    /// Execution failed
    #[error("Execution failed: {0}")]
    ExecutionFailed(String),

    /// Unsupported data type
    #[error("Unsupported data type: {0:?}")]
    UnsupportedDataType(String),

    /// Unsupported layout
    #[error("Unsupported layout: {0:?}")]
    UnsupportedLayout(String),

    /// Out of memory
    #[error("Out of memory")]
    OutOfMemory,
}

/// Memory related errors
#[derive(Debug, Error)]
pub enum MemoryError {
    /// Allocation failed
    #[error("Allocation failed: {0}")]
    AllocationFailed(String),

    /// Out of memory limit
    #[error("Out of memory limit: tried to allocate {size} bytes, limit is {limit}")]
    OutOfMemoryLimit {
        /// Size of the attempted allocation
        size: usize,
        /// Memory limit
        limit: usize,
    },

    /// Invalid alignment
    #[error("Invalid alignment: {0}")]
    InvalidAlignment(usize),

    /// Memory access error
    #[error("Memory access error: {0}")]
    AccessError(String),
}

/// Operator related errors
#[derive(Debug, Error)]
pub enum OperatorError {
    /// Operator not found
    #[error("Operator not found: {0}")]
    NotFound(String),

    /// Shape mismatch
    #[error("Shape mismatch: {0}")]
    ShapeMismatch(String),

    /// Data type mismatch
    #[error("Data type mismatch: {0}")]
    DataTypeMismatch(String),

    /// Invalid attribute
    #[error("Invalid attribute {name}: {message}")]
    InvalidAttribute {
        /// Attribute name
        name: String,
        /// Error message
        message: String,
    },
}
