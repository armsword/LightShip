//! Model loading errors

use thiserror::Error;

/// Model related errors
#[derive(Debug, Error)]
pub enum ModelError {
    /// File not found
    #[error("File not found: {0}")]
    FileNotFound(String),

    /// Invalid format
    #[error("Invalid format: {0}")]
    InvalidFormat(String),

    /// Parse error
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

/// Model loader errors
#[derive(Debug, Error)]
pub enum ModelLoaderError {
    /// File not found
    #[error("File not found: {0}")]
    FileNotFound(String),

    /// Invalid format
    #[error("Invalid format: {0}")]
    InvalidFormat(String),

    /// Invalid magic number
    #[error("Invalid magic number: expected {expected}, got {got}")]
    InvalidMagicNumber {
        /// Expected magic bytes
        expected: String,
        /// Actual bytes received
        got: String,
    },

    /// Unsupported version
    #[error("Unsupported version: {0}")]
    UnsupportedVersion(String),

    /// Parse error
    #[error("Parse error at {location}: {message}")]
    ParseError {
        /// Location
        location: String,
        /// Message
        message: String,
    },

    /// Invalid operator
    #[error("Invalid operator: {0}")]
    InvalidOperator(String),

    /// Invalid tensor
    #[error("Invalid tensor: {0}")]
    InvalidTensor(String),

    /// Unsupported operator
    #[error("Unsupported operator: {0}")]
    UnsupportedOperator(String),

    /// Shape inference failed
    #[error("Shape inference failed: {0}")]
    ShapeInferenceFailed(String),

    /// Validation failed
    #[error("Validation failed: {0}")]
    ValidationFailed(String),

    /// IO error
    #[error("IO error: {0}")]
    IoError(String),
}

/// Validation error
#[derive(Debug, Clone)]
pub struct ValidationError {
    /// Error code
    pub code: ValidationErrorCode,
    /// Error message
    pub message: String,
    /// Related node name
    pub node_name: Option<String>,
    /// Related tensor name
    pub tensor_name: Option<String>,
}

/// Validation error code
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidationErrorCode {
    /// Missing required input
    MissingInput,
    /// Invalid tensor shape
    InvalidShape,
    /// Invalid data type
    InvalidDataType,
    /// Unsupported operator
    UnsupportedOperator,
    /// Cyclic dependency
    CyclicDependency,
    /// Other error
    Other,
}
