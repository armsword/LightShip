//! C API error types
//!
//! Error codes and handling for the C API.

use std::fmt;

/// LightShip error codes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum LightShipErrorCode {
    /// Success
    Success = 0,
    /// Unknown error
    Unknown = 1,
    /// Invalid argument
    InvalidArgument = 2,
    /// Invalid handle
    InvalidHandle = 3,
    /// Model not found
    ModelNotFound = 4,
    /// Model parse error
    ModelParseError = 5,
    /// Backend not available
    BackendNotAvailable = 6,
    /// Backend create error
    BackendCreateError = 7,
    /// Operator not supported
    OperatorNotSupported = 8,
    /// Compilation error
    CompilationError = 9,
    /// Execution error
    ExecutionError = 10,
    /// Out of memory
    OutOfMemory = 11,
    /// Tensor not found
    TensorNotFound = 12,
    /// Dimension mismatch
    DimensionMismatch = 13,
    /// Data type mismatch
    DataTypeMismatch = 14,
    /// Session not ready
    SessionNotReady = 15,
    /// Async in progress
    AsyncInProgress = 16,
    /// Cancellation
    Cancelled = 17,
}

impl Default for LightShipErrorCode {
    fn default() -> Self {
        LightShipErrorCode::Success
    }
}

impl fmt::Display for LightShipErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LightShipErrorCode::Success => write!(f, "Success"),
            LightShipErrorCode::Unknown => write!(f, "Unknown error"),
            LightShipErrorCode::InvalidArgument => write!(f, "Invalid argument"),
            LightShipErrorCode::InvalidHandle => write!(f, "Invalid handle"),
            LightShipErrorCode::ModelNotFound => write!(f, "Model not found"),
            LightShipErrorCode::ModelParseError => write!(f, "Model parse error"),
            LightShipErrorCode::BackendNotAvailable => write!(f, "Backend not available"),
            LightShipErrorCode::BackendCreateError => write!(f, "Backend create error"),
            LightShipErrorCode::OperatorNotSupported => write!(f, "Operator not supported"),
            LightShipErrorCode::CompilationError => write!(f, "Compilation error"),
            LightShipErrorCode::ExecutionError => write!(f, "Execution error"),
            LightShipErrorCode::OutOfMemory => write!(f, "Out of memory"),
            LightShipErrorCode::TensorNotFound => write!(f, "Tensor not found"),
            LightShipErrorCode::DimensionMismatch => write!(f, "Dimension mismatch"),
            LightShipErrorCode::DataTypeMismatch => write!(f, "Data type mismatch"),
            LightShipErrorCode::SessionNotReady => write!(f, "Session not ready"),
            LightShipErrorCode::AsyncInProgress => write!(f, "Async in progress"),
            LightShipErrorCode::Cancelled => write!(f, "Cancelled"),
        }
    }
}

impl LightShipErrorCode {
    /// Check if this is a success code
    pub fn is_success(&self) -> bool {
        matches!(self, LightShipErrorCode::Success)
    }

    /// Check if this is an error code
    pub fn is_error(&self) -> bool {
        !self.is_success()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_code_default() {
        assert_eq!(LightShipErrorCode::default(), LightShipErrorCode::Success);
    }

    #[test]
    fn test_error_code_is_success() {
        assert!(LightShipErrorCode::Success.is_success());
        assert!(!LightShipErrorCode::Unknown.is_success());
    }

    #[test]
    fn test_error_code_display() {
        assert_eq!(format!("{}", LightShipErrorCode::Success), "Success");
        assert_eq!(format!("{}", LightShipErrorCode::OutOfMemory), "Out of memory");
    }
}
