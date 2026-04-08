//! Async handle for asynchronous inference
//!
//! Provides a handle for tracking asynchronous inference operations.

use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

/// Async inference status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AsyncStatus {
    /// Operation is pending
    Pending,
    /// Operation is running
    Running,
    /// Operation completed successfully
    Completed,
    /// Operation failed with an error
    Failed,
    /// Operation was cancelled
    Cancelled,
}

impl Default for AsyncStatus {
    fn default() -> Self {
        AsyncStatus::Pending
    }
}

/// Async inference result
#[derive(Debug, Clone)]
pub struct AsyncResult {
    /// Whether the operation completed
    pub completed: bool,
    /// Error message if failed
    pub error: Option<String>,
}

impl AsyncResult {
    /// Create a new async result
    pub fn new() -> Self {
        Self {
            completed: false,
            error: None,
        }
    }

    /// Mark as completed
    pub fn set_completed(&mut self) {
        self.completed = true;
    }

    /// Mark as failed with error
    pub fn set_failed(&mut self, error: String) {
        self.completed = true;
        self.error = Some(error);
    }
}

impl Default for AsyncResult {
    fn default() -> Self {
        Self::new()
    }
}

/// Handle for tracking asynchronous inference operations
///
/// Allows polling for completion and retrieving results.
#[derive(Debug)]
pub struct AsyncHandle {
    /// Current status
    status: AsyncStatus,
    /// Result data
    result: AsyncResult,
    /// Operation ID
    id: u64,
}

impl AsyncHandle {
    /// Create a new async handle
    pub fn new(id: u64) -> Self {
        Self {
            status: AsyncStatus::Pending,
            result: AsyncResult::new(),
            id,
        }
    }

    /// Get the operation ID
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Get the current status
    pub fn status(&self) -> AsyncStatus {
        self.status
    }

    /// Check if the operation is complete
    pub fn is_done(&self) -> bool {
        matches!(
            self.status,
            AsyncStatus::Completed | AsyncStatus::Failed | AsyncStatus::Cancelled
        )
    }

    /// Check if the operation is still pending
    pub fn is_pending(&self) -> bool {
        self.status == AsyncStatus::Pending
    }

    /// Mark as running
    pub fn set_running(&mut self) {
        self.status = AsyncStatus::Running;
    }

    /// Mark as completed
    pub fn set_completed(&mut self) {
        self.status = AsyncStatus::Completed;
        self.result.set_completed();
    }

    /// Mark as failed
    pub fn set_failed(&mut self, error: String) {
        self.status = AsyncStatus::Failed;
        self.result.set_failed(error);
    }

    /// Mark as cancelled
    pub fn set_cancelled(&mut self) {
        self.status = AsyncStatus::Cancelled;
    }

    /// Get the error message if failed
    pub fn error(&self) -> Option<&str> {
        self.result.error.as_deref()
    }

    /// Get the result
    pub fn result(&self) -> &AsyncResult {
        &self.result
    }

    /// Wait for completion (blocking)
    pub fn wait(self) -> AsyncResult {
        // In a real implementation, this would block until completion
        self.result
    }
}

impl Default for AsyncHandle {
    fn default() -> Self {
        Self::new(0)
    }
}

impl Future for AsyncHandle {
    type Output = AsyncResult;

    fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
        if self.is_done() {
            Poll::Ready(self.result.clone())
        } else {
            Poll::Pending
        }
    }
}

impl Clone for AsyncHandle {
    fn clone(&self) -> Self {
        Self {
            status: self.status,
            result: self.result.clone(),
            id: self.id,
        }
    }
}

impl PartialEq for AsyncHandle {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_async_handle_creation() {
        let handle = AsyncHandle::new(1);
        assert_eq!(handle.id(), 1);
        assert_eq!(handle.status(), AsyncStatus::Pending);
        assert!(!handle.is_done());
    }

    #[test]
    fn test_async_handle_status_transitions() {
        let mut handle = AsyncHandle::new(1);

        handle.set_running();
        assert_eq!(handle.status(), AsyncStatus::Running);

        handle.set_completed();
        assert_eq!(handle.status(), AsyncStatus::Completed);
        assert!(handle.is_done());
    }

    #[test]
    fn test_async_handle_failure() {
        let mut handle = AsyncHandle::new(1);
        handle.set_failed("Test error".to_string());

        assert_eq!(handle.status(), AsyncStatus::Failed);
        assert!(handle.is_done());
        assert_eq!(handle.error(), Some("Test error"));
    }

    #[test]
    fn test_async_handle_cancelled() {
        let mut handle = AsyncHandle::new(1);
        handle.set_cancelled();

        assert_eq!(handle.status(), AsyncStatus::Cancelled);
        assert!(handle.is_done());
    }

    #[test]
    fn test_async_result() {
        let mut result = AsyncResult::new();
        assert!(!result.completed);
        assert!(result.error.is_none());

        result.set_completed();
        assert!(result.completed);

        let mut result2 = AsyncResult::new();
        result2.set_failed("error".to_string());
        assert!(result2.completed);
        assert_eq!(result2.error, Some("error".to_string()));
    }

    #[test]
    fn test_async_status_default() {
        assert_eq!(AsyncStatus::default(), AsyncStatus::Pending);
    }
}
