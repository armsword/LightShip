//! C API for LightShip
//!
//! Provides C-compatible API for integration with other languages
//! and platforms (Python, Android NDK, iOS, etc.).

mod types;
mod handle;
mod error;

pub use types::*;
pub use handle::*;
pub use error::*;
