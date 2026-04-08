//! Model loading and parsing module

pub mod error;
pub mod format;
pub mod loader;
pub mod metadata;

pub use error::{ModelError, ModelLoaderError, ValidationError};
pub use format::MagicNumber;
pub use loader::{ModelFile, ModelLoader, ModelLoaderRegistry, ValidationResult};
pub use metadata::ModelMetadata;
