//! Model loading and parsing module

pub mod error;
pub mod format;
pub mod loader;
pub mod metadata;
pub mod native;
pub mod onnx;

pub use error::{ModelError, ModelLoaderError, ValidationError};
pub use format::MagicNumber;
pub use loader::{ModelFile, ModelLoader, ModelLoaderRegistry, ValidationResult};
pub use metadata::ModelMetadata;
pub use native::NativeSerializer;
pub use onnx::{OnnxLoader, create_default_registry};
