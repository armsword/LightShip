//! Model loader trait and registry

use crate::common::{LightShipError, ModelFormat};
use crate::ir::Graph;
use crate::model::{error::ModelLoaderError, metadata::ModelMetadata};
use std::collections::HashMap;
use std::fmt::Debug;
use std::path::Path;
use std::sync::Arc;
use std::result::Result as StdResult;

/// Model file representation
#[derive(Debug, Clone)]
pub struct ModelFile {
    /// Model format
    pub format: ModelFormat,
    /// Model metadata
    pub metadata: ModelMetadata,
    /// IR version
    pub ir_version: u32,
    /// Compute graph
    pub graph: Graph,
    /// Extra data from the original format
    pub extra_data: HashMap<String, Vec<u8>>,
}

impl ModelFile {
    /// Create a new model file
    pub fn new(format: ModelFormat, graph: Graph) -> Self {
        Self {
            format,
            metadata: ModelMetadata::default(),
            ir_version: 1,
            graph,
            extra_data: HashMap::new(),
        }
    }

    /// Get the number of operators in the model
    pub fn num_operators(&self) -> usize {
        self.graph.nodes.len()
    }

    /// Get the number of parameters (static tensors)
    pub fn num_parameters(&self) -> usize {
        self.graph.variables.len()
    }
}

/// Model loader trait - implemented by format-specific loaders
pub trait ModelLoader: Send + Sync {
    /// Get the formats this loader supports
    fn supported_formats(&self) -> Vec<ModelFormat>;

    /// Check if this loader can handle the given format
    fn can_load(&self, format: ModelFormat) -> bool {
        self.supported_formats().contains(&format)
    }

    /// Load a model from a file path
    fn load_from_file(&self, path: &Path) -> StdResult<ModelFile, ModelLoaderError>;

    /// Load a model from bytes
    fn load_from_bytes(&self, bytes: &[u8], format: ModelFormat) -> StdResult<ModelFile, ModelLoaderError>;

    /// Validate a model file
    fn validate(&self, model: &ModelFile) -> StdResult<ValidationResult, ModelLoaderError>;

    /// Get the loader name
    fn name(&self) -> &'static str;
}

/// Validation result
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Whether the model is valid
    pub valid: bool,
    /// Validation errors
    pub errors: Vec<crate::model::error::ValidationError>,
    /// Validation warnings
    pub warnings: Vec<ValidationWarning>,
}

/// Validation warning
#[derive(Debug, Clone)]
pub struct ValidationWarning {
    /// Warning message
    pub message: String,
    /// Related node name
    pub node_name: Option<String>,
}

/// Model loader registry
pub struct ModelLoaderRegistry {
    loaders: HashMap<ModelFormat, Arc<dyn ModelLoader>>,
    format_detectors: Vec<Arc<dyn FormatDetector>>,
}

impl Debug for ModelLoaderRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelLoaderRegistry")
            .field("num_loaders", &self.loaders.len())
            .field("num_detectors", &self.format_detectors.len())
            .finish()
    }
}

impl ModelLoaderRegistry {
    /// Create a new empty registry
    pub fn new() -> Self {
        Self {
            loaders: HashMap::new(),
            format_detectors: Vec::new(),
        }
    }

    /// Register a model loader
    pub fn register<L: ModelLoader + 'static>(&mut self, loader: L) -> &mut Self {
        let formats = loader.supported_formats();
        let arc_loader = Arc::new(loader);
        for format in formats {
            self.loaders.insert(format, arc_loader.clone());
        }
        self
    }

    /// Register a format detector
    pub fn register_detector<D: FormatDetector + 'static>(&mut self, detector: D) -> &mut Self {
        self.format_detectors.push(Arc::new(detector));
        self
    }

    /// Get a loader for a specific format
    pub fn get(&self, format: ModelFormat) -> Option<&dyn ModelLoader> {
        self.loaders.get(&format).map(|l| l.as_ref())
    }

    /// Detect the format of a model file
    pub fn detect_format(&self, bytes: &[u8]) -> Option<ModelFormat> {
        // Try magic number detection first
        if let Some(format) = crate::model::format::MagicNumber::detect(bytes) {
            return Some(format);
        }

        // Try registered detectors
        for detector in &self.format_detectors {
            if let Some(format) = detector.detect(bytes) {
                return Some(format);
            }
        }

        None
    }

    /// Load a model, auto-detecting the format
    pub fn load(&self, path: &Path) -> StdResult<ModelFile, LightShipError> {
        let bytes = std::fs::read(path).map_err(|e| {
            LightShipError::Model(crate::common::error::ModelError::FileNotFound(format!("Failed to read file: {}", e)))
        })?;

        let format = self
            .detect_format(&bytes)
            .ok_or_else(|| LightShipError::Model(crate::common::error::ModelError::InvalidFormat("Cannot detect model format".into())))?;

        self.load_from_bytes(&bytes, format)
    }

    /// Load a model from bytes with a specific format
    pub fn load_from_bytes(&self, bytes: &[u8], format: ModelFormat) -> StdResult<ModelFile, LightShipError> {
        let loader = self
            .get(format)
            .ok_or_else(|| LightShipError::Model(crate::common::error::ModelError::InvalidFormat(
                format!("No loader registered for format: {:?}", format),
            )))?;

        loader
            .load_from_bytes(bytes, format)
            .map_err(|e| LightShipError::Model(crate::common::error::ModelError::InvalidFormat(e.to_string())))
    }
}

impl Default for ModelLoaderRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Format detector trait
pub trait FormatDetector: Send + Sync {
    /// Detect the format from bytes
    fn detect(&self, bytes: &[u8]) -> Option<ModelFormat>;
}

/// Simple ONNX format detector
pub struct OnnxFormatDetector;

impl Debug for OnnxFormatDetector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OnnxFormatDetector").finish()
    }
}

impl FormatDetector for OnnxFormatDetector {
    fn detect(&self, bytes: &[u8]) -> Option<ModelFormat> {
        // ONNX files are ZIP archives starting with PK
        if bytes.len() >= 2 && bytes[0] == 0x50 && bytes[1] == 0x4B {
            return Some(ModelFormat::ONNX);
        }

        // Or protobuf with ONNX magic
        if bytes.len() >= 4 && bytes[0] == 0x08 {
            return Some(ModelFormat::ONNX);
        }

        None
    }
}
