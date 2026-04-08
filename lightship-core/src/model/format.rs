//! Model format definitions

use crate::common::ModelFormat;

/// Magic numbers for format detection
#[derive(Debug, Clone)]
pub struct MagicNumber {
    /// Expected magic bytes
    pub bytes: Vec<u8>,
    /// Format name
    pub format: ModelFormat,
}

impl MagicNumber {
    /// Detect format from magic number
    pub fn detect(data: &[u8]) -> Option<ModelFormat> {
        // Check for ONNX (PK\x03\x04 is ZIP-based)
        if data.len() >= 4 && data[0] == 0x08 && data[1] == 0x00 {
            return Some(ModelFormat::ONNX);
        }

        // Check for protobuf starting with specific ONNX bytes
        if data.len() >= 4 && data[0] == 0x08 && (data[1] & 0x10) == 0 {
            return Some(ModelFormat::ONNX);
        }

        // Check for LightShip native format
        if data.len() >= 9 {
            let prefix = std::str::from_utf8(&data[0..9]).unwrap_or("");
            if prefix.starts_with("LIGHTSHIP") {
                return Some(ModelFormat::Native);
            }
        }

        None
    }
}

impl ModelFormat {
    /// Get file extensions for this format
    pub fn extensions(&self) -> Vec<&'static str> {
        match self {
            ModelFormat::Native => vec!["lsmodel", "lightship"],
            ModelFormat::ONNX => vec!["onnx"],
            ModelFormat::TensorFlow => vec!["pb", "savedmodel"],
            ModelFormat::TFLite => vec!["tflite"],
            ModelFormat::Caffe => vec!["caffemodel", "prototxt"],
        }
    }

    /// Check if this format is protobuf-based
    pub fn is_protobuf(&self) -> bool {
        matches!(
            self,
            ModelFormat::ONNX | ModelFormat::TensorFlow | ModelFormat::TFLite | ModelFormat::Caffe
        )
    }
}
