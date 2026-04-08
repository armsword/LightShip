//! Model metadata

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Model metadata information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// Model name
    pub name: String,

    /// Model version
    pub version: String,

    /// Model author
    pub author: Option<String>,

    /// Model description
    pub description: Option<String>,

    /// Created timestamp (ISO 8601 string)
    pub created_at: Option<String>,

    /// Modified timestamp (ISO 8601 string)
    pub modified_at: Option<String>,

    /// License
    pub license: Option<String>,

    /// Custom metadata
    pub custom: HashMap<String, String>,
}

impl ModelMetadata {
    /// Create new empty metadata
    pub fn new(name: impl Into<String>) -> Self {
        let now = chrono::Utc::now().to_rfc3339();
        Self {
            name: name.into(),
            version: "1.0.0".to_string(),
            author: None,
            description: None,
            created_at: Some(now.clone()),
            modified_at: Some(now),
            license: None,
            custom: HashMap::new(),
        }
    }

    /// Set the version
    pub fn with_version(mut self, version: impl Into<String>) -> Self {
        self.version = version.into();
        self
    }

    /// Set the author
    pub fn with_author(mut self, author: impl Into<String>) -> Self {
        self.author = Some(author.into());
        self
    }

    /// Set the description
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Add custom metadata
    pub fn with_custom(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.custom.insert(key.into(), value.into());
        self
    }
}

impl Default for ModelMetadata {
    fn default() -> Self {
        Self::new("unnamed_model")
    }
}
