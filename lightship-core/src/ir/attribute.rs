//! Attribute definitions for LightShip IR

use std::collections::HashMap;
use std::fmt;

/// Attribute value types
#[derive(Debug, Clone)]
pub enum AttributeValue {
    /// Integer value
    Int(i64),
    /// Float value
    Float(f32),
    /// String value
    String(String),
    /// Integer list
    IntList(Vec<i64>),
    /// Float list
    FloatList(Vec<f32>),
    /// String list
    StringList(Vec<String>),
}

impl AttributeValue {
    /// Get integer value
    pub fn as_int(&self) -> Option<i64> {
        match self {
            AttributeValue::Int(v) => Some(*v),
            _ => None,
        }
    }

    /// Get float value
    pub fn as_float(&self) -> Option<f32> {
        match self {
            AttributeValue::Float(v) => Some(*v),
            _ => None,
        }
    }

    /// Get string value
    pub fn as_string(&self) -> Option<&String> {
        match self {
            AttributeValue::String(v) => Some(v),
            _ => None,
        }
    }

    /// Get int list
    pub fn as_int_list(&self) -> Option<&[i64]> {
        match self {
            AttributeValue::IntList(v) => Some(v),
            _ => None,
        }
    }

    /// Get float list
    pub fn as_float_list(&self) -> Option<&[f32]> {
        match self {
            AttributeValue::FloatList(v) => Some(v),
            _ => None,
        }
    }
}

/// Single attribute
#[derive(Debug, Clone)]
pub struct Attribute {
    /// Attribute name
    pub name: String,
    /// Attribute value
    pub value: AttributeValue,
    /// Documentation
    pub doc: Option<String>,
}

impl Attribute {
    /// Create a new integer attribute
    pub fn new_int(name: impl Into<String>, value: i64) -> Self {
        Self {
            name: name.into(),
            value: AttributeValue::Int(value),
            doc: None,
        }
    }

    /// Create a new float attribute
    pub fn new_float(name: impl Into<String>, value: f32) -> Self {
        Self {
            name: name.into(),
            value: AttributeValue::Float(value),
            doc: None,
        }
    }

    /// Create a new string attribute
    pub fn new_string(name: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            value: AttributeValue::String(value.into()),
            doc: None,
        }
    }

    /// Create a new int list attribute
    pub fn new_int_list(name: impl Into<String>, value: Vec<i64>) -> Self {
        Self {
            name: name.into(),
            value: AttributeValue::IntList(value),
            doc: None,
        }
    }

    /// Create a new float list attribute
    pub fn new_float_list(name: impl Into<String>, value: Vec<f32>) -> Self {
        Self {
            name: name.into(),
            value: AttributeValue::FloatList(value),
            doc: None,
        }
    }
}

/// Attribute map for storing operator attributes
#[derive(Debug, Clone, Default)]
pub struct AttributeMap {
    attrs: HashMap<String, Attribute>,
}

impl AttributeMap {
    /// Create a new empty attribute map
    pub fn new() -> Self {
        Self {
            attrs: HashMap::new(),
        }
    }

    /// Insert an attribute
    pub fn insert(&mut self, attr: Attribute) -> Option<Attribute> {
        self.attrs.insert(attr.name.clone(), attr)
    }

    /// Get an attribute by name
    pub fn get(&self, name: &str) -> Option<&Attribute> {
        self.attrs.get(name)
    }

    /// Check if contains an attribute
    pub fn contains(&self, name: &str) -> bool {
        self.attrs.contains_key(name)
    }

    /// Get an integer attribute
    pub fn get_int(&self, name: &str) -> Option<i64> {
        self.get(name).and_then(|a| a.value.as_int())
    }

    /// Get a float attribute
    pub fn get_float(&self, name: &str) -> Option<f32> {
        self.get(name).and_then(|a| a.value.as_float())
    }

    /// Get a string attribute
    pub fn get_string(&self, name: &str) -> Option<&String> {
        self.get(name).and_then(|a| a.value.as_string())
    }

    /// Get an int list attribute
    pub fn get_int_list(&self, name: &str) -> Option<&[i64]> {
        self.get(name).and_then(|a| a.value.as_int_list())
    }

    /// Get a float list attribute
    pub fn get_float_list(&self, name: &str) -> Option<&[f32]> {
        self.get(name).and_then(|a| a.value.as_float_list())
    }

    /// Iterate over attributes
    pub fn iter(&self) -> impl Iterator<Item = &Attribute> {
        self.attrs.values()
    }

    /// Number of attributes
    pub fn len(&self) -> usize {
        self.attrs.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.attrs.is_empty()
    }
}
