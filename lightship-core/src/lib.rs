//! LightShip Core Library
//!
//! A lightweight edge-side neural network inference engine.

#![doc(html_root_url = "https://docs.rs/lightship-core")]
#![deny(missing_debug_implementations)]
#![deny(missing_docs)]
#![allow(unused_imports)]

pub mod common;
pub mod ir;
pub mod api;
pub mod backend;
pub mod model;
pub mod memory;
pub mod executor;
pub mod quantization;

pub use common::{error::LightShipError, types::*};
