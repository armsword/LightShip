//! Unit tests for Backend and Memory structures

use lightship_core::backend::{BackendCapabilities, BackendFeatureFlags, MemoryBlock, SimdFlags};
use lightship_core::common::{BackendType, DataType, StorageLayout};
use lightship_core::ir::OperatorType;
use std::ptr::NonNull;

#[test]
fn test_backend_capabilities_default() {
    let caps = BackendCapabilities::default();

    assert_eq!(caps.backend_type, BackendType::CPU);
    assert!(caps.supported_data_types.contains(&DataType::F32));
    assert!(caps.max_threads >= 1);
    assert_eq!(caps.memory_alignment, 64);
}

#[test]
fn test_backend_capabilities_supports() {
    let mut caps = BackendCapabilities::default();
    caps.supported_operators.push(OperatorType::Conv2d);
    caps.supported_operators.push(OperatorType::ReLU);
    caps.supported_data_types.push(DataType::I8);

    assert!(caps.supports_operator(OperatorType::Conv2d));
    assert!(caps.supports_operator(OperatorType::ReLU));
    assert!(!caps.supports_operator(OperatorType::SelfAttention));
    assert!(caps.supports_data_type(DataType::I8));
    assert!(!caps.supports_data_type(DataType::F64));
}

#[test]
fn test_backend_capabilities_supports_layout() {
    let caps = BackendCapabilities::default();

    assert!(caps.supports_layout(StorageLayout::NCHW));
    assert!(caps.supports_layout(StorageLayout::NHWC));
    assert!(!caps.supports_layout(StorageLayout::OIHW));
}

#[test]
fn test_simd_flags() {
    let mut flags = SimdFlags::default();
    assert!(!flags.avx);
    assert!(!flags.avx2);
    assert!(!flags.neon);

    flags.avx2 = true;
    flags.neon = true;
    assert!(flags.avx2);
    assert!(flags.neon);
}

#[test]
fn test_backend_feature_flags() {
    let flags = BackendFeatureFlags {
        fp16: true,
        int8: true,
        bf16: false,
        ..Default::default()
    };

    assert!(flags.fp16);
    assert!(flags.int8);
    assert!(!flags.bf16);
}

#[test]
fn test_memory_block_creation() {
    let mut data = [0u8; 64];
    // Safety: data is valid
    let ptr = unsafe { NonNull::new_unchecked(data.as_mut_ptr()) };
    let block = MemoryBlock::new(1, ptr, 64, 1, Default::default());

    assert_eq!(block.id, 1);
    assert_eq!(block.size, 64);
    assert_eq!(block.alignment, 1);
    assert!(block.is_aligned());
}

#[test]
fn test_memory_block_address() {
    let mut data = [0u8; 64];
    // Safety: data is valid
    let ptr = unsafe { NonNull::new_unchecked(data.as_mut_ptr()) };
    let block = MemoryBlock::new(0, ptr, 64, 1, Default::default());

    assert_eq!(block.address(), data.as_ptr() as usize);
}

#[test]
fn test_memory_block_display() {
    let mut data = [0u8; 128];
    // Safety: data is valid
    let ptr = unsafe { NonNull::new_unchecked(data.as_mut_ptr()) };
    let block = MemoryBlock::new(42, ptr, 128, 64, Default::default());

    let display = format!("{}", block);
    assert!(display.contains("42"));
    assert!(display.contains("128"));
}
