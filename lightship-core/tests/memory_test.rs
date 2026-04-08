//! Unit tests for Memory management

use lightship_core::memory::{
    MemoryLayout, MemoryPool, MemoryPoolConfig, MemoryStats,
};
use lightship_core::memory::allocator::{Allocation, BumpAllocator, MemoryAllocationError};

#[test]
fn test_memory_layout() {
    let layout = MemoryLayout::new(1024, 64);
    assert_eq!(layout.size, 1024);
    assert_eq!(layout.alignment, 64);
    assert!(MemoryLayout::is_valid_alignment(64));
    assert!(!MemoryLayout::is_valid_alignment(3));
}

#[test]
fn test_memory_stats() {
    let stats = MemoryStats::new();
    assert_eq!(stats.current_allocated(), 0);
    assert_eq!(stats.peak_allocated(), 0);
    assert_eq!(stats.num_allocations(), 0);

    stats.record_allocation(100);
    assert_eq!(stats.current_allocated(), 100);
    assert_eq!(stats.peak_allocated(), 100);
    assert_eq!(stats.num_allocations(), 1);

    stats.record_allocation(200);
    assert_eq!(stats.current_allocated(), 300);
    assert_eq!(stats.peak_allocated(), 300);

    stats.record_deallocation(100);
    assert_eq!(stats.current_allocated(), 200);

    stats.reset();
    assert_eq!(stats.current_allocated(), 0);
}

#[test]
fn test_bump_allocator() {
    let mut allocator = BumpAllocator::new(1024);

    let layout = MemoryLayout::new(100, 1);
    let alloc = allocator.alloc(layout).unwrap();

    assert_eq!(alloc.size, 100);
    assert_eq!(alloc.alignment, 1);
    assert!(alloc.is_aligned());

    assert!(allocator.stats().num_allocations() >= 1);
}

#[test]
fn test_bump_allocator_alignment() {
    let mut allocator = BumpAllocator::new(1024);

    // Allocate with 64-byte alignment
    let layout = MemoryLayout::new(100, 64);
    let alloc = allocator.alloc(layout).unwrap();

    assert!(alloc.is_aligned());
}

#[test]
fn test_bump_allocator_out_of_memory() {
    let mut allocator = BumpAllocator::new(100);

    let layout = MemoryLayout::new(100, 1);
    let result = allocator.alloc(layout);
    assert!(result.is_ok());

    // Next allocation should fail
    let layout2 = MemoryLayout::new(100, 1);
    let result2 = allocator.alloc(layout2);
    assert!(result2.is_err());
}

#[test]
fn test_bump_allocator_invalid_alignment() {
    let mut allocator = BumpAllocator::new(1024);

    let layout = MemoryLayout::new(100, 3); // 3 is not power of 2
    let result = allocator.alloc(layout);

    assert!(result.is_err());
}

#[test]
fn test_bump_allocator_reset() {
    let mut allocator = BumpAllocator::new(1024);

    let layout = MemoryLayout::new(100, 1);
    allocator.alloc(layout).unwrap();

    assert_eq!(allocator.remaining(), 1024 - 100);

    allocator.reset();
    assert_eq!(allocator.remaining(), 1024);
}

#[test]
fn test_memory_pool_config_default() {
    let config = MemoryPoolConfig::default();
    assert_eq!(config.initial_capacity, 64 * 1024);
    assert!(config.max_capacity.is_none());
    assert_eq!(config.min_allocation_size, 64);
    assert!(config.reuse_enabled);
}

#[test]
fn test_memory_pool_creation() {
    let pool = MemoryPool::new();
    assert_eq!(pool.current_usage(), 0);
    assert_eq!(pool.peak_usage(), 0);
}

#[test]
fn test_memory_pool_with_config() {
    let config = MemoryPoolConfig {
        initial_capacity: 32 * 1024,
        max_capacity: Some(128 * 1024),
        min_allocation_size: 32,
        reuse_enabled: true,
    };
    let pool = MemoryPool::with_config(config);
    assert_eq!(pool.config().initial_capacity, 32 * 1024);
}
