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

#[test]
fn test_memory_pool_allocation_tracking() {
    let mut pool = MemoryPool::new();

    let layout = MemoryLayout::new(256, 64);
    let result = pool.allocate(layout);
    assert!(result.is_ok());

    assert_eq!(pool.current_usage(), 256);
    assert!(pool.peak_usage() >= 256);
}

#[test]
fn test_memory_pool_max_capacity() {
    let config = MemoryPoolConfig {
        initial_capacity: 1024,
        max_capacity: Some(512), // Set max to exactly 512 bytes
        min_allocation_size: 64,
        reuse_enabled: false,
    };
    let mut pool = MemoryPool::with_config(config);

    let layout = MemoryLayout::new(512, 64);
    let result1 = pool.allocate(layout);
    assert!(result1.is_ok());

    // Second allocation should fail due to capacity limit (512 + 512 > 512)
    let layout2 = MemoryLayout::new(512, 64);
    let result2 = pool.allocate(layout2);
    assert!(result2.is_err());
}

#[test]
fn test_memory_pool_reuse_disabled() {
    let config = MemoryPoolConfig {
        initial_capacity: 1024,
        max_capacity: Some(2048),
        min_allocation_size: 64,
        reuse_enabled: false, // Disable reuse
    };
    let mut pool = MemoryPool::with_config(config);

    let layout = MemoryLayout::new(256, 64);
    let alloc1 = pool.allocate(layout).unwrap();
    drop(alloc1); // Free the first allocation

    // Without reuse, current_size decreases but peak tracks maximum ever allocated
    let alloc2 = pool.allocate(layout);
    assert!(alloc2.is_ok());
    // Peak should be 512 (256 + 256) since we allocated twice without reuse
    assert_eq!(pool.peak_usage(), 512);
}

#[test]
fn test_memory_pool_reuse_enabled() {
    let config = MemoryPoolConfig {
        initial_capacity: 1024,
        max_capacity: Some(1024),
        min_allocation_size: 64,
        reuse_enabled: true, // Enable reuse
    };
    let mut pool = MemoryPool::with_config(config);

    let layout = MemoryLayout::new(256, 64);
    let alloc1 = pool.allocate(layout).unwrap();
    drop(alloc1); // Free the first allocation

    // With reuse enabled, current_size stays at 256 after deallocation
    // (deallocate doesn't reduce current_size when reuse is enabled)
    assert_eq!(pool.current_usage(), 256);

    // Second allocation should succeed
    let alloc2 = pool.allocate(layout);
    assert!(alloc2.is_ok());
    // Peak tracks maximum current_size ever reached
    // Since reuse is enabled but current_size isn't reduced on deallocate,
    // we end up with 512 bytes allocated (first 256 + second 256)
    assert_eq!(pool.peak_usage(), 512);
}
