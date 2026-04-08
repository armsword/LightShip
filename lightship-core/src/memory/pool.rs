//! Memory pool implementation

use crate::memory::allocator::{Allocation, MemoryAllocationError, MemoryAllocator, MemoryLayout};
use crate::memory::stats::MemoryStats;
use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::Arc;

/// Memory pool configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryPoolConfig {
    /// Initial pool capacity in bytes
    pub initial_capacity: usize,
    /// Maximum pool capacity in bytes (None = unlimited)
    pub max_capacity: Option<usize>,
    /// Minimum allocation size
    pub min_allocation_size: usize,
    /// Enable memory reuse
    pub reuse_enabled: bool,
}

impl Default for MemoryPoolConfig {
    fn default() -> Self {
        Self {
            initial_capacity: 64 * 1024, // 64KB
            max_capacity: None,
            min_allocation_size: 64,
            reuse_enabled: true,
        }
    }
}

/// Memory pool for efficient memory management
pub struct MemoryPool {
    config: MemoryPoolConfig,
    /// Small allocations pool (for common tensor sizes)
    small_allocations: HashMap<usize, Vec<Allocation>>,
    /// Large allocations pool
    large_allocations: Vec<Allocation>,
    /// Current total allocated size
    current_size: usize,
    /// Peak total allocated size
    peak_size: usize,
    /// Statistics
    stats: MemoryStats,
}

impl Debug for MemoryPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemoryPool")
            .field("config", &self.config)
            .field("num_small_slots", &self.small_allocations.len())
            .field("current_size", &self.current_size)
            .field("peak_size", &self.peak_size)
            .finish()
    }
}

impl MemoryPool {
    /// Create a new memory pool with default configuration
    pub fn new() -> Self {
        Self::with_config(MemoryPoolConfig::default())
    }

    /// Create a new memory pool with custom configuration
    pub fn with_config(config: MemoryPoolConfig) -> Self {
        Self {
            config,
            small_allocations: HashMap::new(),
            large_allocations: Vec::new(),
            current_size: 0,
            peak_size: 0,
            stats: MemoryStats::new(),
        }
    }

    /// Allocate memory from the pool
    pub fn allocate(&mut self, layout: MemoryLayout) -> Result<Allocation, MemoryAllocationError> {
        // Check limits
        if let Some(max) = self.config.max_capacity {
            if self.current_size + layout.size > max {
                return Err(MemoryAllocationError::OutOfMemory {
                    requested: layout.size,
                    limit: max - self.current_size,
                });
            }
        }

        // Try to reuse existing allocation
        if self.config.reuse_enabled {
            if let Some(cached) = self.small_allocations.get_mut(&layout.size) {
                if let Some(allocation) = cached.pop() {
                    self.stats.record_allocation(layout.size);
                    return Ok(allocation);
                }
            }
        }

        // Create new allocation
        // For now, just track stats - actual allocation would use system allocator
        self.current_size += layout.size;
        self.peak_size = self.peak_size.max(self.current_size);

        self.stats.record_allocation(layout.size);

        // Return a dummy allocation for now
        let ptr = unsafe { std::ptr::NonNull::new_unchecked(vec![0u8; layout.size].leak().as_mut_ptr()) };
        Ok(Allocation::new(ptr, layout.size, layout.alignment))
    }

    /// Return memory to the pool for reuse
    pub fn deallocate(&mut self, allocation: Allocation) {
        let size = allocation.size;
        if self.config.reuse_enabled && size >= self.config.min_allocation_size {
            self.small_allocations
                .entry(size)
                .or_default()
                .push(allocation);
            // Don't reduce current_size as memory is still allocated
        } else {
            // For non-reusable allocations, just track deallocation
            self.current_size = self.current_size.saturating_sub(size);
        }

        self.stats.record_deallocation(size);
    }

    /// Get current memory usage
    pub fn current_usage(&self) -> usize {
        self.current_size
    }

    /// Get peak memory usage
    pub fn peak_usage(&self) -> usize {
        self.peak_size
    }

    /// Get configuration
    pub fn config(&self) -> &MemoryPoolConfig {
        &self.config
    }

    /// Clear all cached allocations
    pub fn clear_cache(&mut self) {
        self.small_allocations.clear();
        self.large_allocations.clear();
    }

    /// Get statistics
    pub fn stats(&self) -> &MemoryStats {
        &self.stats
    }
}

impl Default for MemoryPool {
    fn default() -> Self {
        Self::new()
    }
}

/// Thread-safe wrapper for MemoryPool
pub struct SharedMemoryPool {
    inner: Arc<parking_lot::Mutex<MemoryPool>>,
}

impl Debug for SharedMemoryPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SharedMemoryPool").finish()
    }
}

impl SharedMemoryPool {
    /// Create a new shared memory pool
    pub fn new() -> Self {
        Self {
            inner: Arc::new(parking_lot::Mutex::new(MemoryPool::new())),
        }
    }

    /// Create with custom config
    pub fn with_config(config: MemoryPoolConfig) -> Self {
        Self {
            inner: Arc::new(parking_lot::Mutex::new(MemoryPool::with_config(config))),
        }
    }

    /// Allocate memory
    pub fn allocate(&self, layout: MemoryLayout) -> Result<Allocation, MemoryAllocationError> {
        self.inner.lock().allocate(layout)
    }

    /// Deallocate memory
    pub fn deallocate(&self, allocation: Allocation) {
        self.inner.lock().deallocate(allocation)
    }

    /// Get current usage
    pub fn current_usage(&self) -> usize {
        self.inner.lock().current_usage()
    }

    /// Get peak usage
    pub fn peak_usage(&self) -> usize {
        self.inner.lock().peak_usage()
    }
}

impl Default for SharedMemoryPool {
    fn default() -> Self {
        Self::new()
    }
}
