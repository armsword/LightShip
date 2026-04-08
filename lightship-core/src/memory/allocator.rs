//! Memory allocator trait

use crate::common::LightShipError;
use crate::memory::MemoryStats;
use std::fmt::Debug;

/// Memory allocation error
#[derive(Debug, Clone)]
pub enum MemoryAllocationError {
    /// Out of memory
    OutOfMemory {
        /// Requested size
        requested: usize,
        /// Memory limit
        limit: usize,
    },
    /// Invalid alignment
    InvalidAlignment {
        /// Requested alignment
        requested: usize,
    },
    /// Allocation failed
    AllocationFailed(String),
}

/// Memory layout for allocation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MemoryLayout {
    /// Size in bytes
    pub size: usize,
    /// Alignment in bytes (must be power of 2)
    pub alignment: usize,
}

impl MemoryLayout {
    /// Create a new memory layout
    pub fn new(size: usize, alignment: usize) -> Self {
        Self { size, alignment }
    }

    /// Create a layout with default alignment
    pub fn new_default_align(size: usize) -> Self {
        Self { size, alignment: 1 }
    }

    /// Check if alignment is valid (power of 2)
    pub fn is_valid_alignment(alignment: usize) -> bool {
        alignment > 0 && (alignment & (alignment - 1)) == 0
    }
}

/// Memory allocator trait
pub trait MemoryAllocator: Send + Sync {
    /// Allocate memory with the given layout
    fn allocate(&self, layout: MemoryLayout) -> Result<Allocation, MemoryAllocationError>;

    /// Deallocate memory
    fn deallocate(&self, allocation: Allocation);

    /// Get statistics
    fn stats(&self) -> MemoryStats;

    /// Reset statistics
    fn reset_stats(&self);
}

/// A memory allocation handle
#[derive(Debug, Clone)]
pub struct Allocation {
    /// Pointer to allocated memory
    pub ptr: std::ptr::NonNull<u8>,
    /// Size of the allocation
    pub size: usize,
    /// Alignment used
    pub alignment: usize,
}

impl Allocation {
    /// Create a new allocation
    pub fn new(ptr: std::ptr::NonNull<u8>, size: usize, alignment: usize) -> Self {
        Self {
            ptr,
            size,
            alignment,
        }
    }

    /// Get the pointer address
    pub fn address(&self) -> usize {
        self.ptr.as_ptr() as usize
    }

    /// Check if the allocation is aligned correctly
    pub fn is_aligned(&self) -> bool {
        self.address() % self.alignment == 0
    }
}

/// Simple bump allocator for testing
pub struct BumpAllocator {
    memory: Vec<u8>,
    offset: usize,
    stats: MemoryStats,
}

impl Debug for BumpAllocator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BumpAllocator")
            .field("capacity", &self.memory.len())
            .field("offset", &self.offset)
            .field("stats", &self.stats)
            .finish()
    }
}

impl BumpAllocator {
    /// Create a new bump allocator with the given capacity
    pub fn new(capacity: usize) -> Self {
        Self {
            memory: vec![0u8; capacity],
            offset: 0,
            stats: MemoryStats::new(),
        }
    }

    /// Allocate memory from the bump allocator
    pub fn alloc(&mut self, layout: MemoryLayout) -> Result<Allocation, MemoryAllocationError> {
        if !MemoryLayout::is_valid_alignment(layout.alignment) {
            return Err(MemoryAllocationError::InvalidAlignment {
                requested: layout.alignment,
            });
        }

        // Align the offset
        let aligned_offset = (self.offset + layout.alignment - 1) & !(layout.alignment - 1);

        if aligned_offset + layout.size > self.memory.len() {
            return Err(MemoryAllocationError::OutOfMemory {
                requested: layout.size,
                limit: self.memory.len() - aligned_offset,
            });
        }

        let ptr = unsafe {
            std::ptr::NonNull::new_unchecked(self.memory.as_mut_ptr().add(aligned_offset))
        };
        self.offset = aligned_offset + layout.size;

        self.stats.record_allocation(layout.size);

        Ok(Allocation::new(ptr, layout.size, layout.alignment))
    }

    /// Get the remaining capacity
    pub fn remaining(&self) -> usize {
        self.memory.len() - self.offset
    }

    /// Get statistics
    pub fn stats(&self) -> &MemoryStats {
        &self.stats
    }

    /// Reset the allocator
    pub fn reset(&mut self) {
        self.offset = 0;
        self.stats.reset();
    }
}
