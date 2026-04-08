//! Memory block for backend allocation

use std::fmt;
use std::num::NonZeroUsize;
use std::ptr::NonNull;

/// Storage location
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StorageLocation {
    /// Heap memory
    Heap,
    /// Stack memory
    Stack,
    /// Shared memory
    Shared,
    /// GPU memory
    GPU,
    /// NPU memory
    NPU,
    /// Memory mapped file
    MemoryMap,
}

impl Default for StorageLocation {
    fn default() -> Self {
        StorageLocation::Heap
    }
}

/// Memory block for backend allocation
#[derive(Debug, Clone)]
pub struct MemoryBlock {
    /// Block ID
    pub id: u64,
    /// Pointer to the memory
    pub ptr: NonNull<u8>,
    /// Size in bytes
    pub size: usize,
    /// Alignment in bytes
    pub alignment: usize,
    /// Storage location
    pub location: StorageLocation,
}

impl MemoryBlock {
    /// Create a new memory block
    pub fn new(
        id: u64,
        ptr: NonNull<u8>,
        size: usize,
        alignment: usize,
        location: StorageLocation,
    ) -> Self {
        Self {
            id,
            ptr,
            size,
            alignment,
            location,
        }
    }

    /// Get the pointer as a slice
    pub fn as_slice(&self) -> &[u8] {
        // Safety: The block is guaranteed to be valid and allocated
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.size) }
    }

    /// Get the mutable pointer as a slice
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // Safety: The block is guaranteed to be valid and allocated
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.size) }
    }

    /// Get the pointer address
    pub fn address(&self) -> usize {
        self.ptr.as_ptr() as usize
    }

    /// Check if the alignment is valid
    pub fn is_aligned(&self) -> bool {
        self.address() % self.alignment == 0
    }
}

impl fmt::Display for MemoryBlock {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MemoryBlock(id={}, addr={:#x}, size={}, align={}, loc={:?})",
            self.id, self.address(), self.size, self.alignment, self.location
        )
    }
}
