//! Memory statistics

use std::sync::atomic::{AtomicUsize, Ordering};

/// Memory statistics for tracking allocations
#[derive(Debug)]
pub struct MemoryStats {
    /// Total allocated bytes
    allocated_bytes: AtomicUsize,
    /// Peak allocated bytes
    peak_bytes: AtomicUsize,
    /// Number of allocations
    allocation_count: AtomicUsize,
    /// Number of deallocations
    deallocation_count: AtomicUsize,
    /// Largest single allocation
    largest_allocation: AtomicUsize,
}

impl MemoryStats {
    /// Create new memory stats
    pub fn new() -> Self {
        Self {
            allocated_bytes: AtomicUsize::new(0),
            peak_bytes: AtomicUsize::new(0),
            allocation_count: AtomicUsize::new(0),
            deallocation_count: AtomicUsize::new(0),
            largest_allocation: AtomicUsize::new(0),
        }
    }

    /// Record an allocation
    pub fn record_allocation(&self, size: usize) {
        let prev = self.allocated_bytes.fetch_add(size, Ordering::Relaxed);
        let new_total = prev + size;

        // Update peak
        let mut current_peak = self.peak_bytes.load(Ordering::Relaxed);
        while new_total > current_peak {
            match self.peak_bytes.compare_exchange(
                current_peak,
                new_total,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => current_peak = actual,
            }
        }

        // Update largest
        let mut current_largest = self.largest_allocation.load(Ordering::Relaxed);
        while size > current_largest {
            match self.largest_allocation.compare_exchange(
                current_largest,
                size,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => current_largest = actual,
            }
        }

        self.allocation_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a deallocation
    pub fn record_deallocation(&self, size: usize) {
        self.allocated_bytes.fetch_sub(size, Ordering::Relaxed);
        self.deallocation_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Get current allocated bytes
    pub fn current_allocated(&self) -> usize {
        self.allocated_bytes.load(Ordering::Relaxed)
    }

    /// Get peak allocated bytes
    pub fn peak_allocated(&self) -> usize {
        self.peak_bytes.load(Ordering::Relaxed)
    }

    /// Get allocation count
    pub fn num_allocations(&self) -> usize {
        self.allocation_count.load(Ordering::Relaxed)
    }

    /// Get deallocation count
    pub fn num_deallocations(&self) -> usize {
        self.deallocation_count.load(Ordering::Relaxed)
    }

    /// Get largest allocation
    pub fn largest_allocation(&self) -> usize {
        self.largest_allocation.load(Ordering::Relaxed)
    }

    /// Reset all statistics
    pub fn reset(&self) {
        self.allocated_bytes.store(0, Ordering::Relaxed);
        self.peak_bytes.store(0, Ordering::Relaxed);
        self.allocation_count.store(0, Ordering::Relaxed);
        self.deallocation_count.store(0, Ordering::Relaxed);
        self.largest_allocation.store(0, Ordering::Relaxed);
    }
}

impl Default for MemoryStats {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for MemoryStats {
    fn clone(&self) -> Self {
        Self {
            allocated_bytes: AtomicUsize::new(self.allocated_bytes.load(Ordering::Relaxed)),
            peak_bytes: AtomicUsize::new(self.peak_bytes.load(Ordering::Relaxed)),
            allocation_count: AtomicUsize::new(self.allocation_count.load(Ordering::Relaxed)),
            deallocation_count: AtomicUsize::new(self.deallocation_count.load(Ordering::Relaxed)),
            largest_allocation: AtomicUsize::new(self.largest_allocation.load(Ordering::Relaxed)),
        }
    }
}
