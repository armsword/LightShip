//! Profiling support for inference
//!
//! Provides detailed performance profiling information.

use super::timing::TimingInfo;
use std::fmt;

/// Profiling level
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProfilingLevel {
    /// No profiling
    Off,
    /// Basic profiling (total time only)
    Basic,
    /// Operator-level profiling
    Operators,
    /// Detailed profiling (memory, cache, etc.)
    Detailed,
    /// Maximum profiling
    Max,
}

impl Default for ProfilingLevel {
    fn default() -> Self {
        ProfilingLevel::Off
    }
}

impl ProfilingLevel {
    /// Check if operator timing should be collected
    pub fn includes_operators(&self) -> bool {
        matches!(
            self,
            ProfilingLevel::Operators | ProfilingLevel::Detailed | ProfilingLevel::Max
        )
    }

    /// Check if memory profiling should be collected
    pub fn includes_memory(&self) -> bool {
        matches!(self, ProfilingLevel::Detailed | ProfilingLevel::Max)
    }

    /// Check if cache profiling should be collected
    pub fn includes_cache(&self) -> bool {
        matches!(self, ProfilingLevel::Max)
    }
}

/// Memory profiling information
#[derive(Debug, Clone)]
pub struct MemoryProfile {
    /// Peak memory usage in bytes
    pub peak_bytes: usize,
    /// Current memory usage in bytes
    pub current_bytes: usize,
    /// Number of allocations
    pub allocation_count: u64,
    /// Number of deallocations
    pub deallocation_count: u64,
    /// Memory allocation time
    pub allocation_time_ns: u64,
}

impl MemoryProfile {
    /// Create a new empty memory profile
    pub fn new() -> Self {
        Self {
            peak_bytes: 0,
            current_bytes: 0,
            allocation_count: 0,
            deallocation_count: 0,
            allocation_time_ns: 0,
        }
    }

    /// Get allocation count
    pub fn net_allocations(&self) -> i64 {
        self.allocation_count as i64 - self.deallocation_count as i64
    }
}

impl Default for MemoryProfile {
    fn default() -> Self {
        Self::new()
    }
}

/// Cache profiling information
#[derive(Debug, Clone)]
pub struct CacheProfile {
    /// L1 cache hits
    pub l1_hits: u64,
    /// L1 cache misses
    pub l1_misses: u64,
    /// L2 cache hits
    pub l2_hits: u64,
    /// L2 cache misses
    pub l2_misses: u64,
    /// L3 cache hits
    pub l3_hits: u64,
    /// L3 cache misses
    pub l3_misses: u64,
}

impl CacheProfile {
    /// Create a new empty cache profile
    pub fn new() -> Self {
        Self {
            l1_hits: 0,
            l1_misses: 0,
            l2_hits: 0,
            l2_misses: 0,
            l3_hits: 0,
            l3_misses: 0,
        }
    }

    /// Get L1 hit rate
    pub fn l1_hit_rate(&self) -> f64 {
        let total = self.l1_hits + self.l1_misses;
        if total > 0 {
            self.l1_hits as f64 / total as f64
        } else {
            0.0
        }
    }

    /// Get overall cache hit rate
    pub fn overall_hit_rate(&self) -> f64 {
        let total_hits = self.l1_hits + self.l2_hits + self.l3_hits;
        let total_misses = self.l1_misses + self.l2_misses + self.l3_misses;
        let total = total_hits + total_misses;
        if total > 0 {
            total_hits as f64 / total as f64
        } else {
            0.0
        }
    }
}

impl Default for CacheProfile {
    fn default() -> Self {
        Self::new()
    }
}

/// Complete profiling information
#[derive(Debug, Clone)]
pub struct ProfilingInfo {
    /// Profiling level
    pub level: ProfilingLevel,
    /// Timing information
    pub timing: TimingInfo,
    /// Memory profiling
    pub memory: Option<MemoryProfile>,
    /// Cache profiling
    pub cache: Option<CacheProfile>,
    /// Number of operators executed
    pub operator_count: u32,
    /// Number of nodes in the execution plan
    pub node_count: u32,
    /// Backend used
    pub backend: String,
}

impl ProfilingInfo {
    /// Create a new profiling info
    pub fn new(level: ProfilingLevel, backend: String) -> Self {
        Self {
            level,
            timing: TimingInfo::new(),
            memory: level.includes_memory().then_some(MemoryProfile::new()),
            cache: level.includes_cache().then_some(CacheProfile::new()),
            operator_count: 0,
            node_count: 0,
            backend,
        }
    }

    /// Create with timing info
    pub fn with_timing(mut self, timing: TimingInfo) -> Self {
        self.timing = timing;
        self
    }

    /// Check if profiling is enabled
    pub fn is_enabled(&self) -> bool {
        self.level != ProfilingLevel::Off
    }

    /// Get throughput (inferences per second)
    pub fn throughput(&self) -> f64 {
        let time_secs = self.timing.total_time.as_secs_f64();
        if time_secs > 0.0 {
            1.0 / time_secs
        } else {
            0.0
        }
    }
}

impl Default for ProfilingInfo {
    fn default() -> Self {
        Self::new(ProfilingLevel::Off, "CPU".to_string())
    }
}

impl fmt::Display for ProfilingInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ProfilingInfo(level={:?}, ", self.level)?;
        write!(
            f,
            "time={:.3}ms, throughput={:.2} inf/s",
            self.timing.total_time.as_secs_f64() * 1000.0,
            self.throughput()
        )?;
        if let Some(mem) = &self.memory {
            write!(
                f,
                ", peak_mem={:.2}MB",
                mem.peak_bytes as f64 / (1024.0 * 1024.0)
            )?;
        }
        if let Some(cache) = &self.cache {
            write!(f, ", cache_hit_rate={:.2}", cache.overall_hit_rate())?;
        }
        write!(f, ")")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_profiling_level_includes() {
        assert!(!ProfilingLevel::Off.includes_operators());
        assert!(!ProfilingLevel::Basic.includes_operators());
        assert!(ProfilingLevel::Operators.includes_operators());
        assert!(!ProfilingLevel::Off.includes_memory());
        assert!(ProfilingLevel::Detailed.includes_memory());
    }

    #[test]
    fn test_memory_profile() {
        let profile = MemoryProfile::new();
        assert_eq!(profile.net_allocations(), 0);

        let mut profile = MemoryProfile::new();
        profile.allocation_count = 100;
        profile.deallocation_count = 90;
        assert_eq!(profile.net_allocations(), 10);
    }

    #[test]
    fn test_cache_profile() {
        let profile = CacheProfile::new();
        assert_eq!(profile.l1_hit_rate(), 0.0);

        let mut profile = CacheProfile::new();
        profile.l1_hits = 80;
        profile.l1_misses = 20;
        assert!((profile.l1_hit_rate() - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_profiling_info_creation() {
        let info = ProfilingInfo::new(ProfilingLevel::Operators, "CPU".to_string());
        assert!(info.is_enabled());
        assert!(info.memory.is_none()); // Operators doesn't include memory
        assert!(info.cache.is_none());
    }

    #[test]
    fn test_profiling_info_throughput() {
        let info = ProfilingInfo::new(ProfilingLevel::Basic, "CPU".to_string())
            .with_timing(TimingInfo::from_execution(Duration::from_millis(10)));

        // 1 / 0.01 = 100 inf/s
        assert!((info.throughput() - 100.0).abs() < 0.01);
    }

    #[test]
    fn test_profiling_info_display() {
        let info = ProfilingInfo::new(ProfilingLevel::Basic, "CPU".to_string());
        let s = format!("{}", info);
        assert!(s.contains("Basic"));
    }
}
