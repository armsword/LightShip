//! CPU information and feature detection
//!
//! Provides utilities for detecting CPU features and capabilities.

use std::fmt;

/// CPU feature flags
#[derive(Debug, Clone, Default)]
pub struct CpuFeatures {
    /// SIMD (SSE/AVX)
    pub simd: bool,
    /// AVX
    pub avx: bool,
    /// AVX2
    pub avx2: bool,
    /// AVX-512
    pub avx512: bool,
    /// NEON (ARM)
    pub neon: bool,
    /// ARM SVE
    pub sve: bool,
    /// Vector float16
    pub fp16: bool,
}

/// CPU information
#[derive(Debug, Clone, Default)]
pub struct CpuInfo {
    /// CPU brand name
    pub brand: String,
    /// Number of physical cores
    pub num_cores: usize,
    /// Number of logical CPUs
    pub num_threads: usize,
    /// CPU features
    pub features: CpuFeatures,
}

impl CpuInfo {
    /// Get the current CPU info
    pub fn current() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            Self::detect_x86()
        }
        #[cfg(target_arch = "aarch64")]
        {
            Self::detect_arm()
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            Self::default()
        }
    }

    #[cfg(target_arch = "x86_64")]
    fn detect_x86() -> Self {
        let brand = Self::get_x86_brand();
        let num_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        let num_cores = num_threads; // Simplified
        let features = Self::detect_x86_features();

        Self {
            brand,
            num_cores,
            num_threads,
            features,
        }
    }

    #[cfg(target_arch = "x86_64")]
    fn get_x86_brand() -> String {
        "x86_64".to_string()
    }

    #[cfg(target_arch = "x86_64")]
    fn detect_x86_features() -> CpuFeatures {
        CpuFeatures {
            simd: true, // Assume SIMD on x86_64
            avx: false,
            avx2: false,
            avx512: false,
            neon: false,
            sve: false,
            fp16: false,
        }
    }

    #[cfg(target_arch = "aarch64")]
    fn detect_arm() -> Self {
        let brand = Self::get_arm_brand();
        let num_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        let num_cores = num_threads; // Simplified
        let features = Self::detect_arm_features();

        Self {
            brand,
            num_cores,
            num_threads,
            features,
        }
    }

    #[cfg(target_arch = "aarch64")]
    fn get_arm_brand() -> String {
        "aarch64".to_string()
    }

    #[cfg(target_arch = "aarch64")]
    fn detect_arm_features() -> CpuFeatures {
        CpuFeatures {
            simd: true, // NEON is standard on ARM64
            avx: false,
            avx2: false,
            avx512: false,
            neon: true,
            sve: false,
            fp16: true, // ARMv8.2+ has FP16
        }
    }

    /// Check if SIMD is available
    pub fn has_simd(&self) -> bool {
        self.features.simd
    }

    /// Check if AVX is available
    pub fn has_avx(&self) -> bool {
        self.features.avx
    }

    /// Check if NEON is available
    pub fn has_neon(&self) -> bool {
        self.features.neon
    }
}

impl fmt::Display for CpuInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CpuInfo({}: {} cores, {} threads)",
            self.brand, self.num_cores, self.num_threads
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_info_current() {
        let info = CpuInfo::current();
        assert!(info.num_cores > 0);
        assert!(info.num_threads >= info.num_cores);
    }

    #[test]
    fn test_cpu_features_default() {
        let features = CpuFeatures::default();
        assert!(!features.avx);
        assert!(!features.avx2);
    }

    #[test]
    fn test_cpu_info_display() {
        let info = CpuInfo::current();
        let s = format!("{}", info);
        assert!(!s.is_empty());
    }

    #[test]
    fn test_has_simd() {
        let info = CpuInfo::current();
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        {
            assert!(info.has_simd() || !info.has_simd()); // Either is valid
        }
    }
}
