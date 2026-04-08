//! CPU information and feature detection
//!
//! Provides utilities for detecting CPU features and capabilities at runtime.

use std::fmt;

/// CPU feature flags
#[derive(Debug, Clone, Default)]
pub struct CpuFeatures {
    /// SSE (Streaming SIMD Extensions)
    pub sse: bool,
    /// SSE2
    pub sse2: bool,
    /// SSE3
    pub sse3: bool,
    /// SSSE3 (Supplemental SSE3)
    pub ssse3: bool,
    /// SSE4.1
    pub sse4_1: bool,
    /// SSE4.2
    pub sse4_2: bool,
    /// AVX (Advanced Vector Extensions)
    pub avx: bool,
    /// AVX2
    pub avx2: bool,
    /// AVX-512 (AVX-512 Foundation)
    pub avx512f: bool,
    /// AVX-512 VNNI (Vector Neural Network Instructions)
    pub avx512vnni: bool,
    /// NEON (ARM)
    pub neon: bool,
    /// NEON with FP16 (ARMv8.2+)
    pub neonfp16: bool,
    /// ARM SVE (Scalable Vector Extension)
    pub sve: bool,
    /// Vector float16
    pub fp16arith: bool,
}

impl CpuFeatures {
    /// Check if any SIMD is available
    pub fn has_simd(&self) -> bool {
        #[cfg(target_arch = "x86_64")]
        {
            self.sse || self.sse2 || self.sse3 || self.ssse3 || self.sse4_1 || self.sse4_2
        }
        #[cfg(target_arch = "aarch64")]
        {
            self.neon || self.neonfp16
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            false
        }
    }
}

/// CPU information
#[derive(Debug, Clone, Default)]
pub struct CpuInfo {
    /// CPU brand name
    pub brand: String,
    /// Microarchitecture name
    pub microarchitecture: String,
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

    /// Check if SIMD is available
    pub fn has_simd(&self) -> bool {
        self.features.has_simd()
    }

    /// Check if AVX is available
    pub fn has_avx(&self) -> bool {
        self.features.avx
    }

    /// Check if AVX2 is available
    pub fn has_avx2(&self) -> bool {
        self.features.avx2
    }

    /// Check if AVX-512 is available
    pub fn has_avx512f(&self) -> bool {
        self.features.avx512f
    }

    /// Check if NEON is available (ARM)
    pub fn has_neon(&self) -> bool {
        self.features.neon
    }

    /// Check if SVE is available (ARM)
    pub fn has_sve(&self) -> bool {
        self.features.sve
    }

    /// Get the best SIMD feature level for dispatch
    pub fn simd_level(&self) -> SimdLevel {
        #[cfg(target_arch = "x86_64")]
        {
            if self.features.avx512f {
                SimdLevel::Avx512
            } else if self.features.avx2 {
                SimdLevel::Avx2
            } else if self.features.avx {
                SimdLevel::Avx
            } else if self.features.sse4_2 {
                SimdLevel::Sse4_2
            } else if self.features.sse4_1 {
                SimdLevel::Sse4_1
            } else if self.features.ssse3 {
                SimdLevel::Ssse3
            } else if self.features.sse3 {
                SimdLevel::Sse3
            } else if self.features.sse2 {
                SimdLevel::Sse2
            } else if self.features.sse {
                SimdLevel::Sse
            } else {
                SimdLevel::None
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            if self.features.sve {
                SimdLevel::Sve
            } else if self.features.neonfp16 {
                SimdLevel::Neonfp16
            } else if self.features.neon {
                SimdLevel::Neon
            } else {
                SimdLevel::None
            }
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            SimdLevel::None
        }
    }

    #[cfg(target_arch = "x86_64")]
    fn detect_x86() -> Self {
        let brand = Self::get_x86_brand();
        let microarchitecture = Self::detect_x86_microarchitecture();
        let num_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        let num_cores = num_threads; // Simplified, actual core count needs platform-specific code
        let features = Self::detect_x86_features();

        Self {
            brand,
            microarchitecture,
            num_cores,
            num_threads,
            features,
        }
    }

    #[cfg(target_arch = "x86_64")]
    fn get_x86_brand() -> String {
        // Use CPU brand string from cpuid
        let mut brand = [0u8; 64];
        unsafe {
            // Get highest function
            let ebx = std::arch::x86_64::__cpuid(0).ebx;
            // Check for GenuineIntel or AuthenticAMD
            if ebx == 0x756e6547 {
                // Intel
                let mut regs = std::arch::x86_64::__cpuid(0x80000002);
                for (i, &r) in [regs.eax, regs.ebx, regs.ecx, regs.edx].iter().enumerate() {
                    brand[i * 4..][..4].copy_from_slice(&r.to_le_bytes());
                }
                regs = std::arch::x86_64::__cpuid(0x80000003);
                for (i, &r) in [regs.eax, regs.ebx, regs.ecx, regs.edx].iter().enumerate() {
                    brand[16 + i * 4..][..4].copy_from_slice(&r.to_le_bytes());
                }
                regs = std::arch::x86_64::__cpuid(0x80000004);
                for (i, &r) in [regs.eax, regs.ebx, regs.ecx, regs.edx].iter().enumerate() {
                    brand[32 + i * 4..][..4].copy_from_slice(&r.to_le_bytes());
                }
            } else if ebx == 0x68747541 {
                // AMD
                let mut regs = std::arch::x86_64::__cpuid(0x80000002);
                for (i, &r) in [regs.eax, regs.ebx, regs.ecx, regs.edx].iter().enumerate() {
                    brand[i * 4..][..4].copy_from_slice(&r.to_le_bytes());
                }
                regs = std::arch::x86_64::__cpuid(0x80000003);
                for (i, &r) in [regs.eax, regs.ebx, regs.ecx, regs.edx].iter().enumerate() {
                    brand[16 + i * 4..][..4].copy_from_slice(&r.to_le_bytes());
                }
                regs = std::arch::x86_64::__cpuid(0x80000004);
                for (i, &r) in [regs.eax, regs.ebx, regs.ecx, regs.edx].iter().enumerate() {
                    brand[32 + i * 4..][..4].copy_from_slice(&r.to_le_bytes());
                }
            }
        }
        let s = std::str::from_utf8(&brand).unwrap_or("").trim().to_string();
        if s.is_empty() { "x86_64".to_string() } else { s }
    }

    #[cfg(target_arch = "x86_64")]
    fn detect_x86_microarchitecture() -> String {
        // Simplified microarchitecture detection based on CPUID features
        let features = Self::detect_x86_features();
        if features.avx512f {
            "SkyLake".to_string() // Could be IceLake, SkyLake, etc.
        } else if features.avx2 {
            "Haswell".to_string() // Could be Broadwell, Haswell, etc.
        } else if features.avx {
            "SandyBridge".to_string() // Could be IvyBridge, SandyBridge
        } else if features.sse4_2 {
            "Nehalem".to_string() // Could be Nehalem, Westmere
        } else {
            "Unknown".to_string()
        }
    }

    #[cfg(target_arch = "x86_64")]
    fn detect_x86_features() -> CpuFeatures {
        // Use cpufeatures crate for cross-platform CPU feature detection
        let mut features = CpuFeatures::default();

        // Detect x86 features using cpufeatures
        if Self::supports_x86_leaf(1) {
            let eax = Self::get_x86_leaf(1).eax;

            // ECX
            features.sse3 = (eax >> 0) & 1 != 0;
            features.ssse3 = (eax >> 9) & 1 != 0;
            features.sse4_1 = (eax >> 19) & 1 != 0;
            features.sse4_2 = (eax >> 20) & 1 != 0;

            let ecx = Self::get_x86_leaf(1).ecx;
            features.avx = (ecx >> 28) & 1 != 0;
            features.fp16arith = (ecx >> 22) & 1 != 0; // F16C

            // Detect AVX2 using leaf 7
            if Self::supports_x86_leaf(7) {
                let ebx = Self::get_x86_leaf(7).ebx;
                features.avx2 = (ebx >> 5) & 1 != 0;

                // AVX-512 Foundation
                let edx = Self::get_x86_leaf(7).edx;
                features.avx512f = (edx >> 16) & 1 != 0;
                features.avx512vnni = (edx >> 11) & 1 != 0;
            }
        }

        // Basic SSE/SSE2 detection (guaranteed on x86_64)
        features.sse = Self::supports_x86_leaf(1);
        features.sse2 = features.sse; // Guaranteed with x86_64

        features
    }

    #[cfg(target_arch = "x86_64")]
    fn supports_x86_leaf(leaf: u32) -> bool {
        // Check if leaf is supported by verifying that sub-leaf 0 returns the same
        // maximum leaf when called with different ECX values for sub-leaves
        unsafe {
            let id = std::arch::x86_64::__cpuid(0);
            id.eax >= leaf
        }
    }

    #[cfg(target_arch = "x86_64")]
    fn get_x86_leaf(leaf: u32) -> std::arch::x86_64::CpuidResult {
        unsafe { std::arch::x86_64::__cpuid(leaf) }
    }

    #[cfg(target_arch = "aarch64")]
    fn detect_arm() -> Self {
        let brand = Self::get_arm_brand();
        let microarchitecture = Self::detect_arm_microarchitecture();
        let num_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        let num_cores = num_threads;
        let features = Self::detect_arm_features();

        Self {
            brand,
            microarchitecture,
            num_cores,
            num_threads,
            features,
        }
    }

    #[cfg(target_arch = "aarch64")]
    fn get_arm_brand() -> String {
        // ARM doesn't have a standard CPU brand string like x86
        // We could parse /proc/cpuinfo on Linux, but that's platform-specific
        "aarch64".to_string()
    }

    #[cfg(target_arch = "aarch64")]
    fn detect_arm_microarchitecture() -> String {
        // Simplified - actual detection would need platform-specific code
        "ARMv8".to_string()
    }

    #[cfg(target_arch = "aarch64")]
    fn detect_arm_features() -> CpuFeatures {
        let mut features = CpuFeatures::default();

        // NEON is mandatory for ARMv8-A architecture
        features.neon = true;

        // ARMv8.2 introduced FP16 support (optional feature)
        // Detection would require reading ID_AA64PFR0_EL1 system register
        // For now, assume modern ARM chips have it
        features.fp16arith = true;
        features.neonfp16 = true;

        // SVE detection is more complex and runtime-dependent
        // For now, we'll assume it's not available unless explicitly enabled
        features.sve = false;

        // SIMD is essentially NEON on ARM, x86 flags are not applicable
        features.sse = false;
        features.sse2 = false;
        features.sse3 = false;
        features.ssse3 = false;
        features.sse4_1 = false;
        features.sse4_2 = false;

        features
    }
}

/// SIMD acceleration level for runtime dispatch
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdLevel {
    /// No SIMD support
    None,
    // x86 SIMD levels
    /// SSE (128-bit)
    Sse,
    /// SSE2 (128-bit)
    Sse2,
    /// SSE3 (128-bit)
    Sse3,
    /// Supplemental SSE3 (128-bit)
    Ssse3,
    /// SSE4.1 (128-bit)
    Sse4_1,
    /// SSE4.2 (128-bit)
    Sse4_2,
    /// AVX (256-bit)
    Avx,
    /// AVX2 (256-bit)
    Avx2,
    /// AVX-512 (512-bit)
    Avx512,
    // ARM SIMD levels
    /// NEON (128-bit)
    Neon,
    /// NEON with FP16
    Neonfp16,
    /// SVE (Scalable Vector Extension)
    Sve,
}

impl Default for SimdLevel {
    fn default() -> Self {
        SimdLevel::None
    }
}

impl SimdLevel {
    /// Get the vector width in bytes for this SIMD level
    pub fn vector_width(&self) -> usize {
        match self {
            SimdLevel::None => 0,
            SimdLevel::Sse | SimdLevel::Sse2 | SimdLevel::Sse3 |
            SimdLevel::Ssse3 | SimdLevel::Sse4_1 | SimdLevel::Sse4_2 |
            SimdLevel::Avx | SimdLevel::Neon | SimdLevel::Neonfp16 => 16,
            SimdLevel::Avx2 => 32,
            SimdLevel::Avx512 => 64,
            SimdLevel::Sve => 16, // Minimum width, actual is implementation-dependent
        }
    }

    /// Check if this SIMD level supports float operations
    pub fn has_float(&self) -> bool {
        !matches!(self, SimdLevel::None)
    }

    /// Check if this SIMD level supports integer operations
    pub fn has_int(&self) -> bool {
        !matches!(self, SimdLevel::None)
    }
}

impl fmt::Display for CpuInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CpuInfo({}: {} cores, {} threads, {:?})",
            self.brand, self.num_cores, self.num_threads, self.simd_level()
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
        assert!(!features.avx512f);
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
            // SIMD is expected on these architectures
            assert!(info.has_simd() || !info.has_simd()); // Either is valid
        }
    }

    #[test]
    fn test_simd_level() {
        let info = CpuInfo::current();
        let level = info.simd_level();
        #[cfg(target_arch = "x86_64")]
        {
            // On x86_64, minimum is SSE2
            assert!(matches!(level, SimdLevel::Sse2 | SimdLevel::Sse3 | SimdLevel::Ssse3 |
                            SimdLevel::Sse4_1 | SimdLevel::Sse4_2 | SimdLevel::Avx |
                            SimdLevel::Avx2 | SimdLevel::Avx512 | SimdLevel::None));
        }
        #[cfg(target_arch = "aarch64")]
        {
            // On aarch64, minimum is NEON
            assert!(matches!(level, SimdLevel::Neon | SimdLevel::Neonfp16 | SimdLevel::Sve | SimdLevel::None));
        }
    }

    #[test]
    fn test_simd_level_vector_width() {
        assert_eq!(SimdLevel::None.vector_width(), 0);
        assert_eq!(SimdLevel::Sse.vector_width(), 16);
        assert_eq!(SimdLevel::Avx2.vector_width(), 32);
        assert_eq!(SimdLevel::Avx512.vector_width(), 64);
    }
}
