use std::env;

fn main() {
    let target = env::var("TARGET").unwrap_or_default();

    // Detect CPU features for SIMD
    if target.contains("x86_64") || target.contains("x86") {
        println!("cargo:rustc-env=HAS_AVX2=1");
        println!("cargo:rustc-env=HAS_AVX512=1");
    } else if target.contains("aarch64") || target.contains("arm") {
        println!("cargo:rustc-env=HAS_NEON=1");
    }

    // Platform-specific configurations
    if target.contains("apple") {
        println!("cargo:rustc-env=PLATFORM=apple");
    } else if target.contains("linux") {
        println!("cargo:rustc-env=PLATFORM=linux");
    } else if target.contains("windows") {
        println!("cargo:rustc-env=PLATFORM=windows");
    }

    println!("cargo:rerun-if-changed=build.rs");
}
