//! Benchmark tool binary
//!
//! Measures inference latency, throughput, and memory usage.

use anyhow::Result;
use clap::{Parser, ValueHint};
use lightship_tools::benchmark::{BenchmarkOptions, BenchmarkRunner};
use std::path::PathBuf;

/// Benchmark tool
#[derive(Parser, Debug)]
#[command(name = "lightship-benchmark")]
#[command(about = "Benchmark model inference performance", long_about = None)]
struct Args {
    /// Model file to benchmark
    #[arg(short, long, value_hint = ValueHint::FilePath)]
    model: PathBuf,

    /// Number of warmup runs
    #[arg(short, long, default_value = "10")]
    warmup: usize,

    /// Number of benchmark runs
    #[arg(short, long, default_value = "100")]
    runs: usize,

    /// Input batch size
    #[arg(short, long, default_value = "1")]
    batch_size: usize,

    /// Input shape (comma-separated, e.g., "1,3,224,224")
    #[arg(short, long, default_value = "1,3,224,224")]
    input_shape: String,

    /// Number of threads (0 = auto)
    #[arg(short, long, default_value = "0")]
    threads: usize,

    /// Enable detailed profiling
    #[arg(long)]
    detailed: bool,

    /// Backend to use
    #[arg(long, default_value = "auto")]
    backend: String,

    /// Output format (text, json)
    #[arg(long, default_value = "text")]
    output_format: String,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Initialize logging
    if args.verbose {
        tracing_subscriber::fmt()
            .with_max_level(tracing::Level::DEBUG)
            .init();
    } else {
        tracing_subscriber::fmt()
            .with_max_level(tracing::Level::INFO)
            .init();
    }

    // Note: Full benchmark requires Phase 4 (Model Loading)
    println!("LightShip Benchmark Tool");
    println!("========================");
    println!();
    println!("Note: Benchmark functionality requires Phase 4 (Model Loading) implementation.");
    println!();
    println!("API Usage:");
    println!("  use lightship_core::api::{{Engine, EngineConfig}};");
    println!("  let engine = Engine::new()?;");
    println!("  let options = BenchmarkOptions {{ ... }};");
    println!("  let runner = BenchmarkRunner::new(options);");
    println!("  let report = runner.run('model.onnx')?;");

    Ok(())
}
