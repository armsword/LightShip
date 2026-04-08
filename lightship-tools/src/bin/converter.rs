//! Model converter binary
//!
//! Converts models from ONNX and other formats to LightShip native format.

use anyhow::Result;
use clap::{Parser, ValueHint};
use lightship_tools::converter::{ConversionPath, ModelConverter, ModelFormat};
use std::path::PathBuf;

/// Model format converter
#[derive(Parser, Debug)]
#[command(name = "lightship-convert")]
#[command(about = "Convert model files to LightShip native format", long_about = None)]
struct Args {
    /// Input model file
    #[arg(short, long, value_hint = ValueHint::FilePath)]
    input: PathBuf,

    /// Output model file
    #[arg(short, long, value_hint = ValueHint::FilePath)]
    output: PathBuf,

    /// Input format (auto-detected if not specified)
    #[arg(short, long, value_enum)]
    input_format: Option<String>,

    /// Output format (default: native)
    #[arg(short, long, value_enum, default_value = "native")]
    output_format: String,

    /// Optimization level (0-3)
    #[arg(short, long, default_value = "3")]
    optimization: u32,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,
}

impl Args {
    fn input_format_detected(&self) -> Option<ConversionPath> {
        ConversionPath::detect(&self.input)
    }
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

    // Auto-detect input format if not specified
    let conversion = args.input_format_detected()
        .ok_or_else(|| anyhow::anyhow!(
            "Could not detect model format from file extension: {}",
            args.input.display()
        ))?;

    tracing::info!("Converting model using path: {:?}", conversion);

    // Create converter and run
    let converter = ModelConverter::new(
        ModelFormat::Onnx,
        ModelFormat::Native,
    )
    .with_optimization(args.optimization);

    let result = converter.convert(&args.input, &args.output)?;

    println!("\n{}", result);

    Ok(())
}
