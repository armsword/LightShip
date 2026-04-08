# 工具链模块

## 概述

Phase 11 实现了 LightShip 推理引擎的工具链，包括模型转换工具、性能基准测试工具、配置管理和示例代码。

## 工具箱结构

```
lightship-tools/
├── Cargo.toml
├── src/
│   ├── lib.rs
│   ├── bin/
│   │   ├── converter.rs    # 模型转换 CLI
│   │   ├── benchmark.rs    # 基准测试 CLI
│   │   └── info.rs         # 模型信息 CLI
│   ├── converter/          # 模型转换模块
│   ├── benchmark/           # 基准测试模块
│   ├── config/             # 配置管理模块
│   └── visualizer/         # 模型可视化模块
└── examples/               # 示例代码目录
    ├── basic_inference.rs
    └── async_inference.rs
```

## 配置管理 (config/)

### Config 结构

```rust
pub struct Config {
    pub engine: EngineConfig,
    pub session: SessionConfig,
    pub benchmark: BenchmarkConfig,
}
```

### EngineConfig

```rust
pub struct EngineConfig {
    pub log_level: String,        // 日志级别
    pub num_threads: usize,       // 线程数 (0 = 自动)
    pub backend: BackendConfig,   // 后端配置
    pub memory: MemoryConfig,     // 内存配置
}
```

### SessionConfig

```rust
pub struct SessionConfig {
    pub backend: BackendConfig,
    pub enable_profiling: bool,
    pub profiling_level: String,
    pub execution_mode: ExecutionMode,
}
```

### BenchmarkConfig

```rust
pub struct BenchmarkConfig {
    pub warmup_runs: usize,       // 预热次数
    pub runs: usize,              // 测试次数
    pub input_shape: Vec<usize>,  // 输入形状
    pub batch_size: usize,        // 批大小
    pub num_threads: usize,       // 线程数
    pub detailed_timing: bool,    // 详细计时
}
```

### 配置加载

```rust
// 从文件加载配置
let config = Config::from_file("config.toml")?;

// 保存配置
config.save("config.json")?;
```

## 模型转换工具 (converter/)

### ModelConverter

```rust
pub struct ModelConverter {
    input_format: ModelFormat,
    output_format: ModelFormat,
    optimization_level: u32,
}

impl ModelConverter {
    pub fn new(input_format: ModelFormat, output_format: ModelFormat) -> Self;
    pub fn with_optimization(mut self, level: u32) -> Self;
    pub fn convert<P: AsRef<Path>>(&self, input: P, output: P) -> Result<ConversionResult>;
}
```

### 支持的转换路径

```rust
pub enum ConversionPath {
    OnnxToNative,
    TFLiteToNative,
    CaffeToNative,
}

impl ConversionPath {
    pub fn detect<P: AsRef<Path>>(path: P) -> Option<Self>;
}
```

### 使用示例

```rust
use lightship_tools::converter::{ModelConverter, ModelFormat};

let converter = ModelConverter::new(
    ModelFormat::Onnx,
    ModelFormat::Native,
)
.with_optimization(3);

let result = converter.convert("model.onnx", "model.lship")?;
println!("{}", result);
```

## 基准测试工具 (benchmark/)

### BenchmarkOptions

```rust
pub struct BenchmarkOptions {
    pub warmup_runs: usize,       // 预热迭代次数
    pub runs: usize,              // 基准测试迭代次数
    pub input_shape: Vec<usize>,  // 输入形状
    pub batch_size: usize,        // 批大小
    pub detailed_profiling: bool, // 详细性能分析
    pub num_threads: usize,       // 线程数
}
```

### BenchmarkStatistics

```rust
pub struct BenchmarkStatistics {
    pub mean_ns: u64,      // 平均延迟
    pub median_ns: u64,    // 中位数延迟
    pub min_ns: u64,       // 最小延迟
    pub max_ns: u64,       // 最大延迟
    pub std_dev_ns: u64,    // 标准差
    pub p50_ns: u64,       // P50 延迟
    pub p90_ns: u64,       // P90 延迟
    pub p95_ns: u64,       // P95 延迟
    pub p99_ns: u64,       // P99 延迟
    pub throughput: f64,   // 吞吐量
    pub memory_bytes: Option<usize>,  // 内存使用
}

impl BenchmarkStatistics {
    pub fn mean_ms(&self) -> f64;
    pub fn median_ms(&self) -> f64;
}
```

### BenchmarkRunner

```rust
pub struct BenchmarkRunner {
    options: BenchmarkOptions,
}

impl BenchmarkRunner {
    pub fn new(options: BenchmarkOptions) -> Self;
    pub fn run(&self, model_path: &str) -> Result<BenchmarkReport>;
}
```

### BenchmarkReport

```rust
pub struct BenchmarkReport {
    pub options: BenchmarkOptions,
    pub statistics: BenchmarkStatistics,
    pub total_time: Duration,
}

impl Display for BenchmarkReport {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result { ... }
}
```

## 模型可视化 (visualizer/)

### VisualizerOptions

```rust
pub struct VisualizerOptions {
    pub format: OutputFormat,     // 输出格式
    pub show_attributes: bool,    // 显示属性
    pub show_shapes: bool,       // 显示形状
    pub show_dtypes: bool,       // 显示数据类型
    pub max_nodes: usize,        // 最大节点数
    pub filter_pattern: Option<String>,  // 过滤模式
}
```

### OutputFormat

```rust
pub enum OutputFormat {
    Text,   // 文本格式
    Json,   // JSON 格式
    Dot,    // Graphviz DOT 格式
}
```

## CLI 工具

### lightship-convert

```bash
# 转换 ONNX 模型到原生格式
lightship-convert -i model.onnx -o model.lship

# 指定优化级别
lightship-convert -i model.onnx -o model.lship --optimization 3

# 详细输出
lightship-convert -i model.onnx -o model.lship -v
```

### lightship-benchmark

```bash
# 基准测试模型
lightship-benchmark -m model.lship

# 自定义参数
lightship-benchmark -m model.lship \
    --warmup 10 \
    --runs 100 \
    --batch-size 1 \
    --input-shape 1,3,224,224

# 输出 JSON 格式
lightship-benchmark -m model.lship --output-format json
```

### lightship-info

```bash
# 查看模型信息
lightship-info -m model.lship

# 显示所有详细信息
lightship-info -m model.lship --all

# 显示算子列表
lightship-info -m model.lship --operators

# 导出 DOT 文件
lightship-info -m model.lship --export-dot model.dot
```

## 示例代码

### 基础推理示例

```rust
use lightship_core::api::{Engine, EngineConfig};

fn main() -> anyhow::Result<()> {
    println!("LightShip Basic Inference Example");

    println!("API Usage:");
    println!("  let engine = Engine::new(config)?;");
    println!("  let session = engine.create_session(&model, config)?;");
    println!("  let output = session.forward(&[('input', &tensor)], &['output'])?;");

    Ok(())
}
```

### 异步推理示例

```rust
use lightship_core::api::{Engine, EngineConfig};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("LightShip Async Inference Example");

    println!("API Usage (async):");
    println!("  let handle = session.forward_async(&[('input', &tensor)], &['output'])?;");
    println!("  // Do other work...");
    println!("  let result = handle.await;");

    Ok(())
}
```

## 配置文件格式

### TOML 配置示例

```toml
[engine]
log_level = "info"
num_threads = 0

[engine.backend]
auto = true

[engine.memory]
max_memory = 0
enable_pooling = true
reuse_strategy = "default"

[session]
enable_profiling = true
profiling_level = "basic"
execution_mode = "synchronous"

[benchmark]
warmup_runs = 10
runs = 100
input_shape = [1, 3, 224, 224]
batch_size = 1
detailed_timing = false
```

### JSON 配置示例

```json
{
  "engine": {
    "log_level": "info",
    "num_threads": 0,
    "backend": {
      "auto": true
    },
    "memory": {
      "max_memory": 0,
      "enable_pooling": true,
      "reuse_strategy": "default"
    }
  },
  "session": {
    "enable_profiling": true,
    "profiling_level": "basic",
    "execution_mode": "synchronous"
  },
  "benchmark": {
    "warmup_runs": 10,
    "runs": 100,
    "input_shape": [1, 3, 224, 224],
    "batch_size": 1,
    "detailed_timing": false
  }
}
```

## 下一步

Phase 11 完成。LightShip 推理引擎的所有计划阶段已实现：

- Phase 1-5: 项目基础设施、核心数据结构、CPU 后端、模型加载、内存管理
- Phase 6-9: 图优化、量化支持、Transformer 算子、异步推理
- Phase 10: GPU/异构后端与跨平台
- Phase 11: 工具链

后续可以继续优化和扩展功能。
