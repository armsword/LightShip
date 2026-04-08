# 异步推理与性能分析模块

## 概述

Phase 9 实现了 LightShip 的异步推理和性能分析支持，包括 AsyncHandle、TimingInfo 和 ProfilingInfo，为推理性能优化提供了完整的工具链。

## AsyncHandle (异步句柄)

### 状态机

```rust
pub enum AsyncStatus {
    Pending,    // 等待中
    Running,    // 执行中
    Completed,  // 已完成
    Failed,     // 失败
    Cancelled,  // 已取消
}
```

### 结构定义

```rust
pub struct AsyncHandle {
    status: AsyncStatus,
    result: AsyncResult,
    id: u64,
}

pub struct AsyncResult {
    pub completed: bool,
    pub error: Option<String>,
}
```

### Future 实现

AsyncHandle 实现了 Rust 的 Future trait，可以用于 async/await：

```rust
impl Future for AsyncHandle {
    type Output = AsyncResult;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        if self.is_done() {
            Poll::Ready(self.result.clone())
        } else {
            Poll::Pending
        }
    }
}
```

### 使用方式

```rust
// 创建异步句柄
let handle = AsyncHandle::new(request_id);

// 状态检查
if handle.is_pending() {
    println!("推理进行中...");
}

// 阻塞等待
let result = handle.wait();

// 或使用 async/await
let result = handle.await;
```

## TimingInfo (计时信息)

### 结构定义

```rust
pub struct TimingInfo {
    pub total_time: Duration,           // 总时间
    pub load_time: Option<Duration>,    // 模型加载时间
    pub compile_time: Option<Duration>, // 编译时间
    pub execution_time: Duration,       // 执行时间
    pub operator_times: Vec<OperatorTiming>, // 算子级计时
    pub memory_time: Option<Duration>,  // 内存操作时间
    pub is_valid: bool,
}

pub struct OperatorTiming {
    pub name: String,           // 算子名称
    pub operator_type: String,   // 算子类型
    pub duration: Duration,       // 执行时间
    pub call_count: usize,        // 调用次数
}
```

### 计时器

```rust
pub struct Timer {
    start: Instant,
    end: Option<Instant>,
}

impl Timer {
    pub fn start() -> Self { ... }
    pub fn stop(&mut self) -> Duration { ... }
    pub fn elapsed(&self) -> Duration { ... }
}

// 使用示例
let mut timer = Timer::start();
// 执行推理
let elapsed = timer.stop();
println!("推理耗时: {:.3}ms", elapsed.as_secs_f64() * 1000.0);
```

### 输出格式

```rust
impl Display for TimingInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TimingInfo(total={:.3}ms", self.total_time.as_secs_f64() * 1000.0)?;
        if let Some(load) = self.load_time {
            write!(f, ", load={:.3}ms", load.as_secs_f64() * 1000.0)?;
        }
        write!(f, ", exec={:.3}ms)", self.execution_time.as_secs_f64() * 1000.0)
    }
}
```

## ProfilingInfo (性能分析)

### 分析级别

```rust
pub enum ProfilingLevel {
    Off,        // 关闭
    Basic,      // 基础 (仅总时间)
    Operators,  // 算子级
    Detailed,   // 详细 (含内存)
    Max,        // 最大 (含缓存)
}

impl ProfilingLevel {
    pub fn includes_operators(&self) -> bool {
        matches!(self, ProfilingLevel::Operators | ProfilingLevel::Detailed | ProfilingLevel::Max)
    }

    pub fn includes_memory(&self) -> bool {
        matches!(self, ProfilingLevel::Detailed | ProfilingLevel::Max)
    }

    pub fn includes_cache(&self) -> bool {
        matches!(self, ProfilingLevel::Max)
    }
}
```

### 内存分析

```rust
pub struct MemoryProfile {
    pub peak_bytes: usize,        // 峰值内存
    pub current_bytes: usize,     // 当前内存
    pub allocation_count: u64,    // 分配次数
    pub deallocation_count: u64,  // 释放次数
    pub allocation_time_ns: u64,  // 分配耗时
}

impl MemoryProfile {
    pub fn net_allocations(&self) -> i64 {
        self.allocation_count as i64 - self.deallocation_count as i64
    }
}
```

### 缓存分析

```rust
pub struct CacheProfile {
    pub l1_hits: u64,   pub l1_misses: u64,
    pub l2_hits: u64,   pub l2_misses: u64,
    pub l3_hits: u64,   pub l3_misses: u64,
}

impl CacheProfile {
    pub fn l1_hit_rate(&self) -> f64 {
        let total = self.l1_hits + self.l1_misses;
        if total > 0 { self.l1_hits as f64 / total as f64 } else { 0.0 }
    }

    pub fn overall_hit_rate(&self) -> f64 {
        let total_hits = self.l1_hits + self.l2_hits + self.l3_hits;
        let total_misses = self.l1_misses + self.l2_misses + self.l3_misses;
        let total = total_hits + total_misses;
        if total > 0 { total_hits as f64 / total as f64 } else { 0.0 }
    }
}
```

### 完整分析信息

```rust
pub struct ProfilingInfo {
    pub level: ProfilingLevel,
    pub timing: TimingInfo,
    pub memory: Option<MemoryProfile>,
    pub cache: Option<CacheProfile>,
    pub operator_count: u32,
    pub node_count: u32,
    pub backend: String,
}

impl ProfilingInfo {
    pub fn throughput(&self) -> f64 {
        let time_secs = self.timing.total_time.as_secs_f64();
        if time_secs > 0.0 { 1.0 / time_secs } else { 0.0 }
    }
}
```

## 使用示例

### 同步计时

```rust
use lightship_core::runtime::{TimingInfo, Timer};

let mut timer = Timer::start();
// 执行推理
session.run(&inputs, &outputs);
let elapsed = timer.stop();

let timing = TimingInfo::from_execution(elapsed);
println!("{}", timing);
```

### 异步推理

```rust
use lightship_core::runtime::AsyncHandle;

let handle = session.run_async(&inputs)?;

// 轮询检查状态
while !handle.is_done() {
    println!("进度: {:?}", handle.status());
    std::thread::sleep(Duration::from_millis(10));
}

// 获取结果
match handle.status() {
    AsyncStatus::Completed => println!("推理完成!"),
    AsyncStatus::Failed => println!("失败: {:?}", handle.error()),
    _ => {}
}
```

### 性能分析

```rust
use lightship_core::runtime::{ProfilingInfo, ProfilingLevel};

let info = ProfilingInfo::new(ProfilingLevel::Operators, "CPU".to_string());
// ... 执行带分析的推理 ...

println!("吞吐量: {:.2} 推理/秒", info.throughput());
println!("{}", info);
```
