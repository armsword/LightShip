# LightShip 多线程调度框架实现原理

## 1. 概述

本文档描述 LightShip CPU 后端的多线程调度框架 `thread_scheduler`，包括：

- Block-wise 调度：将计算密集型算子切分为多个 block 并行执行
- Rayon 线程池管理
- CPU 亲和性设置
- 负载均衡策略
- 流水线并行支持

**目标**：利用多核 CPU 提升推理性能，实现算子级的多线程并行。

---

## 2. 架构设计

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                     ThreadScheduler                         │
│  ┌─────────────────┐  ┌─────────────────────────────────┐  │
│  │ SchedulerConfig │  │          ThreadPool              │  │
│  │  - num_threads  │  │  ┌──────────────────────────┐   │  │
│  │  - max_block_size│  │  │   Rayon Global Pool      │   │  │
│  │  - use_affinity │  │  │  ┌────┐ ┌────┐ ┌────┐    │   │  │
│  │  - load_balance │  │  │  │ W0 │ │ W1 │ │ W2 │ ...│   │  │
│  └─────────────────┘  │  │  └────┘ └────┘ └────┘    │   │  │
│                        │  └──────────────────────────┘   │  │
│  ┌─────────────────────────────────────────────────────┐  │  │
│  │              BlockScheduler                         │  │  │
│  │  - decompose_conv2d() → Vec<ComputeBlock>           │  │  │
│  │  - decompose_matmul()  → Vec<ComputeBlock>           │  │  │
│  └─────────────────────────────────────────────────────┘  │  │
│  ┌─────────────────────────────────────────────────────┐  │  │
│  │           LoadBalancingStrategy                     │  │  │
│  │  - Greedy: 最少负载 worker 获取下一个任务            │  │  │
│  │  - RoundRobin: 轮询分配                              │  │  │
│  │  - WorkStealing: 工作窃取（暂未实现）                │  │  │
│  └─────────────────────────────────────────────────────┘  │  │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 核心组件

| 组件 | 说明 |
|------|------|
| `ThreadScheduler` | 主调度器，协调线程池和 Block 分解 |
| `ThreadPool` | 线程池封装（基于 Rayon） |
| `BlockScheduler` | Block 分解器，将大算子切分为小 Block |
| `ComputeBlock` | 可并行执行的计算 Block |
| `WorkloadEstimate` | 工作量估计（FLOPs、内存占用） |
| `LoadBalancingStrategy` | 负载均衡策略 |
| `CpuAffinity` | CPU 亲和性管理 |
| `PipelineSchedule` | 流水线调度（支持算子间并行） |

---

## 3. Block Decomposition

### 3.1 Conv2d Block 分解

Conv2d 沿 batch 维度切分，每个 Block 处理一个或多个 batch 元素：

```
输入: [N=4, C=64, H=56, W=56]   输出: [N=4, OC=64, OH=56, OW=56]
           ↓                    ↑
  ┌────────┬────────┬────────┬────────┐
  │Block 0 │Block 1 │Block 2 │Block 3 │  ← 每个 Block 处理一个 batch
  │Batch 0 │Batch 1 │Batch 2 │Batch 3 │
  └────────┴────────┴────────┴────────┘
```

**分解算法**：

```rust
fn decompose_conv2d(input_shape, output_shape, num_threads) -> Vec<ComputeBlock> {
    let batch = input_shape[0];
    let elements_per_batch = out_channels * out_height * out_width;
    let total_elements = batch * elements_per_batch;

    // 计算每个 Block 的大小
    let block_size = (total_elements / num_threads).max(elements_per_batch);

    // 生成 Block
    for thread_id in 0..num_threads {
        blocks.push(ComputeBlock::new(
            node_id: thread_id,
            start_idx: thread_id * block_size,
            end_idx: min((thread_id + 1) * block_size, total_elements),
            workload: WorkloadEstimate::for_conv2d(...)
        ));
    }
}
```

### 3.2 MatMul Block 分解

MatMul 沿输出维度（M*N）切分：

```
Matrix A: [M=128, K=256]    Matrix B: [K=256, N=128]
              ↓                         ↓
     ┌─────────────────┬─────────────────┐
     │   Block 0       │   Block 1       │
     │   [0..64K)       │   [64K..128K)    │  ← 每个 Block 处理一部分输出
     └─────────────────┴─────────────────┘
                    ↓
              Output: [M=128, N=128]
```

### 3.3 工作量估计

```rust
pub struct WorkloadEstimate {
    /// 估计的 FLOPs
    pub flops: usize,
    /// 估计的内存占用（字节）
    pub memory_bytes: usize,
    /// 计算密度（FLOPs/byte）
    pub intensity: f32,
}

impl WorkloadEstimate {
    // Conv2d FLOPs: N * OC * OH * OW * IC * KH * KW * 2
    pub fn for_conv2d(batch, out_channels, out_height, out_width,
                      in_channels, kernel_h, kernel_w) -> Self { ... }

    // MatMul FLOPs: 2 * M * K * N
    pub fn for_matmul(m, k, n) -> Self { ... }
}
```

---

## 4. 负载均衡

### 4.1 Greedy 策略

将每个 Block 分配给当前负载最轻的 Worker：

```rust
fn greedy_assign(workloads, num_workers) -> Vec<WorkerAssignment> {
    let mut worker_loads = vec![0usize; num_workers];

    for (i, workload) in workloads.iter().enumerate() {
        // 找到负载最轻的 worker
        let min_worker = worker_loads.iter().enumerate()
            .min_by_key(|&(_, load)| load)
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        worker_loads[min_worker] += workload.flops;
        assignments.push(WorkerAssignment {
            block_id: i as u32,
            thread_id: min_worker as u32,
        });
    }
}
```

### 4.2 Round-Robin 策略

轮询分配 Block 到 Worker：

```rust
fn round_robin_assign(workloads, num_workers) -> Vec<WorkerAssignment> {
    workloads.iter().enumerate().map(|(i, _)| WorkerAssignment {
        block_id: i as u32,
        thread_id: (i % num_workers) as u32,
    }).collect()
}
```

---

## 5. CPU 亲和性

### 5.1 获取当前 CPU 亲和性

```rust
impl CpuAffinity {
    pub fn current() -> Self {
        #[cfg(target_os = "linux")]
        unsafe {
            let mut set = std::mem::MaybeUninit::<libc::cpu_set_t>::uninit();
            libc::CPU_ZERO(set.as_mut_ptr());

            if libc::pthread_getaffinity_np(libc::pthread_self(), size, set.as_mut_ptr()) == 0 {
                // 收集可用的 CPU 核心
                let mut cpus = Vec::new();
                for i in 0..libc::CPU_SETSIZE {
                    if libc::CPU_ISSET(i, &set) {
                        cpus.push(i);
                    }
                }
                CpuAffinity { cpus }
            } else {
                CpuAffinity { cpus: vec![] }
            }
        }

        #[cfg(not(target_os = "linux"))]
        CpuAffinity {
            cpus: (0..available_parallelism()).collect()
        }
    }
}
```

### 5.2 设置 CPU 亲和性

```rust
pub fn set_for_current(&self) -> Result<(), &'static str> {
    #[cfg(target_os = "linux")]
    unsafe {
        let mut set = std::mem::MaybeUninit::<libc::cpu_set_t>::uninit();
        libc::CPU_ZERO(set.as_mut_ptr());

        for &cpu in &self.cpus {
            libc::CPU_SET(cpu, set.as_mut_ptr());
        }

        let result = libc::pthread_setaffinity_np(libc::pthread_self(), size, set.as_mut_ptr());
        if result == 0 { Ok(()) } else { Err("Failed") }
    }

    #[cfg(not(target_os = "linux"))]
    Ok(())
}
```

---

## 6. 流水线并行

### 6.1 PipelineSchedule

支持算子间的流水线并行（类似 MNN 的调度方式）：

```
Graph: Conv1 → Conv2 → Conv3

调度方案:
  Stage 0: Thread1 → Conv1 (全部输出)
  Stage 1: Thread2 → Conv2 (block 1)
                  → Conv2 (block 2)  ← 流水线并行
  Stage 2: Thread3 → Conv3 (block 1)
```

### 6.2 实现

```rust
pub struct PipelineSchedule {
    num_stages: usize,
    stages: Vec<Vec<PipelineOp>>,
    dependencies: HashMap<usize, Vec<usize>>,
}

impl PipelineSchedule {
    pub fn new(num_stages: usize) -> Self { ... }

    pub fn add_op(&mut self, stage: usize, node_id: u32, workload: WorkloadEstimate) {
        self.stages[stage].push(PipelineOp { node_id, workload, ... });
    }

    pub fn add_dependency(&mut self, stage_a: usize, stage_b: usize) {
        // stage_a 必须先完成才能开始 stage_b
    }

    pub fn build(&self) -> Vec<Vec<ComputeBlock>> {
        // 将每个 Stage 的操作分解为 Block
    }
}
```

---

## 7. 使用示例

### 7.1 基本用法

```rust
use lightship_core::cpu::{ThreadScheduler, SchedulerConfig, OperatorType};

// 创建调度器
let scheduler = ThreadScheduler::new();

// 定义要调度的算子
let ops = vec![
    (0, OperatorType::Conv2d, vec![4, 64, 56, 56], vec![4, 64, 56, 56]),
    (1, OperatorType::Conv2d, vec![4, 64, 56, 56], vec![4, 64, 56, 56]),
];

// 调度并执行
scheduler.schedule_and_execute_ops(&ops, |node_id, blocks| {
    println!("Node {} executed with {} blocks", node_id, blocks.len());
});
```

### 7.2 自定义配置

```rust
let scheduler = ThreadScheduler::with_config(SchedulerConfig {
    num_threads: 8,
    enable_parallel: true,
    max_block_size: 32 * 1024,  // 32K 元素 per block
    use_affinity: true,
    load_balance: LoadBalanceConfig::Greedy,
});
```

### 7.3 Block 级执行

```rust
let blocks = BlockScheduler::decompose_conv2d(
    &[4, 64, 56, 56],
    &[4, 64, 56, 56],
    4
);

// 并行执行每个 Block
scheduler.execute_blocks(&blocks, |block| {
    // 处理 block.start_idx .. block.end_idx
    compute_conv_block(block);
});
```

---

## 8. 性能特性

### 8.1 Block 大小选择

| 场景 | 推荐 Block 大小 |
|------|----------------|
| 小 batch（1-2） | 整个 tensor 为一个 Block |
| 大 batch（4+） | 按 batch 均匀切分 |
| MatMul | 按 M*N 均匀切分 |

### 8.2 线程数选择

- 默认使用 `std::thread::available_parallelism()` 自动检测
- 可通过 `SchedulerConfig::num_threads` 手动设置
- 建议设置为物理核心数（非逻辑核心）

### 8.3 负载均衡效果

| 策略 | 适用场景 |
|------|----------|
| Greedy | 异构 workload（如不同大小的 Conv） |
| RoundRobin | 同构 workload（如相同的 Conv 链） |

---

## 9. 测试验证

```bash
# 运行所有 thread_scheduler 测试
cargo test --release --package lightship-core --lib thread_scheduler

# 运行结果
test cpu::thread_scheduler::tests::test_block_creation ... ok
test cpu::thread_scheduler::tests::test_block_decompose_conv2d ... ok
test cpu::thread_scheduler::tests::test_block_decompose_matmul ... ok
test cpu::thread_scheduler::tests::test_cpu_affinity_current ... ok
test cpu::thread_scheduler::tests::test_load_balancing_greedy ... ok
test cpu::thread_scheduler::tests::test_scheduler_config ... ok
test cpu::thread_scheduler::tests::test_thread_pool_creation ... ok
test cpu::thread_scheduler::tests::test_thread_pool_custom_threads ... ok
test cpu::thread_scheduler::tests::test_workload_conv2d ... ok
test cpu::thread_scheduler::tests::test_workload_matmul ... ok
test cpu::thread_scheduler::tests::test_pipeline_schedule ... ok

test result: ok. 11 passed; 0 failed
```

---

## 10. 与现有模块集成

### 10.1 与 GraphExecutor 集成

`thread_scheduler` 可与 `executor/graph_executor.rs` 集成，在拓扑层级并行基础上增加 Block 级并行：

```rust
// 在 GraphExecutor::execute_parallel 中
for level in levels {
    // 1. 收集同层算子
    let level_ops: Vec<_> = level.iter()
        .map(|&node_id| prepare_operator(node_id))
        .collect();

    // 2. 使用 ThreadScheduler 进行 Block 分解和并行执行
    scheduler.schedule_and_execute_ops(&level_ops, |node_id, blocks| {
        // 执行算子的各个 Block
        execute_blocks(node_id, blocks);
    });
}
```

### 10.2 与 CpuBackend 集成

`CpuBackend` 可使用 `ThreadScheduler` 来并行执行 Conv2d/MatMul：

```rust
impl CpuBackend {
    fn execute_conv2d_parallel(&self, inputs, outputs) -> Result<()> {
        let scheduler = ThreadScheduler::new();
        let blocks = BlockScheduler::decompose_conv2d(...);

        scheduler.execute_blocks(&blocks, |block| {
            self.compute_conv_block(block, inputs, outputs);
        });

        Ok(())
    }
}
```

---

## 11. 参考资料

- [Rayon Documentation](https://docs.rs/rayon/)
- [Linux CPU Affinity](https://man7.org/linux/man-pages/man3/CPU_SET.3.html)
- [MNN Multi-thread Scheduling](https://github.com/alibaba/MNN/blob/master/source/backend/cpu/CPUBackend.cpp)
