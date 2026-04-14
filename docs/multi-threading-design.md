# LightShip 多线程执行设计原理

## 1. 概述

本文档描述 LightShip CPU 后端多线程执行的实现原理，包括：
1. Conv2d 算子的 batch 维度并行化
2. GraphExecutor 的拓扑层级并行执行

**目标**：利用多核 CPU 提升推理性能，缩小与 MNN 的差距（当前单线程 Conv2d 11.8ms vs MNN ~2ms）。

---

## 2. Conv2d 多线程实现

### 2.1 并行策略

Conv2d 计算中，不同 batch 元素之间完全独立：

```
输入: [N, C, H, W]   输出: [N, OC, OH, OW]
           ↓                    ↑
  ┌────────┬────────┬─────────┐
  │Batch 0 │Batch 1 │  ... N-1 │  ← 各 batch 独立，可并行
  └────────┴────────┴─────────┘
```

**关键约束**：
- 输出缓冲区分片：每个线程写入 `output[n * out_c * oh * ow .. (n+1) * out_c * oh * ow - 1]`
- 写操作不重叠，无需同步（除最终的 Arc::clone）
- 组卷积（groups）内部：同一 batch 内 group 之间无数据依赖，可进一步并行

### 2.2 实现：`std::thread::scope`

使用 Rust 1.63+ 的 `std::thread::scope` 而非手动 thread pool：

```rust
std::thread::scope(|s| {
    for (n_idx, batch_out) in chunks.into_iter().enumerate() {
        s.spawn(move || {
            Self::compute_batch_element(n_idx, batch_out, ...);
        });
    }
});
```

**为什么用 `scope`**：
| 方案 | 优点 | 缺点 |
|------|------|------|
| `scope` | 无需 `Arc<Mutex>`，栈式借阅，零成本抽象 | 需稳定 Rust |
| `Arc<Mutex<Vec>>` | 兼容旧版 Rust | 原子操作开销，代码复杂 |
| `rayon` | 简洁 | 引入外部依赖 |

### 2.3 内存布局与 GEMM

```
batch_out: [out_c * oh * ow]  (NCHW 格式)

线程 n 处理:
  for group in 0..groups:
    1. im2col: 为当前 batch 构建 [kernel_size, oh*ow] 矩阵
    2. GEMM:   [c_out_per_group, oh*ow] = filter_matrix @ col_T
    3. scatter: 结果写入 batch_out 的对应 channel 区间
```

### 2.4 im2col 正确性修复

原有内联代码有 padding bug（未减 pad_h/pad_w），修复为：

```rust
let in_h_idx = (oh * stride_h + kh * dilation_h).wrapping_sub(pad_h);
let in_w_idx = (ow * stride_w + kw * dilation_w).wrapping_sub(pad_w);
let valid = in_h_idx < in_h && in_w_idx < in_w;
```

---

## 3. GraphExecutor 并行执行

### 3.1 拓扑层级并行

神经网络计算图具有 DAG 结构：同一层（level）的节点无依赖，可并行执行。

```
Graph:
  input1 → ReLU1 ──┐
                   ├──→ Add → output
  input2 → ReLU2 ──┘

拓扑层级:
  Level 0: [ReLU1, ReLU2]  ← 可并行执行
  Level 1: [Add]           ← 等待 Level 0 完成
```

### 3.2 compute_levels 实现

基于 Kahn's 算法的 BFS 层分解：

```rust
fn compute_levels(&self, graph: &Graph) -> Vec<Vec<NodeId>> {
    let mut in_degree = vec![0u32; n];
    let mut adjacency: HashMap<NodeId, Vec<NodeId>> = HashMap::new();

    // 构建邻接表和入度
    for node in &graph.nodes {
        for input in &node.inputs {
            if let Some(producer) = graph.find_tensor_producer(&input.tensor_name) {
                adjacency.entry(producer).or_default().push(node.id);
                in_degree[node.id as usize] += 1;
            }
        }
    }

    // BFS 分层
    let mut levels: Vec<Vec<NodeId>> = Vec::new();
    let mut current: Vec<NodeId> = (0..n as NodeId)
        .filter(|&id| in_degree[id as usize] == 0)
        .collect();

    while !current.is_empty() {
        levels.push(current.clone());
        let mut next = Vec::new();
        for &node_id in &current {
            for &dep in adjacency.get(&node_id).unwrap_or(&vec![]) {
                in_degree[dep as usize] -= 1;
                if in_degree[dep as usize] == 0 {
                    next.push(dep);
                }
            }
        }
        current = next;
    }
    levels
}
```

### 3.3 execute_parallel 实现

```
execute_parallel(graph, inputs, outputs):
  1. 初始化 tensor_storage（HashMap<String, Arc<Tensor>>）
  2. 拓扑分层 → levels: Vec<Vec<NodeId>>
  3. for each level:
       a. 收集 level 内所有节点的输入（Arc::clone）
       b. std::thread::scope 并行执行所有节点
       c. join 所有线程
       d. 合并结果到 tensor_storage
```

**线程安全保证**：
- `tensor_storage` 是 `HashMap<String, Arc<Tensor>>`，读操作 Arc clone 无成本
- 同一 level 内节点写入不同的 `tensor_name` key，无写冲突
- `output_tensors` 在各线程栈上分配，线程间无共享可变状态

### 3.4 execute_node_single 隔离执行

每个节点在独立线程执行，后端通过 `&self` 共享：

```rust
fn execute_node_single(
    node: &Node,
    compiled: &CompiledOperator,
    inputs: Vec<&Tensor>,
) -> Result<Vec<(String, Arc<Tensor>)>> {
    let backend: Arc<dyn Backend + Send + Sync> = Arc::new(CpuBackend::new());
    let mut output_tensors: Vec<Tensor> = node.outputs.iter().map(|o| {
        Tensor::new(o.tensor_name.clone(), vec![1], o.data_type)
    }).collect();
    let mut output_refs: Vec<&mut Tensor> = output_tensors.iter_mut().collect();
    backend.execute(compiled, &inputs, &mut output_refs)?;
    Ok(output_tensors.into_iter().map(|t| (t.name.clone(), Arc::new(t))).collect())
}
```

---

## 4. 性能分析

### 4.1 Conv2d batch 并行

| Batch Size | 理论加速比 | 备注 |
|-----------|-----------|------|
| N=1 | 1x | 无并行收益 |
| N=4 | ~2-3x | 4 核 |
| N=8 | ~4-6x | 8 核（含 GEMM SIMD）|

### 4.2 GraphExecutor 层并行

| 场景 | 收益 |
|------|------|
| 单层单节点 | 无收益 |
| 单层多节点（如 split/concat 并行分支）| 线性加速 |
| 多层流水线 | 无额外加速（层间串行）|

### 4.3 当前瓶颈

1. **每节点创建新 CpuBackend**：`execute_node_single` 每次调用 `CpuBackend::new()`，有微小开销
2. **GEMM 未分块**：当前为朴素三层循环 + SIMD，未使用 cache-friendly blocking
3. **Winograd 未实现**：3x3 卷积可额外加速 2x+

---

## 5. 后续优化方向

### 5.1 高优先级

- **CpuBackend 复用**：将 backend 传给 `execute_node_single` 而非每次新建
- **Conv2d group 内并行**：同一 batch 内不同 group 可并行（需更细粒度同步）

### 5.2 中优先级

- **Winograd 算法**：3x3 卷积专用，减少乘法次数 2.25x
- **GEMM 分块**：cache-friendly blocking，参考 MNN/NCNN

### 5.3 低优先级

- **Rayon 集成**：用 rayon 替代手写 thread scope，代码更简洁
- **CPU Affinity**：将线程绑定到特定物理核心，减少跨 NUMA 访问

---

## 6. 测试验证

| 测试 | 目的 |
|------|------|
| `test_conv2d_batch_parallel` | 验证 batch=4 时多线程结果正确 |
| `test_conv2d_basic` | 单 batch 基准测试 |
| `test_conv2d_groups` | 组卷积正确性 |
| `test_graph_executor_parallel_level` | 拓扑层级并行正确性 |
| `test_graph_executor_single_relu` | 单节点基准 |

---

## 7. 参考资料

- Rust `std::thread::scope`: <https://doc.rust-lang.org/std/thread/fn.scope.html>
- Kahn's Algorithm: <https://en.wikipedia.org/wiki/Topological_sorting#Kahn's_algorithm>
- MNN Multi-thread: <https://github.com/alibaba/MNN/blob/master/source/backend/cpu/CPUBackend.cpp>
