# LightShip vs MNN CPU 后端对比分析

## 1. 概述

| 维度 | LightShip | MNN |
|------|-----------|-----|
| **开源时间** | 2024 | 2019 (阿里) |
| **语言** | Rust | C++ |
| **定位** | 轻量级边缘推理引擎 | 面向移动端的高性能引擎 |
| **代码规模** | ~50K LOC | ~100K LOC |

## 2. Conv2d 实现对比

### 2.1 算法策略

| 特性 | LightShip | MNN |
|------|-----------|-----|
| **主要实现** | Im2col + GEMM | Winograd + Im2col+GEMM |
| **3x3卷积** | Im2col + GEMM | Winograd (加速4x) |
| **大卷积核** | Im2col + GEMM | Im2col + GEMM |
| **分组卷积** | 支持 | 支持 |
| **深度卷积优化** | 基本实现 | Winograd + 专用优化 |

### 2.2 LightShip 当前实现

```rust
// convolution.rs - Im2col + GEMM 模式
pub fn forward(&self, input: &Tensor, filter: &Tensor) -> Result<Tensor> {
    // 1. Im2col 转换: 将滑动窗口展平为矩阵列
    let col_matrix = self.im2col(&input_data, ...);

    // 2. Filter reshape: [oc, ic, kh, kw] -> [oc, kernel_size]

    // 3. GEMM: output = filter_matrix @ col_matrix.T
    gemm_simd(&filter_matrix, &col_transposed, &mut output_slice, ...);
}
```

**特点**：
- 通用性强，支持任意kernel size
- 内存开销较大（需要存储im2col展开的矩阵）
- 依赖GEMM的SIMD优化

### 2.3 MNN Winograd 实现

MNN 对 3x3 卷积使用 Winograd 算法，将卷积转换为更少的乘法运算：

```
Winograd(2x2 输出):
- 原始: 9次乘法 (3x3 kernel * 2x2 input)
- Winograd: 4次乘法 + 一些加法
- 加速比: ~2.25x
```

MNN 还使用 **NC4HW4** 内存布局优化，对齐到4通道或8通道。

## 3. SIMD 优化对比

### 3.1 支持的指令集

| 指令集 | LightShip | MNN |
|--------|-----------|-----|
| SSE | ✅ | ✅ |
| SSE2/3/4 | ✅ | ✅ |
| AVX | ✅ | ✅ |
| AVX2 | ✅ | ✅ |
| AVX-512 | ✅ | ✅ |
| NEON | ✅ | ✅ |
| ARM VFP | ❌ | ✅ |
| MSA (MIPS) | ❌ | ✅ |

### 3.2 GEMM 优化

**LightShip 当前实现** (simd.rs):

```rust
// 朴素的三层循环GEMM，使用SIMD加速
unsafe fn gemm_avx2(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    let nr = 8;  // 每个内循环处理8列
    for m_idx in 0..m {
        for n_idx in (0..n).step_by(nr) {
            let mut sum = [_mm256_setzero256(); 8];
            for k_idx in 0..k {
                let a_val = _mm256_set1_ps(a[m_idx * k + k_idx]);
                // ...
                sum[j] = _mm256_fmadd_ps(a_val, b_val, sum[j]);
            }
        }
    }
}
```

**MNN 优化策略**:
1. **分块 (Blocking)**: 将矩阵划分为cache-friendly的块
2. **寄存器重排**: 优化数据布局提高缓存命中率
3. **预取 (Prefetch)**: 提前加载数据减少内存延迟
4. **多线程GEMM**: 细粒度并行化

### 3.3 算子级优化

| 算子 | LightShip | MNN |
|------|-----------|-----|
| ReLU | ✅ SIMD max | ✅ SIMD max |
| ReLU6 | ✅ SIMD clamp | ✅ SIMD clamp |
| Sigmoid | ✅ exp实现 | ✅ 查表+多项式 |
| Tanh | ✅ exp实现 | ✅ 查表+多项式 |
| Softmax | ✅ 数值稳定 | ✅ 数值稳定+SIMD |
| BatchNorm | ✅ | ✅ |
| Add/Mul | ✅ SIMD | ✅ SIMD fusion |

## 4. 内存管理对比

### 4.1 LightShip 内存策略

```rust
// cpu.rs - 简单的内存分配
fn allocate(&self, size: usize, alignment: usize) -> Result<MemoryBlock> {
    let layout = unsafe { Layout::from_size_align_unchecked(size, alignment) };
    let ptr = unsafe { alloc(layout) };
    // ...
}
```

**特点**:
- 使用标准 `std::alloc`
- 每次分配/释放都是系统调用
- 无内存池
- 无复用策略

### 4.2 MNN 内存优化

MNN 实现了 **三级内存池**：

```
┌─────────────────────────────────────┐
│         Runtime Memory Pool        │
├─────────────────────────────────────┤
│  Thread-Local Memory Pool (TLS)    │
├─────────────────────────────────────┤
│      Global Memory Pool            │
└─────────────────────────────────────┘
```

1. **Global Pool**: 所有线程共享，分配大块内存
2. **TLS Pool**: 每个线程独立，减少锁竞争
3. **Memory Reuse**: 同一图中的中间张量复用内存

## 5. 线程调度对比

### 5.1 LightShip

当前实现为**单线程执行**，多线程仅用于SIMD并行：

```rust
// cpu.rs
fn execute(&self, op: &CompiledOperator, inputs: &[&Tensor], outputs: &mut [&mut Tensor]) -> Result<()> {
    match op.operator_type {
        OperatorType::Conv2d => self.execute_conv2d(inputs, outputs),
        // ...
    }
}
```

**问题**: 算子间无并行化，依赖SIMD的指令级并行

### 5.2 MNN 多线程调度

MNN 使用 **Block-wise 调度**:

```
Graph:
  Conv1 → Conv2 → Conv3

MNN 调度:
  ┌─────────┐
  │ Thread1 │ → Conv1 (全部输出)
  │ Thread2 │ → Conv2 (block 1)
  │ Thread2 │ → Conv2 (block 2)  ← 流水线并行
  │ Thread3 │ → Conv3 (block 1)
  └─────────┘
```

**特点**:
- 计算密度高的算子（Conv、MatMul）自动切分
- 细粒度负载均衡
- 支持 CPU亲和性 设置

## 6. 准确性问题

### 6.1 数值精度对比

| 测试 | LightShip | MNN | 差异原因 |
|------|-----------|-----|---------|
| ReLU | ✅ 精确 | ✅ 精确 | 无差异 |
| Softmax (大输入) | ✅ 数值稳定 | ✅ 数值稳定 | 减最大值 |
| Sigmoid | ⚠️ exp精度 | ⚠️ 查表精度 | 近似方法不同 |
| Tanh | ⚠️ exp精度 | ⚠️ 查表精度 | 近似方法不同 |
| GEMM | ✅ 累积精度 | ✅ 累积精度 | FMA指令 |

### 6.2 LightShip 当前问题

```rust
// exp实现依赖std::exp，精度高但可能不是最优
fn exp_scalar(input: &[f32], output: &mut [f32], len: usize) {
    for i in 0..len {
        output[i] = input[i].exp();  // 使用libm的exp
    }
}
```

MNN 可能使用 **多项式逼近** 或 **查表法**，牺牲精度换取性能。

## 7. 性能优化空间

### 7.1 LightShip 当前瓶颈

1. **内存分配**: 每次推理都分配/释放内存
2. **无算子融合**: Conv+ReLU 分离执行，增加内存访问
3. **单线程执行**: 无法利用多核并行
4. **Im2col开销**: 大内存占用和拷贝开销
5. **GEMM未优化**: 未使用分块、预取等优化

### 7.2 MNN 性能优化技术

| 技术 | 说明 | LightShip当前状态 |
|------|------|------------------|
| Winograd | 3x3卷积加速2x+ | ❌ 未实现 |
| Im2Col优化 | 内存布局优化 | ⚠️ 基础实现 |
| 内存池 | 减少分配开销 | ❌ 未实现 |
| 算子融合 | 减少内存访问 | ❌ 未实现 |
| 多线程调度 | 算子间并行 | ❌ 单线程 |
| GEMM分块 | 缓存友好 | ⚠️ 基础实现 |
| 量化推理 | Int8/FP16加速 | ⚠️ 框架在，kernel未实现 |

## 8. 总结与建议

### 8.1 性能差距预估

| 场景 | LightShip | MNN | 差距 |
|------|-----------|-----|------|
| 小模型 (<1M参数) | 基准 | 基准 | ~1x |
| ResNet18 (CPU单线程) | 基准 | 基准 | ~3-5x |
| ResNet18 (CPU多线程) | 基准 | 基准 | ~5-10x |
| 3x3 Conv特定优化 | 基准 | 基准 | ~2-3x |

### 8.2 优先优化项

1. **高优先级**:
   - 实现内存复用池
   - 添加算子融合 (Conv+ReLU)
   - 多线程调度框架

2. **中优先级**:
   - Winograd算法 (3x3卷积)
   - GEMM分块优化
   - 量化推理kernel

3. **低优先级**:
   - ARM VFP支持
   - 更多SIMD指令集优化

### 8.3 准确性保障

LightShip 使用 `std::exp` 等标准库函数，数值精度**高于**使用查表法的实现，但性能可能略低。这是正确的工程权衡，建议保持。

## 9. 参考资料

- MNN GitHub: https://github.com/alibaba/MNN
- MNN Paper: "MNN: A Universal and Efficient Inference Engine"
- LightShip 内部代码实现
