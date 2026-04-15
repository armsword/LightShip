# GEMM 分块优化原理

## 1. 概述

本文档描述 LightShip 中 GEMM（通用矩阵乘法）的分块优化实现。GEMM 是深度学习中最核心的计算操作，几乎所有算子（Conv2d、MatMul、FC 等）最终都依赖于高效的矩阵乘法实现。

## 2. 问题背景

### 2.1 朴素 GEMM 实现

传统的三层循环 GEMM 实现：

```rust
for i in 0..m {
    for j in 0..n {
        for p in 0..k {
            c[i*n + j] += a[i*k + p] * b[p*n + j];
        }
    }
}
```

问题：
- **缓存不友好**：A 按行访问（连续），B 按列访问（跳转）
- **无法利用 SIMD**：每次只计算一个输出元素
- **内存带宽瓶颈**：CPU 大部分时间在等待数据加载

### 2.2 分块优化原理

将大矩阵划分为 cache-friendly 的小块：

```
┌─────────────────┬─────────────────┐
│                 │                 │
│   Block 0       │   Block 1       │
│   (64x64)       │   (64x64)       │
│                 │                 │
├─────────────────┼─────────────────┤
│                 │                 │
│   Block 2       │   Block 3       │
│   (64x64)       │   (64x64)       │
│                 │                 │
└─────────────────┴─────────────────┘
```

每个小块可以完全放入 L1/L2 缓存，显著减少缓存未命中。

## 3. 分块策略

### 3.1 三层分块架构

```
Level 1 (MC x KC): A 的行块，B 的 K 块
Level 2 (KC x NC): B 的 N 块，K 循环展开
Level 3 (Register): SIMD 寄存器级别的向量操作
```

### 3.2 块大小选择

| 参数 | 值 | 说明 |
|------|-----|------|
| MC | 64 | M 方向块大小（64 行），约 64 * K * 4 bytes，适合 L1 DCache |
| KC | 256 | K 方向块大小（256），平衡缓存效率和向量化的 K 展开 |
| NC | 128 | N 方向块大小（128 列） |
| NR | 8/16 | AVX2/AVX512 每次处理的列数 |

### 3.3 数据排布优化（B Packing）

关键优化：将 B 的 KC x NR 块重新排列为连续内存：

```
原始 B 布局 (stride = n):
B[k, j] 访问模式: k 递增（跨 stride），j 连续

Packed B 布局:
packed_b[p * NR + j] = B[k_block + p, n_block + j]
                        └──p 方向连续──┘ └──j 方向连续──┘
```

好处：
- 加载 B 时是连续内存访问
- 充分利用 SIMD 加载指令
- 减少缓存带宽压力

## 4. AVX2 实现详解

### 4.1 寄存器使用

```asm
; AVX2 256-bit = 8 float32
; 每个累加器 acc[j] 存储 8 列的中间结果

; 初始化
vxorps ymm0, ymm0, ymm0    ; acc[0] = 0
vxorps ymm1, ymm1, ymm1    ; acc[1] = 0
; ... (共 8 个累加器)
```

### 4.2 K 循环展开

```rust
// 每次迭代处理 4 个 K 值，产生 32 次 FMA（4 K × 8 列）
while kk + 4 <= k_len {
    // 加载 A 的 4 个值并广播
    let a0 = _mm256_set1_ps(A[i, kk + 0]);
    let a1 = _mm256_set1_ps(A[i, kk + 1]);
    let a2 = _mm256_set1_ps(A[i, kk + 2]);
    let a3 = _mm256_set1_ps(A[i, kk + 3]);

    // 加载 B 的 4 个块（每块 8 floats）
    let b0 = _mm256_loadu_ps(&packed_b[kk * NR]);
    let b1 = _mm256_loadu_ps(&packed_b[(kk+1) * NR]);
    let b2 = _mm256_loadu_ps(&packed_b[(kk+2) * NR]);
    let b3 = _mm256_loadu_ps(&packed_b[(kk+3) * NR]);

    // FMA 融合乘加
    acc[0] = _mm256_fmadd_ps(a0, b0, acc[0]);
    acc[0] = _mm256_fmadd_ps(a1, b1, acc[0]);
    acc[0] = _mm256_fmadd_ps(a2, b2, acc[0]);
    acc[0] = _mm256_fmadd_ps(a3, b3, acc[0]);

    kk += 4;
}
```

### 4.3 预取策略

```rust
// 预取 A 的下一行
_mm_prefetch(A.as_ptr().add(a_row_base + kk + 8) as *const f32, _MM_HINT_T0);

// 预取 B 的后续行
if p % 8 == 0 && b_row + 16 < k {
    _mm_prefetch(B.as_ptr().add((b_row + 16) * n + n_block) as *const f32, _MM_HINT_T0);
}
```

- `_MM_HINT_T0`: 预取到 L1 cache
- 每 8 次迭代预取一次，避免过于频繁的预取开销

## 5. 性能优化技术

### 5.1 软件流水线

通过预取实现软件流水线，隐藏内存访问延迟：

```
当前迭代使用的数据    ←←← 预取
上一次迭代预取的数据 ←←← 已加载到缓存
```

### 5.2 指令级并行

K 循环展开产生多个独立的 FMA 操作，CPU 可以：
- 乱序执行
- 发射多个内存操作
- 利用执行单元流水线

### 5.3 缓存层级利用

```
L1 Cache: ~32KB, 访问 ~4 cycles
L2 Cache: ~256KB, 访问 ~12 cycles
L3 Cache: ~8MB, 访问 ~40 cycles
Main Memory: ~200 cycles
```

通过 MC=64, KC=256 的块大小，确保大部分数据访问落在 L1/L2 cache。

## 6. AVX-512 优化

AVX-512 每次处理 16 个 float32，寄存器宽度翻倍：

```rust
// AVX-512: 512-bit = 16 float32
// 单一累加器处理 16 列

let mut acc = _mm512_setzero_ps();

while kk + 4 <= k_len {
    let a0 = _mm512_set1_ps(A[i, kk + 0]);
    let b0 = _mm512_loadu_ps(&packed_b[kk * NR]);

    acc = _mm512_fmadd_ps(a0, b0, acc);
    // ...
}

// 使用 AVX-512 专有指令进行水平求和
let result = _mm512_reduce_add_ps(acc);
```

`_mm512_reduce_add_ps` 是 AVX-512 新增的指令，可以高效地完成向量水平求和。

## 7. 性能对比

### 理论分析

| 实现 | 每周期理论吞吐 |
|------|---------------|
| 朴素 3-loop | ~2-4 FLOPs |
| SIMD 单核 | ~16 FLOPs (AVX2) |
| 分块 + SIMD | ~32 FLOPs (利用缓存) |

### 实测预期

对于 512x512x512 矩阵乘法：
- 朴素实现：~200-300 ms
- 分块优化：~50-80 ms
- **预期提升：2-3x**

## 8. 代码结构

```
lightship-core/src/platform/simd.rs
├── gemm_avx2()     - AVX2 分块实现
├── gemm_avx512()   - AVX-512 分块实现  
├── gemm_sse()      - SSE 分块实现
└── gemm_neon()     - ARM NEON 分块实现
```

### 8.1 核心数据结构

```rust
const NR: usize = 8;    // AVX2: 8 floats per register
const KC: usize = 256;  // K blocking for cache

let mut packed_b = vec![0.0f32; KC * NR];
```

### 8.2 主循环结构

```rust
for m_block in (0..m).step_by(MC) {
    for n_block in (0..n).step_by(NC) {
        for k_block in (0..k).step_by(KC) {
            // 1. Pack B block
            pack_b_block(&mut packed_b, b, k_block, n_block, k_len, nr_cur);

            // 2. Compute C block
            for i in m_block..m_end {
                // 初始化累加器
                // K 循环展开
                // 水平求和并存储
            }
        }
    }
}
```

## 9. 未来优化方向

1. **多线程并行**：将 M 方向分块分配给不同线程
2. **NN/PB 格式支持**：添加 TF 风格的 filter 格式支持
3. **混合精度**：int8/fp16 GEMM 支持
4. **Winograd 优化**：针对 Conv2d 的 Winograd 快速算法

## 10. 参考资料

- [1] Goto, K., & Van De Geijn, R. (2008). Anatomy of high-performance matrix multiplication.
- [2] Intel SGEMM Implementation: https://github.com/intel/mkl-dnn
- [3] OpenBLAS gemm implementation
- [4] MNN GEMM optimization paper