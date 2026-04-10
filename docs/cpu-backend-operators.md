# LightShip CPU 后端算子执行实现

## 1. 概述

CPU 后端是 LightShip 推理引擎的默认计算后端，负责在 CPU 上执行神经网络算子。本文档记录已实现的算子执行逻辑和 SIMD 优化。

## 2. Backend Trait

### 2.1 Execute 方法签名

```rust
fn execute(
    &self,
    op: &CompiledOperator,
    inputs: &[&Tensor],
    outputs: &mut [&mut Tensor],  // 每个元素是 &mut Tensor
) -> Result<()>;
```

## 3. SIMD 加速

### 3.1 SIMD 级别检测

运行时自动检测 CPU 支持的最高 SIMD 级别：

```rust
pub fn detect_simd_level() -> SimdLevel {
    // x86_64: AVX512 > AVX2 > AVX > SSE4.2 > SSE2
    // ARM64: NEON
    // 其他: None
}
```

### 3.2 支持的 SIMD 指令集

| 平台 | 最高级别 | 矢量宽度 |
|------|---------|---------|
| x86_64 | AVX-512 | 512-bit (16 x f32) |
| x86_64 | AVX2 | 256-bit (8 x f32) |
| x86_64 | AVX | 256-bit (8 x f32) |
| x86_64 | SSE2 | 128-bit (4 x f32) |
| ARM64 | NEON | 128-bit (4 x f32) |

## 4. 已实现算子（SIMD 加速）

### 4.1 ReLU

**公式**: `f(x) = max(x, 0)`

**SIMD 实现**: 使用 `relu_simd` 函数，自动选择最优指令

```rust
fn execute_relu(&self, inputs: &[&Tensor], outputs: &mut [&mut Tensor]) -> Result<()> {
    // 转换为 f32 数组
    let input_f32: Vec<f32> = input_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let mut output_f32 = vec![0.0f32; num_elements];

    // SIMD 加速
    let simd_level = detect_simd_level();
    relu_simd(&input_f32, &mut output_f32, simd_level);
}
```

### 4.2 ReLU6

**公式**: `f(x) = clamp(x, 0, 6)`

**SIMD 实现**: 使用 `relu6_simd` 函数

### 4.3 Add (元素级加法)

**公式**: `c[i] = a[i] + b[i]`

**SIMD 实现**: 使用 `add_simd` 函数，并行处理多个元素

### 4.4 Sub (元素级减法)

**公式**: `c[i] = a[i] - b[i]`

**SIMD 实现**: 使用 `sub_simd` 函数

### 4.5 Mul (元素级乘法)

**公式**: `c[i] = a[i] * b[i]`

**SIMD 实现**: 使用 `mul_simd` 函数

### 4.6 Div (元素级除法)

**公式**: `c[i] = a[i] / b[i]`

**SIMD 实现**: 使用 `div_simd` 函数（AVX512/AVX2/AVX/SSE/NEON）

### 4.7 Sigmoid

**公式**: `sigmoid(x) = 1 / (1 + exp(-x))`

**SIMD 实现**:
```rust
// neg_x = -x (标量)
neg_x[i] = -x[i];

// exp(-x) 使用 exp_simd
exp_simd(&neg_x, &mut exp_neg_x, simd_level);

// sigmoid = 1 / (1 + exp(-x))
sigmoid[i] = 1.0 / (1.0 + exp_neg_x[i]);
```

### 4.8 Tanh

**公式**: `tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))`

**SIMD 实现**: 利用已有的 `exp_simd` 函数计算 `exp(x)` 和 `exp(-x)`

### 4.9 MatMul (矩阵乘法)

**公式**: `C = A * B`，其中 A:[M,K] B:[K,N] C:[M,N]

**SIMD 实现**: 使用 `gemm_simd` 函数
- AVX512: 16元素并行
- AVX2: 8元素并行
- SSE/NEON: 4元素并行

### 4.10 Softmax

**公式**: `softmax[i] = exp(x[i]) / sum(exp(x[j]))`

**SIMD 实现**:
```rust
// 找最大值用于数值稳定
let max_val = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

// exp(x - max) 使用 SIMD
exp_simd(&shifted, &mut exp_values, simd_level);

// 求和使用 horizontal_sum
let sum_exp = horizontal_sum(&exp_values, simd_level);

// 除以 sum 使用 div_scalar_simd
div_scalar_simd(&exp_values, &mut output, sum_exp, simd_level);
```

### 4.11 MaxPool2d

**使用**: `Pool2d::max_pool2d_simd`

**SIMD 实现**: 2x2 窗口 SIMD 优化

### 4.12 AvgPool2d

**使用**: `Pool2d::avg_pool2d`

## 5. 数据布局

### 5.1 字节顺序

所有数据使用 **小端序 (Little Endian)** 编码：
- f32 值 `1.0` 存储为字节 `[0, 0, 128, 63]`

### 5.2 内存布局

```
TensorData::Owned(bytes)
    ├── Vec<u8> 存储原始字节
    ├── 每个 f32 占 4 字节
    └── 通过 chunks_exact(4) 迭代处理
```

## 6. 测试覆盖

| 测试文件 | 测试数 |
|---------|-------|
| cpu_backend_test | 25 |
| simd_benchmark_test | 5 |

运行测试：
```bash
cargo test -p lightship-core --test cpu_backend_test
cargo test -p lightship-core --test simd_benchmark_test
```

## 7. 性能基准

典型操作延迟（MacBook Pro M1, 1M 元素）：

| 算子 | 延迟 |
|------|------|
| ReLU (1M) | < 1ms |
| MatMul (128x256 @ 256x128) | < 5ms |
| Conv2d (1x3x32x32 @ 16x3x3x3) | < 10ms |
| Softmax (1K) | < 0.5ms |

## 8. 后续优化方向

1. **AvgPool2d SIMD**: 目前使用标量实现
2. **多线程**: 大张量使用线程池并行
3. **内存预分配**: 避免算子执行时频繁分配
4. **Fusion**: Conv+ReLU, Conv+BN 融合执行
