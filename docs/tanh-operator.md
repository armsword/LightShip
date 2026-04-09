# Tanh 算子原理与实现

## 1. 概述

Tanh（双曲正切）是一种常用的激活函数，输出范围为(-1, 1)，相比Sigmoid更适合处理有正负值的数据。

**公式：**
```
tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
```

**特性：**
- 输出范围：(-1, 1)
- 零中心（zero-centered）
- 平滑非线性
- 梯度在原点处为1

## 2. 数学原理

### 2.1 基本恒等式

```
tanh(x) = sinh(x) / cosh(x)
        = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
```

### 2.2 与Sigmoid的关系

```
tanh(x) = 2 * sigmoid(2x) - 1
```

**推导：**
```
sigmoid(z) = 1 / (1 + exp(-z))

tanh(x) = 2 * sigmoid(2x) - 1
        = 2 / (1 + exp(-2x)) - 1
        = (2 - (1 + exp(-2x))) / (1 + exp(-2x))
        = (1 - exp(-2x)) / (1 + exp(-2x))
        = (exp(2x) - 1) / (exp(2x) + 1)
        = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
```

### 2.3 导数

```
d/dx tanh(x) = sech²(x) = 1 - tanh²(x)
```

**特性：**
- 梯度范围：(0, 1]
- 在x=0处梯度最大为1
- 随|x|增大梯度减小

## 3. 数值稳定性

### 3.1 直接计算的问题

对于大的正数x：
```
exp(x) - exp(-x) ≈ exp(x)  (exp(-x) → 0)
exp(x) + exp(-x) ≈ exp(x)
tanh(x) ≈ 1
```

对于大的负数x：
```
exp(x) - exp(-x) ≈ -exp(-x)  (exp(x) → 0)
exp(x) + exp(-x) ≈ exp(-x)
tanh(x) ≈ -1
```

### 3.2 边界处理

```rust
if z > 20.0 {  // z = 2x
    return 1.0;
}
if z < -20.0 {
    return -1.0;
}
```

## 4. SIMD优化实现

### 4.1 计算流程

```
Input: [x0, x1, x2, ..., xn]

Step 1: mul_scalar_simd (z = 2 * x)
Output: [2x0, 2x1, 2x2, ..., 2xn]

Step 2: Compute -z and exp(-z)
Output: [exp(-2x0), exp(-2x1), ...]

Step 3: tanh = (1 - exp(-2x)) / (1 + exp(-2x))
Output: [tanh(x0), tanh(x1), ...]
```

### 4.2 为什么用这个公式

```
tanh(x) = (1 - exp(-2x)) / (1 + exp(-2x))
```

优点：
- 只需计算一次exp（exp(-2x)）
- 避免单独的exp(x)和exp(-x)计算
- 数值稳定

## 5. 与其他激活函数的对比

| 函数 | 公式 | 输出范围 | 中心 |
|------|------|---------|------|
| Sigmoid | 1/(1+exp(-x)) | (0, 1) | 偏正 |
| Tanh | (exp(x)-exp(-x))/(exp(x)+exp(-x)) | (-1, 1) | 零中心 |
| ReLU | max(0, x) | [0, +∞) | 偏正 |
| SiLU | x/(1+exp(-x)) | (-0.278, +∞) | 偏正 |
| GELU | 0.5*x*(1+tanh(...)) | (-0.17, +∞) | 偏正 |

## 6. 使用示例

```rust
use lightship_core::operator::Tanh;

let tanh = Tanh::new();

// 单值计算
let result = Tanh::compute(1.0);  // ≈ 0.7616

// 切片计算（标量）
let mut data = vec![-1.0, 0.0, 1.0, 2.0];
Tanh::compute_slice(&mut data);

// SIMD加速
let tanh_simd = Tanh::with_simd_level(SimdLevel::Avx2);
tanh_simd.compute_slice_simd(&mut data);
```

## 7. 性能特性

| 指标 | 数值 |
|------|------|
| 计算复杂度 | O(n) |
| SIMD友好度 | 高（exp已优化） |
| 内存访问 | 顺序 |
| 数值稳定性 | 良好 |

## 8. 应用场景

- **LSTM/GRU**：门控机制常用tanh
- **Autoencoder**：编码器输出层
- **生成对抗网络**：部分架构使用
- **Transformer**：早期版本使用
