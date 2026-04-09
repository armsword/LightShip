# LightShip CPU 后端算子执行实现

## 1. 概述

CPU 后端是 LightShip 推理引擎的默认计算后端，负责在 CPU 上执行神经网络算子。本文档记录已实现的算子执行逻辑。

## 2. Backend Trait 修改

### 2.1 Execute 方法签名

原始签名存在问题：`&mut [&Tensor]` 实际上每个元素是 immutable reference `&Tensor`。

```rust
// 修改前（有问题）
fn execute(
    &self,
    op: &CompiledOperator,
    inputs: &[&Tensor],
    outputs: &mut [&Tensor],  // 每个元素是 &Tensor，不是 &mut Tensor
) -> Result<()>;

// 修改后（正确）
fn execute(
    &self,
    op: &CompiledOperator,
    inputs: &[&Tensor],
    outputs: &mut [&mut Tensor],  // 每个元素是 &mut Tensor
) -> Result<()>;
```

## 3. 已实现算子

### 3.1 ReLU

**公式**: `f(x) = max(x, 0)`

**实现**:
```rust
fn execute_relu(&self, inputs: &[&Tensor], outputs: &mut [&mut Tensor]) -> Result<()> {
    // 遍历输入字节，按 4 字节解析为 f32
    for chunk in input_bytes.chunks_exact(4) {
        let value = f32::from_le_bytes([...]);  // [chunk[0], chunk[1], chunk[2], chunk[3]]
        let result = if value > 0.0 { value } else { 0.0 };
        output_bytes.extend_from_slice(&result.to_le_bytes());
    }
}
```

**测试**:
- 输入: `[-1.0, 0.0, 1.0, 2.0]`
- 输出: `[0.0, 0.0, 1.0, 2.0]`

### 3.2 Add (元素级加法)

**公式**: `c[i] = a[i] + b[i]`

**实现**:
```rust
fn execute_add(&self, inputs: &[&Tensor], outputs: &mut [&mut Tensor]) -> Result<()> {
    // 同时遍历两个输入的字节
    for (chunk_a, chunk_b) in input_bytes_a.chunks_exact(4)
        .zip(input_bytes_b.chunks_exact(4))
    {
        let a = f32::from_le_bytes([...]);
        let b = f32::from_le_bytes([...]);
        output_bytes.extend_from_slice(&(a + b).to_le_bytes());
    }
}
```

**测试**:
- 输入 a: `[1.0, 2.0, 3.0]`
- 输入 b: `[4.0, 5.0, 6.0]`
- 输出: `[5.0, 7.0, 9.0]`

### 3.3 Mul (元素级乘法)

**公式**: `c[i] = a[i] * b[i]`

**实现**: 与 Add 类似，使用乘法操作。

**测试**:
- 输入 a: `[2.0, 3.0, 4.0]`
- 输入 b: `[5.0, 6.0, 7.0]`
- 输出: `[10.0, 18.0, 28.0]`

### 3.4 Sigmoid

**公式**: `sigmoid(x) = 1 / (1 + exp(-x))`

**实现**:
```rust
fn execute_sigmoid(&self, inputs: &[&Tensor], outputs: &mut [&mut Tensor]) -> Result<()> {
    for chunk in input_bytes.chunks_exact(4) {
        let x = f32::from_le_bytes([...]);
        let result = 1.0 / (1.0 + (-x).exp());
        output_bytes.extend_from_slice(&result.to_le_bytes());
    }
}
```

**测试**:
- 输入: `[0.0, 1.0, -1.0]`
- 输出: `[0.5, 0.73105858, 0.26894142]` (允许 0.01 误差)

## 4. 数据布局

### 4.1 字节顺序

所有数据使用 **小端序 (Little Endian)** 编码：
- f32 值 `1.0` 存储为字节 `[0, 0, 128, 63]`
- `to_le_bytes()` 和 `from_le_bytes()` 进行转换

### 4.2 内存布局

```
TensorData::Owned(bytes)
    ├── Vec<u8> 存储原始字节
    ├── 每个 f32 占 4 字节
    └── 通过 chunks_exact(4) 迭代处理
```

### 4.3 数据访问模式

```rust
// 输入: &[u8] 原始字节
let input_bytes = input.data_as_bytes();

// 迭代: 每 4 字节解析为一个 f32
for chunk in input_bytes.chunks_exact(4) {
    let value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    // 处理 value...
}

// 输出: 构建新的字节 Vec
let mut output_bytes = Vec::with_capacity(input_bytes.len());
output_bytes.extend_from_slice(&result.to_le_bytes());
output.data = TensorData::Owned(output_bytes);
```

## 5. 错误处理

每个算子执行函数都包含以下验证：

1. **输入输出数量检查**: 确保有足够的输入和输出 tensor
2. **数据类型检查**: 确保是 F32 类型
3. **尺寸匹配检查**: 确保两个输入的字节数相同

错误时返回 `LightShipError::InvalidParam` 或 `LightShipError::Backend`。

## 6. 尚未实现的算子

| 算子 | 状态 |
|------|------|
| ReLU6 | 待实现 |
| Tanh | 待实现 |
| MaxPool2d | 待实现 |
| AvgPool2d | 待实现 |
| Conv2d | 待实现 |
| FullyConnected | 待实现 |
| BatchNorm | 待实现 |
| Softmax | 待实现 |

## 7. 性能优化方向

### 7.1 SIMD 加速

使用 SIMD 指令并行处理多个元素：
- ARM64: NEON SIMD (`std::arch::aarch64`)
- x86_64: SSE/AVX2 (`std::arch::x86_64`)

```rust
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

unsafe fn relu_simd(input: &[f32], output: &mut [f32]) {
    // 使用 NEON 矢量操作并行处理
}
```

### 7.2 多线程

对于大张量，使用线程池并行处理：
- 将张量分割成多个 chunk
- 每个线程处理一个 chunk
- 使用 `join` 等待所有线程完成

### 7.3 内存预分配

避免在算子执行时频繁分配 Vec：
- 预先分配输出 buffer
- 重复使用临时 buffer

## 8. 测试覆盖

每个算子都有对应的单元测试：
- 正确性测试：验证计算结果
- 边界测试：空输入、错误类型等
- 压力测试：大张量性能

运行测试：
```bash
cargo test -p lightship-core --test cpu_backend_test
```
