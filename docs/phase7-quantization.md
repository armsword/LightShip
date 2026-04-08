# 量化支持模块

## 概述

Phase 7 实现了 LightShip 的量化支持模块，提供了从高精度（FP32）到低精度（INT8/FP16）量化推理的能力。量化可以显著减小模型大小、降低内存占用、提升推理速度。

## 量化类型

### QuantizationType

```rust
pub enum QuantizationType {
    Symmetric,      // 对称量化 (zero-point = 0)
    Asymmetric,      // 非对称量化 (zero-point ≠ 0)
    PerTensor,       // 张量级量化 (单一 scale)
    PerChannel,      // 通道级量化 (每个通道独立 scale)
}
```

### QuantizationAxis

指定 per-channel 量化的轴：

```rust
pub enum QuantizationAxis {
    Channel,   // 沿通道维度 (通常 axis=1 for NCHW)
    Spatial,   // 沿空间维度
    Batch,    // 沿 batch 维度
}
```

## 量化参数

```rust
pub struct QuantizationParameters {
    pub scales: Vec<f32>,      // 缩放因子
    pub zero_points: Vec<i32>, // 零点偏移
    pub bit_width: u8,         // 位宽 (通常 8)
    pub quantized_min: i32,    // 量化最小值
    pub quantized_max: i32,    // 量化最大值
}
```

## 量化方案

### QuantizationScheme

```rust
pub struct QuantizationScheme {
    pub quant_type: QuantizationType,     // 量化类型
    pub axis: QuantizationAxis,           // 量化轴
    pub source_dtype: DataType,           // 源数据类型
    pub target_dtype: DataType,          // 目标量化类型
    pub parameters: QuantizationParameters,
}
```

### 预定义方案

```rust
// Int8 对称量化 (用于 weights)
let scheme = QuantizationScheme::int8_symmetric();

// Uint8 非对称量化 (用于 activations)
let scheme = QuantizationScheme::uint8_asymmetric();

// Per-channel 量化 (用于大模型)
let scheme = QuantizationScheme::per_channel(
    scales, zero_points, QuantizationAxis::Channel, 8
);
```

## 量化算法

### Min-Max 量化

最常用的线性量化方法：

```
scale = (max - min) / (qmax - qmin)
zero_point = qmin - min / scale
```

其中 `qmin`, `qmax` 是量化类型的范围：
- QUInt8: [0, 255]
- QInt8: [-128, 127]

```rust
pub fn find_scale_zp(data: &[f32], dtype: DataType, is_symmetric: bool) -> QuantizationParameters {
    let min_val = data.iter().fold(f32::INFINITY, f32::min);
    let max_val = data.iter().fold(f32::NEG_INFINITY, f32::max);

    if is_symmetric {
        let abs_max = max_val.abs().max(min_val.abs());
        let scale = abs_max / (qmax as f32);
        QuantizationParameters::new_per_tensor(scale, 0, 8)
    } else {
        let scale = (max_val - min_val) / ((qmax - qmin) as f32);
        let zero_point = qmin as f32 - min_val / scale;
        QuantizationParameters::new_per_tensor(scale, zero_point.round() as i32, 8)
    }
}
```

### 量化/反量化

```rust
// 量化: float → int
pub fn quantize_value(value: f32, scale: f32, zero_point: i32, dtype: DataType) -> i32 {
    let qmin = match dtype { DataType::QUInt8 => 0, DataType::QInt8 => -128, _ => 0 };
    let qmax = match dtype { DataType::QUInt8 => 255, DataType::QInt8 => 127, _ => 255 };
    let quantized = (value / scale).round() as i32 + zero_point;
    quantized.clamp(qmin, qmax)
}

// 反量化: int → float
pub fn dequantize_value(quantized: i32, scale: f32, zero_point: i32) -> f32 {
    (quantized as f32 - zero_point as f32) * scale
}
```

## Scale Encoding

支持多种 scale 编码方式：

```rust
pub enum ScaleEncoding {
    Float32,           // 标准 FP32
    Float16,           // 半精度
    BlockWise { block_size: usize },  // 分块量化
    LookupTable { num_entries: usize }, // 查表法
}
```

## 量化误差评估

```rust
pub fn compute_quantization_error(original: &[f32], reconstructed: &[f32]) -> f32 {
    let sum_sq_error: f32 = original.iter()
        .zip(reconstructed.iter())
        .map(|(o, r)| (o - r).powi(2))
        .sum();
    let mse = sum_sq_error / original.len() as f32;
    mse.sqrt()
}
```

## 使用示例

```rust
use lightship_core::quantization::{QuantizationScheme, find_scale_zp};

let data = vec![-1.0, -0.5, 0.0, 0.5, 1.0];

// 自动计算量化参数
let params = find_scale_zp(&data, DataType::QInt8, true);

// 创建量化方案
let scheme = QuantizationScheme::int8_symmetric();

// 量化数据
let quantized = quantize_value(0.5, params.scales[0], params.zero_points[0], DataType::QInt8);

// 反量化验证
let dequantized = dequantize_value(quantized, params.scales[0], params.zero_points[0]);
```
