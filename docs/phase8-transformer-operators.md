# Transformer 算子模块

## 概述

Phase 8 实现了 Transformer 模型的核心算子，包括矩阵乘法、Softmax、LayerNorm、GELU 激活函数和注意力机制。这些是现代 Transformer 架构（如 BERT、GPT、T5）的基础组件。

## MatMul (矩阵乘法)

### 定义

矩阵乘法是神经网络中最核心的运算之一：

```
Y = X × W + B
```

- X: 输入张量 [M, K]
- W: 权重张量 [K, N]
- B: 偏置 [N] (可选)
- Y: 输出张量 [M, N]

### 配置选项

```rust
pub struct MatMul {
    pub transpose_a: bool,  // 转置第一个输入
    pub transpose_b: bool,  // 转置第二个输入
    pub alpha: f32,          // 输出缩放因子
    pub beta: f32,          // 偏置缩放因子
}

impl MatMul {
    pub fn with_transpose_a(mut self, transpose: bool) -> Self {
        self.transpose_a = transpose;
        self
    }

    pub fn with_transpose_b(mut self, transpose: bool) -> Self {
        self.transpose_b = transpose;
        self
    }
}
```

### 输出形状计算

```rust
pub fn output_shape(&self, input_shape: &[usize], weight_shape: &[usize]) -> Vec<usize> {
    let (m, k) = if self.transpose_a {
        (input_shape[1], input_shape[0])
    } else {
        (input_shape[0], input_shape[1])
    };

    let (k2, n) = if self.transpose_b {
        (weight_shape[1], weight_shape[0])
    } else {
        (weight_shape[0], weight_shape[1])
    };

    assert_eq!(k, k2);
    vec![m, n]
}
```

## Softmax

### 定义

Softmax 将向量转换为概率分布：

```
softmax(x_i) = exp(x_i) / Σexp(x_j)
```

### 数值稳定版本

直接计算 exp 可能导致上溢。使用 max 减去来保证数值稳定：

```
softmax(x_i) = exp(x_i - max(x)) / Σexp(x_j - max(x))
```

```rust
pub fn compute_sum_exp(data: &[f32]) -> (f32, f32) {
    let max_val = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_sum: f32 = data.iter()
        .map(|&x| (x - max_val).exp())
        .sum();
    (max_val, exp_sum)
}

pub fn compute(data: &mut [f32]) {
    let (max_val, exp_sum) = Self::compute_sum_exp(data);
    if exp_sum > 0.0 {
        let scale = 1.0 / exp_sum;
        for x in data.iter_mut() {
            *x = ((*x - max_val).exp()) * scale;
        }
    }
}
```

### 轴选项

```rust
pub enum SoftmaxAxis {
    Last,     // 沿最后一维 (NLP 默认)
    First,    // 沿第一维 (CNN)
    Axis(usize),  // 沿指定维
}
```

## LayerNorm (层归一化)

### 定义

LayerNorm 对最后一维进行归一化：

```
LN(x) = (x - mean) / sqrt(variance + ε) × γ + β
```

### 实现

```rust
pub struct LayerNorm {
    pub normalized_shape: Vec<usize>,  // 归一化维度
    pub epsilon: f32,                  // 数值稳定项
    pub affine: LayerNormAffine,       // 仿射变换参数
}

pub fn compute_mean(data: &[f32]) -> f32 {
    data.iter().sum::<f32>() / data.len() as f32
}

pub fn compute_variance(data: &[f32], mean: f32) -> f32 {
    data.iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f32>() / data.len() as f32
}

pub fn normalize(data: &mut [f32], epsilon: f32) -> (f32, f32) {
    let mean = Self::compute_mean(data);
    let variance = Self::compute_variance(data, mean);
    let std_dev = (variance + epsilon).sqrt();

    for x in data.iter_mut() {
        *x = (*x - mean) / std_dev;
    }
    (mean, variance)
}
```

## GELU (高斯误差线性单元)

### 定义

GELU 是 Transformer 中常用的激活函数：

```
GELU(x) = x × Φ(x) = 0.5 × x × (1 + erf(x/√2))
```

其中 Φ 是标准正态分布的 CDF。

### 近似实现

使用 tanh 近似以提高性能：

```
GELU(x) ≈ 0.5 × x × (1 + tanh(√(2/π) × (x + 0.044715x³)))
```

```rust
pub fn tanh_gelu(x: f32) -> f32 {
    let sqrt_2_over_pi: f32 = 0.7978845608;
    let c: f32 = 0.044715;

    let x_cube = x * x * x;
    let tanh_arg = sqrt_2_over_pi * (x + c * x_cube);
    0.5 * x * (1.0 + Self::tanh_approx(tanh_arg))
}

// tanh 近似
fn tanh_approx(x: f32) -> f32 {
    if x > 5.0 { 1.0 }
    else if x < -5.0 { -1.0 }
    else {
        // 泰勒展开: tanh(x) ≈ x - x³/3 + x⁵/5 - x⁷/7
        let x2 = x * x;
        let x3 = x2 * x;
        let x5 = x3 * x2;
        let x7 = x5 * x2;
        x - x3 / 3.0 + x5 / 5.0 - x7 / 7.0
    }
}
```

## 自注意力机制

### SelfAttention

```rust
pub struct SelfAttention {
    pub config: SelfAttentionConfig,
}

pub struct SelfAttentionConfig {
    pub num_heads: usize,     // 注意力头数
    pub head_dim: usize,      // 每个头的维度
    pub scale: bool,          // 是否缩放
    pub dropout_prob: f32,    // Dropout 概率
    pub causal: bool,          // 是否使用因果掩码
}

impl SelfAttention {
    // Attention(Q, K, V) = softmax(QK^T / √d_k) V
    pub fn get_scale(&self) -> f32 {
        1.0 / (self.config.head_dim as f32).sqrt()
    }
}
```

### MultiHeadAttention

```rust
pub struct MultiHeadAttention {
    pub config: SelfAttentionConfig,
    pub out_projection_weight: Option<Tensor>,
    pub out_projection_bias: Option<Tensor>,
}

// 多头注意力的输出形状
pub fn output_shape(&self, seq_len: usize, embed_dim: usize) -> Vec<usize> {
    vec![seq_len, embed_dim]
}
```

## SkipConnection (残差连接)

### 类型

```rust
pub enum SkipConnectionType {
    Residual,    // x + f(x)
    PreNorm,     // x + f(norm(x))  (Pre-LN Transformer)
    Gated { gate: f32 },  // x + gate * f(x)
}
```

### 前向传播

```rust
pub fn forward(&self, input: Tensor, output: Tensor) -> Tensor {
    match self.connection_type {
        SkipConnectionType::Residual => input + output,
        SkipConnectionType::PreNorm => input + output, // norm 在主层之前应用
        SkipConnectionType::Gated { gate } => input + gate * output,
    }
}
```

## 使用示例

```rust
use lightship_core::operator::{
    MatMul, Softmax, LayerNorm, Gelu, SelfAttention
};

// 矩阵乘法
let matmul = MatMul::new().with_transpose_b(true);
let output_shape = matmul.output_shape(&[2, 3], &[4, 3]);

// Softmax
let mut data = vec![1.0, 2.0, 3.0];
Softmax::compute(&mut data);

// LayerNorm
let layernorm = LayerNorm::new(vec![512]).with_epsilon(1e-5);
let mut input = vec![0.0f32; 512];
layernorm.normalize(&mut input, 1e-5);

// GELU
let gelu = Gelu::new();
let result = Gelu::compute(0.5);

// 自注意力
let attention = SelfAttention::new(SelfAttentionConfig::new(8, 64));
```
