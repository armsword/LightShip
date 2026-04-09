# BatchNorm (批归一化) 原理与实现

## 1. 概述

BatchNorm 是深度学习中最重要的正则化技术之一，由 Sergey Ioffe 和 Christian Szegedy 在 2015 年提出。它通过规范化层的输入来解决 internal covariate shift 问题，显著加速了深度网络的训练。

## 2. 数学原理

### 2.1 标准 BatchNorm 公式

对于输入 $x$ 的每个通道 $c$：

```
x_norm = gamma_c * (x_c - mean_c) / sqrt(var_c + eps) + beta_c
```

其中：
- $mean_c$：该通道的均值
- $var_c$：该通道的方差
- $\gamma_c$：缩放参数（可学习）
- $\beta_c$：偏移参数（可学习）
- $\epsilon$：数值稳定性常数（通常 1e-5）

### 2.2 Running Statistics（推理时使用）

训练时计算 batch 统计量，推理时使用移动平均的 running statistics：

```
running_mean = momentum * running_mean + (1 - momentum) * batch_mean
running_var = momentum * running_var + (1 - momentum) * batch_var
```

### 2.3 训练 vs 推理

| 阶段 | 均值 | 方差 | gamma/beta |
|------|------|------|------------|
| 训练 | Batch 计算 | Batch 计算 | 可学习 |
| 推理 | Running mean | Running var | 可学习 |

## 3. 数据结构

```rust
pub struct BatchNorm {
    num_features: usize,        // 通道数
    eps: f32,                   // 数值稳定常数
    momentum: f32,              // 移动平均动量
    training: bool,            // 训练/推理模式

    // 可学习参数
    gamma: Option<Arc<Tensor>>, // 缩放参数
    beta: Option<Arc<Tensor>>,  // 偏移参数

    // Running 统计量（训练时更新）
    running_mean: Vec<f32>,
    running_var: Vec<f32>,
}
```

## 4. 实现算法

### 4.1 前向传播 (Inference)

```
对于每个通道 c:
    1. 从 running_mean[c] 获取均值
    2. 从 running_var[c] 获取方差
    3. 对输入的每个元素:
         output = gamma[c] * (input - mean) / sqrt(var + eps) + beta[c]
```

### 4.2 前向传播 (Training)

```
对于每个通道 c:
    1. 计算 batch 均值 mean_c
    2. 计算 batch 方差 var_c
    3. 更新 running_mean[c] 和 running_var[c]
    4. 对输入的每个元素:
         output = gamma[c] * (input - mean) / sqrt(var + eps) + beta[c]
```

### 4.3 方差计算

使用贝塞尔校正的无偏估计：

```
var = sum((x - mean)^2) / (N - 1)   // 样本方差
```

## 5. SIMD 加速

### 5.1 ARM64 NEON 实现

```rust
unsafe fn forward_neon_internal(...) {
    // 加载 gamma 和 beta
    let gamma_vec = vld1q_f32(gamma_ptr);
    let beta_vec = vld1q_f32(beta_ptr);
    let mean_vec = vld1q_f32(mean_ptr);

    // 计算 (x - mean)
    let x_minus_mean = vsubq_f32(x_vec, mean_vec);

    // 计算 sqrt(var + eps)
    let var_eps = vaddq_f32(var_vec, eps_vec);
    let std_vec = vsqrtq_f32(var_eps);

    // 计算 (x - mean) / std
    let normalized = vdivq_f32(x_minus_mean, std_vec);

    // 计算 gamma * normalized + beta
    let scaled = vmulq_f32(gamma_vec, normalized);
    let result = vaddq_f32(scaled, beta_vec);
}
```

### 5.2 x86_64 AVX2/AVX-512

使用对应的 SIMD intrinsic 进行向量化计算：

```rust
// AVX2 示例
let gamma_vec = _mm256_loadu_ps(gamma_ptr);
let x_minus_mean = _mm256_sub_ps(x_vec, mean_vec);
let normalized = _mm256_div_ps(x_minus_mean, std_vec);
let scaled = _mm256_mul_ps(gamma_vec, normalized);
let result = _mm256_add_ps(scaled, beta_vec);
```

## 6. 使用示例

```rust
use lightship_core::operator::BatchNorm;

// 创建 BatchNorm 算子 (4 通道)
let mut bn = BatchNorm::new(4)
    .with_gamma(vec![1.0, 1.0, 1.0, 1.0])  // 可选：缩放参数
    .with_beta(vec![0.0, 0.0, 0.0, 0.0])  // 可选：偏移参数
    .with_running_stats(vec![0.0, 0.0, 0.0, 0.0], vec![1.0, 1.0, 1.0, 1.0]);

// 推理模式
let input = Tensor::new("input".to_string(), vec![1, 4, 224, 224], DataType::F32);
let output = bn.forward(&input).unwrap();

// 训练模式 (会更新 running statistics)
let mut bn_train = BatchNorm::new(4);
bn_train.set_training(true);
let output_train = bn_train.forward_training(&input).unwrap();
```

## 7. 融合优化

BatchNorm 可与卷积算子融合：

### 7.1 Conv + BatchNorm 融合

卷积后接 BatchNorm 可以融合为一个算子：

```
y = W * x + b
z = gamma * (y - mean) / sqrt(var + eps) + beta
```

融合后可直接计算：

```
z = (gamma / sqrt(var + eps)) * (W * x + b - mean) + beta
  = (gamma / sqrt(var + eps)) * W * x + (gamma / sqrt(var + eps)) * (b - mean) + beta
```

融合后只需要一次卷积操作，减少内存访问。

## 8. 局限性与注意事项

### 8.1 Batch Size 敏感性

- BatchNorm 需要足够大的 batch size 才能准确估计统计量
- Batch size 过小会导致方差估计不准确
- 建议 batch size >= 32

### 8.2 序列模型中的问题

- RNN/LSTM 等序列模型不适合使用 BatchNorm
- LayerNorm 是更好的选择

### 8.3 训练/推理模式切换

- 必须正确切换 training/inference 模式
- 错误地使用 training 模式进行推理会导致 running statistics 不更新

## 9. 测试验证

```rust
#[test]
fn test_batchnorm_inference_single_channel() {
    let bn = BatchNorm::new(1);
    let input = Tensor::new("input".to_string(), vec![1, 1, 4], DataType::F32)
        .with_data(vec![0.0, 1.0, 2.0, 3.0]);

    let output = bn.forward(&input).unwrap();
    // 验证输出形状和数据类型
}

#[test]
fn test_batchnorm_training() {
    let mut bn = BatchNorm::new(1);
    bn.set_training(true);

    let input = Tensor::new("input".to_string(), vec![1, 1, 4], DataType::F32)
        .with_data(vec![0.0, 1.0, 2.0, 3.0]);

    let output = bn.forward_training(&input).unwrap();
    // 验证 running statistics 已更新
}
```

## 10. 性能指标

| 平台 | 通道数 | 单通道延迟 |
|------|--------|----------|
| ARM64 (Apple M1) | 64 | ~0.1ms |
| x86_64 (AVX2) | 64 | ~0.05ms |
| x86_64 (AVX-512) | 64 | ~0.03ms |
