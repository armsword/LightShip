# Pooling 算子原理与实现

## 1. 概述

Pooling（池化）算子是卷积神经网络中的重要组成部分，用于对特征图进行下采样，减少空间尺寸同时保留重要特征。LightShip 实现了四种池化操作：

| 类型 | 描述 |
|------|------|
| MaxPool2d | 最大池化，取窗口内的最大值 |
| AvgPool2d | 平均池化，取窗口内的平均值 |
| GlobalMaxPool2d | 全局最大池化，对整个空间维度进行最大操作 |
| GlobalAvgPool2d | 全局平均池化，对整个空间维度进行平均操作 |

## 2. 数学原理

### 2.1 Max Pooling

最大池化选择窗口中的最大值，用于捕获最显著的特征：

```
output[h][w] = max(input[h*stride_h + kh][w*stride_w + kw]) for all (kh, kw) in kernel
```

**优点**：对噪声具有较好的鲁棒性，保留纹理特征

**应用场景**：图像分类、目标检测中的特征提取

### 2.2 Average Pooling

平均池化计算窗口内所有值的平均：

```
output[h][w] = (1/K) * sum(input[h*stride_h + kh][w*stride_w + kw]) for all (kh, kw) in kernel
```

其中 K = kernel_h × kernel_w

**优点**：对噪声更平滑，适用于需要全局信息的场景

**应用场景**：图像分类、注意力机制中的token混合

### 2.3 全局池化

全局池化将整个空间维度（H × W）压缩为单个值：

- GlobalMaxPool2d：提取最显著的特征响应
- GlobalAvgPool2d：提取特征的全局统计信息

**优势**：
- 输入尺寸无关性
- 减少参数量（全连接层可以替换为全局池化）
- 更好的空间不变性

## 3. 配置参数

```rust
pub struct Pool2dConfig {
    pub kernel_h: usize,      // 核高度
    pub kernel_w: usize,     // 核宽度
    pub stride_h: usize,     // 垂直步长
    pub stride_w: usize,     // 水平步长
    pub pad_h: usize,        // 垂直填充
    pub pad_w: usize,        // 水平填充
    pub dilation_h: usize,   // 垂直膨胀
    pub dilation_w: usize,   // 水平膨胀
    pub count_include_pad: bool, // 平均池化是否计入填充
}
```

### 3.1 输出尺寸计算

对于输入 [N, C, H, W]，输出尺寸计算：

```
out_h = (H + 2*pad_h - dilation_h*(kernel_h-1) - 1) / stride_h + 1
out_w = (W + 2*pad_w - dilation_w*(kernel_w-1) - 1) / stride_w + 1
```

### 3.2 填充处理

**count_include_pad = true**（ONNX兼容）：
- 填充区域视为有效参与计算
- 平均池化会计入填充值（通常为0）

**count_include_pad = false**：
- 填充区域不参与计算
- 仅计算实际数据点的平均

## 4. 实现架构

```
Pool2d
├── PoolType          // 池化类型枚举
│   ├── Max           // 最大池化
│   ├── Avg           // 平均池化
│   ├── GlobalMax     // 全局最大池化
│   └── GlobalAvg     // 全局平均池化
├── Pool2dConfig      // 池化配置
└── SIMD加速层        // 水平max/sum操作
```

## 5. SIMD优化策略

### 5.1 全局池化的SIMD优化

对于全局池化，核心操作是水平方向的max/sum：

```rust
// 全局最大池化
let max_val = channel_slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

// 全局平均池化
let sum: f32 = channel_slice.iter().sum();
let avg_val = sum / spatial_size as f32;
```

### 5.2 局部池化的SIMD优化

对于 2×2 stride=2 的常见配置，可利用SIMD并行处理多个窗口：

```rust
// 收集4个元素进行SIMD max
let vals = [f32::NEG_INFINITY; 4];
vals[0] = input[h0 * in_w + w0];      // 左上
vals[1] = input[h0 * in_w + w0 + 1];  // 右上
vals[2] = input[(h0+1) * in_w + w0];  // 左下
vals[3] = input[(h0+1) * in_w + w0+1]; // 右下

let max_val = vals[0].max(vals[1]).max(vals[2]).max(vals[3]);
```

## 6. 使用示例

### 6.1 最大池化

```rust
use lightship_core::operator::{Pool2d, Pool2dConfig};

let pool = Pool2d::max_pool(Pool2dConfig {
    kernel_h: 2,
    kernel_w: 2,
    stride_h: 2,
    stride_w: 2,
    pad_h: 0,
    pad_w: 0,
    dilation_h: 1,
    dilation_w: 1,
    count_include_pad: true,
});

let input = Tensor::new("input".to_string(), vec![1, 64, 32, 32], DataType::F32);
let output = pool.forward(&input)?;
// output.shape = [1, 64, 16, 16]
```

### 6.2 全局平均池化

```rust
use lightship_core::operator::{Pool2d, PoolType, Pool2dConfig};

let pool = Pool2d::with_simd_level(
    PoolType::GlobalAvg,
    Pool2dConfig::default(),
    SimdLevel::Avx2
);

let input = Tensor::new("input".to_string(), vec![1, 512, 7, 7], DataType::F32);
let output = pool.forward_simd(&input)?;
// output.shape = [1, 512, 1, 1]
```

## 7. 性能特性

| 配置 | 计算复杂度 | 内存访问模式 |
|------|-----------|-------------|
| MaxPool2d 2×2 stride=2 | O(NCHW/4) | 顺序访问 |
| AvgPool2d 2×2 stride=2 | O(NCHW/4) | 顺序访问 |
| GlobalMaxPool2d | O(NCHW) | 非连续+规约 |
| GlobalAvgPool2d | O(NCHW) | 非连续+规约 |

## 8. 与其他框架的兼容性

### ONNX算子映射

| LightShip | ONNX |
|-----------|------|
| MaxPool2d | MaxPool |
| AvgPool2d | AveragePool |
| GlobalMaxPool2d | GlobalMaxPool |
| GlobalAvgPool2d | GlobalAveragePool |

### PyTorch映射

```python
# PyTorch
torch.nn.MaxPool2d(kernel_size=2, stride=2)
torch.nn.AvgPool2d(kernel_size=2, stride=2)
torch.nn.AdaptiveAvgPool2d(1)  # GlobalAvgPool2d
```

## 9. 数值稳定性

### 最大池化

- 使用 `f32::NEG_INFINITY` 初始化最大值
- 无特殊数值问题

### 平均池化

- 当 count=0 时返回0（避免除零）
- 对于极大输入求和，可能出现溢出，但f32范围足够大

### 全局平均池化

- 累加和可能在大张量上溢出
- 未来考虑使用Kahan求和算法提高精度
