# SiLU 算子原理与实现

## 1. 概述

SiLU（Sigmoid Linear Unit）是一种自门控激活函数，由Ramachandran等人在2017年提出。也被称为**Swish**激活函数。

**公式：**
```
SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
```

## 2. 数学特性

### 2.1 核心性质

| 性质 | 描述 |
|------|------|
| **非单调** | SiLU是非单调激活函数，介于线性与非线性之间 |
| **自门控** | 输入x通过sigmoid函数门控自身的响应 |
| **平滑** | 处处可导，无ReLU的"dead zone"问题 |
| **有下界** | x → -∞ 时，SiLU(x) → 0 |
| **无上界** | x → +∞ 时，SiLU(x) ≈ x（线性增长） |

### 2.2 与其他激活函数的关系

```
SiLU(x) = x * sigmoid(x)
        = x / (1 + exp(-x))

ReLU(x) = max(0, x)

GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))

Swish(x) = SiLU(x)  # 同一函数的不同名称
```

### 2.3 导数（梯度）

```
dSiLU/dx = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
         = sigmoid(x) + x * sigmoid(x) - x * sigmoid(x)^2
```

梯度特性：
- 当x很大时：sigmoid(x) ≈ 1，梯度 ≈ 1（无梯度消失）
- 当x为负时：梯度可以小于0（不同于ReLU）
- 当x=0时：sigmoid(0) = 0.5，SiLU'(0) = 0.5

## 3. 数值稳定性

### 3.1 原始公式

```
SiLU(x) = x / (1 + exp(-x))
```

对于x > 20：exp(-x) ≈ 0，SiLU(x) ≈ x
对于x < -20：exp(-x) → ∞，SiLU(x) → 0

### 3.2 SIMD实现中的数值处理

```rust
let sigmoid = if exp_xi.is_infinite() && exp_xi.is_sign_positive() {
    // Large positive x: sigmoid ≈ 1
    1.0f32
} else if exp_xi == 0.0 {
    // Large negative x: sigmoid ≈ 0
    0.0f32
} else {
    exp_xi / (1.0 + exp_xi)
};
```

### 3.3 实现策略

LightShip采用以下计算路径：

1. **exp(x)计算**：使用SIMD优化的`exp_simd`
2. **sigmoid计算**：exp(x) / (1 + exp(x))
3. **SiLU计算**：x * sigmoid(x)

对于大正值，exp(x)会溢出为无穷大，此时sigmoid(x) ≈ 1，SiLU(x) ≈ x
对于大负值，exp(x)会下溢为0，此时sigmoid(x) ≈ 0，SiLU(x) ≈ 0

## 4. SIMD优化

### 4.1 计算流程

```
Input: [x0, x1, x2, ..., xn]

Step 1: exp_simd
Output: [exp(x0), exp(x1), exp(x2), ..., exp(xn)]

Step 2: Loop (scalar add)
Temp: [1+exp(x0), 1+exp(x1), 1+exp(x2), ..., 1+exp(xn)]

Step 3: sigmoid (loop with division)
sigmoid: [exp(x0)/(1+exp(x0)), exp(x1)/(1+exp(x1)), ...]

Step 4: mul_simd
Output: [x0*sigmoid0, x1*sigmoid1, ...]
```

### 4.2 为什么不使用SIMD做除法

平台SIMD库未提供元素级除法(`div_simd`)，因此采用：
- SIMD计算exp(x)
- 标量循环处理 1 + exp(x) 和除法
- SIMD处理最终的乘法

这在实际应用中足够高效，因为：
- exp是最昂贵的操作（已SIMD化）
- 加法和除法是廉价操作
- 乘法可以合并到后续操作中

## 5. 使用示例

```rust
use lightship_core::operator::{SiLU, Swish};

let silu = SiLU::new();

// 单值计算
let result = SiLU::compute(1.0);  // ≈ 0.731

// 切片计算（标量）
let mut data = vec![-1.0, 0.0, 1.0, 2.0];
SiLU::compute_slice(&mut data);

// SIMD加速
let silu_simd = SiLU::with_simd_level(SimdLevel::Avx2);
silu_simd.compute_slice_simd(&mut data);

// Swish是SiLU的别名
let swish = Swish::new();
```

## 6. 性能对比

| 激活函数 | 单次计算复杂度 | SIMD友好度 | 数值稳定性 |
|---------|--------------|-----------|-----------|
| ReLU | O(1) | 极高 | 优秀（有dead zone） |
| Sigmoid | O(1) | 高 | 中等（exp溢出） |
| SiLU | O(1) | 高 | 良好（边界处理） |
| GELU | O(1) | 高 | 良好（tanh近似） |

## 7. 与ReLU的对比

### SiLU优势

1. **平滑梯度**：避免ReLU的梯度突变
2. **负值响应**：对负值有非零响应（不同于ReLU）
3. **自适应**：门控机制自适应调节

### 实验发现

论文"Swish: a Self-Gated Activation Function"表明：
- SiLU在图像分类任务中优于ReLU
- 在深层网络中表现更稳定
- 特别是当网络超过20层时优势明显

## 8. 实现代码

```rust
pub fn compute(x: f32) -> f32 {
    if x > 20.0 {
        return x;  // SiLU(x) ≈ x for large positive x
    }
    if x < -20.0 {
        return 0.0;  // SiLU(x) ≈ 0 for large negative x
    }
    x / (1.0 + (-x).exp())
}

pub fn compute_slice(data: &mut [f32]) {
    for x in data.iter_mut() {
        *x = Self::compute(*x);
    }
}
```
