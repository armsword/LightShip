# Winograd算法实现原理

## 概述

Winograd算法是一种快速卷积算法，由Shmuel Winograd于1980年代提出。该算法通过预计算变换矩阵，将卷积操作中的乘法次数减少到接近理论最小值。对于3x3卷积核的F(2x2, 3x3)算法，可以将原始9次乘法减少到4次乘法，理论上提升约2.25倍的性能。

## 数学原理

### 标准卷积

对于输入X和3x3卷积核W，输出Y的每个元素计算如下：

```
Y[i,j] = Σ(kh=0..2) Σ(kw=0..2) W[kh,kw] * X[i+kh, j+kw]
```

对于2x2输出块（对应F(2x2, 3x3)），需要计算：
- Y[0,0], Y[0,1], Y[1,0], Y[1,1]
- 每个输出需要9次乘法（3x3卷积核）
- 总计需要36次乘法

### Winograd F(2x2, 3x3)

Winograd算法的核心思想是利用张量积分解，将卷积操作转换为矩阵乘法+元素级操作：

```
Y = A^T * (G * W * G^T) * (B^T * d * B) * A
```

其中：
- G: 卷积核变换矩阵 (4x3)
- B: 输入变换矩阵 (4x4)
- A: 输出变换矩阵 (2x4)
- d: 输入数据 (4x4 tile)
- W: 卷积核 (3x3)

### 变换矩阵

```
     [1  0  0]
G =  [1  1  1]      (4x3)
     [1 -1  1]
     [0  0  1]

     [1  0 -1  0]
B =  [0  1  1  0]   (4x4)
     [0 -1  1  0]
     [0  1  1  0]

     [1  1  1  1]
A =  [0  1 -1 -1]   (2x4)
```

### 计算流程

1. **输入变换**: B^T * d * B -> 4个元素
2. **核变换**: G * W * G^T -> 4个元素（预计算，可缓存）
3. **元素级乘法**: 4次乘法
4. **输出逆变换**: A^T * m * A -> 2x2输出

## 实现细节

### 数据布局

- 输入: NCHW格式 [N, C, H, W]
- 输出: NCHW格式 [N, out_channels, out_h, out_w]
- 卷积核: OIHW格式 [out_channels, in_channels, 3, 3]

### Tile划分

将输出空间划分为2x2的块，每个块使用Winograd算法计算：

```
输出 (out_h x out_w)
  -> tile_h = ceil(out_h / 2) 个tile
  -> tile_w = ceil(out_w / 2) 个tile
```

### 边界处理

- 输入提取时考虑padding
- 对于越界的像素，用0填充
- 边界tile可能只有部分有效输出

## 性能分析

### 理论加速比

原始卷积: 9次乘法/输出
Winograd: 4次乘法/输出 + 若干加法
加速比: 9/4 = 2.25x

### 实际考虑

1. **变换开销**: 输入/输出变换需要额外的加法操作
2. **缓存友好**: 4x4输入tile可以完全放入L1缓存
3. **适用场景**: stride=1的3x3卷积（如ResNet中的大部分卷积层）

### 限制条件

- 仅支持3x3卷积核
- 仅支持stride=1
- 多通道需要累加中间结果

## 代码结构

```rust
pub struct WinogradConv2d {
    config: WinogradConfig,
    simd_level: SimdLevel,
}

impl WinogradConv2d {
    pub fn forward(&self, input: &Tensor, filter: &Tensor) -> Result<Tensor>
    fn compute_tile(&self, ...) -> [f32; 4]  // 计算单个2x2输出块
    fn transform_kernel(&self, ...) -> [f32; 4]  // 核变换
    fn transform_input(&self, ...) -> [f32; 4]  // 输入变换
    fn inverse_transform(&self, ...) -> [f32; 4]  // 输出逆变换
}
```

## SIMD优化

### ARM64 NEON

- 使用NEON 128位向量（4个f32同时处理）
- 输入提取和变换可以向量化
- 元素级乘法可以并行处理4个元素

### x86 AVX2

- 使用AVX2 256位向量（8个f32同时处理）
- 类似NEON的优化策略

## 与Im2col+GEMM的对比

| 特性 | Im2col+GEMM | Winograd |
|------|-------------|----------|
| 乘法次数(3x3, 2x2) | 36次 | 4次 |
| 内存占用 | O(k*out_h*out_w) | O(4*4) per tile |
| 适用场景 | 任意卷积 | stride=1, 3x3 |
| 缓存效率 | 中等 | 高 |

## 未来优化方向

1. **多通道融合**: 将多通道的输入变换合并，减少重复计算
2. **预变换核缓存**: 核变换只计算一次，跨batch复用
3. **更大的Tile**: F(4x4, 3x3)可以将乘法减少到16/36 = 2.25x，但变换开销增加
4. **混合精度**: 结合int8量化进一步加速