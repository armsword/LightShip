# SIMD 运行时优化原理

## 概述

SIMD (Single Instruction Multiple Data) 是提升神经网络推理性能的关键技术。本文档介绍 LightShip 中的 SIMD 优化实现原理。

## CPU Feature 检测

### x86_64 架构

x86_64 使用 `CPUID` 指令在运行时检测 CPU 特性：

```rust
// 获取 CPUID leaf 1 的特性
let result = unsafe { std::arch::x86_64::__cpuid(1) };
let eax = result.eax;  // 主要特性标志
let ecx = result.ecx;  // 扩展特性 (SSE3, AVX, etc.)
```

**检测特性列表：**

| 特性 | Flag | 说明 |
|------|------|------|
| SSE | CPUID.1:EDX[25] | 128-bit SIMD, 单精度浮点 |
| SSE2 | CPUID.1:EDX[26] | 128-bit SIMD, 双精度浮点和整数 |
| SSE3 | CPUID.1:ECX[0] | 128-bit SIMD, 水平运算 |
| SSSE3 | CPUID.1:ECX[9] | 补充 SSE3 |
| SSE4.1 | CPUID.1:ECX[19] | 128-bit, 点积, 插入/提取 |
| SSE4.2 | CPUID.1:ECX[20] | 128-bit, 字符串指令 |
| AVX | CPUID.1:ECX[28] | 256-bit, 三操作数 VEX |
| AVX2 | CPUID.7:EBX[5] | 256-bit, 整数 SIMD |
| AVX-512F | CPUID.7:EDX[16] | 512-bit, Foundation |

### ARM64 架构

ARM64 的 NEON 是架构标准特性，FP16 (ARMv8.2+) 是可选特性：

- **NEON**: 所有 ARMv8-A 处理器支持，128-bit SIMD
- **FP16**: ARMv8.2-A 引入，用于半精度浮点加速
- **SVE**: 可伸缩向量扩展，长度不固定 (128-2048 bits)

## SimdLevel 枚举

LightShip 使用 `SimdLevel` 枚举表示运行时支持的最高 SIMD 级别：

```rust
pub enum SimdLevel {
    None,       // 无 SIMD
    Sse,        // 128-bit SSE
    Sse2,       // 128-bit SSE2
    Sse3,       // 128-bit SSE3
    Ssse3,      // 128-bit SSSE3
    Sse4_1,     // 128-bit SSE4.1
    Sse4_2,     // 128-bit SSE4.2
    Avx,        // 256-bit AVX
    Avx2,       // 256-bit AVX2
    Avx512,     // 512-bit AVX-512
    Neon,       // 128-bit ARM NEON
    Neonfp16,   // ARM NEON + FP16
    Sve,        // ARM SVE (可伸缩)
}
```

### 优先级判断

```rust
pub fn simd_level(&self) -> SimdLevel {
    if self.features.avx512f { SimdLevel::Avx512 }
    else if self.features.avx2 { SimdLevel::Avx2 }
    else if self.features.avx { SimdLevel::Avx }
    else if self.features.sse4_2 { SimdLevel::Sse4_2 }
    // ...
}
```

## SIMD 调度策略

### 向量宽度

不同 SIMD 级别对应不同向量宽度：

| 级别 | 向量宽度 |
|------|----------|
| SSE/NEON | 16 字节 (128-bit) |
| AVX2 | 32 字节 (256-bit) |
| AVX-512 | 64 字节 (512-bit) |

```rust
pub fn vector_width(&self) -> usize {
    match self {
        SimdLevel::Sse | SimdLevel::Neon => 16,
        SimdLevel::Avx2 => 32,
        SimdLevel::Avx512 => 64,
        _ => 0,
    }
}
```

### 性能影响

**理论加速比估算：**

| 算子 | SSE/NEON | AVX2 | AVX-512 |
|------|----------|------|---------|
| ReLU (激活) | ~4x | ~8x | ~16x |
| Conv2d (3x3) | ~4x | ~8x | ~16x |
| GEMM (矩阵乘) | ~4x | ~8x | ~16x |

### 调度示例

```rust
fn compute_relu_simd(input: &[f32], output: &mut [f32], simd_level: SimdLevel) {
    match simd_level {
        SimdLevel::Avx2 => unsafe { avx2_relu_kernel(input, output) },
        SimdLevel::Avx => unsafe { avx_relu_kernel(input, output) },
        SimdLevel::Sse2 | SimdLevel::Neon => unsafe { sse_relu_kernel(input, output) },
        SimdLevel::None => scalar_relu_kernel(input, output),
    }
}
```

## 核心优化算子

### 1. ReLU 激活

```c
// SSE/NEON 16元素并行
__m128 mask = _mm_loadu_ps(input);
__m128 zero = _mm_setzero_ps();
__m128 result = _mm_max_ps(zero, mask);  // ReLU: max(0, x)
_mm_storeu_ps(output, result);
```

### 2. Conv2d (Im2col + GEMM)

```
输入特征图 → Im2col展开 → SIMD矩阵乘法 → 累加 → 激活 → 输出
```

Im2col 将卷积转换为矩阵乘法：
- 3x3 卷积核 → 每输出像素需要 9 个输入元素
- 展开后利用 SIMD 一次性处理多个乘加操作

### 3. GEMM 矩阵乘法

分块矩阵乘法充分利用缓存：

```
┌─────────────────┐
│  A      B       │
│  ┌────┬────┐    │
│  │    │    │    │
│  │A11 │A12 │    │
│  │    │    │    │
│  ├────┼────┤    │
│  │A21 │A22 │    │
│  │    │    │    │
│  └────┴────┘    │
└─────────────────┘

每个块使用 SIMD 向量化处理
```

## 内存对齐

SIMD 操作通常需要内存对齐：

| 类型 | 对齐要求 |
|------|----------|
| SSE (__m128) | 16 字节 |
| AVX (__m256) | 32 字节 |
| AVX-512 (__m512) | 64 字节 |

```rust
// 不对齐加载（性能略有损失）
let data = _mm_loadu_ps(ptr);  // unaligned

// 对齐加载（更快）
let data = _mm_load_ps(aligned_ptr);  // 必须16字节对齐
```

## 已实现的算子内核

LightShip 在 `platform/simd.rs` 中实现了以下向量化算子：

### 1. ReLU / ReLU6

```rust
pub fn relu_simd(input: &[f32], output: &mut [f32], level: SimdLevel)
pub fn relu6_simd(input: &[f32], output: &mut [f32], level: SimdLevel)
```

**实现方式**：
- AVX-512: `_mm512_max_ps` 一次处理 16 个浮点
- AVX2: `_mm256_max_ps` 一次处理 8 个浮点
- SSE: `_mm_max_ps` 一次处理 4 个浮点
- NEON: `vmaxq_f32` 一次处理 4 个浮点

### 2. Element-wise Add / Mul

```rust
pub fn add_simd(a: &[f32], b: &[f32], c: &mut [f32], level: SimdLevel)
pub fn mul_simd(a: &[f32], b: &[f32], c: &mut [f32], level: SimdLevel)
```

### 3. GEMM 矩阵乘法

```rust
pub fn gemm_simd(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize, level: SimdLevel)
```

采用分块策略：
- 按输出行/列分块
- 每块内按 SIMD 向量宽度展开
- 使用 FMA (Fused Multiply-Add) 指令减少乘加延迟

### 4. Horizontal Sum

```rust
pub fn horizontal_sum(arr: &[f32], level: SimdLevel) -> f32
```

用于 Softmax、LayerNorm 等需要归约操作的算子。

## 运行时调度

```rust
pub fn detect_simd_level() -> SimdLevel

// 使用示例
let level = detect_simd_level();
relu_simd(&input, &mut output, level);
```

## Conv2d 算子优化

LightShip 在 `operator/convolution.rs` 中实现了 Conv2d 卷积算子：

### Im2col 变换

Im2col (Image to Column) 将卷积操作转换为矩阵乘法：

```
输入特征图                    Im2col 展开
┌─────────────┐              ┌────────────────────┐
│ a b c d     │              │ b c d 0            │
│ e f g h     │   Im2col     │ c d 0 0            │
│ i j k l     │ ──────────▶  │ d 0 0 0            │
│ m n o p     │              │ e f g h            │
└─────────────┘              │ f g h 0            │
                             │ ...                │
                             └────────────────────┘
```

**优点**：
- 将稀疏的卷积操作转换为密集的矩阵乘法
- 可以利用优化的 BLAS (GEMM) 库

### Conv2d 配置

```rust
pub struct Conv2dConfig {
    pub out_channels: usize,
    pub kernel_h: usize,
    pub kernel_w: usize,
    pub stride_h: usize,
    pub stride_w: usize,
    pub pad_h: usize,
    pub pad_w: usize,
    pub dilation_h: usize,
    pub dilation_w: usize,
    pub groups: usize,  // 分组卷积
}
```

### 前向传播

```rust
let conv = Conv2d::new(config);
let output = conv.forward(&input, &filter)?;
```

### 支持的特性

- 标准卷积 (groups=1)
- 分组卷积 (groups>1)
- Stride 卷积
- 空洞卷积 (Dilation)
- Padding

### 后续优化方向

1. **Winograd 算法**: 对于 3x3 卷积可减少乘法次数
2. **Direct 卷积优化**: 对于小卷积核的直接实现
3. **内存布局优化**: NHWC 格式替代 NCHW

## 未来优化方向

1. **运行时 JIT 编译**: 根据检测到的 CPU 特性动态生成最优代码
2. **混合精度调度**: int8/FP16/FP32 自动选择
3. **Cache 优化**: 预取、阻塞策略优化
4. **SVE 支持**: ARM 可伸缩向量扩展

## 参考资料

- [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/)
- [ARM NEON Programmer's Guide](https://developer.arm.com/architectures/instruction-sets/simd-isas/neon)
- [Agner Fog's Optimization Manuals](https://www.agner.org/optimize/)
