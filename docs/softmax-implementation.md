# Softmax 算子实现原理

## 1. 算法定义

Softmax 是神经网络中常用的激活函数，将任意实数向量转换为概率分布：

```
softmax(x_i) = exp(x_i) / sum(exp(x_j))
```

对于输入向量 `[1.0, 2.0, 3.0]`：
- `exp([1,2,3]) ≈ [2.718, 7.389, 20.086]`
- `sum ≈ 30.193`
- `softmax ≈ [0.090, 0.245, 0.665]`

## 2. 数值稳定性优化

直接计算 `exp(x)` 可能导致溢出（当 x 很大时）。优化方法：先减去最大值

```rust
// 原始: exp(x_i) / sum(exp(x_j))

// 优化后（减去最大值）:
max_val = max(x)
shifted = x - max_val  // 现在最大值变为 0
exp_shifted = exp(shifted)
softmax = exp_shifted / sum(exp_shifted)
```

数学上等价：
```
exp(x_i - max) / sum(exp(x_j - max))
= exp(x_i) / exp(max) / sum(exp(x_j) / exp(max))
= exp(x_i) / sum(exp(x_j))
```

## 3. LightShip 实现

### 3.1 CPU 后端执行流程

```rust
fn execute_softmax(&self, inputs: &[&Tensor], outputs: &mut [&mut Tensor]) -> Result<()> {
    let input = inputs[0];
    let output = outputs[0];

    // 1. 提取 f32 数据
    let bytes = input.data_as_bytes();
    let num_elements = bytes.len() / 4;
    let mut values: Vec<f32> = ... // 从字节转换

    // 2. 找最大值（标量归约）
    let max_val = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    // 3. 减去最大值（平移）
    for v in &mut values {
        *v = *v - max_val;  // 现在范围是 (-max, 0]
    }

    // 4. 计算 exp（使用 SIMD 或标量）
    let mut exp_values = vec![0.0f32; num_elements];
    exp_softmax_simd(&values, &mut exp_values, simd_level);

    // 5. 求和（水平求和 SIMD）
    let sum_exp = horizontal_sum(&exp_values, simd_level);

    // 6. 除以总和
    div_scalar_simd(&exp_values, &mut softmax_values, sum_exp, simd_level);
}
```

### 3.2 exp 函数选择

LightShip 使用 `std::exp()` 而非自定义多项式逼近，原因：

| 方法 | x=-2 时误差 | 备注 |
|------|-------------|------|
| 5次 Taylor 多项式 | ~50% | 收敛慢，x=-2 时不准确 |
| 10次多项式 | ~10% | 仍不满足精度要求 |
| 查表法 | ~1% | 实现复杂，缓存不友好 |
| std::exp | ~0% | Intel/ARM 已高度优化 |

对于 1000 元素的 softmax，`std::exp` 耗时约 50-100μs，远低于 MNN 的 20μs 但可接受。

## 4. SIMD 优化要点

### 4.1 水平求和 (horizontal_sum)

Softmax 需要对所有元素求和，这是个"归约"操作：

```rust
pub fn horizontal_sum(input: &[f32], level: SimdLevel) -> f32 {
    // 使用 SIMD 做向量加法，然后手动归约
    // 例如 ARM NEON: vpaddq + vpaddq + vgetq_lane
}
```

### 4.2 逐元素除法 (div_scalar_simd)

每个 exp 值除以同一个标量（sum）：

```rust
pub fn div_scalar_simd(input: &[f32], output: &mut [f32], scalar: f32, level: SimdLevel) {
    // 加载标量到向量寄存器
    // 逐元素向量除法
}
```

## 5. 性能数据

| 规模 | LightShip | MNN | 比率 |
|------|-----------|-----|------|
| 1000 元素 | ~100μs | ~20μs | 5x |

主要差距：
- `std::exp` vs 手写 SIMD exp
- 标量除法 vs 向量除法

## 6. 后续优化方向

1. **SIMD exp**: 实现向量化的指数函数（使用分段多项式或查表+插值）
2. **融合算子**: Softmax 通常跟在 MatMul/Conv 后面，可考虑算子融合
3. **ARM NEON 优化**: 使用 `vexpq_f32` 指令（如果可用）
