# 算子融合 (Operator Fusion) 原理与实现

## 1. 概述

算子融合是图优化中的重要技术，通过将多个连续的算子合并为一个 fused 算子来减少内存访问和kernel启动开销。

## 2. 支持的融合模式

| 融合类型 | 模式 | 说明 |
|---------|------|------|
| Conv+ReLU | Conv → ReLU | 卷积后接ReLU激活 |
| Conv+ReLU6 | Conv → ReLU6 | 卷积后接ReLU6激活 |
| Add+ReLU | Add → ReLU | 残差连接中的ReLU |
| Conv+BatchNorm | Conv → BN | 卷积后接批归一化（待实现） |
| Conv+Sigmoid | Conv → Sigmoid | 卷积后接Sigmoid（待实现） |

## 3. 融合条件

不是所有的算子对都可以融合，需要满足以下条件：

### 3.1 通用条件

1. **单一消费者**：被融合的激活函数（如ReLU）必须只有**一个**消费者
   - 原因：如果ReLU有多个下游消费者，直接融合会破坏图的连接关系

2. **输出形状不变**：融合后输出tensor的shape必须保持不变

### 3.2 具体示例

**可以融合：**
```
Conv (1 output) → ReLU (1 consumer) → Output
```
- ReLU只有1个消费者（Output）
- 融合后：Conv带ReLU属性，Output直接消费Conv的输出

**不可以融合：**
```
Conv (1 output) → ReLU (2 consumers: A, B)
```
- ReLU有2个消费者（A和B）
- 如果融合，A和B都需要连接到Conv的输出
- 这要求复制数据或改变计算图结构

## 4. 数据结构

### 4.1 FusionType 枚举

```rust
pub enum FusionType {
    ConvReLU,       // Conv + ReLU
    ConvReLU6,      // Conv + ReLU6
    ConvSigmoid,    // Conv + Sigmoid
    ConvBatchNorm,  // Conv + BatchNorm
    BatchNormReLU,  // BatchNorm + ReLU
    AddReLU,        // Add + ReLU (residual)
    MulReLU,        // Mul + ReLU
}
```

### 4.2 FusionInfo 结构

```rust
pub struct FusionInfo {
    pub fusion_type: FusionType,
    pub original_ops: Vec<OperatorType>,  // 融合前的算子列表
    pub eliminate_batch_norm: bool,       // 是否可消除BatchNorm
}
```

### 4.3 Node 扩展

```rust
pub struct Node {
    // ... existing fields ...
    pub fusion: Option<FusionInfo>,  // 融合信息（如果有）
}
```

## 5. 实现算法

### 5.1 FusionPass

```rust
pub struct FusionPass {
    fusion_types: Vec<FusionType>,
}
```

### 5.2 融合候选查找

```
For each node in graph:
    If node is Conv2d or Add:
        Find ReLU consumer
        If ReLU has exactly ONE consumer:
            Add to fusion candidates
```

### 5.3 融合执行

对于 Conv → ReLU 融合：

1. **标记融合信息**：在Conv节点上设置 `FusionInfo`
2. **重定向消费者**：将ReLU的消费者的输入从ReLU的输出改为Conv的输出
3. **删除ReLU节点**：从图中移除ReLU节点

```rust
fn fuse_conv_relu(&self, graph: &mut Graph, conv_id: NodeId, relu_id: NodeId) {
    // 1. 收集信息
    let relu_output = ...;
    let conv_output = ...;

    // 2. 设置融合信息
    if let Some(conv_node) = graph.nodes.iter_mut().find(|n| n.id == conv_id) {
        conv_node.fusion = Some(FusionInfo::conv_relu());
    }

    // 3. 重定向消费者
    for node in &mut graph.nodes {
        for input in &mut node.inputs {
            if input.tensor_name == relu_output {
                input.tensor_name = conv_output.clone();
            }
        }
    }

    // 4. 删除ReLU
    graph.retain_nodes(|n| n.id != relu_id);
}
```

## 6. 使用示例

```rust
use lightship_core::ir::optimizer::FusionPass;
use lightship_core::ir::Graph;

let mut graph = Graph::new("test".to_string());
// ... 构建图 ...

let fusion = FusionPass::new();
fusion.optimize(&mut graph);

// 检查融合结果
for node in &graph.nodes {
    if let Some(info) = &node.fusion {
        println!("Node {} fused with type {}", node.name, info.fusion_type);
    }
}
```

## 7. 融合优势

| 优势 | 说明 |
|------|------|
| 减少内存访问 | 减少中间结果写入/读取 |
| 减少kernel启动开销 | 合并多个操作为一个 |
| 提高缓存利用率 | 数据局部性更好 |
| 自动图简化 | 无需手动优化 |

## 8. 融合限制

1. **多消费者限制**：只有单消费者的激活函数才能融合
2. **形状约束**：融合后形状必须兼容
3. **数值精度**：某些融合可能影响数值精度（如Conv+BatchNorm的融合需要处理BN的均值/方差）

## 9. 未来扩展

- Conv+BatchNorm 融合（需要融合权重）
- Conv+Sigmoid/Tanh 融合
- 多元素融合（如 Conv+BN+ReLU）
- 循环融合（用于RNN/LSTM）
