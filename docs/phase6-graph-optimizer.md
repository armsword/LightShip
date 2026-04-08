# 图优化与执行计划

## 概述

Phase 6 实现了 LightShip 推理引擎的图优化模块和执行计划模块，为模型推理提供了高效的图优化能力和执行调度能力。

## 图优化模块 (ir/optimizer.rs)

### ConstantFolding (常量折叠)

常量折叠是一种编译器优化技术，通过在编译时计算常量表达式来减少运行时的计算量。

**优化策略：**
- `Add(0)` → 恒等变换，可消除
- `Mul(1)` → 恒等变换，可消除
- `Sub(x, x)` → 结果为 0

**实现原理：**
```rust
pub struct ConstantFolding;
impl GraphOptimizer for ConstantFolding {
    fn optimize(&self, graph: &mut Graph) {
        // 识别单输入的 Add/Mul/Sub 节点（简化版本）
        // 真实实现会检查实际的常量值
        let nodes_to_remove: HashSet<NodeId> = graph.nodes.iter()
            .filter(|node| matches!(node.operator_type,
                OperatorType::Add | OperatorType::Mul | OperatorType::Sub))
            .filter(|node| node.inputs.len() == 1)
            .map(|node| node.id)
            .collect();
        graph.retain_nodes(|n| !nodes_to_remove.contains(&n.id));
    }
}
```

### DeadCodeElimination (死代码消除)

移除图中不可达的节点，减少不必要的计算。

**算法：**
1. 从模型输入和输出开始进行 BFS
2. 标记所有可达节点
3. 移除所有不可达节点

```rust
fn find_reachable_nodes(&self, graph: &Graph) -> HashSet<NodeId> {
    let mut reachable = HashSet::new();
    let mut worklist = VecDeque::new();

    // 从输入开始
    for input_name in &self.inputs {
        if let Some(node) = graph.node(input_name) {
            worklist.push_back(node.id);
        }
    }

    // BFS 遍历
    while let Some(node_id) = worklist.pop_front() {
        if reachable.contains(&node_id) { continue; }
        reachable.insert(node_id);
        // ... 添加消费者节点到 worklist
    }
    reachable
}
```

### ShapeInference (形状推导)

根据算子的语义从输入形状推导出输出形状，无需实际执行就能知道张量的维度。

**支持的算子：**

| 算子 | 输入形状 | 输出形状 |
|------|----------|----------|
| Conv2d | [N,C,H,W] | [N,C_out,H_out,W_out] |
| ReLU/Sigmoid/Tanh | 任意 | 保持不变 |
| MaxPool/AvgPool | [N,C,H,W] | [N,C,H/2,W/2] |
| Reshape | 任意 | 从属性获取 |

## 执行计划模块 (executor/)

### ScheduledNode (调度节点)

表示图中一个准备好执行的节点，包含运行时所需的所有信息：

```rust
pub struct ScheduledNode {
    pub node_id: NodeId,           // 原图节点 ID
    pub operator_type: OperatorType, // 算子类型
    pub input_ids: Vec<String>,    // 输入张量 ID
    pub output_ids: Vec<String>,   // 输出张量 ID
    pub backend: BackendType,       // 执行的 backend
    pub fusion: Option<FusionInfo>, // 融合信息
    pub parallelizable: bool,       // 是否可并行
    pub parallel_group: Option<usize>, // 并行组 ID
    pub memory_estimate: usize,    // 内存估计
}
```

### MemoryPlan (内存计划)

管理推理过程中的内存分配和复用：

```rust
pub struct MemoryPlan {
    pub total_memory: usize,
    pub peak_memory: usize,
    pub node_allocations: HashMap<NodeId, MemoryAllocation>,
    pub reuse_pairs: Vec<(NodeId, NodeId)>,  // 可共享内存的节点对
    pub input_sizes: HashMap<String, usize>,
    pub output_sizes: HashMap<String, usize>,
    pub temp_sizes: HashMap<String, usize>,
}
```

### ExecutionPlan (执行计划)

完整的模型执行计划：

```rust
pub struct ExecutionPlan {
    pub nodes: Vec<ScheduledNode>,      // 按拓扑排序的节点
    pub parallel_groups: Vec<ParallelGroup>, // 并行组
    pub memory_plan: MemoryPlan,       // 内存计划
    pub total_cycles: u64,             // 估计周期数
    pub supports_async: bool,          // 是否支持异步
}
```

### Scheduler (调度器)

将 IR 图转换为可执行的计划：

1. **拓扑排序**：确定节点执行顺序
2. **并行化分析**：识别可并行执行的节点
3. **内存规划**：计算最优内存分配
4. **生成执行计划**：创建包含所有运行时信息的 ExecutionPlan

```rust
impl Scheduler {
    pub fn schedule(&self, graph: &Graph) -> ExecutionPlan {
        let order = graph.topological_sort();  // 拓扑排序
        let parallelizable = self.find_parallelizable_nodes(graph, &order);
        // ... 构建 ExecutionPlan
    }
}
```

## Graph 增强

新增方法支持优化过程中的节点管理：

```rust
impl Graph {
    pub fn retain_nodes<F>(&mut self, pred: F)
    where F: FnMut(&Node) -> bool {
        // 保留满足条件的节点，并重建索引
        self.nodes.retain(pred);
        self.rebuild_index();
    }

    pub fn update_node_name(&mut self, old: &str, new: String, id: NodeId) {
        self.node_name_index.remove(old);
        self.node_name_index.insert(new, id);
    }
}
```

## 使用示例

```rust
use lightship_core::ir::optimizer::{ConstantFolding, DeadCodeElimination};
use lightship_core::executor::Scheduler;

// 创建优化器
let constant_folding = ConstantFolding::new();
let dce = DeadCodeElimination::new();

// 应用优化
constant_folding.optimize(&mut graph);
dce.optimize(&mut graph);

// 生成执行计划
let scheduler = Scheduler::new();
let plan = scheduler.schedule(&graph);
```
