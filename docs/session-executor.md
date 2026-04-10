# LightShip Session 和 GraphExecutor 架构

## 1. 概述

Session 是 LightShip 推理引擎的核心 API，提供了模型加载、图准备和推理执行的整体流程。GraphExecutor 负责图的端到端执行。

## 2. 架构层次

```
┌─────────────────────────────────────┐
│           SessionHandle              │  ← 用户API
├─────────────────────────────────────┤
│         GraphExecutor                │  ← 图执行器
├─────────────────────────────────────┤
│  CompiledOperator (HashMap)        │  ← 编译后的算子
├─────────────────────────────────────┤
│         Backend (Arc<dyn>)          │  ← CPU/GPU/NPU后端
└─────────────────────────────────────┘
```

## 3. SessionHandle

### 3.1 核心结构

```rust
pub struct SessionHandle {
    /// Backend for execution (owned via Arc)
    backend: Arc<CpuBackend>,
    /// Graph executor
    executor: GraphExecutor,
    /// Prepared graph
    graph: Option<Graph>,
    /// Input tensor names
    input_names: Vec<String>,
    /// Output tensor names
    output_names: Vec<String>,
}
```

### 3.2 生命周期管理

- `backend: Arc<CpuBackend>` - 使用 Arc 共享后端所有权
- `executor: GraphExecutor` - 持有编译后的算子和执行逻辑
- `graph: Option<Graph>` - 准备阶段的图

### 3.3 主要方法

| 方法 | 功能 |
|------|------|
| `new()` | 创建新 Session |
| `prepare_graph(graph)` | 编译图中的所有算子 |
| `forward(inputs, outputs)` | 执行推理 |

## 4. GraphExecutor

### 4.1 核心结构

```rust
pub struct GraphExecutor {
    backend: Arc<dyn Backend + Send + Sync>,
    compiled_ops: HashMap<String, CompiledOperator>,
}
```

### 4.2 执行流程

```
prepare():
  for node in graph.nodes:
    def = OperatorDef(node.name, node.operator_type)
    compiled = backend.compile_operator(def)
    compiled_ops[node.name] = compiled

execute():
  1. tensor_storage = {}  // 创建张量存储
  2. 插入输入张量
  3. order = topological_sort(graph)
  4. for node_id in order:
       - 获取输入张量
       - 获取/创建输出张量
       - backend.execute(compiled_ops[node.name], inputs, outputs)
       - 存储输出到 tensor_storage
  5. 复制输出到调用者
```

## 5. 执行流程示例

### 5.1 创建 Session

```rust
let session = SessionHandle::new().unwrap();
```

### 5.2 准备图

```rust
let mut graph = Graph::new("test".to_string());
// 添加节点...

// 设置图的输入输出
graph.inputs.push(GraphIO {
    name: "input".into(),
    io: NodeIO { tensor_name: "input".into(), data_type: DataType::F32 },
    is_model_input: true,
    is_model_output: false,
});
graph.outputs.push(GraphIO {
    name: "output".into(),
    io: NodeIO { tensor_name: "output".into(), data_type: DataType::F32 },
    is_model_input: false,
    is_model_output: true,
});

session.prepare_graph(graph).unwrap();
```

### 5.3 执行推理

```rust
let input = Tensor::from_data("input".into(), vec![6], DataType::F32, input_data);
let mut outputs: &mut [(&str, Tensor)] = &mut [
    ("output", Tensor::new("output".into(), vec![6], DataType::F32))
];
session.forward(&[("input", input)], outputs).unwrap();
```

## 6. 多节点图执行

对于图 `input → conv → relu → pool → output`：

```
拓扑排序: [input_node, conv_node, relu_node, pool_node, output_node]

执行顺序:
1. input_node: 生成 "input_tensor"
2. conv_node: 读取 "input_tensor"，生成 "conv_out"
3. relu_node: 读取 "conv_out"，生成 "relu_out"
4. pool_node: 读取 "relu_out"，生成 "pool_out"
5. output_node: 读取 "pool_out"，生成 "output_tensor"
```

## 7. 内存管理

- `tensor_storage: HashMap<String, Tensor>` - 存储所有中间张量
- 输入张量由调用者提供
- 输出张量预分配并注册到 storage
- 执行过程中复用中间结果的内存

## 8. 错误处理

| 错误场景 | 处理方式 |
|---------|---------|
| 节点未找到 | `LightShipError::InvalidParam` |
| 算子未编译 | `LightShipError::InvalidParam` |
| 输入张量缺失 | `LightShipError::InvalidParam` |
| 后端执行失败 | `LightShipError::Backend` |

## 9. 线程安全

- `Backend: Send + Sync` - 后端必须是线程安全的
- `Arc<dyn Backend + Send + Sync>` - 使用 Arc 确保共享访问安全
- `GraphExecutor` - 内部使用 Arc 共享后端

## 10. 后续优化方向

1. **内存复用** - 节点间复用中间张量内存
2. **算子融合** - Conv+ReLU, Conv+BN 融合执行
3. **异步执行** - 支持异步 forward
4. **批量推理** - 批量输入推理
5. **多后端** - 支持 GPU/NPU 后端
