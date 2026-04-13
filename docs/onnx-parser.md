# LightShip ONNX 解析器原理

## 1. 概述

ONNX (Open Neural Network Exchange) 是一种开放的神经网络模型交换格式。LightShip 的 ONNX 解析器负责将 ONNX 模型转换为内部 IR 表示。

## 2. ONNX 文件结构

### 2.1 文件格式

ONNX 文件有两种格式：
1. **ZIP 压缩格式** (`.onnx`) - 包含 `graph.pb` 和元数据
2. **原始 Protobuf** (`.pb`) - 直接包含序列化后的 ModelProto

### 2.2 Protobuf 结构

```
ModelProto
├── ir_version (field 1, varint)
├── producer_name (field 2, string)
├── producer_version (field 3, string)
├── domain (field 4, string)
├── model_version (field 5, varint)
├── doc_string (field 6, string)
├── opset_import (field 8, repeated)
└── graph (field 7, message)
    ├── node (field 1, repeated NodeProto)
    ├── name (field 2, string)
    ├── initializer (field 5, repeated TensorProto)
    ├── input (field 11, repeated ValueInfoProto)
    ├── output (field 12, repeated ValueInfoProto)
    └── ...
```

## 3. Protobuf 编码基础

### 3.1 Tag 编码

每个字段以 tag 开头，格式为：
```
tag = (field_number << 3) | wire_type
```

- `field_number`: 字段编号 (1-15 用 1 byte, 16-2047 用 2 bytes)
- `wire_type`: 数据类型 (0-5)

### 3.2 Wire Type

| Type | Value | 说明 |
|------|-------|------|
| Varint | 0 | 可变长整数编码 |
| 64-bit | 1 | 8 字节固定长度 |
| Length-delimited | 2 | 字符串、字节数组、嵌套消息 |
| Start group | 3 | (已废弃) |
| End group | 4 | (已废弃) |
| 32-bit | 5 | 4 字节固定长度 |

### 3.3 Varint 编码

小数值 (0-127) 用 1 byte:
```
byte[0] = value
```

大数值用多 byte:
```
byte[i] = (value >> (7*i)) & 0x7F | 0x80  (for i > 0)
```

## 4. 解析实现

### 4.1 解析入口

```rust
fn load_from_bytes(&self, bytes: &[u8], format: ModelFormat) -> Result<ModelFile>
```

1. 检测文件格式 (ZIP 或 原始 protobuf)
2. 提取 graph.pb (如果是 ZIP)
3. 调用 `parse_graph_protobuf`

### 4.2 ModelProto 解析

```rust
fn parse_graph_protobuf(&self, data: &[u8]) -> Result<ModelFile>
```

按顺序读取 tag，处理各个字段：
- field 1-6: 元数据 (跳过)
- field 7: graph (调用 `parse_nodes_from_graph`)
- field 8: opset_import (跳过)

### 4.3 GraphProto 解析

```rust
fn parse_nodes_from_graph(&self, data: &[u8], offset, end) -> Result<(Vec<Node>, Vec<GraphIO>, Vec<GraphIO>, HashMap<String, Arc<Tensor>>)>
```

关键字段处理：

| Field | Wire | 类型 | 处理 |
|-------|------|------|------|
| 1 | 2 | NodeProto | 调用 `parse_node` |
| 5 | 2 | TensorProto | 调用 `parse_tensor_proto` (initializers) |
| 11 | 2 | ValueInfoProto | 调用 `parse_value_info_proto` (inputs) |
| 12 | 2 | ValueInfoProto | 调用 `parse_value_info_proto` (outputs) |

### 4.4 NodeProto 解析

```rust
fn parse_node(&self, data: &[u8], offset, node_id) -> Result<(Node, usize)>
```

ONNX NodeProto 字段：

| Field | Wire | 说明 | LightShip 映射 |
|-------|------|------|----------------|
| 1 | 2 | input (repeated string) | `node.inputs` |
| 2 | 2 | output (repeated string) | `node.outputs` |
| 3 | 2 | name (string) | `node.name` |
| 4 | 2 | op_type (string) | `node.operator_type` |
| 5 | 2 | attribute | 跳过 |

### 4.5 TensorProto 解析 (Initializers)

```rust
fn parse_tensor_proto(&self, data: &[u8], offset, end) -> Result<Option<Tensor>>
```

TensorProto 字段：

| Field | Wire | 说明 |
|-------|------|------|
| 1 | 0/2 | dims (repeated int64) |
| 2 | 0 | data_type (int) |
| 4 | 2 | float_data (repeated float) |
| 8 | 2 | name (string) |
| 9 | 2 | raw_data (bytes) |

### 4.6 ValueInfoProto 解析

```rust
fn parse_value_info_proto(&self, data: &[u8], offset, end, is_input) -> Result<Option<GraphIO>>
```

ValueInfoProto 字段：

| Field | Wire | 说明 |
|-------|------|------|
| 1 | 2 | name (string) |
| 2 | 2 | type (TypeProto) |

TypeProto → TensorTypeProto → elem_type:
- 1 = FLOAT (F32)
- 7 = INT32 (I32)

## 5. 解析流程图

```
load_from_bytes(bytes)
    │
    ├─[ZIP格式]─> 解压 ─> 读取 graph.pb
    │
    └─[Protobuf]─> 直接使用 bytes
            │
            v
    parse_graph_protobuf(data)
            │
            ├─ 跳过 field 1-6 (元数据)
            │
            ├─ field 7: graph
            │       │
            │       v
            │   parse_nodes_from_graph()
            │       │
            │       ├─ field 1: NodeProto ──────────> parse_node()
            │       │                                   │
            │       │                                   ├─ field 1: inputs
            │       │                                   ├─ field 2: outputs
            │       │                                   ├─ field 3: name
            │       │                                   └─ field 4: op_type
            │       │
            │       ├─ field 5: TensorProto ──────────> parse_tensor_proto()
            │       │                                   │
            │       │                                   ├─ field 1: dims
            │       │                                   ├─ field 2: data_type
            │       │                                   ├─ field 4/9: data
            │       │                                   └─ field 8: name
            │       │
            │       ├─ field 11: ValueInfoProto ─────> parse_value_info_proto() → inputs
            │       │
            │       └─ field 12: ValueInfoProto ─────> parse_value_info_proto() → outputs
            │
            v
    ModelFile { graph, variables, inputs, outputs }
```

## 6. 算子类型映射

```rust
pub fn from_onnx_op(op_type: &str) -> OperatorType {
    match op_type {
        "Conv"           => OperatorType::Conv2d,
        "Relu"           => OperatorType::ReLU,
        "MaxPool"        => OperatorType::MaxPool2d,
        "MatMul"         => OperatorType::MatMul,
        "Add"            => OperatorType::Add,
        "Softmax"        => OperatorType::Softmax,
        "BatchNormalization" => OperatorType::BatchNorm,
        // ...
        _                => OperatorType::Custom,
    }
}
```

## 7. 数据类型映射

```rust
fn from_onnx_dtype(dtype: i32) -> Option<DataType> {
    match dtype {
        1  => Some(DataType::F32),    // FLOAT
        7  => Some(DataType::I32),    // INT32
        10 => Some(DataType::U16),     // UINT16
        // ...
        _  => None,
    }
}
```

## 8. 已知限制

1. **稀疏张量**: 未实现 `sparse_initializer` 解析
2. **量化注解**: 未实现 `quantization_annotation` 解析
3. **属性解析**: NodeProto 的 `attribute` 字段被跳过
4. **维度参数**: ValueInfoProto 中的动态维度 (`dim_param`) 未处理

## 9. 调试技巧

### 9.1 查看原始字节

```bash
xxd -l 50 model.onnx
```

### 9.2 Python 调试

```python
import onnx
from onnx.onnx_ml_pb2 import ModelProto

with open('model.onnx', 'rb') as f:
    data = f.read()

model = ModelProto()
model.ParseFromString(data)

# 检查模型结构
print(f"IR version: {model.ir_version}")
print(f"Producer: {model.producer_name}")
print(f"Graph name: {model.graph.name}")
print(f"Nodes: {len(model.graph.node)}")
for node in model.graph.node:
    print(f"  {node.op_type}: {list(node.input)} -> {list(node.output)}")
```

### 9.3 Protobuf 手动解析

```python
def parse_tag(data, offset):
    tag = data[offset]
    field = tag >> 3
    wire = tag & 7
    return field, wire

def parse_varint(data, offset):
    result = 0
    shift = 0
    pos = offset
    while data[pos] & 0x80:
        result |= (data[pos] & 0x7F) << shift
        pos += 1
        shift += 7
    result |= data[pos] << shift
    return result, pos + 1
```
