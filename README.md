# LightShip

一款用 Rust 编写的**轻量级边缘侧神经网络推理引擎**，灵感来源于 MNN（阿里巴巴的移动端推理引擎）。专为资源受限的嵌入式设备和边缘计算场景设计，提供高 performance、低内存占用的模型部署能力。

## 核心特性

- **高 Performance**: SIMD 硬件加速 (SSE/AVX/NEON)，多线程调度，GEMM 分块优化
- **低内存占用**: Buddy + Slab 混合内存分配器，减少碎片
- **多后端支持**: CPU / GPU / Metal / Vulkan / NPU
- **40+ 算子支持**: Conv2d, MatMul, Attention, LayerNorm, BatchNorm, Softmax, GELU 等
- **多模型格式**: 原生格式、ONNX
- **量化支持**: INT8 / FP16 per-tensor / per-channel 量化
- **C API 接口**: 便于 C/C++ 项目集成
- **TDD 开发**: 完整的测试覆盖

## 架构概览

```
LightShip
├── lightship-core/          # 核心推理引擎库
│   └── src/
│       ├── api/             # 对外 API 层 (Engine, Session, Tensor)
│       ├── ir/              # 中间表示 (Graph, Operator, Tensor)
│       ├── backend/         # 计算后端 (CPU, GPU, Metal)
│       ├── operator/        # 算子实现 (Conv, GEMM, Attention...)
│       ├── model/           # 模型加载 (ONNX 解析器)
│       ├── executor/        # 图执行器与调度
│       ├── memory/          # 内存管理 (Allocator, Pool)
│       ├── platform/        # 平台适配 (SIMD, ThreadPool)
│       ├── cpu/             # CPU 特定优化
│       └── common/          # 公共基础 (Error, Logger, Types)
├── lightship-tools/         # 工具集
├── examples/                # 示例代码
├── c-api/                   # C API 接口
└── docs/                    # 技术文档
```

## 快速开始

### Rust 项目依赖

```toml
[dependencies]
lightship-core = { path = "./lightship-core" }
```

### 基本用法

```rust
use lightship_core::{Engine, Tensor};

let engine = Engine::from_file("model.mnn")?;
let mut session = engine.create_session()?;

let input = Tensor::from_slice::<f32>(&input_data, &[batch, channels, height, width])?;
session.set_input("input", input)?;

session.run()?;

let output = session.get_output("output")?;
```

### C API

```c
#include "lightship.h"

LSEngine engine;
ls_engine_from_file(&engine, "model.mnn");

LSSession session;
ls_session_create(engine, &session);

LSTensor input;
ls_tensor_from_data(&input, data, shape, 4, LS_DATA_TYPE_F32);
ls_session_set_input(session, "input", &input);

ls_session_run(session);

LSTensor output;
ls_session_get_output(session, "output", &output);
```

## 支持的算子

### 卷积与池化
| 算子 | 说明 |
|------|------|
| Conv2d | 2D 卷积，支持分组、空洞 |
| ConvTranspose2d | 转置卷积 |
| MaxPool2d / AvgPool2d | 最大/平均池化 |
| GlobalAvgPool2d / GlobalMaxPool2d | 全局池化 |

### 矩阵运算
| 算子 | 说明 |
|------|------|
| MatMul | 矩阵乘法 |
| GEMM | 通用矩阵乘法 (三层分块优化) |
| FullyConnected | 全连接层 |

### 归一化
| 算子 | 说明 |
|------|------|
| BatchNorm | 批归一化 |
| LayerNorm | 层归一化 |
| InstanceNorm | 实例归一化 |

### 注意力机制
| 算子 | 说明 |
|------|------|
| SelfAttention | 自注意力 |
| MultiHeadAttention | 多头注意力 |

### 激活函数
| 算子 | 说明 |
|------|------|
| ReLU / ReLU6 | 线性整流 |
| Sigmoid | Sigmoid |
| Tanh | 双曲正切 |
| GELU | Gaussian Error Linear Unit |
| SiLU / Swish | Sigmoid Linear Unit |
| Softmax | Softmax |

### 形状操作
| 算子 | 说明 |
|------|------|
| Reshape / Flatten | 形状变换 |
| Transpose | 转置 |
| Concat / Split | 拼接/分割 |
| Slice / Pad | 切片/填充 |
| Squeeze / Unsqueeze | 维度压缩/扩展 |

### 广播运算
| 算子 | 说明 |
|------|------|
| Add / Sub / Mul / Div | 逐元素运算 |
| Broadcast | 广播 |

## 技术亮点

### SIMD 优化
- 运行时 CPU feature 检测 (SSE/AVX/NEON)
- 算子级 SIMD 向量化 (ReLU, Sigmoid, Softmax, GEMM)
- 查找表 + 插值优化 exp 函数

### 多线程调度
- 基于 rayon 的数据并行
- 算子级并行调度
- 与 CPU Backend 深度集成

### 内存优化
- Buddy + Slab 混合分配器
- 静态/临时/输入/输出张量生命周期管理
- 内存池复用

### 图优化
- 常量折叠
- 死代码消除
- 算子融合 (Conv+BN, Conv+ReLU)

### Winograd 算法
- F(2x2, 3x3) 优化算法
- 减少乘法运算次数

## 性能对比

基于标准 benchmark 测试，LightShip 在多个场景下性能优于 MNN：

| 场景 | 性能提升 |
|------|----------|
| Conv2d | +15% |
| GEMM | +20% |
| Softmax | +10x (SIMD exp) |
| ReLU | +10x (bytes 操作) |

详见 [性能报告](./docs/lightship-vs-mnn-comparison.md)

## 文档

详细的技术原理文档位于 `docs/` 目录：

| 文档 | 说明 |
|------|------|
| [batchnorm.md](./docs/batchnorm.md) | BatchNorm 原理与实现 |
| [cpu-backend-comparison.md](./docs/cpu-backend-comparison.md) | CPU 后端性能对比 |
| [gemm-block-optimization.md](./docs/gemm-block-optimization.md) | GEMM 分块优化原理 |
| [lightship-vs-mnn-comparison.md](./docs/lightship-vs-mnn-comparison.md) | 与 MNN 对比分析 |
| [multi-threading-design.md](./docs/multi-threading-design.md) | 多线程调度设计 |
| [onnx-parser.md](./docs/onnx-parser.md) | ONNX 解析器原理 |
| [operator-fusion.md](./docs/operator-fusion.md) | 算子融合策略 |
| [simd-optimization.md](./docs/simd-optimization.md) | SIMD 优化详解 |
| [softmax-implementation.md](./docs/softmax-implementation.md) | Softmax 实现原理 |
| [winograd-algorithm.md](./docs/winograd-algorithm.md) | Winograd 算法详解 |

## 项目结构

```
LightShip/
├── Cargo.toml              # Workspace 配置
├── lightship-core/         # 核心库
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs
│       ├── api/            # Engine, Session, Tensor API
│       ├── ir/             # Graph, Node, Operator, Tensor
│       ├── backend/        # Backend trait, CPU/GPU/Metal
│       ├── operator/       # 算子实现
│       ├── model/          # ONNX 解析器
│       ├── executor/       # 图执行器
│       ├── memory/         # 内存管理
│       ├── platform/       # SIMD, ThreadPool
│       ├── cpu/            # CPU 优化
│       └── common/        # Error, Logger, Types
├── lightship-tools/        # 工具集
├── examples/               # 示例
├── c-api/                  # C API
│   └── include/
│       └── lightship.h
├── docs/                   # 技术文档
└── tests/                  # 集成测试
```

## License

MIT License
