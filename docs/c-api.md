# LightShip C API 设计与实现

## 1. 概述

C API 是 LightShip 推理引擎的跨语言绑定接口，提供 C 兼容的函数接口供 Python、Android NDK、iOS 等平台调用。C API 使用 opaque pointer（不透明指针）模式，将内部实现细节完全封装。

## 2. 架构设计

### 2.1 设计原则

- **零成本抽象**：C API 函数签名字面量对应 Rust 实现，无额外运行时开销
- **内存安全**：通过 opaque pointer 避免 C 调用方直接访问内部数据
- **错误处理**：统一的错误码枚举，通过 `LightShipErrorCode` 返回状态
- **线程安全**：内部状态管理使用 `HashMap` 存储，支持多线程并发访问

### 2.2 核心组件

```
┌─────────────────────────────────────────────────────────────┐
│                      C API Layer                            │
│  LightShipEngine_Create / Destroy / LoadModel               │
│  LightShipSession_Create / Run / GetTiming                  │
│  LightShipTensor_Create / Destroy / GetData                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Internal State (ApiState)                 │
│  engines: HashMap<usize, EngineState>                       │
│  models: HashMap<usize, ModelState>                         │
│  sessions: HashMap<usize, SessionState>                     │
│  tensors: HashMap<usize, TensorState>                       │
└─────────────────────────────────────────────────────────────┘
```

## 3. 类型映射

### 3.1 Handle 类型

| C API Type | Rust Internal Type | 说明 |
|------------|-------------------|------|
| `LightShipEngine` | `*mut c_void` | 引擎句柄，存储引擎 ID |
| `LightShipModel` | `*mut c_void` | 模型句柄，存储模型 ID |
| `LightShipSession` | `*mut c_void` | 会话语柄，存储会话 ID |
| `LightShipTensor` | `*mut c_void` | 张量句柄，存储张量 ID |

### 3.2 数据类型映射

| C API `LightShipDataType` | Rust `DataType` |
|---------------------------|-----------------|
| `F32` (0) | `F32` |
| `F16` (1) | `F16` |
| `I8` (3) | `I8` |
| `I32` (5) | `I32` |
| `QUInt8` (12) | `QUInt8` |
| `QInt8` (13) | `QInt8` |

### 3.3 后端类型映射

| C API `LightShipBackend` | Rust `BackendType` |
|-------------------------|-------------------|
| `CPU` (0) | `CPU` |
| `GPU` (1) | `GPU` |
| `NPU` (2) | `NPU` |
| `Vulkan` (3) | `Vulkan` |
| `Metal` (4) | `Metal` |

## 4. 内部状态管理

### 4.1 ApiState 结构

```rust
struct ApiState {
    engines: HashMap<usize, EngineState>,
    models: HashMap<usize, ModelState>,
    sessions: HashMap<usize, SessionState>,
    tensors: HashMap<usize, TensorState>,
    next_id: usize,
    last_error: String,
}
```

- `next_id`：原子递增 ID 生成器，确保每个资源有唯一标识
- `last_error`：线程安全的错误消息存储
- 使用 `HashMap` 存储各种资源，key 为资源 ID

### 4.2 资源状态结构

```rust
struct EngineState {
    id: usize,
    log_level: LightShipLogLevel,
}

struct ModelState {
    id: usize,
    name: String,
    num_inputs: u32,
    num_outputs: u32,
    is_quantized: bool,
}

struct SessionState {
    id: usize,
    engine_id: usize,
    model_id: usize,
    backend: LightShipBackend,
    num_threads: usize,
    last_timing: Option<LightShipTiming>,
}

struct TensorState {
    id: usize,
    shape: Vec<usize>,
    data_type: LightShipDataType,
    data: Vec<u8>,
}
```

## 5. 核心函数实现

### 5.1 引擎管理

#### LightShipEngine_Create

```c
LightShipErrorCode LightShipEngine_Create(
    LightShipEngine* out_engine,
    LightShipLogLevel log_level
);
```

**实现逻辑**：
1. 参数校验：`out_engine` 不能为空
2. 生成唯一 ID：`state.next_id()`
3. 创建 `EngineState` 并存入 `HashMap`
4. 将 ID 指针转换后写入 `*out_engine`

#### LightShipEngine_Destroy

```c
LightShipErrorCode LightShipEngine_Destroy(LightShipEngine engine);
```

**实现逻辑**：
1. 参数校验：`engine` 不能为空
2. 查找并移除对应的 `EngineState`
3. 清理关联的 sessions（会话依赖引擎）
4. 返回成功或 `InvalidHandle` 错误

### 5.2 模型加载

#### LightShipEngine_LoadModel

```c
LightShipErrorCode LightShipEngine_LoadModel(
    LightShipEngine engine,
    const char* path,
    LightShipModel* out_model
);
```

**实现逻辑**：
1. 验证 `engine` 和 `path` 有效性
2. 检查引擎是否存在
3. 解析路径字符串（处理 UTF-8 编码）
4. 创建 `ModelState`（目前为 stub，元数据待 ONNX 解析完成后填充）
5. 返回模型句柄

### 5.3 会话管理

#### LightShipEngine_CreateSession

```c
LightShipErrorCode LightShipEngine_CreateSession(
    LightShipEngine engine,
    LightShipModel model,
    const LightShipSessionConfig* config,
    LightShipSession* out_session
);
```

**实现逻辑**：
1. 验证所有参数非空
2. 检查引擎和模型存在性
3. 从 `config` 提取后端类型、线程数等配置
4. 创建 `SessionState` 并关联引擎和模型

#### LightShipSession_Run

```c
LightShipErrorCode LightShipSession_Run(
    LightShipSession session,
    const LightShipTensor* inputs,
    uint32_t num_inputs,
    const LightShipTensor* outputs,
    uint32_t num_outputs
);
```

**实现逻辑**：
1. 验证会话和输入输出有效性
2. 目前为 stub，记录假时序数据
3. 实际推理执行需 Phase 3 (CPU 后端) 实现后完成

### 5.4 张量操作

#### LightShipTensor_Create

```c
LightShipErrorCode LightShipTensor_Create(
    const LightShipShape* shape,
    LightShipDataType data_type,
    LightShipTensor* out_tensor
);
```

**实现逻辑**：
1. 验证 shape 和 out_tensor 非空
2. 计算元素数量：`prod(dims)`
3. 根据 data_type 获取字节大小
4. 分配 `vec![0u8; total_size]`
5. 创建 `TensorState` 并存储

#### LightShipTensor_GetData

```c
LightShipErrorCode LightShipTensor_GetData(
    LightShipTensor tensor,
    void** out_data
);
```

**实现逻辑**：
1. 验证参数非空
2. 查找张量状态
3. 返回内部数据缓冲区的原始指针

**注意**：C 调用方获得指针后，张量生命周期由 C API 管理，张量销毁前指针有效。

#### LightShipTensor_GetShape

```c
LightShipErrorCode LightShipTensor_GetShape(
    LightShipTensor tensor,
    LightShipShape* out_shape
);
```

**实现逻辑**：
1. 查找张量状态
2. 复制 shape 到输出参数

## 6. 错误处理

### 6.1 错误码枚举

```rust
pub enum LightShipErrorCode {
    Success = 0,
    Unknown = 1,
    InvalidArgument = 2,
    InvalidHandle = 3,
    ModelNotFound = 4,
    ModelParseError = 5,
    BackendNotAvailable = 6,
    // ...
}
```

### 6.2 错误消息传递

```c
LightShipErrorCode LightShipEngine_GetLastError(
    LightShipEngine engine,
    const char** out_message
);
```

C 调用方通过此函数获取最后一条错误消息的 UTF-8 指针。

## 7. 内存管理约定

### 7.1 所有权模型

- **创建函数**（`_Create`, `_LoadModel`）：C API 分配内存，返回句柄
- **销毁函数**（`_Destroy`）：C API 释放内存
- **调用方职责**：必须调用对应的 `_Destroy` 函数避免内存泄漏

### 7.2 句柄失效

调用 `_Destroy` 后，相同句柄再次使用会返回 `InvalidHandle`。

## 8. 使用示例

### 8.1 C 语言调用

```c
#include "lightship.h"

int main() {
    LightShipEngine engine;
    LightShipModel model;
    LightShipSession session;
    LightShipTensor input, output;
    LightShipTiming timing;

    // 创建引擎
    LightShipEngine_Create(&engine, LIGHTSHIP_LOG_LEVEL_INFO);

    // 加载模型
    LightShipEngine_LoadModel(engine, "/path/to/model.onnx", &model);

    // 创建会话
    LightShipSessionConfig config = {
        .backend = LIGHTSHIP_BACKEND_CPU,
        .num_threads = 4,
    };
    LightShipEngine_CreateSession(engine, model, &config, &session);

    // 创建输入张量
    LightShipShape input_shape = {{1, 3, 224, 224}};
    LightShipTensor_Create(&input_shape, LIGHTSHIP_DATATYPE_F32, &input);

    // 运行推理
    LightShipSession_Run(session, &input, 1, &output, 1);

    // 获取时序
    LightShipSession_GetTiming(session, &timing);
    printf("Inference time: %lu us\n", timing.execution_time_us);

    // 清理资源
    LightShipTensor_Destroy(input);
    LightShipTensor_Destroy(output);
    LightShipSession_Destroy(session);
    LightShipModel_Destroy(model);
    LightShipEngine_Destroy(engine);

    return 0;
}
```

### 8.2 Android JNI 调用

```java
public class LightShipDemo {
    static {
        System.loadLibrary("lightship_jni");
    }

    public native int createEngine();
    public native int loadModel(String modelPath);
    public native int createSession(int numThreads, boolean useGPU);
    public native int run(ByteBuffer[] inputs, ByteBuffer[] outputs);
    public native long[] getTiming();

    public static void main(String[] args) {
        LightShipDemo demo = new LightShipDemo();
        demo.createEngine();
        demo.loadModel("/sdcard/model.onnx");
        demo.createSession(4, false);

        // 运行推理...
    }
}
```

## 9. 已知限制

1. **模型加载**：目前为 stub，实际 ONNX 解析需 Phase 4 完成
2. **推理执行**：目前为 stub，实际执行需 Phase 3 (CPU 后端) 完成
3. **错误消息**：目前存储在 static 变量，非线程安全
4. **张量数据**：目前仅支持零初始化，不支持预填充数据

## 10. 后续计划

- Phase 3: 实现 CPU 后端，使 `LightShipSession_Run` 可真正执行推理
- Phase 4: 实现完整 ONNX 解析，使 `LightShipEngine_LoadModel` 可加载真实模型
- Phase 10: 实现 Vulkan/Metal 后端扩展
