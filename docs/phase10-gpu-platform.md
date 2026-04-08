# GPU/异构后端与跨平台支持

## 概述

Phase 10 实现了 LightShip 的 GPU/异构后端抽象和跨平台支持，包括 GpuBackend trait、Vulkan/Metal 后端接口、NPU 适配层以及 C API 导出。

## GPU 后端架构

### GpuBackend trait

所有 GPU 后端必须实现 GpuBackend trait：

```rust
pub trait GpuBackend: Send + Sync {
    fn backend_type(&self) -> BackendType;
    fn is_available(&self) -> bool;
    fn capabilities(&self) -> GpuCapabilities;

    // 内存管理
    fn allocate_buffer(&self, size: usize) -> Result<GpuBuffer, GpuError>;
    fn deallocate_buffer(&self, buffer: &GpuBuffer) -> Result<(), GpuError>;

    // 着色器/内核编译
    fn compile_shader(&self, source: &str, options: &ShaderOptions)
        -> Result<CompiledShader, GpuError>;

    // 资源同步
    fn create_fence(&self) -> Result<GpuFence, GpuError>;
    fn wait_for_fence(&self, fence: &GpuFence, timeout_ns: u64) -> Result<bool, GpuError>;

    // 命令执行
    fn create_command_buffer(&self) -> Result<CommandBuffer, GpuError>;
    fn submit_commands(&self, buffer: &CommandBuffer) -> Result<(), GpuError>;
}
```

### GpuCapabilities

```rust
pub struct GpuCapabilities {
    pub backend: BackendType,
    pub supports_float16: bool,
    pub supports_int8: bool,
    pub supports_storage_buffer: bool,
    pub max_workgroup_size: usize,
    pub max_memory_size: usize,
    pub unified_memory: bool,
}
```

## Vulkan 后端

### VulkanBackend

```rust
pub struct VulkanBackend {
    device: Arc<VulkanDevice>,
    queue: VulkanQueue,
    shader_compiler: Arc<ShaderCompiler>,
    memory_allocator: Arc<VulkanMemoryAllocator>,
}

impl VulkanBackend {
    pub fn new(instance: &VulkanInstance) -> Result<Self, VulkanError> {
        // 选择物理设备和创建逻辑设备
        let physical_device = instance.enumerate_physical_devices()?
            .into_iter()
            .max_by_key(|dev| dev.score())
            .ok_or(VulkanError::NoDevice)?;

        let (device, queue) = physical_device.create_device()?;
        Ok(Self { device, queue, ... })
    }
}
```

### VulkanShader

```rust
pub struct VulkanShader {
    module: vk::ShaderModule,
    entry_point: String,
    specialization_info: Option<vk::SpecializationInfo>,
}
```

### VulkanBuffer

```rust
pub struct VulkanBuffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    size: usize,
    mapped: Option<*mut std::ffi::c_void>,
}
```

## Metal 后端

### MetalBackend

```rust
pub struct MetalBackend {
    device: Arc<mtl::Device>,
    command_queue: Arc<mtl::CommandQueue>,
    library: Arc<mtl::Library>,
    shader_compiler: Arc<ShaderCompiler>,
}

impl MetalBackend {
    pub fn new() -> Result<Self, MetalError> {
        let device = mtl::SystemDefaultDevice::ok_or(MetalError::NoDevice)?;
        let command_queue = device.new_command_queue();
        Ok(Self { device, command_queue, ... })
    }
}
```

### MetalShader

```rust
pub struct MetalShader {
    library: mtl::Library,
    function: mtl::Function,
    pipeline_state: mtl::ComputePipelineState,
}
```

## NPU 后端 (Apple Neural Engine)

### AppleANEBackend

```rust
pub struct AppleANEBackend {
    context: ANEContext,
    compiled_graphs: HashMap<GraphId, CompiledANEGraph>,
}

impl AppleANEBackend {
    pub fn new() -> Result<Self, ANEError> {
        // 初始化 ANEContext
        let context = ANEContext::create()?;
        Ok(Self { context, compiled_graphs: HashMap::new() })
    }
}
```

### 支持的 ANE 操作

```rust
pub enum ANEOperation {
    Conv2d { kernel_size: (u32, u32), stride: (u32, u32) },
    FullyConnected { output_size: u32 },
    BatchNorm { epsilon: f32 },
    Activation { activation_type: ANEActivationType },
    Pooling { pool_type: ANEPoolType, window_size: (u32, u32) },
}
```

## 跨平台线程池

### ThreadPoolConfig

```rust
#[derive(Debug, Clone)]
pub struct ThreadPoolConfig {
    pub num_threads: usize,      // 0 = 自动检测
    pub name_prefix: String,
    pub stack_size: Option<usize>,
    pub use_affinity: bool,
}
```

### ThreadPool

```rust
pub struct ThreadPool {
    handles: Vec<thread::JoinHandle<()>>,
    task_sender: std::sync::mpsc::Sender<Task>,
    config: ThreadPoolConfig,
}

impl ThreadPool {
    pub fn with_config(config: ThreadPoolConfig) -> Self {
        let num_threads = if config.num_threads == 0 {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
        } else {
            config.num_threads
        };
        // 创建线程池...
    }

    pub fn submit<F>(&self, task: F)
    where
        F: FnOnce() + Send + 'static,
    {
        let _ = self.task_sender.send(Box::new(task));
    }
}
```

### ParallelExecutor

```rust
pub struct ParallelExecutor {
    pool: ThreadPool,
    chunk_size: usize,
}

impl ParallelExecutor {
    pub fn parallel_for<F>(&self, range: std::ops::Range<usize>, f: F)
    where
        F: Fn(usize) + Send + Sync + 'static,
    {
        let f = Arc::new(f);
        let chunk_size = self.chunk_size.max(1);

        for chunk_start in (0..range.len()).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(range.len());
            let f = Arc::clone(&f);

            self.pool.submit(move || {
                for i in chunk_start..chunk_end {
                    f(range.start + i);
                }
            });
        }
    }
}
```

## CPU 信息检测

### CpuInfo

```rust
pub struct CpuInfo {
    pub vendor: String,
    pub brand: String,
    pub num_cores: usize,
    pub num_threads: usize,
    pub features: CpuFeatures,
}

pub struct CpuFeatures {
    pub sse4_2: bool,
    pub avx: bool,
    pub avx2: bool,
    pub avx512f: bool,
    pub neon: bool,           // ARM
    pub sve: bool,             // ARM SVE
}
```

## C API 导出

### 核心类型

```rust
pub type LightShipEngine = *mut std::ffi::c_void;
pub type LightShipModel = *mut std::ffi::c_void;
pub type LightShipSession = *mut std::ffi::c_void;
pub type LightShipTensor = *mut std::ffi::c_void;
```

### 错误处理

```c
typedef enum {
    LIGHTSHIP_SUCCESS = 0,
    LIGHTSHIP_ERROR_INVALID_ARGUMENT = 1,
    LIGHTSHIP_ERROR_OUT_OF_MEMORY = 2,
    LIGHTSHIP_ERROR_MODEL_NOT_FOUND = 3,
    LIGHTSHIP_ERROR_BACKEND_UNAVAILABLE = 4,
    // ...
} LightShipError;
```

### 核心 API

```c
// 引擎生命周期
LightShipError LightShipEngine_Create(
    LightShipEngine* out_engine,
    LightShipLogLevel log_level
);
LightShipError LightShipEngine_Destroy(LightShipEngine engine);

// 模型加载
LightShipError LightShipEngine_LoadModel(
    LightShipEngine engine,
    const char* path,
    LightShipModel* out_model
);
LightShipError LightShipModel_Destroy(LightShipModel model);

// 会话管理
LightShipError LightShipEngine_CreateSession(
    LightShipEngine engine,
    LightShipModel model,
    const LightShipSessionConfig* config,
    LightShipSession* out_session
);
LightShipError LightShipSession_Destroy(LightShipSession session);
LightShipError LightShipSession_Run(
    LightShipSession session,
    const LightShipTensor* inputs,
    int input_count,
    LightShipTensor* outputs,
    int output_count
);

// 张量操作
LightShipError LightShipTensor_Create(
    const int64_t* shape,
    int num_dims,
    LightShipDataType data_type,
    LightShipTensor* out_tensor
);
LightShipError LightShipTensor_Destroy(LightShipTensor tensor);
LightShipError LightShipTensor_SetData(
    LightShipTensor tensor,
    const void* data,
    size_t size
);
LightShipError LightShipTensor_GetData(
    LightShipTensor tensor,
    void* out_data,
    size_t* in_out_size
);
```

## 后端选择策略

```rust
pub enum BackendSelection {
    Auto,       // 自动选择最快后端
    CpuOnly,    // 仅 CPU
    GpuOnly,    // 仅 GPU
    NpuOnly,    // 仅 NPU
    Preferred { preferred: BackendType, fallback: BackendType },
}

impl BackendSelection {
    pub fn select(&self, available: &[BackendType]) -> Option<BackendType> {
        match self {
            BackendSelection::Auto => available.first().copied(),
            BackendSelection::CpuOnly => available.iter()
                .find(|&&t| t == BackendType::Cpu).copied(),
            BackendSelection::Preferred { preferred, fallback } => {
                available.iter()
                    .find(|&&t| t == *preferred)
                    .or_else(|| available.iter().find(|&&t| t == *fallback))
                    .copied()
            }
            // ...
        }
    }
}
```

## 使用示例

### 多后端自动选择

```rust
use lightship_core::{Engine, EngineConfig, BackendSelection};

let engine = Engine::new(EngineConfig {
    backend_selection: BackendSelection::Auto,
    ..Default::default()
})?;

let available = engine.available_backends();
println!("可用后端: {:?}", available);
```

### GPU 推理

```rust
use lightship_core::{Session, SessionConfig, BackendType};

let config = SessionConfig {
    backend: BackendType::Vulkan,
    enable_profiling: true,
    ..Default::default()
};

let session = engine.create_session(&model, config)?;
session.run(&inputs, &outputs)?;
```

### C API 使用

```c
#include "lightship.h"

LightShipEngine engine;
LightShipEngine_Create(&engine, LIGHTSHIP_LOG_LEVEL_INFO);

LightShipModel model;
LightShipEngine_LoadModel(engine, "model.onnx", &model);

LightShipSession session;
LightShipSessionConfig config = { .backend = LIGHTSHIP_BACKEND_CPU };
LightShipEngine_CreateSession(engine, model, &config, &session);

// 创建输入张量
LightShipTensor input;
int64_t shape[] = {1, 3, 224, 224};
LightShipTensor_Create(shape, 4, LIGHTSHIP_DATA_TYPE_FLOAT32, &input);
LightShipTensor_SetData(input, image_data, sizeof(image_data));

// 执行推理
LightShipTensor outputs[1];
LightShipSession_Run(session, &input, 1, outputs, 1);

// 获取结果
float result[1000];
size_t size = sizeof(result);
LightShipTensor_GetData(outputs[0], result, &size);
```
