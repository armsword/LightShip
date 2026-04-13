//! C API function implementations
//!
//! These are C-compatible function implementations for the LightShip API.

use std::sync::RwLock;
use std::collections::HashMap;
use std::ffi::CStr;
use std::os::raw::c_char;

use super::{LightShipBackend, LightShipDataType, LightShipInferenceMode, LightShipLogLevel, LightShipSessionConfig, LightShipShape, LightShipTiming};
use super::error::LightShipErrorCode;
use crate::common::types::{DataType, StorageLayout, BackendType};
use crate::ir::Tensor;

// ============================================================================
// Internal State Management
// ============================================================================

/// Global state for C API
static mut C_API_STATE: Option<ApiState> = None;

/// API state container
struct ApiState {
    /// Created engines
    engines: HashMap<usize, EngineState>,
    /// Created models
    models: HashMap<usize, ModelState>,
    /// Created sessions
    sessions: HashMap<usize, SessionState>,
    /// Created tensors
    tensors: HashMap<usize, TensorState>,
    /// Next available ID
    next_id: usize,
    /// Last error message
    last_error: String,
}

/// Engine internal state
struct EngineState {
    /// Engine ID
    id: usize,
    /// Log level
    log_level: LightShipLogLevel,
}

/// Model internal state
struct ModelState {
    /// Model ID
    id: usize,
    /// Model name
    name: String,
    /// Number of inputs
    num_inputs: u32,
    /// Number of outputs
    num_outputs: u32,
    /// Whether quantized
    is_quantized: bool,
}

/// Session internal state
struct SessionState {
    /// Session ID
    id: usize,
    /// Parent engine ID
    engine_id: usize,
    /// Parent model ID
    model_id: usize,
    /// Backend type
    backend: LightShipBackend,
    /// Number of threads
    num_threads: usize,
    /// Last timing
    last_timing: Option<LightShipTiming>,
}

/// Tensor internal state
struct TensorState {
    /// Tensor ID
    id: usize,
    /// Tensor shape
    shape: Vec<usize>,
    /// Data type
    data_type: LightShipDataType,
    /// Data buffer
    data: Vec<u8>,
}

impl ApiState {
    fn new() -> Self {
        Self {
            engines: HashMap::new(),
            models: HashMap::new(),
            sessions: HashMap::new(),
            tensors: HashMap::new(),
            next_id: 1,
            last_error: String::new(),
        }
    }

    fn next_id(&mut self) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    fn set_error(&mut self, msg: impl Into<String>) {
        self.last_error = msg.into();
    }

    fn get_error(&self) -> String {
        self.last_error.clone()
    }
}

/// Get mutable access to global state
unsafe fn get_state() -> &'static mut ApiState {
    if C_API_STATE.is_none() {
        C_API_STATE = Some(ApiState::new());
    }
    C_API_STATE.as_mut().unwrap()
}

// ============================================================================
// Helper Functions
// ============================================================================

fn to_data_type(c_dtype: LightShipDataType) -> DataType {
    match c_dtype {
        LightShipDataType::F32 => DataType::F32,
        LightShipDataType::F16 => DataType::F16,
        LightShipDataType::F64 => DataType::F64,
        LightShipDataType::I8 => DataType::I8,
        LightShipDataType::I16 => DataType::I16,
        LightShipDataType::I32 => DataType::I32,
        LightShipDataType::I64 => DataType::I64,
        LightShipDataType::U8 => DataType::U8,
        LightShipDataType::QUInt8 => DataType::QUInt8,
        LightShipDataType::QInt8 => DataType::QInt8,
        LightShipDataType::QInt32 => DataType::QInt32,
        _ => DataType::F32,
    }
}

fn from_backend_type(backend: LightShipBackend) -> BackendType {
    match backend {
        LightShipBackend::CPU => BackendType::CPU,
        LightShipBackend::GPU => BackendType::GPU,
        LightShipBackend::NPU => BackendType::NPU,
        LightShipBackend::Vulkan => BackendType::Vulkan,
        LightShipBackend::Metal => BackendType::Metal,
        _ => BackendType::CPU,
    }
}

// ============================================================================
// C API Implementations
// ============================================================================

/// Create a new LightShip engine
///
/// # Safety
/// This function is unsafe and must be called with valid pointers.
#[no_mangle]
pub unsafe extern "C" fn LightShipEngine_Create(
    out_engine: *mut *mut std::ffi::c_void,
    log_level: LightShipLogLevel,
) -> LightShipErrorCode {
    if out_engine.is_null() {
        return LightShipErrorCode::InvalidArgument;
    }

    let state = get_state();
    let id = state.next_id();

    let engine = EngineState {
        id,
        log_level,
    };

    state.engines.insert(id, engine);

    // Return the ID as the handle (cast to pointer)
    *out_engine = id as *mut std::ffi::c_void;
    tracing::debug!("Created engine with ID: {}", id);

    LightShipErrorCode::Success
}

/// Destroy a LightShip engine
///
/// # Safety
/// This function is unsafe and must be called with valid pointers.
#[no_mangle]
pub unsafe extern "C" fn LightShipEngine_Destroy(engine: *mut std::ffi::c_void) -> LightShipErrorCode {
    if engine.is_null() {
        return LightShipErrorCode::InvalidHandle;
    }

    let id = engine as usize;
    let state = get_state();

    // Clean up associated sessions
    let sessions_to_remove: Vec<usize> = state.sessions
        .iter()
        .filter(|(_, s)| s.engine_id == id)
        .map(|(sid, _)| *sid)
        .collect();

    for sid in sessions_to_remove {
        state.sessions.remove(&sid);
    }

    // Clean up associated models
    let models_to_remove: Vec<usize> = state.models
        .iter()
        .filter(|(_, _m)| {
            // Models are not directly associated with engines, but we track them
            false
        })
        .map(|(mid, _)| *mid)
        .collect();

    for _mid in models_to_remove {
        // Currently models are not associated with engines
    }

    if state.engines.remove(&id).is_some() {
        tracing::debug!("Destroyed engine with ID: {}", id);
        LightShipErrorCode::Success
    } else {
        state.set_error("Engine not found");
        LightShipErrorCode::InvalidHandle
    }
}

/// Get the number of available backends
#[no_mangle]
pub extern "C" fn LightShipEngine_GetAvailableBackends(count: *mut u32) -> LightShipErrorCode {
    if count.is_null() {
        return LightShipErrorCode::InvalidArgument;
    }

    // For now, only CPU is available
    unsafe { *count = 1; }
    LightShipErrorCode::Success
}

/// Load a model from file
///
/// # Safety
/// This function is unsafe and must be called with valid pointers.
#[no_mangle]
pub unsafe extern "C" fn LightShipEngine_LoadModel(
    engine: *mut std::ffi::c_void,
    path: *const c_char,
    out_model: *mut *mut std::ffi::c_void,
) -> LightShipErrorCode {
    if engine.is_null() || path.is_null() || out_model.is_null() {
        return LightShipErrorCode::InvalidArgument;
    }

    let engine_id = engine as usize;
    let state = get_state();

    // Verify engine exists
    if !state.engines.contains_key(&engine_id) {
        state.set_error("Engine not found");
        return LightShipErrorCode::InvalidHandle;
    }

    // Get path as Rust string
    let c_str = CStr::from_ptr(path);
    let path_str = match c_str.to_str() {
        Ok(s) => s,
        Err(_) => {
            state.set_error("Invalid path encoding");
            return LightShipErrorCode::InvalidArgument;
        }
    };

    // Create a stub model for now (full ONNX loading would require Phase 4)
    let model_id = state.next_id();
    let model = ModelState {
        id: model_id,
        name: path_str.to_string(),
        num_inputs: 1,
        num_outputs: 1,
        is_quantized: false,
    };

    state.models.insert(model_id, model);
    *out_model = model_id as *mut std::ffi::c_void;

    tracing::debug!("Loaded model '{}' with ID: {}", path_str, model_id);
    LightShipErrorCode::Success
}

/// Create a new inference session
///
/// # Safety
/// This function is unsafe and must be called with valid pointers.
#[no_mangle]
pub unsafe extern "C" fn LightShipEngine_CreateSession(
    engine: *mut std::ffi::c_void,
    model: *mut std::ffi::c_void,
    config: *const LightShipSessionConfig,
    out_session: *mut *mut std::ffi::c_void,
) -> LightShipErrorCode {
    if engine.is_null() || model.is_null() || out_session.is_null() {
        return LightShipErrorCode::InvalidArgument;
    }

    let engine_id = engine as usize;
    let model_id = model as usize;
    let state = get_state();

    // Verify engine exists
    if !state.engines.contains_key(&engine_id) {
        state.set_error("Engine not found");
        return LightShipErrorCode::InvalidHandle;
    }

    // Verify model exists
    if !state.models.contains_key(&model_id) {
        state.set_error("Model not found");
        return LightShipErrorCode::InvalidHandle;
    }

    // Get config
    let cfg = if config.is_null() {
        LightShipSessionConfig::default()
    } else {
        (*config).clone()
    };

    let session_id = state.next_id();
    let session = SessionState {
        id: session_id,
        engine_id,
        model_id,
        backend: cfg.backend,
        num_threads: cfg.num_threads,
        last_timing: None,
    };

    state.sessions.insert(session_id, session);
    *out_session = session_id as *mut std::ffi::c_void;

    tracing::debug!("Created session with ID: {} for model: {}", session_id, model_id);
    LightShipErrorCode::Success
}

/// Run synchronous inference
///
/// # Safety
/// This function is unsafe and must be called with valid pointers.
#[no_mangle]
pub unsafe extern "C" fn LightShipSession_Run(
    session: *mut std::ffi::c_void,
    _inputs: *const *mut std::ffi::c_void,
    num_inputs: u32,
    _outputs: *const *mut std::ffi::c_void,
    num_outputs: u32,
) -> LightShipErrorCode {
    if session.is_null() {
        return LightShipErrorCode::InvalidHandle;
    }

    let session_id = session as usize;
    let state = get_state();

    let sess = match state.sessions.get_mut(&session_id) {
        Some(s) => s,
        None => {
            state.set_error("Session not found");
            return LightShipErrorCode::InvalidHandle;
        }
    };

    // Validate input/output counts match
    if num_inputs == 0 || num_outputs == 0 {
        state.set_error("Invalid input/output count");
        return LightShipErrorCode::InvalidArgument;
    }

    // For now, just record timing stub (real execution would be Phase 3)
    let timing = LightShipTiming {
        total_time_us: 1000,
        load_time_us: 100,
        compile_time_us: 200,
        execution_time_us: 700,
    };
    sess.last_timing = Some(timing);

    tracing::debug!("Ran inference on session: {}", session_id);
    LightShipErrorCode::Success
}

/// Create a tensor
///
/// # Safety
/// This function is unsafe and must be called with valid pointers.
#[no_mangle]
pub unsafe extern "C" fn LightShipTensor_Create(
    shape: *const LightShipShape,
    data_type: LightShipDataType,
    out_tensor: *mut *mut std::ffi::c_void,
) -> LightShipErrorCode {
    if shape.is_null() || out_tensor.is_null() {
        return LightShipErrorCode::InvalidArgument;
    }

    let state = get_state();
    let tensor_id = state.next_id();

    let shape_ref = &*shape;
    let dims = shape_ref.dims.clone();

    // Calculate element count and allocate
    let element_count: usize = dims.iter().product();
    let dtype = to_data_type(data_type);
    let element_size = dtype.byte_size();
    let total_size = element_count * element_size;

    let tensor = TensorState {
        id: tensor_id,
        shape: dims,
        data_type,
        data: vec![0u8; total_size],
    };

    state.tensors.insert(tensor_id, tensor);
    *out_tensor = tensor_id as *mut std::ffi::c_void;

    tracing::debug!("Created tensor with ID: {}, size: {} bytes", tensor_id, total_size);
    LightShipErrorCode::Success
}

/// Destroy a tensor
///
/// # Safety
/// This function is unsafe and must be called with valid pointers.
#[no_mangle]
pub unsafe extern "C" fn LightShipTensor_Destroy(tensor: *mut std::ffi::c_void) -> LightShipErrorCode {
    if tensor.is_null() {
        return LightShipErrorCode::InvalidHandle;
    }

    let id = tensor as usize;
    let state = get_state();

    if state.tensors.remove(&id).is_some() {
        tracing::debug!("Destroyed tensor with ID: {}", id);
        LightShipErrorCode::Success
    } else {
        state.set_error("Tensor not found");
        LightShipErrorCode::InvalidHandle
    }
}

/// Get tensor data pointer
///
/// # Safety
/// This function is unsafe and must be called with valid pointers.
#[no_mangle]
pub unsafe extern "C" fn LightShipTensor_GetData(
    tensor: *mut std::ffi::c_void,
    out_data: *mut *mut std::ffi::c_void,
) -> LightShipErrorCode {
    if tensor.is_null() || out_data.is_null() {
        return LightShipErrorCode::InvalidArgument;
    }

    let id = tensor as usize;
    let state = get_state();

    let tensor_state = match state.tensors.get(&id) {
        Some(t) => t,
        None => {
            state.set_error("Tensor not found");
            return LightShipErrorCode::InvalidHandle;
        }
    };

    // Return pointer to internal buffer
    *out_data = tensor_state.data.as_ptr() as *mut std::ffi::c_void;
    LightShipErrorCode::Success
}

/// Get tensor shape
///
/// # Safety
/// This function is unsafe and must be called with valid pointers.
#[no_mangle]
pub unsafe extern "C" fn LightShipTensor_GetShape(
    tensor: *mut std::ffi::c_void,
    out_shape: *mut LightShipShape,
) -> LightShipErrorCode {
    if tensor.is_null() || out_shape.is_null() {
        return LightShipErrorCode::InvalidArgument;
    }

    let id = tensor as usize;
    let state = get_state();

    let tensor_state = match state.tensors.get(&id) {
        Some(t) => t,
        None => {
            state.set_error("Tensor not found");
            return LightShipErrorCode::InvalidHandle;
        }
    };

    // Return shape by copying to output
    let dims = tensor_state.shape.clone();
    let shape = LightShipShape::new(dims);
    std::ptr::write(out_shape, shape);

    LightShipErrorCode::Success
}

/// Get timing information from session
///
/// # Safety
/// This function is unsafe and must be called with valid pointers.
#[no_mangle]
pub unsafe extern "C" fn LightShipSession_GetTiming(
    session: *mut std::ffi::c_void,
    out_timing: *mut LightShipTiming,
) -> LightShipErrorCode {
    if session.is_null() || out_timing.is_null() {
        return LightShipErrorCode::InvalidArgument;
    }

    let session_id = session as usize;
    let state = get_state();

    let sess = match state.sessions.get(&session_id) {
        Some(s) => s,
        None => {
            state.set_error("Session not found");
            return LightShipErrorCode::InvalidHandle;
        }
    };

    match &sess.last_timing {
        Some(timing) => {
            std::ptr::write(out_timing, timing.clone());
            LightShipErrorCode::Success
        }
        None => {
            // Return default timing if no inference run yet
            std::ptr::write(out_timing, LightShipTiming::default());
            LightShipErrorCode::Success
        }
    }
}

/// Set the backend for a session
///
/// # Safety
/// This function is unsafe and must be called with valid pointers.
#[no_mangle]
pub unsafe extern "C" fn LightShipSession_SetBackend(
    session: *mut std::ffi::c_void,
    backend: LightShipBackend,
) -> LightShipErrorCode {
    if session.is_null() {
        return LightShipErrorCode::InvalidHandle;
    }

    let session_id = session as usize;
    let state = get_state();

    let sess = match state.sessions.get_mut(&session_id) {
        Some(s) => s,
        None => {
            state.set_error("Session not found");
            return LightShipErrorCode::InvalidHandle;
        }
    };

    sess.backend = backend;
    tracing::debug!("Set session {} backend to {:?}", session_id, backend);
    LightShipErrorCode::Success
}

/// Get the last error message
///
/// # Safety
/// This function is unsafe and must be called with valid pointers.
#[no_mangle]
pub unsafe extern "C" fn LightShipEngine_GetLastError(
    _engine: *mut std::ffi::c_void,
    out_message: *mut *const c_char,
) -> LightShipErrorCode {
    if out_message.is_null() {
        return LightShipErrorCode::InvalidArgument;
    }

    let state = get_state();
    let error_msg = state.get_error();

    // Store error in a static for C API to read
    static mut LAST_ERROR: String = String::new();
    unsafe {
        LAST_ERROR = error_msg;
        if LAST_ERROR.is_empty() {
            *out_message = std::ptr::null();
        } else {
            *out_message = LAST_ERROR.as_ptr() as *const c_char;
        }
    }

    LightShipErrorCode::Success
}

/// Get model metadata
///
/// # Safety
/// This function is unsafe and must be called with valid pointers.
#[no_mangle]
pub unsafe extern "C" fn LightShipModel_GetMetadata(
    model: *mut std::ffi::c_void,
    out_metadata: *mut super::types::LightShipModelMetadata,
) -> LightShipErrorCode {
    if model.is_null() || out_metadata.is_null() {
        return LightShipErrorCode::InvalidArgument;
    }

    let model_id = model as usize;
    let state = get_state();

    let model_state = match state.models.get(&model_id) {
        Some(m) => m,
        None => {
            state.set_error("Model not found");
            return LightShipErrorCode::InvalidHandle;
        }
    };

    // Store metadata strings in static
    static mut MODEL_NAME: String = String::new();
    static mut MODEL_VERSION: String = String::new();

    unsafe {
        MODEL_NAME = model_state.name.clone();
        MODEL_VERSION = "1.0.0".to_string();

        *out_metadata = super::types::LightShipModelMetadata {
            name: if MODEL_NAME.is_empty() { std::ptr::null() } else { MODEL_NAME.as_ptr() as *const c_char },
            version: if MODEL_VERSION.is_empty() { std::ptr::null() } else { MODEL_VERSION.as_ptr() as *const c_char },
            num_inputs: model_state.num_inputs,
            num_outputs: model_state.num_outputs,
            is_quantized: model_state.is_quantized,
        };
    }

    LightShipErrorCode::Success
}
