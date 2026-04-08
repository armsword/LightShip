//! C API function declarations
//!
//! These are C-compatible function declarations for the LightShip API.

use super::{LightShipBackend, LightShipDataType, LightShipInferenceMode, LightShipLogLevel, LightShipSessionConfig, LightShipShape, LightShipTiming};

/// Create a new LightShip engine
///
/// # Safety
/// This function is unsafe and must be called with valid pointers.
#[no_mangle]
pub unsafe extern "C" fn LightShipEngine_Create(
    out_engine: *mut *mut std::ffi::c_void,
    log_level: LightShipLogLevel,
) -> super::error::LightShipErrorCode {
    super::error::LightShipErrorCode::Success
}

/// Destroy a LightShip engine
///
/// # Safety
/// This function is unsafe and must be called with valid pointers.
#[no_mangle]
pub unsafe extern "C" fn LightShipEngine_Destroy(engine: *mut std::ffi::c_void) -> super::error::LightShipErrorCode {
    super::error::LightShipErrorCode::Success
}

/// Get the number of available backends
#[no_mangle]
pub extern "C" fn LightShipEngine_GetAvailableBackends(_count: *mut u32) -> super::error::LightShipErrorCode {
    super::error::LightShipErrorCode::Success
}

/// Load a model from file
///
/// # Safety
/// This function is unsafe and must be called with valid pointers.
#[no_mangle]
pub unsafe extern "C" fn LightShipEngine_LoadModel(
    _engine: *mut std::ffi::c_void,
    _path: *const std::ffi::c_char,
    _out_model: *mut *mut std::ffi::c_void,
) -> super::error::LightShipErrorCode {
    super::error::LightShipErrorCode::Success
}

/// Create a new inference session
///
/// # Safety
/// This function is unsafe and must be called with valid pointers.
#[no_mangle]
pub unsafe extern "C" fn LightShipEngine_CreateSession(
    _engine: *mut std::ffi::c_void,
    _model: *mut std::ffi::c_void,
    _config: *const LightShipSessionConfig,
    _out_session: *mut *mut std::ffi::c_void,
) -> super::error::LightShipErrorCode {
    super::error::LightShipErrorCode::Success
}

/// Run synchronous inference
///
/// # Safety
/// This function is unsafe and must be called with valid pointers.
#[no_mangle]
pub unsafe extern "C" fn LightShipSession_Run(
    _session: *mut std::ffi::c_void,
    _inputs: *const *mut std::ffi::c_void,
    _num_inputs: u32,
    _outputs: *const *mut std::ffi::c_void,
    _num_outputs: u32,
) -> super::error::LightShipErrorCode {
    super::error::LightShipErrorCode::Success
}

/// Create a tensor
///
/// # Safety
/// This function is unsafe and must be called with valid pointers.
#[no_mangle]
pub unsafe extern "C" fn LightShipTensor_Create(
    _shape: *const LightShipShape,
    _data_type: LightShipDataType,
    _out_tensor: *mut *mut std::ffi::c_void,
) -> super::error::LightShipErrorCode {
    super::error::LightShipErrorCode::Success
}

/// Destroy a tensor
///
/// # Safety
/// This function is unsafe and must be called with valid pointers.
#[no_mangle]
pub unsafe extern "C" fn LightShipTensor_Destroy(_tensor: *mut std::ffi::c_void) -> super::error::LightShipErrorCode {
    super::error::LightShipErrorCode::Success
}

/// Get tensor data pointer
///
/// # Safety
/// This function is unsafe and must be called with valid pointers.
#[no_mangle]
pub unsafe extern "C" fn LightShipTensor_GetData(
    _tensor: *mut std::ffi::c_void,
    _out_data: *mut *mut std::ffi::c_void,
) -> super::error::LightShipErrorCode {
    super::error::LightShipErrorCode::Success
}

/// Get tensor shape
///
/// # Safety
/// This function is unsafe and must be called with valid pointers.
#[no_mangle]
pub unsafe extern "C" fn LightShipTensor_GetShape(
    _tensor: *mut std::ffi::c_void,
    _out_shape: *mut LightShipShape,
) -> super::error::LightShipErrorCode {
    super::error::LightShipErrorCode::Success
}

/// Get timing information from session
///
/// # Safety
/// This function is unsafe and must be called with valid pointers.
#[no_mangle]
pub unsafe extern "C" fn LightShipSession_GetTiming(
    _session: *mut std::ffi::c_void,
    _out_timing: *mut LightShipTiming,
) -> super::error::LightShipErrorCode {
    super::error::LightShipErrorCode::Success
}

/// Set the backend for a session
///
/// # Safety
/// This function is unsafe and must be called with valid pointers.
#[no_mangle]
pub unsafe extern "C" fn LightShipSession_SetBackend(
    _session: *mut std::ffi::c_void,
    _backend: LightShipBackend,
) -> super::error::LightShipErrorCode {
    super::error::LightShipErrorCode::Success
}

/// Get the last error message
///
/// # Safety
/// This function is unsafe and must be called with valid pointers.
#[no_mangle]
pub unsafe extern "C" fn LightShipEngine_GetLastError(
    _engine: *mut std::ffi::c_void,
    _out_message: *mut *const std::ffi::c_char,
) -> super::error::LightShipErrorCode {
    super::error::LightShipErrorCode::Success
}

/// Get model metadata
///
/// # Safety
/// This function is unsafe and must be called with valid pointers.
#[no_mangle]
pub unsafe extern "C" fn LightShipModel_GetMetadata(
    _model: *mut std::ffi::c_void,
    _out_metadata: *mut super::types::LightShipModelMetadata,
) -> super::error::LightShipErrorCode {
    super::error::LightShipErrorCode::Success
}
