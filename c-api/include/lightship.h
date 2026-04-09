/* LightShip C API
 *
 * This header defines the C API for LightShip inference engine.
 * It provides a stable ABI for integration with C/C++, Python, and mobile platforms.
 */

#ifndef LIGHTSHIP_H
#define LIGHTSHIP_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*==============================================================================
 * Version Information
 *============================================================================*/

/** LightShip major version */
#define LIGHTSHIP_VERSION_MAJOR 0
/** LightShip minor version */
#define LIGHTSHIP_VERSION_MINOR 1
/** LightShip patch version */
#define LIGHTSHIP_VERSION_PATCH 0

/** Get full version string */
const char* LightShip_GetVersion(void);

/*==============================================================================
 * Error Handling
 *============================================================================*/

/** Error codes returned by LightShip API functions */
typedef enum {
    /** Success */
    LIGHTSHIP_SUCCESS = 0,
    /** Generic error */
    LIGHTSHIP_ERROR = 1,
    /** Invalid argument */
    LIGHTSHIP_ERROR_INVALID_ARGUMENT = 2,
    /** Out of memory */
    LIGHTSHIP_ERROR_OUT_OF_MEMORY = 3,
    /** Model not found */
    LIGHTSHIP_ERROR_MODEL_NOT_FOUND = 4,
    /** Invalid model format */
    LIGHTSHIP_ERROR_INVALID_MODEL = 5,
    /** Backend not available */
    LIGHTSHIP_ERROR_BACKEND_NOT_AVAILABLE = 6,
    /** Tensor shape mismatch */
    LIGHTSHIP_ERROR_SHAPE_MISMATCH = 7,
    /** Data type not supported */
    LIGHTSHIP_ERROR_UNSUPPORTED_DATATYPE = 8,
    /** Device error */
    LIGHTSHIP_ERROR_DEVICE_ERROR = 9,
    /** Not implemented */
    LIGHTSHIP_ERROR_NOT_IMPLEMENTED = 10,
} LightShipErrorCode;

/** Get human-readable error message */
const char* LightShipError_GetMessage(LightShipErrorCode code);

/*==============================================================================
 * Data Types
 *============================================================================*/

/** Data types supported by LightShip */
typedef enum {
    /** 32-bit floating point */
    LIGHTSHIP_DATATYPE_F32 = 0,
    /** 16-bit floating point */
    LIGHTSHIP_DATATYPE_F16 = 1,
    /** 64-bit floating point */
    LIGHTSHIP_DATATYPE_F64 = 2,
    /** 8-bit signed integer */
    LIGHTSHIP_DATATYPE_I8 = 3,
    /** 16-bit signed integer */
    LIGHTSHIP_DATATYPE_I16 = 4,
    /** 32-bit signed integer */
    LIGHTSHIP_DATATYPE_I32 = 5,
    /** 64-bit signed integer */
    LIGHTSHIP_DATATYPE_I64 = 6,
    /** 8-bit unsigned integer */
    LIGHTSHIP_DATATYPE_U8 = 7,
    /** Quantized unsigned 8-bit */
    LIGHTSHIP_DATATYPE_QUINT8 = 8,
    /** Quantized signed 8-bit */
    LIGHTSHIP_DATATYPE_QINT8 = 9,
} LightShipDataType;

/** Storage layouts for tensors */
typedef enum {
    /** Channels first (NCHW) */
    LIGHTSHIP_LAYOUT_NCHW = 0,
    /** Channels last (NHWC) */
    LIGHTSHIP_LAYOUT_NHWC = 1,
    /** OIHW for conv weights */
    LIGHTSHIP_LAYOUT_OIHW = 2,
} LightShipStorageLayout;

/** Backend types */
typedef enum {
    /** CPU backend */
    LIGHTSHIP_BACKEND_CPU = 0,
    /** GPU backend */
    LIGHTSHIP_BACKEND_GPU = 1,
    /** NPU backend */
    LIGHTSHIP_BACKEND_NPU = 2,
    /** Vulkan compute */
    LIGHTSHIP_BACKEND_VULKAN = 3,
    /** Metal (Apple) */
    LIGHTSHIP_BACKEND_METAL = 4,
} LightShipBackendType;

/** Logging levels */
typedef enum {
    /** No logging */
    LIGHTSHIP_LOG_LEVEL_NONE = 0,
    /** Error only */
    LIGHTSHIP_LOG_LEVEL_ERROR = 1,
    /** Warnings and errors */
    LIGHTSHIP_LOG_LEVEL_WARN = 2,
    /** Info, warnings, and errors */
    LIGHTSHIP_LOG_LEVEL_INFO = 3,
    /** Debug info */
    LIGHTSHIP_LOG_LEVEL_DEBUG = 4,
} LightShipLogLevel;

/*==============================================================================
 * Opaque Handle Types
 *============================================================================*/

/** Engine handle */
typedef struct LightShipEngineImpl* LightShipEngine;
/** Model handle */
typedef struct LightShipModelImpl* LightShipModel;
/** Session handle */
typedef struct LightShipSessionImpl* LightShipSession;
/** Tensor handle */
typedef struct LightShipTensorImpl* LightShipTensor;
/** Backend handle */
typedef struct LightShipBackendImpl* LightShipBackend;

/*==============================================================================
 * Engine Functions
 *============================================================================*/

/** Create a new LightShip engine instance
 *
 * @param out_engine Pointer to store the created engine
 * @param log_level Minimum log level to output
 * @return LIGHTSHIP_SUCCESS on success, error code otherwise
 */
LightShipErrorCode LightShipEngine_Create(
    LightShipEngine* out_engine,
    LightShipLogLevel log_level
);

/** Destroy a LightShip engine instance
 *
 * @param engine Engine to destroy
 * @return LIGHTSHIP_SUCCESS on success, error code otherwise
 */
LightShipErrorCode LightShipEngine_Destroy(
    LightShipEngine engine
);

/** Get the last error message for the engine
 *
 * @param engine Engine instance
 * @return Error message string (valid until next call)
 */
const char* LightShipEngine_GetLastError(
    LightShipEngine engine
);

/** Load a model from file
 *
 * @param engine Engine instance
 * @param path Path to model file
 * @param out_model Pointer to store the loaded model
 * @return LIGHTSHIP_SUCCESS on success, error code otherwise
 */
LightShipErrorCode LightShipEngine_LoadModel(
    LightShipEngine engine,
    const char* path,
    LightShipModel* out_model
);

/** Load a model from memory
 *
 * @param engine Engine instance
 * @param data Model data pointer
 * @param size Size of model data in bytes
 * @param out_model Pointer to store the loaded model
 * @return LIGHTSHIP_SUCCESS on success, error code otherwise
 */
LightShipErrorCode LightShipEngine_LoadModelFromMemory(
    LightShipEngine engine,
    const uint8_t* data,
    size_t size,
    LightShipModel* out_model
);

/** Get model metadata
 *
 * @param model Model instance
 * @param out_name Pointer to store model name (optional, can be NULL)
 * @param out_version Pointer to store version string (optional, can be NULL)
 * @param out_num_inputs Pointer to store number of inputs (optional, can be NULL)
 * @param out_num_outputs Pointer to store number of outputs (optional, can be NULL)
 * @return LIGHTSHIP_SUCCESS on success, error code otherwise
 */
LightShipErrorCode LightShipModel_GetMetadata(
    LightShipModel model,
    const char** out_name,
    const char** out_version,
    uint32_t* out_num_inputs,
    uint32_t* out_num_outputs
);

/** Get model input tensor info
 *
 * @param model Model instance
 * @param index Input index (0-based)
 * @param out_name Pointer to store input name (optional, can be NULL)
 * @param out_dtype Pointer to store data type (optional, can be NULL)
 * @param out_shape Pointer to store shape (optional, can be NULL)
 * @param out_num_dims Pointer to store number of dimensions (optional, can be NULL)
 * @return LIGHTSHIP_SUCCESS on success, error code otherwise
 */
LightShipErrorCode LightShipModel_GetInputInfo(
    LightShipModel model,
    uint32_t index,
    const char** out_name,
    LightShipDataType* out_dtype,
    int64_t* out_shape,
    uint32_t* out_num_dims
);

/** Get model output tensor info
 *
 * @param model Model instance
 * @param index Output index (0-based)
 * @param out_name Pointer to store output name (optional, can be NULL)
 * @param out_dtype Pointer to store data type (optional, can be NULL)
 * @param out_shape Pointer to store shape (optional, can be NULL)
 * @param out_num_dims Pointer to store number of dimensions (optional, can be NULL)
 * @return LIGHTSHIP_SUCCESS on success, error code otherwise
 */
LightShipErrorCode LightShipModel_GetOutputInfo(
    LightShipModel model,
    uint32_t index,
    const char** out_name,
    LightShipDataType* out_dtype,
    int64_t* out_shape,
    uint32_t* out_num_dims
);

/** Destroy a model instance
 *
 * @param model Model to destroy
 * @return LIGHTSHIP_SUCCESS on success, error code otherwise
 */
LightShipErrorCode LightShipModel_Destroy(
    LightShipModel model
);

/*==============================================================================
 * Session Functions
 *============================================================================*/

/** Session configuration */
typedef struct {
    /** Backend type to use */
    LightShipBackendType backend_type;
    /** Number of threads for CPU backend (0 = auto) */
    uint32_t num_threads;
    /** Whether to use async execution */
    bool async_execution;
    /** Memory allocation alignment */
    uint32_t alignment;
} LightShipSessionConfig;

/** Create a new inference session
 *
 * @param engine Engine instance
 * @param model Model to run
 * @param config Session configuration (can be NULL for defaults)
 * @param out_session Pointer to store the created session
 * @return LIGHTSHIP_SUCCESS on success, error code otherwise
 */
LightShipErrorCode LightShipEngine_CreateSession(
    LightShipEngine engine,
    LightShipModel model,
    const LightShipSessionConfig* config,
    LightShipSession* out_session
);

/** Destroy an inference session
 *
 * @param session Session to destroy
 * @return LIGHTSHIP_SUCCESS on success, error code otherwise
 */
LightShipErrorCode LightShipSession_Destroy(
    LightShipSession session
);

/** Run inference synchronously
 *
 * @param session Session instance
 * @param inputs Array of input tensors
 * @param num_inputs Number of input tensors
 * @param outputs Array of output tensors (must be pre-allocated)
 * @param num_outputs Number of output tensors
 * @return LIGHTSHIP_SUCCESS on success, error code otherwise
 */
LightShipErrorCode LightShipSession_Run(
    LightShipSession session,
    LightShipTensor* inputs,
    uint32_t num_inputs,
    LightShipTensor* outputs,
    uint32_t num_outputs
);

/** Set input tensor by name
 *
 * @param session Session instance
 * @param name Input tensor name
 * @param tensor Input tensor
 * @return LIGHTSHIP_SUCCESS on success, error code otherwise
 */
LightShipErrorCode LightShipSession_SetInput(
    LightShipSession session,
    const char* name,
    LightShipTensor tensor
);

/** Get output tensor by name
 *
 * @param session Session instance
 * @param name Output tensor name
 * @param out_tensor Pointer to store output tensor
 * @return LIGHTSHIP_SUCCESS on success, error code otherwise
 */
LightShipErrorCode LightShipSession_GetOutput(
    LightShipSession session,
    const char* name,
    LightShipTensor* out_tensor
);

/** Get timing information for last inference
 *
 * @param session Session instance
 * @param out_total_time Pointer to store total time in microseconds
 * @param out_operator_time Pointer to store per-operator times (optional, can be NULL)
 * @param out_num_operators Pointer to store number of operators (optional, can be NULL)
 * @return LIGHTSHIP_SUCCESS on success, error code otherwise
 */
LightShipErrorCode LightShipSession_GetTiming(
    LightShipSession session,
    uint64_t* out_total_time,
    uint64_t* out_operator_time,
    uint32_t* out_num_operators
);

/** Set session backend
 *
 * @param session Session instance
 * @param backend_type Backend type to use
 * @return LIGHTSHIP_SUCCESS on success, error code otherwise
 */
LightShipErrorCode LightShipSession_SetBackend(
    LightShipSession session,
    LightShipBackendType backend_type
);

/*==============================================================================
 * Tensor Functions
 *============================================================================*/

/** Create a tensor
 *
 * @param shape Tensor shape
 * @param num_dims Number of dimensions
 * @param data_type Tensor data type
 * @param out_tensor Pointer to store created tensor
 * @return LIGHTSHIP_SUCCESS on success, error code otherwise
 */
LightShipErrorCode LightShipTensor_Create(
    const int64_t* shape,
    uint32_t num_dims,
    LightShipDataType data_type,
    LightShipTensor* out_tensor
);

/** Create a tensor with external data
 *
 * @param shape Tensor shape
 * @param num_dims Number of dimensions
 * @param data_type Tensor data type
 * @param data Data pointer
 * @param size Data size in bytes
 * @param out_tensor Pointer to store created tensor
 * @return LIGHTSHIP_SUCCESS on success, error code otherwise
 */
LightShipErrorCode LightShipTensor_CreateWithData(
    const int64_t* shape,
    uint32_t num_dims,
    LightShipDataType data_type,
    const void* data,
    size_t size,
    LightShipTensor* out_tensor
);

/** Destroy a tensor
 *
 * @param tensor Tensor to destroy
 * @return LIGHTSHIP_SUCCESS on success, error code otherwise
 */
LightShipErrorCode LightShipTensor_Destroy(
    LightShipTensor tensor
);

/** Get tensor data pointer
 *
 * @param tensor Tensor instance
 * @param out_data Pointer to store data pointer
 * @param out_size Pointer to store data size in bytes
 * @return LIGHTSHIP_SUCCESS on success, error code otherwise
 */
LightShipErrorCode LightShipTensor_GetData(
    LightShipTensor tensor,
    const void** out_data,
    size_t* out_size
);

/** Set tensor data
 *
 * @param tensor Tensor instance
 * @param data Data pointer
 * @param size Data size in bytes
 * @return LIGHTSHIP_SUCCESS on success, error code otherwise
 */
LightShipErrorCode LightShipTensor_SetData(
    LightShipTensor tensor,
    const void* data,
    size_t size
);

/** Get tensor shape
 *
 * @param tensor Tensor instance
 * @param out_shape Pointer to store shape
 * @param out_num_dims Pointer to store number of dimensions
 * @return LIGHTSHIP_SUCCESS on success, error code otherwise
 */
LightShipErrorCode LightShipTensor_GetShape(
    LightShipTensor tensor,
    int64_t* out_shape,
    uint32_t* out_num_dims
);

/** Get tensor data type
 *
 * @param tensor Tensor instance
 * @param out_dtype Pointer to store data type
 * @return LIGHTSHIP_SUCCESS on success, error code otherwise
 */
LightShipErrorCode LightShipTensor_GetDataType(
    LightShipTensor tensor,
    LightShipDataType* out_dtype
);

/*==============================================================================
 * Backend Functions
 *============================================================================*/

/** Get available backends
 *
 * @param out_backends Pointer to store available backends array
 * @param out_num_backends Pointer to store number of backends
 * @return LIGHTSHIP_SUCCESS on success, error code otherwise
 */
LightShipErrorCode LightShipEngine_GetAvailableBackends(
    LightShipBackendType* out_backends,
    uint32_t* out_num_backends
);

/** Create a backend
 *
 * @param engine Engine instance
 * @param backend_type Backend type to create
 * @param out_backend Pointer to store created backend
 * @return LIGHTSHIP_SUCCESS on success, error code otherwise
 */
LightShipErrorCode LightShipEngine_CreateBackend(
    LightShipEngine engine,
    LightShipBackendType backend_type,
    LightShipBackend* out_backend
);

/** Destroy a backend
 *
 * @param backend Backend to destroy
 * @return LIGHTSHIP_SUCCESS on success, error code otherwise
 */
LightShipErrorCode LightShipBackend_Destroy(
    LightShipBackend backend
);

/** Get backend info
 *
 * @param backend Backend instance
 * @param out_name Pointer to store backend name
 * @param out_vendor Pointer to store vendor name
 * @return LIGHTSHIP_SUCCESS on success, error code otherwise
 */
LightShipErrorCode LightShipBackend_GetInfo(
    LightShipBackend backend,
    const char** out_name,
    const char** out_vendor
);

#ifdef __cplusplus
}
#endif

#endif /* LIGHTSHIP_H */
