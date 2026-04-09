/* LightShip Java/JNI Wrapper
 *
 * This package provides Java bindings for the LightShip inference engine.
 */

package com.lightship;

import java.nio.ByteBuffer;

/**
 * LightShip inference engine wrapper for Android.
 *
 * This class provides a Java API for loading and running neural network models
 * using the LightShip inference engine via JNI.
 */
public class LightShip {
    private long engineHandle;
    private long modelHandle;
    private long sessionHandle;

    static {
        System.loadLibrary("lightship_jni");
    }

    /** Initialize the LightShip engine */
    public LightShip() {
        this.engineHandle = 0;
        this.modelHandle = 0;
        this.sessionHandle = 0;
    }

    /**
     * Create a new LightShip engine instance.
     *
     * @return 0 on success, error code otherwise
     */
    public native int createEngine();

    /**
     * Destroy the LightShip engine instance.
     *
     * @return 0 on success, error code otherwise
     */
    public native int destroyEngine();

    /**
     * Load a model from file.
     *
     * @param modelPath Path to the model file
     * @return 0 on success, error code otherwise
     */
    public native int loadModel(String modelPath);

    /**
     * Load a model from assets.
     *
     * @param assetManager Asset manager for reading assets
     * @param modelName Name of the model asset
     * @return 0 on success, error code otherwise
     */
    public native int loadModelFromAsset(Object assetManager, String modelName);

    /**
     * Create an inference session.
     *
     * @param numThreads Number of threads (0 = auto)
     * @param useGPU Use GPU if available
     * @return 0 on success, error code otherwise
     */
    public native int createSession(int numThreads, boolean useGPU);

    /**
     * Destroy the inference session.
     *
     * @return 0 on success, error code otherwise
     */
    public native int destroySession();

    /**
     * Get the number of model inputs.
     *
     * @return Number of inputs
     */
    public native int getNumInputs();

    /**
     * Get the number of model outputs.
     *
     * @return Number of outputs
     */
    public native int getNumOutputs();

    /**
     * Get input tensor info.
     *
     * @param index Input index
     * @return Input tensor info
     */
    public native TensorInfo getInputInfo(int index);

    /**
     * Get output tensor info.
     *
     * @param index Output index
     * @return Output tensor info
     */
    public native TensorInfo getOutputInfo(int index);

    /**
     * Run inference synchronously.
     *
     * @param inputs Array of input buffers
     * @param outputs Array of output buffers
     * @return 0 on success, error code otherwise
     */
    public native int run(ByteBuffer[] inputs, ByteBuffer[] outputs);

    /**
     * Run inference with direct byte arrays.
     *
     * @param inputData Array of input byte arrays
     * @param outputData Array of output byte arrays
     * @return 0 on success, error code otherwise
     */
    public native int run(byte[][] inputData, byte[][] outputData);

    /**
     * Get the last error message.
     *
     * @return Error message string
     */
    public native String getLastError();

    /**
     * Get inference timing info.
     *
     * @return Timing info in microseconds
     */
    public native long[] getTiming();

    /**
     * Tensor information.
     */
    public static class TensorInfo {
        public String name;
        public int[] shape;
        public int dataType;
        public int byteSize;

        public TensorInfo(String name, int[] shape, int dataType, int byteSize) {
            this.name = name;
            this.shape = shape;
            this.dataType = dataType;
            this.byteSize = byteSize;
        }

        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder();
            sb.append("TensorInfo{name='").append(name).append("', shape=[");
            for (int i = 0; i < shape.length; i++) {
                if (i > 0) sb.append(", ");
                sb.append(shape[i]);
            }
            sb.append("], dtype=").append(dataType).append("}");
            return sb.toString();
        }
    }

    /** Data type constants */
    public static final int DATATYPE_F32 = 0;
    public static final int DATATYPE_F16 = 1;
    public static final int DATATYPE_I8 = 3;
    public static final int DATATYPE_U8 = 7;
    public static final int DATATYPE_QUINT8 = 12;
    public static final int DATATYPE_QINT8 = 13;

    /** Error codes */
    public static final int SUCCESS = 0;
    public static final int ERROR = 1;
    public static final int ERROR_INVALID_ARGUMENT = 2;
    public static final int ERROR_OUT_OF_MEMORY = 3;
    public static final int ERROR_MODEL_NOT_FOUND = 4;
    public static final int ERROR_INVALID_MODEL = 5;
    public static final int ERROR_BACKEND_NOT_AVAILABLE = 6;
    public static final int ERROR_SHAPE_MISMATCH = 7;
}
