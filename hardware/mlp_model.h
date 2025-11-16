#ifndef MLP_MODEL_H
#define MLP_MODEL_H

#include "mlp_weights.h"
#include "hls_stream.h"
#include "ap_axi_sdata.h"
#include <stdint.h>

typedef ap_axiu<32,0,0,0, AXIS_ENABLE_DATA|AXIS_ENABLE_LAST, true> AXIS_wLAST;

// Gesture class names (for reference) - 2-IMU Test Model (8 classes)
// 0: no_gesture
// 1: swipe_left  
// 2: swipe_right
// 3: roll_wrist_clockwise
// 4: roll_wrist_counter_clockwise
// 5: swipe_up
// 6: swipe_down
// 7: shake

// Fixed-point arithmetic helper functions for MLP

/**
 * Multiply two Q8.8 fixed-point numbers
 * Result is Q16.16, then shifted back to Q8.8
 */
static inline MLP_DTYPE mlp_fixed_multiply(MLP_DTYPE a, MLP_DTYPE b) {
    int32_t result = ((int32_t)a * (int32_t)b) >> MLP_SCALE_SHIFT;
    // Saturate to prevent overflow
    if (result > 32767) result = 32767;
    if (result < -32768) result = -32768;
    return (MLP_DTYPE)result;
}

/**
 * Add two Q8.8 fixed-point numbers with saturation
 */
static inline MLP_DTYPE mlp_fixed_add(MLP_DTYPE a, MLP_DTYPE b) {
    int32_t result = (int32_t)a + (int32_t)b;
    // Saturate to prevent overflow
    if (result > 32767) result = 32767;
    if (result < -32768) result = -32768;
    return (MLP_DTYPE)result;
}

/**
 * ReLU activation function for fixed-point
 */
static inline MLP_DTYPE mlp_relu(MLP_DTYPE x) {
    return (x > 0) ? x : 0;
}

/**
 * Convert float input to Q8.8 fixed-point
 */
static inline MLP_DTYPE mlp_float_to_fixed(float x) {
    // Clamp to valid range
    if (x > 127.996f) x = 127.996f;
    if (x < -128.0f) x = -128.0f;
    return (MLP_DTYPE)(x * MLP_SCALE_FACTOR);
}

/**
 * Convert Q8.8 fixed-point to float
 */
static inline float mlp_fixed_to_float(MLP_DTYPE x) {
    return ((float)x) / MLP_SCALE_FACTOR;
}

// Main MLP functions
void mlp_gesture_detection(
    hls::stream<AXIS_wLAST> &input_stream,
    hls::stream<AXIS_wLAST> &output_stream);

// Helper functions - now using fixed-point arithmetic
void mlp_relu_activation(MLP_DTYPE* data, int size);
void mlp_softmax_activation(MLP_DTYPE* data, int size);
int mlp_argmax(MLP_DTYPE* data, int size);
void mlp_dense_layer(
    const MLP_DTYPE* input, 
    const MLP_DTYPE* weights, // Flattened weight matrix
    const MLP_DTYPE* bias,
    MLP_DTYPE* output,
    int input_size,
    int output_size);

// Matrix operations
void mlp_matrix_vector_mult(const MLP_DTYPE* W, const MLP_DTYPE* x, const MLP_DTYPE* b,
                           MLP_DTYPE* y, int rows, int cols);
void mlp_apply_relu(MLP_DTYPE* vector, int size);

#endif // MLP_MODEL_H

