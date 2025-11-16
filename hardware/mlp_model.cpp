/*
 * MLP Gesture Recognition IP Core for Vitis HLS - 2-IMU Test Model
 * Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
 * SPDX-License-Identifier: X11
 */

#include "mlp_model.h"
#include "mlp_weights.h"
#include <string.h>
#include <stdio.h>
#include <math.h>
#include "hls_stream.h"

void mlp_relu_activation(MLP_DTYPE* data, int size) {
    #pragma HLS INLINE off
    
    RELU_LOOP:
    for (int i = 0; i < size; i++) {
        #pragma HLS PIPELINE II=1
        if (data[i] < 0) {
            data[i] = 0;
        }
    }
}

void mlp_softmax_activation(MLP_DTYPE* data, int size) {
    #pragma HLS INLINE off
    
    // For fixed-point, we'll use a simplified softmax approximation
    // Find maximum value for numerical stability
    MLP_DTYPE max_val = data[0];
    
    SOFTMAX_MAX_LOOP:
    for (int i = 1; i < size; i++) {
        #pragma HLS PIPELINE II=1
        if (data[i] > max_val) {
            max_val = data[i];
        }
    }
    
    // Subtract max and apply simplified exponential approximation
    // For fixed-point, we'll use a piecewise linear approximation
    int32_t sum_exp = 0;
    
    SOFTMAX_EXP_LOOP:
    for (int i = 0; i < size; i++) {
        #pragma HLS PIPELINE II=1
        MLP_DTYPE shifted = data[i] - max_val;
        
        // Simplified exponential approximation for fixed-point
        // exp(x) ≈ max(0, 1 + x) for small x (good enough for classification)
        if (shifted < -MLP_SCALE_FACTOR) {
            data[i] = 1; // Very small positive value
        } else {
            data[i] = MLP_SCALE_FACTOR + shifted; // 1 + x in fixed-point
            if (data[i] < 1) data[i] = 1;
        }
        sum_exp += data[i];
    }
    
    // Normalize by sum (simplified division)
    SOFTMAX_NORM_LOOP:
    for (int i = 0; i < size; i++) {
        #pragma HLS PIPELINE II=1
        // Use shift for approximate division
        data[i] = (data[i] * MLP_SCALE_FACTOR) / (sum_exp >> 8);
        if (data[i] > MLP_SCALE_FACTOR) data[i] = MLP_SCALE_FACTOR;
    }
}

int mlp_argmax(MLP_DTYPE* data, int size) {
    #pragma HLS INLINE off
    
    MLP_DTYPE max_val = data[0];
    int max_idx = 0;
    
    ARGMAX_LOOP:
    for (int i = 1; i < size; i++) {
        #pragma HLS PIPELINE II=1
        if (data[i] > max_val) {
            max_val = data[i];
            max_idx = i;
        }
    }
    return max_idx;
}

void mlp_dense_layer(
    const MLP_DTYPE* input, 
    const MLP_DTYPE* weights, // Flattened weight matrix
    const MLP_DTYPE* bias,
    MLP_DTYPE* output,
    int input_size,
    int output_size) {
    #pragma HLS INLINE off
    
    DENSE_OUTPUT_LOOP:
    for (int i = 0; i < output_size; i++) {
        #pragma HLS PIPELINE II=1
        
        int32_t accumulator = 0;
        
        DENSE_INPUT_LOOP:
        for (int j = 0; j < input_size; j++) {
            #pragma HLS UNROLL factor=4
            // Access weights in row-major order: weights[j * output_size + i]
            accumulator += (int32_t)input[j] * (int32_t)weights[j * output_size + i];
        }
        
        // Scale back from Q16.16 to Q8.8 and add bias
        accumulator >>= MLP_SCALE_SHIFT;
        accumulator += (int32_t)bias[i];
        
        // Saturate to prevent overflow
        if (accumulator > 32767) accumulator = 32767;
        if (accumulator < -32768) accumulator = -32768;
        
        output[i] = (MLP_DTYPE)accumulator;
    }
}

void mlp_gesture_detection(
    hls::stream<AXIS_wLAST> &input_stream,
    hls::stream<AXIS_wLAST> &output_stream) {

#pragma HLS INTERFACE axis port=input_stream
#pragma HLS INTERFACE axis port=output_stream
#pragma HLS INTERFACE s_axilite port=return bundle=control

    // Internal buffers for intermediate computations - now using fixed-point
    MLP_DTYPE layer1_output[MLP_LAYER1_SIZE];
    MLP_DTYPE layer2_output[MLP_LAYER2_SIZE];
    #pragma HLS ARRAY_PARTITION variable=layer1_output complete
    #pragma HLS ARRAY_PARTITION variable=layer2_output complete
    
    // Local buffers for input/output - now using fixed-point
    MLP_DTYPE input_buffer[MLP_INPUT_SIZE];
    MLP_DTYPE output_buffer[MLP_OUTPUT_SIZE];
    #pragma HLS ARRAY_PARTITION variable=input_buffer complete
    #pragma HLS ARRAY_PARTITION variable=output_buffer complete
    
    // Read input data from stream and convert to fixed-point
    STREAM_INPUT_LOOP:
    for (int i = 0; i < MLP_INPUT_SIZE; i++) {
        #pragma HLS PIPELINE II=1
        AXIS_wLAST temp = input_stream.read();
        // Convert from uint32_t bits to float, then to Q8.8 fixed-point
        union { float f; uint32_t i; } converter;
        converter.i = temp.data;
        input_buffer[i] = mlp_float_to_fixed(converter.f);
    }
    
    // Forward pass through the MLP
    
    // Layer 1: Input(84) -> Dense(64) -> ReLU
    mlp_dense_layer(input_buffer, 
                   (const MLP_DTYPE*)mlp_dense_64_weights, 
                   mlp_dense_64_bias, 
                   layer1_output, 
                   MLP_INPUT_SIZE, 
                   MLP_LAYER1_SIZE);
    mlp_relu_activation(layer1_output, MLP_LAYER1_SIZE);
    
    // Layer 2: Dense(64) -> Dense(32) -> ReLU  
    mlp_dense_layer(layer1_output, 
                   (const MLP_DTYPE*)mlp_dense_32_weights, 
                   mlp_dense_32_bias, 
                   layer2_output, 
                   MLP_LAYER1_SIZE, 
                   MLP_LAYER2_SIZE);
    mlp_relu_activation(layer2_output, MLP_LAYER2_SIZE);
    
    // Output Layer: Dense(32) -> Dense(8) (logits)
    mlp_dense_layer(layer2_output, 
                   (const MLP_DTYPE*)mlp_logits_weights, 
                   mlp_logits_bias, 
                   output_buffer, 
                   MLP_LAYER2_SIZE, 
                   MLP_OUTPUT_SIZE);
    
    // Write output data to stream (raw logits, convert back to float for compatibility)
    STREAM_OUTPUT_LOOP:
    for (int i = 0; i < MLP_OUTPUT_SIZE; i++) {
        #pragma HLS PIPELINE II=1
        AXIS_wLAST temp;
        // Convert Q8.8 fixed-point back to float, then to uint32_t bits
        union { float f; uint32_t i; } converter;
        converter.f = mlp_fixed_to_float(output_buffer[i]);
        temp.data = converter.i;
        temp.last = (i == MLP_OUTPUT_SIZE - 1) ? 1 : 0;
        output_stream.write(temp);
    }
}

// Matrix operations implementation
void mlp_matrix_vector_mult(const MLP_DTYPE* W, const MLP_DTYPE* x, const MLP_DTYPE* b,
                           MLP_DTYPE* y, int rows, int cols) {
    #pragma HLS INLINE off
    
    MATRIX_MULT_ROWS:
    for (int i = 0; i < rows; i++) {
        #pragma HLS PIPELINE II=1
        int32_t accumulator = 0;
        
        // Compute dot product of row i with input vector x
        MATRIX_MULT_COLS:
        for (int j = 0; j < cols; j++) {
            #pragma HLS UNROLL factor=4
            accumulator += (int32_t)W[i * cols + j] * (int32_t)x[j];
        }
        
        // Scale back from Q16.16 to Q8.8 and add bias
        accumulator >>= MLP_SCALE_SHIFT;
        accumulator += (int32_t)b[i];
        
        // Saturate to prevent overflow
        if (accumulator > 32767) accumulator = 32767;
        if (accumulator < -32768) accumulator = -32768;
        
        y[i] = (MLP_DTYPE)accumulator;
    }
}

void mlp_apply_relu(MLP_DTYPE* vector, int size) {
    #pragma HLS INLINE off
    
    APPLY_RELU_LOOP:
    for (int i = 0; i < size; i++) {
        #pragma HLS PIPELINE II=1
        if (vector[i] < 0) {
            vector[i] = 0;
        }
    }
}

