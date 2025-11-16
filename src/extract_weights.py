#!/usr/bin/env python3
"""
Extract weights from numpy files and generate C header files for 4am Model
"""

import struct
import os

def read_npy_data(filename):
    """Read numpy array data from .npy file"""
    with open(filename, 'rb') as f:
        # Read magic string
        magic = f.read(6)
        # Read version
        version = struct.unpack('<BB', f.read(2))
        # Read header length
        header_len = struct.unpack('<H', f.read(2))[0]
        # Read header
        header = f.read(header_len).decode('ascii').rstrip('\x00')
        
        # Parse header to get shape and dtype
        import ast
        header_dict = ast.literal_eval(header)
        shape = header_dict['shape']
        dtype = header_dict['descr']
        
        # Read data
        if dtype == '<f4':
            dtype_size = 4
            fmt = '<f'
        else:
            raise ValueError(f'Unsupported dtype: {dtype}')
        
        data_size = 1
        for dim in shape:
            data_size *= dim
        data_size *= dtype_size
        data_bytes = f.read(data_size)
        
        # Unpack data
        num_elements = 1
        for dim in shape:
            num_elements *= dim
        data = struct.unpack(f'<{num_elements}f', data_bytes)
        
        # Reshape
        if len(shape) == 1:
            return list(data), shape
        elif len(shape) == 2:
            result = []
            for i in range(shape[0]):
                row = []
                for j in range(shape[1]):
                    row.append(data[i * shape[1] + j])
                result.append(row)
            return result, shape
        else:
            raise ValueError(f'Unsupported shape: {shape}')

def format_c_array_1d(data, name, indent=""):
    """Format 1D array as C array"""
    lines = []
    values_per_line = 8
    for i in range(0, len(data), values_per_line):
        chunk = data[i:i+values_per_line]
        formatted_values = [f"{val:.8f}f" for val in chunk]
        if i == 0:
            lines.append(f"{indent}{', '.join(formatted_values)},")
        else:
            lines.append(f"{indent}{', '.join(formatted_values)},")
    
    # Remove last comma
    if lines:
        lines[-1] = lines[-1].rstrip(',')
    
    return lines

def format_c_array_2d(data, name, indent=""):
    """Format 2D array as C array"""
    lines = []
    for i, row in enumerate(data):
        row_lines = format_c_array_1d(row, f"{name}[{i}]", indent + "    ")
        if i == 0:
            lines.append(f"{indent}{{ {row_lines[0].strip()}")
        else:
            lines.append(f"{indent} {{ {row_lines[0].strip()}")
        
        for line in row_lines[1:]:
            lines.append(f"{indent}   {line.strip()}")
        
        if i < len(data) - 1:
            lines.append(f"{indent} }},")
        else:
            lines.append(f"{indent} }}")
    
    return lines

def format_c_array_2d_fixed_point(data, name, indent=""):
    """Format 2D array as C array with fixed-point conversion"""
    lines = []
    for i, row in enumerate(data):
        # Convert to fixed-point
        fixed_row = [int(round(val * 256)) for val in row]  # Q8.8 format
        
        # Format as hex for better readability
        values_per_line = 8
        row_lines = []
        for j in range(0, len(fixed_row), values_per_line):
            chunk = fixed_row[j:j+values_per_line]
            formatted_values = [f"{val:d}" for val in chunk]
            row_lines.append(f"{', '.join(formatted_values)},")
        
        # Remove last comma from last line
        if row_lines:
            row_lines[-1] = row_lines[-1].rstrip(',')
        
        if i == 0:
            lines.append(f"{indent}{{ {row_lines[0].strip()}")
        else:
            lines.append(f"{indent} {{ {row_lines[0].strip()}")
        
        for line in row_lines[1:]:
            lines.append(f"{indent}   {line.strip()}")
        
        if i < len(data) - 1:
            lines.append(f"{indent} }},")
        else:
            lines.append(f"{indent} }}")
    
    return lines

def format_c_array_1d_fixed_point(data, name, indent=""):
    """Format 1D array as C array with fixed-point conversion"""
    # Convert to fixed-point
    fixed_data = [int(round(val * 256)) for val in data]  # Q8.8 format
    
    lines = []
    values_per_line = 8
    for i in range(0, len(fixed_data), values_per_line):
        chunk = fixed_data[i:i+values_per_line]
        formatted_values = [f"{val:d}" for val in chunk]
        lines.append(f"{indent}{', '.join(formatted_values)},")
    
    # Remove last comma
    if lines:
        lines[-1] = lines[-1].rstrip(',')
    
    return lines

def main():
    # Read all weight files from 4am directory
    weights_dir = 'weights_npy/'
    
    print("Reading weight files...")
    
    # Layer 1: 84 -> 64
    dense_64_W, dense_64_W_shape = read_npy_data(os.path.join(weights_dir, 'dense_64_W.npy'))
    dense_64_b, dense_64_b_shape = read_npy_data(os.path.join(weights_dir, 'dense_64_b.npy'))
    print(f"Dense 64 weights: {dense_64_W_shape}, bias: {dense_64_b_shape}")
    
    # Layer 2: 64 -> 32
    dense_32_W, dense_32_W_shape = read_npy_data(os.path.join(weights_dir, 'dense_32_W.npy'))
    dense_32_b, dense_32_b_shape = read_npy_data(os.path.join(weights_dir, 'dense_32_b.npy'))
    print(f"Dense 32 weights: {dense_32_W_shape}, bias: {dense_32_b_shape}")
    
    # Output layer: 32 -> 8
    logits_W, logits_W_shape = read_npy_data(os.path.join(weights_dir, 'logits_W.npy'))
    logits_b, logits_b_shape = read_npy_data(os.path.join(weights_dir, 'logits_b.npy'))
    print(f"Logits weights: {logits_W_shape}, bias: {logits_b_shape}")
    
    # Generate header file
    with open('mlp_weights.h', 'w') as f:
        f.write("""#ifndef MLP_WEIGHTS_H
#define MLP_WEIGHTS_H

#include <stdint.h>

// MLP Network Architecture: 84 -> 64 -> 32 -> 8
// Gesture classes: ['no_gesture', 'swipe_left', 'swipe_right','roll_wrist_clockwise', 'roll_wrist_counter_clockwise', 
//                   'swipe_up', 'swipe_down', 'shake']


#define MLP_INPUT_SIZE 84
#define MLP_LAYER1_SIZE 64
#define MLP_LAYER2_SIZE 32
#define MLP_OUTPUT_SIZE 8

// Using Q8.8 fixed-point format (16-bit signed integer)
// Range: -128.0 to +127.996 with 1/256 precision
typedef int16_t MLP_DTYPE;
#define MLP_SCALE_FACTOR 256  // 2^8 for Q8.8 format
#define MLP_SCALE_SHIFT 8     // Shift amount for scaling

// Conversion macros
#define FLOAT_TO_FIXED(x) ((int16_t)((x) * MLP_SCALE_FACTOR))
#define FIXED_TO_FLOAT(x) (((float)(x)) / MLP_SCALE_FACTOR)

// Layer 1: 84 -> 64
static const MLP_DTYPE mlp_dense_64_weights[84][64] = {
""")
        
        # Write dense_64 weights
        weight_lines = format_c_array_2d_fixed_point(dense_64_W, "mlp_dense_64_weights", "    ")
        for line in weight_lines:
            f.write(line + "\n")
        
        f.write("};\n\nstatic const MLP_DTYPE mlp_dense_64_bias[64] = {\n")
        
        # Write dense_64 bias
        bias_lines = format_c_array_1d_fixed_point(dense_64_b, "mlp_dense_64_bias", "    ")
        for line in bias_lines:
            f.write(line + "\n")
        
        f.write("};\n\n// Layer 2: 64 -> 32\nstatic const MLP_DTYPE mlp_dense_32_weights[64][32] = {\n")
        
        # Write dense_32 weights
        weight_lines = format_c_array_2d_fixed_point(dense_32_W, "mlp_dense_32_weights", "    ")
        for line in weight_lines:
            f.write(line + "\n")
        
        f.write("};\n\nstatic const MLP_DTYPE mlp_dense_32_bias[32] = {\n")
        
        # Write dense_32 bias
        bias_lines = format_c_array_1d_fixed_point(dense_32_b, "mlp_dense_32_bias", "    ")
        for line in bias_lines:
            f.write(line + "\n")
        
        f.write("};\n\n// Output layer: 32 -> 8\nstatic const MLP_DTYPE mlp_logits_weights[32][8] = {\n")
        
        # Write logits weights
        weight_lines = format_c_array_2d_fixed_point(logits_W, "mlp_logits_weights", "    ")
        for line in weight_lines:
            f.write(line + "\n")
        
        f.write("};\n\nstatic const MLP_DTYPE mlp_logits_bias[8] = {\n")
        
        # Write logits bias
        bias_lines = format_c_array_1d_fixed_point(logits_b, "mlp_logits_bias", "    ")
        for line in bias_lines:
            f.write(line + "\n")
        
        f.write("};\n\n#endif // MLP_WEIGHTS_H\n")
    
    print("Generated mlp_weights.h")
    
    # Now generate test data header
    io_dir = 'io_npy/'
    
    test_x, test_x_shape = read_npy_data(os.path.join(io_dir, 'test_x.npy'))
    test_y, test_y_shape = read_npy_data(os.path.join(io_dir, 'test_y.npy'))
    print(f"Test X: {test_x_shape}, Test Y: {test_y_shape}")
    
    with open('mlp_test_data.h', 'w') as f:
        f.write("""#ifndef MLP_TEST_DATA_H
#define MLP_TEST_DATA_H

#include "mlp_weights.h"

#define TEST_SAMPLES 10

// Test input data (10 samples) - converted to Q8.8 fixed-point
static const float test_input_data_float[10][84] = {
""")
        
        # Write test input data
        for i, sample in enumerate(test_x):
            sample_lines = format_c_array_1d(sample, f"test_input_data_float[{i}]", "    ")
            if i == 0:
                f.write("    { " + sample_lines[0].strip() + "\n")
            else:
                f.write("     { " + sample_lines[0].strip() + "\n")
            
            for line in sample_lines[1:]:
                f.write("       " + line.strip() + "\n")
            
            if i < len(test_x) - 1:
                f.write("     },\n")
            else:
                f.write("     }\n")
        
        f.write("};\n\n// Expected output data (10 samples) - raw logits\nstatic const float test_expected_output[10][8] = {\n")
        
        # Write expected output data
        for i, sample in enumerate(test_y):
            sample_lines = format_c_array_1d(sample, f"test_expected_output[{i}]", "    ")
            if i == 0:
                f.write("    { " + sample_lines[0].strip() + "\n")
            else:
                f.write("     { " + sample_lines[0].strip() + "\n")
            
            for line in sample_lines[1:]:
                f.write("       " + line.strip() + "\n")
            
            if i < len(test_y) - 1:
                f.write("     },\n")
            else:
                f.write("     }\n")
        
        f.write("};\n\n#endif // MLP_TEST_DATA_H\n")
    
    print("Generated mlp_test_data.h")
    print("\nWeight extraction complete!")

if __name__ == "__main__":
    main()

