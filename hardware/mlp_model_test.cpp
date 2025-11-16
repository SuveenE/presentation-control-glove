#include "mlp_model.h"
#include "mlp_test_data.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include "hls_stream.h"

// Wrapper function for testing with arrays
void mlp_gesture_detection_wrapper(float input_data[MLP_INPUT_SIZE], float output_data[MLP_OUTPUT_SIZE]) {
    hls::stream<AXIS_wLAST> input_stream;
    hls::stream<AXIS_wLAST> output_stream;
    
    // Write input data to stream
    for (int i = 0; i < MLP_INPUT_SIZE; i++) {
        AXIS_wLAST temp;
        // Use union to correctly interpret the float bits
        union { float f; uint32_t i; } converter;
        converter.f = input_data[i];
        temp.data = converter.i;
        temp.last = (i == MLP_INPUT_SIZE - 1) ? 1 : 0;
        input_stream.write(temp);
    }
    
    // Call the actual function
    mlp_gesture_detection(input_stream, output_stream);
    
    // Read output data from stream
    for (int i = 0; i < MLP_OUTPUT_SIZE; i++) {
        AXIS_wLAST temp = output_stream.read();
        // Use union to correctly interpret the float bits
        union { float f; uint32_t i; } converter;
        converter.i = temp.data;
        output_data[i] = converter.f;
    }
}

void print_array(const char* name, const float* arr, int size) {
    std::cout << name << ": [";
    for (int i = 0; i < size; i++) {
        std::cout << std::fixed << std::setprecision(8) << arr[i];
        if (i < size - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}

int argmax_float(const float* data, int size) {
    float max_val = data[0];
    int max_idx = 0;
    
    for (int i = 1; i < size; i++) {
        if (data[i] > max_val) {
            max_val = data[i];
            max_idx = i;
        }
    }
    return max_idx;
}

int main() {
    std::cout << "=== MLP Gesture Detection Test - 2-IMU Test Model ===" << std::endl;
    std::cout << "Model Architecture: " << MLP_INPUT_SIZE << " -> " 
              << MLP_LAYER1_SIZE << " -> " << MLP_LAYER2_SIZE 
              << " -> " << MLP_OUTPUT_SIZE << std::endl << std::endl;
    
    const char* gesture_names[MLP_OUTPUT_SIZE] = {
        "no_gesture", "swipe_left", "swipe_right", "v_sign", "fist_squeeze",
        "roll_wrist_clockwise", "roll_wrist_counter_clockwise", 
        "single_knock_on_wrist"
    };
    
    float output[MLP_OUTPUT_SIZE];
    
    int correct_predictions = 0;
    int total_predictions = 10; // We have 10 test samples
    
    for (int sample = 0; sample < total_predictions; sample++) {
        std::cout << "--- Test Sample " << (sample + 1) << " ---" << std::endl;
        
        // Copy input data to non-const array (convert from float to match interface)
        float input_data[MLP_INPUT_SIZE];
        for (int i = 0; i < MLP_INPUT_SIZE; i++) {
            input_data[i] = test_input_data_float[sample][i];
        }
        
        // Run inference
        mlp_gesture_detection_wrapper(input_data, output);
        
        // Print results
        print_array("Input (first 10)", input_data, 10);
        print_array("Output logits", output, MLP_OUTPUT_SIZE);
        print_array("Expected", (float*)test_expected_output[sample], MLP_OUTPUT_SIZE);
        
        // Get predictions using float argmax for test comparison
        int predicted_class = argmax_float(output, MLP_OUTPUT_SIZE);
        int expected_class = argmax_float(test_expected_output[sample], MLP_OUTPUT_SIZE);
        
        std::cout << "Predicted: " << gesture_names[predicted_class] 
                  << " (class " << predicted_class << ")" << std::endl;
        std::cout << "Expected:  " << gesture_names[expected_class] 
                  << " (class " << expected_class << ")" << std::endl;
        
        if (predicted_class == expected_class) {
            std::cout << "✓ CORRECT" << std::endl;
            correct_predictions++;
        } else {
            std::cout << "✗ INCORRECT" << std::endl;
        }
        
        std::cout << std::endl;
    }
    
    std::cout << "=== Results ===" << std::endl;
    std::cout << "Accuracy: " << correct_predictions << "/" << total_predictions 
              << " (" << (100.0 * correct_predictions / total_predictions) << "%)" << std::endl;
    
    return 0;
}

