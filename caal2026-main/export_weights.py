"""
Weight Export Script for C/C++ Inference (Pure Python)
=========================================================

This script exports trained model weights to C-compatible format.
Uses only standard library - no PyTorch required!

The weights.txt file has a special format where the parameter name and shape 
are on one line, followed by the values on subsequent lines.
"""

import os
import re
from pathlib import Path

# Configuration
WEIGHTS_FILE = "caal2026/model_params/weights.txt"
OUTPUT_DIR = "caal2026/c_inference"
D_MODEL = 64
D_STATE = 64
NUM_CLASSES = 4
SEQ_LEN = 4096
INPUT_CHANNELS = 1  # Grayscale for GalaxyMNIST
HALF_STATE = 32  # d_state // 2


def parse_weights_file(filename):
    """Parse the weights.txt file and extract all weights."""
    print(f"Loading weights from: {filename}")
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    params = {}
    current_name = None
    current_shape = None
    current_values = []
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Skip empty lines
        if not line:
            i += 1
            continue
        
        # Check for parameter header like [name] Shape: [shape]
        if line.startswith('[') and 'Shape:' in line:
            # Save previous parameter
            if current_name and current_values:
                params[current_name] = {
                    'shape': current_shape,
                    'data': current_values
                }
            
            # Parse new parameter name and shape
            match = re.match(r'\[([^\]]+)\]\s+Shape:\s*\[(\d+(?:,\s*\d+)*)\]', line)
            if match:
                current_name = match.group(1)
                shape_str = match.group(2)
                current_shape = [int(x.strip()) for x in shape_str.split(',')]
                current_values = []
                
                # Check if values are on the same line (for small tensors)
                remaining = line.split(']', 1)
                if len(remaining) > 1 and remaining[1].strip():
                    # Values on same line after closing bracket
                    values_str = remaining[1].strip()
                    current_values = [float(v) for v in re.findall(r'-?\d+\.?\d*', values_str)]
        else:
            # This is a data line - parse values
            # Handle both comma-separated and space-separated values
            values = re.findall(r'-?\d+\.?\d*', line)
            if values:
                current_values.extend([float(v) for v in values])
        
        i += 1
    
    # Save last parameter
    if current_name and current_values:
        params[current_name] = {
            'shape': current_shape,
            'data': current_values
        }
    
    print(f"Found {len(params)} parameters")
    for name, info in params.items():
        print(f"  {name}: {info['shape']} ({len(info['data'])} values)")
    
    return params


def export_weights():
    """Export model weights to C header file format."""
    
    print("=" * 60)
    print("S4 Galaxy Classifier - Weight Export (Pure Python)")
    print("=" * 60)
    
    # Parse weights file
    params = parse_weights_file(WEIGHTS_FILE)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate C header file
    print(f"\nGenerating C header file: {OUTPUT_DIR}/model_weights.h")
    
    with open(f"{OUTPUT_DIR}/model_weights.h", 'w') as f:
        f.write("""/**
 * S4 Galaxy Classifier - Model Weights
 * =====================================
 * Auto-generated from weights.txt
 * 
 * Model Configuration (Grayscale):
 * - d_model: 64
 * - d_state: 64  
 * - num_classes: 4
 * - seq_len: 4096
 * - input_channels: 1
 * - half_state: 32
 * 
 * DO NOT EDIT MANUALLY
 */

#ifndef MODEL_WEIGHTS_H
#define MODEL_WEIGHTS_H

#ifdef __cplusplus
extern "C" {
#endif

// Float precision for model weights
#define MODEL_FLOAT float

""")
        
        # Write dimension constants
        f.write(f"// Model dimensions\n")
        f.write(f"#define D_MODEL {D_MODEL}\n")
        f.write(f"#define D_STATE {D_STATE}\n")
        f.write(f"#define HALF_STATE {HALF_STATE}\n")
        f.write(f"#define NUM_CLASSES {NUM_CLASSES}\n")
        f.write(f"#define SEQ_LEN {SEQ_LEN}\n")
        f.write(f"#define INPUT_CHANNELS {INPUT_CHANNELS}\n\n")
        
        # Write each weight array
        # Map parameter names to C array names
        param_mapping = {
            'hilbert_scan.indices': 'hilbert_indices',
            'uproject.weight': 'uproject_weight',
            'uproject.bias': 'uproject_bias',
            's4_1.log_dt': 's4d1_log_dt',
            's4_1.log_A_real': 's4d1_log_A_real',
            's4_1.A_imag': 's4d1_A_imag',
            's4_1.C': 's4d1_C',
            's4_1.D': 's4d1_D',
            's4_2.log_dt': 's4d2_log_dt',
            's4_2.log_A_real': 's4d2_log_A_real',
            's4_2.A_imag': 's4d2_A_imag',
            's4_2.C': 's4d2_C',
            's4_2.D': 's4d2_D',
            'fc.weight': 'fc_weight',
            'fc.bias': 'fc_bias',
        }
        
        for orig_name, c_name in param_mapping.items():
            if orig_name in params:
                data = params[orig_name]['data']
                shape = params[orig_name]['shape']
            else:
                print(f"  Warning: {orig_name} not found in weights file")
                # Calculate expected size
                if 'C' in orig_name:
                    data = [0.0] * (D_MODEL * HALF_STATE * 2)
                elif 'A_real' in orig_name or 'A_imag' in orig_name:
                    data = [0.0] * (D_MODEL * HALF_STATE)
                elif 'log_dt' in orig_name or 'D' in orig_name:
                    data = [0.0] * D_MODEL
                elif 'weight' in orig_name and 'uproject' in orig_name:
                    data = [0.0] * (D_MODEL * INPUT_CHANNELS)
                elif 'bias' in orig_name and 'uproject' in orig_name:
                    data = [0.0] * D_MODEL
                elif 'weight' in orig_name and 'fc' in orig_name:
                    data = [0.0] * (NUM_CLASSES * D_MODEL)
                elif 'bias' in orig_name and 'fc' in orig_name:
                    data = [0.0] * NUM_CLASSES
                else:
                    data = [0.0]
                shape = [len(data)]
            
            total_size = len(data)
            
            # Write dimension comment
            f.write(f"// {orig_name}\n")
            f.write(f"// Shape: {shape}\n")
            
            # Write size define
            safe_name = c_name.upper()
            f.write(f"#define {safe_name}_SIZE {total_size}\n")
            
            # Write data as static const array
            f.write(f"static const MODEL_FLOAT {c_name}[{total_size}] = {{\n")
            
            # Write values in rows of 8
            for i in range(0, len(data), 8):
                row = data[i:i+8]
                f.write("    " + ", ".join([f"{v:.10f}f" for v in row]) + ",\n")
            
            f.write("};\n\n")
        
        # Also write Hilbert indices directly if available
        if 'hilbert_scan.indices' in params:
            hilbert_data = params['hilbert_scan.indices']['data']
            f.write(f"// Hilbert curve lookup table\n")
            f.write(f"#define HILBERT_LOOKUP_SIZE {len(hilbert_data)}\n")
            f.write(f"static const int32_t hilbert_lookup[{len(hilbert_data)}] = {{\n")
            for i in range(0, len(hilbert_data), 16):
                row = hilbert_data[i:i+16]
                f.write("    " + ", ".join([str(int(v)) for v in row]) + ",\n")
            f.write("};\n\n")
        
        # Close header
        f.write("""#ifdef __cplusplus
}
#endif

#endif // MODEL_WEIGHTS_H
""")
    
    # Generate parameter summary
    total_params = sum(len(p['data']) for p in params.values())
    
    print(f"\n" + "=" * 60)
    print("Export Complete!")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Output: {OUTPUT_DIR}/model_weights.h")
    print("=" * 60)
    
    return params


if __name__ == "__main__":
    export_weights()
