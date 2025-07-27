#!/usr/bin/env python3
"""
Analyze BRATS dataset to find optimal crop boundaries
Identifies consistent brain regions across all cases for efficient cropping
"""

import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from tqdm import tqdm
import json

def find_brain_bounds_3d(img_data, threshold=0.01):
    """Find 3D bounding box of brain tissue"""
    brain_mask = img_data > threshold
    
    # Find bounds in each dimension
    coords = np.where(brain_mask)
    
    if len(coords[0]) == 0:  # No brain tissue found
        return None
        
    bounds = {
        'x_min': int(coords[0].min()),
        'x_max': int(coords[0].max()),
        'y_min': int(coords[1].min()),
        'y_max': int(coords[1].max()),
        'z_min': int(coords[2].min()),
        'z_max': int(coords[2].max())
    }
    
    return bounds

def analyze_single_case(case_dir, modalities=['t1n', 't1c', 't2w', 't2f']):
    """Analyze brain bounds for a single case across all modalities"""
    case_name = os.path.basename(case_dir)
    case_bounds = {}
    
    for modality in modalities:
        file_path = os.path.join(case_dir, f"{case_name}-{modality}.nii.gz")
        
        if not os.path.exists(file_path):
            continue
            
        try:
            img = nib.load(file_path)
            img_data = img.get_fdata()
            
            bounds = find_brain_bounds_3d(img_data)
            if bounds:
                case_bounds[modality] = bounds
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    return case_bounds

def analyze_dataset(data_dir, max_cases=None):
    """Analyze entire dataset to find optimal crop boundaries"""
    print(f"🔍 Analyzing dataset: {data_dir}")
    
    # Find all case directories
    case_dirs = [d for d in os.listdir(data_dir) 
                if os.path.isdir(os.path.join(data_dir, d))]
    case_dirs.sort()
    
    if max_cases:
        case_dirs = case_dirs[:max_cases]
    
    print(f"Found {len(case_dirs)} cases to analyze")
    
    # Collect bounds from all cases
    all_bounds = {
        'x_min': [], 'x_max': [],
        'y_min': [], 'y_max': [],
        'z_min': [], 'z_max': []
    }
    
    original_shapes = []
    valid_cases = 0
    
    for case_dir_name in tqdm(case_dirs, desc="Analyzing cases"):
        case_dir = os.path.join(data_dir, case_dir_name)
        case_bounds = analyze_single_case(case_dir)
        
        if not case_bounds:
            continue
            
        valid_cases += 1
        
        # Get original shape (should be consistent)
        case_name = os.path.basename(case_dir)
        sample_file = os.path.join(case_dir, f"{case_name}-t1n.nii.gz")
        if os.path.exists(sample_file):
            img = nib.load(sample_file)
            original_shapes.append(img.shape)
        
        # Aggregate bounds across all modalities for this case
        case_aggregate = {
            'x_min': min(b['x_min'] for b in case_bounds.values()),
            'x_max': max(b['x_max'] for b in case_bounds.values()),
            'y_min': min(b['y_min'] for b in case_bounds.values()),
            'y_max': max(b['y_max'] for b in case_bounds.values()),
            'z_min': min(b['z_min'] for b in case_bounds.values()),
            'z_max': max(b['z_max'] for b in case_bounds.values())
        }
        
        # Add to global statistics
        for key, value in case_aggregate.items():
            all_bounds[key].append(value)
    
    print(f"✅ Successfully analyzed {valid_cases} cases")
    
    if valid_cases == 0:
        print("❌ No valid cases found!")
        return None
    
    return all_bounds, original_shapes

def calculate_optimal_crop(all_bounds, safety_margin=5):
    """Calculate optimal crop boundaries with safety margin"""
    
    # Use percentiles to handle outliers
    crop_bounds = {
        'x_min': max(0, int(np.percentile(all_bounds['x_min'], 5)) - safety_margin),
        'x_max': int(np.percentile(all_bounds['x_max'], 95)) + safety_margin,
        'y_min': max(0, int(np.percentile(all_bounds['y_min'], 5)) - safety_margin),
        'y_max': int(np.percentile(all_bounds['y_max'], 95)) + safety_margin,
        'z_min': max(0, int(np.percentile(all_bounds['z_min'], 5)) - safety_margin),
        'z_max': int(np.percentile(all_bounds['z_max'], 95)) + safety_margin,
    }
    
    return crop_bounds

def visualize_crop_analysis(all_bounds, crop_bounds, original_shape, output_dir):
    """Create visualizations of the crop analysis"""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    dimensions = ['x', 'y', 'z']
    bounds_types = ['min', 'max']
    
    for i, dim in enumerate(dimensions):
        for j, bound_type in enumerate(bounds_types):
            ax = axes[j, i]
            key = f"{dim}_{bound_type}"
            
            # Plot histogram of bounds
            ax.hist(all_bounds[key], bins=30, alpha=0.7, edgecolor='black')
            
            # Add optimal crop line
            crop_value = crop_bounds[key]
            ax.axvline(crop_value, color='red', linestyle='--', linewidth=2, 
                      label=f'Optimal crop: {crop_value}')
            
            # Add percentile lines
            p5 = np.percentile(all_bounds[key], 5)
            p95 = np.percentile(all_bounds[key], 95)
            ax.axvline(p5, color='orange', linestyle=':', alpha=0.7, label='5th percentile')
            ax.axvline(p95, color='orange', linestyle=':', alpha=0.7, label='95th percentile')
            
            ax.set_title(f'{dim.upper()}-dimension {bound_type}')
            ax.set_xlabel('Pixel coordinate')
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'crop_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Visualization saved to: {output_dir}/crop_analysis.png")

def calculate_efficiency_gains(original_shape, crop_bounds):
    """Calculate memory and computation efficiency gains"""
    
    original_volume = np.prod(original_shape)
    
    cropped_shape = (
        crop_bounds['x_max'] - crop_bounds['x_min'],
        crop_bounds['y_max'] - crop_bounds['y_min'],
        crop_bounds['z_max'] - crop_bounds['z_min']
    )
    
    cropped_volume = np.prod(cropped_shape)
    
    efficiency_gain = original_volume / cropped_volume
    memory_reduction = (1 - cropped_volume / original_volume) * 100
    
    return {
        'original_shape': original_shape,
        'cropped_shape': cropped_shape,
        'original_volume': original_volume,
        'cropped_volume': cropped_volume,
        'efficiency_gain': efficiency_gain,
        'memory_reduction_percent': memory_reduction
    }

def save_crop_config(crop_bounds, efficiency_stats, output_dir):
    """Save crop configuration for use in training"""
    
    crop_config = {
        'crop_bounds': crop_bounds,
        'efficiency_stats': efficiency_stats,
        'usage_instructions': {
            'description': 'Apply these crop bounds to BRATS images before training',
            'code_example': {
                'cropping': 'img_cropped = img_data[x_min:x_max, y_min:y_max, z_min:z_max]',
                'integration': 'Add to bratsloader.py __getitem__ method'
            }
        }
    }
    
    config_path = os.path.join(output_dir, 'optimal_crop_config.json')
    with open(config_path, 'w') as f:
        json.dump(crop_config, f, indent=2)
    
    print(f"💾 Crop configuration saved to: {config_path}")
    return config_path

def generate_integration_code(crop_bounds):
    """Generate code snippet for integration into bratsloader.py"""
    
    code_snippet = f'''
# Add this to bratsloader.py __getitem__ method after loading image data

def apply_optimal_crop(img_data):
    """Apply optimal crop bounds determined by dataset analysis"""
    # Optimal crop bounds (from analysis)
    x_min, x_max = {crop_bounds['x_min']}, {crop_bounds['x_max']}
    y_min, y_max = {crop_bounds['y_min']}, {crop_bounds['y_max']}
    z_min, z_max = {crop_bounds['z_min']}, {crop_bounds['z_max']}
    
    # Apply crop
    img_cropped = img_data[x_min:x_max, y_min:y_max, z_min:z_max]
    return img_cropped

# Usage in __getitem__:
# img_np = nibabel.load(filedict['t1n']).get_fdata()
# img_cropped = apply_optimal_crop(img_np)  # Apply optimal crop
# img_normalized = clip_and_normalize(img_cropped)
# # Continue with existing processing...
'''
    
    return code_snippet

def main():
    parser = argparse.ArgumentParser(description="Analyze BRATS dataset for optimal cropping")
    parser.add_argument("--data_dir", default="./datasets/BRATS2023/training",
                       help="Directory containing BRATS cases")
    parser.add_argument("--output_dir", default="./crop_analysis_results",
                       help="Output directory for analysis results")
    parser.add_argument("--max_cases", type=int, default=None,
                       help="Maximum number of cases to analyze (default: all)")
    parser.add_argument("--safety_margin", type=int, default=5,
                       help="Safety margin in pixels (default: 5)")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Analyze dataset
    result = analyze_dataset(args.data_dir, args.max_cases)
    
    if result is None:
        print("❌ Analysis failed - no valid cases found")
        return
    
    all_bounds, original_shapes = result
    
    # Get the most common original shape
    if original_shapes:
        original_shape = max(set(map(tuple, original_shapes)), key=original_shapes.count)
        print(f"📏 Original image shape: {original_shape}")
    else:
        original_shape = (240, 240, 155)  # BRATS default
        print(f"⚠️  Using default BRATS shape: {original_shape}")
    
    # Calculate optimal crop
    crop_bounds = calculate_optimal_crop(all_bounds, args.safety_margin)
    
    # Calculate efficiency gains
    efficiency_stats = calculate_efficiency_gains(original_shape, crop_bounds)
    
    # Print results
    print(f"\n🎯 OPTIMAL CROP ANALYSIS RESULTS")
    print(f"=" * 50)
    print(f"Original shape: {efficiency_stats['original_shape']}")
    print(f"Optimal crop shape: {efficiency_stats['cropped_shape']}")
    print(f"Memory reduction: {efficiency_stats['memory_reduction_percent']:.1f}%")
    print(f"Efficiency gain: {efficiency_stats['efficiency_gain']:.2f}x")
    
    print(f"\n📐 CROP BOUNDARIES:")
    print(f"X: [{crop_bounds['x_min']}:{crop_bounds['x_max']}] (width: {crop_bounds['x_max']-crop_bounds['x_min']})")
    print(f"Y: [{crop_bounds['y_min']}:{crop_bounds['y_max']}] (height: {crop_bounds['y_max']-crop_bounds['y_min']})")
    print(f"Z: [{crop_bounds['z_min']}:{crop_bounds['z_max']}] (depth: {crop_bounds['z_max']-crop_bounds['z_min']})")
    
    # Generate visualizations
    visualize_crop_analysis(all_bounds, crop_bounds, original_shape, args.output_dir)
    
    # Save configuration
    config_path = save_crop_config(crop_bounds, efficiency_stats, args.output_dir)
    
    # Generate integration code
    integration_code = generate_integration_code(crop_bounds)
    code_path = os.path.join(args.output_dir, 'integration_code.py')
    with open(code_path, 'w') as f:
        f.write(integration_code)
    
    print(f"🔧 Integration code saved to: {code_path}")
    
    # Summary
    print(f"\n✅ ANALYSIS COMPLETE!")
    print(f"📁 Results saved to: {args.output_dir}")
    print(f"💡 Next steps:")
    print(f"   1. Review the crop boundaries and visualizations")
    print(f"   2. Integrate the cropping code into bratsloader.py")
    print(f"   3. Update model input dimensions accordingly")
    print(f"   4. Retrain with {efficiency_stats['memory_reduction_percent']:.1f}% memory reduction!")

if __name__ == "__main__":
    main()