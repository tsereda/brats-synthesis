#!/usr/bin/env python3
"""
Enhanced BRATS dataset analysis to find optimal crop boundaries with validation
Identifies consistent brain regions across all cases for efficient cropping
Includes comprehensive validation to ensure no brain tissue loss
NOW WITH WAVELET-FRIENDLY DIMENSION CONSTRAINTS for Fast-CWDM

# NOTE: The validated crop bounds (160x224x160, x:39-199, y:17-225, z:0-160) are used everywhere in the codebase for loader, training, and conversion.
"""

import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from tqdm import tqdm
import json
import random
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

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

def make_wavelet_friendly(crop_bounds, divisor=16, original_shape=(240, 240, 155)):
    """
    Adjust crop bounds to be wavelet-friendly (divisible by divisor)
    
    Args:
        crop_bounds: Dict with x_min, x_max, y_min, y_max, z_min, z_max
        divisor: Divisor for wavelet compatibility (8, 16, or 32)
        original_shape: Original image dimensions for boundary checks
    
    Returns:
        Dict with wavelet-friendly crop bounds
    """
    print(f"\n🔧 Making crop bounds wavelet-friendly (divisible by {divisor})...")
    
    friendly_bounds = crop_bounds.copy()
    dimensions = ['x', 'y', 'z']
    
    for i, dim in enumerate(dimensions):
        # Current size
        current_size = crop_bounds[f'{dim}_max'] - crop_bounds[f'{dim}_min']
        
        # Round up to nearest multiple of divisor
        friendly_size = ((current_size + divisor - 1) // divisor) * divisor
        
        # Calculate expansion needed
        expansion = friendly_size - current_size
        
        # Try to expand symmetrically, but respect original image boundaries
        min_expansion = expansion // 2
        max_expansion = expansion - min_expansion
        
        new_min = max(0, crop_bounds[f'{dim}_min'] - min_expansion)
        new_max = min(original_shape[i], crop_bounds[f'{dim}_max'] + max_expansion)
        
        # If we can't expand max, try expanding min more
        if new_max - new_min < friendly_size:
            shortfall = friendly_size - (new_max - new_min)
            new_min = max(0, new_min - shortfall)
        
        # If still not enough space, expand max beyond boundary (will be clipped later)
        if new_max - new_min < friendly_size:
            new_max = new_min + friendly_size
            if new_max > original_shape[i]:
                print(f"⚠️  Warning: {dim.upper()}-dimension wavelet-friendly size ({friendly_size}) "
                      f"exceeds original dimension ({original_shape[i]})")
                new_max = original_shape[i]
                new_min = max(0, new_max - friendly_size)
        
        friendly_bounds[f'{dim}_min'] = int(new_min)
        friendly_bounds[f'{dim}_max'] = int(new_max)
        
        actual_size = friendly_bounds[f'{dim}_max'] - friendly_bounds[f'{dim}_min']
        
        print(f"  {dim.upper()}: {current_size} → {actual_size} "
              f"({actual_size % divisor == 0 and '✅' or '❌'} divisible by {divisor})")
    
    return friendly_bounds

def test_wavelet_compatibility(crop_bounds, test_divisors=[8, 16, 32]):
    """Test if crop bounds are compatible with different wavelet levels"""
    
    x_size = crop_bounds['x_max'] - crop_bounds['x_min']
    y_size = crop_bounds['y_max'] - crop_bounds['y_min']
    z_size = crop_bounds['z_max'] - crop_bounds['z_min']
    
    print(f"\n🧪 WAVELET COMPATIBILITY TEST:")
    print(f"Crop dimensions: {x_size} × {y_size} × {z_size}")
    
    for divisor in test_divisors:
        levels = int(np.log2(divisor))
        x_ok = x_size % divisor == 0
        y_ok = y_size % divisor == 0
        z_ok = z_size % divisor == 0
        all_ok = x_ok and y_ok and z_ok
        
        status = "✅" if all_ok else "❌"
        print(f"  {levels} levels (÷{divisor}): {status} "
              f"X:{x_ok and '✅' or '❌'} Y:{y_ok and '✅' or '❌'} Z:{z_ok and '✅' or '❌'}")
    
    return x_size % 8 == 0 and y_size % 8 == 0 and z_size % 8 == 0

def analyze_single_case(case_dir, modalities=['t1n', 't1c', 't2w', 't2f'], include_seg=False):
    """Analyze brain bounds for a single case across all modalities"""
    case_name = os.path.basename(case_dir)
    case_bounds = {}
    seg_bounds = None
    
    # Analyze imaging modalities
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
    
    # Analyze segmentation mask if available and requested
    if include_seg:
        seg_path = os.path.join(case_dir, f"{case_name}-seg.nii.gz")
        if os.path.exists(seg_path):
            try:
                seg_img = nib.load(seg_path)
                seg_data = seg_img.get_fdata()
                seg_bounds = find_brain_bounds_3d(seg_data, threshold=0.5)  # Different threshold for masks
            except Exception as e:
                print(f"Error processing segmentation {seg_path}: {e}")
    
    return case_bounds, seg_bounds

def validate_crop_with_samples(data_dir, crop_bounds, num_samples=5, output_dir="./crop_analysis_results", 
                              bounds_name="optimal"):
    """Validate crop boundaries by visualizing random samples"""
    print(f"\n🔍 VALIDATION: Checking {num_samples} random samples for {bounds_name} crop...")
    
    # Find all case directories
    case_dirs = [d for d in os.listdir(data_dir) 
                if os.path.isdir(os.path.join(data_dir, d))]
    
    # Select random samples
    sample_cases = random.sample(case_dirs, min(num_samples, len(case_dirs)))
    
    validation_results = {
        'samples_checked': len(sample_cases),
        'brain_tissue_preserved': 0,
        'segmentation_preserved': 0,
        'samples_with_seg': 0,
        'details': []
    }
    
    for i, case_name in enumerate(sample_cases):
        case_dir = os.path.join(data_dir, case_name)
        print(f"  📋 Validating case {i+1}/{len(sample_cases)}: {case_name}")
        
        # Load T1n for validation
        t1n_path = os.path.join(case_dir, f"{case_name}-t1n.nii.gz")
        seg_path = os.path.join(case_dir, f"{case_name}-seg.nii.gz")
        
        if not os.path.exists(t1n_path):
            continue
        
        try:
            # Load T1n image
            t1n_img = nib.load(t1n_path)
            t1n_data = t1n_img.get_fdata()
            
            # Check if brain tissue exists outside crop boundaries
            brain_bounds = find_brain_bounds_3d(t1n_data)
            
            if brain_bounds:
                # Check if any brain tissue would be lost
                tissue_preserved = (
                    brain_bounds['x_min'] >= crop_bounds['x_min'] and
                    brain_bounds['x_max'] <= crop_bounds['x_max'] and
                    brain_bounds['y_min'] >= crop_bounds['y_min'] and
                    brain_bounds['y_max'] <= crop_bounds['y_max'] and
                    brain_bounds['z_min'] >= crop_bounds['z_min'] and
                    brain_bounds['z_max'] <= crop_bounds['z_max']
                )
                
                if tissue_preserved:
                    validation_results['brain_tissue_preserved'] += 1
                
                sample_result = {
                    'case': case_name,
                    'brain_preserved': tissue_preserved,
                    'brain_bounds': brain_bounds,
                    'crop_bounds': crop_bounds
                }
                
                # Check segmentation if available
                seg_data = None
                if os.path.exists(seg_path):
                    validation_results['samples_with_seg'] += 1
                    seg_img = nib.load(seg_path)
                    seg_data = seg_img.get_fdata()
                    seg_bounds = find_brain_bounds_3d(seg_data, threshold=0.5)
                    
                    if seg_bounds:
                        seg_preserved = (
                            seg_bounds['x_min'] >= crop_bounds['x_min'] and
                            seg_bounds['x_max'] <= crop_bounds['x_max'] and
                            seg_bounds['y_min'] >= crop_bounds['y_min'] and
                            seg_bounds['y_max'] <= crop_bounds['y_max'] and
                            seg_bounds['z_min'] >= crop_bounds['z_min'] and
                            seg_bounds['z_max'] <= crop_bounds['z_max']
                        )
                        
                        if seg_preserved:
                            validation_results['segmentation_preserved'] += 1
                        
                        sample_result['seg_preserved'] = seg_preserved
                        sample_result['seg_bounds'] = seg_bounds
                
                validation_results['details'].append(sample_result)
                
                # Create validation visualization
                create_validation_visualization(
                    t1n_data, crop_bounds, case_name, i, output_dir,
                    seg_data=seg_data, bounds_name=bounds_name
                )
                
        except Exception as e:
            print(f"    ⚠️  Error validating {case_name}: {e}")
    
    return validation_results

def create_validation_visualization(img_data, crop_bounds, case_name, sample_idx, output_dir, 
                                  seg_data=None, bounds_name="optimal"):
    """Create before/after visualization for validation"""
    os.makedirs(f"{output_dir}/validation_samples", exist_ok=True)
    
    # Select middle slices for visualization
    mid_x = img_data.shape[0] // 2
    mid_y = img_data.shape[1] // 2
    mid_z = img_data.shape[2] // 2
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original images (top row)
    axes[0, 0].imshow(img_data[mid_x, :, :], cmap='gray')
    axes[0, 0].set_title(f'Original - Sagittal (X={mid_x})')
    axes[0, 0].add_patch(Rectangle(
        (crop_bounds['z_min'], crop_bounds['y_min']),
        crop_bounds['z_max'] - crop_bounds['z_min'],
        crop_bounds['y_max'] - crop_bounds['y_min'],
        linewidth=2, edgecolor='red', facecolor='none'
    ))
    
    axes[0, 1].imshow(img_data[:, mid_y, :], cmap='gray')
    axes[0, 1].set_title(f'Original - Coronal (Y={mid_y})')
    axes[0, 1].add_patch(Rectangle(
        (crop_bounds['z_min'], crop_bounds['x_min']),
        crop_bounds['z_max'] - crop_bounds['z_min'],
        crop_bounds['x_max'] - crop_bounds['x_min'],
        linewidth=2, edgecolor='red', facecolor='none'
    ))
    
    axes[0, 2].imshow(img_data[:, :, mid_z], cmap='gray')
    axes[0, 2].set_title(f'Original - Axial (Z={mid_z})')
    axes[0, 2].add_patch(Rectangle(
        (crop_bounds['y_min'], crop_bounds['x_min']),
        crop_bounds['y_max'] - crop_bounds['y_min'],
        crop_bounds['x_max'] - crop_bounds['x_min'],
        linewidth=2, edgecolor='red', facecolor='none'
    ))
    
    # Cropped images (bottom row)
    cropped_img = img_data[
        crop_bounds['x_min']:crop_bounds['x_max'],
        crop_bounds['y_min']:crop_bounds['y_max'],
        crop_bounds['z_min']:crop_bounds['z_max']
    ]
    
    crop_mid_x = cropped_img.shape[0] // 2
    crop_mid_y = cropped_img.shape[1] // 2
    crop_mid_z = cropped_img.shape[2] // 2
    
    axes[1, 0].imshow(cropped_img[crop_mid_x, :, :], cmap='gray')
    axes[1, 0].set_title(f'Cropped - Sagittal')
    
    axes[1, 1].imshow(cropped_img[:, crop_mid_y, :], cmap='gray')
    axes[1, 1].set_title(f'Cropped - Coronal')
    
    axes[1, 2].imshow(cropped_img[:, :, crop_mid_z], cmap='gray')
    axes[1, 2].set_title(f'Cropped - Axial')
    
    # Add segmentation overlay if available
    if seg_data is not None:
        for ax, view in zip([axes[0, 0], axes[0, 1], axes[0, 2]], 
                           [seg_data[mid_x, :, :], seg_data[:, mid_y, :], seg_data[:, :, mid_z]]):
            ax.contour(view, levels=[0.5], colors='yellow', linewidths=1, alpha=0.7)
    
    plt.suptitle(f'{bounds_name.title()} Crop Validation - Sample {sample_idx + 1}: {case_name}', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/validation_samples/{bounds_name}_sample_{sample_idx + 1}_{case_name}.png", 
                dpi=150, bbox_inches='tight')
    plt.close()

def analyze_dataset(data_dir, max_cases=None, include_seg_validation=True):
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
    
    seg_bounds_list = []
    original_shapes = []
    valid_cases = 0
    
    for case_dir_name in tqdm(case_dirs, desc="Analyzing cases"):
        case_dir = os.path.join(data_dir, case_dir_name)
        case_bounds, seg_bounds = analyze_single_case(case_dir, include_seg=include_seg_validation)
        
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
        
        # Collect segmentation bounds if available
        if seg_bounds:
            seg_bounds_list.append(seg_bounds)
    
    print(f"✅ Successfully analyzed {valid_cases} cases")
    if include_seg_validation:
        print(f"📊 Found {len(seg_bounds_list)} cases with segmentation masks")
    
    if valid_cases == 0:
        print("❌ No valid cases found!")
        return None
    
    return all_bounds, original_shapes, seg_bounds_list

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

def validate_segmentation_coverage(seg_bounds_list, crop_bounds):
    """Validate that segmentation masks fit within crop boundaries"""
    if not seg_bounds_list:
        return None
    
    validation_stats = {
        'total_cases': len(seg_bounds_list),
        'fully_covered': 0,
        'coverage_percentage': []
    }
    
    for seg_bounds in seg_bounds_list:
        # Check if segmentation is fully within crop bounds
        fully_covered = (
            seg_bounds['x_min'] >= crop_bounds['x_min'] and
            seg_bounds['x_max'] <= crop_bounds['x_max'] and
            seg_bounds['y_min'] >= crop_bounds['y_min'] and
            seg_bounds['y_max'] <= crop_bounds['y_max'] and
            seg_bounds['z_min'] >= crop_bounds['z_min'] and
            seg_bounds['z_max'] <= crop_bounds['z_max']
        )
        
        if fully_covered:
            validation_stats['fully_covered'] += 1
        
        # Calculate coverage percentage (rough estimate)
        seg_volume = (
            (seg_bounds['x_max'] - seg_bounds['x_min']) *
            (seg_bounds['y_max'] - seg_bounds['y_min']) *
            (seg_bounds['z_max'] - seg_bounds['z_min'])
        )
        
        # Intersection volume
        intersect_x = max(0, min(seg_bounds['x_max'], crop_bounds['x_max']) - 
                         max(seg_bounds['x_min'], crop_bounds['x_min']))
        intersect_y = max(0, min(seg_bounds['y_max'], crop_bounds['y_max']) - 
                         max(seg_bounds['y_min'], crop_bounds['y_min']))
        intersect_z = max(0, min(seg_bounds['z_max'], crop_bounds['z_max']) - 
                         max(seg_bounds['z_min'], crop_bounds['z_min']))
        
        intersect_volume = intersect_x * intersect_y * intersect_z
        coverage_pct = (intersect_volume / seg_volume) * 100 if seg_volume > 0 else 0
        validation_stats['coverage_percentage'].append(coverage_pct)
    
    return validation_stats

def visualize_crop_analysis(all_bounds, optimal_crop_bounds, wavelet_crop_bounds, original_shape, 
                           output_dir, seg_validation_stats=None):
    """Create visualizations of the crop analysis with both optimal and wavelet-friendly bounds"""
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
            optimal_value = optimal_crop_bounds[key]
            ax.axvline(optimal_value, color='red', linestyle='--', linewidth=2, 
                      label=f'Optimal: {optimal_value}')
            
            # Add wavelet-friendly crop line
            wavelet_value = wavelet_crop_bounds[key]
            ax.axvline(wavelet_value, color='blue', linestyle='-', linewidth=2, 
                      label=f'Wavelet-friendly: {wavelet_value}')
            
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
    plt.savefig(os.path.join(output_dir, 'crop_analysis_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create validation summary plot if segmentation data available
    if seg_validation_stats:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Coverage percentage histogram
        ax1.hist(seg_validation_stats['coverage_percentage'], bins=20, alpha=0.7, edgecolor='black')
        ax1.axvline(np.mean(seg_validation_stats['coverage_percentage']), 
                   color='red', linestyle='--', label=f"Mean: {np.mean(seg_validation_stats['coverage_percentage']):.1f}%")
        ax1.set_xlabel('Segmentation Coverage (%)')
        ax1.set_ylabel('Number of cases')
        ax1.set_title('Segmentation Coverage by Crop Bounds')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Summary bar chart
        fully_covered_pct = (seg_validation_stats['fully_covered'] / seg_validation_stats['total_cases']) * 100
        ax2.bar(['Fully Covered', 'Partially Covered'], 
               [fully_covered_pct, 100 - fully_covered_pct],
               color=['green', 'orange'], alpha=0.7)
        ax2.set_ylabel('Percentage of Cases')
        ax2.set_title('Segmentation Coverage Summary')
        ax2.set_ylim(0, 100)
        
        # Add percentage labels on bars
        ax2.text(0, fully_covered_pct + 2, f'{fully_covered_pct:.1f}%', ha='center')
        ax2.text(1, (100 - fully_covered_pct) + 2, f'{100 - fully_covered_pct:.1f}%', ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'segmentation_validation.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"📊 Visualization saved to: {output_dir}/crop_analysis_comparison.png")
    if seg_validation_stats:
        print(f"📊 Segmentation validation saved to: {output_dir}/segmentation_validation.png")

def calculate_efficiency_gains(original_shape, crop_bounds, name=""):
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
        'name': name,
        'original_shape': original_shape,
        'cropped_shape': cropped_shape,
        'original_volume': int(original_volume),
        'cropped_volume': int(cropped_volume),
        'efficiency_gain': float(efficiency_gain),
        'memory_reduction_percent': float(memory_reduction)
    }

def save_crop_config(optimal_crop_bounds, wavelet_crop_bounds, optimal_efficiency_stats, 
                    wavelet_efficiency_stats, validation_results, output_dir, wavelet_divisor):
    """Save crop configuration for use in training"""
    
    # Convert numpy types to native Python types for JSON serialization
    optimal_bounds_json = {k: int(v) for k, v in optimal_crop_bounds.items()}
    wavelet_bounds_json = {k: int(v) for k, v in wavelet_crop_bounds.items()}
    
    crop_config = {
        'optimal_crop_bounds': optimal_bounds_json,
        'wavelet_friendly_crop_bounds': wavelet_bounds_json,
        'wavelet_divisor': wavelet_divisor,
        'optimal_efficiency_stats': optimal_efficiency_stats,
        'wavelet_friendly_efficiency_stats': wavelet_efficiency_stats,
        'validation_results': validation_results,
        'recommended_bounds': 'wavelet_friendly_crop_bounds',  # Recommend wavelet-friendly
        'usage_instructions': {
            'description': 'Apply wavelet-friendly crop bounds to BRATS images before training with Fast-CWDM',
            'code_example': {
                'cropping': 'img_cropped = img_data[x_min:x_max, y_min:y_max, z_min:z_max]',
                'integration': 'Add to bratsloader.py __getitem__ method',
                'wavelet_note': f'These bounds ensure dimensions are divisible by {wavelet_divisor} for perfect DWT/IDWT reconstruction'
            }
        }
    }
    
    config_path = os.path.join(output_dir, 'optimal_crop_config.json')
    with open(config_path, 'w') as f:
        json.dump(crop_config, f, indent=2)
    
    print(f"💾 Crop configuration saved to: {config_path}")
    return config_path

def generate_integration_code(wavelet_crop_bounds, wavelet_divisor):
    """Generate code snippet for integration into bratsloader.py"""
    
    code_snippet = f'''
# Add this to bratsloader.py __getitem__ method after loading image data
# WAVELET-FRIENDLY CROP BOUNDS (divisible by {wavelet_divisor} for Fast-CWDM)

def apply_optimal_crop(img_data):
    """Apply wavelet-friendly crop bounds determined by dataset analysis"""
    # Wavelet-friendly crop bounds (validated for Fast-CWDM)
    x_min, x_max = {wavelet_crop_bounds['x_min']}, {wavelet_crop_bounds['x_max']}
    y_min, y_max = {wavelet_crop_bounds['y_min']}, {wavelet_crop_bounds['y_max']}
    z_min, z_max = {wavelet_crop_bounds['z_min']}, {wavelet_crop_bounds['z_max']}
    
    # Verify dimensions are wavelet-compatible
    x_size = x_max - x_min  # {wavelet_crop_bounds['x_max'] - wavelet_crop_bounds['x_min']}
    y_size = y_max - y_min  # {wavelet_crop_bounds['y_max'] - wavelet_crop_bounds['y_min']}
    z_size = z_max - z_min  # {wavelet_crop_bounds['z_max'] - wavelet_crop_bounds['z_min']}
    
    assert x_size % {wavelet_divisor} == 0, f"X dimension {{x_size}} not divisible by {wavelet_divisor}"
    assert y_size % {wavelet_divisor} == 0, f"Y dimension {{y_size}} not divisible by {wavelet_divisor}"
    assert z_size % {wavelet_divisor} == 0, f"Z dimension {{z_size}} not divisible by {wavelet_divisor}"
    
    # Apply crop
    img_cropped = img_data[x_min:x_max, y_min:y_max, z_min:z_max]
    return img_cropped

# Usage in __getitem__:
# img_np = nibabel.load(filedict['t1n']).get_fdata()
# img_cropped = apply_optimal_crop(img_np)  # Apply wavelet-friendly crop
# img_normalized = clip_and_normalize(img_cropped)
# # Continue with existing processing...

# For uncropping (inference):
def apply_uncrop(cropped_output, original_shape=(240, 240, 155)):
    """Uncrop model output back to original dimensions"""
    if isinstance(cropped_output, torch.Tensor):
        uncropped = torch.zeros(original_shape, dtype=cropped_output.dtype, device=cropped_output.device)
    else:
        uncropped = np.zeros(original_shape, dtype=cropped_output.dtype)
    
    uncropped[{wavelet_crop_bounds['x_min']}:{wavelet_crop_bounds['x_max']}, 
              {wavelet_crop_bounds['y_min']}:{wavelet_crop_bounds['y_max']}, 
              {wavelet_crop_bounds['z_min']}:{wavelet_crop_bounds['z_max']}] = cropped_output
    
    return uncropped
'''
    
    return code_snippet

def main():
    parser = argparse.ArgumentParser(description="Analyze BRATS dataset for wavelet-friendly optimal cropping")
    parser.add_argument("--data_dir", default="./datasets/BRATS2023/training",
                       help="Directory containing BRATS cases")
    parser.add_argument("--output_dir", default="./crop_analysis_results",
                       help="Output directory for analysis results")
    parser.add_argument("--max_cases", type=int, default=None,
                       help="Maximum number of cases to analyze (default: all)")
    parser.add_argument("--safety_margin", type=int, default=5,
                       help="Safety margin in pixels (default: 5)")
    parser.add_argument("--validation_samples", type=int, default=10,
                       help="Number of random samples to validate (default: 10)")
    parser.add_argument("--skip_seg_validation", action="store_true",
                       help="Skip segmentation mask validation")
    parser.add_argument("--wavelet_divisor", type=int, default=16,
                       help="Wavelet divisor for dimension constraints (8, 16, or 32)")
    parser.add_argument("--skip_wavelet_friendly", action="store_true",
                       help="Skip wavelet-friendly adjustment (not recommended for Fast-CWDM)")
    
    args = parser.parse_args()
    
    # Validate wavelet divisor
    if args.wavelet_divisor not in [8, 16, 32]:
        print(f"⚠️  Warning: Unusual wavelet divisor {args.wavelet_divisor}. Recommended: 8, 16, or 32")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"🎯 WAVELET-FRIENDLY CROP ANALYSIS")
    print(f"Divisor constraint: {args.wavelet_divisor} (for {int(np.log2(args.wavelet_divisor))} DWT levels)")
    
    # Analyze dataset
    include_seg = not args.skip_seg_validation
    result = analyze_dataset(args.data_dir, args.max_cases, include_seg_validation=include_seg)
    
    if result is None:
        print("❌ Analysis failed - no valid cases found")
        return
    
    all_bounds, original_shapes, seg_bounds_list = result
    
    # Get the most common original shape
    if original_shapes:
        original_shape = max(set(map(tuple, original_shapes)), key=original_shapes.count)
        print(f"📏 Original image shape: {original_shape}")
    else:
        original_shape = (240, 240, 155)  # BRATS default
        print(f"⚠️  Using default BRATS shape: {original_shape}")
    
    # Calculate optimal crop (brain-tissue based)
    optimal_crop_bounds = calculate_optimal_crop(all_bounds, args.safety_margin)
    
    # Calculate efficiency gains for optimal crop
    optimal_efficiency_stats = calculate_efficiency_gains(original_shape, optimal_crop_bounds, "optimal")
    
    # Test wavelet compatibility of optimal crop
    optimal_wavelet_compatible = test_wavelet_compatibility(optimal_crop_bounds, [8, 16, 32])
    
    if args.skip_wavelet_friendly:
        print("⚠️  Skipping wavelet-friendly adjustment (may cause DWT/IDWT issues)")
        wavelet_crop_bounds = optimal_crop_bounds
        wavelet_efficiency_stats = optimal_efficiency_stats
    else:
        # Make wavelet-friendly version
        wavelet_crop_bounds = make_wavelet_friendly(
            optimal_crop_bounds, args.wavelet_divisor, original_shape
        )
        
        # Calculate efficiency gains for wavelet-friendly crop
        wavelet_efficiency_stats = calculate_efficiency_gains(
            original_shape, wavelet_crop_bounds, "wavelet_friendly"
        )
        
        # Test wavelet compatibility of adjusted crop
        test_wavelet_compatibility(wavelet_crop_bounds, [8, 16, 32])
    
    # Validate segmentation coverage for both crop types
    optimal_seg_validation = None
    wavelet_seg_validation = None
    
    if include_seg and seg_bounds_list:
        optimal_seg_validation = validate_segmentation_coverage(seg_bounds_list, optimal_crop_bounds)
        if not args.skip_wavelet_friendly:
            wavelet_seg_validation = validate_segmentation_coverage(seg_bounds_list, wavelet_crop_bounds)
    
    # Validate with random samples for both crop types
    optimal_sample_validation = validate_crop_with_samples(
        args.data_dir, optimal_crop_bounds, args.validation_samples, args.output_dir, "optimal"
    )
    
    wavelet_sample_validation = optimal_sample_validation
    if not args.skip_wavelet_friendly:
        wavelet_sample_validation = validate_crop_with_samples(
            args.data_dir, wavelet_crop_bounds, args.validation_samples, args.output_dir, "wavelet_friendly"
        )
    
    # Print comparison results
    print(f"\n🎯 CROP ANALYSIS COMPARISON")
    print(f"=" * 60)
    
    print(f"\n📐 OPTIMAL CROP (brain-tissue based):")
    opt_shape = optimal_efficiency_stats['cropped_shape']
    print(f"  Shape: {opt_shape} (W×H×D)")
    print(f"  Memory reduction: {optimal_efficiency_stats['memory_reduction_percent']:.1f}%")
    print(f"  Wavelet compatible: {optimal_wavelet_compatible and '✅' or '❌'}")
    
    if not args.skip_wavelet_friendly:
        print(f"\n🔧 WAVELET-FRIENDLY CROP:")
        wav_shape = wavelet_efficiency_stats['cropped_shape']
        print(f"  Shape: {wav_shape} (W×H×D)")
        print(f"  Memory reduction: {wavelet_efficiency_stats['memory_reduction_percent']:.1f}%")
        print(f"  Wavelet compatible: ✅ (divisible by {args.wavelet_divisor})")
        
        # Compare memory reduction loss
        memory_loss = optimal_efficiency_stats['memory_reduction_percent'] - wavelet_efficiency_stats['memory_reduction_percent']
        print(f"  Memory reduction loss: {memory_loss:.1f}% (trade-off for wavelet compatibility)")
    
    # Print validation results
    print(f"\n✅ VALIDATION COMPARISON:")
    print(f"📋 Optimal crop validation:")
    print(f"   Brain tissue preserved: {optimal_sample_validation['brain_tissue_preserved']}/{optimal_sample_validation['samples_checked']} cases")
    
    if not args.skip_wavelet_friendly:
        print(f"📋 Wavelet-friendly crop validation:")
        print(f"   Brain tissue preserved: {wavelet_sample_validation['brain_tissue_preserved']}/{wavelet_sample_validation['samples_checked']} cases")
    
    # Generate visualizations
    visualize_crop_analysis(
        all_bounds, optimal_crop_bounds, wavelet_crop_bounds, original_shape, 
        args.output_dir, wavelet_seg_validation
    )
    
    # Compile validation results
    validation_summary = {
        'optimal_sample_validation': optimal_sample_validation,
        'wavelet_sample_validation': wavelet_sample_validation,
        'optimal_segmentation_validation': optimal_seg_validation,
        'wavelet_segmentation_validation': wavelet_seg_validation
    }
    
    # Save configuration (prefer wavelet-friendly)
    config_path = save_crop_config(
        optimal_crop_bounds, wavelet_crop_bounds, 
        optimal_efficiency_stats, wavelet_efficiency_stats,
        validation_summary, args.output_dir, args.wavelet_divisor
    )
    
    # Generate integration code (use wavelet-friendly bounds)
    integration_code = generate_integration_code(wavelet_crop_bounds, args.wavelet_divisor)
    code_path = os.path.join(args.output_dir, 'integration_code.py')
    with open(code_path, 'w') as f:
        f.write(integration_code)
    
    print(f"🔧 Integration code saved to: {code_path}")
    
    # Final recommendation
    recommended_bounds = wavelet_crop_bounds if not args.skip_wavelet_friendly else optimal_crop_bounds
    recommended_shape = (
        recommended_bounds['x_max'] - recommended_bounds['x_min'],
        recommended_bounds['y_max'] - recommended_bounds['y_min'],
        recommended_bounds['z_max'] - recommended_bounds['z_min']
    )
    
    print(f"\n🎉 FINAL RECOMMENDATION:")
    if not args.skip_wavelet_friendly:
        print(f"✅ Use WAVELET-FRIENDLY crop: {recommended_shape}")
        print(f"   - Guaranteed DWT/IDWT compatibility")
        print(f"   - {wavelet_efficiency_stats['memory_reduction_percent']:.1f}% memory reduction")
        print(f"   - Perfect for Fast-CWDM training")
    else:
        print(f"⚠️  Using OPTIMAL crop: {recommended_shape}")
        print(f"   - May have DWT/IDWT issues if not wavelet-compatible")
        print(f"   - {optimal_efficiency_stats['memory_reduction_percent']:.1f}% memory reduction")
    
    # Summary
    print(f"\n✅ ANALYSIS COMPLETE!")
    print(f"📁 Results saved to: {args.output_dir}")
    print(f"🖼️  Validation samples: {args.output_dir}/validation_samples/")
    print(f"💡 Next steps:")
    print(f"   1. Review validation visualizations")
    print(f"   2. Use crop bounds from: {config_path}")
    print(f"   3. Update bratsloader.py with integration code")
    print(f"   4. Update model args: --image_size should match crop dimensions")
    print(f"   5. Test DWT/IDWT compatibility before training")

if __name__ == "__main__":
    main()