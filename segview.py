#!/usr/bin/env python3
"""
Segmentation Map Visualizer
Visualizes a folder of segmentation maps with optional W&B logging

Usage:
python visualize_segmentations.py --input_dir /path/to/segmentations --output_dir ./visualizations --use_wandb
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import nibabel as nib
import glob
from pathlib import Path
import warnings

# Optional W&B import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("W&B not available. Install with: pip install wandb")

warnings.filterwarnings("ignore")


class SegmentationVisualizer:
    """Visualize BraTS-style segmentation maps"""
    
    def __init__(self, use_wandb=False, wandb_project="Segmentation-Visualization"):
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        
        # BraTS label colors
        self.colors = {
            0: [0, 0, 0],           # Background - Black
            1: [255, 0, 0],         # Tumor Core (TC) - Red  
            2: [0, 255, 0],         # Whole Tumor (WT) - Green
            3: [0, 0, 255],         # NCR/NET - Blue (if using 0,1,2,3 format)
            4: [255, 255, 0],       # Enhancing Tumor (ET) - Yellow
        }
        
        if self.use_wandb:
            wandb.init(project=wandb_project, name="seg_visualization")
            print("✓ W&B initialized")
    
    def create_colored_segmentation(self, seg_array):
        """Convert segmentation array to colored RGB image"""
        height, width = seg_array.shape
        colored = np.zeros((height, width, 3), dtype=np.uint8)
        
        unique_labels = np.unique(seg_array)
        print(f"    Unique labels found: {unique_labels}")
        
        for label in unique_labels:
            if label in self.colors:
                mask = seg_array == label
                colored[mask] = self.colors[label]
            else:
                # Use a random color for unknown labels
                mask = seg_array == label
                colored[mask] = [128, 128, 128]  # Gray for unknown
        
        return colored
    
    def get_interesting_slices(self, seg_array, num_slices=5):
        """Find slices with the most segmentation content"""
        slice_scores = []
        
        for i in range(seg_array.shape[2]):
            slice_2d = seg_array[:, :, i]
            # Score based on number of non-background pixels
            score = np.sum(slice_2d > 0)
            slice_scores.append((i, score))
        
        # Sort by score and take top slices
        slice_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Get well-distributed slices
        if len(slice_scores) >= num_slices:
            selected_indices = [x[0] for x in slice_scores[:num_slices]]
        else:
            # If not enough interesting slices, add some evenly spaced ones
            selected_indices = [x[0] for x in slice_scores]
            step = seg_array.shape[2] // (num_slices - len(selected_indices) + 1)
            for i in range(step, seg_array.shape[2], step):
                if i not in selected_indices and len(selected_indices) < num_slices:
                    selected_indices.append(i)
        
        return sorted(selected_indices[:num_slices])
    
    def create_multi_slice_visualization(self, seg_array, case_name, slices_to_show=5):
        """Create a multi-slice visualization"""
        interesting_slices = self.get_interesting_slices(seg_array, slices_to_show)
        
        fig, axes = plt.subplots(2, len(interesting_slices), figsize=(4*len(interesting_slices), 8))
        if len(interesting_slices) == 1:
            axes = axes.reshape(2, 1)
        
        for idx, slice_idx in enumerate(interesting_slices):
            slice_2d = seg_array[:, :, slice_idx]
            colored_slice = self.create_colored_segmentation(slice_2d)
            
            # Top row: Original segmentation with labels
            axes[0, idx].imshow(slice_2d, cmap='tab10', vmin=0, vmax=4)
            axes[0, idx].set_title(f'Slice {slice_idx}\nLabels: {np.unique(slice_2d)}')
            axes[0, idx].axis('off')
            
            # Bottom row: Colored visualization
            axes[1, idx].imshow(colored_slice)
            axes[1, idx].set_title(f'Colored')
            axes[1, idx].axis('off')
        
        plt.suptitle(f'{case_name}\nShape: {seg_array.shape}', fontsize=14)
        plt.tight_layout()
        
        return fig
    
    def create_summary_visualization(self, seg_array, case_name):
        """Create a comprehensive summary visualization"""
        # Get middle slice and some interesting slices
        mid_slice = seg_array.shape[2] // 2
        interesting_slices = self.get_interesting_slices(seg_array, 3)
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        
        # First row: Different views of middle slice
        mid_slice_2d = seg_array[:, :, mid_slice]
        colored_mid = self.create_colored_segmentation(mid_slice_2d)
        
        axes[0, 0].imshow(mid_slice_2d, cmap='tab10', vmin=0, vmax=4)
        axes[0, 0].set_title(f'Middle Slice ({mid_slice})')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(colored_mid)
        axes[0, 1].set_title('Colored')
        axes[0, 1].axis('off')
        
        # Sagittal view (middle)
        sag_slice = seg_array[seg_array.shape[0]//2, :, :]
        axes[0, 2].imshow(sag_slice, cmap='tab10', vmin=0, vmax=4)
        axes[0, 2].set_title('Sagittal View')
        axes[0, 2].axis('off')
        
        # Coronal view (middle)  
        cor_slice = seg_array[:, seg_array.shape[1]//2, :]
        axes[0, 3].imshow(cor_slice, cmap='tab10', vmin=0, vmax=4)
        axes[0, 3].set_title('Coronal View')
        axes[0, 3].axis('off')
        
        # Second row: Three most interesting slices
        for idx, slice_idx in enumerate(interesting_slices[:3]):
            slice_2d = seg_array[:, :, slice_idx]
            colored_slice = self.create_colored_segmentation(slice_2d)
            
            axes[1, idx].imshow(colored_slice)
            axes[1, idx].set_title(f'Slice {slice_idx}')
            axes[1, idx].axis('off')
        
        # Statistics panel
        unique_labels, counts = np.unique(seg_array, return_counts=True)
        stats_text = f"Shape: {seg_array.shape}\n"
        stats_text += f"Unique labels: {unique_labels}\n"
        for label, count in zip(unique_labels, counts):
            percentage = (count / seg_array.size) * 100
            stats_text += f"Label {label}: {count} ({percentage:.2f}%)\n"
        
        axes[1, 3].text(0.1, 0.9, stats_text, transform=axes[1, 3].transAxes, 
                       fontsize=10, verticalalignment='top', fontfamily='monospace')
        axes[1, 3].set_title('Statistics')
        axes[1, 3].axis('off')
        
        plt.suptitle(f'{case_name}', fontsize=16)
        plt.tight_layout()
        
        return fig
    
    def visualize_segmentation_file(self, file_path, output_dir, create_summary=True):
        """Visualize a single segmentation file"""
        case_name = Path(file_path).stem.replace('.nii', '')
        print(f"\nProcessing: {case_name}")
        
        try:
            # Load the segmentation
            seg_img = nib.load(file_path)
            seg_array = seg_img.get_fdata().astype(np.int32)
            
            print(f"  Shape: {seg_array.shape}")
            print(f"  Data type: {seg_array.dtype}")
            print(f"  Value range: {seg_array.min()} - {seg_array.max()}")
            print(f"  Unique values: {np.unique(seg_array)}")
            
            # Create visualizations
            if create_summary:
                # Summary visualization
                fig_summary = self.create_summary_visualization(seg_array, case_name)
                summary_path = os.path.join(output_dir, f"{case_name}_summary.png")
                fig_summary.savefig(summary_path, dpi=150, bbox_inches='tight')
                print(f"  ✓ Summary saved: {summary_path}")
                
                if self.use_wandb:
                    wandb.log({f"summary/{case_name}": wandb.Image(fig_summary)})
                
                plt.close(fig_summary)
            
            # Multi-slice visualization
            fig_slices = self.create_multi_slice_visualization(seg_array, case_name, 5)
            slices_path = os.path.join(output_dir, f"{case_name}_slices.png")
            fig_slices.savefig(slices_path, dpi=150, bbox_inches='tight')
            print(f"  ✓ Slices saved: {slices_path}")
            
            if self.use_wandb:
                wandb.log({f"slices/{case_name}": wandb.Image(fig_slices)})
            
            plt.close(fig_slices)
            
            return True
            
        except Exception as e:
            print(f"  ❌ Error processing {case_name}: {e}")
            return False
    
    def visualize_directory(self, input_dir, output_dir, file_pattern="*.nii.gz", 
                          max_files=None, create_summary=True):
        """Visualize all segmentation files in a directory"""
        
        print(f"🔍 Scanning directory: {input_dir}")
        print(f"📊 Output directory: {output_dir}")
        print(f"🎨 Using W&B: {self.use_wandb}")
        
        # Find all segmentation files
        search_pattern = os.path.join(input_dir, "**", file_pattern)
        seg_files = glob.glob(search_pattern, recursive=True)
        
        if not seg_files:
            print(f"❌ No files found matching pattern: {file_pattern}")
            return
        
        print(f"📁 Found {len(seg_files)} segmentation files")
        
        if max_files:
            seg_files = seg_files[:max_files]
            print(f"📊 Processing first {len(seg_files)} files")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Process each file
        successful = 0
        failed = 0
        
        for i, file_path in enumerate(seg_files):
            print(f"\n[{i+1}/{len(seg_files)}]", end="")
            
            success = self.visualize_segmentation_file(
                file_path, output_dir, create_summary
            )
            
            if success:
                successful += 1
            else:
                failed += 1
        
        # Summary
        print(f"\n{'='*60}")
        print(f"📊 VISUALIZATION COMPLETE")
        print(f"{'='*60}")
        print(f"✅ Successful: {successful}")
        print(f"❌ Failed: {failed}")
        print(f"📁 Total files: {len(seg_files)}")
        print(f"💾 Output directory: {output_dir}")
        
        if self.use_wandb:
            wandb.log({
                "summary/total_files": len(seg_files),
                "summary/successful": successful,
                "summary/failed": failed
            })
            print(f"🔗 Check your W&B project for interactive visualizations!")
    
    def finish(self):
        """Clean up"""
        if self.use_wandb:
            wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Visualize segmentation maps")
    parser.add_argument("--input_dir", type=str, required=True,
                       help="Directory containing segmentation files")
    parser.add_argument("--output_dir", type=str, default="./seg_visualizations",
                       help="Output directory for visualizations")
    parser.add_argument("--pattern", type=str, default="*.nii.gz",
                       help="File pattern to search for")
    parser.add_argument("--max_files", type=int, default=None,
                       help="Maximum number of files to process")
    parser.add_argument("--use_wandb", action="store_true",
                       help="Log visualizations to W&B")
    parser.add_argument("--wandb_project", type=str, default="Segmentation-Visualization",
                       help="W&B project name")
    parser.add_argument("--no_summary", action="store_true",
                       help="Skip summary visualizations (only create slice views)")
    
    args = parser.parse_args()
    
    # Check if input directory exists
    if not os.path.exists(args.input_dir):
        print(f"❌ Input directory does not exist: {args.input_dir}")
        return
    
    # Create visualizer
    visualizer = SegmentationVisualizer(
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project
    )
    
    try:
        # Run visualization
        visualizer.visualize_directory(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            file_pattern=args.pattern,
            max_files=args.max_files,
            create_summary=not args.no_summary
        )
    finally:
        visualizer.finish()


if __name__ == "__main__":
    main()