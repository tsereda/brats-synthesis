#!/usr/bin/env python3
"""
BraTS-Lighthouse 2025 Challenge Submission Entrypoint
Fast-CWDM: Conditional Wavelet Diffusion Model for Medical Image Synthesis
"""

import os
import sys
import argparse

# Add current directory to path for imports
sys.path.append(".")
sys.path.append("/app")

def main():
    """Main entrypoint for BraTS challenge submission"""
    
    print("🚀 Fast-CWDM BraTS Challenge Submission Starting...")
    print("=" * 60)
    
    # Challenge paths (mounted by Docker)
    input_dir = "/input"
    output_dir = "/output" 
    checkpoint_dir = "/app/checkpoints"
    
    # Verify paths exist
    if not os.path.exists(input_dir):
        print(f"❌ Input directory not found: {input_dir}")
        sys.exit(1)
        
    if not os.path.exists(checkpoint_dir):
        print(f"❌ Checkpoint directory not found: {checkpoint_dir}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Count input cases
    input_cases = [d for d in os.listdir(input_dir) 
                   if os.path.isdir(os.path.join(input_dir, d)) and d.startswith('BraTS')]
    
    print(f"📁 Input directory: {input_dir}")
    print(f"📁 Output directory: {output_dir}")
    print(f"🏥 Found {len(input_cases)} cases to process")
    print(f"🧠 Using checkpoints from: {checkpoint_dir}")
    
    # Import and run the complete dataset synthesis
    try:
        from scripts.complete_dataset import main as synthesis_main
        
        # Override sys.argv to pass arguments to the synthesis script
        original_argv = sys.argv.copy()
        sys.argv = [
            'complete_dataset.py',
            '--input_dir', input_dir,
            '--output_dir', output_dir,
            '--checkpoint_dir', checkpoint_dir,
            '--device', 'cuda:0',
            '--evaluation_mode',  # Use complete cases and artificially exclude modalities
            '--wandb_mode', 'disabled',  # Disable wandb for submission
            '--diffusion_steps', '100',  # Fast sampling for challenge
        ]
        
        print("🔄 Starting synthesis pipeline...")
        synthesis_main()
        
        # Restore original argv
        sys.argv = original_argv
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all dependencies are installed in the container")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Synthesis failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Verify outputs were created
    output_files = [f for f in os.listdir(output_dir) if f.endswith('.nii.gz')]
    if output_files:
        print(f"✅ Successfully created {len(output_files)} synthesis outputs")
        print("📋 Output files:")
        for f in sorted(output_files)[:5]:  # Show first 5
            print(f"  - {f}")
        if len(output_files) > 5:
            print(f"  ... and {len(output_files) - 5} more")
    else:
        print("❌ No output files were created!")
        sys.exit(1)
    
    print("=" * 60)
    print("🎉 Fast-CWDM synthesis completed successfully!")
    print(f"📤 Ready for BraTS Challenge evaluation")

if __name__ == "__main__":
    main()