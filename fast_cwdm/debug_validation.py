#!/usr/bin/env python3
"""
Quick diagnostic script to test validation synthesis components
Run this before training to identify issues early
"""

import torch as th
import numpy as np
import sys
import os

sys.path.append(".")

from fast_cwdm.guided_diffusion import dist_util
from fast_cwdm.guided_diffusion.bratsloader import BRATSVolumes
from fast_cwdm.guided_diffusion.script_util import (
    model_and_diffusion_defaults, create_model_and_diffusion, args_to_dict
)
from fast_cwdm.guided_diffusion.synthesis_utils import ComprehensiveMetrics
from fast_cwdm.DWT_IDWT.DWT_IDWT_layer import DWT_3D, IDWT_3D

def test_dataset_loading():
    """Test dataset loading and basic operations"""
    print("🔍 Testing dataset loading...")
    
    try:
        # Test dataset loading
        data_dir = "./fast_cwdm/datasets/BRATS2023/training"  # Adjust path as needed
        ds = BRATSVolumes(data_dir, mode='train', split='val', val_split_ratio=0.2)
        
        print(f"✅ Dataset loaded successfully")
        print(f"   Validation cases: {len(ds)}")
        print(f"   Crop info: {ds.get_crop_info()}")
        print(f"   Output dims: {ds.get_output_dimensions()}")
        print(f"   DWT dims: {ds.get_dwt_dimensions()}")
        
        # Test loading a single case
        if len(ds) > 0:
            batch = ds[0]
            print(f"✅ Sample batch loaded:")
            for key in ['t1n', 't1c', 't2w', 't2f']:
                print(f"   {key}: {batch[key].shape}")
            
            return True, ds
        else:
            print("❌ No validation cases found")
            return False, None
            
    except Exception as e:
        print(f"❌ Dataset loading failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, None

def test_dwt_operations():
    """Test DWT/IDWT operations with expected dimensions"""
    print("\n🔍 Testing DWT/IDWT operations...")
    
    try:
        device = th.device('cuda' if th.cuda.is_available() else 'cpu')
        
        # Create test tensor with validation dataset dimensions
        test_shape = (1, 1, 160, 224, 160)  # Cropped dimensions
        test_tensor = th.randn(*test_shape, device=device)
        print(f"   Test tensor shape: {test_tensor.shape}")
        
        # Test DWT
        dwt = DWT_3D('haar')
        dwt_components = dwt(test_tensor)
        LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = dwt_components
        
        print(f"✅ DWT successful:")
        print(f"   LLL shape: {LLL.shape}")
        print(f"   Other components: {LLH.shape}")
        
        # Test conditioning tensor creation (3 modalities)
        conditioning = th.cat([
            LLL / 3., LLH, LHL, LHH, HLL, HLH, HHL, HHH,
            LLL / 3., LLH, LHL, LHH, HLL, HLH, HHL, HHH,
            LLL / 3., LLH, LHL, LHH, HLL, HLH, HHL, HHH
        ], dim=1)
        
        print(f"✅ Conditioning tensor shape: {conditioning.shape}")
        
        # Test noise tensor
        noise = th.randn(1, 8, *LLL.shape[2:], device=device)
        print(f"✅ Noise tensor shape: {noise.shape}")
        
        # Test concatenation
        combined = th.cat([noise, conditioning], dim=1)
        print(f"✅ Combined input shape: {combined.shape}")
        
        if combined.shape[1] != 32:
            print(f"❌ Expected 32 channels, got {combined.shape[1]}")
            return False
        
        # Test IDWT
        idwt = IDWT_3D('haar')
        reconstructed = idwt(
            noise[:, 0:1] * 3., noise[:, 1:2], noise[:, 2:3], noise[:, 3:4],
            noise[:, 4:5], noise[:, 5:6], noise[:, 6:7], noise[:, 7:8]
        )
        print(f"✅ IDWT successful: {reconstructed.shape}")
        
        # Check if dimensions match original
        if reconstructed.shape == test_tensor.shape:
            print(f"✅ Dimension consistency maintained")
            return True
        else:
            print(f"❌ Dimension mismatch: {reconstructed.shape} vs {test_tensor.shape}")
            return False
            
    except Exception as e:
        print(f"❌ DWT/IDWT test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_model_creation():
    """Test model and diffusion creation"""
    print("\n🔍 Testing model creation...")
    
    try:
        # Create test arguments
        class Args:
            pass
        
        args = Args()
        args.image_size = 224
        args.num_channels = 64
        args.num_res_blocks = 2
        args.channel_mult = "1,2,2,4,4"
        args.learn_sigma = False
        args.class_cond = False
        args.use_checkpoint = False
        args.attention_resolutions = ""
        args.num_heads = 1
        args.num_head_channels = -1
        args.num_heads_upsample = -1
        args.use_scale_shift_norm = False
        args.dropout = 0.0
        args.resblock_updown = True
        args.use_fp16 = False
        args.use_new_attention_order = False
        args.dims = 3
        args.num_groups = 32
        args.bottleneck_attention = False
        args.resample_2d = False
        args.additive_skips = False
        args.use_freq = False
        args.predict_xstart = True
        args.noise_schedule = "linear"
        args.timestep_respacing = ""
        args.use_kl = False
        args.rescale_timesteps = False
        args.rescale_learned_sigmas = False
        args.in_channels = 32
        args.out_channels = 8
        args.diffusion_steps = 100
        args.sample_schedule = 'sampled'
        args.mode = 'i2i'
        args.dataset = "brats"
        
        # Create model and diffusion
        model, diffusion = create_model_and_diffusion(
            **args_to_dict(args, model_and_diffusion_defaults().keys())
        )
        diffusion.mode = 'i2i'
        
        device = th.device('cuda' if th.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        
        print(f"✅ Model created successfully")
        print(f"   Model device: {next(model.parameters()).device}")
        print(f"   Diffusion timesteps: {diffusion.num_timesteps}")
        print(f"   Diffusion mode: {getattr(diffusion, 'mode', 'default')}")
        
        # Test forward pass with dummy input
        dummy_input = th.randn(1, 32, 80, 112, 80, device=device)  # DWT dimensions
        dummy_timesteps = th.randint(0, diffusion.num_timesteps, (1,), device=device)
        
        with th.no_grad():
            output = model(dummy_input, dummy_timesteps)
            print(f"✅ Forward pass successful: {output.shape}")
            
        return True, model, diffusion
        
    except Exception as e:
        print(f"❌ Model creation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, None, None

def test_metrics_calculation():
    """Test metrics calculation"""
    print("\n🔍 Testing metrics calculation...")
    
    try:
        device = th.device('cuda' if th.cuda.is_available() else 'cpu')
        metrics_calc = ComprehensiveMetrics(device)
        
        # Create dummy data
        pred = th.randn(160, 224, 160, device=device)
        target = th.randn(160, 224, 160, device=device)
        
        # Ensure positive values for realistic brain data
        pred = th.clamp(pred, 0, 1)
        target = th.clamp(target, 0, 1)
        
        metrics = metrics_calc.calculate_metrics(pred, target, "test")
        
        print(f"✅ Metrics calculation successful:")
        for key, value in metrics.items():
            print(f"   {key}: {value}")
            
        return True
        
    except Exception as e:
        print(f"❌ Metrics calculation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all diagnostic tests"""
    print("🔬 Running FAST-CWDM Validation Diagnostics")
    print("=" * 50)
    
    results = {}
    
    # Test dataset loading
    results['dataset'], dataset = test_dataset_loading()
    
    # Test DWT operations
    results['dwt'] = test_dwt_operations()
    
    # Test model creation
    results['model'], model, diffusion = test_model_creation()
    
    # Test metrics
    results['metrics'] = test_metrics_calculation()
    
    # Summary
    print(f"\n📊 DIAGNOSTIC SUMMARY")
    print("=" * 30)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:12} {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print(f"\n🎉 All tests passed! Your validation should work.")
        print(f"   If you're still seeing errors, the issue might be in:")
        print(f"   1. Checkpoint loading/compatibility")
        print(f"   2. Memory issues during sampling")
        print(f"   3. Specific edge cases in your data")
    else:
        print(f"\n❌ Some tests failed. Fix these issues before training.")
    
    print(f"\n💡 Next steps:")
    print(f"   1. Apply the enhanced error handling code")
    print(f"   2. Run training with detailed debugging")
    print(f"   3. Check the full error messages")

if __name__ == "__main__":
    main()