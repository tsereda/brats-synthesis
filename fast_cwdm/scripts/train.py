"""
Enhanced training script with proper train/val split and validation-based model selection.
"""

import argparse
import numpy as np
import random
import sys
import torch as th
import wandb

sys.path.append(".")
sys.path.append("..")

from guided_diffusion import dist_util, logger
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (model_and_diffusion_defaults, create_model_and_diffusion,
                                          args_to_dict, add_dict_to_argparser)
from guided_diffusion.train_util import TrainLoop
from guided_diffusion.bratsloader import BRATSVolumes
from torch.utils.tensorboard import SummaryWriter


def main():
    args = create_argparser().parse_args()
    seed = args.seed
    th.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # --- wandb integration ---
    wandb.init(
        project="fast-cwmd-brats",
        entity="timgsereda",
        config=vars(args),
        mode="online"
    )
    # --- end wandb integration ---

    summary_writer = None
    if args.use_tensorboard:
        logdir = None
        if args.tensorboard_path:
            logdir = args.tensorboard_path
        summary_writer = SummaryWriter(log_dir=logdir)
        summary_writer.add_text(
            'config',
            '\n'.join([f'--{k}={repr(v)} <br/>' for k, v in vars(args).items()])
        )
        logger.configure(dir=summary_writer.get_logdir())
    else:
        logger.configure()

    dist_util.setup_dist(devices=args.devices)

    # Log configuration
    print(f"[CONFIG] sample_schedule: {getattr(args, 'sample_schedule', 'direct')}")
    print(f"[CONFIG] diffusion_steps: {getattr(args, 'diffusion_steps', 1000)}")
    print(f"[CONFIG] val_split_ratio: {args.val_split_ratio}")
    print(f"[CONFIG] val_interval: {args.val_interval}")
    print(f"[CONFIG] early_stopping_patience: {args.early_stopping_patience}")
    
    print("Creating model and diffusion...")
    arguments = args_to_dict(args, model_and_diffusion_defaults().keys())
    model, diffusion = create_model_and_diffusion(**arguments)
    model.to(dist_util.dev([0, 1]) if len(args.devices) > 1 else dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion, maxt=diffusion.num_timesteps)
    
    if args.dataset == 'brats':
        # Create training dataset
        print("🚂 Creating training dataset...")
        ds_train = BRATSVolumes(
            args.data_dir, 
            mode='train', 
            split='train',
            val_split_ratio=args.val_split_ratio,
            seed=args.seed
        )
        
        # Create validation dataset
        print("🧪 Creating validation dataset...")
        ds_val = BRATSVolumes(
            args.data_dir,
            mode='train',  # Same directory, different split
            split='val', 
            val_split_ratio=args.val_split_ratio,
            seed=args.seed
        )
        
        print(f"✅ Dataset split complete:")
        print(f"   Training: {len(ds_train)} cases")
        print(f"   Validation: {len(ds_val)} cases")
        print(f"   Split ratio: {args.val_split_ratio}")
        
        # Log split info to wandb
        wandb.log({
            'dataset/train_cases': len(ds_train),
            'dataset/val_cases': len(ds_val),
            'dataset/total_cases': len(ds_train) + len(ds_val),
            'dataset/val_ratio': args.val_split_ratio
        })
    
    # Create data loaders
    datal_train = th.utils.data.DataLoader(
        ds_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
    )
    
    # Note: Validation dataset is passed directly to TrainLoop, not as DataLoader
    # because validation is done on individual cases, not batches
    
    print("🚀 Starting enhanced training with validation...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=datal_train,
        val_data=ds_val,  # NEW: Pass validation dataset
        batch_size=args.batch_size,
        in_channels=args.in_channels,
        image_size=args.image_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        val_interval=args.val_interval,  # NEW: Validation interval
        resume_checkpoint=args.resume_checkpoint,
        resume_step=args.resume_step,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        dataset=args.dataset,
        summary_writer=summary_writer,
        mode='i2i',
        contr=args.contr,
        sample_schedule=args.sample_schedule,
        diffusion_steps=args.diffusion_steps,
        early_stopping_patience=args.early_stopping_patience,  # NEW: Early stopping
    ).run_loop()


def create_argparser():
    defaults = dict(
        seed=0,
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,
        ema_rate="0.9999",
        log_interval=100,
        save_interval=5000,
        val_interval=100,  # NEW: Validate every 100 steps
        val_split_ratio=0.2,  # NEW: 20% of data for validation
        early_stopping_patience=10,  # NEW: Stop if no improvement for 10 validations
        resume_checkpoint='',
        resume_step=0,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        dataset='brats',
        use_tensorboard=True,
        tensorboard_path='',
        devices=[0],
        dims=3,
        learn_sigma=False,
        num_groups=32,
        channel_mult="1,2,2,4,4",
        in_channels=24,
        out_channels=8,
        bottleneck_attention=False,
        num_workers=0,
        mode='default',
        renormalize=True,
        additive_skips=False,
        use_freq=False,
        contr='t1n',
        sample_schedule='direct',
    )
    from guided_diffusion.script_util import model_and_diffusion_defaults, add_dict_to_argparser
    defaults.update(model_and_diffusion_defaults())
    import argparse
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()