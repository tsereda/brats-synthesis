import copy
import functools
import os
import glob
import time

import blobfile as bf
import torch as th
import torch.distributed as dist
import torch.utils.tensorboard
from torch.optim import AdamW
import torch.cuda.amp as amp
import wandb

import itertools
import numpy as np

from . import dist_util, logger
from .resample import LossAwareSampler, UniformSampler
from fast_cwdm.DWT_IDWT.DWT_IDWT_layer import DWT_3D, IDWT_3D
from fast_cwdm.guided_diffusion.synthesis_utils import (
    ComprehensiveMetrics, synthesize_modality_shared, MODALITIES
)

INITIAL_LOG_LOSS_SCALE = 20.0

def visualize(img):
    _min = img.min()
    _max = img.max()
    if _max > _min:
        normalized_img = (img - _min) / (_max - _min)
    else:
        normalized_img = np.zeros_like(img)
    return normalized_img

class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        val_data=None,  # NEW: Validation dataset
        batch_size,
        in_channels,
        image_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        contr,
        save_interval,
        val_interval=100,  # NEW: Validation interval
        resume_checkpoint,
        resume_step,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        dataset='brats',
        summary_writer=None,
        mode='default',
        loss_level='image',
        sample_schedule='direct',
        diffusion_steps=1000,
        early_stopping_patience=10,  # NEW: Early stopping
    ):
        self.summary_writer = summary_writer
        self.mode = mode
        self.model = model
        self.diffusion = diffusion
        self.datal = data
        self.val_data = val_data  # NEW: Validation data
        self.dataset = dataset
        self.iterdatal = iter(data)
        self.batch_size = batch_size
        self.in_channels = in_channels
        self.image_size = image_size
        self.contr = contr
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.val_interval = val_interval  # NEW
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        if self.use_fp16:
            self.grad_scaler = amp.GradScaler()
        else:
            self.grad_scaler = amp.GradScaler(enabled=False)
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        self.dwt = DWT_3D('haar')
        self.idwt = IDWT_3D('haar')
        self.loss_level = loss_level
        self.step = 1
        self.resume_step = resume_step
        self.global_batch = self.batch_size * dist.get_world_size()
        self.sync_cuda = th.cuda.is_available()
        self.sample_schedule = sample_schedule
        self.diffusion_steps = diffusion_steps
        self.early_stopping_patience = early_stopping_patience
        
        # NEW: Enhanced validation tracking
        self.best_val_ssim = -np.inf
        self.steps_without_improvement = 0
        self.best_checkpoint_path = None
        self.checkpoint_dir = os.path.join(get_blob_logdir(), 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # NEW: Metrics calculator for validation
        if self.val_data is not None:
            self.metrics_calculator = ComprehensiveMetrics(dist_util.dev())
            print(f"🧠 Validation enabled: {len(self.val_data)} cases, interval={self.val_interval}")
        else:
            self.metrics_calculator = None
            print("⚠️  No validation data provided")
        
        # Load existing best SSIM if resuming
        self._load_best_metrics()
        
        self._load_and_sync_parameters()
        self.opt = AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if self.resume_step:
            print("Resume Step: " + str(self.resume_step))
            self._load_optimizer_state()
        if not th.cuda.is_available():
            logger.warn(
                "Training requires CUDA. "
            )

    def _load_best_metrics(self):
        """Load best validation metrics from file if it exists"""
        best_metrics_file = os.path.join(self.checkpoint_dir, 'best_val_metrics.txt')
        if os.path.exists(best_metrics_file):
            try:
                with open(best_metrics_file, 'r') as f:
                    for line in f:
                        if line.strip() and 'best_val_ssim:' in line:
                            self.best_val_ssim = float(line.strip().split(':')[1])
                print(f"Loaded best validation SSIM: {self.best_val_ssim:.4f}")
            except Exception as e:
                print(f"Error loading best metrics: {e}")
                self.best_val_ssim = -np.inf

    def _save_best_metrics(self):
        """Save best validation metrics to file"""
        best_metrics_file = os.path.join(self.checkpoint_dir, 'best_val_metrics.txt')
        try:
            with open(best_metrics_file, 'w') as f:
                f.write(f"best_val_ssim:{self.best_val_ssim}\n")
                f.write(f"modality:{self.contr}\n")
                f.write(f"step:{self.step + self.resume_step}\n")
        except Exception as e:
            print(f"Error saving best metrics: {e}")

    def run_validation(self):
        """
        Run validation synthesis and calculate SSIM metrics.
        Returns average validation SSIM.
        """
        if self.val_data is None or self.metrics_calculator is None:
            return None
        
        print(f"\n🧪 Running validation at step {self.step + self.resume_step}...")
        self.model.eval()
        
        val_metrics = []
        val_start_time = time.time()
        
        # Sample a few validation cases for efficiency
        max_val_cases = min(10, len(self.val_data))  # Limit for speed
        val_indices = np.random.choice(len(self.val_data), max_val_cases, replace=False)
        
        with th.no_grad():
            for i, val_idx in enumerate(val_indices):
                try:
                    batch = self.val_data[val_idx]
                    
                    # Move to device
                    batch['t1n'] = batch['t1n'].to(dist_util.dev()).unsqueeze(0)  # Add batch dim
                    batch['t1c'] = batch['t1c'].to(dist_util.dev()).unsqueeze(0)
                    batch['t2w'] = batch['t2w'].to(dist_util.dev()).unsqueeze(0)
                    batch['t2f'] = batch['t2f'].to(dist_util.dev()).unsqueeze(0)
                    
                    # Select target and conditioning modalities based on self.contr
                    if self.contr == 't1n':
                        target = batch['t1n'][0]  # Remove batch dim for metrics
                        available_modalities = {'t1c': batch['t1c'], 't2w': batch['t2w'], 't2f': batch['t2f']}
                    elif self.contr == 't1c':
                        target = batch['t1c'][0]
                        available_modalities = {'t1n': batch['t1n'], 't2w': batch['t2w'], 't2f': batch['t2f']}
                    elif self.contr == 't2w':
                        target = batch['t2w'][0]
                        available_modalities = {'t1n': batch['t1n'], 't1c': batch['t1c'], 't2f': batch['t2f']}
                    elif self.contr == 't2f':
                        target = batch['t2f'][0]
                        available_modalities = {'t1n': batch['t1n'], 't1c': batch['t1c'], 't2w': batch['t2w']}
                    else:
                        continue
                    
                    # Synthesize using shared function
                    synthesized, metrics = synthesize_modality_shared(
                        self.model, self.diffusion, available_modalities, self.contr,
                        dist_util.dev(), self.metrics_calculator, target
                    )
                    
                    if metrics and 'ssim' in metrics:
                        val_metrics.append(metrics['ssim'])
                        print(f"  Val case {i+1}/{max_val_cases}: SSIM={metrics['ssim']:.4f}")
                    
                except Exception as e:
                    print(f"  Error in validation case {i+1}: {e}")
                    continue
        
        self.model.train()  # Back to training mode
        
        val_end_time = time.time()
        val_duration = val_end_time - val_start_time
        
        if val_metrics:
            avg_val_ssim = np.mean(val_metrics)
            std_val_ssim = np.std(val_metrics)
            print(f"🧠 Validation SSIM: {avg_val_ssim:.4f} ± {std_val_ssim:.4f} "
                  f"({len(val_metrics)} cases, {val_duration:.1f}s)")
            
            # Log to wandb and tensorboard
            wandb_log_dict = {
                'val/ssim_mean': avg_val_ssim,
                'val/ssim_std': std_val_ssim,
                'val/num_cases': len(val_metrics),
                'val/duration': val_duration,
                'step': self.step + self.resume_step
            }
            wandb.log(wandb_log_dict, step=self.step + self.resume_step)
            
            if self.summary_writer is not None:
                self.summary_writer.add_scalar('val/ssim_mean', avg_val_ssim, 
                                               global_step=self.step + self.resume_step)
                self.summary_writer.add_scalar('val/ssim_std', std_val_ssim,
                                               global_step=self.step + self.resume_step)
            
            return avg_val_ssim
        else:
            print("❌ No valid validation metrics calculated")
            return None

    def save_if_best_val_ssim(self, val_ssim):
        """Save checkpoint only if validation SSIM improves"""
        if val_ssim is None:
            return False
        
        is_best = val_ssim > self.best_val_ssim
        
        if is_best and dist.get_rank() == 0:
            # Update best metrics
            old_best = self.best_val_ssim
            self.best_val_ssim = val_ssim
            self.steps_without_improvement = 0
            
            print(f"🎯 NEW BEST VAL SSIM for {self.contr}! {old_best:.4f} → {val_ssim:.4f}")
            
            # Remove old best checkpoint if it exists
            if self.best_checkpoint_path and os.path.exists(self.best_checkpoint_path):
                try:
                    os.remove(self.best_checkpoint_path)
                    print(f"Removed old checkpoint: {self.best_checkpoint_path}")
                except Exception as e:
                    print(f"Error removing old checkpoint: {e}")
            
            # Save new best checkpoint
            filename = f"brats_{self.contr}_BEST_val_ssim_{val_ssim:.4f}_step_{(self.step+self.resume_step):06d}_{self.sample_schedule}_{self.diffusion_steps}.pt"
            self.best_checkpoint_path = os.path.join(self.checkpoint_dir, filename)
            
            try:
                with bf.BlobFile(self.best_checkpoint_path, "wb") as f:
                    th.save(self.model.state_dict(), f)
                
                print(f"✅ Saved new best checkpoint: {self.best_checkpoint_path}")
                
                # Save best metrics to file
                self._save_best_metrics()
                
                # Save optimizer state for best model
                opt_save_path = os.path.join(self.checkpoint_dir, f"opt_best_{self.contr}_val_ssim.pt")
                with bf.BlobFile(opt_save_path, "wb") as f:
                    th.save(self.opt.state_dict(), f)
                print(f"💾 Saved optimizer state: {opt_save_path}")
                
            except Exception as e:
                print(f"❌ Error saving checkpoint: {e}")
                return False
            
            return True
        else:
            if not is_best:
                self.steps_without_improvement += 1
                print(f"Val SSIM {val_ssim:.4f} not better than best {self.best_val_ssim:.4f} "
                      f"({self.steps_without_improvement}/{self.early_stopping_patience} patience)")
            return False

    def should_early_stop(self):
        """Check if early stopping criteria is met"""
        return self.steps_without_improvement >= self.early_stopping_patience

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            print('resume model ...')
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                self.model.load_state_dict(
                    dist_util.load_state_dict(
                        resume_checkpoint, map_location=dist_util.dev()
                    )
                )

        dist_util.sync_params(self.model.parameters())

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)
        else:
            print('no optimizer checkpoint exists')

    def run_loop(self):
        import time
        total_data_time = 0.0
        total_step_time = 0.0
        total_log_time = 0.0
        total_save_time = 0.0
        total_val_time = 0.0
        start_time = time.time()
        t = time.time()
        
        while not self.lr_anneal_steps or self.step + self.resume_step < self.lr_anneal_steps:
            # Check early stopping
            if self.should_early_stop():
                print(f"\n🛑 Early stopping triggered after {self.steps_without_improvement} steps without improvement")
                print(f"Best validation SSIM: {self.best_val_ssim:.4f}")
                break
            
            t_total = time.time() - t
            t = time.time()
            
            # --- Data loading ---
            data_load_start = time.time()
            if self.dataset in ['brats']:
                try:
                    batch = next(self.iterdatal)
                    cond = {}
                except StopIteration:
                    self.iterdatal = iter(self.datal)
                    batch = next(self.iterdatal)
                    cond = {}
            data_load_end = time.time()
            total_data_time += data_load_end - data_load_start

            # --- Move to device ---
            if self.mode=='i2i':
                batch['t1n'] = batch['t1n'].to(dist_util.dev())
                batch['t1c'] = batch['t1c'].to(dist_util.dev())
                batch['t2w'] = batch['t2w'].to(dist_util.dev())
                batch['t2f'] = batch['t2f'].to(dist_util.dev())
            else:
                batch = batch.to(dist_util.dev())

            # --- Model forward/backward ---
            step_proc_start = time.time()
            lossmse, sample, sample_idwt = self.run_step(batch, cond)
            step_proc_end = time.time()
            total_step_time += step_proc_end - step_proc_start

            names = ["LLL", "LLH", "LHL", "LHH", "HLL", "HLH", "HHL", "HHH"]

            # --- Logging ---
            log_start = time.time()
            if self.summary_writer is not None:
                self.summary_writer.add_scalar('time/load', total_data_time, global_step=self.step + self.resume_step)
                self.summary_writer.add_scalar('time/forward', total_step_time, global_step=self.step + self.resume_step)
                self.summary_writer.add_scalar('time/total', t_total, global_step=self.step + self.resume_step)
                self.summary_writer.add_scalar('loss/MSE', lossmse.item(), global_step=self.step + self.resume_step)

            wandb_log_dict = {
                'time/load': total_data_time,
                'time/forward': total_step_time,
                'time/total': t_total,
                'loss/MSE': lossmse.item(),
                'train/best_val_ssim': self.best_val_ssim,
                'train/steps_without_improvement': self.steps_without_improvement,
                'step': self.step + self.resume_step
            }

            if self.step % 200 == 0:
                image_size = sample_idwt.size()[2]
                midplane = sample_idwt[0, 0, :, :, image_size // 2]
                if self.summary_writer is not None:
                    self.summary_writer.add_image('sample/x_0', midplane.unsqueeze(0),
                                                  global_step=self.step + self.resume_step)
                img = (visualize(midplane.detach().cpu().numpy()) * 255).astype('uint8')
                wandb_log_dict['sample/x_0'] = wandb.Image(img, caption='sample/x_0')

                image_size = sample.size()[2]
                for ch in range(8):
                    midplane = sample[0, ch, :, :, image_size // 2]
                    if self.summary_writer is not None:
                        self.summary_writer.add_image('sample/{}'.format(names[ch]), midplane.unsqueeze(0),
                                                      global_step=self.step + self.resume_step)
                    img = (visualize(midplane.detach().cpu().numpy()) * 255).astype('uint8')
                    wandb_log_dict[f'sample/{names[ch]}'] = wandb.Image(img, caption=f'sample/{names[ch]}')

                if self.mode == 'i2i':
                    if not self.contr == 't1n':
                        image_size = batch['t1n'].size()[2]
                        midplane = batch['t1n'][0, 0, :, :, image_size // 2]
                        if self.summary_writer is not None:
                            self.summary_writer.add_image('source/t1n', midplane.unsqueeze(0),
                                                          global_step=self.step + self.resume_step)
                        img = (visualize(midplane.detach().cpu().numpy()) * 255).astype('uint8')
                        wandb_log_dict['source/t1n'] = wandb.Image(img, caption='source/t1n')
                    if not self.contr == 't1c':
                        image_size = batch['t1c'].size()[2]
                        midplane = batch['t1c'][0, 0, :, :, image_size // 2]
                        if self.summary_writer is not None:
                            self.summary_writer.add_image('source/t1c', midplane.unsqueeze(0),
                                                          global_step=self.step + self.resume_step)
                        img = (visualize(midplane.detach().cpu().numpy()) * 255).astype('uint8')
                        wandb_log_dict['source/t1c'] = wandb.Image(img, caption='source/t1c')
                    if not self.contr == 't2w':
                        midplane = batch['t2w'][0, 0, :, :, image_size // 2]
                        if self.summary_writer is not None:
                            self.summary_writer.add_image('source/t2w', midplane.unsqueeze(0),
                                                          global_step=self.step + self.resume_step)
                        img = (visualize(midplane.detach().cpu().numpy()) * 255).astype('uint8')
                        wandb_log_dict['source/t2w'] = wandb.Image(img, caption='source/t2w')
                    if not self.contr == 't2f':
                        midplane = batch['t2f'][0, 0, :, :, image_size // 2]
                        if self.summary_writer is not None:
                            self.summary_writer.add_image('source/t2f', midplane.unsqueeze(0),
                                                          global_step=self.step + self.resume_step)
                        img = (visualize(midplane.detach().cpu().numpy()) * 255).astype('uint8')
                        wandb_log_dict['source/t2f'] = wandb.Image(img, caption='source/t2f')

            wandb.log(wandb_log_dict, step=self.step + self.resume_step)
            log_end = time.time()
            total_log_time += log_end - log_start

            if self.step % self.log_interval == 0:
                logger.dumpkvs()

            # --- NEW: Validation step ---
            if self.step % self.val_interval == 0 and self.val_data is not None:
                val_start = time.time()
                val_ssim = self.run_validation()
                self.save_if_best_val_ssim(val_ssim)
                val_end = time.time()
                total_val_time += val_end - val_start

            # --- Saving (regular intervals for safety) ---
            if self.step % self.save_interval == 0:
                save_start = time.time()
                self.save_regular_checkpoint()
                save_end = time.time()
                total_save_time += save_end - save_start
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1

            # Print profiling info every log_interval
            if self.step % self.log_interval == 0:
                elapsed = time.time() - start_time
                print(f"[PROFILE] Step {self.step}: Data {total_data_time:.2f}s, Step {total_step_time:.2f}s, "
                      f"Log {total_log_time:.2f}s, Val {total_val_time:.2f}s, Save {total_save_time:.2f}s, Total {elapsed:.2f}s")
                # Reset counters for next interval
                total_data_time = 0.0
                total_step_time = 0.0
                total_log_time = 0.0
                total_save_time = 0.0
                total_val_time = 0.0

        # Final validation and save
        if self.val_data is not None:
            final_val_ssim = self.run_validation()
            self.save_if_best_val_ssim(final_val_ssim)
            print(f"\n🎯 FINAL RESULTS:")
            print(f"Best validation SSIM for {self.contr}: {self.best_val_ssim:.4f}")
            if self.best_checkpoint_path:
                print(f"Best model saved to: {self.best_checkpoint_path}")

    def save_regular_checkpoint(self):
        """Save regular checkpoint for safety (not necessarily best)"""
        if dist.get_rank() == 0:
            filename = f"brats_{self.contr}_step_{(self.step+self.resume_step):06d}_{self.sample_schedule}_{self.diffusion_steps}.pt"
            full_save_path = os.path.join(self.checkpoint_dir, filename)
            
            try:
                with bf.BlobFile(full_save_path, "wb") as f:
                    th.save(self.model.state_dict(), f)
                print(f"💾 Saved regular checkpoint: {filename}")
            except Exception as e:
                print(f"❌ Error saving regular checkpoint: {e}")

    def run_step(self, batch, cond, label=None, info=dict()):
        lossmse, sample, sample_idwt = self.forward_backward(batch, cond, label)

        if self.use_fp16:
            self.grad_scaler.unscale_(self.opt)

        # compute norms
        with th.no_grad():
            param_max_norm = max([p.abs().max().item() for p in self.model.parameters()])
            grad_max_norm = max([p.grad.abs().max().item() for p in self.model.parameters()])
            info['norm/param_max'] = param_max_norm
            info['norm/grad_max'] = grad_max_norm

        if not th.isfinite(lossmse):
            if not th.isfinite(th.tensor(param_max_norm)):
                logger.log(f"Model parameters contain non-finite value {param_max_norm}, entering breakpoint", level=logger.ERROR)
                breakpoint()
            else:
                logger.log(f"Model parameters are finite, but loss is not: {lossmse}"
                           "\n -> update will be skipped in grad_scaler.step()", level=logger.WARN)

        if self.use_fp16:
            self.grad_scaler.step(self.opt)
            self.grad_scaler.update()
            info['scale'] = self.grad_scaler.get_scale()
        else:
            self.opt.step()
        self._anneal_lr()
        self.log_step()
        return lossmse, sample, sample_idwt

    def forward_backward(self, batch, cond, label=None):
        for p in self.model.parameters():
            p.grad = None

        if self.mode == 'i2i':
            batch_size = batch['t1n'].shape[0]
        else:
            batch_size = batch.shape[0]

        t, weights = self.schedule_sampler.sample(batch_size, dist_util.dev())

        compute_losses = functools.partial(
            self.diffusion.training_losses,
            self.model,
            x_start=batch,
            t=t,
            model_kwargs=cond,
            labels=label,
            mode=self.mode,
            contr=self.contr
        )
        losses1 = compute_losses()

        if isinstance(self.schedule_sampler, LossAwareSampler):
            self.schedule_sampler.update_with_local_losses(
                t, losses1["loss"].detach())

        losses = losses1[0]
        sample = losses1[1]
        sample_idwt = losses1[2]

        # Log wavelet level loss
        if self.summary_writer is not None:
            self.summary_writer.add_scalar('loss/mse_wav_lll', losses["mse_wav"][0].item(),
                                           global_step=self.step + self.resume_step)
            self.summary_writer.add_scalar('loss/mse_wav_llh', losses["mse_wav"][1].item(),
                                           global_step=self.step + self.resume_step)
            self.summary_writer.add_scalar('loss/mse_wav_lhl', losses["mse_wav"][2].item(),
                                           global_step=self.step + self.resume_step)
            self.summary_writer.add_scalar('loss/mse_wav_lhh', losses["mse_wav"][3].item(),
                                           global_step=self.step + self.resume_step)
            self.summary_writer.add_scalar('loss/mse_wav_hll', losses["mse_wav"][4].item(),
                                           global_step=self.step + self.resume_step)
            self.summary_writer.add_scalar('loss/mse_wav_hlh', losses["mse_wav"][5].item(),
                                           global_step=self.step + self.resume_step)
            self.summary_writer.add_scalar('loss/mse_wav_hhl', losses["mse_wav"][6].item(),
                                           global_step=self.step + self.resume_step)
            self.summary_writer.add_scalar('loss/mse_wav_hhh', losses["mse_wav"][7].item(),
                                           global_step=self.step + self.resume_step)

        weights = th.ones(len(losses["mse_wav"])).cuda()

        loss = (losses["mse_wav"] * weights).mean()
        lossmse = loss.detach()

        log_loss_dict(self.diffusion, t, {k: v * weights for k, v in losses.items()})

        if not th.isfinite(loss):
            logger.log(f"Encountered non-finite loss {loss}")
        if self.use_fp16:
            self.grad_scaler.scale(loss).backward()
        else:
            loss.backward()

        return lossmse.detach(), sample, sample_idwt

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self):
        """Legacy save method - prints deprecation warning"""
        print("⚠️  Warning: Using legacy save(). The enhanced training uses validation-based saving.")
        self.save_regular_checkpoint()


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = os.path.basename(filename)
    split = split.split(".")[-2]
    split = split.split("_")[-1]
    reversed_split = []
    for c in reversed(split):
        if not c.isdigit():
            break
        reversed_split.append(c)
    split = ''.join(reversed(reversed_split))
    split = ''.join(c for c in split if c.isdigit())
    try:
        return int(split)
    except ValueError:
        return 0


def get_blob_logdir():
    """
    Modified to save checkpoints to /data/ directory where persistent volume is mounted
    """
    return "/data"


def find_resume_checkpoint():
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)