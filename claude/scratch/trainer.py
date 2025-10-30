import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm
import logging
from config import Config, Factory
from customs import EarlyStopping, PhysicsLossFn
import numpy as np

#Abstract class for physics loss, the physics loss should have format: phy_loss(u, x), u for output, x for grid position


# ============================================================================
# FNO-EBM Combined Model
# ============================================================================

class FNO_EBM(nn.Module):
    """
    Combined FNO-EBM model
    Total Energy: E(u, X) = 0.5 * ||u - u_FNO(X)||^2 + V(u, X)

    This wrapper class combines an FNO model (for predictions) and an EBM model
    (for uncertainty quantification).
    """
    def __init__(self, fno_model, ebm_model):
        super().__init__()
        self.u_fno = fno_model
        self.V_ebm = ebm_model

    def energy(self, u, x, training=False, use_anchor_in_training=False, u_fno=None, sigma_squared_train=None, sigma_squared_inference=None):
        """
        Compute total energy E(u, X)

        Supports two modes:
        - Mode 1 (use_anchor_in_training=False):
            Training: E = V_EBM(u, x) only
            Inference: E = ||u - u_FNO||²/(2*σ²) + V_EBM(u, x)
        - Mode 2 (use_anchor_in_training=True):
            Training: E = ||u - u_FNO||²/(2*σ²_train) + V_EBM(u, x)
            Inference: E = ||u - u_FNO||²/(2*σ²_inference) + V_EBM(u, x)

        Args:
            u: candidate solution (batch, n_x, n_y, 1)
            x: input coordinates (batch, n_x, n_y, 3)
            training: if True, we're in training mode
            use_anchor_in_training: if True, use FNO anchor even during training
            u_fno: pre-computed FNO solution (optional)
            sigma_squared_train: variance for training anchor (default: 100.0, weak anchor)
            sigma_squared_inference: variance for inference anchor (default: 1.0, strong anchor)
        Returns:
            E: total energy (batch,)
        """
        # Potential term: learned energy function
        potential_term = self.V_ebm(u, x)

        # Decide whether to use anchor
        use_anchor = (not training) or (training and use_anchor_in_training)

        if not use_anchor:
            # Mode 1 Training: V_EBM only (standard EBM training)
            return potential_term
        else:
            # Mode 1 Inference OR Mode 2 (Training/Inference with anchor)
            if u_fno is None:
                with torch.no_grad():
                    u_fno = self.u_fno(x)

            # Quadratic term: anchors to FNO solution
            # sigma_squared controls how tightly samples are pulled toward FNO
            if training:
                # Training with anchor: use weak anchor by default
                sigma_squared = sigma_squared_train if sigma_squared_train is not None else 100.0
            else:
                # Inference: use strong anchor by default
                sigma_squared = sigma_squared_inference if sigma_squared_inference is not None else 1.0

            quadratic_term = 0.5 / sigma_squared * torch.mean((u - u_fno)**2, dim=[1, 2, 3])

            #print(f"quadratic_term: {quadratic_term.mean().item()}, potential_term: {potential_term.mean().item()}")
            return quadratic_term + potential_term

    def forward(self, x):
        """Direct FNO prediction"""
        return self.u_fno(x)


# ============================================================================
# Trainer Class
# ============================================================================

class Trainer:
    def __init__(self, model: FNO_EBM, phy_loss: PhysicsLossFn,
                 train_loader, val_loader, config: Config,
                 ebm_train_loader=None, ebm_val_loader=None):
        """
        Initialize trainer with models, data loaders and configuration.

        Supports two training modes:
        1. Single-dataset mode (train_loader only):
           - Both FNO and EBM train on same noisy data
           - Physics loss should be disabled (lambda_phys=0)

        2. Dual-dataset mode (provide ebm_*_loader):
           - FNO trains on clean data (train_loader) with physics loss
           - EBM trains on noisy data (ebm_*_loader) for uncertainty
           - Physics loss can be enabled (lambda_phys>0)

        Args:
            model: FNO_EBM combined model
            phy_loss: Physics loss function
            train_loader: Dataloader for FNO training
            val_loader: Dataloader for FNO validation
            config: Configuration object
            ebm_train_loader: Optional separate dataloader for EBM training (noisy data)
            ebm_val_loader: Optional separate dataloader for EBM validation (noisy data)
        """
        self.model = model  # Store full FNO_EBM model for energy() method
        self.fno_model = model.u_fno
        self.ebm_model = model.V_ebm

        # FNO dataloaders (clean data in dual-dataset mode)
        self.fno_train_loader = train_loader
        self.fno_val_loader = val_loader

        # EBM dataloaders (noisy data in dual-dataset mode)
        # If not provided, use same as FNO (single-dataset mode)
        self.ebm_train_loader = ebm_train_loader if ebm_train_loader is not None else train_loader
        self.ebm_val_loader = ebm_val_loader if ebm_val_loader is not None else val_loader

        # Detect training mode
        self.dual_dataset_mode = (ebm_train_loader is not None)

        self.config = config
        self.accumulation_steps = getattr(config, 'accumulation_steps', 1)

        # Setup optimizers using Factory pattern
        # Check if optimizer configs exist, otherwise fall back to legacy learning rate configs
        if hasattr(config, 'fno_optimizer_config') and config.fno_optimizer_config:
            self.fno_optimizer = Factory.create_optimizer(
                config.fno_optimizer_config,
                self.fno_model.parameters()
            )
        else:
            # Backward compatibility: use legacy fno_learning_rate
            self.fno_optimizer = optim.Adam(
                self.fno_model.parameters(),
                lr=config.fno_learning_rate
            )

        if hasattr(config, 'ebm_optimizer_config') and config.ebm_optimizer_config:
            self.ebm_optimizer = Factory.create_optimizer(
                config.ebm_optimizer_config,
                self.ebm_model.parameters()
            )
        else:
            # Backward compatibility: use legacy ebm_learning_rate
            self.ebm_optimizer = optim.Adam(
                self.ebm_model.parameters(),
                lr=config.ebm_learning_rate
            )

        # Setup schedulers using Factory pattern
        self.fno_scheduler = None
        if hasattr(config, 'fno_scheduler_config') and config.fno_scheduler_config:
            self.fno_scheduler = Factory.create_scheduler(
                config.fno_scheduler_config,
                self.fno_optimizer
            )

        self.ebm_scheduler = None
        if hasattr(config, 'ebm_scheduler_config') and config.ebm_scheduler_config:
            self.ebm_scheduler = Factory.create_scheduler(
                config.ebm_scheduler_config,
                self.ebm_optimizer
            )
        
        # Setup loss functions
        self.fno_criterion = nn.MSELoss() #Maybe I should change this to a function of PDE residual 
        self.phy_loss_fn = phy_loss

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.lambda_phys = config.lambda_phys  # Weight for physics loss
        
        self.early_stopper_fno = EarlyStopping(
            patience=config.patience, 
            verbose=True
        )
        self.early_stopper_ebm = EarlyStopping(
            patience=config.patience,
            verbose=True
        )

        # Persistent Contrastive Divergence (PCD) replay buffer
        self.use_pcd = getattr(config, 'use_pcd', True)  # Enable PCD by default
        self.pcd_buffer_size = getattr(config, 'pcd_buffer_size', 1000)
        self.pcd_reinit_prob = getattr(config, 'pcd_reinit_prob', 0.05)  # 5% fresh samples
        self.replay_buffer = []  # Store (x, y_neg) tuples for PCD

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        if self.use_pcd:
            self.logger.info(f"PCD enabled: buffer_size={self.pcd_buffer_size}, reinit_prob={self.pcd_reinit_prob}")
        self.logger.info("Negative sampling traces will be printed to terminal during training")

    def checkpoint(self, epoch, val_loss, stage_type, is_best=False):
        """
        Save model checkpoint - only keeps best and current for each stage.

        Args:
            epoch: Current epoch number
            val_loss: Validation loss
            stage_type: 'fno' or 'ebm'
            is_best: Whether this is the best model so far
        """
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'fno_model': self.fno_model.state_dict(),
            'ebm_model': self.ebm_model.state_dict(),
            'fno_optimizer': self.fno_optimizer.state_dict(),
            'ebm_optimizer': self.ebm_optimizer.state_dict(),
            'val_loss': val_loss,
        }

        # Save current epoch checkpoint (overwrites previous current)
        current_path = os.path.join(
            self.config.checkpoint_dir,
            f'current_{stage_type}.pt'
        )
        torch.save(checkpoint, current_path)

        # Save best model if this is the best
        if is_best:
            best_path = os.path.join(
                self.config.checkpoint_dir,
                f'best_model_{stage_type}.pt'
            )
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best {stage_type.upper()} model (epoch {epoch}, val_loss={val_loss:.6f})")

    def resume(self):
        """
        Resume training from checkpoint if available
        """
        checkpoint_path = os.path.join(
            self.config.checkpoint_dir,
            'best_model.pt'
        )
        
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            self.fno_model.load_state_dict(checkpoint['fno_model'])
            self.ebm_model.load_state_dict(checkpoint['ebm_model'])
            self.fno_optimizer.load_state_dict(checkpoint['fno_optimizer'])
            self.ebm_optimizer.load_state_dict(checkpoint['ebm_optimizer'])
            self.current_epoch = checkpoint['epoch']
            self.best_val_loss = checkpoint['val_loss']
            self.logger.info(f"Resumed from epoch {self.current_epoch}")
            return True
        return False

    def train_step(self, x, y):
        """
        Single training step for both FNO and EBM
        """
        # FNO Training Step
        self.fno_optimizer.zero_grad()
        fno_output = self.fno_model(x)
        data_loss = self.fno_criterion(fno_output, y)

        x_coords = x[..., 0].requires_grad_(True)
        y_coords = x[..., 1].requires_grad_(True)
        residual = self.phy_loss_fn(fno_output, x_coords, y_coords)
        #residual = compute_pde_residual(fno_output, x_coords, y_coords)
        physics_loss = torch.mean(residual**2)

        total_loss = data_loss + self.lambda_phys * physics_loss
        total_loss.backward()
        self.fno_optimizer.step()
        
        # EBM Training Step
        self.ebm_optimizer.zero_grad()

        # Positive phase
        pos_energy = self.ebm_model(x, y)

        # Negative phase (MCMC sampling with Langevin dynamics)
        # Initialize from CORRUPTED ground truth to create diverse negative samples
        y_neg = y + 0.2 * torch.randn_like(y)  # Add significant noise to ground truth
        y_neg = y_neg.detach()
        y_neg.requires_grad = True

        noise_scale = np.sqrt(2 * self.config.mcmc_step_size)

        for _ in range(self.config.mcmc_steps):
            y_neg.requires_grad = True
            neg_energy = self.ebm_model(x, y_neg)
            neg_grad = torch.autograd.grad(
                neg_energy.sum(),
                y_neg,
                create_graph=True
            )[0]

            # Langevin update: gradient descent + noise
            with torch.no_grad():
                noise = torch.randn_like(y_neg) * noise_scale
                y_neg = y_neg - self.config.mcmc_step_size * neg_grad + noise

            y_neg = y_neg.detach()
        
        neg_energy = self.ebm_model(x, y_neg)
        ebm_loss = pos_energy.mean() - neg_energy.mean()
        ebm_loss.backward()
        self.ebm_optimizer.step()
        
        return total_loss.item(), ebm_loss.item()

    def validate_old(self):
        """
        Validation step
        """
        self.fno_model.eval()
        self.ebm_model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(self.config.device), y.to(self.config.device)
                fno_output = self.fno_model(x)
                val_loss = self.fno_criterion(fno_output, y)
                total_val_loss += val_loss.item()
                
        avg_val_loss = total_val_loss / len(self.val_loader)
        return avg_val_loss
    
    def validate(self):
        """
        Validation step including physics loss
        """
        self.fno_model.eval()
        self.ebm_model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for x, y in self.fno_val_loader:
                x, y = x.to(self.config.device), y.to(self.config.device)
                fno_output = self.fno_model(x)
                
                # Data loss
                data_loss = self.fno_criterion(fno_output, y)
                
                # Physics loss
                x_coords = x[..., 0].requires_grad_(True)
                y_coords = x[..., 1].requires_grad_(True)
                # Pass full x_grid for Darcy physics loss
                residual = self.phy_loss_fn(fno_output, x_coords, y_coords, x_grid=x)
                physics_loss = torch.mean(residual**2)
                
                # Total validation loss
                val_loss = data_loss + self.lambda_phys * physics_loss
                total_val_loss += val_loss.item()
                
        avg_val_loss = total_val_loss / len(self.fno_val_loader)
        return avg_val_loss
    
    def validate_ebm(self):
        """
        Validation step for the EBM.
        Computes -(energy gap) = pos_energy - neg_energy on validation set.

        Metric interpretation:
        - Energy gap = neg_energy - pos_energy (should be LARGE and POSITIVE)
        - Validation loss = pos - neg = -(gap) (should be NEGATIVE and become MORE negative)
        - Lower (more negative) validation loss = better separation = better model

        Example:
        - Good model: pos=1, neg=10 → gap=9 → val_loss=-9 (low, good!)
        - Bad model:  pos=5, neg=6  → gap=1 → val_loss=-1 (high, bad!)
        """
        self.ebm_model.eval()
        self.fno_model.eval()
        total_ebm_val_loss = 0

        for x, y in self.ebm_val_loader:
            x, y = x.to(self.config.device), y.to(self.config.device)

            # Pre-compute FNO predictions once
            with torch.no_grad():
                u_fno = self.fno_model(x)

            # Positive phase: energy of ground truth (no grad needed)
            with torch.no_grad():
                #pos_energy = self.model.energy(y, x, training=False, u_fno=u_fno)
                pos_energy = self.ebm_model(y,x)

            # Negative phase (MCMC sampling - needs gradients)
            # Initialize from FNO prediction for inference
            y_neg = u_fno.clone().detach()
            noise_scale = np.sqrt(2 * self.config.mcmc_step_size)

            for _ in range(self.config.mcmc_steps):
                y_neg.requires_grad_(True)
                #neg_energy_for_grad = self.model.energy(y_neg, x, training=False, u_fno=u_fno)
                neg_energy_for_grad = self.ebm_model(y_neg, x)
                neg_grad = torch.autograd.grad(neg_energy_for_grad.sum(), y_neg)[0]

                with torch.no_grad():
                    noise = torch.randn_like(y_neg) * noise_scale
                    y_neg = y_neg - self.config.mcmc_step_size * neg_grad + noise
                    y_neg = y_neg.detach()

            with torch.no_grad():
                #neg_energy = self.model.energy(y_neg, x, training=False, u_fno=u_fno)
                neg_energy = self.ebm_model(y_neg, x)
                ebm_loss = pos_energy.mean() - neg_energy.mean()
                total_ebm_val_loss += ebm_loss.item()

        avg_ebm_val_loss = total_ebm_val_loss / len(self.ebm_val_loader)
        return avg_ebm_val_loss

    def train_no_stage(self):
        """
        Main training loop
        """
        # Try to resume training
        self.resume()
        
        for epoch in range(self.current_epoch, self.config.epochs):
            self.fno_model.train()
            self.ebm_model.train()
            
            # Training loop with progress bar
            loop = tqdm(self.train_loader)
            epoch_fno_loss = 0
            epoch_ebm_loss = 0
            
            for batch_idx, (x, y) in enumerate(loop):
                x, y = x.to(self.config.device), y.to(self.config.device)
                fno_loss, ebm_loss = self.train_step(x, y)
                
                epoch_fno_loss += fno_loss
                epoch_ebm_loss += ebm_loss
                
                # Update progress bar
                loop.set_description(f"Epoch [{epoch}/{self.config.epochs}]")
                loop.set_postfix(
                    fno_loss=fno_loss,
                    ebm_loss=ebm_loss
                )
            
            # Validation phase
            val_loss = self.validate()

            if self.fno_scheduler:
                self.fno_scheduler.step()
            if self.ebm_scheduler:
                self.ebm_scheduler.step()
            
            # Log metrics
            self.logger.info(
                f"Epoch {epoch}: FNO Loss={epoch_fno_loss/len(self.train_loader):.4f}, "
                f"EBM Loss={epoch_ebm_loss/len(self.train_loader):.4f}, "
                f"Val Loss={val_loss:.4f}"
            )
            
            # Save checkpoint
            self.checkpoint(epoch, val_loss)
            self.current_epoch = epoch + 1
    
    # ...existing code...

    def train_step_fno(self, x, y):
        """
        Single training step for FNO with both data and physics losses

        Returns:
            tuple: (total_loss, data_loss, physics_loss) as floats
        """
        #self.fno_optimizer.zero_grad()

        # Forward pass
        fno_output = self.fno_model(x)

        # Data loss
        data_loss = self.fno_criterion(fno_output, y)

        # Physics loss (PDE residual)
        x_coords = x[..., 0].requires_grad_(True)
        y_coords = x[..., 1].requires_grad_(True)
        # Pass full x_grid for Darcy physics loss (includes permeability in x[..., 2])
        residual = self.phy_loss_fn(fno_output, x_coords, y_coords, x_grid=x)
        physics_loss = torch.mean(residual**2)

        # Total loss
        total_loss = data_loss + self.lambda_phys * physics_loss
        total_loss_scaled = total_loss / self.accumulation_steps

        total_loss_scaled.backward()
        #self.fno_optimizer.step()

        return total_loss.item(), data_loss.item(), physics_loss.item()

    def train_step_ebm(self, x, y, langevin=True, use_anchor=None, trace_samples=False):
        """
        FIXED EBM training - Following UvA Deep Energy Models Tutorial

        CRITICAL FIXES from UvA tutorial:
        1. NO create_graph=True during MCMC (was killing training!)
        2. Random uniform initialization instead of corrupted data
        3. Small noise added to positive samples
        4. 60 MCMC steps instead of 200

        Args:
            trace_samples: If True, collect detailed tracing information
        """
        batch_size = x.shape[0]

        # STEP 1: Add small noise to positive samples (UvA trick!)
        # Prevents overfitting to exact training data
        small_noise = torch.randn_like(y) * 0.005
        y_noisy = y + small_noise
        y_noisy = torch.clamp(y_noisy, -3, 3)

        # STEP 2: Compute positive energy (V_EBM only, no anchor during training)
        pos_energy = -self.ebm_model(y_noisy, x)  # Fixed: y_noisy is u (solution), x is conditioning
        #print(f"  Pos Energy (mean): {pos_energy.mean().item():.4f}")
        # STEP 3: Initialize negative samples from buffer (PCD)
        if self.use_pcd and len(self.replay_buffer) > 0 and np.random.random() > self.pcd_reinit_prob:
            # Sample from buffer (95% of time)
            buffer_indices = np.random.choice(
                len(self.replay_buffer),
                size=min(batch_size, len(self.replay_buffer)),
                replace=False
            )
            y_neg = torch.stack([self.replay_buffer[i][1] for i in buffer_indices]).to(x.device)

            # Pad if needed with random uniform (UvA style)
            if y_neg.shape[0] < batch_size:
                n_missing = batch_size - y_neg.shape[0]
                y_neg_extra = torch.rand(n_missing, *y.shape[1:], device=x.device) * 2 - 1
                y_neg = torch.cat([y_neg, y_neg_extra], dim=0)
            init_source = "buffer"
        else:
            # Fresh initialization (5% of time): random uniform [-1, 1] (UvA style)
            y_neg = torch.rand_like(y) * 2 - 1
            init_source = "random"

        # Store initial state for tracing
        if trace_samples:
            y_neg_init = y_neg.clone().detach()
            energy_trajectory = []
            grad_norm_trajectory = []

        # STEP 4: MCMC Sampling - CRITICAL FIX: NO create_graph!
        mcmc_steps = 60 #60  # UvA uses 60 during training (not 200!)
        step_size = 0.001 #self.config.mcmc_step_size  #0.001)  # Cap step size to prevent explosion
        grad_clip =  1.0 #0.1  # Reduced gradient clipping
        noise_scale = np.sqrt(2 * step_size)  # Proper Langevin noise scale
        
        # Set EBM to eval mode during sampling to avoid batch norm issues
        self.ebm_model.eval()
        
        y_neg = y_neg.detach()  # Start fresh, no gradients

        for i in range(mcmc_steps):
            # Clamp to reasonable bounds
            y_neg = torch.clamp(y_neg, -3, 3)
            
            # Enable gradients for this step only
            y_neg.requires_grad_(True)

            # Compute energy
            neg_energy_for_grad = -self.ebm_model(y_neg, x)

            # CRITICAL FIX: NO create_graph=True!
            # Only backprop to get gradients, don't keep computation graph
            neg_grad = torch.autograd.grad(
                neg_energy_for_grad.sum(),
                y_neg,
                retain_graph=False,
                create_graph=False  # FIXED: Was True, killed training!
            )[0]

            # Track gradient norm before clipping
            if trace_samples and i % 10 == 0:
                with torch.no_grad():
                    grad_norm_trajectory.append(neg_grad.norm().item())
                    energy_trajectory.append(neg_energy_for_grad.mean().item())

            # Clip gradients to prevent explosion
            grad_norm_before_clip = neg_grad.norm().item()
            neg_grad = torch.clamp(neg_grad, -grad_clip, grad_clip)
            grad_norm_after_clip = neg_grad.norm().item()
            
            # Debug print for first few steps
            if trace_samples and i < 5:
                print(f"    Step {i:2d}: Energy={neg_energy_for_grad.mean().item():.4f}, "
                      f"GradNorm={grad_norm_before_clip:.4f}->{grad_norm_after_clip:.4f}")

            # LANGEVIN UPDATE: Move against energy gradient + noise
            with torch.no_grad():
                if langevin:
                    noise = torch.randn_like(y_neg) * noise_scale
                    # CRITICAL: Move AGAINST energy gradient (minus sign) to find LOW energy
                    #y_neg = y_neg + step_size * neg_grad + noise
                    y_neg = y_neg - step_size * neg_grad + noise
                else:
                    # Gradient descent without noise
                    y_neg = y_neg - step_size * neg_grad
                    #y_neg = y_neg + step_size * neg_grad
                    
                # Optional: Add more aggressive clamping if values go extreme
                y_neg = torch.clamp(y_neg, -5, 5)

            # Detach for next iteration
            y_neg = y_neg.detach()
        
        # Set EBM back to train mode
        self.ebm_model.train()

        # STEP 5: Final negative energy
        neg_energy = -self.ebm_model(y_neg, x)
        #print(f"  Neg Energy (mean): {neg_energy.mean().item():.4f}")
        # STEP 6: Update buffer
        if self.use_pcd:
            for i in range(batch_size):
                self.replay_buffer.append((
                    x[i].detach().cpu(),
                    y_neg[i].detach().cpu()
                ))

            if len(self.replay_buffer) > self.pcd_buffer_size:
                self.replay_buffer = self.replay_buffer[-self.pcd_buffer_size:]

        # STEP 7: Compute loss (UvA formula)
        alpha = 0.08 # UvA default, set somewhere between 0.1 - 0.03
        reg_loss = alpha * (pos_energy ** 2 + neg_energy ** 2).mean()

        # Contrastive divergence, maybe pos - neg
        cd_loss = neg_energy.mean() - pos_energy.mean()
        #cd_loss = pos_energy.mean() - neg_energy.mean()  # FIXED: UvA style

        # Total loss
        ebm_loss = reg_loss + cd_loss

        ebm_loss_scaled = ebm_loss / self.accumulation_steps
        ebm_loss_scaled.backward()

        # Return tracing info if requested
        if trace_samples:
            trace_info = {
                'init_source': init_source,
                'pos_energy_mean': pos_energy.mean().item(),
                'neg_energy_mean': neg_energy.mean().item(),
                'energy_gap': (neg_energy - pos_energy).mean().item(),
                'pos_stats': {
                    'mean': y.mean().item(),
                    'std': y.std().item(),
                    'min': y.min().item(),
                    'max': y.max().item()
                },
                'neg_init_stats': {
                    'mean': y_neg_init.mean().item(),
                    'std': y_neg_init.std().item(),
                    'min': y_neg_init.min().item(),
                    'max': y_neg_init.max().item()
                },
                'neg_final_stats': {
                    'mean': y_neg.mean().item(),
                    'std': y_neg.std().item(),
                    'min': y_neg.min().item(),
                    'max': y_neg.max().item()
                },
                'energy_trajectory': energy_trajectory,
                'grad_norm_trajectory': grad_norm_trajectory
            }
            return ebm_loss.item(), trace_info

        return ebm_loss.item()

    def train_fno(self, num_epochs):
        """
        Train FNO model only.

        Uses fno_train_loader (clean data in dual-dataset mode, noisy data otherwise).
        """
        mode_str = "clean data" if self.dual_dataset_mode else "single dataset (noisy)"
        self.logger.info(f"Starting FNO training on {mode_str}...")
        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            self.fno_model.train()
            epoch_total_loss = 0
            epoch_data_loss = 0
            epoch_physics_loss = 0

            loop = tqdm(self.fno_train_loader)  # Use FNO-specific loader
            for batch_idx, (x, y) in enumerate(loop):
                x, y = x.to(self.config.device), y.to(self.config.device)
                total_loss, data_loss, physics_loss = self.train_step_fno(x, y)
                epoch_total_loss += total_loss
                epoch_data_loss += data_loss
                epoch_physics_loss += physics_loss

                if (batch_idx + 1) % self.accumulation_steps == 0:
                    self.fno_optimizer.step()  #apply gradient update
                    self.fno_optimizer.zero_grad()  #reset gradient to zero

                loop.set_description(f"FNO Training Epoch [{epoch}/{num_epochs}]")
                loop.set_postfix(
                    total=f"{total_loss:.4f}",
                    data=f"{data_loss:.4f}",
                    phys=f"{physics_loss:.4f}"
                )

            val_loss = self.validate()
            avg_total_loss = epoch_total_loss / len(self.fno_train_loader)
            avg_data_loss = epoch_data_loss / len(self.fno_train_loader)
            avg_physics_loss = epoch_physics_loss / len(self.fno_train_loader)

            self.logger.info(
                f"FNO Epoch {epoch}: "
                f"Total={avg_total_loss:.6f}, "
                f"Data={avg_data_loss:.6f}, "
                f"Physics={avg_physics_loss:.6f}, "
                f"Val={val_loss:.6f}"
            )

            if self.fno_scheduler:
                self.fno_scheduler.step()

            # Save checkpoint every epoch
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss

            self.checkpoint(epoch, val_loss, 'fno', is_best=is_best)

            self.early_stopper_fno(val_loss)
            if self.early_stopper_fno.early_stop:
                self.logger.info("EARLY STOPPING FNO TRIGGERED, NO IMPROVEMENT IN SEVERAL EPOCHS")
                break


    def train_ebm(self, num_epochs):
        """
        Train EBM model only, with checkpointing and early stopping.

        Uses ebm_train_loader (noisy data in dual-dataset mode, same as FNO otherwise).
        """
        mode_str = "noisy observations" if self.dual_dataset_mode else "same dataset as FNO"
        self.logger.info(f"Starting EBM training on {mode_str}...")
        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            self.ebm_model.train()
            epoch_loss = 0

            loop = tqdm(self.ebm_train_loader)  # Use EBM-specific loader
            for batch_idx, (x, y) in enumerate(loop):
                x, y = x.to(self.config.device), y.to(self.config.device)

                # Enable tracing for first batch of each epoch and every 50 batches
                trace_samples = (batch_idx == 0) or (batch_idx % 50 == 0)

                if trace_samples:
                    loss, trace_info = self.train_step_ebm(x, y, trace_samples=True)
                    # Print trace information to terminal
                    self._print_trace_info(epoch, batch_idx, trace_info)
                else:
                    loss = self.train_step_ebm(x, y, trace_samples=False)

                epoch_loss += loss

                if (batch_idx + 1) % self.accumulation_steps == 0:
                    self.ebm_optimizer.step()
                    self.ebm_optimizer.zero_grad()

                loop.set_description(f"EBM Training Epoch [{epoch+1}/{num_epochs}]")
                loop.set_postfix(ebm_loss=loss)

            # EBM Validation (uses EBM val loader)
            val_loss = self.validate_ebm()
            avg_epoch_loss = epoch_loss / len(self.ebm_train_loader)

            self.logger.info(
                f"EBM Epoch {epoch+1}: Train Loss={avg_epoch_loss:.4f}, "
                f"Val Loss (Energy Gap)={val_loss:.4f}"
            )

            if self.ebm_scheduler:
                self.ebm_scheduler.step()

            # Save checkpoint every epoch
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss

            self.checkpoint(epoch, val_loss, 'ebm', is_best=is_best)

            self.early_stopper_ebm(val_loss)
            if self.early_stopper_ebm.early_stop:
                self.logger.info("EARLY STOPPING TRIGGERED FOR EBM, NO IMPROVEMENT IN SEVERAL EPOCHS")
                break

    def _print_trace_info(self, epoch, batch_idx, trace_info):
        """Print negative sampling trace information to terminal"""
        self.logger.info("=" * 80)
        self.logger.info(f"NEGATIVE SAMPLING TRACE - Epoch {epoch}, Batch {batch_idx}")
        self.logger.info("=" * 80)

        # Initialization
        self.logger.info(f"Initialization Source: {trace_info['init_source']}")

        # Positive samples statistics
        self.logger.info("\nPOSITIVE SAMPLES (Ground Truth):")
        self.logger.info(f"  Mean: {trace_info['pos_stats']['mean']:8.4f}  Std: {trace_info['pos_stats']['std']:8.4f}")
        self.logger.info(f"  Min:  {trace_info['pos_stats']['min']:8.4f}  Max: {trace_info['pos_stats']['max']:8.4f}")

        # Negative samples - initial
        self.logger.info(f"\nNEGATIVE SAMPLES - Initial ({trace_info['init_source']}):")
        self.logger.info(f"  Mean: {trace_info['neg_init_stats']['mean']:8.4f}  Std: {trace_info['neg_init_stats']['std']:8.4f}")
        self.logger.info(f"  Min:  {trace_info['neg_init_stats']['min']:8.4f}  Max: {trace_info['neg_init_stats']['max']:8.4f}")

        # Negative samples - final
        self.logger.info("\nNEGATIVE SAMPLES - After MCMC:")
        self.logger.info(f"  Mean: {trace_info['neg_final_stats']['mean']:8.4f}  Std: {trace_info['neg_final_stats']['std']:8.4f}")
        self.logger.info(f"  Min:  {trace_info['neg_final_stats']['min']:8.4f}  Max: {trace_info['neg_final_stats']['max']:8.4f}")

        # Energy statistics
        self.logger.info("\nENERGY STATISTICS:")
        self.logger.info(f"  Positive Energy:  {trace_info['pos_energy_mean']:8.4f}")
        self.logger.info(f"  Negative Energy:  {trace_info['neg_energy_mean']:8.4f}")
        self.logger.info(f"  Energy Gap:       {trace_info['energy_gap']:8.4f} (should be POSITIVE!)")

        # MCMC trajectory
        if trace_info['energy_trajectory']:
            self.logger.info("\nMCMC TRAJECTORY (sampled every 10 steps):")
            self.logger.info("  Step   Energy    Grad Norm")
            for i, (energy, grad_norm) in enumerate(zip(trace_info['energy_trajectory'],
                                                         trace_info['grad_norm_trajectory'])):
                step = i * 10
                self.logger.info(f"  {step:4d}  {energy:8.4f}  {grad_norm:8.4f}")

        # Diagnosis
        self.logger.info("\nDIAGNOSIS:")
        if abs(trace_info['neg_final_stats']['mean'] - trace_info['pos_stats']['mean']) < 0.1:
            self.logger.info("  ⚠ WARNING: Negative samples too similar to positive samples (mode collapse?)")
        if trace_info['energy_gap'] < 0:
            self.logger.info("  ⚠ WARNING: Negative energy gap is NEGATIVE (should be positive!)")
        if trace_info['neg_final_stats']['std'] < 0.01:
            self.logger.info("  ⚠ WARNING: Negative samples have very low variance")
        if abs(trace_info['neg_init_stats']['mean'] - trace_info['neg_final_stats']['mean']) < 0.05:
            self.logger.info("  ⚠ WARNING: MCMC sampling didn't change samples much (gradients too small?)")

        self.logger.info("=" * 80)


    def train_staged(self):
        """
        Staged training: First FNO, then EBM
        """
        self.logger.info("Starting staged training...")
        
        # Stage 1: Train FNO
        self.logger.info("Stage 1: Training FNO")
        self.train_fno(self.config.fno_epochs)
        
        # Stage 2: Train EBM
        self.logger.info("Stage 2: Training EBM")
        self.train_ebm(self.config.ebm_epochs)
        
        self.logger.info("Staged training completed")

# ...existing code...