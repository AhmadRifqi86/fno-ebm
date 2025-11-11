import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm
import logging
from config import Config, Factory
from customs import EarlyStopping, PhysicsLossFn, combined_ebm_loss
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
        print(f"DEBUG: lambda_phys = {self.lambda_phys}")
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
        # torch.save(checkpoint, current_path)  # Commented out to save disk space

        # Save best model if this is the best
        if is_best:
            best_path = os.path.join(
                self.config.checkpoint_dir,
                f'best_model_{stage_type}.pt'
            )
            # torch.save(checkpoint, best_path)  # Commented out to save disk space
            self.logger.info(f"Best {stage_type.upper()} model checkpoint skipped (epoch {epoch}, val_loss={val_loss:.6f})")

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
        Validation step for the EBM using Score Matching.

        Computes denoising score matching loss on validation set:
        - Add noise to ground truth: y_noisy = y + σ * ε
        - Compute predicted score: s_θ = -∇_y E(y_noisy, x)
        - Target score: -ε / σ
        - Loss: ||s_θ - target_score||²

        Lower validation loss = better score prediction = better model
        """
        self.ebm_model.eval()
        self.fno_model.eval()
        total_ebm_val_loss = 0

        # Use the same noise levels as training
        sigma_levels = getattr(self.config, 'score_matching_sigmas', [0.1, 0.5, 1.0])

        for x, y in self.ebm_val_loader:
            x, y = x.to(self.config.device), y.to(self.config.device)

            batch_loss = 0.0

            # Compute score matching loss for each noise level
            for sigma in sigma_levels:
                # Add noise to clean data
                noise = torch.randn_like(y)
                y_noisy = y + sigma * noise
                y_noisy.requires_grad_(True)

                # Compute energy of noisy data
                energy = self.ebm_model(y_noisy, x)

                # Compute score: s_θ(y_noisy) = -∇_y E(y_noisy, x)
                score = -torch.autograd.grad(
                    outputs=energy.sum(),
                    inputs=y_noisy,
                    create_graph=False  # No backprop in validation
                )[0]

                # Target score (points from noisy data back to clean data)
                target_score = -noise / sigma

                # Score matching loss (MSE between predicted and target score)
                with torch.no_grad():
                    score_loss = torch.mean((score - target_score) ** 2)
                    batch_loss += score_loss.item()

            # Average over noise levels
            batch_loss = batch_loss / len(sigma_levels)
            total_ebm_val_loss += batch_loss

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
                # ReduceLROnPlateau requires metric, other schedulers don't
                if isinstance(self.fno_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.fno_scheduler.step(val_loss)
                else:
                    self.fno_scheduler.step()
            if self.ebm_scheduler:
                # ReduceLROnPlateau requires metric, other schedulers don't
                if isinstance(self.ebm_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.ebm_scheduler.step(val_loss)
                else:
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

    def train_step_ebm(self, x, y, langevin=True, trace_samples=False):
        """
        EBM training with Contrastive Divergence using Langevin MCMC sampling.

        Algorithm:
        1. Positive phase: E_pos = E(y_real, x) - energy of real data
        2. Negative phase: Sample y_neg using Langevin dynamics
           - Initialize from RANDOM noise (high energy region)
           - Update: y_t+1 = y_t - ε∇E(y_t, x) + √(2ε)z  (walks to lower energy)
        3. Loss: E_pos - E_neg (maximize energy gap)
        4. Regularization: Prevent energy values from diverging

        Args:
            x: input coordinates (batch, n_x, n_y, 3)
            y: ground truth solution (batch, n_x, n_y, 1)
            langevin: if True, use Langevin dynamics; else simple gradient descent
            trace_samples: if True, return detailed trace info for debugging

        Returns:
            loss (float) or (loss, trace_info) if trace_samples=True
        """
        # ========================================================================
        # POSITIVE PHASE: Energy of real data (ground truth)
        # ========================================================================
        y_pos = y.detach()  # No gradient needed for positive samples
        pos_energy = self.ebm_model(y_pos, x)  # (batch,)

        # ========================================================================
        # NEGATIVE PHASE: Generate samples via Langevin MCMC
        # ========================================================================
        # CRITICAL: Initialize from RANDOM noise, NOT from data
        # This ensures we start in HIGH energy regions and walk down

        # Get data statistics for proper initialization scale
        y_mean = y.mean()
        y_std = y.std()

        # Initialize from random Gaussian noise scaled to data range
        # This puts us in a high-energy region of the energy landscape
        y_neg = torch.randn_like(y) * y_std + y_mean
        y_neg = y_neg.detach()

        # Langevin dynamics parameters
        step_size = self.config.mcmc_step_size
        noise_scale = np.sqrt(2 * step_size) if langevin else 0.0

        # Storage for tracing (debugging)
        if trace_samples:
            trace_info = {
                'init_source': 'random_noise',
                'pos_stats': self._get_tensor_stats(y_pos),
                'neg_init_stats': self._get_tensor_stats(y_neg),
                'energy_trajectory': [],
                'grad_norm_trajectory': []
            }

        # Run Langevin MCMC to generate negative samples
        for step in range(self.config.mcmc_steps):
            y_neg.requires_grad_(True)

            # Compute energy from EBM model
            energy_neg = self.ebm_model(y_neg, x)  # (batch,)

            # Compute gradient of energy w.r.t. y_neg
            # We use create_graph=False in training to save memory
            # (gradient is only used for sampling, not for backprop through sampling)
            grad_energy = torch.autograd.grad(
                outputs=energy_neg.sum(),
                inputs=y_neg,
                create_graph=False  # Don't backprop through MCMC
            )[0]

            # Trace for debugging (sample every 10 steps)
            if trace_samples and step % 10 == 0:
                with torch.no_grad():
                    trace_info['energy_trajectory'].append(energy_neg.mean().item())
                    trace_info['grad_norm_trajectory'].append(grad_energy.norm().item())

            # Langevin dynamics update:
            # y_t+1 = y_t - ε * ∇E(y_t) + √(2ε) * noise
            #
            # This performs GRADIENT DESCENT on energy, moving samples toward LOWER energy
            # (lower energy = higher probability under p(y) ∝ exp(-E(y)))
            with torch.no_grad():
                # Gradient descent step: move toward lower energy
                y_neg = y_neg - step_size * grad_energy #need to check sign of y_neg and grad_energy

                # Add Brownian noise (exploration)
                if langevin:
                    noise = torch.randn_like(y_neg) * noise_scale
                    y_neg = y_neg + noise

                y_neg = y_neg.detach()

        # Final negative energy (after MCMC)
        y_neg.requires_grad_(True)
        neg_energy = self.ebm_model(y_neg, x)

        # ========================================================================
        # CONTRASTIVE DIVERGENCE LOSS
        # ========================================================================
        # Goal: minimize energy on real data (E_pos should be LOW)
        #       maximize energy on generated samples (E_neg should be HIGH)
        #
        # Standard CD loss: L = E_pos - E_neg
        #
        # When we minimize this loss:
        # - E_pos decreases (real data gets lower energy)
        # - E_neg increases (generated samples get higher energy)
        #
        # The energy gap (E_neg - E_pos) should be POSITIVE and LARGE
        cd_loss = pos_energy.mean() - neg_energy.mean()
        #cd_loss = neg_energy.mean() - pos_energy.mean()

        # ========================================================================
        # REGULARIZATION (prevents energy divergence)
        # ========================================================================
        # Add L2 regularization on energy values to prevent them from growing unbounded
        # This is critical for stable training
        alpha_reg = getattr(self.config, 'ebm_energy_reg', 0.05)
        energy_reg = alpha_reg * (pos_energy.pow(2).mean() + neg_energy.pow(2).mean())

        # Total loss
        total_loss = cd_loss + energy_reg

        # Backward pass
        total_loss_scaled = total_loss / self.accumulation_steps
        total_loss_scaled.backward()

        # ========================================================================
        # TRACING (debugging)
        # ========================================================================
        if trace_samples:
            with torch.no_grad():
                trace_info['neg_final_stats'] = self._get_tensor_stats(y_neg)
                trace_info['pos_energy_mean'] = pos_energy.mean().item()
                trace_info['neg_energy_mean'] = neg_energy.mean().item()
                trace_info['energy_gap'] = (neg_energy.mean() - pos_energy.mean()).item()
                trace_info['cd_loss'] = cd_loss.item()
                trace_info['energy_reg'] = energy_reg.item()
            return total_loss.item(), trace_info

        return total_loss.item()

    def train_step_ebm_scorematching(self, x, y, trace_score=False):
        """
        EBM training with Denoising Score Matching.

        Algorithm (Denoising Score Matching):
        1. Add noise to real data: y_noisy = y + σ * ε, where ε ~ N(0, I)
        2. Compute score (gradient of log-density): s_θ(y_noisy) = -∇_y E(y_noisy, x)
        3. Target score: s_target = -ε / σ  (points toward clean data)
        4. Loss: ||s_θ(y_noisy) - s_target||²

        This avoids MCMC sampling entirely, making training faster and more stable.
        The learned energy E(y, x) implicitly defines: p(y|x) ∝ exp(-E(y, x))

        Args:
            x: input coordinates (batch, n_x, n_y, 3)
            y: ground truth solution (batch, n_x, n_y, 1)
            trace_score: if True, return detailed trace info for debugging

        Returns:
            loss (float) or (loss, trace_info) if trace_score=True
        """
        # ========================================================================
        # DENOISING SCORE MATCHING
        # ========================================================================
        # Noise level (sigma) - can be fixed or annealed during training
        # Multiple noise levels are often used (similar to NCSN), but we start simple
        sigma_levels = getattr(self.config, 'score_matching_sigmas', [0.005, 0.02, 0.05])
        #print(f"DEBUG: y stats - mean={y.mean():.4f}, std={y.std():.4f}, min={y.min():.4f}, max={y.max():.4f}")
        total_loss = 0.0

        if trace_score:
            trace_info = {
                'sigma_levels': sigma_levels,
                'score_losses': [],
                'score_norms': [],
                'target_score_norms': []
            }

        for sigma in sigma_levels:
            # 1. Add noise to clean data
            noise = torch.randn_like(y)
            y_noisy = y + sigma * noise
            y_noisy.requires_grad_(True)

            # 2. Compute energy of noisy data
            energy = self.ebm_model(y_noisy, x)  # (batch,), should I add noise_level as parameter?

            # 3. Compute score: s_θ(y_noisy) = -∇_y E(y_noisy, x)
            # This is the gradient of log-density
            score = -torch.autograd.grad(
                outputs=energy.sum(),
                inputs=y_noisy,
                create_graph=True  # Need gradients for backprop,
            )[0]  # (batch, n_x, n_y, 1)

            # 4. Target score (points from noisy data back to clean data)
            # Derived from: ∇_y log p(y_noisy | y_clean) = -noise / σ²
            # But we use simplified version: -noise / σ (equivalent up to scaling)
            target_score = -noise / sigma

            # 5. Score matching loss (MSE between predicted and target score)
            score_loss = torch.mean((score - target_score) ** 2)

            total_loss += score_loss

            # Tracing
            if trace_score:
                with torch.no_grad():
                    trace_info['score_losses'].append(score_loss.item())
                    trace_info['score_norms'].append(score.norm().item())
                    trace_info['target_score_norms'].append(target_score.norm().item())

        # Average over noise levels
        total_loss = total_loss / len(sigma_levels)

        # ========================================================================
        # ADDITIONAL REGULARIZATION (optional)
        # ========================================================================
        # L2 regularization on energy magnitude (prevents unbounded energies)
        alpha_reg = getattr(self.config, 'ebm_energy_reg', 0.01)

        # Compute energy on clean data for regularization
        with torch.no_grad():
            y_clean = y.detach()
            y_clean.requires_grad_(True)
        energy_clean = self.ebm_model(y_clean, x)
        energy_reg = alpha_reg * energy_clean.pow(2).mean()

        # Total loss
        total_loss_with_reg = total_loss + energy_reg

        # Backward pass
        total_loss_scaled = total_loss_with_reg / self.accumulation_steps
        total_loss_scaled.backward()

        if trace_score:
            trace_info['total_score_loss'] = total_loss.item()
            trace_info['energy_reg'] = energy_reg.item()
            trace_info['total_loss'] = total_loss_with_reg.item()
            return total_loss_with_reg.item(), trace_info

        return total_loss_with_reg.item()

    def train_step_ebm_scorematching_combined(self, x, y, trace_score=False):
        """
        EBM training with Enhanced Score Matching + Error-Aware Calibration.

        Combines:
        1. Weighted multi-scale score matching (balanced learning across noise levels)
        2. Error-aware calibration loss (teaches EBM to predict high uncertainty where FNO fails)
        3. Energy regularization (prevents unbounded energies)

        This produces spatially-structured uncertainty maps that correlate with FNO errors,
        instead of uniform random noise.

        Args:
            x: input coordinates (batch, n_x, n_y, 3)
            y: ground truth solution (batch, n_x, n_y, 1)
            trace_score: if True, return detailed trace info for debugging

        Returns:
            loss (float) or (loss, trace_info) if trace_score=True
        """
        # ========================================================================
        # Get FNO predictions for calibration
        # ========================================================================
        with torch.no_grad():
            fno_output = self.fno_model(x)  # (batch, n_x, n_y, 1)

        # Prepare inputs for combined_ebm_loss
        # u_clean: FNO output for score matching (we add noise to this)
        # x_coords: Input coordinates
        # fno_pred: FNO output for calibration (squeezed for error calculation)
        # ground_truth: True solution for calibration
        u_clean = fno_output  # (batch, n_x, n_y, 1)
        x_coords = x  # (batch, n_x, n_y, 3)
        fno_pred = fno_output.squeeze(-1) if fno_output.dim() == 4 else fno_output  # (batch, n_x, n_y)
        ground_truth = y.squeeze(-1) if y.dim() == 4 else y  # (batch, n_x, n_y)

        # ========================================================================
        # Get hyperparameters from config
        # ========================================================================
        sigma_levels = getattr(self.config, 'score_matching_sigmas', [0.01, 0.02, 0.05])
        weight_score = getattr(self.config, 'ebm_weight_score', 1.0)
        weight_calibration = getattr(self.config, 'ebm_weight_calibration', 0.0)
        energy_reg_weight = getattr(self.config, 'ebm_energy_reg', 0.0)

        # ========================================================================
        # COMBINED EBM LOSS (from customs.py)
        # ========================================================================
        loss, loss_dict = combined_ebm_loss(
            ebm_model=self.ebm_model,
            u_clean=u_clean,
            x_coords=x_coords,
            fno_pred=fno_pred,
            ground_truth=ground_truth,
            weight_score=weight_score,
            weight_calibration=weight_calibration,
            sigmas=sigma_levels,
            energy_reg_weight=energy_reg_weight
        )

        # Backward pass
        loss_scaled = loss / self.accumulation_steps
        loss_scaled.backward()

        # ========================================================================
        # TRACING (for debugging and monitoring)
        # ========================================================================
        if trace_score:
            # Reconstruct trace_info compatible with existing logging
            trace_info = {
                'sigma_levels': sigma_levels,
                'score_losses': [loss_dict.get(f'score_loss_{s}', 0.0) for s in sigma_levels],
                'score_norms': [0.0] * len(sigma_levels),  # Not provided by combined_ebm_loss
                'target_score_norms': [0.0] * len(sigma_levels),  # Not provided
                'total_score_loss': loss_dict['score'],
                'energy_reg': loss_dict['energy_reg'],
                'calibration_loss': loss_dict['calibration'],
                'total_loss': loss_dict['total'],
                'score_norm_ratio_0.01': loss_dict.get('score_norm_ratio_0.01', 0.0),
                'score_norm_ratio_0.05': loss_dict.get('score_norm_ratio_0.05', 0.0),
            }
            return loss.item(), trace_info

        return loss.item()

    def _get_tensor_stats(self, tensor):
        """Helper function to get statistics of a tensor for debugging"""
        with torch.no_grad():
            return {
                'mean': tensor.mean().item(),
                'std': tensor.std().item(),
                'min': tensor.min().item(),
                'max': tensor.max().item()
            }
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
                # ReduceLROnPlateau requires metric, other schedulers don't
                if isinstance(self.fno_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.fno_scheduler.step(val_loss)
                else:
                    self.fno_scheduler.step()

            # Save checkpoint every epoch
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                self.best_val_loss = val_loss

            self.checkpoint(epoch, val_loss, 'fno', is_best=is_best)

            self.early_stopper_fno(val_loss)
            if self.early_stopper_fno.early_stop:
                self.logger.info("EARLY STOPPING FNO TRIGGERED, NO IMPROVEMENT IN SEVERAL EPOCHS")
                break


    def train_ebm(self, num_epochs):
        """
        Train EBM model only using Score Matching, with checkpointing and early stopping.

        Uses ebm_train_loader (noisy data in dual-dataset mode, same as FNO otherwise).
        """
        mode_str = "noisy observations" if self.dual_dataset_mode else "same dataset as FNO"
        self.logger.info(f"Starting EBM training with Score Matching on {mode_str}...")
        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            self.ebm_model.train()
            epoch_loss = 0

            loop = tqdm(self.ebm_train_loader)  # Use EBM-specific loader
            for batch_idx, (x, y) in enumerate(loop):
                x, y = x.to(self.config.device), y.to(self.config.device)

                # Enable tracing for first batch of each epoch and every 50 batches
                trace_score = (batch_idx == 0) #or (batch_idx % 50 == 0)

                if trace_score:
                    loss, trace_info = self.train_step_ebm_scorematching(x, y, trace_score=True)
                    # Print trace information to terminal
                    self._print_score_trace_info(epoch, batch_idx, trace_info)
                else:
                    loss = self.train_step_ebm_scorematching(x, y, trace_score=False)

                epoch_loss += loss

                if (batch_idx + 1) % self.accumulation_steps == 0:
                    self.ebm_optimizer.step()
                    self.ebm_optimizer.zero_grad()

                loop.set_description(f"EBM Training Epoch [{epoch+1}/{num_epochs}]")
                loop.set_postfix(ebm_loss=loss)

            # EBM Validation (uses EBM val loader with score matching)
            val_loss = self.validate_ebm()
            avg_epoch_loss = epoch_loss / len(self.ebm_train_loader)

            self.logger.info(
                f"EBM Epoch {epoch+1}: Train Loss={avg_epoch_loss:.4f}, "
                f"Val Loss (Score Matching)={val_loss:.4f}"
            )

            if self.ebm_scheduler:
                # ReduceLROnPlateau requires metric, other schedulers don't
                if isinstance(self.ebm_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.ebm_scheduler.step(val_loss)
                else:
                    self.ebm_scheduler.step()

            # Save checkpoint every epoch
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                self.best_val_loss = val_loss

            self.checkpoint(epoch, val_loss, 'ebm', is_best=is_best)

            # self.early_stopper_ebm(val_loss)
            # if self.early_stopper_ebm.early_stop:
            #     self.logger.info("EARLY STOPPING TRIGGERED FOR EBM, NO IMPROVEMENT IN SEVERAL EPOCHS")
            #     break

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

    def _print_score_trace_info(self, epoch, batch_idx, trace_info):
        """Print enhanced score matching trace information to terminal"""
        self.logger.info("=" * 80)
        self.logger.info(f"ENHANCED SCORE MATCHING TRACE - Epoch {epoch}, Batch {batch_idx}")
        self.logger.info("=" * 80)

        # Noise levels used
        self.logger.info(f"Noise Levels (σ): {trace_info['sigma_levels']}")

        # Score losses per noise level
        self.logger.info("\nSCORE LOSSES BY NOISE LEVEL:")
        for i, (sigma, loss) in enumerate(zip(trace_info['sigma_levels'], trace_info['score_losses'])):
            self.logger.info(f"  σ={sigma:5.2f}: Loss={loss:8.4f}")

        # Score norm ratios (key diagnostic for convergence)
        self.logger.info("\nSCORE NORM RATIOS (should approach 1.0):")
        if trace_info.get('score_norm_ratio_0.01', 0.0) > 0:
            self.logger.info(f"  σ=0.01: Ratio={trace_info['score_norm_ratio_0.01']:.4f}")
        if trace_info.get('score_norm_ratio_0.05', 0.0) > 0:
            self.logger.info(f"  σ=0.05: Ratio={trace_info['score_norm_ratio_0.05']:.4f}")

        # Total loss breakdown
        self.logger.info("\nLOSS BREAKDOWN:")
        self.logger.info(f"  Score Loss:         {trace_info['total_score_loss']:8.4f}")
        self.logger.info(f"  Calibration Loss:   {trace_info.get('calibration_loss', 0.0):8.4f}")
        self.logger.info(f"  Energy Reg:         {trace_info['energy_reg']:8.4f}")
        self.logger.info(f"  Total Loss:         {trace_info['total_loss']:8.4f}")

        # Diagnosis
        self.logger.info("\nDIAGNOSIS:")
        avg_score_loss = sum(trace_info['score_losses']) / len(trace_info['score_losses']) if trace_info['score_losses'] else 0
        if avg_score_loss > 10.0:
            self.logger.info("  ⚠ WARNING: High score matching loss - model may need more training")
        if trace_info['energy_reg'] > trace_info['total_score_loss']:
            self.logger.info("  ⚠ WARNING: Energy regularization dominates loss - consider reducing ebm_energy_reg")

        # Check calibration loss
        if trace_info.get('calibration_loss', 0.0) > trace_info['total_score_loss']:
            self.logger.info("  ⚠ WARNING: Calibration loss dominates - consider reducing ebm_weight_calibration")

        # Check score norm ratios (critical for convergence)
        ratio_001 = trace_info.get('score_norm_ratio_0.01', 0.0)
        ratio_005 = trace_info.get('score_norm_ratio_0.05', 0.0)

        if ratio_001 > 0 and (ratio_001 < 0.5 or ratio_001 > 1.5):
            self.logger.info(f"  ⚠ WARNING: Score norm mismatch at σ=0.01 (ratio={ratio_001:.2f}, should be ~1.0)")
        if ratio_005 > 0 and (ratio_005 < 0.5 or ratio_005 > 1.5):
            self.logger.info(f"  ⚠ WARNING: Score norm mismatch at σ=0.05 (ratio={ratio_005:.2f}, should be ~1.0)")

        # Success indicators
        if ratio_001 > 0.7 and ratio_001 < 1.3:
            self.logger.info(f"  ✓ Good score matching convergence at σ=0.01 (ratio={ratio_001:.2f})")

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
        # self.logger.info("Stage 2: Training EBM")
        # self.train_ebm(self.config.ebm_epochs)
        
        self.logger.info("Staged training completed")

# ...existing code...


#Original
# cd_loss = pos_energy.mean() - neg_energy.mean()
# return energy, gonna try return -energy
# y_neg = y_neg - step_size * grad_energy + noise