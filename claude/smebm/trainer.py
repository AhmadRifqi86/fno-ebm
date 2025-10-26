import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm
import logging
from config import Config, Factory
from customs import EarlyStopping, PhysicsLossFn
import numpy as np

# Import torchebm library
try:
    from torchebm.samplers import LangevinDynamics as LangevinSampler
    from torchebm.losses import ContrastiveDivergence
    TORCHEBM_AVAILABLE = True
    print("Using torchebm library for EBM training")
except ImportError:
    TORCHEBM_AVAILABLE = False
    print("Warning: torchebm library not available")
    print("Install with: pip install torchebm")

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

            return quadratic_term + potential_term

    def forward(self, x):
        """Direct FNO prediction"""
        return self.u_fno(x)


# ============================================================================
# Trainer Class (Using torchebm library)
# ============================================================================

class Trainer:
    def __init__(self, model: FNO_EBM, phy_loss: PhysicsLossFn,
                 train_loader, val_loader, config: Config,
                 ebm_train_loader=None, ebm_val_loader=None):
        """
        Initialize trainer with models, data loaders and configuration.

        This version uses the torchebm library for EBM training instead of
        from-scratch implementation.

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
        if not TORCHEBM_AVAILABLE:
            raise ImportError(
                "torchebm library is required for this trainer. "
                "Install with: pip install torchebm"
            )

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

        # ======================================================================
        # torchebm library components
        # ======================================================================
        # Create Langevin sampler for MCMC sampling
        # Note: The sampler will be passed to ContrastiveDivergence
        self.langevin_sampler = LangevinSampler(
            energy_function=self.ebm_model,
            step_size=config.mcmc_step_size,
            noise_scale=np.sqrt(2 * config.mcmc_step_size)
        )

        # Create Contrastive Divergence loss
        # The EBM model is the energy function
        self.cd_loss_fn = ContrastiveDivergence(
            energy_function=self.ebm_model,
            sampler=self.langevin_sampler,
            k_steps=config.mcmc_steps  # Number of MCMC steps for negative sample generation
        )

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self.logger.info("Using torchebm library for EBM training")
        self.logger.info(f"  Langevin sampler: step_size={config.mcmc_step_size}, num_steps={config.mcmc_steps}")

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
        Validation step for the EBM using torchebm library.
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
                pos_energy = self.model.energy(y, x, training=False, u_fno=u_fno)

            # Negative phase: Use torchebm's Langevin sampler
            # Initialize from FNO prediction for inference
            y_neg = u_fno.clone().detach()

            # Sample using torchebm's Langevin sampler
            y_neg = self.langevin_sampler.sample(
                init_samples=y_neg,
                condition=x  # Pass condition if needed
            )

            with torch.no_grad():
                neg_energy = self.model.energy(y_neg, x, training=False, u_fno=u_fno)
                ebm_loss = pos_energy.mean() - neg_energy.mean()
                total_ebm_val_loss += ebm_loss.item()

        avg_ebm_val_loss = total_ebm_val_loss / len(self.ebm_val_loader)
        return avg_ebm_val_loss

    def train_step_fno(self, x, y):
        """
        Single training step for FNO with both data and physics losses

        Returns:
            tuple: (total_loss, data_loss, physics_loss) as floats
        """
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

        return total_loss.item(), data_loss.item(), physics_loss.item()

    def train_step_ebm(self, x, y):
        """
        EBM training step using torchebm library.

        This replaces the from-scratch MCMC implementation with torchebm's
        optimized samplers and loss functions.

        Args:
            x: Input coordinates (batch, n_x, n_y, 3)
            y: Ground truth solutions (batch, n_x, n_y, 1)

        Returns:
            ebm_loss: EBM loss value (float)
        """
        # STEP 1: Add small noise to positive samples (prevents overfitting)
        small_noise = torch.randn_like(y) * 0.005
        y_noisy = y + small_noise
        y_noisy = torch.clamp(y_noisy, -3, 3)

        # STEP 2: Use torchebm's Contrastive Divergence loss
        # This handles:
        # - Computing positive energy
        # - MCMC sampling for negative samples (via sampler)
        # - Computing negative energy
        # - Computing CD loss
        # Returns: (loss, negative_samples)
        x_combine = torch.cat([y_noisy, x], dim=-1)
        ebm_loss, _ = self.cd_loss_fn(x_combine)

        # Scale loss for gradient accumulation
        ebm_loss_scaled = ebm_loss / self.accumulation_steps
        ebm_loss_scaled.backward()

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
        Train EBM model only using torchebm library.

        Uses ebm_train_loader (noisy data in dual-dataset mode, same as FNO otherwise).
        """
        mode_str = "noisy observations" if self.dual_dataset_mode else "same dataset as FNO"
        self.logger.info(f"Starting EBM training with torchebm on {mode_str}...")
        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            self.ebm_model.train()
            epoch_loss = 0

            loop = tqdm(self.ebm_train_loader)  # Use EBM-specific loader
            for batch_idx, (x, y) in enumerate(loop):
                x, y = x.to(self.config.device), y.to(self.config.device)
                loss = self.train_step_ebm(x, y)
                epoch_loss += loss

                if (batch_idx + 1) % self.accumulation_steps == 0:
                    self.ebm_optimizer.step()
                    self.ebm_optimizer.zero_grad()

                loop.set_description(f"EBM Training Epoch [{epoch+1}/{num_epochs}] (torchebm)")
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


    def train_staged(self):
        """
        Staged training: First FNO, then EBM
        """
        self.logger.info("Starting staged training (using torchebm library)...")

        # Stage 1: Train FNO
        self.logger.info("Stage 1: Training FNO")
        self.train_fno(self.config.fno_epochs)

        # Stage 2: Train EBM with torchebm
        self.logger.info("Stage 2: Training EBM with torchebm library")
        self.train_ebm(self.config.ebm_epochs)

        self.logger.info("Staged training completed")

# ...existing code...