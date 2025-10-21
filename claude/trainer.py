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

    def energy(self, u, x, u_fno=None):
        """
        Compute total energy E(u, X)

        Args:
            u: candidate solution (batch, n_x, n_y, 1)
            x: input coordinates (batch, n_x, n_y, 3)
            u_fno: pre-computed FNO solution (optional)
        Returns:
            E: total energy (batch,)
        """
        if u_fno is None:
            with torch.no_grad():
                u_fno = self.u_fno(x)

        # Quadratic term: anchors to FNO solution
        quadratic_term = 0.5 * torch.mean((u - u_fno)**2, dim=[1, 2, 3]) #configurable squared sigma

        # Potential term: captures uncertainty structure
        potential_term = self.V_ebm(u, x)

        return quadratic_term + potential_term

    def forward(self, x):
        """Direct FNO prediction"""
        return self.u_fno(x)


# ============================================================================
# Trainer Class
# ============================================================================

class Trainer:
    def __init__(self, model: FNO_EBM,phy_loss: PhysicsLossFn, train_loader, val_loader, config: Config):
        """
        Initialize trainer with models, data loaders and configuration
        """
        self.fno_model = model.u_fno
        self.ebm_model = model.V_ebm
        self.train_loader = train_loader
        self.val_loader = val_loader
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
        
        self.early_stopper = EarlyStopping(
            patience=config.patience, 
            verbose=True
        )
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def checkpoint(self, epoch, val_loss):
        """
        Save model checkpoint
        """
        checkpoint = {
            'epoch': epoch,
            'fno_model': self.fno_model.state_dict(),
            'ebm_model': self.ebm_model.state_dict(),
            'fno_optimizer': self.fno_optimizer.state_dict(),
            'ebm_optimizer': self.ebm_optimizer.state_dict(),
            'val_loss': val_loss,
        }
        
        checkpoint_path = os.path.join(
            self.config.checkpoint_dir, 
            f'checkpoint_epoch_{epoch}.pt'
        )
        torch.save(checkpoint, checkpoint_path)
        
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            best_path = os.path.join(
                self.config.checkpoint_dir,
                'best_model.pt'
            )
            torch.save(checkpoint, best_path)

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
        
        # Negative phase (MCMC sampling)
        y_neg = y.clone().detach()
        y_neg.requires_grad = True
        
        for _ in range(self.config.mcmc_steps):
            y_neg.requires_grad = True
            neg_energy = self.ebm_model(x, y_neg)
            neg_grad = torch.autograd.grad(
                neg_energy.sum(), 
                y_neg,
                create_graph=True
            )[0]
            y_neg = y_neg - self.config.mcmc_step_size * neg_grad
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
            for x, y in self.val_loader:
                x, y = x.to(self.config.device), y.to(self.config.device)
                fno_output = self.fno_model(x)
                
                # Data loss
                data_loss = self.fno_criterion(fno_output, y)
                
                # Physics loss
                x_coords = x[..., 0].requires_grad_(True)
                y_coords = x[..., 1].requires_grad_(True)
                residual = self.phy_loss_fn(fno_output, x_coords, y_coords)
                #residual = compute_pde_residual(fno_output, x_coords, y_coords)
                physics_loss = torch.mean(residual**2)
                
                # Total validation loss
                val_loss = data_loss + self.lambda_phys * physics_loss
                total_val_loss += val_loss.item()
                
        avg_val_loss = total_val_loss / len(self.val_loader)
        return avg_val_loss
    
    def validate_ebm(self):
        """
        Validation step for the EBM.
        Computes the average energy gap on the validation set.
        A lower (more negative) loss is better.
        """
        self.ebm_model.eval()
        total_ebm_val_loss = 0

        for x, y in self.val_loader:
            x, y = x.to(self.config.device), y.to(self.config.device)

            # Positive phase (no grad needed)
            with torch.no_grad():
                pos_energy = self.ebm_model(x, y)

            # Negative phase (MCMC sampling - needs gradients)
            y_neg = y.clone().detach()
            noise_scale = np.sqrt(2 * self.config.mcmc_step_size)

            for _ in range(self.config.mcmc_steps):
                y_neg.requires_grad_(True)
                neg_energy_for_grad = self.ebm_model(x, y_neg)
                neg_grad = torch.autograd.grad(neg_energy_for_grad.sum(), y_neg)[0]

                with torch.no_grad():
                    noise = torch.randn_like(y_neg) * noise_scale
                    y_neg = y_neg - self.config.mcmc_step_size * neg_grad + noise
                    y_neg = y_neg.detach()

            with torch.no_grad():
                neg_energy = self.ebm_model(x, y_neg)
                ebm_loss = pos_energy.mean() - neg_energy.mean()
                total_ebm_val_loss += ebm_loss.item()

        avg_ebm_val_loss = total_ebm_val_loss / len(self.val_loader)
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
        """
        #self.fno_optimizer.zero_grad()
        
        # Forward pass
        fno_output = self.fno_model(x)
        
        # Data loss
        data_loss = self.fno_criterion(fno_output, y)
        
        # Physics loss (PDE residual)
        x_coords = x[..., 0].requires_grad_(True)
        y_coords = x[..., 1].requires_grad_(True)
        residual = self.phy_loss_fn(fno_output, x_coords, y_coords)
        #residual = compute_pde_residual(fno_output, x_coords, y_coords)
        physics_loss = torch.mean(residual**2)
        
        # Total loss
        total_loss = data_loss + self.lambda_phys * physics_loss
        total_loss_scaled = total_loss / self.accumulation_steps

        total_loss_scaled.backward()
        #self.fno_optimizer.step()
        
        return total_loss.item()

    def train_step_ebm(self, x, y, langevin=True):
        """
        Single training step for EBM only
        """
        #self.ebm_optimizer.zero_grad()
        
        # Positive phase
        pos_energy = self.ebm_model(x, y)
        
        # Negative phase (MCMC sampling)
        y_neg = y.clone().detach()
        y_neg.requires_grad = True
        
        for _ in range(self.config.mcmc_steps):   #Apakah ini langevin dynamics tanpa function
            y_neg.requires_grad = True
            neg_energy = self.ebm_model(x, y_neg)
            neg_grad = torch.autograd.grad(
                neg_energy.sum(), 
                y_neg,
                create_graph=True
            )[0]
            #No langevin noise
            if langevin:
                noise = torch.randn_like(y_neg) * np.sqrt(2 * self.config.mcmc_step_size)
                y_neg = y_neg - self.config.mcmc_step_size * neg_grad + noise
            else:
                y_neg = y_neg - self.config.mcmc_step_size * neg_grad
            y_neg = y_neg.detach()
        
        neg_energy = self.ebm_model(x, y_neg)
        ebm_loss = pos_energy.mean() - neg_energy.mean()
        ebm_loss_scaled = ebm_loss / self.accumulation_steps
        ebm_loss_scaled.backward()
        #self.ebm_optimizer.step()

        return ebm_loss.item()

    def train_fno(self, num_epochs):
        """
        Train FNO model only
        """
        self.logger.info("Starting FNO training...")
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            self.fno_model.train()
            epoch_loss = 0
            
            loop = tqdm(self.train_loader)
            for batch_idx, (x, y) in enumerate(loop):
                x, y = x.to(self.config.device), y.to(self.config.device)
                loss = self.train_step_fno(x, y)
                epoch_loss += loss

                if (batch_idx + 1) % self.accumulation_steps == 0:
                    self.fno_optimizer.step()  #apply gradient update
                    self.fno_optimizer.zero_grad()  #reset gradient to zero
                
                loop.set_description(f"FNO Training Epoch [{epoch}/{num_epochs}]")
                loop.set_postfix(fno_loss=loss)
            
            val_loss = self.validate()
            avg_epoch_loss = epoch_loss / len(self.train_loader)
            
            self.logger.info(
                f"FNO Epoch {epoch}: Train Loss={avg_epoch_loss:.4f}, "
                f"Val Loss={val_loss:.4f}"
            )

            if self.fno_scheduler:
                self.fno_scheduler.step()
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.checkpoint(epoch, val_loss)
            
            self.early_stopper(val_loss)
            if self.early_stopper.early_stop:
                self.logger.info("EARLY STOPPING FNO TRIGGERED, NO IMPROVEMENT IN SEVERAL EPOCHS")
                break


    def train_ebm(self, num_epochs):
        """
        Train EBM model only, with checkpointing and early stopping.
        """
        best_val_loss = float('inf')
        self.logger.info("Starting EBM training...")
        # Reset early stopper for the EBM stage
        
        for epoch in range(num_epochs):
            self.ebm_model.train()
            epoch_loss = 0
            
            loop = tqdm(self.train_loader)
            for batch_idx, (x, y) in enumerate(loop):
                x, y = x.to(self.config.device), y.to(self.config.device)
                loss = self.train_step_ebm(x, y)
                epoch_loss += loss

                if (batch_idx + 1) % self.accumulation_steps == 0:
                    self.ebm_optimizer.step()
                    self.ebm_optimizer.zero_grad()
                
                loop.set_description(f"EBM Training Epoch [{epoch+1}/{num_epochs}]")
                loop.set_postfix(ebm_loss=loss)
            
            # EBM Validation
            val_loss = self.validate_ebm()
            avg_epoch_loss = epoch_loss / len(self.train_loader)
            
            self.logger.info(
                f"EBM Epoch {epoch+1}: Train Loss={avg_epoch_loss:.4f}, "
                f"Val Loss (Energy Gap)={val_loss:.4f}"
            )

            if self.ebm_scheduler:
                self.ebm_scheduler.step()

            # Checkpoint and Early Stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.checkpoint(epoch, val_loss)

            self.early_stopper(val_loss)
            if self.early_stopper.early_stop:
                self.logger.info("EARLY STOPPING TRIGGERED FOR EBM, NO IMPROVEMENT IN SEVERAL EPOCHS")
                break


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