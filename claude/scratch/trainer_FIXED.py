"""
FIXED train_step_ebm() - Based on UvA Deep Energy Models Tutorial
https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial8/Deep_Energy_Models.html

CRITICAL FIXES:
1. NO create_graph during MCMC sampling
2. Random uniform initialization for buffer
3. Add small noise to positive samples
4. Use 60 MCMC steps (not 200)
5. Proper loss computation
"""

def train_step_ebm_FIXED(self, x, y, langevin=True):
    """
    FIXED EBM training step following UvA tutorial exactly

    Key changes from broken version:
    - NO create_graph=True during MCMC (was causing gradient issues)
    - Random initialization instead of corrupted ground truth
    - Small noise added to real samples
    - 60 MCMC steps instead of 200
    """
    import torch
    import numpy as np

    batch_size = x.shape[0]

    # STEP 1: Add small noise to positive samples (UvA trick!)
    # This prevents overfitting to exact training data
    small_noise = torch.randn_like(y) * 0.005
    y_noisy = y + small_noise
    y_noisy = torch.clamp(y_noisy, -3, 3)  # Prevent extreme values

    # STEP 2: Compute positive energy
    pos_energy = self.ebm_model(x, y_noisy)

    # STEP 3: Initialize negative samples from buffer (PCD)
    if self.use_pcd and len(self.replay_buffer) > 0 and np.random.random() > self.pcd_reinit_prob:
        # Sample from buffer (95% of time)
        buffer_indices = np.random.choice(
            len(self.replay_buffer),
            size=min(batch_size, len(self.replay_buffer)),
            replace=False
        )
        y_neg = torch.stack([self.replay_buffer[i][1] for i in buffer_indices]).to(x.device)

        # Pad if needed
        if y_neg.shape[0] < batch_size:
            n_missing = batch_size - y_neg.shape[0]
            # UvA initialization: random uniform [-1, 1]
            y_neg_extra = torch.rand(n_missing, *y.shape[1:], device=x.device) * 2 - 1
            y_neg = torch.cat([y_neg, y_neg_extra], dim=0)
    else:
        # Fresh initialization (5% of time): random uniform [-1, 1]
        y_neg = torch.rand_like(y) * 2 - 1

    # STEP 4: MCMC Sampling (CRITICAL: NO create_graph!)
    # UvA uses 60 steps during training, 256 for generation
    mcmc_steps = 60  # Fixed to UvA value
    step_size = self.config.mcmc_step_size
    grad_clip = 0.03  # UvA value

    y_neg = y_neg.detach()  # Ensure no gradients from initialization

    for i in range(mcmc_steps):
        # Add noise and clamp
        noise_this_step = torch.randn_like(y_neg) * 0.005
        y_neg = y_neg + noise_this_step
        y_neg = torch.clamp(y_neg, -3, 3)

        # Enable gradients for this step only
        y_neg.requires_grad_(True)

        # Compute energy
        neg_energy_for_grad = self.ebm_model(x, y_neg)

        # Compute gradients WITHOUT create_graph
        # CRITICAL FIX: No create_graph=True!
        neg_grad = torch.autograd.grad(
            neg_energy_for_grad.sum(),
            y_neg,
            retain_graph=False,  # Don't keep computation graph
            create_graph=False   # FIXED: Was True, now False!
        )[0]

        # Clip gradients (UvA uses 0.03)
        neg_grad = torch.clamp(neg_grad, -grad_clip, grad_clip)

        # Update y_neg (detach to break gradient flow)
        with torch.no_grad():
            # Langevin dynamics update
            if langevin:
                noise = torch.randn_like(y_neg) * np.sqrt(2 * step_size)
                y_neg = y_neg - step_size * neg_grad + noise
            else:
                y_neg = y_neg - step_size * neg_grad

        y_neg = y_neg.detach()  # Critical: detach after each step

    # STEP 5: Compute final negative energy
    neg_energy = self.ebm_model(x, y_neg)

    # STEP 6: Update buffer
    if self.use_pcd:
        for i in range(batch_size):
            self.replay_buffer.append((
                x[i].detach().cpu(),
                y_neg[i].detach().cpu()
            ))

        # Keep buffer size manageable
        if len(self.replay_buffer) > self.pcd_buffer_size:
            self.replay_buffer = self.replay_buffer[-self.pcd_buffer_size:]

    # STEP 7: Compute loss (UvA formula)
    # Regularization: keep energies bounded
    alpha = 0.1  # UvA default
    reg_loss = alpha * (pos_energy ** 2 + neg_energy ** 2).mean()

    # Contrastive divergence: push up neg energy, push down pos energy
    # UvA: fake.mean() - real.mean()
    # In our case: neg_energy.mean() - pos_energy.mean()
    cd_loss = neg_energy.mean() - pos_energy.mean()

    # Total loss
    ebm_loss = reg_loss + cd_loss

    # Backward (will only backprop through energy, not MCMC)
    ebm_loss_scaled = ebm_loss / self.accumulation_steps
    ebm_loss_scaled.backward()

    return ebm_loss.item()


# ALSO FIX THE OPTIMIZER:
# In Trainer.__init__(), change EBM optimizer to:

"""
self.ebm_optimizer = optim.Adam(
    self.ebm_model.parameters(),
    lr=config.ebm_learning_rate,
    betas=(0.0, 0.999)  # CRITICAL: beta1=0.0 like UvA!
)
"""