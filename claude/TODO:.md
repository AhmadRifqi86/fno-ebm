# TODO:


## Find out similar paper to find out whether my idea already done or not
## Create Main entry-point [DONE] [Mon 13 Oct 2025]
## Implement factory.py [DONE] [Wed 15 Oct 2025] 
## Sanity check to find code vs concept mismatch [DONE] [Wed 15 Oct 2025]
## Installing physics simulator, possibly using dockerized fenics [DONE] [Mon 13 Oct 2025]
## Implementing datautils.py to store dataloader and visualization tools [DONE] [Wed 15 Oct 2025]
## Diagnostic tools [DONE] [Sun 19 Oct 2025]
## Creating generate data script [DONE] [Mon 20 Oct 2025]
## Creating sythetics data
## Using transformer based and mamba based model for FNO [DONE] [Fri 17 Oct 2025]
## Using KAN based model for EBM  [DONE]  [Fri 17 Oct 2025]
## Implementing multi-channel attention for transformerFNO 
## Separating model for FNO and EBM [DONE] [Sun 19 Oct 2025]
## Implementing deploy.py 
## Test main.py using dummy data [DONE] [Wed 21 Oct 2025]
## Fixing checkpointing mechanism, only save if there is improvement, with removing the older file first [DONE] [Wed 21 Oct 2025]
## Creating two type of main: Single-noisy, double-noisy-separate, try to train for both [DONE] [Wed 21 Oct 2025]
## Log the physics loss and data loss separately [DONE] [Thu 23 Oct 2025]
## Fixing physics loss scale and lambda physics param on trainer and config [DONE] [Thu 23 Oct 2025]
## Baseline Paper and Reference Paper [DONE] [Thu 23 Oct 2025]  
## Download dataset PDEBench, Command [PLAN] [Fri 24 Oct 2025] (baru darcy flow)
## Training using single noisy data [PLAN] [Mon 27 Oct 2025]
## Maybe i should try using torchebm library [PLAN] [Mon 27 Oct 2025] [Probably]
## Create training automation script so i could leave the laptop
## Test using other synthetic PDE for single and separated data [PLAN] [Mon 27 Oct 2025]
## Normalize permeability data [PLAN] [Sat 26 Oct 2025] [Probably]
## One more training to visualize negative sample plot [PLAN] [Sun 26 Oct 2025] [Probably]
## Try Score Matching EBM [PLAN]
## Training FNO model using EBM pos-neg sampling and contrastive divergence loss (plug FNO model into EBM training regime)[FAR] 
wget https://darus.uni-stuttgart.de/api/access/datafile/133139 \
       -O data/pdebench/2D_DarcyFlow_beta1.0_Train.hdf5
  wget https://darus.uni-stuttgart.de/api/access/datafile/133140 \
       -O data/pdebench/2D_DarcyFlow_beta1.0_Valid.hdf5


# Recommendation todo for EBM (24 Oct 2025)
 My Recommended Order:
  1. Start from FNO in training (#2) - Easy change, big impact
  2. Increase MCMC steps in training (#1) - From 40 → 200
  3. Reduce step size (#3A) - From 0.1 → 0.01
  4. Diagnostic check (#10) - See what was actually learned
  5. If still fails: Convolutional EBM (#4) - Architectural change
  6. If desperate: Persistent CD (#7) - More advanced technique

# Recommendation todo for EBM (25 Oct 2025)
Recommended Solutions (in order of impact):

  Solution 1: Implement Persistent Contrastive Divergence (HIGHEST IMPACT)

  class Trainer:
      def __init__(self, ...):
          # Add replay buffer
          self.replay_buffer = []  # Store previous negatives

      def train_step_ebm(self, x, y):
          # Initialize negatives from replay buffer (80%) + noise (20%)
          if len(self.replay_buffer) > 0 and random.random() < 0.8:
              y_neg = self.replay_buffer.pop(0)  # Reuse old negative
          else:
              y_neg = torch.randn_like(y)  # Fresh random sample

          # Run MCMC
          for _ in range(mcmc_steps):
              # ... langevin dynamics ...

          # Store back to buffer
          self.replay_buffer.append(y_neg.detach())
          if len(self.replay_buffer) > 1000:  # Keep buffer size manageable
              self.replay_buffer.pop(0)

  This is THE KEY missing piece!

  Solution 2: Try Single-Dataset Mode (SECOND HIGHEST IMPACT)

  Stop using dual-dataset mode entirely:
  # Both FNO and EBM train on SAME noisy data
  trainer = Trainer(
      model=model,
      train_loader=noisy_data_loader,  # Same for both!
      ebm_train_loader=None  # Don't provide separate loader
  )

  This eliminates the distribution mismatch.

  Solution 3: Better Negative Initialization

  Instead of y + 0.2*noise, try:
  # Option A: Uniform random on grid
  y_neg = torch.rand_like(y) * 2 - 1  # [-1, 1]

  # Option B: Mix of strategies
  if random.random() < 0.33:
      y_neg = torch.randn_like(y)  # Pure noise
  elif random.random() < 0.66:
      y_neg = y + 0.5 * torch.randn_like(y)  # More corruption
  else:
      y_neg = self.fno_model(x) + 0.3 * torch.randn_like(y)  # From FNO

  Solution 4: Add Spectral Normalization

  Quick fix to ebm.py:
  import torch.nn.utils.spectral_norm as spectral_norm

  class ConvEBM(nn.Module):
      def __init__(self, ...):
          layers = []
          for ...
              layers.append(spectral_norm(
                  nn.Conv2d(prev_channels, hidden_ch, kernel_size=3, padding=1)
              ))


# Note
- When lamda_physics sets to 0.0, the FNO learns from data only and giving good results on synthetic data 
- For some reason, unset lambda_phys for physics loss regularizer has a huge impact on total loss
- For now EBM training result is mediocre


# Log
- Applying noise to EBM training step instead of deterministic gradient descent, still retraining


# Next Question for Claude
- Will spectral normalization help both PCD and SM EBM training?
- Does the input data for EBM already normalized?
- What about training FNO under EBM score matching loss? [web]
- What about bayesian-FNO? [web]
