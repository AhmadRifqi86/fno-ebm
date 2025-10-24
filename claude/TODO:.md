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
## Underfitting Indication, try to fix underfitting
## Log the physics loss and data loss separately [DONE] [Thu 23 Oct 2025]
## Fixing physics loss scale and lambda physics param on trainer and config [DONE] [Thu 23 Oct 2025]
## Baseline Paper and Reference Paper [PLAN] [Thu 23 Oct 2025]  
## Download dataset PDEBench, Command [PLAN] [Fri 24 Oct 2025]
## Training with mcmc_steps set to 200 and 0.01
## Create training automation script so i could leave the laptop
## Test using other synthetic PDE for single and separated data [PLAN] [Mon 27 Oct 2025]
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

# Note
- When lamda_physics sets to 0.0, the FNO learns from data only and giving good results on synthetic data 
- For some reason, unset lambda_phys for physics loss regularizer has a huge impact on total loss
- For now EBM training result is mediocre


# Log
- Applying noise to EBM training step instead of deterministic gradient descent, still retraining
