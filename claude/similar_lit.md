1. Probabilistic Predictions with Fourier Neural Operators (OpenReview)

  - Relevance: Directly combines FNO with generative modeling based on strictly proper scoring rules
  - Key insight: Uses energy score as a strictly proper scoring rule in Hilbert spaces for probabilistic predictions
  - Link: https://openreview.net/forum?id=orKA6gJwlB

  2. Probabilistic Neural Operators for Functional Uncertainty Quantification (2025)

  - Paper: arXiv:2502.12902
  - Relevance: PNO framework extends neural operators with generative modeling for learning probability distributions
  - Key contribution: Integrates uncertainty information directly into training using energy-based scoring rules
  - Link: https://arxiv.org/abs/2502.12902

  3. DGenNO: Deep Generative Neural Operator for Forward and Inverse PDE Problems (2025)

  - Paper: arXiv:2502.06250
  - Relevance: Physics-aware neural operator based on deep generative probabilistic modeling
  - Key feature: Combines generative framework with neural operators for both forward/inverse problems
  - Link: https://arxiv.org/html/2502.06250

  4. Integrating Fourier Neural Operators with Diffusion Models (2025)

  - Paper: arXiv:2504.00757
  - Relevance: Combines neural operators with denoising diffusion probabilistic models
  - Application: Physics-based probabilistic modeling for synthetic earthquake ground motion
  - Link: https://arxiv.org/html/2504.00757

  5. Improved Training of PINNs Using Energy-Based Priors (OpenReview)

  - Relevance: Directly uses energy-based models as Bayesian priors for physics-informed networks
  - Key result: EBM priors expedite PINN convergence by 10x faster
  - Link: https://openreview.net/forum?id=zqkfJA6R1-r

  6. Neural Operator induced Gaussian Process (NOGaP) Framework

  - Journal: Computer Methods in Applied Mechanics and Engineering
  - Relevance: Combines neural operators with probabilistic GP framework for uncertainty quantification
  - Key feature: Provides both predictions and uncertainty measures for parametric PDEs
  - Link: https://www.sciencedirect.com/science/article/abs/pii/S0045782524005218

  7. Adversarially Trained Neural Operators with Generative Models (2025)

  - Paper: arXiv:2505.23106
  - Relevance: Combines operator learning with adversarial generative modeling
  - Application: Turbulent flow predictions with 15x reduction in energy-spectrum error
  - Innovation: Overcomes oversmoothing in standard L2-trained neural operators

  8. Distribution-Free Uncertainty Quantification in Neural Operators (2024)

  - Paper: arXiv:2412.09369
  - Relevance: Conformalized Randomized Prior Operator (CRP-O) for uncertainty quantification
  - Approach: Combines neural operators with conformal prediction for probabilistic guarantees
  - Link: https://arxiv.org/html/2412.09369


Key Differences Between Your Approach and the Papers

  1. Fundamental Architecture Difference

  Papers #1 & #2 (Probabilistic Neural Operators):
  - Use sample-based approaches (dropout/reparameterization) to generate probabilistic predictions
  - Learn empirical distributions through multiple stochastic forward passes
  - No explicit energy function - just sample generation

  Your FNO-EBM (from model.py:706-787):
  - Uses explicit energy-based formulation: E(u, X) = 0.5 * ||u - u_FNO(X)||^2 + V(u, X)
  - Has a dedicated energy potential network V_ebm (EBMPotential)
  - Employs Langevin dynamics MCMC for sampling (trainer.py:355-369)

  2. Training Methodology

  Papers #1 & #2:
  - Train using energy score as loss function:
  L = E[||u - u_true||] - 0.5 * E[||u_i - u_j||]
  - Single-stage training with proper scoring rules
  - No explicit contrastive divergence

  Your Approach (trainer.py:342-377):
  - Uses contrastive divergence training:
  pos_energy = self.ebm_model(x, y)  # positive phase
  # Langevin MCMC for negative samples
  ebm_loss = pos_energy.mean() - neg_energy.mean()
  - Two-stage training (trainer.py:472-486):
    a. Stage 1: Train FNO with physics loss
    b. Stage 2: Train EBM with contrastive learning

  3. Uncertainty Quantification Mechanism

  Papers #1 & #2:
  - Uncertainty from stochastic dropout in weights/Fourier modes
  - "Fourier dropout" - randomly zeros frequency components
  - Reparameterization trick for Gaussian distributions

  Your Approach:
  - Uncertainty from energy landscape learned by EBM
  - Samples via gradient-based MCMC (Langevin dynamics)
  - Energy captures distributional structure explicitly

  4. Physics Integration

  Papers #1 & #2:
  - No explicit physics constraint mentioned
  - Focus on data-driven uncertainty

  Your Approach (trainer.py:327-331):
  - Physics-informed training with PDE residual loss:
  physics_loss = torch.mean(residual**2)
  total_loss = data_loss + self.lambda_phys * physics_loss
  - FNO explicitly optimized for physics constraints

  5. Loss Function Design

  | Aspect             | Papers #1 & #2                     | Your FNO-EBM           |
  |--------------------|------------------------------------|------------------------|
  | FNO Loss           | MSE + Energy Score                 | MSE + Physics Residual |
  | Probabilistic Loss | Energy Score (proper scoring rule) | Contrastive Divergence |
  | Sampling           | Forward passes with dropout        | Langevin MCMC          |

  6. Model Coupling

  Papers #1 & #2:
  - Single integrated model
  - Probabilistic FNO as unified architecture

  Your Approach (model.py:751-787):
  - Two separate networks:
    - u_fno: Deterministic predictor (various architectures available)
    - V_ebm: Energy potential (MLP/KAN/GNN options)
  - Coupled through energy formulation

  7. Advanced EBM Variants

  Your implementation includes unique EBM architectures not found in the papers:

  - KAN-based EBM (model.py:883-954): Uses spline-based learnable activations
  - Hybrid FNO-KAN EBM (model.py:957-1056): FNO encoder + KAN energy head
  - GNN-based EBM (model.py:1104-1250): Graph convolution for spatial structure

  ---
  Summary: Your Novel Contributions

  1. ✅ Explicit energy-based formulation (not just sampling)
  2. ✅ Physics-informed training integrated with EBM
  3. ✅ Contrastive divergence instead of proper scoring rules
  4. ✅ Langevin dynamics MCMC for sampling
  5. ✅ Modular architecture (separate FNO + EBM)
  6. ✅ Multiple EBM backbone options (MLP/KAN/GNN)
  7. ✅ Two-stage training strategy