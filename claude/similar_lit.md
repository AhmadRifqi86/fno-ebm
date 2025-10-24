# Research Novelty and Similar Paper 
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
  ## Langevin MCMC for negative samples
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


# Baseline Paper

## Priority 1: Papers Using PDEBench

### 1. **PDEBench: An Extensive Benchmark for Scientific Machine Learning** (2022)
- **Link**: https://arxiv.org/abs/2210.07182
- **GitHub**: https://github.com/pdebench/PDEBench
- **Relevance**: **PRIMARY BASELINE** - Official benchmark suite with baseline FNO results
- **Dataset**: 11 PDEs including Darcy Flow, Navier-Stokes, Burgers, and more
- **Key FNO Benchmark Results**:
  - Strong general performance across most PDEs
  - Darcy Flow: Relative L2 error ~0.05-0.10 (depending on resolution)
  - Limitations identified: High-frequency dynamics, scaling issues
  - RMSE increases 2 orders of magnitude for high-frequency problems
- **Why cite**: Establishes baseline FNO performance on standard benchmarks

### 2. **Probabilistic Neural Operators (PNO)** (February 2025)
- **Link**: https://arxiv.org/abs/2502.12902
- **Code**: https://github.com/cbuelt/pfno
- **Relevance**: **MOST SIMILAR TO YOUR WORK** - Extends FNO with probabilistic framework for UQ
- **Method**: Generative modeling using strictly proper scoring rules (energy score)
- **Key Difference from Your Work**: Single-stage training with energy score vs your two-stage (FNO→EBM)
- **Benchmark Results**:
  - Darcy Flow: Improved calibration over baseline FNO
  - 2D/3D Navier-Stokes: Better calibration error than deterministic FNO
  - Real weather data: Demonstrates practical applicability
- **Metrics Reported**: CRPS, calibration error, relative L2 error
- **Why cite**: Direct competitor for probabilistic operator learning with UQ

### 3. **DGenNO: Deep Generative Neural Operator for Forward and Inverse PDE Problems** (2025)
- **Link**: https://arxiv.org/abs/2502.06250
- **Relevance**: Physics-aware neural operator with generative probabilistic modeling
- **Method**: Combines neural operators with deep generative framework
- **Application**: Both forward and inverse PDE problems
- **Benchmark**: Multiple PDEs including Darcy flow and Burgers equation
- **Why cite**: Alternative approach to probabilistic neural operators

## Priority 2: Recent UQ Methods for Neural Operators (2024-2025)

### 4. **Conformalized-DeepONet: Uncertainty Quantification with Conformal Prediction** (February 2024)
- **Link**: https://arxiv.org/abs/2402.15406
- **Relevance**: **GOLD STANDARD FOR CALIBRATION** - Distribution-free UQ with coverage guarantees
- **Method**: Conformal prediction framework for neural operators
- **Benchmark Results**:
  - Turbulent flow: 100% empirical coverage at 95% confidence level
  - Mean relative L2 error: 4.82%
  - Elastoplastic deformation: 92.59% coverage, 10.72% error
- **Key Strength**: Theoretical coverage guarantees (your EBM approach doesn't have this)
- **Why cite**: State-of-the-art for calibrated uncertainty quantification

### 5. **Distribution-Free Uncertainty Quantification in Neural Operators** (December 2024)
- **Link**: https://arxiv.org/abs/2412.09369
- **Method**: Conformalized Randomized Prior Operator (CRP-O)
- **Relevance**: Combines neural operators with conformal prediction
- **Key Feature**: Probabilistic guarantees without distributional assumptions
- **Why cite**: Recent alternative UQ approach with theoretical guarantees

### 6. **Approximate Bayesian Neural Operators (ABNO)** (2022-2024)
- **Papers**: Multiple works on Bayesian DeepONet and FNO
- **Method**: Variational inference or MCMC for Bayesian neural operators
- **Benchmark**: Standard operator learning problems
- **Key Feature**: Provides both aleatoric and epistemic uncertainty
- **Limitation**: Computationally expensive (similar to your MCMC sampling)
- **Why cite**: Alternative Bayesian approach to UQ in neural operators

### 7. **DIVERSENO: Diverse Ensembles of Neural Operators** (ICLR 2024)
- **Link**: https://openreview.net/forum?id=zP8cWiSxqy
- **Method**: Ensemble-based uncertainty quantification
- **Relevance**: Simple baseline for UQ - train multiple FNOs with different initializations
- **Benchmark**: Multiple PDEs from PDEBench
- **Advantages**: Easy to implement, well-calibrated for interpolation
- **Disadvantages**: Expensive (multiple models), poor extrapolation
- **Why cite**: Simple but effective baseline to compare against

## Priority 3: Energy-Based Approaches to Neural Operators

### 8. **Energy-Conserving Neural Operators (ENO)** (February 2024)
- **Link**: https://arxiv.org/abs/2402.08166
- **Relevance**: Uses energy formulation for physics constraints (similar philosophy to your work)
- **Method**: Enforces energy conservation in neural operator architecture
- **Key Difference**: Uses energy for **physics constraints**, not for **uncertainty quantification**
- **Benchmark**: Hamiltonian systems, wave equations
- **Why cite**: Related energy-based approach but different application

### 9. **Variational Inference Neural Operators (VINO)** (2025)
- **Link**: https://arxiv.org/abs/2501.10257
- **Method**: Variational energy formulation for operator learning
- **Relevance**: Energy-based framework for learning operators
- **Application**: PDE solution with energy minimization
- **Why cite**: Another energy-based approach to operator learning

### 10. **Improved Training of Physics-Informed Networks Using Energy-Based Priors** (OpenReview)
- **Link**: https://openreview.net/forum?id=zqkfJA6R1-r
- **Relevance**: Uses EBM priors for physics-informed networks
- **Key Result**: EBM priors expedite PINN convergence by 10x
- **Difference**: Applied to PINNs not neural operators, used as prior not for UQ
- **Why cite**: Related use of EBMs in scientific ML

## Priority 4: Other Relevant Baselines

### 11. **Adversarially Trained Neural Operators with Generative Models** (May 2025)
- **Link**: https://arxiv.org/abs/2505.23106
- **Method**: GAN-based training for neural operators
- **Relevance**: Generative modeling approach to neural operators
- **Key Result**: 15x reduction in energy-spectrum error for turbulent flows
- **Application**: Overcomes oversmoothing in standard L2-trained neural operators
- **Why cite**: Alternative generative approach (GAN vs your EBM)

### 12. **Neural Operator induced Gaussian Process (NOGaP)** (2024)
- **Link**: https://www.sciencedirect.com/science/article/abs/pii/S0045782524005218
- **Journal**: Computer Methods in Applied Mechanics and Engineering
- **Method**: Combines neural operators with GP framework
- **Relevance**: Probabilistic framework with uncertainty quantification
- **Why cite**: GP-based alternative for UQ in operator learning

## Comparison Strategy

### What Metrics to Report

Based on these papers, you should report:

1. **Accuracy Metrics** (to match PDEBench):
   - Relative L2 error: ||u_pred - u_true||₂ / ||u_true||₂
   - Mean Squared Error (MSE)
   - R² coefficient

2. **Uncertainty Calibration** (to match PNO, Conformalized-DeepONet):
   - Calibration error (most important!)
   - Continuous Ranked Probability Score (CRPS)
   - Coverage at different confidence levels (e.g., 50%, 90%, 95%)
   - Sharpness (average prediction interval width)

3. **Physics Compliance** (your unique strength):
   - PDE residual: ||R(u_pred)||₂ where R is PDE residual operator
   - Boundary condition violation
   - Conservation error (if applicable)

4. **Computational Cost**:
   - Training time
   - Inference time per sample
   - Number of MCMC steps needed

### Expected Performance Summary

Based on the papers, here's honest expectation for your FNO-EBM:

| Metric              | PDEBench FNO | PNO (2025) | Conformalized-DeepONet | Your FNO-EBM (Expected) |
|---------------------|--------------|------------|------------------------|-------------------------|
| Relative L2 Error   | 0.05-0.10    | 0.04-0.09  | 0.048                  | **0.05-0.10** (comparable) |
| Calibration Error   | N/A          | ~0.05      | <0.05                  | **~0.05-0.08** (competitive) |
| Coverage (95%)      | N/A          | ~92%       | ~95%                   | **~90-94%** (good) |
| Physics Residual    | Moderate     | Moderate   | High                   | **Low** (your strength) |
| Inference Time      | Fast         | Fast       | Fast                   | **Slow** (MCMC sampling) |

### Your Honest Positioning

**What to emphasize**:
1. ✅ **Physics compliance**: Lower PDE residuals than all baselines
2. ✅ **Interpretable uncertainty**: Energy-based framework is more interpretable than dropout/ensembles
3. ✅ **Flexible architecture**: Multiple EBM backbones (MLP/KAN/GNN)
4. ✅ **Modular design**: Can swap FNO for other neural operators

**What to acknowledge**:
1. ⚠️ **No coverage guarantees**: Unlike Conformalized-DeepONet, no theoretical guarantees
2. ⚠️ **Higher computational cost**: MCMC sampling is slower than dropout/ensemble
3. ⚠️ **Accuracy comparable not better**: Similar MSE/L2 error to baselines
4. ⚠️ **Calibration competitive not superior**: Similar calibration to PNO

**Your niche**: *"Physics-informed probabilistic operator learning when physical constraints are critical and interpretable uncertainty is valued, even at higher computational cost."*


# Sanity checks result  (Sun 19 Oct 2025)

● Short Answer: YES, but with caveats

  Your current approach can work, but it will learn something different from what you might expect. Let me explain what actually happens:

  ---
  What Your Current Implementation Actually Does

  Stage 1: Train FNO Alone

  # Learns: f(x) → g(x)
  u_pred = FNO(x)
  loss = MSE(u_pred, u_true) + λ * physics_loss

  FNO learns: The mean/mode of the distribution p(u|x) from data
  - If data has noise/uncertainty, FNO learns the average behavior
  - This is your "best guess" solution

  Stage 2: Train EBM on V(u,x) Alone

  # Current: trainer.py lines 391-426
  pos_energy = V(u_true, x)  # Energy of real data
  neg_energy = V(u_sampled, x)  # Energy of generated samples
  loss = E_data[V] - E_model[V]

  V(u,x) learns: The data distribution directly
  - V(u,x) is low when u looks like real data
  - V(u,x) is high when u looks unrealistic
  - Does NOT know about FNO during training!

  Stage 3: Inference with Combined Energy

  # inference.py line 22
  E(u,x) = 0.5 * ||u - u_FNO||² + V(u,x)

  Sampling from this gives:
  - Pulled toward FNO solution by ||u - u_FNO||²
  - Shaped by data realism from V(u,x)
  - The quadratic term at inference only acts as a regularizer

  ---
  Does It Capture Uncertainty? YES, Kind Of...

● What V(u,x) Actually Learns:

  Since V is trained independently on real data:

  # It learns to assign low energy to:
  V(u_realistic, x) → low

  # High energy to:
  V(u_unrealistic, x) → high

  This means V(u,x) learns:
  1. ✅ Physical plausibility: Solutions that violate physics get high energy
  2. ✅ Data manifold: Solutions consistent with training data get low energy
  3. ✅ Implicit uncertainty: Regions with high data density get low V

  Example:
  - If your training data has noise in region A
  - V learns that multiple solutions near A are all "realistic" (low energy)
  - At inference, Langevin will sample from this region → captures uncertainty!

  ---
  The Catch: What Happens at Inference

  Your combined energy at inference:
  E(u,x) = ||u - μ_FNO||²/(2σ²) + V(u,x)
         = Gaussian_pull + Data_realism

  Three Scenarios:

  Scenario 1: FNO is Accurate, Data has Low Noise

  u_FNO ≈ u_true
  V(u_true, x) is LOW
  Result:
  - Langevin samples cluster tightly around FNO
  - Low uncertainty ✅ Correct!

  Scenario 2: FNO is Accurate, Data has High Noise

  u_FNO ≈ E[u_true]  (mean of noisy data)
  V learns that region around u_FNO has many realistic solutions
  Result:
  - Langevin samples spread around FNO
  - High uncertainty ✅ Correct!

  Scenario 3: FNO is Wrong, But V Learned True Distribution

  u_FNO is biased
  V(u_correct, x) is LOW
  V(u_FNO, x) might be HIGH
  Result:
  - Quadratic term pulls toward wrong u_FNO
  - V term pulls toward correct region
  - Competition between two forces ⚠️
  - Outcome depends on relative strength (implicit σ²)

  ---
  Practical Analysis: Will It Work?

● When Your Approach Works Well:

  ✅ 1. FNO learns good mean prediction
  - If FNO is well-trained, it gives good starting point

  ✅ 2. Data uncertainty is in the training set
  - If your training data includes noisy observations
  - V will learn that this noise region is realistic

  ✅ 3. V has enough capacity
  - Complex V (like your KAN/GNN variants) can capture intricate uncertainty patterns

  ✅ 4. Langevin sampling works
  - With proper step size and iterations
  - Can explore the energy landscape effectively

  When It Might Fail:

  ❌ 1. FNO is systematically biased
  - Quadratic term pulls samples to wrong location
  - V can't overcome strong quadratic pull

  ❌ 2. Training data doesn't contain uncertainty
  - If data is deterministic (no noise/variations)
  - V learns "only one solution is realistic"
  - Won't capture true uncertainty

  ❌ 3. V overfit to training data
  - Might reject valid solutions outside training distribution
  - Poor generalization

  ❌ 4. Implicit σ² is wrong
  - The coefficient 0.5 in 0.5 * ||u - u_FNO||² assumes σ²=1
  - If actual uncertainty is σ²=10, quadratic term too strong
  - If actual uncertainty is σ²=0.01, quadratic term too weak

  ---
  Key Insight: The Implicit Coupling

  Even though you train FNO and EBM separately, they implicitly couple at inference through:

  E(u,x) = 0.5 * ||u - u_FNO(x)||² + V(u,x)
           ↑                          ↑
           Learned mean           Learned data distribution

  This is similar to (but not exactly):
  - A Gaussian with learned mean (FNO)
  - And learned non-Gaussian corrections (V)

  It's like saying:
  p(u|x) ∝ exp(-0.5||u-μ||²) * exp(-V(u,x))
         = Gaussian_base   * Non-Gaussian_correction

  This can work if:
  - V learns the right "correction" to Gaussian
  - FNO learns the right mean μ
  - The implicit variance (from coefficient 0.5) is reasonable

  ---
  Concrete Example

  Imagine you're solving Poisson equation with noisy boundary conditions:

  Training data:
  Input: f(x) with noise
  Output: u(x) with corresponding noise propagation

  What happens:

  1. FNO learns: E[u|f] - average solution for average input
  2. V learns: "realistic solutions have smooth gradients, satisfy BC approximately, etc."
  3. At inference:
    - Start near FNO prediction
    - Langevin explores nearby "realistic" solutions
    - Samples spread where V is flat (uncertain regions)
    - Samples concentrate where V is sharp (certain regions)

  Result: ✅ Captures both solution and uncertainty!

  ---
  My Verdict

  Will it work?

  YES, with high probability, IF:

  1. ✅ Your FNO is reasonably accurate (doesn't need to be perfect)
  2. ✅ Your training data contains the uncertainty you want to model
  3. ✅ V has enough capacity (your KAN/GNN/FNO-KAN models likely do)
  4. ✅ Langevin sampling converges (use enough steps)
  5. ⚠️ You tune the implicit σ² (the 0.5 coefficient) - this is critical!

  What uncertainty will it capture?

  It will capture aleatoric uncertainty (data noise) well, but epistemic uncertainty (model uncertainty in unseen regions) less reliably.

  Should you change anything?

  Optional improvements (not strictly necessary):

  1. Make σ² explicit and tunable:
  # Instead of:
  E = 0.5 * ||u - u_FNO||² + V(u,x)

  # Use:
  E = ||u - u_FNO||² / (2*sigma²) + V(u,x)
  # Where sigma² is a hyperparameter or learned

  2. Add validation: Check that:
    - Samples actually vary in uncertain regions
    - Samples concentrate in certain regions
    - Predicted std correlates with true error
  3. Monitor during training:
    - Are Langevin samples escaping FNO too much? (σ² too high)
    - Are Langevin samples stuck at FNO? (σ² too low)

  ---

● Bottom Line

  Yes, your current implementation can work!

  The separate training of FNO and EBM, combined with Langevin sampling from the joint energy landscape, is a valid approach that can capture both the solution and uncertainty
  present in the data.

  It's not the most theoretically elegant approach, but it's:
  - ✅ Pragmatic
  - ✅ Likely to work in practice
  - ✅ Easier to train (staged training is more stable)
  - ⚠️ Requires careful tuning of the quadratic term's strength

  My recommendation: Try it as-is, monitor the results, and only add complexity if you see specific failure modes.

  Want me to add diagnostic tools to help you verify it's working correctly?



# New Problem formulation

Part 1: Reformulated Problem Statement

  🔴 OLD (Problematic) Problem Statement:

  "Capture theoretical solution computed via FEM over PDE operators using FNO, along with uncertainty due to data collection shifts"

  Problems:
  - "Theoretical solution" implies you're solving PDEs (you're not, you're learning from data)
  - "Data collection shifts" is vague and suggests systematic bias
  - Doesn't clarify what uncertainty means
  - Conflates multiple concepts

  ---
  ✅ NEW (Honest) Problem Statement:

  Core Research Question:

  "How can we learn nonlinear operator mappings from noisy observations while providing calibrated uncertainty estimates that respect physical constraints?"

  Specific Problem Formulation:

  Given:
  - Noisy paired observations: {(fᵢ, uᵢ + εᵢ)}ᴺᵢ₌₁ where
    - fᵢ: Input function (e.g., forcing term, boundary conditions)
    - uᵢ: True solution to operator G: f → u
    - εᵢ: Observation noise (potentially heteroscedastic)
  - Physical constraints: PDE residuals R(u, f) ≈ 0

  Goal:
  Learn a probabilistic operator G̃_θ: f → p(u|f) such that:
  1. Accuracy: Mean prediction E[u|f] approximates true operator G(f)
  2. Uncertainty: Predictive distribution captures observation noise and model uncertainty
  3. Physics: Samples from p(u|f) satisfy physical constraints
  4. Calibration: Predicted uncertainty correlates with actual prediction error

  Non-goals (to be explicit about limitations):
  - ❌ Solving PDEs from scratch (we learn from data)
  - ❌ Handling arbitrary distribution shifts (out of scope)
  - ❌ Beating all existing methods on all metrics

  ---
  Proposed Solution (Your FNO-EBM):

  Architecture:
  p(u|f) ∝ exp(-E(u,f))

  where E(u,f) = ½||u - μ_FNO(f)||² / σ² + V_EBM(u,f)
                ︸━━━━━━━━━━━━━━━━━━━━━┛   ︸━━━━━━━━━━━┛
                Anchoring to learned mean   Learned deviation

  Training Strategy:
  - Stage 1 (FNO): Learn mean operator with physics constraints
  L_FNO = MSE(μ_FNO(f), u_obs) + λ·||R(μ_FNO(f), f)||²
  - Stage 2 (EBM): Learn uncertainty structure via contrastive divergence
  L_EBM = E_data[V(u,f)] - E_model[V(u',f)]
  - where u' sampled via Langevin dynamics

  Inference:
  - Sample from p(u|f) via Langevin MCMC
  - Compute statistics: mean, std, quantiles

  ---
  Why This Reformulation is Better:

  | Aspect   | Old                                  | New                                     |
  |----------|--------------------------------------|-----------------------------------------|
  | Clarity  | Vague "capture theoretical solution" | Precise probabilistic operator learning |
  | Honesty  | Implies solving PDEs                 | Clear: learning from observations       |
  | Scope    | Overpromises (distribution shift)    | Realistic scope (noisy data + UQ)       |
  | Novelty  | Unclear contribution                 | Explicit: physics + energy-based UQ     |
  | Testable | Hard to validate                     | Clear evaluation criteria               |

  ---

● Part 2: Experimental Design - Comparison Framework

  📋 Overview of Comparison

  Let me fetch the papers to understand their exact methods:

● Fetch(https://openreview.net/forum?id=orKA6gJwlB)
  ⎿  Received 40.5KB (200 OK)

● Fetch(https://arxiv.org/abs/2502.12902)
  ⎿  Received 48.4KB (200 OK)

● Let me work from what we know from your literature review and standard practices in this field.

  ---
  🔬 Experimental Design: Three-Way Comparison

  Methods to Compare:

  | Method                     | Architecture          | Uncertainty Source        | Training                       | Physics               |
  |----------------------------|-----------------------|---------------------------|--------------------------------|-----------------------|
  | Paper #1: Energy Score FNO | FNO + Dropout         | Stochastic forward passes | Single-stage with energy score | No explicit physics   |
  | Paper #2: PNO              | FNO + Generative head | Learned distributions     | Energy-based scoring           | Implicit in data      |
  | Your FNO-EBM               | FNO + EBM             | Langevin sampling         | Two-stage (FNO→EBM)            | Explicit physics loss |

  ---
  📊 Benchmark Problems (Proposed)

  To make fair comparisons, use standard operator learning benchmarks:

  Problem 1: Darcy Flow (Standard Benchmark)

  -∇·(a(x)∇u(x)) = f(x)  in Ω=[0,1]²
  u(x) = 0                on ∂Ω

  Input:  a(x) ~ GP (permeability field)
  Output: u(x) (pressure field)

  Why this problem:
  - ✅ Standard in FNO papers
  - ✅ Has analytic solutions for validation
  - ✅ Can add controlled noise
  - ✅ Clear physics (Darcy's law)

  Data generation:
  ## Pseudocode
  N_train = 1000
  N_test = 200

  for i in 1:N_train:
      a_i = sample_from_GP(lengthscale=0.1)
      u_i = solve_darcy(a_i)  # Using FEM
      u_noisy_i = u_i + ε_i   # ε ~ N(0, σ²) heteroscedastic

      save (a_i, u_noisy_i)

  Noise models to test:
  1. Homoscedastic: σ² = 0.01 (constant)
  2. Heteroscedastic: σ²(x) = 0.001 + 0.02·|u(x)| (depends on solution)
  3. Spatially correlated: ε ~ GP(0, K) (realistic sensor noise)

  ---
  Problem 2: Burgers' Equation (Nonlinear)

  ∂u/∂t + u·∂u/∂x = ν·∂²u/∂x²
  u(x,0) = u₀(x)

  Why this problem:
  - ✅ Nonlinear (tests model capacity)
  - ✅ Time-dependent (operator maps IC → solution)
  - ✅ Can develop shocks (challenging)
  - ✅ Used in Paper #2

  ---
  Problem 3: Navier-Stokes (2D, Realistic)

  ∂u/∂t + (u·∇)u = -∇p + ν∇²u
  ∇·u = 0

  Why this problem:
  - ✅ Highly relevant application
  - ✅ Complex dynamics
  - ✅ Tests scalability
  - ⚠️ More expensive (use if time permits)

  ---
  📏 Evaluation Metrics

  Category 1: Accuracy Metrics

  ## 1. Mean Squared Error (point prediction)
  def mse(u_pred_mean, u_true):
      return np.mean((u_pred_mean - u_true)**2)

  ## 2. Relative L2 Error
  def relative_l2(u_pred_mean, u_true):
      return np.linalg.norm(u_pred_mean - u_true) / np.linalg.norm(u_true)

  ---
  Category 2: Uncertainty Calibration Metrics (CRITICAL)

  ## 3. Calibration Error (most important!)
  def calibration_error(u_samples, u_true, num_bins=10):
      """
      Check if predicted confidence matches actual coverage
      
      For each confidence level α, check if true solution
      falls within predicted α-credible interval α% of the time
      """
      alphas = np.linspace(0.05, 0.95, num_bins)
      coverage_errors = []

      for alpha in alphas:
          # Predicted credible interval
          lower = np.quantile(u_samples, (1-alpha)/2, axis=0)
          upper = np.quantile(u_samples, (1+alpha)/2, axis=0)

          # Actual coverage
          actual_coverage = np.mean((u_true >= lower) & (u_true <= upper))

          # Calibration error
          coverage_errors.append(abs(actual_coverage - alpha))

      return np.mean(coverage_errors)

  ## 4. Negative Log-Likelihood (proper scoring rule)
  def nll(u_samples, u_true):
      """
      Estimate log p(u_true | u_samples) via kernel density
      """
      from scipy.stats import gaussian_kde

      kde = gaussian_kde(u_samples)
      log_prob = kde.logpdf(u_true)
      return -np.mean(log_prob)

  ## 5. Continuous Ranked Probability Score (CRPS)
  def crps(u_samples, u_true):
      """
      Proper scoring rule for probabilistic forecasts
      Lower is better
      """
      n_samples = len(u_samples)

      # E[|X - y|]
      term1 = np.mean(np.abs(u_samples - u_true))

      # 0.5 * E[|X - X'|]
      term2 = 0
      for i in range(n_samples):
          for j in range(i+1, n_samples):
              term2 += np.abs(u_samples[i] - u_samples[j])
      term2 = term2 / (n_samples * (n_samples - 1) / 2)

      return term1 - 0.5 * term2

  ---
  Category 3: Physics Compliance Metrics

  ## 6. Physics Residual (for Darcy flow)
  def physics_residual(u_samples, a, f):
      """
      Measure how well samples satisfy PDE
      """
      residuals = []
      for u in u_samples:
          # Compute -∇·(a∇u) - f
          R = compute_darcy_residual(u, a, f)
          residuals.append(np.mean(R**2))

      return {
          'mean': np.mean(residuals),
          'std': np.std(residuals),
          'max': np.max(residuals)
      }

  ## 7. Boundary Condition Violation
  def bc_violation(u_samples):
      """
      Measure boundary condition errors
      """
      boundary_errors = []
      for u in u_samples:
          bc_error = np.mean(u[boundary_mask]**2)  # Should be zero
          boundary_errors.append(bc_error)

      return np.mean(boundary_errors)

  ---
  Category 4: Computational Metrics

  ## 8. Training Time
  training_time_seconds

  ## 9. Inference Time (per sample)
  inference_time_per_sample

  ## 10. Memory Usage
  peak_memory_mb

  ---
  🧪 Experimental Protocol

  Experiment 1: Clean Data Baseline

  Setup:
  - Darcy flow with NO noise (σ = 0)
  - 1000 training samples
  - 200 test samples

  Goal: Establish upper bound on accuracy

  Metrics: MSE, Relative L2, Physics Residual

  Expected result:
  - All methods should perform similarly well
  - Tests if methods work in ideal case

  ---
  Experiment 2: Noise Robustness

  Setup:
  - Darcy flow with increasing noise levels: σ ∈ {0.01, 0.05, 0.1, 0.2}
  - Same train/test split

  Goal: Test which method handles noise better

  Metrics: All accuracy + uncertainty metrics

  Hypotheses:
  - Paper #1: Should handle noise via dropout
  - Paper #2: Should learn noise distribution
  - Your FNO-EBM: Should capture noise via EBM

  Key question: Does your two-stage training help or hurt?

  ---
  Experiment 3: Uncertainty Calibration (MOST IMPORTANT)

  Setup:
  - Darcy flow with heteroscedastic noise
  - Noise varies spatially: high near boundaries, low in center

  Goal: Which method gives CALIBRATED uncertainty?

  Metrics:
  - Calibration Error (primary metric)
  - CRPS
  - Coverage plots (confidence vs actual coverage)

  Critical plots to generate:

  ## Calibration curve
  plt.figure(figsize=(8,6))
  plt.plot(predicted_confidence, actual_coverage, 'o-')
  plt.plot([0,1], [0,1], 'k--', label='Perfect calibration')
  plt.xlabel('Predicted Confidence Level')
  plt.ylabel('Actual Coverage')
  plt.title('Calibration Curve')
  plt.legend()

  ## Uncertainty heatmap
  fig, axes = plt.subplots(1, 3, figsize=(15,4))
  axes[0].imshow(u_true)
  axes[0].set_title('True Solution')
  axes[1].imshow(u_pred_mean)
  axes[1].set_title('Predicted Mean')
  axes[2].imshow(u_pred_std)
  axes[2].set_title('Predicted Std (should be high near boundaries)')

  Success criterion:
  - Calibration error < 0.05
  - High uncertainty where noise is high
  - Low uncertainty where noise is low

  ---
  Experiment 4: Physics Constraint Comparison

  Setup:
  - Train all three methods on Darcy flow
  - For Paper #1 & #2: Add physics loss (fair comparison)
  - For Your FNO-EBM: Ablation with/without physics loss

  Goal: Quantify benefit of explicit physics

  Metrics:
  - Physics Residual
  - BC Violation
  - Samples should satisfy PDE better

  Expected result:
  - Your method should have lowest physics residual
  - Demonstrates value of two-stage + physics

  ---
  Experiment 5: Ablation Study (For Your Method)

  Variants to test:

  | Variant              | FNO | EBM | Physics Loss |
  |----------------------|-----|-----|--------------|
  | FNO-only             | ✓   | ✗   | ✓            |
  | EBM-only             | ✗   | ✓   | ✗            |
  | FNO-EBM (no physics) | ✓   | ✓   | ✗            |
  | FNO-EBM (full)       | ✓   | ✓   | ✓            |

  Goal: Show each component contributes

  ---

● 📊 Summary Comparison Table

  What You Need to Show:

  | Claim                                     | Evidence Needed                     | Experiment |
  |-------------------------------------------|-------------------------------------|------------|
  | "Our method is accurate"                  | MSE comparable to baselines         | Exp 1, 2   |
  | "Our method gives calibrated uncertainty" | Low calibration error               | Exp 3 ⭐    |
  | "Physics constraints help"                | Lower residuals than non-physics    | Exp 4      |
  | "Two-stage training is beneficial"        | Ablation shows improvement          | Exp 5      |
  | "Competitive with SOTA"                   | Better on ≥1 metric vs Papers #1,#2 | All exps   |

  ---
  🎯 Honest Assessment of Expected Results

  Based on the methods, here's what I honestly expect:

  Accuracy (MSE, L2):

  - Paper #1: ⭐⭐⭐⭐ (very good)
  - Paper #2: ⭐⭐⭐⭐⭐ (slightly better, integrated training)
  - Your FNO-EBM: ⭐⭐⭐⭐ (comparable)

  Why: Your two-stage training might slightly hurt accuracy compared to joint training.

  Uncertainty Calibration:

  - Paper #1: ⭐⭐⭐ (good but dropout-based uncertainty can be overconfident)
  - Paper #2: ⭐⭐⭐⭐ (good, energy score is proper scoring rule)
  - Your FNO-EBM: ⭐⭐⭐⭐ (potentially better if tuned right)

  Why: Explicit energy modeling CAN give better calibration, but depends on σ² tuning.

  Physics Compliance:

  - Paper #1: ⭐⭐ (no explicit physics)
  - Paper #2: ⭐⭐⭐ (implicit in data)
  - Your FNO-EBM: ⭐⭐⭐⭐⭐ (best, explicit physics loss)

  Why: This is your strongest advantage.

  Computational Cost:

  - Paper #1: ⭐⭐⭐⭐⭐ (fast, dropout)
  - Paper #2: ⭐⭐⭐⭐ (moderate)
  - Your FNO-EBM: ⭐⭐⭐ (slower, MCMC sampling)

  Why: Langevin dynamics is expensive.

  ---
  💡 Your Selling Points (Honest)

  Based on this analysis, your honest narrative should be:

  "We present FNO-EBM, a physics-informed probabilistic operator learning framework that provides calibrated uncertainty estimates while respecting physical constraints.

  Compared to existing methods [Papers #1, #2], our approach:
  - ✅ Explicitly enforces physics through PDE residuals
  - ✅ Provides interpretable energy-based uncertainty
  - ✅ Achieves better physics compliance (X% lower residual)
  - ✅ Competitive uncertainty calibration
  - ⚠️ At the cost of higher computational expense

  We demonstrate on [Darcy flow, Burgers] that our method is particularly advantageous when physical constraints are critical, even with moderate training data."

  ---

● 🛠️ Implementation Roadmap

  Would you like me to implement:

  1. Evaluation framework - All the metrics above as Python code
  2. Darcy flow dataset generator - Clean + noisy variants
  3. Baseline implementations - Paper #1 style (FNO + dropout) for fair comparison
  4. Calibration diagnostic tools - Plots and statistical tests
  5. Experiment runner - Automated comparison across all methods

  Let me know which you want first, and I'll create production-quality code.

  ---
  Final Integrity Check ✅

  With this reformulated approach:

  ✅ Problem statement is honest - No overclaiming
  ✅ Experiments are rigorous - Fair comparisons with SOTA
  ✅ Metrics are standard - Calibration, CRPS, physics residual
  ✅ Expected results realistic - Not claiming to beat everything
  ✅ Novelty is clear - Physics + energy-based UQ + two-stage

  This is publishable, honest research. You should feel confident defending it.


# Data Format

Let me give you concrete examples.

  ---
  Data Format for FNO-EBM Training

  Problem Setup

  Let's use Darcy Flow as example (standard benchmark):

  PDE: -∇·(a(x)∇u(x)) = f(x) in Ω = [0,1]²

  Goal: Learn operator G: a → u that maps permeability field to pressure field

  ---
  Stage 1: FNO Training Data

  Input Format: x

  FNO learns operator mappings, so input contains both spatial coordinates AND input function values.

  Shape: (batch, nx, ny, 3)

  Content:
  x[:, :, :, 0] = x-coordinates (meshgrid)
  x[:, :, :, 1] = y-coordinates (meshgrid)
  x[:, :, :, 2] = input function values (e.g., permeability a(x,y))

  Example for one sample (nx=64, ny=64):
  # Single training sample
  x_coords = np.linspace(0, 1, 64)  # [0, 0.015625, 0.03125, ..., 1]
  y_coords = np.linspace(0, 1, 64)

  X_grid, Y_grid = np.meshgrid(x_coords, y_coords, indexing='ij')
  # X_grid.shape = (64, 64)
  # Y_grid.shape = (64, 64)

  # Input function (permeability field) - this is what varies per sample
  a_field = generate_permeability()  # Shape: (64, 64)
  # Example: a_field could be from Gaussian Process, or real measurements

  # Combine into input tensor
  x = np.stack([X_grid, Y_grid, a_field], axis=-1)
  # x.shape = (64, 64, 3)

  Concrete numbers for one grid point (i=32, j=32):
  x[32, 32, :] = [0.5, 0.5, 2.47]
                   ↑    ↑     ↑
                   x    y    a(x,y)

  Output Format: u (Ground Truth)

  FNO predicts the solution field.

  Shape: (batch, nx, ny, 1)

  Content:
  u[:, :, :, 0] = solution field u(x,y) - pressure/temperature/etc.

  How you get this:
  # Option 1: Solve PDE numerically (FEM/FDM)
  u_true = solve_darcy_equation(a_field)  # Shape: (64, 64)

  # Option 2: From simulation/measurements
  u_measured = load_from_simulation()

  # Add observation noise (important for UQ!)
  noise = np.random.normal(0, 0.01, size=u_true.shape)
  u_noisy = u_true + noise

  # Format for training
  u = u_noisy.reshape(64, 64, 1)  # Shape: (64, 64, 1)

  Concrete numbers at same point:
  u[32, 32, 0] = 0.873
                 ↑
              solution value at (x=0.5, y=0.5)

  FNO Training Batch

  Batch of N=1000 samples:

  x_batch.shape = (1000, 64, 64, 3)
      [:, :, :, 0] = x-coords (same for all samples)
      [:, :, :, 1] = y-coords (same for all samples)
      [:, :, :, 2] = permeability (DIFFERENT for each sample)

  u_batch.shape = (1000, 64, 64, 1)
      [:, :, :, 0] = pressure solutions (DIFFERENT for each sample)

  FNO loss:
  u_pred = FNO(x_batch)  # Shape: (1000, 64, 64, 1)
  loss = MSE(u_pred, u_batch) + λ * physics_loss(u_pred, x_batch)

  ---
  Stage 2: EBM Training Data

  EBM learns to assign low energy to realistic (solution, input) pairs and high energy to unrealistic pairs.

  Input Format for EBM: (u, x)

  EBM takes both the solution candidate AND the input.

  Shape of inputs:
  - u: (batch, nx, ny, 1) - candidate solution
  - x: (batch, nx, ny, 3) - input coordinates + function

  Inside EBM forward:
  def forward(self, u, x):
      # u.shape = (batch, 64, 64, 1)
      # x.shape = (batch, 64, 64, 3)

      # Concatenate along feature dimension
      combined = torch.cat([u, x], dim=-1)
      # combined.shape = (batch, 64, 64, 4)

      # Process to get scalar energy
      energy = self.network(combined)
      # energy.shape = (batch,)  <- SINGLE NUMBER per sample

      return energy

  Training Data for EBM

  Positive samples (real data pairs):
  x_pos = x_batch  # Same as FNO input: (1000, 64, 64, 3)
  u_pos = u_batch  # Real solutions: (1000, 64, 64, 1)

  # These are the SAME x and u you used to train FNO!

  Negative samples (generated via Langevin):
  # Start from positive samples
  u_neg = u_pos.clone()  # (1000, 64, 64, 1)

  # Run Langevin dynamics to generate "unrealistic" solutions
  for k in range(num_mcmc_steps):
      energy = EBM(u_neg, x_pos)
      grad_u = compute_gradient(energy, u_neg)

      noise = torch.randn_like(u_neg) * noise_scale
      u_neg = u_neg - step_size * grad_u + noise

  # u_neg.shape still (1000, 64, 64, 1)

  EBM Training Loss:
  # Positive phase: energy of real data
  E_pos = EBM(u_pos, x_pos)  # Shape: (1000,)

  # Negative phase: energy of generated samples
  E_neg = EBM(u_neg, x_pos)  # Shape: (1000,)

  # Contrastive divergence loss
  loss = E_pos.mean() - E_neg.mean()

  ---
  Concrete Example: One Training Sample

  Darcy Flow Example

  Problem: Given permeability field a(x,y), predict pressure field u(x,y)

  Training sample #42:

  FNO Input x[42]:

  Grid: 64×64

  x[42, :, :, 0] = [[0.0,    0.015625, 0.03125, ..., 1.0],
                    [0.0,    0.015625, 0.03125, ..., 1.0],
                    ...
                    [0.0,    0.015625, 0.03125, ..., 1.0]]
                    ↑ x-coordinates (constant across y)

  x[42, :, :, 1] = [[0.0,    0.0,      0.0,     ..., 0.0],
                    [0.015625, 0.015625, ..., 0.015625],
                    ...
                    [1.0,    1.0,      1.0,     ..., 1.0]]
                    ↑ y-coordinates (constant across x)

  x[42, :, :, 2] = [[1.2,    1.5,      2.1,     ..., 0.9],
                    [1.8,    2.3,      1.7,     ..., 1.2],
                    ...
                    [0.7,    1.1,      1.9,     ..., 1.6]]
                    ↑ Permeability values a(x,y) - UNIQUE to sample #42

  FNO Target u[42]:

  u[42, :, :, 0] = [[0.0,    0.12,     0.23,    ..., 0.0],
                    [0.15,   0.34,     0.41,    ..., 0.18],
                    ...
                    [0.0,    0.09,     0.17,    ..., 0.0]]
                    ↑ Pressure solution from solving PDE with noise added

  EBM Input (same data, used differently):

  # Positive pair (real)
  u_positive = u[42]  # Shape: (64, 64, 1) - the TRUE solution
  x_input = x[42]     # Shape: (64, 64, 3) - the permeability + coords

  # EBM assigns low energy
  E_positive = EBM(u_positive, x_input)  # Scalar, e.g., -2.3

  # Negative pair (generated)
  u_negative = langevin_sample(...)  # Shape: (64, 64, 1) - UNREALISTIC solution
  # Same input!
  x_input = x[42]  # Shape: (64, 64, 3)

  # EBM assigns high energy
  E_negative = EBM(u_negative, x_input)  # Scalar, e.g., +1.7

  ---
  Key Insights

  1. Same Data, Different Usage

  FNO Training:
      Input:  x (coordinates + permeability)
      Output: u (pressure)
      Goal:   Learn u = FNO(x)

  EBM Training:
      Input:  (u, x) pairs
      Output: Energy scalar
      Goal:   E(u_real, x) < E(u_fake, x)

  2. Why x includes coordinates?

  For FNO: Needs to know where in space each value is
  For EBM: Needs to check if solution is realistic for THAT specific input

  3. Noise is Critical for UQ

  # Without noise:
  u_clean = solve_PDE(a)
  # EBM learns: "only ONE solution is realistic"
  # Uncertainty = zero (wrong!)

  # With noise:
  u_noisy = solve_PDE(a) + ε
  # EBM learns: "solutions in a REGION are realistic"
  # Uncertainty = spread of that region (correct!)

  4. Data Generation Pipeline

  1. Generate input functions (permeabilities):
     a₁, a₂, ..., aₙ ~ GP or physical model

  2. Solve PDE for each:
     uᵢ = solve(aᵢ) using FEM/FDM

  3. Add observation noise:
     uᵢ_obs = uᵢ + εᵢ where εᵢ ~ N(0, σ²)

  4. Create grid coordinates:
     X, Y = meshgrid(...)

  5. Package for FNO:
     x[i] = [X, Y, aᵢ]  # Shape: (nx, ny, 3)
     u[i] = uᵢ_obs      # Shape: (nx, ny, 1)

  6. Same data used for EBM!

  ---
  Dimensions Summary

  | Quantity          | Shape          | Example           | Content                           |
  |-------------------|----------------|-------------------|-----------------------------------|
  | Single FNO input  | (nx, ny, 3)    | (64, 64, 3)       | [x_coords, y_coords, input_field] |
  | Single FNO output | (nx, ny, 1)    | (64, 64, 1)       | [solution_field]                  |
  | FNO batch input   | (B, nx, ny, 3) | (1000, 64, 64, 3) | Batch of inputs                   |
  | FNO batch output  | (B, nx, ny, 1) | (1000, 64, 64, 1) | Batch of solutions                |
  | EBM input u       | (B, nx, ny, 1) | (1000, 64, 64, 1) | Solution candidates               |
  | EBM input x       | (B, nx, ny, 3) | (1000, 64, 64, 3) | Context (same as FNO)             |
  | EBM output        | (B,)           | (1000,)           | Scalar energy per sample          |

  ---
  What About Different Problems?

  Burgers Equation (1D time evolution)

  Input x.shape = (batch, nt, nx, 2)
      [:, :, :, 0] = time coordinate
      [:, :, :, 1] = initial condition u₀(x)

  Output u.shape = (batch, nt, nx, 1)
      [:, :, :, 0] = solution u(t,x)

  Navier-Stokes (2D velocity field)

  Input x.shape = (batch, nx, ny, 3)
      [:, :, :, 0] = x-coordinate
      [:, :, :, 1] = y-coordinate
      [:, :, :, 2] = forcing term f(x,y)

  Output u.shape = (batch, nx, ny, 2)
      [:, :, :, 0] = velocity u_x(x,y)
      [:, :, :, 1] = velocity u_y(x,y)

  ---
  Critical: Noise Levels for UQ

  Your noise level determines how much uncertainty EBM can learn:

  Too little noise (σ = 0.001):
  - EBM learns very narrow energy wells
  - Predicted uncertainty too small
  - Underconfident

  Too much noise (σ = 0.1):
  - EBM learns very flat energy landscape
  - Predicted uncertainty too large
  - Overconfident

  Good practice:
  - Start with σ = 0.01 (1% of typical solution magnitude)
  - Use heteroscedastic noise: σ(x,y) = 0.001 + 0.02 * |u(x,y)|
  - Measure noise from real data if available

  ---


# Why the physics loss dominating total loss?

Excellent question! Let me give you the mathematical reasoning for why physics loss dominates when data is normalized.

  ---
  Mathematical Explanation: Why Physics Loss Explodes on Normalized Data

  The Core Problem: Derivative Scaling

  Setup

  Original Darcy equation:
  -∇·(a(x)∇u(x)) = f(x)

  Data normalization:
  u_normalized = (u - u_mean) / u_std

  where for your Darcy flow: u_std ≈ 0.016 (very small!)

  ---
  Derivative Chain Rule Analysis

  Spatial derivatives transform as:

  ∂u_normalized/∂x = ∂/∂x[(u - u_mean)/u_std]
                   = (1/u_std) · ∂u/∂x
                   = (1/0.016) · ∂u/∂x
                   ≈ 62.5 · ∂u/∂x

  Key insight: Derivatives are inversely proportional to the normalization scale!

  ---
  Physics Loss Scaling

  Original physics residual (unnormalized):

  R_original = -∇·(a∇u) - f

  Magnitude: If u ~ O(0.01), then:
  - ∇u ~ O(0.01/L) where L=1 (domain size)
  - ∇·(a∇u) ~ O(0.01)
  - R_original ~ O(0.01)

  Physics loss (unnormalized):
  L_phys_original = ||R_original||² ~ O(0.0001)

  ---
  Normalized physics residual:

  When you normalize u → u_normalized:

  ∇u_normalized = (1/u_std) · ∇u = 62.5 · ∇u

  So the residual becomes:
  R_normalized = -∇·(a∇u_normalized) - f_normalized
               = -∇·(a · 62.5∇u) - f_normalized
               = 62.5 · [-∇·(a∇u)] - f_normalized

  But wait! You're likely not normalizing the coefficient a or source f, only the solution u. This creates asymmetry:

  R_normalized ≈ 62.5 · [-∇·(a∇u)] - f_unnormalized
               └─ scaled up by 1/u_std ─┘

  Magnitude:
  ||R_normalized|| ~ 62.5 × ||R_original|| = 62.5 × 0.01 = 0.625

  Physics loss (normalized):
  L_phys_normalized = ||R_normalized||² 
                    ~ (62.5)² × ||R_original||²
                    = 3906 × L_phys_original
                    = 3906 × 0.0001
                    = 0.39

  Result: Physics loss is ~4000× larger on normalized data!

  ---
  Data Loss Scaling

  Original data loss:

  L_data_original = ||u_pred - u_true||²
                  ~ O(0.016²) = O(0.000256)

  Normalized data loss:

  L_data_normalized = ||u_pred_normalized - u_true_normalized||²

  After normalization: std(u_normalized) = 1, so:
  L_data_normalized ~ O(1²) = O(1)

  But in practice, with good training:
  L_data_normalized ~ 0.0005  (0.05% of variance)

  ---
  The Imbalance: Complete Mathematical Derivation

  Total loss comparison:

  Unnormalized case:
  L_total = L_data + λ · L_phys
          = 0.000256 + λ × 0.0001

  For balance: λ · L_phys ~ L_data
  λ = 0.000256 / 0.0001 = 2.56
  So λ ~ O(1) is reasonable.

  ---
  Normalized case:
  L_total = L_data_normalized + λ · L_phys_normalized
          = 0.0005 + λ × 0.39

  For balance: λ · L_phys ~ L_data
  λ = 0.0005 / 0.39 ≈ 0.0013

  You need λ ~ 0.001 not λ ~ 1!

  ---
  Why Your λ = 0.00009 Works

  Your physics loss is actually even larger than my estimate:

  λ · L_phys = L_data
  0.00009 × L_phys = 0.0005
  L_phys = 0.0005 / 0.00009 ≈ 5.5

  This means your physics residual is VERY large (~5.5), confirming the derivative amplification!

  ---
  General Formula: Optimal Lambda_phys

  Scaling relationship:

  λ_normalized = λ_original × (u_std)²

  For your case:
  λ_normalized = λ_original × (0.016)²
               = λ_original × 0.000256

  Example:
  - If λ_original = 1.0 would work on unnormalized data
  - Then λ_normalized = 1.0 × 0.000256 = 0.000256 on normalized data

  Your λ = 0.00009 is even smaller, suggesting physics residuals are extra large (possibly due to not normalizing a or f).

  ---
  The Root Cause: Three Factors

  1. Derivative Amplification

  ∇u_normalized ~ (1/u_std) · ∇u = 62.5 · ∇u

  2. Quadratic Scaling in Loss

  ||∇u_normalized||² ~ (1/u_std)² · ||∇u||² = 3906 · ||∇u||²

  3. Asymmetric Normalization

  You normalize u but likely not a and f, creating:
  -∇·(a∇u_normalized) - f
  = -(1/u_std)∇·(a∇u) - f
    └─ scaled ─┘   └─ unscaled! ─┘

  This mismatch makes the residual even worse!

  ---
  Solutions (In Order of Best Practice)

  ✅ Option 1: Scale lambda_phys (What you're doing)

  lambda_phys = 0.00009  # Scale down by (u_std)² ≈ 2.5e-4

  Pros: Simple, works
  Cons: Non-intuitive tiny values

  ---
  ✅ Option 2: Compute physics loss on UNNORMALIZED data

  ## Denormalize predictions first
  u_pred_real = u_pred * u_std + u_mean

  ## Then compute physics residual
  residual = compute_residual(u_pred_real, a, f)  # Original scale
  physics_loss = torch.mean(residual**2)

  ## Now lambda can be O(1)
  total_loss = data_loss + 1.0 * physics_loss

  Pros: Intuitive lambda values, physically meaningful
  Cons: Need to denormalize during training

  ---
  ✅ Option 3: Normalize physics residual to match data loss scale

  physics_loss_raw = torch.mean(residual**2)
  physics_loss_normalized = physics_loss_raw / (u_std**2)

  total_loss = data_loss + lambda_phys * physics_loss_normalized

  Pros: Both losses on same scale
  Cons: Extra normalization step

  ---
  ✅ Option 4: Adaptive weighting

  ## Automatically balance losses
  data_loss_val = torch.mean((u_pred - u_true)**2)
  physics_loss_val = torch.mean(residual**2)

  ## Make them equal magnitude
  lambda_phys = data_loss_val.detach() / (physics_loss_val.detach() + 1e-8)

  total_loss = data_loss + lambda_phys * physics_loss

  Pros: Automatic balancing
  Cons: Lambda changes during training

  ---
  Summary: The Mathematical Answer

  Your physics loss explodes because:

  1. Normalization amplifies derivatives by 1/u_std = 62.5
  2. Squared terms in loss amplify by (1/u_std)² ≈ 3906
  3. Asymmetric normalization (normalizing u but not a, f) creates larger residuals

  Your solution (tiny lambda) is mathematically correct:
  λ_normalized = λ_original × (u_std)²
  0.00009 ≈ 0.35 × (0.016)²

  This compensates for the ~4000× amplification of physics loss!

  Bottom line: There's nothing wrong with your approach - it's the mathematically rigorous solution to handle normalized data with physics constraints. The "strange" tiny lambda
  is actually the correct scaling factor given your normalization scheme