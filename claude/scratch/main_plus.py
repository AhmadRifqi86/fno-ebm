#!/usr/bin/env python3
"""
main_noisy.py - Single-Dataset Training Mode with Enhanced Diagnostics

Trains FNO-EBM on a single noisy dataset with comprehensive diagnostics
to identify root causes of uncertainty map issues.

⭐ ENHANCED VERSION with failure mode analysis! ⭐
"""

import yaml
import torch
from torch.utils.data import DataLoader
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from config import Config
from fno import FNO2d
from ebm import SimpleFNO_EBM
from trainer import Trainer, FNO_EBM
from customs import DarcyPhysicsLoss
from inference import inference_deterministic, inference_probabilistic
from datautils import PDEDataset, visualize_inference_results


def diagnose_energy_landscape(model, val_loader, config, num_samples=5):
    """
    Diagnose whether EBM learned a structured energy landscape.
    
    Tests:
    1. Energy discrimination: Do positive/negative samples have different energies?
    2. Energy smoothness: Is energy spatially smooth or chaotic?
    3. Energy sensitivity: Does energy change with spatial perturbations?
    """
    print("\n" + "=" * 80)
    print("DIAGNOSTIC 1: ENERGY LANDSCAPE ANALYSIS")
    print("=" * 80)
    
    model.eval()
    device = config.device
    
    # Get validation batch
    x_batch, y_batch = next(iter(val_loader))
    x_batch = x_batch[:num_samples].to(device)
    y_batch = y_batch[:num_samples].to(device)
    
    with torch.no_grad():
        # 1. Energy of ground truth (positive samples)
        energy_positive = model.energy(y_batch, x_batch, training=False)
        
        # 2. Energy of FNO prediction
        y_fno = model.u_fno(x_batch)
        energy_fno = model.energy(y_fno, x_batch, training=False)
        
        # 3. Energy of random noise (negative samples)
        y_noise = torch.randn_like(y_batch) * 0.1
        energy_noise = model.energy(y_noise, x_batch, training=False)
        
        # 4. Energy of spatially smooth perturbation
        y_smooth = y_batch + 0.05 * torch.randn(y_batch.shape[0], 1, 1, 1, device=device)
        energy_smooth = model.energy(y_smooth, x_batch, training=False)
    
    # Compute statistics
    results = {
        'positive': energy_positive.cpu().numpy(),
        'fno': energy_fno.cpu().numpy(),
        'noise': energy_noise.cpu().numpy(),
        'smooth': energy_smooth.cpu().numpy()
    }
    
    print("\nEnergy Statistics (lower = more probable):")
    print(f"  Ground Truth:     mean={results['positive'].mean():.4f}, std={results['positive'].std():.4f}")
    print(f"  FNO Prediction:   mean={results['fno'].mean():.4f}, std={results['fno'].std():.4f}")
    print(f"  Random Noise:     mean={results['noise'].mean():.4f}, std={results['noise'].std():.4f}")
    print(f"  Smooth Perturb:   mean={results['smooth'].mean():.4f}, std={results['smooth'].std():.4f}")
    
    # Test 1: Energy discrimination
    discrimination_gap = results['noise'].mean() - results['positive'].mean()
    print(f"\n✓ Test 1: Energy Discrimination")
    print(f"  Gap (noise - positive): {discrimination_gap:.4f}")
    if discrimination_gap < 0.1:
        print(f"  ❌ FAIL: Energy barely discriminates! (gap < 0.1)")
        print(f"     → EBM cannot distinguish good from bad solutions")
        print(f"     → ROOT CAUSE: Negative sampling is ineffective")
    else:
        print(f"  ✓ PASS: EBM discriminates between good/bad samples")
    
    # Test 2: Energy smoothness
    fno_gap = abs(results['fno'].mean() - results['positive'].mean())
    print(f"\n✓ Test 2: Energy Smoothness")
    print(f"  Gap (FNO - positive): {fno_gap:.4f}")
    if fno_gap > 0.5:
        print(f"  ❌ FAIL: Energy landscape is too rough! (gap > 0.5)")
        print(f"     → FNO predictions (good approximations) have high energy")
        print(f"     → ROOT CAUSE: Training didn't converge properly")
    else:
        print(f"  ✓ PASS: Energy is smooth around good solutions")
    
    # Test 3: Spatial sensitivity
    spatial_sensitivity = results['smooth'].std()
    print(f"\n✓ Test 3: Spatial Sensitivity")
    print(f"  Energy variance under perturbation: {spatial_sensitivity:.4f}")
    if spatial_sensitivity < 0.01:
        print(f"  ❌ FAIL: Energy is spatially insensitive! (var < 0.01)")
        print(f"     → Small spatial changes don't affect energy")
        print(f"     → ROOT CAUSE: Architecture doesn't capture spatial structure")
    else:
        print(f"  ✓ PASS: Energy responds to spatial variations")
    
    return results, discrimination_gap, spatial_sensitivity


def diagnose_negative_samples(model, val_loader, config, num_viz=3):
    """
    Diagnose negative sample quality during training.
    
    Tests:
    1. Sample diversity: Are negatives exploring the space?
    2. Sample quality: Are negatives realistic but wrong?
    3. Energy progression: Do samples move toward lower energy?
    """
    print("\n" + "=" * 80)
    print("DIAGNOSTIC 2: NEGATIVE SAMPLE QUALITY")
    print("=" * 80)
    
    model.eval()
    device = config.device
    
    # Get validation batch
    x_batch, y_batch = next(iter(val_loader))
    x_batch = x_batch[:num_viz].to(device)
    y_batch = y_batch[:num_viz].to(device)
    
    # Generate negative samples with intermediate tracking
    with torch.no_grad():
        y_init = model.u_fno(x_batch).clone()
    
    y_neg = y_init.clone()
    y_neg.requires_grad_(True)
    
    energies_over_time = []
    samples_over_time = [y_neg.detach().cpu().clone()]
    
    step_size = config.mcmc_step_size
    noise_scale = np.sqrt(2 * step_size)
    
    # Track 20 snapshots during MCMC
    snapshot_steps = np.linspace(0, config.mcmc_steps - 1, 20, dtype=int)
    
    for k in range(config.mcmc_steps):
        energy = model.energy(y_neg, x_batch, training=False).sum()
        grad_u = torch.autograd.grad(energy, y_neg)[0]
        
        with torch.no_grad():
            noise = torch.randn_like(y_neg) * noise_scale
            y_neg = y_neg - step_size * grad_u + noise
            y_neg.requires_grad_(True)
            
            if k in snapshot_steps:
                energies_over_time.append(model.energy(y_neg, x_batch, training=False).mean().item())
                samples_over_time.append(y_neg.detach().cpu().clone())
    
    samples_over_time = torch.stack(samples_over_time)
    
    # Test 1: Sample diversity
    diversity_initial = (samples_over_time[0] - samples_over_time[5]).abs().mean().item()
    diversity_final = (samples_over_time[-1] - samples_over_time[-6]).abs().mean().item()
    
    print(f"\n✓ Test 1: Sample Diversity")
    print(f"  Early diversity (steps 0-5):   {diversity_initial:.6f}")
    print(f"  Late diversity (steps 15-20):  {diversity_final:.6f}")
    
    if diversity_final < 0.001:
        print(f"  ❌ FAIL: Samples collapse to single mode! (diversity < 0.001)")
        print(f"     → MCMC isn't exploring the energy landscape")
        print(f"     → ROOT CAUSE: Step size too small or energy too peaked")
    elif diversity_initial < 0.001:
        print(f"  ⚠️  WARNING: Samples start identical (FNO mode collapse?)")
    else:
        print(f"  ✓ PASS: Samples show sufficient diversity")
    
    # Test 2: Energy progression
    if len(energies_over_time) > 1:
        energy_drop = energies_over_time[0] - energies_over_time[-1]
        print(f"\n✓ Test 2: Energy Progression")
        print(f"  Initial energy: {energies_over_time[0]:.4f}")
        print(f"  Final energy:   {energies_over_time[-1]:.4f}")
        print(f"  Energy drop:    {energy_drop:.4f}")
        
        if energy_drop < 0:
            print(f"  ❌ FAIL: Energy INCREASES! Gradient descent is broken!")
            print(f"     → ROOT CAUSE: Gradients are wrong or step size too large")
        elif energy_drop < 0.01:
            print(f"  ⚠️  WARNING: Energy barely changes (drop < 0.01)")
            print(f"     → Negative samples aren't moving much")
        else:
            print(f"  ✓ PASS: Samples move toward lower energy")
    
    # Test 3: Spatial structure in negatives
    final_sample = samples_over_time[-1][0, ..., 0].numpy()
    
    # Compute spatial autocorrelation (simple version)
    shifted = np.roll(final_sample, shift=1, axis=0)
    autocorr = np.corrcoef(final_sample.flatten(), shifted.flatten())[0, 1]
    
    print(f"\n✓ Test 3: Spatial Structure in Negatives")
    print(f"  Spatial autocorrelation: {autocorr:.4f}")
    
    if abs(autocorr) < 0.1:
        print(f"  ❌ FAIL: Negative samples have no spatial structure! (|corr| < 0.1)")
        print(f"     → Samples look like random noise")
        print(f"     → ROOT CAUSE: EBM doesn't learn spatial patterns")
    else:
        print(f"  ✓ PASS: Negatives have spatial coherence")
    
    # Visualize negative sample evolution
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    sample_idx = 0
    
    viz_indices = [0, 5, 10, 15, 19]
    for i, idx in enumerate(viz_indices):
        sample = samples_over_time[idx, sample_idx, ..., 0].numpy()
        axes[0, i].imshow(sample, cmap='viridis')
        axes[0, i].set_title(f'Step {snapshot_steps[idx] if idx < len(snapshot_steps) else config.mcmc_steps}')
        axes[0, i].axis('off')
    
    # Bottom row: Energy progression and diversity
    axes[1, 0].plot(snapshot_steps[:len(energies_over_time)], energies_over_time, 'o-')
    axes[1, 0].set_xlabel('MCMC Step')
    axes[1, 0].set_ylabel('Energy')
    axes[1, 0].set_title('Energy Evolution')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Compute per-step diversity
    diversities = []
    for i in range(len(samples_over_time) - 1):
        div = (samples_over_time[i] - samples_over_time[i+1]).abs().mean().item()
        diversities.append(div)
    
    axes[1, 1].plot(diversities, 'o-')
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('Change magnitude')
    axes[1, 1].set_title('Sample Movement per Step')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Histogram of final sample
    axes[1, 2].hist(final_sample.flatten(), bins=50, alpha=0.7)
    axes[1, 2].set_xlabel('Value')
    axes[1, 2].set_title('Final Sample Distribution')
    axes[1, 2].grid(True, alpha=0.3)
    
    # Ground truth for comparison
    gt = y_batch[sample_idx, ..., 0].cpu().numpy()
    axes[1, 3].imshow(gt, cmap='viridis')
    axes[1, 3].set_title('Ground Truth')
    axes[1, 3].axis('off')
    
    axes[1, 4].axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(config.checkpoint_dir, 'negative_samples_diagnosis.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved negative sample visualization to {save_path}")
    plt.close()
    
    return diversity_final, autocorr


def diagnose_model_capacity(model, config):
    """
    Diagnose whether model architecture has sufficient capacity.
    
    Tests:
    1. Parameter utilization: Are parameters actually being used?
    2. Gradient flow: Are gradients flowing through the network?
    3. Representation power: Can model represent spatial features?
    """
    print("\n" + "=" * 80)
    print("DIAGNOSTIC 3: MODEL ARCHITECTURE CAPACITY")
    print("=" * 80)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.V_ebm.parameters())
    trainable_params = sum(p.numel() for p in model.V_ebm.parameters() if p.requires_grad)
    
    print(f"\nParameter Count:")
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Check gradient statistics from last training step
    print(f"\nGradient Flow Analysis:")
    grad_norms = []
    param_norms = []
    zero_grad_count = 0
    
    for name, param in model.V_ebm.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            param_norm = param.norm().item()
            grad_norms.append(grad_norm)
            param_norms.append(param_norm)
            
            if grad_norm < 1e-7:
                zero_grad_count += 1
                print(f"  ⚠️  {name}: grad_norm={grad_norm:.2e} (near zero!)")
    
    if len(grad_norms) > 0:
        avg_grad_norm = np.mean(grad_norms)
        avg_param_norm = np.mean(param_norms)
        grad_param_ratio = avg_grad_norm / (avg_param_norm + 1e-8)
        
        print(f"\n  Average gradient norm:  {avg_grad_norm:.6f}")
        print(f"  Average parameter norm: {avg_param_norm:.6f}")
        print(f"  Grad/param ratio:       {grad_param_ratio:.6f}")
        print(f"  Layers with zero grad:  {zero_grad_count}/{len(grad_norms)}")
        
        if grad_param_ratio < 1e-5:
            print(f"\n  ❌ FAIL: Gradients are too small! (ratio < 1e-5)")
            print(f"     → Parameters aren't being updated effectively")
            print(f"     → ROOT CAUSE: Learning rate too small or vanishing gradients")
        elif zero_grad_count > len(grad_norms) * 0.3:
            print(f"\n  ⚠️  WARNING: {zero_grad_count} layers have near-zero gradients")
            print(f"     → Some layers aren't learning")
        else:
            print(f"\n  ✓ PASS: Gradient flow appears healthy")
    else:
        print(f"\n  ⚠️  No gradients found (model not in training mode?)")
    
    # Architecture-specific checks
    print(f"\nArchitecture Configuration:")
    if hasattr(model.V_ebm, 'fno_width'):
        print(f"  FNO width:  {model.V_ebm.fno_width}")
    if hasattr(model.V_ebm, 'fno_layers'):
        print(f"  FNO layers: {model.V_ebm.fno_layers}")
    
    # Receptive field estimate (for ConvNets)
    if hasattr(model.V_ebm, 'conv1'):
        print(f"\n✓ Using convolutional architecture (spatial inductive bias)")
    else:
        print(f"\n⚠️  Not using convolutional architecture")
    
    return total_params, trainable_params


def generate_diagnostic_report(energy_results, diversity, autocorr, discrimination_gap, 
                               spatial_sensitivity, config):
    """
    Generate comprehensive diagnostic report with actionable recommendations.
    """
    print("\n" + "=" * 80)
    print("COMPREHENSIVE DIAGNOSTIC REPORT")
    print("=" * 80)
    
    # Classify the failure mode
    failure_modes = []
    
    if discrimination_gap < 0.1:
        failure_modes.append({
            'name': 'Weak Energy Discrimination',
            'severity': 'CRITICAL',
            'description': 'EBM cannot distinguish positive from negative samples',
            'likely_cause': 'Negative sampling is ineffective',
            'recommendations': [
                'Increase MCMC steps during training (try 100-200)',
                'Increase MCMC step size (try 0.01-0.05)',
                'Add noise schedule: start high, anneal to low',
                'Try different negative sample initialization (uniform, Gaussian)',
                'Consider using persistent contrastive divergence (PCD)',
            ]
        })
    
    if spatial_sensitivity < 0.01:
        failure_modes.append({
            'name': 'No Spatial Structure Learning',
            'severity': 'CRITICAL',
            'description': 'Energy function is spatially uniform',
            'likely_cause': 'Architecture cannot capture spatial patterns',
            'recommendations': [
                'Increase model capacity: fno_width=64, fno_layers=4',
                'Add convolutional layers if not present',
                'Ensure input includes spatial coordinates',
                'Try ConvEBM instead of SimpleFNO_EBM',
                'Check if receptive field covers full domain',
            ]
        })
    
    if diversity < 0.001:
        failure_modes.append({
            'name': 'Mode Collapse in Negative Sampling',
            'severity': 'HIGH',
            'description': 'MCMC samples collapse to single mode',
            'likely_cause': 'Energy landscape is too peaked or step size too small',
            'recommendations': [
                'Increase MCMC step size for inference (try 0.05-0.1)',
                'Increase number of MCMC steps (try 500-1000)',
                'Add temperature parameter to soften energy landscape',
                'Use multiple chains with different initializations',
                'Consider annealed importance sampling',
            ]
        })
    
    if abs(autocorr) < 0.1:
        failure_modes.append({
            'name': 'Negative Samples Lack Spatial Coherence',
            'severity': 'HIGH',
            'description': 'Negative samples look like random noise',
            'likely_cause': 'MCMC cannot find structured samples',
            'recommendations': [
                'Initialize negatives from FNO + smooth noise',
                'Use spatially correlated noise (Gaussian process)',
                'Increase MCMC steps significantly',
                'Add spatial smoothness prior to energy function',
                'Try score matching instead of contrastive divergence',
            ]
        })
    
    # Print failure modes
    if len(failure_modes) == 0:
        print("\n🎉 NO CRITICAL ISSUES DETECTED!")
        print("   Model appears to be working correctly.")
        print("   If uncertainty maps still look noisy, try:")
        print("   - Increase num_samples during inference (100-200)")
        print("   - Adjust color scale normalization")
        print("   - Check denormalization of std values")
    else:
        print(f"\n⚠️  DETECTED {len(failure_modes)} FAILURE MODE(S):\n")
        
        for i, mode in enumerate(failure_modes, 1):
            print(f"{i}. {mode['name']} [{mode['severity']}]")
            print(f"   Description: {mode['description']}")
            print(f"   Likely cause: {mode['likely_cause']}")
            print(f"   Recommendations:")
            for rec in mode['recommendations']:
                print(f"     • {rec}")
            print()
    
    # Priority action items
    print("=" * 80)
    print("PRIORITY ACTION ITEMS (try in order):")
    print("=" * 80)
    
    if discrimination_gap < 0.1:
        print("\n1. FIX NEGATIVE SAMPLING FIRST (most critical)")
        print("   → Negative samples aren't diverse enough for training")
        print("   → Modify trainer.py to log negative sample diversity")
        print("   → Increase mcmc_steps in config.yaml to 100-200")
        print("   → Try step_size=0.02 instead of 0.005")
    
    elif spatial_sensitivity < 0.01:
        print("\n1. INCREASE MODEL CAPACITY (architecture issue)")
        print("   → Current architecture can't learn spatial patterns")
        print("   → In ebm.py, increase fno_width to 64")
        print("   → In ebm.py, increase fno_layers to 4")
        print("   → Consider switching to ConvEBM")
    
    elif diversity < 0.001:
        print("\n1. FIX INFERENCE SAMPLING (MCMC collapse)")
        print("   → Inference MCMC step size is too small")
        print("   → In inference call, try step_size=0.05-0.1")
        print("   → Increase num_mcmc_steps to 500-1000")
        print("   → Add temperature parameter: energy/T")
    
    else:
        print("\n✓ All diagnostic tests passed!")
        print("  If uncertainty maps still look poor, investigate:")
        print("  - Training convergence (check loss curves)")
        print("  - Hyperparameter tuning (learning rates, batch size)")
        print("  - Data quality (verify spatial structure exists)")
    
    print("\n" + "=" * 80)
    
    # Save report to file
    report_path = os.path.join(config.checkpoint_dir, 'diagnostic_report.txt')
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("FNO-EBM DIAGNOSTIC REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("METRICS:\n")
        f.write(f"  Energy discrimination gap: {discrimination_gap:.4f}\n")
        f.write(f"  Spatial sensitivity:       {spatial_sensitivity:.4f}\n")
        f.write(f"  Negative sample diversity: {diversity:.6f}\n")
        f.write(f"  Spatial autocorrelation:   {autocorr:.4f}\n\n")
        
        f.write("FAILURE MODES:\n")
        if len(failure_modes) == 0:
            f.write("  None detected!\n")
        else:
            for mode in failure_modes:
                f.write(f"\n  {mode['name']} [{mode['severity']}]\n")
                f.write(f"    {mode['description']}\n")
                f.write(f"    Likely cause: {mode['likely_cause']}\n")
    
    print(f"✓ Diagnostic report saved to {report_path}")


def main():
    """
    Single-dataset training pipeline with enhanced diagnostics.
    """

    print("=" * 70)
    print("FNO-EBM TRAINING: SINGLE-DATASET MODE (Noisy Data Only)")
    print("WITH ENHANCED DIAGNOSTICS")
    print("=" * 70)

    # 1. Load Configuration
    print("\n--- Loading Configuration ---")
    with open('config.yaml', 'r') as f:
        config_dict = yaml.safe_load(f)
    config = Config(config_dict)

    print(f"Device: {config.device}")
    print(f"PDE Type: {config.pde_type}")
    print(f"Complexity: {config.complexity}")
    print(f"Noise Type: {config.noise_type}")
    print(f"Physics Loss Weight: {config.lambda_phys} (should be 0.0 for noisy data)")

    if config.lambda_phys > 0:
        print("\n⚠️  WARNING: lambda_phys > 0 detected!")
        print("   For noisy data training, physics loss should be disabled (lambda_phys=0.0)")
        config.lambda_phys = 0.0

    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # 2. Initialize Models
    print("\n--- Initializing Models ---")

    fno_model = FNO2d(
        modes1=config.fno_modes,
        modes2=config.fno_modes,
        width=config.fno_width,
        num_layers=4
    )

    ebm_model = SimpleFNO_EBM(in_channels=4, fno_width=32, fno_layers=3)

    model = FNO_EBM(fno_model, ebm_model).to(config.device)

    print(f"FNO: modes={config.fno_modes}, width={config.fno_width}")
    print(f"EBM: ConvEBM with spatial convolutions")

    # 3. Load Noisy Dataset
    print("\n--- Loading Noisy Dataset ---")

    data_dir = Path(config.data_dir) if hasattr(config, 'data_dir') else Path('../data')
    resolution = config.grid_size if hasattr(config, 'grid_size') else 64
    complexity = config.complexity if hasattr(config, 'complexity') else 'medium'
    noise_type = config.noise_type if hasattr(config, 'noise_type') else 'heteroscedastic'

    noisy_train_file = data_dir / f"darcy_{complexity}_{noise_type}_res{resolution}_train.npz"
    noisy_val_file = data_dir / f"darcy_{complexity}_{noise_type}_res{resolution}_val.npz"

    if not noisy_train_file.exists() or not noisy_val_file.exists():
        print(f"\n❌ ERROR: Noisy dataset files not found!")
        print(f"Expected files:")
        print(f"  - {noisy_train_file}")
        print(f"  - {noisy_val_file}")
        return

    print(f"Loading noisy dataset from {data_dir}...")
    train_dataset = PDEDataset.from_file(str(noisy_train_file), normalize_output=True)
    val_dataset = PDEDataset.from_file(str(noisy_val_file), normalize_output=True)

    print("✓ Dataset loaded (automatically normalized)")

    # 3b. Create DataLoaders
    print("\n--- Creating DataLoaders ---")

    batch_size = config.batch_size if hasattr(config, 'batch_size') else 16
    num_workers = config.num_workers if hasattr(config, 'num_workers') else 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"✓ DataLoaders created (batch_size={batch_size})")

    # 4. Initialize Trainer
    print("\n--- Initializing Trainer (Single-Dataset Mode) ---")

    phy_loss = DarcyPhysicsLoss(source_term=1.0)

    trainer = Trainer(
        model=model,
        phy_loss=phy_loss,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config
    )

    print("Training mode: Single-dataset (noisy)")

    # 5. Train
    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)
    trainer.train_staged()
    print("\n" + "=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)

    # 6. POST-TRAINING DIAGNOSTICS
    print("\n" + "=" * 70)
    print("RUNNING POST-TRAINING DIAGNOSTICS")
    print("=" * 70)
    
    best_fno_path = os.path.join(config.checkpoint_dir, 'best_model_fno.pt')
    best_ebm_path = os.path.join(config.checkpoint_dir, 'best_model_ebm.pt')

    if os.path.exists(best_fno_path) and os.path.exists(best_ebm_path):
        # Load best models
        fno_checkpoint = torch.load(best_fno_path, map_location=config.device)
        ebm_checkpoint = torch.load(best_ebm_path, map_location=config.device)

        model.u_fno.load_state_dict(fno_checkpoint['fno_model'])
        model.V_ebm.load_state_dict(ebm_checkpoint['ebm_model'])

        print("✓ Loaded best models for diagnostics\n")
        
        # Run all diagnostic tests
        energy_results, discrimination_gap, spatial_sensitivity = diagnose_energy_landscape(
            model, val_loader, config, num_samples=5
        )
        
        diversity, autocorr = diagnose_negative_samples(
            model, val_loader, config, num_viz=3
        )
        
        total_params, trainable_params = diagnose_model_capacity(model, config)
        
        # Generate comprehensive report
        generate_diagnostic_report(
            energy_results, diversity, autocorr, 
            discrimination_gap, spatial_sensitivity, config
        )
        
        # 7. Standard Inference (after diagnostics)
        print("\n" + "=" * 70)
        print("RUNNING STANDARD INFERENCE")
        print("=" * 70)
        
        # Get validation batch
        x_samples, y_true_samples = next(iter(val_loader))
        x_samples = x_samples.to(config.device)

        # Deterministic inference (FNO)
        print("Running FNO deterministic inference...")
        y_fno_pred = inference_deterministic(model, x_samples, device=config.device)

        # Probabilistic inference (EBM)
        print("Running EBM probabilistic inference...")
        _, stats = inference_probabilistic(
            model,
            x_samples,
            num_samples=50,
            num_mcmc_steps=config.mcmc_steps,
            step_size=config.mcmc_step_size,
            device=config.device
        )

        # Denormalize predictions for visualization
        y_fno_pred_denorm = train_dataset.denormalize(y_fno_pred)
        y_true_denorm = train_dataset.denormalize(y_true_samples)

        stats_denorm = {
            'mean': train_dataset.denormalize(stats['mean']),
            'std': stats['std'] * train_dataset.u_std,
        }

        # Visualize
        visualize_inference_results(y_true_denorm, y_fno_pred_denorm, stats_denorm, config)
        print("✓ Inference results saved")

    else:
        print("⚠️  Could not find best model checkpoints - skipping diagnostics")

    print("\n" + "=" * 70)
    print("SUMMARY - SINGLE-DATASET MODE WITH DIAGNOSTICS")
    print("=" * 70)
    print(f"Checkpoints saved in: {config.checkpoint_dir}/")
    print("  - best_model_fno.pt")
    print("  - best_model_ebm.pt")
    print("  - current_fno.pt")
    print("  - current_ebm.pt")
    print("\nDiagnostic outputs:")
    print(f"  - diagnostic_report.txt (comprehensive analysis)")
    print(f"  - negative_samples_diagnosis.png (MCMC evolution)")
    print(f"  - inference_results.png (standard output)")
    print("\nWhat the diagnostics tell you:")
    print("=" * 70)
    print("1. ENERGY LANDSCAPE ANALYSIS")
    print("   → Tests if EBM learned to discriminate good/bad solutions")
    print("   → Tests if energy is smooth or chaotic")
    print("   → Tests if energy responds to spatial changes")
    print()
    print("2. NEGATIVE SAMPLE QUALITY")
    print("   → Tests if MCMC explores the space (diversity)")
    print("   → Tests if samples have spatial structure")
    print("   → Tests if energy decreases during sampling")
    print()
    print("3. MODEL CAPACITY")
    print("   → Tests if gradients flow properly")
    print("   → Tests if architecture has enough capacity")
    print("   → Tests parameter utilization")
    print()
    print("ROOT CAUSE IDENTIFICATION:")
    print("  • Low discrimination gap (<0.1)")
    print("    → Negative sampling failed during training")
    print("    → Fix: Increase MCMC steps, step size, or use PCD")
    print()
    print("  • Low spatial sensitivity (<0.01)")
    print("    → Architecture can't capture spatial patterns")
    print("    → Fix: Increase model capacity or use ConvEBM")
    print()
    print("  • Low sample diversity (<0.001)")
    print("    → MCMC collapses during inference")
    print("    → Fix: Increase inference step size and num_steps")
    print()
    print("  • Low spatial autocorrelation (<0.1)")
    print("    → Samples lack spatial coherence")
    print("    → Fix: Better initialization or longer MCMC chains")
    print()
    print("Next steps:")
    print("  1. Read diagnostic_report.txt for detailed analysis")
    print("  2. View negative_samples_diagnosis.png to see MCMC behavior")
    print("  3. Follow priority action items in the report")
    print("  4. Re-run training after fixing identified issues")
    print("=" * 70)


if __name__ == '__main__':
    main()