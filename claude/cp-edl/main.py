#!/usr/bin/env python3
"""
main.py - Comprehensive UQ Experiment Runner

Supports 17 UQ methods across 3 families:
- 6 Conformal Prediction methods (Split, Full, Cross, CQR, Adaptive, Mondrian)
- 6 Evidential Deep Learning methods (DER, Improved DER, Prior Networks, Posterior Networks, Natural Posterior, Dirichlet)
- 5 Cross-family baselines (MC Dropout, Ensemble, Bayesian, Standard FNO, MLP-EBM)
"""

import argparse
import json
import os
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
import pandas as pd
import matplotlib.pyplot as plt

from config import Config, Factory, get_baseline_configs, get_conformal_methods_configs, get_evidential_methods_configs
from torch.utils.data import DataLoader, Subset
from datautils import (
    load_pde_data, create_dataloaders, create_dataloaders_with_calibration,
    create_kfold_splits, create_stratified_splits, create_ensemble_splits,
    create_ood_test_data, get_calibration_dataset, create_subset_loader,
    load_ns_exp4, PDEDataset
)
from fno import (
    FNO2d, FNOTrainer, EvidentialFNO2d, AblationEvidentialFNO2d, EvidentialFNOTrainer,
    MCDropoutFNO2d, FNOEnsemble, BayesianFNO2d, QuantileFNO2d, PriorNetworkFNO2d,
    DirichletEvidentialFNO2d, PosteriorNetworkFNO2d
)
from customs import (
    # Conformal Prediction
    conformal_calibrate, conformal_predict, conformal_coverage,
    full_conformal_calibrate, cross_conformal_calibrate,
    adaptive_conformal_calibrate, mondrian_conformal_calibrate, mondrian_conformal_predict,
    cqr_calibrate, cqr_predict, quantile_loss,
    # Evidential Deep Learning
    nig_nll, evidential_loss, evidential_uncertainty, evidential_regularization,
    improved_evidential_loss, improved_evidential_regularization,
    natural_nig_loss, prior_network_loss,
    # Additional Regularization Schemes
    uncertainty_aware_regularization, annealed_regularization,
    l2_evidence_regularization, adaptive_regularization,
    kl_divergence_regularization, general_evidential_loss,
    # Cross-family
    bayesian_elbo_loss,
    # Evaluation
    evaluate_uq_method, expected_calibration_error, uncertainty_error_correlation
)
from kanebm import EBMTrainer
from visualize import generate_ebm_visualization, plot_training_curves, visualize_evidential_parameters


def train_fno_ebm(config_dict: dict, data_path: str, pde_type: str,
                  nu_values: list = None, output_dir: str = 'experiments'):
    """
    Main training function for FNO-EBM on a single PDE.

    Args:
        config_dict: Configuration dictionary
        data_path: Path to data file
        pde_type: 'burgers', 'advection', 'diffusion_reaction', 'navier_stokes'
        nu_values: List of nu values (for 1D PDEs)
        output_dir: Output directory for checkpoints/logs
    """
    # Setup output directories
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = f"{pde_type}_nu{len(nu_values) if nu_values else 'all'}_{timestamp}"
    exp_dir = Path(output_dir) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    config_dict['log_dir'] = str(exp_dir / 'logs')
    config_dict['checkpoint_dir'] = str(exp_dir / 'checkpoints')
    os.makedirs(config_dict['log_dir'], exist_ok=True)
    os.makedirs(config_dict['checkpoint_dir'], exist_ok=True)

    # Save config
    with open(exp_dir / 'config.json', 'w') as f:
        json.dump(config_dict, f, indent=2)

    config = Config(config_dict)

    print(f"\n{'='*80}")
    print(f"Training: {pde_type}")
    print(f"Nu values: {nu_values if nu_values else 'all'}")
    print(f"Device: {config.device}")
    print(f"Output: {exp_dir}")
    print(f"{'='*80}\n")

    # Load data with memory-safe defaults
    print("Loading data...")
    max_samples = getattr(config, 'max_samples', 100)  # Default to 100 samples per nu value
    time_step_spacing = getattr(config, 'time_step_spacing', 10)
    max_pairs_per_sample = getattr(config, 'max_pairs_per_sample', 20)

    print(f"Data loading settings:")
    print(f"  max_samples: {max_samples}")
    print(f"  time_step_spacing: {time_step_spacing}")
    print(f"  max_pairs_per_sample: {max_pairs_per_sample}")

    dataset = load_pde_data(
        filepath=data_path,
        pde_type=pde_type,
        nu_values=nu_values,
        max_samples=max_samples,
        time_step_spacing=time_step_spacing,
        max_pairs_per_sample=max_pairs_per_sample
    )
    print(f"Dataset size: {len(dataset)}")
    print(f"Input shape: {dataset.X[0].shape}")
    print(f"Output shape: {dataset.U[0].shape}")

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset=dataset,
        train_ratio=config.train_ratio if hasattr(config, 'train_ratio') else 0.8,
        val_ratio=config.val_ratio if hasattr(config, 'val_ratio') else 0.1,
        batch_size=config.batch_size,
        seed=config.seed if hasattr(config, 'seed') else 42
    )
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")

    # Initialize FNO model
    print("\nInitializing FNO model...")
    fno_model = Factory.create_fno(config, pde_type=pde_type)
    print(f"FNO Model type: {fno_model.__class__.__name__}")
    print(f"FNO parameters: {sum(p.numel() for p in fno_model.parameters()):,}")

    # Train FNO
    print("\n" + "="*80)
    print("Stage 1: Training FNO")
    print("="*80)
    fno_trainer = FNOTrainer(model=fno_model, config=config)
    fno_epochs = getattr(config, 'fno_epochs', 100)
    fno_trainer.train(train_loader, val_loader, fno_epochs)

    # Initialize EBM
    print("\n" + "="*80)
    print("Stage 2: Training EBM")
    print("="*80)

    # Calculate EBM input_dim based on data shape
    sample_x, sample_u = dataset[0]
    n_spatial = sample_x.shape[0] * sample_x.shape[1]  # n_x * n_y
    n_input_channels = sample_x.shape[-1]  # coordinate channels
    n_output_channels = sample_u.shape[-1]  # output channels

    # Total flattened size: u + x + u_fno
    ebm_input_dim = n_spatial * (n_output_channels + n_input_channels + n_output_channels)

    print(f"EBM input calculation: n_spatial={n_spatial}, input_ch={n_input_channels}, output_ch={n_output_channels}")
    print(f"EBM input_dim: {ebm_input_dim}")

    # Update config_dict BEFORE creating Config object
    config_dict['ebm_input_dim'] = ebm_input_dim
    config_dict['ebm_n_input_channels'] = n_input_channels  # For CNN: x coordinate channels
    config_dict['ebm_n_output_channels'] = n_output_channels  # For CNN: u field channels

    # Recreate config with updated input_dim
    config = Config(config_dict)

    ebm_model = Factory.create_ebm(config)
    print(f"EBM parameters: {sum(p.numel() for p in ebm_model.parameters()):,}")

    ebm_trainer = EBMTrainer(
        model=ebm_model,
        config=config,
        fno_model=fno_model
    )
    ebm_epochs = getattr(config, 'ebm_epochs', 50)
    ebm_trainer.train(train_loader, val_loader, ebm_epochs)

    # Evaluate on test set
    print("\n" + "="*80)
    print("Final Evaluation on Test Set")
    print("="*80)
    test_loss = ebm_trainer.validate(test_loader)
    test_metrics = {'test_loss': test_loss}
    print(f"Test Metrics:")
    print(f"  test_loss: {test_loss:.6f}")

    # Save final results
    with open(exp_dir / 'results.json', 'w') as f:
        json.dump(test_metrics, f, indent=2)

    # Generate visualizations
    print("\n" + "="*80)
    print("Generating Visualizations and Calibration Analysis")
    print("="*80)

    # Generate comparison visualizations (GT, FNO, FNO Error, EBM) + calibration plots
    num_viz_samples = getattr(config, 'num_viz_samples', 4)
    generate_ebm_visualization(
        ebm_trainer=ebm_trainer,
        fno_model=fno_model,
        test_loader=test_loader,
        config=config,
        pde_type=pde_type,
        output_dir=str(exp_dir),
        num_samples=num_viz_samples,
    )

    print("\nCalibration plot saved to: " + str(exp_dir / 'ebm_calibration.png'))

    # Plot training curves for both FNO and EBM
    if hasattr(fno_trainer, 'train_losses') and len(fno_trainer.train_losses) > 0:
        plot_training_curves(
            train_losses=fno_trainer.train_losses,
            val_losses=fno_trainer.val_losses,
            save_path=str(exp_dir / 'fno_training_curves.png'),
            title='FNO Training Curves',
        )

    if hasattr(ebm_trainer, 'train_losses') and len(ebm_trainer.train_losses) > 0:
        plot_training_curves(
            train_losses=ebm_trainer.train_losses,
            val_losses=ebm_trainer.val_losses,
            save_path=str(exp_dir / 'ebm_training_curves.png'),
            title='EBM Training Curves',
        )

    print(f"\n{'='*80}")
    print("Training Complete!")
    print(f"Results saved to: {exp_dir}")
    print(f"{'='*80}\n")


def train_conformal_method(method_name: str, dataset, config_dict: dict,
                          output_dir: Path) -> Dict:
    """
    Train and evaluate a Conformal Prediction method.

    Args:
        method_name: 'split', 'full', 'cross', 'cqr', 'adaptive', 'mondrian'
        dataset: PDEDataset
        config_dict: Configuration dictionary
        output_dir: Output directory

    Returns:
        Dictionary with metrics (coverage, interval_width, mse, mae, etc.)
    """
    config = Config(config_dict)
    method_configs = get_conformal_methods_configs()
    method_config = method_configs.get(f'{method_name}_conformal', method_configs.get('split_conformal'))

    print(f"\n{'='*70}")
    print(f"CONFORMAL METHOD: {method_name.upper()}")
    print(f"{'='*70}")

    # Create 3-way split for all CP methods
    train_loader, cal_loader, test_loader = create_dataloaders_with_calibration(
        dataset,
        train_ratio=0.8,
        cal_ratio=method_config.get('calibration_split', 0.1),
        test_ratio=0.1,
        batch_size=config.batch_size,
        seed=config.seed if hasattr(config, 'seed') else 42
    )

    # PHASE 1: Train base model
    print("\nPHASE 1: Training base FNO...")
    if method_name == 'cqr':
        # CQR uses quantile model
        model = QuantileFNO2d(modes1=12, modes2=12, width=32, n_layers=4,
                             quantiles=method_config['quantiles'])
    else:
        # Standard FNO for all other CP methods
        model = FNO2d(modes1=12, modes2=12, width=32, n_layers=4, in_channels=3)

    model = model.to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=method_config.get('lr', 1e-3))

    # Initialize tracker
    tracker = None
    if getattr(config, 'enable_tracking', False):
        try:
            from track import GradientTracker
            tracker = GradientTracker(model, log_dir=str(output_dir / 'logs'),
                                     experiment_name=f'conformal_{method_name}')
        except ImportError:
            pass

    # Training loop - prioritize config_dict over method_config
    epochs = config_dict.get('epochs', method_config.get('epochs', 30))
    print(f"Training for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for x, y in train_loader:
            x, y = x.to(config.device), y.to(config.device)

            if method_name == 'cqr':
                q_low, q_high = model(x)
                loss = quantile_loss(q_low, y, tau=0.025) + quantile_loss(q_high, y, tau=0.975)
            else:
                pred = model(x)
                loss = F.mse_loss(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if tracker:
                tracker.track(loss=loss)

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        print(f"[{method_name.upper()}] Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.6f}")

    # PHASE 2: Calibrate
    print("\nPHASE 2: Calibrating conformal threshold...")
    if method_name == 'split':
        quantile = conformal_calibrate(model, cal_loader, alpha=method_config['alpha'],
                                       score_fn=method_config['score_fn'], device=config.device)
    elif method_name == 'cqr':
        quantile = cqr_calibrate(model, cal_loader, alpha=method_config['alpha'], device=config.device)
    # Add other methods as needed (full, cross, adaptive, mondrian would go here)
    else:
        # Fallback to split conformal
        quantile = conformal_calibrate(model, cal_loader, alpha=method_config['alpha'],
                                       score_fn=method_config.get('score_fn', 'l2'), device=config.device)

    print(f"Calibrated threshold: {quantile:.6f}")

    # PHASE 3: Evaluate
    print("\nPHASE 3: Evaluating on test set...")
    if method_name == 'cqr':
        metrics = evaluate_uq_method(model, test_loader, method_name='CQR',
                                     device=config.device, quantile=quantile)
    else:
        metrics = conformal_coverage(model, test_loader, quantile,
                                     score_fn=method_config.get('score_fn', 'l2'), device=config.device)

    print(f"Coverage: {metrics.get('coverage', 0):.3f}, Width: {metrics.get('avg_interval_width', 0):.6f}")

    # Save results
    with open(output_dir / f'{method_name}_conformal_results.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    return metrics


def get_regularization_function(reg_name: str):
    """
    Get regularization function by name.

    Args:
        reg_name: Name of regularization scheme

    Returns:
        Regularization function or None
    """
    if reg_name is None:
        return None

    reg_map = {
        'standard': evidential_regularization,
        'improved': improved_evidential_regularization,
        'uncertainty_aware': uncertainty_aware_regularization,
        #'annealed': annealed_regularization,
        'l2_evidence': l2_evidence_regularization,
        'adaptive': adaptive_regularization,
        'kl_divergence': kl_divergence_regularization
    }

    return reg_map.get(reg_name, None)


def train_evidential_method(method_name: str, data_path: str, pde_type: str,
                            config_dict: dict, output_dir: Path,
                            reg_name: str = None, max_samples: int = 500) -> Dict:
    """
    Train and evaluate an Evidential Deep Learning method.

    Args:
        method_name: 'der_nig', 'improved_der', 'prior_networks', 'posterior_networks',
                    'natural_posterior', 'dirichlet_evidential'
        data_path: Path to data file
        pde_type: PDE type ('burgers', 'advection', etc.)
        config_dict: Configuration dictionary
        output_dir: Output directory
        reg_name: Regularization scheme name
        max_samples: Maximum samples to load

    Returns:
        Dictionary with metrics (ECE, correlation, epistemic/aleatoric split, etc.)
    """
    config = Config(config_dict)
    method_configs = get_evidential_methods_configs()
    method_config = method_configs.get(method_name, method_configs.get('der_nig'))

    print(f"\n{'='*70}")
    print(f"EVIDENTIAL METHOD: {method_name.upper()}")
    print(f"{'='*70}")

    # Load RAW dataset (NO LEAKAGE MODE)
    print("\nLoading RAW dataset...")
    X_raw, U_raw = load_pde_data(data_path, pde_type, max_samples=max_samples, return_raw=True)
    print(f"Total dataset size: {len(X_raw)}")

    # Create train/val/test splits with NO LEAKAGE
    # This splits FIRST, then normalizes with TRAIN stats only
    from datautils import create_dataloaders_no_leakage
    train_loader, val_loader, test_loader = create_dataloaders_no_leakage(
        X_raw, U_raw,
        train_ratio=0.85,
        val_ratio=0.05,
        batch_size=config.batch_size,
        seed=config.seed if hasattr(config, 'seed') else 42
    )
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")

    # Create model
    print("\nInitializing evidential model...")
    if method_name in ['der_nig', 'natural_posterior']:
        model = EvidentialFNO2d(
            modes1=12, modes2=12, width=32, n_layers=4, in_channels=3,
            nu_min=method_config['nu_min'],
            alpha_min=method_config['alpha_min'],
            beta_min=method_config['beta_min']
        )
    elif method_name == 'prior_networks':
        model = PriorNetworkFNO2d(
            modes1=12, modes2=12, width=32, n_layers=4, in_channels=3,
            n_bins=method_config['n_bins'],
            output_range=tuple(method_config['output_range'])
        )
    elif method_name == 'posterior_networks':
        model = PosteriorNetworkFNO2d(
            modes1=12, modes2=12, width=32, n_layers=4, in_channels=3,
            n_flows=method_config['n_flows'],
            flow_hidden_dim=method_config['flow_hidden_dim']
        )
    elif method_name == 'dirichlet_evidential':
        model = DirichletEvidentialFNO2d(
            modes1=12, modes2=12, width=32, n_layers=4, in_channels=3,
            n_bins=method_config['n_bins'],
            output_range=tuple(method_config['output_range'])
        )
    else:
        raise ValueError(f"Unknown evidential method: {method_name}")

    model = model.to(config.device)

    # Training loop - prioritize config_dict over method_config
    epochs = config_dict.get('epochs', method_config.get('epochs', 30))

    # Use EvidentialFNOTrainer for ALL evidential methods
    print(f"\nUsing EvidentialFNOTrainer ({method_name}) for {epochs} epochs...")

    # Create optimizer and scheduler
    optimizer_config = {
        'type': method_config.get('optimizer', 'adam'),
        'lr': method_config.get('lr', 1e-4),
        'weight_decay': method_config.get('weight_decay', 0.0)
    }
    optimizer = Factory.create_optimizer(optimizer_config, model.parameters())

    scheduler = None
    if 'scheduler' in method_config and method_config['scheduler'] is not None:
        scheduler = Factory.create_scheduler(method_config['scheduler'], optimizer)
    else:
        # Default: exponential_lr for evidential methods
        scheduler_config = {
            'type': 'exponential_lr',
            'gamma': 0.93
        }
        scheduler = Factory.create_scheduler(scheduler_config, optimizer)

    # Setup trainer config
    trainer_config = Config({
        'device': config.device,
        'lr': method_config.get('lr', 1e-4),
        'weight_decay': method_config.get('weight_decay', 0.0),
        'enable_tracking': getattr(config, 'enable_tracking', False),
        'log_dir': str(output_dir / 'logs'),
        'experiment_name': f'evidential_{method_name}',
        'checkpoint_dir': str(output_dir / 'checkpoints'),
        'save_every': 50,
        'use_tensorboard': False
    })

    # Get regularization function if specified
    if reg_name is None:
        reg_name = method_name
        
    reg_fn = get_regularization_function(reg_name)
    if reg_name:
        print(f"Using regularization: {reg_name}")

    # Add max_epochs to method_config for annealed regularization
    if reg_name == 'annealed':
        method_config['max_epochs'] = epochs

    # Create trainer with method-specific configuration
    trainer = EvidentialFNOTrainer(
        model=model,
        config=trainer_config,
        method_name=method_name,
        optimizer=optimizer,
        scheduler=scheduler,
        method_config=method_config,
        reg_fn=reg_fn
    )

    # Train
    trainer.train(train_loader, val_loader, epochs=epochs)
        
    # Evaluate
    print("\nEvaluating uncertainty quantification...")
    metrics = evaluate_uq_method(model, test_loader, method_name=method_name, device=config.device)

    print(f"ECE: {metrics.get('ece', 0):.4f}, Correlation: {metrics.get('correlation', 0):.3f}")
    print(f"NLL: {metrics.get('nll', 0):.6f}, rel_l2: {metrics.get('rel_l2', 0):.3f}")
    print(f"MSE: {metrics.get('mse', 0):.6f}, MAE: {metrics.get('mae', 0):.6f}")
    print(f"coverage: {metrics.get('coverage', 0):.3f}, interval_width: {metrics.get('interval_width', 0):.6f}")

    # Generate visualization of NIG parameters (only for NIG-based methods)
    if method_name in ['der_nig', 'improved_der', 'natural_posterior']:
        print("\nGenerating evidential parameter visualization...")
        try:
            # Get samples from test loader
            for x, y in test_loader:
                pde_type = config_dict.get('pde_type', 'unknown')
                visualize_evidential_parameters(
                    model=model,
                    x=x,
                    u_gt=y,
                    save_path=str(output_dir / f'{method_name}_nig_parameters.png'),
                    n_samples=12,  # Visualize 3 samples
                    pde_type=pde_type,
                    device=config.device
                )
                print(f"NIG parameter visualizations saved to {output_dir}")
                break  # Only visualize first batch
        except Exception as e:
            print(f"Warning: Could not generate NIG parameter visualization: {e}")

    # Save results
    with open(output_dir / f'{method_name}_results.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    return metrics


def train_baseline_method(method_name: str, dataset, config_dict: dict,
                          output_dir: Path) -> Dict:
    """
    Train and evaluate a cross-family baseline UQ method.

    Args:
        method_name: 'mc_dropout', 'ensemble', 'bayesian', 'standard_fno', 'mlp_ebm'
        dataset: PDEDataset
        config_dict: Configuration dictionary
        output_dir: Output directory

    Returns:
        Dictionary with metrics (uncertainty quality, MSE, MAE, etc.)
    """
    config = Config(config_dict)
    baseline_configs = get_baseline_configs()
    method_config = baseline_configs.get(method_name, {})

    print(f"\n{'='*70}")
    print(f"BASELINE METHOD: {method_name.upper()}")
    print(f"{'='*70}")

    # Standard 2-way split for baselines
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset,
        train_ratio=0.8,
        val_ratio=0.1,
        batch_size=config.batch_size,
        seed=config.seed if hasattr(config, 'seed') else 42
    )

    # Create model based on method
    print(f"\nInitializing {method_name} model...")

    if method_name == 'mc_dropout':
        model = MCDropoutFNO2d(
            modes1=12, modes2=12, width=32, n_layers=4, in_channels=3,
            dropout_rate=method_config.get('dropout_rate', 0.1),
            n_samples=method_config.get('n_forward_passes', 30)
        )
        model = model.to(config.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Initialize tracker
        tracker = None
        if getattr(config, 'enable_tracking', False):
            try:
                from track import GradientTracker
                tracker = GradientTracker(model, log_dir=str(output_dir / 'logs'),
                                         experiment_name=f'baseline_{method_name}')
            except ImportError:
                pass

        # Training loop
        epochs = 500
        for epoch in range(epochs):
            model.train()
            for x, y in train_loader:
                x, y = x.to(config.device), y.to(config.device)
                pred = model.forward_single(x)  # Single pass during training
                loss = F.mse_loss(pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if tracker:
                    tracker.track(loss=loss)

            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

    elif method_name == 'ensemble':
        n_models = method_config.get('n_models', 5)
        print(f"Training {n_models} ensemble members...")

        # Create ensemble splits
        ensemble_splits = create_ensemble_splits(
            dataset, n_models=n_models, bootstrap=True,
            train_ratio=0.8, batch_size=config.batch_size,
            seed=config.seed if hasattr(config, 'seed') else 42
        )

        models = []
        for i, (train_loader_i, val_loader_i) in enumerate(ensemble_splits):
            print(f"\nTraining ensemble member {i+1}/{n_models}...")
            model_i = FNO2d(modes1=12, modes2=12, width=32, n_layers=4, in_channels=3)
            model_i = model_i.to(config.device)
            optimizer_i = torch.optim.Adam(model_i.parameters(), lr=1e-3)

            # Training loop for each ensemble member
            epochs = 500
            for epoch in range(epochs):
                model_i.train()
                for x, y in train_loader_i:
                    x, y = x.to(config.device), y.to(config.device)
                    pred = model_i(x)
                    loss = F.mse_loss(pred, y)

                    optimizer_i.zero_grad()
                    loss.backward()
                    optimizer_i.step()

                if (epoch + 1) % 200 == 0:
                    print(f"  Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

            models.append(model_i)

        # Create ensemble wrapper
        model = FNOEnsemble(models=models)

    elif method_name == 'bayesian':
        model = BayesianFNO2d(
            modes1=12, modes2=12, width=32, n_layers=4, in_channels=3,
            prior_std=method_config.get('prior_std', 1.0)
        )
        model = model.to(config.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Initialize tracker
        tracker = None
        if getattr(config, 'enable_tracking', False):
            try:
                from track import GradientTracker
                tracker = GradientTracker(model, log_dir=str(output_dir / 'logs'),
                                         experiment_name=f'baseline_{method_name}')
            except ImportError:
                pass

        # Training loop with ELBO loss
        kl_weight = method_config.get('kl_weight', 0.01)
        epochs = 500
        for epoch in range(epochs):
            model.train()
            for x, y in train_loader:
                x, y = x.to(config.device), y.to(config.device)
                pred = model(x)
                nll = F.mse_loss(pred, y)
                kl = model.kl_divergence()
                loss = nll + kl_weight * kl

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if tracker:
                    tracker.track(loss=loss)

            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs}, NLL: {nll.item():.6f}, KL: {kl.item():.6f}")

    elif method_name == 'standard_fno':
        # Standard FNO without UQ
        model = FNO2d(modes1=12, modes2=12, width=32, n_layers=4, in_channels=3)
        model = model.to(config.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Initialize tracker
        tracker = None
        if getattr(config, 'enable_tracking', False):
            try:
                from track import GradientTracker
                tracker = GradientTracker(model, log_dir=str(output_dir / 'logs'),
                                         experiment_name=f'baseline_{method_name}')
            except ImportError:
                pass

        # Standard training loop
        epochs = 500
        for epoch in range(epochs):
            model.train()
            for x, y in train_loader:
                x, y = x.to(config.device), y.to(config.device)
                pred = model(x)
                loss = F.mse_loss(pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if tracker:
                    tracker.track(loss=loss)

            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

    elif method_name == 'mlp_ebm':
        # Train FNO + MLP-EBM (original approach from train_fno_ebm)
        # Simplified version - just return placeholder for now
        print("MLP-EBM baseline uses the full train_fno_ebm pipeline.")
        print("Skipping for comprehensive comparison (use --mode fno_ebm for full training)")
        return {'method': 'mlp_ebm', 'note': 'Use fno_ebm mode for full training'}

    else:
        raise ValueError(f"Unknown baseline method: {method_name}")

    # Evaluate
    print("\nEvaluating baseline method...")
    metrics = evaluate_uq_method(model, test_loader, method_name=method_name, device=config.device)

    print(f"MSE: {metrics.get('mse', 0):.6f}, MAE: {metrics.get('mae', 0):.6f}")

    # Save results
    with open(output_dir / f'{method_name}_baseline_results.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    return metrics


def run_comprehensive_comparison(data_path: str, pde_type: str,
                                 output_dir: str = 'comprehensive_results',
                                 methods: List[str] = None):
    """
    Run comprehensive comparison of all 17 UQ methods.

    Args:
        data_path: Path to data file
        pde_type: PDE type
        output_dir: Output directory
        methods: List of methods to run (None = all 17 methods)
    """
    # Setup
    exp_dir = Path(output_dir) / f"{pde_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE UQ COMPARISON: {pde_type.upper()}")
    print(f"Output: {exp_dir}")
    print(f"{'='*80}\n")

    # Load data
    print("Loading dataset...")
    dataset = load_pde_data(data_path, pde_type, max_samples=500)

    config_dict = create_default_config(pde_type)

    # Define all methods
    if methods is None:
        all_methods = {
            # Conformal Prediction (6 methods)
            'cp': ['split', 'full', 'cross', 'cqr', 'adaptive', 'mondrian'],
            # Evidential Deep Learning (6 methods)
            'edl': ['der_nig', 'improved_der', 'prior_networks', 'posterior_networks',
                    'natural_posterior', 'dirichlet_evidential'],
            # Cross-family baselines (5 methods)
            'baseline': ['mc_dropout', 'ensemble', 'bayesian', 'standard_fno', 'mlp_ebm']
        }
    else:
        # Parse user-specified methods
        all_methods = {'cp': [], 'edl': [], 'baseline': []}
        for m in methods:
            if m in ['split', 'full', 'cross', 'cqr', 'adaptive', 'mondrian']:
                all_methods['cp'].append(m)
            elif m in ['der_nig', 'improved_der', 'prior_networks', 'posterior_networks',
                       'natural_posterior', 'dirichlet_evidential']:
                all_methods['edl'].append(m)
            elif m in ['mc_dropout', 'ensemble', 'bayesian', 'standard_fno', 'mlp_ebm']:
                all_methods['baseline'].append(m)

    results = {}

    # Run CP methods
    print(f"\n{'#'*80}")
    print("CONFORMAL PREDICTION METHODS")
    print(f"{'#'*80}")
    for method in all_methods['cp']:
        try:
            metrics = train_conformal_method(method, dataset, config_dict, exp_dir)
            results[f'CP_{method}'] = metrics
        except Exception as e:
            print(f"Error running {method}: {e}")
            results[f'CP_{method}'] = {'error': str(e)}

    # Run EDL methods
    print(f"\n{'#'*80}")
    print("EVIDENTIAL DEEP LEARNING METHODS")
    print(f"{'#'*80}")
    for method in all_methods['edl']:
        try:
            metrics = train_evidential_method(
                method_name=method,
                data_path=data_path,
                pde_type=pde_type,
                config_dict=config_dict,
                output_dir=exp_dir,
                max_samples=500
            )
            results[f'EDL_{method}'] = metrics
        except Exception as e:
            print(f"Error running {method}: {e}")
            results[f'EDL_{method}'] = {'error': str(e)}

    # Run Baseline methods
    if 'baseline' in all_methods and len(all_methods['baseline']) > 0:
        print(f"\n{'#'*80}")
        print("CROSS-FAMILY BASELINE METHODS")
        print(f"{'#'*80}")
        for method in all_methods['baseline']:
            try:
                metrics = train_baseline_method(method, dataset, config_dict, exp_dir)
                results[f'BASELINE_{method}'] = metrics
            except Exception as e:
                print(f"Error running {method}: {e}")
                results[f'BASELINE_{method}'] = {'error': str(e)}

    # Save comprehensive results
    results_df = pd.DataFrame(results).T
    results_df.to_csv(exp_dir / 'comprehensive_results.csv')

    with open(exp_dir / 'all_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*80}")
    print("COMPREHENSIVE COMPARISON COMPLETE!")
    print(f"Results saved to: {exp_dir}")
    print(f"{'='*80}\n")

    return results


def experiment_epistemic_aleatoric(data_path: str, pde_type: str,
                                   config_dict: dict,
                                   output_dir: str = 'experiments',
                                   reg_name: str = None) -> Dict:
    """
    Experiment 3: Epistemic vs Aleatoric Decomposition.

    Validates that epistemic uncertainty decreases with more training data
    while aleatoric uncertainty remains constant.

    Args:
        data_path: Path to data file
        pde_type: PDE type
        config_dict: Configuration dictionary
        output_dir: Output directory
        reg_name: Regularization scheme name (e.g., 'standard', 'improved', etc.)

    Returns:
        Dictionary with experimental results
    """
    # Setup output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = Path(output_dir) / f"experiment3_{pde_type}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"EXPERIMENT 3: Epistemic vs Aleatoric Decomposition")
    print(f"PDE: {pde_type}")
    print(f"Output: {exp_dir}")
    print(f"{'='*80}\n")

    # Load RAW dataset (NO LEAKAGE MODE)
    print("Loading RAW dataset...")
    X_raw, U_raw = load_pde_data(data_path, pde_type, max_samples=5000, return_raw=True)
    print(f"Total dataset size: {len(X_raw)}")

    # Create fixed train/val/test sets using NO LEAKAGE approach
    # This splits FIRST, then normalizes with TRAIN stats only
    config = Config(config_dict)
    from datautils import create_dataloaders_no_leakage
    full_train_loader, val_loader, test_loader = create_dataloaders_no_leakage(
        X_raw, U_raw,
        train_ratio=0.8,
        val_ratio=0.1,
        batch_size=config.batch_size,
        seed=config.seed if hasattr(config, 'seed') else 42
    )
    print(f"Full training set size: {len(full_train_loader.dataset)}")
    print(f"Fixed validation set size: {len(val_loader.dataset)}")
    print(f"Fixed test set size: {len(test_loader.dataset)}")

    # Get the full training dataset for creating subsets
    # We'll use this to create smaller training sets while keeping val/test fixed
    full_train_dataset = full_train_loader.dataset
    n_train_total = len(full_train_dataset)

    # Training data sizes to test
    train_sizes = [100, 200, 500, 1000, 2000, 5000]
    #train_sizes = [8000]
    # Filter out sizes larger than available training data
    train_sizes = [n for n in train_sizes if n <= n_train_total]

    print(f"\nTraining sizes to test: {train_sizes}")

    # Store results
    results = []

    # Train model for each data size
    for n_train in train_sizes:
        print(f"\n{'='*70}")
        print(f"Training with N={n_train} samples")
        print(f"{'='*70}")

        # Create subset loader by sampling ONLY from training dataset
        # This prevents data leakage into val/test sets
        rng = np.random.RandomState(seed=config.seed if hasattr(config, 'seed') else 42)
        sampled_indices = rng.choice(n_train_total, size=n_train, replace=False)

        # Create subset and loader from the training dataset
        train_subset = Subset(full_train_dataset, sampled_indices.tolist())
        train_loader = DataLoader(train_subset, batch_size=config.batch_size, shuffle=True)
        print(f"Created training subset: {n_train} samples from training set only")

        # Create evidential model
        model = EvidentialFNO2d(
            modes1=12, modes2=12, width=32, n_layers=4, in_channels=3,
            nu_min=1.0, alpha_min=1.0, beta_min=0.1
        )
        model = model.to(config.device)

        # Get method config for evidential training
        method_configs = get_evidential_methods_configs()
        method_config = method_configs.get('der_nig')

        # Enable UR-ERN (Uncertainty Regularization) to prevent gradient vanishing
        # From "Uncertainty Regularized Evidential Regression" (Oh et al., AAAI 2024)
        method_config['ur_weight'] = 0.0
        method_config['reg_weight'] = 0.001

        # Create optimizer and scheduler
        optimizer_config = {
            'type': 'adam',
            'lr': method_config.get('lr', 1e-4),
            'weight_decay': 1e-5
        }
        optimizer = Factory.create_optimizer(optimizer_config, model.parameters())

        scheduler_config = {
            'type': 'exponential_lr',
            'gamma': 0.93
        }
        scheduler = Factory.create_scheduler(scheduler_config, optimizer)

        # Setup trainer config
        trainer_config = Config({
            'device': config.device,
            'lr': method_config.get('lr', 1e-4),
            'weight_decay': 0.0,
            'enable_tracking': False,
            'log_dir': str(exp_dir / 'logs'),
            'experiment_name': f'exp3_n{n_train}',
            'checkpoint_dir': str(exp_dir / 'checkpoints'),
            'save_every': 50,
            'use_tensorboard': False
        })

        # Get regularization function if specified
        reg_fn = get_regularization_function(reg_name)
        if reg_name:
            print(f"Using regularization: {reg_name}")

        # Add max_epochs for annealed regularization
        epochs = config_dict.get('epochs', 200)
        if reg_name == 'annealed':
            method_config['max_epochs'] = epochs

        # Create trainer
        trainer = EvidentialFNOTrainer(
            model=model,
            config=trainer_config,
            method_name='der_nig',
            optimizer=optimizer,
            scheduler=scheduler,
            method_config=method_config,
            reg_fn=reg_fn
        )

        # Train - use val_loader for validation, not test_loader
        print(f"Training for {epochs} epochs...")
        trainer.train(train_loader, val_loader, epochs=epochs)

        # Evaluate uncertainty on test set
        print("\nEvaluating uncertainties on test set...")
        model.eval()
        all_aleatoric = []
        all_epistemic = []

        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(config.device), y.to(config.device)

                # Get evidential parameters
                gamma, nu, alpha, beta = model(x)

                # Compute uncertainties
                uq_dict = evidential_uncertainty(gamma, nu, alpha, beta)

                all_aleatoric.append(uq_dict['aleatoric'].cpu().numpy())
                all_epistemic.append(uq_dict['epistemic'].cpu().numpy())

        # Aggregate results
        aleatoric = np.concatenate(all_aleatoric).mean()
        epistemic = np.concatenate(all_epistemic).mean()
        ratio = epistemic / (aleatoric + 1e-8)

        print(f"Results for N={n_train}:")
        print(f"  Aleatoric: {aleatoric:.6f}")
        print(f"  Epistemic: {epistemic:.6f}")
        print(f"  Ratio (Epi/Ale): {ratio:.3f}")

        results.append({
            'n_train': n_train,
            'aleatoric': float(aleatoric),
            'epistemic': float(epistemic),
            'ratio': float(ratio)
        })

    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(exp_dir / 'experiment3_results.csv', index=False)
    print(f"\nResults saved to: {exp_dir / 'experiment3_results.csv'}")

    # Generate plot
    try:
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Uncertainties vs Training Size
        ax1.plot(results_df['n_train'], results_df['aleatoric'], 'o-', label='Aleatoric (σ²_ale)', linewidth=2)
        ax1.plot(results_df['n_train'], results_df['epistemic'], 's-', label='Epistemic (σ²_epi)', linewidth=2)
        ax1.set_xlabel('Training Data Size (N)', fontsize=12)
        ax1.set_ylabel('Uncertainty', fontsize=12)
        ax1.set_title('Epistemic vs Aleatoric Uncertainty', fontsize=14)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log')

        # Plot 2: Log-log plot to show σ²_epi ∝ 1/N
        ax2.loglog(results_df['n_train'], results_df['epistemic'], 's-', label='Epistemic (σ²_epi)', linewidth=2)
        ax2.loglog(results_df['n_train'], results_df['aleatoric'], 'o-', label='Aleatoric (σ²_ale)', linewidth=2)

        # Add reference line for 1/N slope
        n_vals = np.array(train_sizes)
        reference = results_df['epistemic'].iloc[0] * (train_sizes[0] / n_vals)
        ax2.plot(n_vals, reference, '--', color='gray', alpha=0.5, label='Reference: 1/N slope')

        ax2.set_xlabel('Training Data Size (N)', fontsize=12)
        ax2.set_ylabel('Uncertainty (log scale)', fontsize=12)
        ax2.set_title('Log-Log: Epistemic ∝ 1/N Validation', fontsize=14)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(exp_dir / 'experiment3_plot.png', dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {exp_dir / 'experiment3_plot.png'}")
        plt.close()

    except Exception as e:
        print(f"Warning: Could not generate plot: {e}")

    # Validation check
    print(f"\n{'='*70}")
    print("VALIDATION:")
    print(f"{'='*70}")

    # Check if epistemic decreases
    epi_start = results_df['epistemic'].iloc[0]
    epi_end = results_df['epistemic'].iloc[-1]
    epi_decrease = (epi_start - epi_end) / epi_start * 100

    print(f"Epistemic uncertainty: {epi_start:.6f} → {epi_end:.6f} ({epi_decrease:.1f}% decrease)")

    # Check if aleatoric remains constant (< 10% variation)
    ale_std = results_df['aleatoric'].std()
    ale_mean = results_df['aleatoric'].mean()
    ale_variation = (ale_std / ale_mean) * 100

    print(f"Aleatoric uncertainty: mean={ale_mean:.6f}, std={ale_std:.6f} ({ale_variation:.1f}% variation)")

    validation = {
        'epistemic_decrease_pct': float(epi_decrease),
        'aleatoric_variation_pct': float(ale_variation),
        'validation_passed': bool(epi_decrease > 20 and ale_variation < 15)
    }

    if validation['validation_passed']:
        print("\n✓ VALIDATION PASSED: Epistemic decreases, aleatoric constant")
    else:
        print("\n✗ VALIDATION FAILED: Check experiment parameters")

    # Save validation results
    with open(exp_dir / 'experiment3_validation.json', 'w') as f:
        json.dump(validation, f, indent=2)

    print(f"\n{'='*80}")
    print("EXPERIMENT 3 COMPLETE!")
    print(f"Results saved to: {exp_dir}")
    print(f"{'='*80}\n")

    return {
        'results': results,
        'validation': validation,
        'output_dir': str(exp_dir)
    }


def experiment_regularization_comparison(data_path: str, pde_type: str,
                                         config_dict: dict,
                                         output_dir: str = 'experiments') -> Dict:
    """
    Experiment 5: Regularization Comparison.

    Systematically compares all 7 regularization schemes for evidential methods:
    1. standard - Linear error penalty
    2. improved - Log barrier (bounded gradients)
    3. uncertainty_aware - Inverse uncertainty weighting
    4. annealed - Time-decaying weight
    5. l2_evidence - L2 penalty on evidence
    6. adaptive - Exponential error weighting
    7. kl_divergence - KL divergence from prior

    For each regularization, tracks:
    - Epistemic/Aleatoric uncertainty across training sizes
    - Prediction accuracy (MSE, MAE)
    - Calibration quality (ECE)
    - Uncertainty-error correlation

    Args:
        data_path: Path to data file
        pde_type: PDE type
        config_dict: Configuration dictionary
        output_dir: Output directory

    Returns:
        Dictionary with experimental results
    """
    # Setup output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = Path(output_dir) / f"experiment5_reg_comparison_{pde_type}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"EXPERIMENT 5: Regularization Comparison")
    print(f"PDE: {pde_type}")
    print(f"Output: {exp_dir}")
    print(f"{'='*80}\n")

    # All regularization schemes to compare
    regularizations = [
        'standard',
        'improved',
        'uncertainty_aware',
        #'annealed',
        'l2_evidence',
        'adaptive',
        'kl_divergence'
    ]

    print(f"Regularization schemes to compare: {len(regularizations)}")
    for i, reg in enumerate(regularizations, 1):
        print(f"  {i}. {reg}")
    print()

    # Load full dataset
    print("Loading dataset...")
    dataset = load_pde_data(data_path, pde_type, max_samples=5000)
    print(f"Total dataset size: {len(dataset)}")

    # Create fixed test set
    config = Config(config_dict)
    _, _, test_loader = create_dataloaders(
        dataset,
        train_ratio=0.8,
        val_ratio=0.1,
        batch_size=config.batch_size,
        seed=config.seed if hasattr(config, 'seed') else 42
    )
    print(f"Fixed test set size: {len(test_loader.dataset)}")

    # Training data sizes to test (same as Experiment 3)
    #train_sizes = [100, 200, 500, 1000, 2000, 5000]
    train_sizes = [8000]
    train_sizes = [n for n in train_sizes if n <= len(dataset)]
    print(f"\nTraining sizes to test: {train_sizes}")

    # Store all results
    all_results = []

    # Get method config for evidential training
    method_configs = get_evidential_methods_configs()
    method_config = method_configs.get('der_nig')

    # Loop over each regularization scheme
    for reg_name in regularizations:
        print(f"\n{'='*70}")
        print(f"REGULARIZATION: {reg_name}")
        print(f"{'='*70}")

        # Get regularization function
        reg_fn = get_regularization_function(reg_name)

        # Loop over training data sizes
        for n_train in train_sizes:
            print(f"\n  Training with N={n_train} samples...")

            # Create subset loader
            train_loader = create_subset_loader(
                dataset,
                n_samples=n_train,
                batch_size=config.batch_size,
                seed=config.seed if hasattr(config, 'seed') else 42
            )

            # Create evidential model
            model = EvidentialFNO2d(
                modes1=12, modes2=12, width=32, n_layers=4, in_channels=3,
                #nu_min=1.0, alpha_min=1.0, beta_min=0.0
                nu_min=0.01, alpha_min=1.01, beta_min=0.01
            )
            model = model.to(config.device)

            # Create optimizer and scheduler
            optimizer_config = {
                'type': 'adam',
                'lr': method_config.get('lr', 1e-4),
                'weight_decay': 0.0
            }
            optimizer = Factory.create_optimizer(optimizer_config, model.parameters())

            scheduler_config = {
                'type': 'exponential_lr',
                'gamma': 0.93
            }
            scheduler = Factory.create_scheduler(scheduler_config, optimizer)

            # Setup trainer config
            trainer_config = Config({
                'device': config.device,
                'lr': method_config.get('lr', 1e-4),
                'weight_decay': 0.0,
                'enable_tracking': False,
                'log_dir': str(exp_dir / 'logs'),
                'experiment_name': f'exp5_{reg_name}_n{n_train}',
                'checkpoint_dir': str(exp_dir / 'checkpoints'),
                'save_every': 50,
                'use_tensorboard': False
            })

            # Add max_epochs for annealed regularization
            epochs = config_dict.get('epochs', 200)
            if reg_name == 'annealed':
                method_config['max_epochs'] = epochs

            # Create trainer with regularization function
            trainer = EvidentialFNOTrainer(
                model=model,
                config=trainer_config,
                method_name='der_nig',
                optimizer=optimizer,
                scheduler=scheduler,
                method_config=method_config,
                reg_fn=reg_fn
            )

            # Train
            trainer.train(train_loader, test_loader, epochs=epochs)

            # Evaluate on test set
            print(f"  Evaluating on test set...")
            model.eval()
            all_aleatoric = []
            all_epistemic = []
            all_mse = []
            all_mae = []
            all_predictions = []
            all_targets = []
            all_total_unc = []

            with torch.no_grad():
                for x, y in test_loader:
                    x, y = x.to(config.device), y.to(config.device)

                    # Get evidential parameters
                    gamma, nu, alpha, beta = model(x)

                    # Compute uncertainties
                    uq_dict = evidential_uncertainty(gamma, nu, alpha, beta)

                    # Compute prediction errors
                    mse = F.mse_loss(gamma, y, reduction='none').mean(dim=(1, 2, 3))
                    mae = F.l1_loss(gamma, y, reduction='none').mean(dim=(1, 2, 3))

                    all_aleatoric.append(uq_dict['aleatoric'].cpu().numpy())
                    all_epistemic.append(uq_dict['epistemic'].cpu().numpy())
                    all_total_unc.append(uq_dict['total'].cpu().numpy())
                    all_mse.append(mse.cpu().numpy())
                    all_mae.append(mae.cpu().numpy())
                    all_predictions.append(gamma.cpu().numpy())
                    all_targets.append(y.cpu().numpy())

            # Aggregate results
            aleatoric = np.concatenate(all_aleatoric).mean()
            epistemic = np.concatenate(all_epistemic).mean()
            total_unc = np.concatenate(all_total_unc).mean()
            mse_val = np.concatenate(all_mse).mean()
            mae_val = np.concatenate(all_mae).mean()

            # Compute ECE (Expected Calibration Error)
            all_preds_flat = np.concatenate([p.flatten() for p in all_predictions])
            all_targets_flat = np.concatenate([t.flatten() for t in all_targets])
            all_unc_flat = np.concatenate([u.flatten() for u in all_total_unc])

            # Compute actual errors
            errors_flat = np.abs(all_preds_flat - all_targets_flat)

            # Compute calibration error
            # predicted_std = uncertainty, actual_error = absolute error
            ece = expected_calibration_error(
                predicted_std=torch.from_numpy(all_unc_flat),
                actual_error=torch.from_numpy(errors_flat),
                n_bins=10
            )

            # Compute uncertainty-error correlation
            corr = np.corrcoef(errors_flat, all_unc_flat)[0, 1]

            print(f"  Results: Ale={aleatoric:.6f}, Epi={epistemic:.6f}, "
                  f"MSE={mse_val:.6f}, MAE={mae_val:.6f}, ECE={ece:.6f}, Corr={corr:.3f}")

            all_results.append({
                'regularization': reg_name,
                'n_train': n_train,
                'aleatoric': float(aleatoric),
                'epistemic': float(epistemic),
                'total_uncertainty': float(total_unc),
                'mse': float(mse_val),
                'mae': float(mae_val),
                'ece': float(ece),
                'uncertainty_error_corr': float(corr)
            })

    # Save results to CSV
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(exp_dir / 'experiment5_results.csv', index=False)
    print(f"\n{'='*70}")
    print(f"Results saved to: {exp_dir / 'experiment5_results.csv'}")
    print(f"{'='*70}")

    # Generate comparison plots
    print("\nGenerating comparison plots...")

    try:
        # Create comprehensive comparison figure
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # Define colors for each regularization
        colors = plt.cm.tab10(np.linspace(0, 1, len(regularizations)))
        color_map = {reg: colors[i] for i, reg in enumerate(regularizations)}

        # Plot 1: Epistemic Uncertainty vs Training Size
        ax1 = fig.add_subplot(gs[0, 0])
        for reg_name in regularizations:
            reg_data = results_df[results_df['regularization'] == reg_name]
            ax1.plot(reg_data['n_train'], reg_data['epistemic'], 'o-',
                    label=reg_name, color=color_map[reg_name], linewidth=2, markersize=6)
        ax1.set_xlabel('Training Data Size (N)', fontsize=11)
        ax1.set_ylabel('Epistemic Uncertainty', fontsize=11)
        ax1.set_title('Epistemic vs N (should ∝ 1/N)', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=8, ncol=2)
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log')

        # Plot 2: Aleatoric Uncertainty vs Training Size
        ax2 = fig.add_subplot(gs[0, 1])
        for reg_name in regularizations:
            reg_data = results_df[results_df['regularization'] == reg_name]
            ax2.plot(reg_data['n_train'], reg_data['aleatoric'], 's-',
                    label=reg_name, color=color_map[reg_name], linewidth=2, markersize=6)
        ax2.set_xlabel('Training Data Size (N)', fontsize=11)
        ax2.set_ylabel('Aleatoric Uncertainty', fontsize=11)
        ax2.set_title('Aleatoric vs N (should be constant)', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=8, ncol=2)
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log')

        # Plot 3: Log-log Epistemic (validate 1/N slope)
        ax3 = fig.add_subplot(gs[0, 2])
        for reg_name in regularizations:
            reg_data = results_df[results_df['regularization'] == reg_name]
            ax3.loglog(reg_data['n_train'], reg_data['epistemic'], 'o-',
                      label=reg_name, color=color_map[reg_name], linewidth=2, markersize=6)
        # Add reference 1/N line
        n_vals = np.array(train_sizes)
        ref_epi = results_df.groupby('regularization')['epistemic'].first().mean()
        reference = ref_epi * (train_sizes[0] / n_vals)
        ax3.plot(n_vals, reference, '--', color='black', alpha=0.5, linewidth=2, label='1/N reference')
        ax3.set_xlabel('Training Data Size (N)', fontsize=11)
        ax3.set_ylabel('Epistemic (log scale)', fontsize=11)
        ax3.set_title('Log-Log: Epistemic ∝ 1/N Validation', fontsize=12, fontweight='bold')
        ax3.legend(fontsize=8, ncol=2)
        ax3.grid(True, alpha=0.3)

        # Plot 4: MSE vs Training Size
        ax4 = fig.add_subplot(gs[1, 0])
        for reg_name in regularizations:
            reg_data = results_df[results_df['regularization'] == reg_name]
            ax4.plot(reg_data['n_train'], reg_data['mse'], 'o-',
                    label=reg_name, color=color_map[reg_name], linewidth=2, markersize=6)
        ax4.set_xlabel('Training Data Size (N)', fontsize=11)
        ax4.set_ylabel('Mean Squared Error', fontsize=11)
        ax4.set_title('Prediction Accuracy (MSE)', fontsize=12, fontweight='bold')
        ax4.legend(fontsize=8, ncol=2)
        ax4.grid(True, alpha=0.3)
        ax4.set_xscale('log')
        ax4.set_yscale('log')

        # Plot 5: MAE vs Training Size
        ax5 = fig.add_subplot(gs[1, 1])
        for reg_name in regularizations:
            reg_data = results_df[results_df['regularization'] == reg_name]
            ax5.plot(reg_data['n_train'], reg_data['mae'], 's-',
                    label=reg_name, color=color_map[reg_name], linewidth=2, markersize=6)
        ax5.set_xlabel('Training Data Size (N)', fontsize=11)
        ax5.set_ylabel('Mean Absolute Error', fontsize=11)
        ax5.set_title('Prediction Accuracy (MAE)', fontsize=12, fontweight='bold')
        ax5.legend(fontsize=8, ncol=2)
        ax5.grid(True, alpha=0.3)
        ax5.set_xscale('log')
        ax5.set_yscale('log')

        # Plot 6: ECE (Calibration Error) vs Training Size
        ax6 = fig.add_subplot(gs[1, 2])
        for reg_name in regularizations:
            reg_data = results_df[results_df['regularization'] == reg_name]
            ax6.plot(reg_data['n_train'], reg_data['ece'], 'o-',
                    label=reg_name, color=color_map[reg_name], linewidth=2, markersize=6)
        ax6.set_xlabel('Training Data Size (N)', fontsize=11)
        ax6.set_ylabel('Expected Calibration Error', fontsize=11)
        ax6.set_title('Calibration Quality (lower is better)', fontsize=12, fontweight='bold')
        ax6.legend(fontsize=8, ncol=2)
        ax6.grid(True, alpha=0.3)
        ax6.set_xscale('log')

        # Plot 7: Uncertainty-Error Correlation vs Training Size
        ax7 = fig.add_subplot(gs[2, 0])
        for reg_name in regularizations:
            reg_data = results_df[results_df['regularization'] == reg_name]
            ax7.plot(reg_data['n_train'], reg_data['uncertainty_error_corr'], 'o-',
                    label=reg_name, color=color_map[reg_name], linewidth=2, markersize=6)
        ax7.set_xlabel('Training Data Size (N)', fontsize=11)
        ax7.set_ylabel('Uncertainty-Error Correlation', fontsize=11)
        ax7.set_title('Uncertainty Quality (higher is better)', fontsize=12, fontweight='bold')
        ax7.legend(fontsize=8, ncol=2)
        ax7.grid(True, alpha=0.3)
        ax7.set_xscale('log')
        ax7.axhline(0, color='black', linestyle='--', alpha=0.3)

        # Plot 8: Final Performance Comparison (N=max)
        ax8 = fig.add_subplot(gs[2, 1])
        max_n_data = results_df[results_df['n_train'] == max(train_sizes)]
        x_pos = np.arange(len(regularizations))
        mse_vals = [max_n_data[max_n_data['regularization'] == reg]['mse'].values[0]
                   for reg in regularizations]
        bars = ax8.bar(x_pos, mse_vals, color=[color_map[reg] for reg in regularizations])
        ax8.set_xticks(x_pos)
        ax8.set_xticklabels(regularizations, rotation=45, ha='right', fontsize=9)
        ax8.set_ylabel('MSE', fontsize=11)
        ax8.set_title(f'Final MSE Comparison (N={max(train_sizes)})', fontsize=12, fontweight='bold')
        ax8.grid(True, alpha=0.3, axis='y')

        # Plot 9: Final ECE Comparison (N=max)
        ax9 = fig.add_subplot(gs[2, 2])
        ece_vals = [max_n_data[max_n_data['regularization'] == reg]['ece'].values[0]
                   for reg in regularizations]
        bars = ax9.bar(x_pos, ece_vals, color=[color_map[reg] for reg in regularizations])
        ax9.set_xticks(x_pos)
        ax9.set_xticklabels(regularizations, rotation=45, ha='right', fontsize=9)
        ax9.set_ylabel('ECE', fontsize=11)
        ax9.set_title(f'Final ECE Comparison (N={max(train_sizes)})', fontsize=12, fontweight='bold')
        ax9.grid(True, alpha=0.3, axis='y')

        plt.suptitle('Experiment 5: Comprehensive Regularization Comparison',
                    fontsize=16, fontweight='bold', y=0.995)
        plt.savefig(exp_dir / 'experiment5_comparison.png', dpi=150, bbox_inches='tight')
        print(f"Comparison plot saved to: {exp_dir / 'experiment5_comparison.png'}")
        plt.close()

    except Exception as e:
        print(f"Warning: Could not generate comparison plot: {e}")

    # Generate summary statistics
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)

    summary = []
    for reg_name in regularizations:
        reg_data = results_df[results_df['regularization'] == reg_name]

        # Check epistemic decrease
        epi_start = reg_data['epistemic'].iloc[0]
        epi_end = reg_data['epistemic'].iloc[-1]
        epi_decrease_pct = (epi_start - epi_end) / epi_start * 100

        # Check aleatoric variation
        ale_std = reg_data['aleatoric'].std()
        ale_mean = reg_data['aleatoric'].mean()
        ale_variation_pct = (ale_std / ale_mean) * 100

        # Get final performance
        final_mse = reg_data['mse'].iloc[-1]
        final_ece = reg_data['ece'].iloc[-1]
        final_corr = reg_data['uncertainty_error_corr'].iloc[-1]

        summary.append({
            'regularization': reg_name,
            'epistemic_decrease_pct': float(epi_decrease_pct),
            'aleatoric_variation_pct': float(ale_variation_pct),
            'final_mse': float(final_mse),
            'final_ece': float(final_ece),
            'final_unc_error_corr': float(final_corr),
            'validation_passed': bool(epi_decrease_pct > 20 and ale_variation_pct < 15)
        })

        print(f"\n{reg_name}:")
        print(f"  Epistemic decrease: {epi_decrease_pct:.1f}%")
        print(f"  Aleatoric variation: {ale_variation_pct:.1f}%")
        print(f"  Final MSE: {final_mse:.6f}")
        print(f"  Final ECE: {final_ece:.6f}")
        print(f"  Final Corr: {final_corr:.3f}")
        print(f"  Validation: {'PASSED' if summary[-1]['validation_passed'] else 'FAILED'}")

    # Save summary
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(exp_dir / 'experiment5_summary.csv', index=False)
    print(f"\nSummary saved to: {exp_dir / 'experiment5_summary.csv'}")

    # Find best regularization
    print("\n" + "="*70)
    print("BEST REGULARIZATION SCHEMES")
    print("="*70)

    best_mse = summary_df.loc[summary_df['final_mse'].idxmin()]
    best_ece = summary_df.loc[summary_df['final_ece'].idxmin()]
    best_corr = summary_df.loc[summary_df['final_unc_error_corr'].idxmax()]

    print(f"\nBest MSE: {best_mse['regularization']} ({best_mse['final_mse']:.6f})")
    print(f"Best ECE: {best_ece['regularization']} ({best_ece['final_ece']:.6f})")
    print(f"Best Uncertainty-Error Correlation: {best_corr['regularization']} ({best_corr['final_unc_error_corr']:.3f})")

    print(f"\n{'='*80}")
    print("EXPERIMENT 5 COMPLETE!")
    print(f"Results saved to: {exp_dir}")
    print(f"{'='*80}\n")

    return {
        'results': all_results,
        'summary': summary,
        'output_dir': str(exp_dir),
        'best_schemes': {
            'mse': best_mse['regularization'],
            'ece': best_ece['regularization'],
            'correlation': best_corr['regularization']
        }
    }


def experiment_ablation(data_path: str, pde_type: str,
                        config_dict: dict,
                        output_dir: str = 'experiments') -> Dict:
    """
    Experiment 8: Ablation Studies.

    Tests the impact of removing architectural components from EvidentialFNO2d:
    - Skip connections (Fourier + Conv)
    - Activation functions (GELU)
    - Evidential head complexity (deep/shallow/linear)
    - Batch normalization
    - Fourier-only mode
    - Residual connections

    Args:
        data_path: Path to data file
        pde_type: PDE type
        config_dict: Configuration dictionary
        output_dir: Output directory

    Returns:
        Dictionary with experimental results
    """
    # Setup output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = Path(output_dir) / f"experiment8_ablation_{pde_type}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"EXPERIMENT 8: Ablation Studies")
    print(f"PDE: {pde_type}")
    print(f"Output: {exp_dir}")
    print(f"{'='*80}\n")

    # Load RAW dataset (NO LEAKAGE MODE)
    print("Loading RAW dataset...")
    X_raw, U_raw = load_pde_data(data_path, pde_type, max_samples=5000, return_raw=True)
    print(f"Total dataset size: {len(X_raw)}")

    # Create train/val/test splits with NO LEAKAGE
    # This splits FIRST, then normalizes with TRAIN stats only
    config = Config(config_dict)
    from datautils import create_dataloaders_no_leakage
    train_loader, val_loader, test_loader = create_dataloaders_no_leakage(
        X_raw, U_raw,
        train_ratio=0.7,
        val_ratio=0.15,
        batch_size=config.batch_size,
        seed=42
    )

    print(f"Train size: {len(train_loader.dataset)}")
    print(f"Val size: {len(val_loader.dataset)}")
    print(f"Test size: {len(test_loader.dataset)}")

    # Define ablation configurations
    ablation_configs = [
        {
            'name': 'baseline',
            'description': 'Full model with all components',
            'use_skip_connections': True,
            'use_activations': F.gelu,
            'head_depth': 'deep',
            'use_batch_norm': False,
            'fourier_only': False,
            'residual_connection': False
        },
        {
            'name': 'no_skip_connections',
            'description': 'Remove skip connections (Fourier only in layers)',
            'use_skip_connections': False,
            'use_activations': F.gelu,
            'head_depth': 'deep',
            'use_batch_norm': False,
            'fourier_only': False,
            'residual_connection': False
        },
        {
            'name': 'no_activations',
            'description': 'Remove GELU activations between layers',
            'use_skip_connections': True,
            'use_activations': None,
            'head_depth': 'deep',
            'use_batch_norm': False,
            'fourier_only': False,
            'residual_connection': False
        },
        {
            'name': 'shallow_heads',
            'description': 'Use shallow evidential heads (64 hidden)',
            'use_skip_connections': True,
            'use_activations': F.gelu,
            'head_depth': 'shallow',
            'use_batch_norm': False,
            'fourier_only': False,
            'residual_connection': False
        },
        {
            'name': 'linear_heads',
            'description': 'Use linear evidential heads (no hidden layer)',
            'use_skip_connections': True,
            'use_activations': F.gelu,
            'head_depth': 'linear',
            'use_batch_norm': False,
            'fourier_only': False,
            'residual_connection': False
        },
        {
            'name': 'with_batch_norm',
            'description': 'Add batch normalization after each layer',
            'use_skip_connections': True,
            'use_activations': F.gelu,
            'head_depth': 'deep',
            'use_batch_norm': True,
            'fourier_only': False,
            'residual_connection': False
        },
        {
            'name': 'fourier_only',
            'description': 'Remove all Conv layers (pure Fourier)',
            'use_skip_connections': True,  # Irrelevant when fourier_only=True
            'use_activations': F.gelu,
            'head_depth': 'deep',
            'use_batch_norm': False,
            'fourier_only': True,
            'residual_connection': False
        },
        {
            'name': 'with_residual',
            'description': 'Add residual connection from input to output',
            'use_skip_connections': True,
            'use_activations': F.gelu,
            'head_depth': 'deep',
            'use_batch_norm': False,
            'fourier_only': False,
            'residual_connection': True
        }
    ]

    # Train and evaluate each ablation variant
    results = []
    method_config = get_evidential_methods_configs()['der_nig']
    epochs = config_dict.get('epochs', 100)

    for idx, ablation_cfg in enumerate(ablation_configs):
        print(f"\n{'='*80}")
        print(f"Ablation {idx+1}/{len(ablation_configs)}: {ablation_cfg['name']}")
        print(f"Description: {ablation_cfg['description']}")
        print(f"{'='*80}\n")

        # Create model with ablation configuration
        model = AblationEvidentialFNO2d(
            modes1=12,
            modes2=12,
            width=32,
            n_layers=4,
            in_channels=3,
            nu_min=1.0,
            alpha_min=1.0,
            beta_min=0.0,
            use_skip_connections=ablation_cfg['use_skip_connections'],
            use_activations=ablation_cfg['use_activations'],
            head_depth=ablation_cfg['head_depth'],
            use_batch_norm=ablation_cfg['use_batch_norm'],
            fourier_only=ablation_cfg['fourier_only'],
            residual_connection=ablation_cfg['residual_connection']
        )
        model = model.to(config.device)

        # Count parameters
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model parameters: {n_params:,}")

        # Setup optimizer and scheduler
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=False
        )

        # Setup trainer
        trainer_config = Config({
            'device': config.device,
            'batch_size': config.batch_size,
            'lr': 1e-3,
            'weight_decay': 1e-5,
            'checkpoint_dir': str(exp_dir / ablation_cfg['name']),
            'save_every': epochs + 1  # Don't save periodic checkpoints
        })

        trainer = EvidentialFNOTrainer(
            model=model,
            config=trainer_config,
            method_name='der_nig',
            optimizer=optimizer,
            scheduler=scheduler,
            method_config=method_config,
            save_flag=False  # Don't save checkpoints for ablation
        )

        # Train
        print(f"Training for {epochs} epochs...")
        trainer.train(train_loader, val_loader, epochs=epochs)

        # Evaluate on test set
        print("\nEvaluating on test set...")
        model.eval()

        all_predictions = []
        all_targets = []
        all_epistemic = []
        all_aleatoric = []
        all_total_unc = []

        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(config.device), y.to(config.device)

                # Forward pass
                gamma, nu, alpha, beta = model(x)

                # Compute uncertainties
                uq_dict = evidential_uncertainty(gamma, nu, alpha, beta)

                all_predictions.append(gamma.cpu())
                all_targets.append(y.cpu())
                all_epistemic.append(uq_dict['epistemic'].cpu())
                all_aleatoric.append(uq_dict['aleatoric'].cpu())
                all_total_unc.append(uq_dict['total'].cpu())

        # Concatenate results
        predictions = torch.cat(all_predictions, dim=0)
        targets = torch.cat(all_targets, dim=0)
        epistemic = torch.cat(all_epistemic, dim=0)
        aleatoric = torch.cat(all_aleatoric, dim=0)
        total_unc = torch.cat(all_total_unc, dim=0)

        # Compute metrics
        mse = F.mse_loss(predictions, targets).item()
        mae = F.l1_loss(predictions, targets).item()
        rel_l2 = (torch.norm(predictions - targets) / torch.norm(targets)).item()

        # Uncertainty metrics
        mean_epistemic = epistemic.mean().item()
        mean_aleatoric = aleatoric.mean().item()
        mean_total = total_unc.mean().item()

        # Compute errors
        errors = (predictions - targets).abs()

        # Calibration error
        ece = expected_calibration_error(total_unc, errors, 10)

        # Uncertainty-error correlation
        unc_err_corr = uncertainty_error_correlation(total_unc, errors)

        # Convert activation function to string for JSON serialization
        activation_str = None
        if ablation_cfg['use_activations'] is not None:
            if ablation_cfg['use_activations'] == F.gelu:
                activation_str = 'gelu'
            elif callable(ablation_cfg['use_activations']):
                activation_str = ablation_cfg['use_activations'].__name__

        # Store results
        result = {
            'name': ablation_cfg['name'],
            'description': ablation_cfg['description'],
            'n_params': n_params,
            'mse': mse,
            'mae': mae,
            'rel_l2': rel_l2,
            'epistemic': mean_epistemic,
            'aleatoric': mean_aleatoric,
            'total_uncertainty': mean_total,
            'calibration_error': ece,
            'unc_err_correlation': unc_err_corr,
            'final_train_loss': trainer.train_losses[-1] if trainer.train_losses else 0.0,
            'final_val_loss': trainer.val_losses[-1] if trainer.val_losses else 0.0,
            # Include ablation flags with JSON-serializable values
            'use_skip_connections': ablation_cfg['use_skip_connections'],
            'use_activations': activation_str,
            'head_depth': ablation_cfg['head_depth'],
            'use_batch_norm': ablation_cfg['use_batch_norm'],
            'fourier_only': ablation_cfg['fourier_only'],
            'residual_connection': ablation_cfg['residual_connection']
        }

        results.append(result)

        print(f"\nResults for {ablation_cfg['name']}:")
        print(f"  MSE: {mse:.6f}")
        print(f"  MAE: {mae:.6f}")
        print(f"  Rel L2: {rel_l2:.6f}")
        print(f"  Epistemic: {mean_epistemic:.6f}")
        print(f"  Aleatoric: {mean_aleatoric:.6f}")
        print(f"  Calibration Error: {ece:.6f}")
        print(f"  Unc-Err Correlation: {unc_err_corr:.6f}")

    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(exp_dir / 'experiment8_ablation_results.csv', index=False)
    print(f"\n\nResults saved to: {exp_dir / 'experiment8_ablation_results.csv'}")

    # Create comparison plots
    print("\nGenerating comparison plots...")

    # 1. Performance comparison (MSE, MAE, Rel L2)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    names = [r['name'] for r in results]
    mses = [r['mse'] for r in results]
    maes = [r['mae'] for r in results]
    rel_l2s = [r['rel_l2'] for r in results]

    # Highlight baseline
    colors = ['green' if name == 'baseline' else 'steelblue' for name in names]

    axes[0].bar(range(len(names)), mses, color=colors)
    axes[0].set_xlabel('Ablation Variant')
    axes[0].set_ylabel('MSE')
    axes[0].set_title('Mean Squared Error')
    axes[0].set_xticks(range(len(names)))
    axes[0].set_xticklabels(names, rotation=45, ha='right')
    axes[0].grid(axis='y', alpha=0.3)

    axes[1].bar(range(len(names)), maes, color=colors)
    axes[1].set_xlabel('Ablation Variant')
    axes[1].set_ylabel('MAE')
    axes[1].set_title('Mean Absolute Error')
    axes[1].set_xticks(range(len(names)))
    axes[1].set_xticklabels(names, rotation=45, ha='right')
    axes[1].grid(axis='y', alpha=0.3)

    axes[2].bar(range(len(names)), rel_l2s, color=colors)
    axes[2].set_xlabel('Ablation Variant')
    axes[2].set_ylabel('Relative L2 Error')
    axes[2].set_title('Relative L2 Error')
    axes[2].set_xticks(range(len(names)))
    axes[2].set_xticklabels(names, rotation=45, ha='right')
    axes[2].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(exp_dir / 'experiment8_performance_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Uncertainty comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    epistemics = [r['epistemic'] for r in results]
    aleatorics = [r['aleatoric'] for r in results]
    totals = [r['total_uncertainty'] for r in results]

    axes[0].bar(range(len(names)), epistemics, color=colors)
    axes[0].set_xlabel('Ablation Variant')
    axes[0].set_ylabel('Epistemic Uncertainty')
    axes[0].set_title('Epistemic Uncertainty')
    axes[0].set_xticks(range(len(names)))
    axes[0].set_xticklabels(names, rotation=45, ha='right')
    axes[0].grid(axis='y', alpha=0.3)

    axes[1].bar(range(len(names)), aleatorics, color=colors)
    axes[1].set_xlabel('Ablation Variant')
    axes[1].set_ylabel('Aleatoric Uncertainty')
    axes[1].set_title('Aleatoric Uncertainty')
    axes[1].set_xticks(range(len(names)))
    axes[1].set_xticklabels(names, rotation=45, ha='right')
    axes[1].grid(axis='y', alpha=0.3)

    axes[2].bar(range(len(names)), totals, color=colors)
    axes[2].set_xlabel('Ablation Variant')
    axes[2].set_ylabel('Total Uncertainty')
    axes[2].set_title('Total Uncertainty')
    axes[2].set_xticks(range(len(names)))
    axes[2].set_xticklabels(names, rotation=45, ha='right')
    axes[2].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(exp_dir / 'experiment8_uncertainty_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 3. Calibration and correlation comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    eces = [r['calibration_error'] for r in results]
    corrs = [r['unc_err_correlation'] for r in results]

    axes[0].bar(range(len(names)), eces, color=colors)
    axes[0].set_xlabel('Ablation Variant')
    axes[0].set_ylabel('Expected Calibration Error')
    axes[0].set_title('Calibration Quality (lower is better)')
    axes[0].set_xticks(range(len(names)))
    axes[0].set_xticklabels(names, rotation=45, ha='right')
    axes[0].grid(axis='y', alpha=0.3)

    axes[1].bar(range(len(names)), corrs, color=colors)
    axes[1].set_xlabel('Ablation Variant')
    axes[1].set_ylabel('Uncertainty-Error Correlation')
    axes[1].set_title('Uncertainty-Error Correlation (higher is better)')
    axes[1].set_xticks(range(len(names)))
    axes[1].set_xticklabels(names, rotation=45, ha='right')
    axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.5, label='No correlation')
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(exp_dir / 'experiment8_calibration_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 4. Performance degradation table
    baseline_result = results[0]  # First config is baseline

    print("\n" + "="*80)
    print("ABLATION STUDY SUMMARY")
    print("="*80)
    print(f"\n{'Variant':<25} {'MSE Δ%':<12} {'Rel L2 Δ%':<12} {'ECE Δ%':<12} {'Corr Δ%':<12}")
    print("-"*80)

    for r in results:
        if r['name'] == 'baseline':
            print(f"{r['name']:<25} {'baseline':<12} {'baseline':<12} {'baseline':<12} {'baseline':<12}")
        else:
            mse_delta = ((r['mse'] - baseline_result['mse']) / baseline_result['mse']) * 100
            rel_l2_delta = ((r['rel_l2'] - baseline_result['rel_l2']) / baseline_result['rel_l2']) * 100
            ece_delta = ((r['calibration_error'] - baseline_result['calibration_error']) / baseline_result['calibration_error']) * 100
            corr_delta = ((r['unc_err_correlation'] - baseline_result['unc_err_correlation']) / baseline_result['unc_err_correlation']) * 100

            print(f"{r['name']:<25} {mse_delta:+.2f}%       {rel_l2_delta:+.2f}%       {ece_delta:+.2f}%       {corr_delta:+.2f}%")

    print("="*80)

    # Save summary to JSON
    summary = {
        'baseline': baseline_result,
        'ablation_variants': results[1:],
        'experiment_config': {
            'pde_type': pde_type,
            'epochs': epochs,
            'train_size': len(train_loader.dataset),
            'val_size': len(val_loader.dataset),
            'test_size': len(test_loader.dataset)
        }
    }

    with open(exp_dir / 'experiment8_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*80}")
    print("ABLATION STUDY COMPLETE!")
    print(f"Results saved to: {exp_dir}")
    print(f"{'='*80}\n")

    return {
        'results': results,
        'summary': summary,
        'output_dir': str(exp_dir)
    }


def experiment_ood_detection(id_data_path: str, ood_data_path: str,
                             config_dict: dict,
                             output_dir: str = 'experiments',
                             reg_name: str = None) -> Dict:
    """
    Experiment 4: OOD Detection using Reynolds Number.

    Trains evidential model on ID data (Re=1000, 2000, 3000) and evaluates
    OOD detection performance on higher Reynolds numbers (Re=5000, 10000).
    Uses total uncertainty as OOD score and computes AUROC.

    Args:
        id_data_path: Path to in-distribution training data
        ood_data_path: Path to out-of-distribution test data
        config_dict: Configuration dictionary
        output_dir: Output directory
        reg_name: Regularization scheme name (e.g., 'standard', 'improved', etc.)

    Returns:
        Dictionary with experimental results including AUROC
    """
    # Setup output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = Path(output_dir) / f"experiment4_ood_detection_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"EXPERIMENT 4: OOD Detection via Reynolds Number")
    print(f"Output: {exp_dir}")
    print(f"{'='*80}\n")

    # Load ID training data (Re=1000, 2000, 3000) - RAW MODE (NO LEAKAGE)
    print("Loading ID training data (Re=1000, 2000, 3000) in RAW mode...")
    X_id_raw, U_id_raw, id_reynolds = load_ns_exp4(
        filepath=id_data_path,
        reynolds_numbers=[1000.0, 2000.0, 3000.0],
        time_pairs=5,
        return_raw=True
    )
    print(f"ID dataset size: {len(X_id_raw)}")
    print(f"Reynolds distribution: {np.unique(id_reynolds, return_counts=True)}")

    # Create ID train/val/test splits with NO LEAKAGE
    # This computes normalization on TRAINING DATA ONLY
    config = Config(config_dict)
    from datautils import create_dataloaders_no_leakage
    train_loader, val_loader, id_test_loader = create_dataloaders_no_leakage(
        X_id_raw, U_id_raw,
        train_ratio=0.8,
        val_ratio=0.1,
        batch_size=config.batch_size,
        seed=42
    )

    # Get the normalization stats from training data for OOD dataset
    train_dataset = train_loader.dataset
    u_mean = train_dataset.u_mean
    u_std = train_dataset.u_std
    x_min = train_dataset.x_min
    x_max = train_dataset.x_max
    y_min = train_dataset.y_min
    y_max = train_dataset.y_max
    x_fields_mean = train_dataset.x_fields_mean
    x_fields_std = train_dataset.x_fields_std

    # Load OOD test data (Re=5000, 10000) - RAW MODE
    print("\nLoading OOD test data (Re=5000, 10000) in RAW mode...")
    X_ood_raw, U_ood_raw, ood_reynolds = load_ns_exp4(
        filepath=ood_data_path,
        reynolds_numbers=[5000.0, 10000.0],
        time_pairs=5,
        return_raw=True
    )
    print(f"OOD dataset size: {len(X_ood_raw)}")
    print(f"Reynolds distribution: {np.unique(ood_reynolds, return_counts=True)}")

    # Create OOD dataset using SAME normalization stats as ID training data
    print("\nCreating OOD dataset with ID training normalization stats...")
    ood_dataset = PDEDataset(
        X_ood_raw, U_ood_raw,
        normalize_output=True,
        normalize_input=True,
        normalize_coords=True,
        precomputed_u_mean=u_mean,  # ← Using ID TRAIN stats!
        precomputed_u_std=u_std,
        precomputed_x_min=x_min,
        precomputed_x_max=x_max,
        precomputed_y_min=y_min,
        precomputed_y_max=y_max,
        precomputed_x_fields_mean=x_fields_mean if x_fields_mean else None,
        precomputed_x_fields_std=x_fields_std if x_fields_std else None
    )
    ood_test_loader = DataLoader(
        ood_dataset,
        batch_size=config.batch_size,
        shuffle=False
    )

    print(f"\nTrain: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")
    print(f"ID Test: {len(id_test_loader.dataset)}, OOD Test: {len(ood_test_loader.dataset)}")

    # Create evidential model
    print("\nInitializing evidential model...")
    model = EvidentialFNO2d(
        modes1=12,
        modes2=12,
        width=64,  # Increased from 32 to 64 for better capacity
        n_layers=4,
        in_channels=3,
        nu_min=1.0,
        alpha_min=1.0,
        beta_min=0.01
    )
    model = model.to(config.device)

    # Get method config
    method_configs = get_evidential_methods_configs()
    method_config = method_configs.get('der_nig')

    # Enable UR-ERN (Uncertainty Regularization) to prevent gradient vanishing
    # From "Uncertainty Regularized Evidential Regression" (Oh et al., AAAI 2024)
    method_config['ur_weight'] = 0.001

    # Get regularization function if specified
    reg_fn = get_regularization_function(reg_name)
    if reg_name:
        print(f"Using regularization: {reg_name}")

    # Add max_epochs for annealed regularization
    epochs = config_dict.get('epochs', 200)  # Increased from 100 to 200 for better uncertainty learning
    if reg_name == 'annealed':
        method_config['max_epochs'] = epochs

    # Setup optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )

    # Setup trainer config
    trainer_config = Config({
        'device': config.device,
        'lr': 1e-3,
        'weight_decay': 1e-5,
        'enable_tracking': False,
        'log_dir': str(exp_dir / 'logs'),
        'experiment_name': 'exp4_ood_detection',
        'checkpoint_dir': str(exp_dir / 'checkpoints'),
        'save_every': 1000,
        'use_tensorboard': False
    })

    # Create trainer
    trainer = EvidentialFNOTrainer(
        model=model,
        config=trainer_config,
        method_name='der_nig',
        optimizer=optimizer,
        scheduler=scheduler,
        method_config=method_config,
        reg_fn=reg_fn,
        save_flag=False
    )

    # Train
    epochs = config_dict.get('epochs', 200)  # Increased from 100 to 200 for better uncertainty learning
    print(f"\nTraining evidential model for {epochs} epochs...")
    trainer.train(train_loader, val_loader, epochs=epochs)

    # Evaluate on ID test set
    print("\nEvaluating on ID test set (same distribution)...")
    model.eval()

    id_uncertainties = []
    id_errors = []
    id_predictions = []
    id_targets = []

    with torch.no_grad():
        for x, y in id_test_loader:
            x, y = x.to(config.device), y.to(config.device)

            # Get evidential parameters
            gamma, nu, alpha, beta = model(x)

            # Compute uncertainties
            uq_dict = evidential_uncertainty(gamma, nu, alpha, beta)
            total_unc = uq_dict['total']

            # Compute errors
            error = (gamma - y).abs()

            id_uncertainties.append(total_unc.cpu().numpy())
            id_errors.append(error.cpu().numpy())
            id_predictions.append(gamma.cpu().numpy())
            id_targets.append(y.cpu().numpy())

    id_uncertainties = np.concatenate(id_uncertainties)
    id_errors = np.concatenate(id_errors)
    id_predictions = np.concatenate(id_predictions)
    id_targets = np.concatenate(id_targets)

    # Evaluate on OOD test set
    print("Evaluating on OOD test set (Re=5000, 10000)...")
    ood_uncertainties = []
    ood_errors = []
    ood_predictions = []
    ood_targets = []

    with torch.no_grad():
        for x, y in ood_test_loader:
            x, y = x.to(config.device), y.to(config.device)

            # Get evidential parameters
            gamma, nu, alpha, beta = model(x)

            # Compute uncertainties
            uq_dict = evidential_uncertainty(gamma, nu, alpha, beta)
            total_unc = uq_dict['total']

            # Compute errors
            error = (gamma - y).abs()

            ood_uncertainties.append(total_unc.cpu().numpy())
            ood_errors.append(error.cpu().numpy())
            ood_predictions.append(gamma.cpu().numpy())
            ood_targets.append(y.cpu().numpy())

    ood_uncertainties = np.concatenate(ood_uncertainties)
    ood_errors = np.concatenate(ood_errors)
    ood_predictions = np.concatenate(ood_predictions)
    ood_targets = np.concatenate(ood_targets)

    # Compute metrics
    id_mse = ((id_predictions - id_targets)**2).mean()
    ood_mse = ((ood_predictions - ood_targets)**2).mean()

    id_mean_unc = id_uncertainties.mean()
    ood_mean_unc = ood_uncertainties.mean()

    print(f"\n{'='*70}")
    print("RESULTS:")
    print(f"{'='*70}")
    print(f"ID Test MSE: {id_mse:.6f}")
    print(f"OOD Test MSE: {ood_mse:.6f}")
    print(f"ID Mean Uncertainty: {id_mean_unc:.6f}")
    print(f"OOD Mean Uncertainty: {ood_mean_unc:.6f}")
    print(f"Uncertainty Ratio (OOD/ID): {ood_mean_unc / (id_mean_unc + 1e-8):.3f}")

    # Compute AUROC for OOD detection
    # Label: 0 for ID, 1 for OOD
    # Score: Total uncertainty (higher for OOD)
    from sklearn.metrics import roc_auc_score, roc_curve

    # Compute mean uncertainty per sample (average over spatial dimensions)
    id_unc_per_sample = id_uncertainties.reshape(len(id_uncertainties), -1).mean(axis=1)
    ood_unc_per_sample = ood_uncertainties.reshape(len(ood_uncertainties), -1).mean(axis=1)

    y_true = np.concatenate([
        np.zeros(len(id_unc_per_sample)),  # ID = 0
        np.ones(len(ood_unc_per_sample))   # OOD = 1
    ])

    y_score = np.concatenate([
        id_unc_per_sample,
        ood_unc_per_sample
    ])

    auroc = roc_auc_score(y_true, y_score)
    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    print(f"\nOOD Detection AUROC: {auroc:.4f}")

    # Save results
    results = {
        'id_mse': float(id_mse),
        'ood_mse': float(ood_mse),
        'id_mean_uncertainty': float(id_mean_unc),
        'ood_mean_uncertainty': float(ood_mean_unc),
        'uncertainty_ratio': float(ood_mean_unc / (id_mean_unc + 1e-8)),
        'auroc': float(auroc)
    }

    with open(exp_dir / 'experiment4_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Generate plots
    print("\nGenerating plots...")

    # Plot 1: Uncertainty distributions (ID vs OOD)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    axes[0].hist(id_uncertainties.flatten(), bins=50, alpha=0.6, label='ID (Re=1k-3k)', density=True, color='blue')
    axes[0].hist(ood_uncertainties.flatten(), bins=50, alpha=0.6, label='OOD (Re=5k-10k)', density=True, color='red')
    axes[0].set_xlabel('Total Uncertainty', fontsize=12)
    axes[0].set_ylabel('Density', fontsize=12)
    axes[0].set_title('Uncertainty Distribution: ID vs OOD', fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].grid(alpha=0.3)

    # Box plot
    axes[1].boxplot([id_uncertainties.flatten(), ood_uncertainties.flatten()],
                   labels=['ID (Re=1k-3k)', 'OOD (Re=5k-10k)'])
    axes[1].set_ylabel('Total Uncertainty', fontsize=12)
    axes[1].set_title('Uncertainty Comparison', fontsize=14)
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(exp_dir / 'experiment4_uncertainty_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 2: ROC Curve
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    ax.plot(fpr, tpr, linewidth=2, label=f'AUROC = {auroc:.4f}')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUROC = 0.5)')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve: OOD Detection via Uncertainty', fontsize=14)
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(exp_dir / 'experiment4_roc_curve.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 3: Uncertainty vs Error scatter (ID and OOD)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ID
    axes[0].scatter(id_uncertainties.flatten()[:1000], id_errors.flatten()[:1000], alpha=0.3, s=10)
    axes[0].set_xlabel('Total Uncertainty', fontsize=12)
    axes[0].set_ylabel('Absolute Error', fontsize=12)
    axes[0].set_title('ID: Uncertainty vs Error', fontsize=14)
    axes[0].grid(alpha=0.3)

    # OOD
    axes[1].scatter(ood_uncertainties.flatten()[:1000], ood_errors.flatten()[:1000], alpha=0.3, s=10, color='red')
    axes[1].set_xlabel('Total Uncertainty', fontsize=12)
    axes[1].set_ylabel('Absolute Error', fontsize=12)
    axes[1].set_title('OOD: Uncertainty vs Error', fontsize=14)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(exp_dir / 'experiment4_uncertainty_vs_error.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n{'='*80}")
    print("EXPERIMENT 4 COMPLETE!")
    print(f"Results saved to: {exp_dir}")
    print(f"AUROC: {auroc:.4f}")
    print(f"{'='*80}\n")

    return {
        'results': results,
        'auroc': auroc,
        'output_dir': str(exp_dir)
    }


def create_default_config(pde_type: str) -> dict:
    """Create default configuration for a PDE type."""
    config = {
        # FNO Model
        'fno_modes': 12,
        'fno_width': 64,
        'fno_depth': 4,

        # EBM Architecture Selection
        'ebm_use_cnn': False,  # Use CNN for 2D spatial data (auto-enabled for 2D PDEs)
        'ebm_use_kan': False,  # Use KAN instead of MLP (not recommended)

        # EBM MLP-based settings (for 1D or when ebm_use_cnn=False)
        'ebm_hidden_dim': 256,  # Increased from 64
        'ebm_layers': 4,  # Increased from 3

        # EBM CNN-specific settings (when ebm_use_cnn=True for 2D PDEs)
        'ebm_base_channels': 64,  # Base channels for CNN
        'ebm_num_blocks': 4,  # Number of convolutional blocks
        'ebm_use_spatial_attn': True,  # Enable spatial attention
        'ebm_use_channel_attn': True,  # Enable channel attention (SE)
        'ebm_mlp_hidden': 256,  # MLP head hidden dimension

        # Training
        'batch_size': 32,
        'fno_epochs': 20,
        'ebm_epochs': 100,  # Increased from 20 - EBM needs longer training
        'fno_lr': 1e-3,
        'ebm_lr': 1e-4,
        'patience': 50,  # Increased for longer training

        # Data - Memory-safe defaults
        'train_ratio': 0.8,
        'val_ratio': 0.1,
        'max_samples': 100,
        'time_step_spacing': 2,  # CRITICAL FIX: Reduced from 10 to capture early dynamics
        'max_pairs_per_sample': 20,  # Increased from 10
        'seed': 42,

        # Tracking
        'enable_tracking': True,
        'tracking_backend': 'custom',  # 'custom' or 'tensorboard'

        # Checkpointing
        'save_checkpoints': False,  # Set to True to save model checkpoints

        # Device
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    }

    # PDE-specific adjustments
    if pde_type in ['burgers', 'advection']:
        # 1D PDEs: Use MLP-based EBM
        config['fno_modes'] = 16
        config['batch_size'] = 32
        config['ebm_use_cnn'] = False
        config['ebm_hidden_dim'] = 256
        config['ebm_layers'] = 4
        config['ebm_epochs'] = 100
        config['time_step_spacing'] = 2  # Critical: capture dynamics before dissipation
        config['max_pairs_per_sample'] = 20
        config['max_samples'] = 200

    elif pde_type in ['navier_stokes', 'diffusion_reaction','darcy']:
        # 2D PDEs: Use CNN-based EBM for spatial data
        config['fno_modes'] = 20
        config['fno_width'] = 96
        config['batch_size'] = 8  # Smaller batch for larger models
        config['ebm_use_cnn'] = True  # Enable CNN architecture
        config['ebm_base_channels'] = 64
        config['ebm_num_blocks'] = 4
        config['ebm_use_spatial_attn'] = True
        config['ebm_use_channel_attn'] = True
        config['ebm_mlp_hidden'] = 512  # Larger MLP head for 2D
        config['ebm_epochs'] = 150  # Longer training for complex 2D patterns
        config['max_samples'] = 100  # Prevent OOM on 2D data

    return config


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive UQ Experiment Runner - 17 Methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run FNO-EBM (original)
  python main.py --mode fno_ebm --data_path data/burgers.hdf5 --pde_type burgers

  # Run comprehensive comparison (all 17 methods)
  python main.py --mode comprehensive --data_path data/burgers.hdf5 --pde_type burgers

  # Run specific conformal method
  python main.py --mode conformal --method split --data_path data/burgers.hdf5 --pde_type burgers

  # Run specific evidential method
  python main.py --mode evidential --method der_nig --data_path data/burgers.hdf5 --pde_type burgers

  # Run specific baseline method
  python main.py --mode baseline --method mc_dropout --data_path data/burgers.hdf5 --pde_type burgers
        """
    )

    # Mode selection
    parser.add_argument('--mode', type=str, required=True,
                        choices=['fno_ebm', 'comprehensive', 'conformal', 'evidential', 'baseline', 'experiment'],
                        help='Experiment mode: fno_ebm (original), comprehensive (all 17 methods), '
                             'conformal (single CP method), evidential (single EDL method), '
                             'baseline (single cross-family baseline), experiment (run specific experiment)')

    # Required arguments
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to data file (.hdf5, .h5, or .pt)')
    parser.add_argument('--pde_type', type=str, required=True,
                        choices=['burgers', 'advection', 'diffusion_reaction', 'navier_stokes', 'darcy'],
                        help='PDE type')

    # Method-specific arguments
    parser.add_argument('--method', type=str, default=None,
                        help='Specific method name (for conformal or evidential mode)')

    # Experiment-specific arguments
    parser.add_argument('--exp_id', type=int, default=None,
                        help='Experiment ID (for experiment mode): '
                             '3=Epistemic vs Aleatoric, 4=OOD Detection, '
                             '5=Regularization Comparison, 8=Ablation Studies')
    parser.add_argument('--ood_data_path', type=str, default=None,
                        help='Path to OOD test data (for experiment 4)')

    # Regularization scheme selection (for evidential methods)
    parser.add_argument('--regularization', type=str, default=None,
                        choices=['standard', 'improved', 'uncertainty_aware', 'annealed',
                                'l2_evidence', 'adaptive', 'kl_divergence'],
                        help='Regularization scheme for evidential methods. '
                             'Options: standard (DER), improved (log barrier), '
                             'uncertainty_aware (inverse unc), annealed (time-decay), '
                             'l2_evidence (L2 penalty), adaptive (exp error), '
                             'kl_divergence (KL prior). Default: depends on method')

    # Optional arguments
    parser.add_argument('--nu_values', nargs='+', type=float, default=None,
                        help='Nu values to train on (for 1D PDEs only)')
    parser.add_argument('--output_dir', type=str, default='experiments',
                        help='Output directory')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to custom config JSON')

    # Model hyperparameters
    parser.add_argument('--fno_modes', type=int, default=None)
    parser.add_argument('--fno_width', type=int, default=None)
    parser.add_argument('--fno_depth', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs for conformal/evidential/baseline methods')
    parser.add_argument('--fno_epochs', type=int, default=None)
    parser.add_argument('--ebm_epochs', type=int, default=None)

    # Tracking backend
    parser.add_argument('--tracking_backend', type=str, default=None,
                        choices=['custom', 'tensorboard'],
                        help='Tracking backend (custom or tensorboard)')

    # Checkpointing
    parser.add_argument('--save_checkpoints', action='store_true',
                        help='Save model checkpoints during training')

    args = parser.parse_args()

    # Create config
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
    else:
        config_dict = create_default_config(args.pde_type)

    # Override with command line arguments
    if args.fno_modes is not None:
        config_dict['fno_modes'] = args.fno_modes
    if args.fno_width is not None:
        config_dict['fno_width'] = args.fno_width
    if args.fno_depth is not None:
        config_dict['fno_depth'] = args.fno_depth
    if args.batch_size is not None:
        config_dict['batch_size'] = args.batch_size
    if args.epochs is not None:
        config_dict['epochs'] = args.epochs
    if args.fno_epochs is not None:
        config_dict['fno_epochs'] = args.fno_epochs
    if args.ebm_epochs is not None:
        config_dict['ebm_epochs'] = args.ebm_epochs
    if args.tracking_backend is not None:
        config_dict['tracking_backend'] = args.tracking_backend
    if args.save_checkpoints:
        config_dict['save_checkpoints'] = True

    # Route to appropriate training function based on mode
    if args.mode == 'fno_ebm':
        # Original FNO-EBM training
        train_fno_ebm(
            config_dict=config_dict,
            data_path=args.data_path,
            pde_type=args.pde_type,
            nu_values=args.nu_values,
            output_dir=args.output_dir
        )

    elif args.mode == 'comprehensive':
        # Run all 17 methods
        run_comprehensive_comparison(
            data_path=args.data_path,
            pde_type=args.pde_type,
            output_dir=args.output_dir
        )

    elif args.mode == 'conformal':
        # Run single conformal method
        if not args.method:
            raise ValueError("--method required for conformal mode. "
                           "Options: split, full, cross, cqr, adaptive, mondrian")

        dataset = load_pde_data(args.data_path, args.pde_type, max_samples=500)
        output_dir = Path(args.output_dir) / f"conformal_{args.method}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output_dir.mkdir(parents=True, exist_ok=True)

        train_conformal_method(args.method, dataset, config_dict, output_dir)

    elif args.mode == 'evidential':
        # Run single evidential method
        if not args.method:
            raise ValueError("--method required for evidential mode. "
                           "Options: der_nig, improved_der, prior_networks, "
                           "posterior_networks, natural_posterior, dirichlet_evidential")

        reg_suffix = f"_{args.regularization}" if args.regularization else ""
        output_dir = Path(args.output_dir) / f"evidential_{args.method}{reg_suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output_dir.mkdir(parents=True, exist_ok=True)

        train_evidential_method(
            method_name=args.method,
            data_path=args.data_path,
            pde_type=args.pde_type,
            config_dict=config_dict,
            output_dir=output_dir,
            reg_name=args.regularization,
            max_samples=500
        )

    elif args.mode == 'baseline':
        # Run single baseline method
        if not args.method:
            raise ValueError("--method required for baseline mode. "
                           "Options: mc_dropout, ensemble, bayesian, standard_fno, mlp_ebm")

        dataset = load_pde_data(args.data_path, args.pde_type, max_samples=500)
        output_dir = Path(args.output_dir) / f"baseline_{args.method}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output_dir.mkdir(parents=True, exist_ok=True)

        train_baseline_method(args.method, dataset, config_dict, output_dir)

    elif args.mode == 'experiment':
        # Run specific experiment
        if not args.exp_id:
            raise ValueError("--exp_id required for experiment mode. "
                           "Options: 3 (Epistemic vs Aleatoric Decomposition), "
                           "4 (OOD Detection), "
                           "5 (Regularization Comparison), "
                           "8 (Ablation Studies)")

        if args.exp_id == 3:
            experiment_epistemic_aleatoric(
                data_path=args.data_path,
                pde_type=args.pde_type,
                config_dict=config_dict,
                output_dir=args.output_dir
            )
        elif args.exp_id == 4:
            # OOD Detection experiment
            if not args.ood_data_path:
                raise ValueError("--ood_data_path required for experiment 4")

            experiment_ood_detection(
                id_data_path=args.data_path,
                ood_data_path=args.ood_data_path,
                config_dict=config_dict,
                output_dir=args.output_dir
            )
        elif args.exp_id == 5:
            # Regularization Comparison experiment
            experiment_regularization_comparison(
                data_path=args.data_path,
                pde_type=args.pde_type,
                config_dict=config_dict,
                output_dir=args.output_dir
            )
        elif args.exp_id == 8:
            experiment_ablation(
                data_path=args.data_path,
                pde_type=args.pde_type,
                config_dict=config_dict,
                output_dir=args.output_dir
            )
        else:
            raise ValueError(f"Unknown experiment ID: {args.exp_id}. Available: 3, 4, 5, 8")


if __name__ == '__main__':
    main()
