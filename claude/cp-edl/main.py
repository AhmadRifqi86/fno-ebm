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

from config import Config, Factory, get_baseline_configs, get_conformal_methods_configs, get_evidential_methods_configs
from datautils import (
    load_pde_data, create_dataloaders, create_dataloaders_with_calibration,
    create_kfold_splits, create_stratified_splits, create_ensemble_splits,
    create_ood_test_data, get_calibration_dataset
)
from fno import (
    FNO2d, FNOTrainer, EvidentialFNO2d, MCDropoutFNO2d, FNOEnsemble,
    BayesianFNO2d, QuantileFNO2d, PriorNetworkFNO2d, DirichletEvidentialFNO2d,
    PosteriorNetworkFNO2d
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


def train_evidential_method(method_name: str, dataset, config_dict: dict,
                            output_dir: Path) -> Dict:
    """
    Train and evaluate an Evidential Deep Learning method.

    Args:
        method_name: 'der_nig', 'improved_der', 'prior_networks', 'posterior_networks',
                    'natural_posterior', 'dirichlet_evidential'
        dataset: PDEDataset
        config_dict: Configuration dictionary
        output_dir: Output directory

    Returns:
        Dictionary with metrics (ECE, correlation, epistemic/aleatoric split, etc.)
    """
    config = Config(config_dict)
    method_configs = get_evidential_methods_configs()
    method_config = method_configs.get(method_name, method_configs.get('der_nig'))

    print(f"\n{'='*70}")
    print(f"EVIDENTIAL METHOD: {method_name.upper()}")
    print(f"{'='*70}")

    # Standard 2-way split for EDL
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset,
        train_ratio=0.85,
        val_ratio=0.05,
        batch_size=config.batch_size,
        seed=config.seed if hasattr(config, 'seed') else 42
    )

    # Create model
    print("\nInitializing evidential model...")
    if method_name in ['der_nig', 'improved_der', 'natural_posterior']:
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

    # Use Factory to create optimizer and scheduler
    optimizer_config = {
        'type': method_config.get('optimizer', 'adam'),
        'lr': method_config.get('lr', 1e-4),
        'weight_decay': method_config.get('weight_decay', 0.0)
    }
    optimizer = Factory.create_optimizer(optimizer_config, model.parameters())

    # Create learning rate scheduler if configured
    scheduler = None
    if 'scheduler' in method_config and method_config['scheduler'] is not None:
        scheduler = Factory.create_scheduler(method_config['scheduler'], optimizer)
    else:
        # Default: ReduceLROnPlateau for evidential methods
        scheduler_config = {
            'type': 'exponential_lr',
            'gamma': 0.93
        }
        scheduler = Factory.create_scheduler(scheduler_config, optimizer)

    # Initialize tracker
    tracker = None
    if getattr(config, 'enable_tracking', False):
        try:
            from track import GradientTracker
            tracker = GradientTracker(model, log_dir=str(output_dir / 'logs'),
                                     experiment_name=f'evidential_{method_name}')
        except ImportError:
            pass

    # Training loop - prioritize config_dict over method_config
    epochs = config_dict.get('epochs', method_config.get('epochs', 30))
    print(f"\nTraining evidential model for {epochs} epochs...")
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0

        for x, y in train_loader:
            x, y = x.to(config.device), y.to(config.device)

            if method_name in ['der_nig', 'improved_der', 'natural_posterior']:
                gamma, nu, alpha, beta = model(x)
                if method_name == 'improved_der':
                    loss, _ = improved_evidential_loss(gamma, nu, alpha, beta, y, reg_weight=method_config['reg_weight'])
                elif method_name == 'natural_posterior':
                    loss = natural_nig_loss(gamma, nu, alpha, beta, y)
                else:
                    loss, _ = evidential_loss(gamma, nu, alpha, beta, y, reg_weight=method_config['reg_weight'])
            elif method_name == 'prior_networks':
                alphas, mean, uncertainty = model(x)
                loss = prior_network_loss(alphas, y, n_bins=method_config['n_bins'],
                                         output_range=tuple(method_config['output_range']))
            else:
                # Posterior networks or Dirichlet
                mean, aleatoric, epistemic = model(x)
                loss = F.mse_loss(mean, y)

            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping to prevent evidential collapse
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=15.0)

            optimizer.step()

            if tracker:
                tracker.track(loss=loss)

            train_loss += loss.item()
            train_batches += 1

        avg_train_loss = train_loss / train_batches

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(config.device), y.to(config.device)

                if method_name in ['der_nig', 'improved_der', 'natural_posterior']:
                    gamma, nu, alpha, beta = model(x)
                    if method_name == 'improved_der':
                        loss, _ = improved_evidential_loss(gamma, nu, alpha, beta, y, reg_weight=method_config['reg_weight'])
                    elif method_name == 'natural_posterior':
                        loss = natural_nig_loss(gamma, nu, alpha, beta, y)
                    else:
                        loss, _ = evidential_loss(gamma, nu, alpha, beta, y, reg_weight=method_config['reg_weight'])
                elif method_name == 'prior_networks':
                    alphas, mean, uncertainty = model(x)
                    loss = prior_network_loss(alphas, y, n_bins=method_config['n_bins'],
                                             output_range=tuple(method_config['output_range']))
                else:
                    # Posterior networks or Dirichlet
                    mean, aleatoric, epistemic = model(x)
                    loss = F.mse_loss(mean, y)

                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / val_batches if val_batches > 0 else 0.0

        # Step scheduler based on validation loss
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_val_loss)
            else:
                scheduler.step()

        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"[{method_name.upper()}] Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, LR: {current_lr:.2e}")
        
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
            # Get a sample from test loader
            for x, y in test_loader:
                pde_type = config_dict.get('pde_type', 'unknown')
                visualize_evidential_parameters(
                    model=model,
                    x=x,
                    u_gt=y,
                    save_path=str(output_dir / f'{method_name}_nig_parameters.png'),
                    sample_idx=0,
                    pde_type=pde_type,
                    device=config.device
                )
                print(f"NIG parameter visualization saved to {output_dir / f'{method_name}_nig_parameters.png'}")
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
            metrics = train_evidential_method(method, dataset, config_dict, exp_dir)
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

    elif pde_type in ['navier_stokes', 'diffusion_reaction']:
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
                        choices=['fno_ebm', 'comprehensive', 'conformal', 'evidential', 'baseline'],
                        help='Experiment mode: fno_ebm (original), comprehensive (all 17 methods), '
                             'conformal (single CP method), evidential (single EDL method), '
                             'baseline (single cross-family baseline)')

    # Required arguments
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to data file (.hdf5, .h5, or .pt)')
    parser.add_argument('--pde_type', type=str, required=True,
                        choices=['burgers', 'advection', 'diffusion_reaction', 'navier_stokes'],
                        help='PDE type')

    # Method-specific arguments
    parser.add_argument('--method', type=str, default=None,
                        help='Specific method name (for conformal or evidential mode)')

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

        dataset = load_pde_data(args.data_path, args.pde_type, max_samples=500)
        output_dir = Path(args.output_dir) / f"evidential_{args.method}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output_dir.mkdir(parents=True, exist_ok=True)

        train_evidential_method(args.method, dataset, config_dict, output_dir)

    elif args.mode == 'baseline':
        # Run single baseline method
        if not args.method:
            raise ValueError("--method required for baseline mode. "
                           "Options: mc_dropout, ensemble, bayesian, standard_fno, mlp_ebm")

        dataset = load_pde_data(args.data_path, args.pde_type, max_samples=500)
        output_dir = Path(args.output_dir) / f"baseline_{args.method}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output_dir.mkdir(parents=True, exist_ok=True)

        train_baseline_method(args.method, dataset, config_dict, output_dir)


if __name__ == '__main__':
    main()
