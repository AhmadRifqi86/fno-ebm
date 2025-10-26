import yaml
import torch
from torch.utils.data import DataLoader, TensorDataset
import os
import matplotlib.pyplot as plt

from config import Config
from fno import FNO2d
from ebm import EBMPotential
from trainer import Trainer, FNO_EBM
from customs import compute_pde_residual
from inference import inference_deterministic, inference_probabilistic
from datautils import dummy_dataloaders, create_dataloaders, visualize_inference_results

def main():
    """
    Main entry-point for the training and inference pipeline.
    """
    # 1. Load Configuration from YAML
    print("--- Loading Configuration ---")
    with open('config.yaml', 'r') as f:
        config_dict = yaml.safe_load(f)
    config = Config(config_dict)
    print(f"Running on device: {config.device}")
    print(f"PDE Type: {config.pde_type}")
    print(f"Complexity: {config.complexity}")
    print(f"Noise Type: {config.noise_type}")

    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # 2. Instantiate Model
    print("--- Initializing Model ---")

    # Create FNO model
    fno_model = FNO2d(
        modes1=config.fno_modes,
        modes2=config.fno_modes,
        width=config.fno_width,
        num_layers=4
    )

    # Create EBM model (input_dim=4 because we concatenate u(1) + x(3))
    ebm_model = EBMPotential(
        input_dim=4,
        hidden_dims=[config.ebm_hidden_dim, config.ebm_hidden_dim * 2,
                     config.ebm_hidden_dim * 2, config.ebm_hidden_dim]
    )

    # Combine into FNO_EBM
    model = FNO_EBM(fno_model, ebm_model).to(config.device)

    # 3. Create DataLoaders
    # Use create_dataloaders for real PDE data (auto-generates if needed)
    # Use dummy_dataloaders for quick testing with random data
    train_loader, val_loader = create_dataloaders(config)

    # 5. Instantiate Trainer
    print("--- Initializing Trainer ---")
    trainer = Trainer(
        model=model,
        phy_loss=compute_pde_residual,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config
    )

    # 6. Run Training
    print("--- Starting Training ---")
    trainer.train_staged()
    print("--- Training Finished ---")

    # 7. Run Inference on the best model
    print("\n--- Loading Best Model for Inference ---")
    best_model_path = os.path.join(config.checkpoint_dir, 'best_model.pt')
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=config.device)
        model.u_fno.load_state_dict(checkpoint['fno_model'])
        model.V_ebm.load_state_dict(checkpoint['ebm_model'])
        
        # Get a batch of data for inference
        x_samples, y_true_samples = next(iter(val_loader))
        x_samples = x_samples.to(config.device)
        
        # Perform deterministic inference (FNO only)
        y_fno_pred = inference_deterministic(model, x_samples, device=config.device)
        
        # Perform probabilistic inference (EBM refinement)
        _, stats = inference_probabilistic(
            model, 
            x_samples, 
            num_samples=50, # Number of samples for stats
            num_mcmc_steps=config.mcmc_steps,
            step_size=config.mcmc_step_size,
            device=config.device
        )
        
        # Visualize the results
        visualize_inference_results(y_true_samples, y_fno_pred, stats, config)
    else:
        print("Could not find best_model.pt to run inference.")

if __name__ == '__main__':
    main()