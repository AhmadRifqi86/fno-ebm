#!/usr/bin/env python3
"""Check if FNO training is actually learning or just predicting constants."""

import torch
import numpy as np
from config import Config
import yaml
import os

# Load config
with open('config.yaml', 'r') as f:
    config_dict = yaml.safe_load(f)
    config = Config(config_dict)

# Load best FNO checkpoint
checkpoint_path = os.path.join(config.checkpoint_dir, 'best_model_fno.pt')
if not os.path.exists(checkpoint_path):
    print(f"❌ Checkpoint not found: {checkpoint_path}")
    exit(1)

checkpoint = torch.load(checkpoint_path, map_location=config.device, weights_only=False)

# Load FNO model
from fno import FNO2d
fno_model = FNO2d(
    modes1=config.fno_modes,
    modes2=config.fno_modes,
    width=config.fno_width,
    num_layers=4
).to(config.device)

# Extract FNO model state dict if checkpoint contains multiple items
if 'fno_model' in checkpoint:
    fno_model.load_state_dict(checkpoint['fno_model'])
else:
    fno_model.load_state_dict(checkpoint)
fno_model.eval()

# Load validation data
from dataset.darcy_flow import DarcyFlowDataset

val_dataset = DarcyFlowDataset(
    data_path=config.data_path,
    mode='val',
    normalize=config.normalize
)

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=config.batch_size,
    shuffle=False
)

# Get predictions on validation set
print("Evaluating on validation set...")
all_y_true = []
all_y_pred = []

with torch.no_grad():
    for x, y in val_loader:
        x = x.to(config.device)
        y = y.to(config.device)

        y_pred = fno_model(x)

        all_y_true.append(y.cpu().numpy())
        all_y_pred.append(y_pred.cpu().numpy())

y_true = np.concatenate(all_y_true, axis=0)
y_pred = np.concatenate(all_y_pred, axis=0)

# Compute statistics
print("\n" + "="*60)
print("TRAINING QUALITY DIAGNOSTIC")
print("="*60)

print(f"\n📊 Ground Truth Statistics:")
print(f"   Mean:  {y_true.mean():+.6f}")
print(f"   Std:   {y_true.std():.6f}")
print(f"   Min:   {y_true.min():+.6f}")
print(f"   Max:   {y_true.max():+.6f}")

print(f"\n🔮 Prediction Statistics:")
print(f"   Mean:  {y_pred.mean():+.6f}")
print(f"   Std:   {y_pred.std():.6f}")
print(f"   Min:   {y_pred.min():+.6f}")
print(f"   Max:   {y_pred.max():+.6f}")

# Compute metrics
mse = np.mean((y_true - y_pred)**2)
variance = y_true.var()
relative_mse = mse / variance if variance > 0 else float('inf')
r2_score = 1 - relative_mse

# Correlation
y_true_flat = y_true.flatten()
y_pred_flat = y_pred.flatten()
correlation = np.corrcoef(y_true_flat, y_pred_flat)[0, 1]

print(f"\n📈 Performance Metrics:")
print(f"   MSE:               {mse:.8f}")
print(f"   Truth Variance:    {variance:.8f}")
print(f"   Relative MSE:      {relative_mse:.4f}")
print(f"   R² Score:          {r2_score:.4f}")
print(f"   Correlation:       {correlation:.4f}")

# Critical checks
print(f"\n🔍 Critical Checks:")

checks_passed = 0
checks_total = 3

# Check 1: Prediction variance
pred_std_ratio = y_pred.std() / y_true.std() if y_true.std() > 0 else 0
if pred_std_ratio > 0.5:
    print(f"   ✅ Prediction variance: {pred_std_ratio:.2%} of truth (GOOD)")
    checks_passed += 1
else:
    print(f"   ❌ Prediction variance: {pred_std_ratio:.2%} of truth (TOO LOW - model predicting constants!)")

# Check 2: Correlation
if correlation > 0.95:
    print(f"   ✅ Correlation: {correlation:.4f} (EXCELLENT)")
    checks_passed += 1
elif correlation > 0.85:
    print(f"   ⚠️  Correlation: {correlation:.4f} (GOOD but could be better)")
    checks_passed += 1
else:
    print(f"   ❌ Correlation: {correlation:.4f} (POOR - model not learning properly)")

# Check 3: R² score
if r2_score > 0.9:
    print(f"   ✅ R² Score: {r2_score:.4f} (EXCELLENT)")
    checks_passed += 1
elif r2_score > 0.7:
    print(f"   ⚠️  R² Score: {r2_score:.4f} (ACCEPTABLE)")
    checks_passed += 1
else:
    print(f"   ❌ R² Score: {r2_score:.4f} (POOR - worse than simple mean baseline)")

print(f"\n📝 Overall Assessment: {checks_passed}/{checks_total} checks passed")

if checks_passed == 3:
    print("   🎉 EXCELLENT TRAINING! Model is learning well.")
elif checks_passed == 2:
    print("   👍 GOOD TRAINING. Model is learning but has room for improvement.")
elif checks_passed == 1:
    print("   ⚠️  MEDIOCRE TRAINING. Model is partially learning.")
else:
    print("   ❌ POOR TRAINING. Model is likely underfitting (predicting constants).")

print("\n" + "="*60)

# Visualization hint
print("\n💡 To visualize predictions, run:")
print("   python -c \"from check_training import *; import matplotlib.pyplot as plt; plt.figure(); plt.scatter(y_true.flatten()[:1000], y_pred.flatten()[:1000], alpha=0.1); plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--'); plt.xlabel('True'); plt.ylabel('Predicted'); plt.savefig('predictions.png'); print('Saved to predictions.png')\"")