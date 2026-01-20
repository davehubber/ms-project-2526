import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
import matplotlib.pyplot as plt
import os
import config

# --- Configuration ---
# Use paths from config.py
GROUND_TRUTH_FILE = config.GROUND_TRUTH_FILE
METAMODEL_FILE = config.METAMODEL_FILE
SCALER_X_FILE = config.SCALER_X_FILE
SCALER_Y_FILE = config.SCALER_Y_FILE

INPUT_SIZE = 676
OUTPUT_SIZE = 48
HIDDEN_LAYERS = [64, 256, 256, 512]

# Optimization Hyperparameters
LEARNING_RATE = 0.01   # Gradient Descent Step Size
ITERATIONS = 1000       # Faster convergence than SPSA

# --- 1. Model Definition (Must Match Training) ---
class ODMetamodel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims):
        super(ODMetamodel, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p=0.1))
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=0.1))
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

# --- 2. Differentiable Inverse Scaler ---
class DifferentiableScalerY(nn.Module):
    """
    Reimplements MinMaxScaler.inverse_transform using PyTorch tensors 
    to allow gradients to flow from the Loss back to the Model Output.
    """
    def __init__(self, sklearn_scaler, device):
        super().__init__()
        # Sklearn formula: X_scaled = X * scale_ + min_
        # Inverse: X = (X_scaled - min_) / scale_
        
        # We handle the tiny division epsilon for stability if needed, 
        # but usually scale_ is non-zero.
        self.scale_ = torch.tensor(sklearn_scaler.scale_, dtype=torch.float32).to(device)
        self.min_ = torch.tensor(sklearn_scaler.min_, dtype=torch.float32).to(device)

    def inverse_transform(self, y_scaled):
        return (y_scaled - self.min_) / self.scale_

# --- 3. Main Optimization Routine ---
def run_gradient_descent():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Differentiable Optimization on: {device}")

    # A. Load Resources
    model = ODMetamodel(INPUT_SIZE, OUTPUT_SIZE, HIDDEN_LAYERS).to(device)
    model.load_state_dict(torch.load(METAMODEL_FILE, map_location=device))
    model.eval() 
    
    # Freeze Model Weights (We are optimizing the Input, not the Network!)
    for param in model.parameters():
        param.requires_grad = False

    # Load Scalers
    scaler_x_sci = joblib.load(SCALER_X_FILE)
    scaler_y_sci = joblib.load(SCALER_Y_FILE)
    
    # Convert Y-Scaler to Differentiable Torch Module
    diff_scaler_y = DifferentiableScalerY(scaler_y_sci, device)

    # B. Load Ground Truth
    df = pd.read_csv(GROUND_TRUTH_FILE)
    df_mid = df[df['TIME_OF_DAY'] == '00:00:00'].sort_values('EQUIPMENTID')
    y_target_np = df_mid['TOTAL_VOLUME'].values.astype(np.float32)
    
    # Target to Tensor
    y_target = torch.tensor(y_target_np).to(device)

    # C. Initialize Decision Vector (OD Matrix)
    # We initialize in the Normalized Space [0, 1]
    # requires_grad=True is the key magic that enables "M-SPSA" behavior
    initial_guess = np.random.uniform(0, 0.05, INPUT_SIZE).astype(np.float32)
    od_matrix_param = torch.tensor(initial_guess, device=device, requires_grad=True)

    # D. Optimizer (Adam is generally robust for this)
    optimizer = optim.Adam([od_matrix_param], lr=LEARNING_RATE)

    print("Starting Gradient Descent...")
    history = []

    for i in range(ITERATIONS):
        optimizer.zero_grad()
        
        # 1. Forward Pass (Model predicts scaled flow)
        # Note: We unsqueeze to make it batch size 1: [1, 676]
        pred_scaled = model(od_matrix_param.unsqueeze(0))
        
        # 2. Differentiable Inverse Transform (Scaled -> Real Vehicles)
        pred_real = diff_scaler_y.inverse_transform(pred_scaled).squeeze(0)
        
        # 3. Calculate Loss (RMSE between Real Prediction and Real Ground Truth)
        loss = torch.sqrt(torch.mean((pred_real - y_target) ** 2))
        
        # 4. Backward Pass (Calculate Exact Gradient)
        loss.backward()

        zones = 26
        diagonal_indices = [i * zones + i for i in range(zones)]

        # 2. Zero-out the gradient for these indices
        # This ensures the optimizer effectively "ignores" them
        od_matrix_param.grad.data[diagonal_indices] = 0.0
        
        # 5. Update Step
        optimizer.step()
        
        # 6. Projection / Constraints (Keep OD between 0 and 1)
        with torch.no_grad():
            od_matrix_param.clamp_(0, 1)
            
        # Logging
        if i % 10 == 0:
            current_loss = loss.item()
            history.append(current_loss)
            if i % 50 == 0:
                print(f"Iter {i:03d} | RMSE: {current_loss:.4f}")

    # --- 4. Final Processing & Saving ---
    final_rmse = history[-1]
    print(f"\nFinal RMSE: {final_rmse:.4f}")
    
    # Extract Final Normalized OD
    final_od_norm = od_matrix_param.detach().cpu().numpy().reshape(1, -1)
    
    # Inverse Transform X (Offline is fine here) to get Real OD Demand
    final_od_real = scaler_x_sci.inverse_transform(final_od_norm)[0]
    
    # Save
    results_file = config.RESULTS_DIR / "estimated_od_matrix_gd.csv"
    pd.DataFrame(final_od_real).to_csv(results_file, index=False, header=["Count"])
    print(f"Saved '{results_file}'")

    # --- Visualizations ---
    # 1. Convergence
    plt.figure(figsize=(10, 5))
    plt.plot(range(0, len(history)*10, 10), history, linewidth=2, color='green')
    plt.title("Gradient-Based Optimization Convergence (M-SPSA Limit)")
    plt.xlabel("Iteration")
    plt.ylabel("RMSE Loss")
    plt.grid(True, alpha=0.3)
    conv_file = config.RESULTS_DIR / "mspsa_convergence.png"
    plt.savefig(conv_file)
    
    # 2. Real vs Pred
    final_pred_real_np = pred_real.detach().cpu().numpy()
    plt.figure(figsize=(8, 8))
    plt.scatter(y_target_np, final_pred_real_np, c='green', alpha=0.6, label='Detectors')
    
    limit = max(y_target_np.max(), final_pred_real_np.max()) * 1.1
    plt.plot([0, limit], [0, limit], 'r--', label='Perfect Fit')
    plt.xlim(0, limit)
    plt.ylim(0, limit)
    plt.xlabel("Real Ground Truth")
    plt.ylabel("Optimized Estimate")
    plt.title("Estimation Quality: Gradient Descent")
    plt.legend()
    plt.grid(True)
    scatter_file = config.RESULTS_DIR / "mspsa_estimation_scatter.png"
    plt.savefig(scatter_file)
    print("Visualizations saved.")

if __name__ == "__main__":
    run_gradient_descent()