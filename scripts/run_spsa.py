import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import matplotlib.pyplot as plt
import os
import config

# --- Configuration ---
# File Paths - Use config.py
GROUND_TRUTH_FILE = config.GROUND_TRUTH_FILE
METAMODEL_FILE = config.METAMODEL_FILE
SCALER_X_FILE = config.SCALER_X_FILE
SCALER_Y_FILE = config.SCALER_Y_FILE

# Dimensions (Must match your trained model)
INPUT_SIZE = 676   # OD Pairs
OUTPUT_SIZE = 48   # Detectors
HIDDEN_LAYERS = [64, 256, 256, 512]

# SPSA Hyperparameters (Based on Min et al. 2024)
MAX_ITERATIONS = 1000   # Sufficient for convergence within ~1 min
A_PARAM = 0.001        # Step size numerator
C_PARAM = 0.1          # Perturbation numerator
DELTA_PARAM = 0.1      # Decay rate (Paper: "delta is a constant... 0.1")

# --- Model Architecture (Must match training) ---
class ODMetamodel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims):
        super(ODMetamodel, self).__init__()
        layers = []
        # Input -> Hidden 1
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p=0.1))
        
        # Hidden -> Hidden
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=0.1))
            
        # Hidden -> Output
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

# --- Helper Functions ---

def load_target_volumes(filename):
    """
    Extracts ground truth volumes for 00:00:00, sorted alphanumerically.
    """
    df = pd.read_csv(filename)
    
    # Filter for the first interval of the day
    df_midnight = df[df['TIME_OF_DAY'] == '00:00:00'].copy()
    
    # Sort by EQUIPMENTID to match the alphanumeric order of model outputs
    df_midnight = df_midnight.sort_values(by='EQUIPMENTID')
    
    # Extract volume column
    target_volumes = df_midnight['TOTAL_VOLUME'].values.astype(np.float32)
    
    print(f"Loaded Target Data: {len(target_volumes)} detectors.")
    print(f"Target Volume Stats: Min={target_volumes.min():.2f}, Max={target_volumes.max():.2f}")
    
    return target_volumes

def rmse_loss(y_true, y_pred):
    """Root Mean Squared Error (Equation 2 in Paper uses RMSE)"""
    return np.sqrt(np.mean((y_true - y_pred)**2))

# --- Main SPSA Optimization ---

def main():
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running SPSA on: {device}")

    # 2. Load Resources
    try:
        # Load Model
        model = ODMetamodel(INPUT_SIZE, OUTPUT_SIZE, HIDDEN_LAYERS).to(device)
        model.load_state_dict(torch.load(METAMODEL_FILE, map_location=device))
        model.eval() # Set to evaluation mode (disable dropout)
        
        # Load Scalers
        scaler_x = joblib.load(SCALER_X_FILE)
        scaler_y = joblib.load(SCALER_Y_FILE)
        
        # Load Ground Truth
        y_real = load_target_volumes(GROUND_TRUTH_FILE)
        
    except FileNotFoundError as e:
        print(f"Error: Missing file. {e}")
        print("Ensure 'od_metamodel.pth', 'scaler_x.pkl', and 'scaler_y.pkl' are present.")
        return

    # 3. Initialize OD Matrix (The Decision Vector 'theta')
    # Paper: "random integers of 0 or 1".
    # Implementation: We optimize in Normalized Space [0, 1].
    # We initialize with small random values (0 to 0.01) to simulate a "conservative" empty network start.
    
    theta = np.random.uniform(0, 0.01, size=(INPUT_SIZE,)).astype(np.float32)
    
    history = []
    print("\nStarting Optimization Loop...")

    for k in range(MAX_ITERATIONS):
        # --- A. Update Hyperparameters (Decay) ---
        # ak = a / (k+1)^delta
        # ck = c / (k+1)^delta
        ak = A_PARAM / ((k + 1) ** DELTA_PARAM)
        ck = C_PARAM / ((k + 1) ** DELTA_PARAM)
        
        # --- B. Perturbation ---
        # Generate Bernoulli Vector (+1 or -1)
        delta_k = np.sign(np.random.randn(INPUT_SIZE)).astype(np.float32)
        
        # Create Two Candidates: Theta+ and Theta-
        theta_plus = theta + ck * delta_k
        theta_minus = theta - ck * delta_k
        
        # Constraint: Keep within normalized bounds [0, 1]
        theta_plus = np.clip(theta_plus, 0, 1)
        theta_minus = np.clip(theta_minus, 0, 1)
        
        # --- C. Evaluate Metamodel (Batch Mode) ---
        # Stack inputs to run both predictions in one forward pass
        inputs_batch = np.vstack([theta_plus, theta_minus])
        inputs_tensor = torch.tensor(inputs_batch, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            outputs_scaled = model(inputs_tensor).cpu().numpy()
            
        # Inverse Transform: Convert predictions back to Real Vehicle Counts
        outputs_real = scaler_y.inverse_transform(outputs_scaled)
        
        y_pred_plus = outputs_real[0]
        y_pred_minus = outputs_real[1]
        
        # --- D. Calculate Loss (RMSE) ---
        loss_plus = rmse_loss(y_real, y_pred_plus)
        loss_minus = rmse_loss(y_real, y_pred_minus)
        
        # --- E. Gradient Approximation & Update ---
        # Gradient Estimate = (Loss+ - Loss-) / (2*ck) * delta_k
        loss_diff = loss_plus - loss_minus
        grad_est = (loss_diff / (2 * ck)) * delta_k
        
        # Update Theta
        theta = theta - ak * grad_est
        
        # Apply Constraints (Non-negative flow)
        theta = np.clip(theta, 0, 1)
        
        # --- F. Monitoring ---
        if k % 10 == 0 or k == MAX_ITERATIONS - 1:
            # Check actual error of current theta
            with torch.no_grad():
                curr_in = torch.tensor(theta.reshape(1, -1), dtype=torch.float32).to(device)
                curr_out_scaled = model(curr_in).cpu().numpy()
                curr_out_real = scaler_y.inverse_transform(curr_out_scaled)[0]
                curr_rmse = rmse_loss(y_real, curr_out_real)
                history.append(curr_rmse)
                print(f"Iter {k:03d} | RMSE: {curr_rmse:.4f} | Step Size (ak): {ak:.5f}")

    # 4. Final Evaluation & Visualization
    final_od_normalized = theta
    
    # Get Final Prediction
    with torch.no_grad():
        final_in = torch.tensor(final_od_normalized.reshape(1, -1), dtype=torch.float32).to(device)
        final_pred_flow = scaler_y.inverse_transform(model(final_in).cpu().numpy())[0]
        
    print(f"\nFinal RMSE: {history[-1]:.4f}")
    
    # --- Visualization 1: Convergence Curve ---
    plt.figure(figsize=(10, 5))
    plt.plot(range(0, len(history)*10, 10), history, marker='o', markersize=3)
    plt.title("SPSA Optimization Convergence (RMSE)")
    plt.xlabel("Iteration")
    plt.ylabel("RMSE (Vehicle Volume)")
    plt.grid(True, alpha=0.3)
    conv_file = config.RESULTS_DIR / 'spsa_convergence.png'
    plt.savefig(conv_file)
    print(f"Saved '{conv_file}'")

    # --- Visualization 2: Estimated vs Real (Scatter) ---
    plt.figure(figsize=(7, 7))
    plt.scatter(y_real, final_pred_flow, alpha=0.6, c='blue', label='Detectors')
    
    # Perfect Fit Line
    limit_max = max(y_real.max(), final_pred_flow.max()) * 1.1
    plt.plot([0, limit_max], [0, limit_max], 'r--', label='Perfect Fit')
    
    plt.xlim(0, limit_max)
    plt.ylim(0, limit_max)
    plt.xlabel("Real World Volume (Ground Truth)")
    plt.ylabel("Estimated Volume (Metamodel Prediction)")
    plt.title("Reality Gap Analysis: Real vs Estimated")
    plt.legend()
    plt.grid(True, alpha=0.3)
    scatter_file = config.RESULTS_DIR / 'spsa_accuracy.png'
    plt.savefig(scatter_file)
    print(f"Saved '{scatter_file}'")

    # Save Estimated OD
    # Note: We need to inverse transform the OD to get real vehicle counts if possible.
    # But we don't have scaler_x loaded in this snippet logic effectively for inverse if it wasn't saved.
    # Assuming scaler_x is available:
    try:
        final_od_real = scaler_x.inverse_transform(final_od_normalized.reshape(1, -1))[0]
        res_file = config.RESULTS_DIR / "estimated_od_matrix.csv"
        np.savetxt(res_file, final_od_real, delimiter=",")
        print(f"Saved '{res_file}'")
    except:
        print("Could not save Real OD (scaler_x issue), saving normalized OD.")
        res_file_norm = config.RESULTS_DIR / "estimated_od_matrix_normalized.csv"
        np.savetxt(res_file_norm, final_od_normalized, delimiter=",")


if __name__ == "__main__":
    main()