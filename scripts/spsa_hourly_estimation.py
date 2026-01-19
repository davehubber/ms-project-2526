import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import joblib
import os
from datetime import datetime

# --- 1. CONFIGURATION ---
INPUT_SIZE = 676        # 26x26 Zones
OUTPUT_SIZE = 48        # 48 Detectors
HIDDEN_LAYERS = [64, 256, 256, 512]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Simulation / Optimization Config
START_HOUR = 8          # e.g., 8 AM
END_HOUR = 10            # e.g., 10 AM (Estimates 08:00 to 1:00 inclusive)
SPSA_ITERATIONS = 1000   # Iterations per time step

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
MODELS_DIR = os.path.join(DATA_DIR, 'models')
SENSOR_DATA_DIR = os.path.join(DATA_DIR, 'sensor_data')

# Artifact Paths
MODEL_PATH = os.path.join(MODELS_DIR, "metamodel.pth")
SCALER_X_PATH = os.path.join(MODELS_DIR, "scaler_x.pkl")
SCALER_Y_PATH = os.path.join(MODELS_DIR, "scaler_y.pkl")
GROUND_TRUTH_FILE = os.path.join(SENSOR_DATA_DIR, "ground_truth.csv")

# --- 2. MODEL DEFINITION (Must match training) ---
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

# --- 3. SPSA OPTIMIZER (Refactored for Sequential Targets) ---
class SPSAOptimizer:
    def __init__(self, model, scaler_x, scaler_y, input_size, device):
        self.model = model
        self.scaler_x = scaler_x
        self.scaler_y = scaler_y
        self.input_size = input_size
        self.device = device
        
        # Hyperparameters
        self.a = 0.001
        self.c = 0.1
        self.delta = 0.1 
        
        # Search space scale vector S
        self.S = torch.tensor(self.scaler_x.data_range_, dtype=torch.float32).to(device)

    def get_loss(self, theta_scaled, target_tensor):
        """Helper to calculate RMSE against a specific target tensor"""
        self.model.eval()
        with torch.no_grad():
            pred_scaled = self.model(theta_scaled.unsqueeze(0))
            pred_real_np = self.scaler_y.inverse_transform(pred_scaled.cpu().numpy())
            pred_real = torch.tensor(pred_real_np, dtype=torch.float32).to(self.device)
            mse = nn.MSELoss()(pred_real, target_tensor)
        return torch.sqrt(mse)

    def run(self, target_data_numpy, iterations=300):
        """
        Runs SPSA for a specific target vector (one time step).
        target_data_numpy: 1D array of real sensor counts.
        """
        target_tensor = torch.tensor(target_data_numpy, dtype=torch.float32).to(self.device)
        
        # Initialization
        init_od_real = np.random.randint(0, 2, size=(1, self.input_size))
        
        # Scale initial guess
        init_od_scaled = self.scaler_x.transform(init_od_real)
        theta = torch.tensor(init_od_scaled.flatten(), dtype=torch.float32).to(self.device)
        
        final_loss = 0.0
        
        for k in range(iterations):
            # Decay Sequences
            ak = (self.a / ((k + 1) ** self.delta)) * self.S
            ck = (self.c / ((k + 1) ** self.delta)) * self.S
            
            # Perturbation
            delta_k = torch.randint(0, 2, (self.input_size,), device=self.device).float() * 2 - 1
            
            # Evaluation
            theta_plus = torch.clamp(theta + ck * delta_k, 0, 1)
            theta_minus = torch.clamp(theta - ck * delta_k, 0, 1)

            y_plus = self.get_loss(theta_plus, target_tensor)
            y_minus = self.get_loss(theta_minus, target_tensor)
            
            # Gradient Update
            ghat = (y_plus - y_minus) / (2.0 * ck * delta_k)
            theta = torch.clamp(theta - ak * ghat, 0, 1)
            
            if k == iterations - 1:
                final_loss = self.get_loss(theta, target_tensor).item()

        # Inverse transform result
        final_od_scaled = theta.cpu().numpy().reshape(1, -1)
        final_od_real = self.scaler_x.inverse_transform(final_od_scaled)
        final_od_real = np.maximum(final_od_real, 0) # Non-negative constraint
        
        return final_od_real.flatten(), final_loss

# --- 4. MAIN EXECUTION ---
if __name__ == "__main__":
    # A. Load Artifacts
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not found. Run training first.")
    
    print("Loading model and scalers...")
    scaler_x = joblib.load(SCALER_X_PATH)
    scaler_y = joblib.load(SCALER_Y_PATH)
    model = ODMetamodel(INPUT_SIZE, OUTPUT_SIZE, HIDDEN_LAYERS).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # B. Load and Filter Data
    print(f"Loading Ground Truth for interval {START_HOUR:02d}:00 - {END_HOUR:02d}:00...")
    df_gt = pd.read_csv(GROUND_TRUTH_FILE)
    
    # Identify time column (case-insensitive search)
    time_col = next((c for c in df_gt.columns if 'time' in c.lower()), None)
    if not time_col:
        raise ValueError("Cannot filter by hour: No 'time' column found in CSV.")
        
    # Convert to datetime objects for filtering
    df_gt['dt_obj'] = pd.to_datetime(df_gt[time_col], format='%H:%M:%S', errors='coerce')
    if df_gt['dt_obj'].isna().all():
         df_gt['dt_obj'] = pd.to_datetime(df_gt[time_col], errors='coerce')

    # Filter Rows
    mask = (df_gt['dt_obj'].dt.hour >= START_HOUR) & (df_gt['dt_obj'].dt.hour < END_HOUR)
    interval_data = df_gt[mask].copy().sort_values(by='dt_obj')
    
    if len(interval_data) == 0:
        print("No data found for the specified time interval.")
        exit()
        
    print(f"Found {len(interval_data)} time steps to process.")

    # C. Sequential Estimation Loop
    optimizer = SPSAOptimizer(model, scaler_x, scaler_y, INPUT_SIZE, DEVICE)
    
    results_od = []
    rmse_history = []
    time_steps = []
    
    # Identify Sensor Columns (exclude time/metadata)
    sensor_cols = [c for c in df_gt.columns if c not in [time_col, 'dt_obj']]
    # Ensure we strictly pick the first 48 numeric columns if headers are messy
    if len(sensor_cols) > OUTPUT_SIZE:
        sensor_cols = sensor_cols[:OUTPUT_SIZE]
    
    for idx, row in interval_data.iterrows():
        current_time = row[time_col]
        target_vals = row[sensor_cols].values.astype(float)
        
        # Run SPSA for this specific 5-min interval
        est_od, final_rmse = optimizer.run(target_vals, iterations=SPSA_ITERATIONS)
        
        print(f"Time: {current_time} | Final RMSE: {final_rmse:.2f}")
        
        # Store results
        res_dict = {f"OD_{i}": val for i, val in enumerate(est_od)}
        res_dict['time'] = current_time
        res_dict['rmse'] = final_rmse
        results_od.append(res_dict)
        
        rmse_history.append(final_rmse)
        time_steps.append(current_time)

    # D. Save & Visualize
    # 1. Save Full Time-Series OD Matrix
    df_results = pd.DataFrame(results_od)
    # Move 'time' and 'rmse' to front
    cols = ['time', 'rmse'] + [c for c in df_results.columns if c not in ['time', 'rmse']]
    df_results = df_results[cols]
    
    output_filename = os.path.join(RESULTS_DIR, f"estimated_od_{START_HOUR:02d}-{END_HOUR:02d}.csv")
    df_results.to_csv(output_filename, index=False)
    print(f"\nTime-series OD estimates saved to {output_filename}")

    # 2. Plot RMSE over Time
    plt.figure(figsize=(10, 5))
    plt.plot(time_steps, rmse_history, marker='o', linestyle='-', color='#1f77b4')
    plt.xticks(rotation=45)
    plt.xlabel('Time of Day')
    plt.ylabel('RMSE (Vehicle Counts)')
    plt.title(f'Estimation Accuracy over Time ({START_HOUR:02d}:00 - {END_HOUR:02d}:00)')
    plt.grid(True)
    plt.tight_layout()
    output_plot_path = os.path.join(RESULTS_DIR, 'hourly_rmse_tracking.png')
    plt.savefig(output_plot_path)
    print(f"Saved plot to {output_plot_path}")