import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import matplotlib.pyplot as plt
import config

GROUND_TRUTH_FILE = config.GROUND_TRUTH_FILE
METAMODEL_FILE = config.METAMODEL_FILE
SCALER_X_FILE = config.SCALER_X_FILE
SCALER_Y_FILE = config.SCALER_Y_FILE

INPUT_SIZE = 676
OUTPUT_SIZE = 48
HIDDEN_LAYERS = [64, 256, 256, 512]

MAX_ITERATIONS = 1000
A_PARAM = 0.001
C_PARAM = 0.1
DELTA_PARAM = 0.1

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

def load_target_volumes(filename):
    df = pd.read_csv(filename)
    
    df_midnight = df[df['TIME_OF_DAY'] == '00:00:00'].copy()
    
    df_midnight = df_midnight.sort_values(by='EQUIPMENTID')
    
    target_volumes = df_midnight['TOTAL_VOLUME'].values.astype(np.float32)
    
    print(f"Loaded Target Data: {len(target_volumes)} detectors.")
    print(f"Target Volume Stats: Min={target_volumes.min():.2f}, Max={target_volumes.max():.2f}")
    
    return target_volumes

def rmse_loss(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running SPSA on: {device}")

    try:
        model = ODMetamodel(INPUT_SIZE, OUTPUT_SIZE, HIDDEN_LAYERS).to(device)
        model.load_state_dict(torch.load(METAMODEL_FILE, map_location=device))
        model.eval()
        
        scaler_x = joblib.load(SCALER_X_FILE)
        scaler_y = joblib.load(SCALER_Y_FILE)
        
        y_real = load_target_volumes(GROUND_TRUTH_FILE)
        
    except FileNotFoundError as e:
        print(f"Error: Missing file. {e}")
        print("Ensure 'od_metamodel.pth', 'scaler_x.pkl', and 'scaler_y.pkl' are present.")
        return

    theta = np.random.uniform(0, 0.01, size=(INPUT_SIZE,)).astype(np.float32)
    
    history = []
    print("\nStarting Optimization Loop...")

    for k in range(MAX_ITERATIONS):
        ak = A_PARAM / ((k + 1) ** DELTA_PARAM)
        ck = C_PARAM / ((k + 1) ** DELTA_PARAM)
        
        delta_k = np.sign(np.random.randn(INPUT_SIZE)).astype(np.float32)
        
        theta_plus = theta + ck * delta_k
        theta_minus = theta - ck * delta_k
        
        theta_plus = np.clip(theta_plus, 0, 1)
        theta_minus = np.clip(theta_minus, 0, 1)
        
        inputs_batch = np.vstack([theta_plus, theta_minus])
        inputs_tensor = torch.tensor(inputs_batch, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            outputs_scaled = model(inputs_tensor).cpu().numpy()
            
        outputs_real = scaler_y.inverse_transform(outputs_scaled)
        
        y_pred_plus = outputs_real[0]
        y_pred_minus = outputs_real[1]
        
        loss_plus = rmse_loss(y_real, y_pred_plus)
        loss_minus = rmse_loss(y_real, y_pred_minus)
        
        loss_diff = loss_plus - loss_minus
        grad_est = (loss_diff / (2 * ck)) * delta_k
        
        theta = theta - ak * grad_est
        
        theta = np.clip(theta, 0, 1)
        
        if k % 10 == 0 or k == MAX_ITERATIONS - 1:
            with torch.no_grad():
                curr_in = torch.tensor(theta.reshape(1, -1), dtype=torch.float32).to(device)
                curr_out_scaled = model(curr_in).cpu().numpy()
                curr_out_real = scaler_y.inverse_transform(curr_out_scaled)[0]
                curr_rmse = rmse_loss(y_real, curr_out_real)
                history.append(curr_rmse)
                print(f"Iter {k:03d} | RMSE: {curr_rmse:.4f} | Step Size (ak): {ak:.5f}")

    final_od_normalized = theta
    
    with torch.no_grad():
        final_in = torch.tensor(final_od_normalized.reshape(1, -1), dtype=torch.float32).to(device)
        final_pred_flow = scaler_y.inverse_transform(model(final_in).cpu().numpy())[0]
        
    print(f"\nFinal RMSE: {history[-1]:.4f}")
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(0, len(history)*10, 10), history, marker='o', markersize=3)
    plt.title("SPSA Optimization Convergence (RMSE)")
    plt.xlabel("Iteration")
    plt.ylabel("RMSE (Vehicle Volume)")
    plt.grid(True, alpha=0.3)
    conv_file = config.RESULTS_DIR / 'spsa_convergence.png'
    plt.savefig(conv_file)
    print(f"Saved '{conv_file}'")

    plt.figure(figsize=(7, 7))
    plt.scatter(y_real, final_pred_flow, alpha=0.6, c='blue', label='Detectors')
    
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