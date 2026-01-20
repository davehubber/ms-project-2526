import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
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

LEARNING_RATE = 0.01
ITERATIONS = 1000

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

class DifferentiableScalerY(nn.Module):
    def __init__(self, sklearn_scaler, device):
        super().__init__()
        self.scale_ = torch.tensor(sklearn_scaler.scale_, dtype=torch.float32).to(device)
        self.min_ = torch.tensor(sklearn_scaler.min_, dtype=torch.float32).to(device)

    def inverse_transform(self, y_scaled):
        return (y_scaled - self.min_) / self.scale_

def run_gradient_descent():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Differentiable Optimization on: {device}")

    model = ODMetamodel(INPUT_SIZE, OUTPUT_SIZE, HIDDEN_LAYERS).to(device)
    model.load_state_dict(torch.load(METAMODEL_FILE, map_location=device))
    model.eval() 
    
    for param in model.parameters():
        param.requires_grad = False

    scaler_x_sci = joblib.load(SCALER_X_FILE)
    scaler_y_sci = joblib.load(SCALER_Y_FILE)
    
    diff_scaler_y = DifferentiableScalerY(scaler_y_sci, device)

    df = pd.read_csv(GROUND_TRUTH_FILE)
    df_mid = df[df['TIME_OF_DAY'] == '00:00:00'].sort_values('EQUIPMENTID')
    y_target_np = df_mid['TOTAL_VOLUME'].values.astype(np.float32)
    
    y_target = torch.tensor(y_target_np).to(device)

    initial_guess = np.random.uniform(0, 0.05, INPUT_SIZE).astype(np.float32)
    od_matrix_param = torch.tensor(initial_guess, device=device, requires_grad=True)

    optimizer = optim.Adam([od_matrix_param], lr=LEARNING_RATE)

    print("Starting Gradient Descent...")
    history = []

    for i in range(ITERATIONS):
        optimizer.zero_grad()
        
        pred_scaled = model(od_matrix_param.unsqueeze(0))
        
        pred_real = diff_scaler_y.inverse_transform(pred_scaled).squeeze(0)
        
        loss = torch.sqrt(torch.mean((pred_real - y_target) ** 2))
        
        loss.backward()

        zones = 26
        diagonal_indices = [i * zones + i for i in range(zones)]

        od_matrix_param.grad.data[diagonal_indices] = 0.0
        
        optimizer.step()
        
        with torch.no_grad():
            od_matrix_param.clamp_(0, 1)
            
        if i % 10 == 0:
            current_loss = loss.item()
            history.append(current_loss)
            if i % 50 == 0:
                print(f"Iter {i:03d} | RMSE: {current_loss:.4f}")

    final_rmse = history[-1]
    print(f"\nFinal RMSE: {final_rmse:.4f}")
    
    final_od_norm = od_matrix_param.detach().cpu().numpy().reshape(1, -1)
    
    final_od_real = scaler_x_sci.inverse_transform(final_od_norm)[0]
    
    results_file = config.RESULTS_DIR / "estimated_od_matrix_gd.csv"
    pd.DataFrame(final_od_real).to_csv(results_file, index=False, header=["Count"])
    print(f"Saved '{results_file}'")

    plt.figure(figsize=(10, 5))
    plt.plot(range(0, len(history)*10, 10), history, linewidth=2, color='green')
    plt.title("Gradient-Based Optimization Convergence (M-SPSA Limit)")
    plt.xlabel("Iteration")
    plt.ylabel("RMSE Loss")
    plt.grid(True, alpha=0.3)
    conv_file = config.RESULTS_DIR / "mspsa_convergence.png"
    plt.savefig(conv_file)
    
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