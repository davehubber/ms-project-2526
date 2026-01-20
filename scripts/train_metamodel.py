import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import joblib
import os

import config

INPUT_SIZE = 676
OUTPUT_SIZE = 48
HIDDEN_LAYERS = [64, 256, 256, 512]

LEARNING_RATE = 0.0001
EPOCHS = 10000
BATCH_SIZE = 64
DROPOUT_RATE = 0.1
L2_LAMBDA = 0.00001
TEST_SPLIT = 0.2
DATASET_FILE = config.DATASET_FILE
MODEL_SAVE_PATH = config.METAMODEL_FILE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_and_process_data(filename):
    print("Loading dataset...")
    df = pd.read_csv(filename)
    
    X = df.iloc[:, :INPUT_SIZE].values
    y = df.iloc[:, INPUT_SIZE:].values
    
    print(f"Input shape: {X.shape}, Output shape: {y.shape}")
    
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    X_scaled = scaler_x.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=TEST_SPLIT, random_state=42
    )
    
    train_dataset = TensorDataset(
        torch.from_numpy(X_train).float(), 
        torch.from_numpy(y_train).float()
    )
    test_dataset = TensorDataset(
        torch.from_numpy(X_test).float(), 
        torch.from_numpy(y_test).float()
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    
    return train_loader, test_loader, scaler_x, scaler_y

class ODMetamodel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims):
        super(ODMetamodel, self).__init__()
        
        layers = []
        
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p=DROPOUT_RATE)) 
        
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=DROPOUT_RATE))
            
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

def train_model():
    if not os.path.exists(DATASET_FILE):
        print(f"Error: {DATASET_FILE} not found. Please run create_dataset.py first.")
        return None, None, None, None

    train_loader, test_loader, scaler_x, scaler_y = load_and_process_data(DATASET_FILE)
    
    model = ODMetamodel(INPUT_SIZE, OUTPUT_SIZE, HIDDEN_LAYERS).to(device)
    print(f"Model Architecture: {model}")
    
    criterion = nn.L1Loss()
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=L2_LAMBDA)
    
    history = {'train_loss': [], 'test_loss': []}
    
    best_test_loss = float('inf')
    
    print("\nStarting training...")
    for epoch in range(EPOCHS):
        model.train()
        batch_losses = []
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
            
        avg_train_loss = np.mean(batch_losses)
        history['train_loss'].append(avg_train_loss)
        
        model.eval()
        test_batch_losses = []
        with torch.no_grad():
            for test_x, test_y in test_loader:
                test_x, test_y = test_x.to(device), test_y.to(device)
                test_outputs = model(test_x)
                test_loss = criterion(test_outputs, test_y).item()
                test_batch_losses.append(test_loss)
        
        avg_test_loss = np.mean(test_batch_losses)
        history['test_loss'].append(avg_test_loss)
        
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}] | Train MAE: {avg_train_loss:.6f} | Test MAE: {avg_test_loss:.6f}")

    print(f"\nTraining complete.")
    print(f"Best model was saved to '{MODEL_SAVE_PATH}' with Test MAE: {best_test_loss:.6f}")

    joblib.dump(scaler_x, config.SCALER_X_FILE)
    joblib.dump(scaler_y, config.SCALER_Y_FILE)
    print("Scalers saved to scaler_x.pkl and scaler_y.pkl")
    
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    
    return history, model, test_loader, scaler_y

def plot_results(history, model, test_loader, scaler_y):
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train MAE Loss')
    plt.plot(history['test_loss'], label='Test MAE Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error (Normalized)')
    plt.title('Training and Validation Loss [cite: 394, 473]')
    plt.legend()
    plt.grid(True)
    plt.savefig(str(config.RESULTS_DIR / 'loss_curve.png'))
    print("Saved loss_curve.png")
    
    model.eval()
    with torch.no_grad():
        for test_x, test_y in test_loader:
            test_x = test_x.to(device)
            preds_scaled = model(test_x).cpu().numpy()
            targets_scaled = test_y.numpy()
            
            preds_real = scaler_y.inverse_transform(preds_scaled)
            targets_real = scaler_y.inverse_transform(targets_scaled)
            break

    num_plot_sensors = min(10, OUTPUT_SIZE)
    plot_data = []
    labels = []
    
    for i in range(num_plot_sensors):
        plot_data.append(targets_real[:, i])
        labels.append(f'Det {i} (Real)')
        plot_data.append(preds_real[:, i])
        labels.append(f'Det {i} (Pred)')
        
    plt.figure(figsize=(12, 6))
    positions = []
    for i in range(num_plot_sensors):
        positions.extend([i*3 + 1, i*3 + 2])
        
    bplot = plt.boxplot(plot_data, positions=positions, patch_artist=True, widths=0.6)
    
    colors = ['#d62728', '#1f77b4'] * num_plot_sensors
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    plt.xticks([i*3 + 1.5 for i in range(num_plot_sensors)], [f'Det {i}' for i in range(num_plot_sensors)])
    plt.ylabel('Vehicle Volume (15 min)')
    plt.title('Box Plot of Real vs Predicted Detector Data (First 10 Detectors)')
    
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='#d62728', lw=4),
                    Line2D([0], [0], color='#1f77b4', lw=4)]
    plt.legend(custom_lines, ['Real Data', 'Predicted Data'])
    
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(str(config.RESULTS_DIR / 'boxplot_comparison.png'))
    print("Saved boxplot_comparison.png")

if __name__ == "__main__":
    history, model, test_loader, scaler_y = train_model()
    if history:
        plot_results(history, model, test_loader, scaler_y)