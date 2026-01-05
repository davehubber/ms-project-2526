import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
import matplotlib.pyplot as plt
import time

# --- CONFIG ---
CSV_FILE = "simulation_dataset.csv"
MODEL_PATH = "traffic_metamodel.pth"
PLOT_PATH = "loss_curve.png"
HISTORY_PATH = "training_history.csv"
BATCH_SIZE = 64
EPOCHS = 1000  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- DATA LOADING ---
if not os.path.exists(CSV_FILE):
    raise FileNotFoundError(f"Dataset {CSV_FILE} not found. Ensure the file is in the directory.")

print(f"Loading {CSV_FILE}...")
df = pd.read_csv(CSV_FILE)

# Auto-detect columns
input_cols = [c for c in df.columns if c.startswith("od_")]
output_cols = [c for c in df.columns if any(x in c for x in ["_flow", "_speed", "_occupancy"])]

print(f"Detected {len(input_cols)} input columns and {len(output_cols)} output columns.")

X = df[input_cols].values.astype(np.float32)
y = df[output_cols].values.astype(np.float32)

# Split and Scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train = scaler_x.fit_transform(X_train)
X_test = scaler_x.transform(X_test)
y_train = scaler_y.fit_transform(y_train)
y_test = scaler_y.transform(y_test)

# Save metadata
joblib.dump(scaler_x, "scaler_x.pkl")
joblib.dump(scaler_y, "scaler_y.pkl")
joblib.dump(input_cols, "input_cols.pkl")
joblib.dump(output_cols, "output_cols.pkl")

# --- MODEL ---
class TrafficModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        return self.net(x)

model = TrafficModel(len(input_cols), len(output_cols)).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
criterion = nn.L1Loss() # MAE Loss

# --- TRAINING SETUP ---
train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
test_ds = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

# History trackers
history = {
    'epoch': [],
    'train_loss': [],
    'val_loss': []
}

best_loss = float('inf')
start_time = time.time()

print(f"Starting training on {device} for {EPOCHS} epochs...")

# --- TRAINING LOOP ---
for epoch in range(EPOCHS):
    model.train()
    running_train_loss = 0.0
    
    for bx, by in train_loader:
        optimizer.zero_grad()
        pred = model(bx.to(device))
        loss = criterion(pred, by.to(device))
        loss.backward()
        optimizer.step()
        running_train_loss += loss.item()

    # Calculate average training loss for this epoch
    avg_train_loss = running_train_loss / len(train_loader)

    # Validation phase
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for bx, by in test_loader:
            pred = model(bx.to(device))
            running_val_loss += criterion(pred, by.to(device)).item()
    
    avg_val_loss = running_val_loss / len(test_loader)

    # Save Checkpoint
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        torch.save(model.state_dict(), MODEL_PATH)

    # Record History
    history['epoch'].append(epoch)
    history['train_loss'].append(avg_train_loss)
    history['val_loss'].append(avg_val_loss)
        
    # Logging
    if epoch % 50 == 0 or epoch == EPOCHS - 1:
        elapsed = time.time() - start_time
        print(f"Epoch {epoch}/{EPOCHS} | Time: {elapsed:.0f}s | "
              f"Train MAE: {avg_train_loss:.5f} | Val MAE: {avg_val_loss:.5f}")

total_time = time.time() - start_time
print(f"\nTraining Complete in {total_time/60:.2f} minutes.")
print(f"Best Validation MAE: {best_loss:.5f}")

# --- REPORTING & PLOTTING ---

# 1. Save history to CSV (for manual graphing later)
hist_df = pd.DataFrame(history)
hist_df.to_csv(HISTORY_PATH, index=False)
print(f"Training history saved to {HISTORY_PATH}")

# 2. Generate Matplotlib Plot
plt.figure(figsize=(10, 6))
plt.plot(history['train_loss'], label='Training Loss (MAE)')
plt.plot(history['val_loss'], label='Validation Loss (MAE)', linestyle="--")
plt.title('Traffic Metamodel Training Convergence')
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Error (Scaled)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(PLOT_PATH)
print(f"Loss plot saved to {PLOT_PATH}")