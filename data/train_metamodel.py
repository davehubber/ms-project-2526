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

# --- CONFIG ---
CSV_FILE = "simulation_dataset.csv"
MODEL_PATH = "traffic_metamodel.pth"
BATCH_SIZE = 64
EPOCHS = 1000  # Min et al. (2024) suggests ample training for convergence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- DATA LOADING ---
if not os.path.exists(CSV_FILE):
    raise FileNotFoundError("Dataset not found. Wait for Student A.")

df = pd.read_csv(CSV_FILE)

# Auto-detect columns
input_cols = [c for c in df.columns if c.startswith("od_")]
output_cols = [c for c in df.columns if any(x in c for x in ["_flow", "_speed", "_occupancy"])]

X = df[input_cols].values.astype(np.float32)
y = df[output_cols].values.astype(np.float32)

# Split and Scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train = scaler_x.fit_transform(X_train)
X_test = scaler_x.transform(X_test)
y_train = scaler_y.fit_transform(y_train)
y_test = scaler_y.transform(y_test)

# Save metadata for Student C
joblib.dump(scaler_x, "scaler_x.pkl")
joblib.dump(scaler_y, "scaler_y.pkl")
joblib.dump(input_cols, "input_cols.pkl")
joblib.dump(output_cols, "output_cols.pkl")

# --- MODEL ---
# Architecture based on Min et al. (2024): Input -> 16 -> 128 -> 128 -> 256 -> Output
# Note: Adjusted first layer to match your actual input size
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

# --- TRAINING ---
train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
test_ds = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

best_loss = float('inf')

print("Starting training...")
for epoch in range(EPOCHS):
    model.train()
    for bx, by in train_loader:
        optimizer.zero_grad()
        pred = model(bx.to(device))
        loss = criterion(pred, by.to(device))
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for bx, by in test_loader:
            pred = model(bx.to(device))
            val_loss += criterion(pred, by.to(device)).item()
    
    avg_val = val_loss / len(test_loader)
    if avg_val < best_loss:
        best_loss = avg_val
        torch.save(model.state_dict(), MODEL_PATH)
        
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Val MAE {avg_val:.5f}")

print(f"Done. Best MAE: {best_loss:.5f}")