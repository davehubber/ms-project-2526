import torch
import torch.nn as nn
import joblib
import numpy as np
import os

# Define model structure (Must match training script)
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

class DigitalTwin:
    def __init__(self, model_path="traffic_metamodel.pth"):
        if not os.path.exists(model_path):
            raise FileNotFoundError("Model file missing. Check with Student B.")

        # Load scalers and metadata
        self.scaler_x = joblib.load("scaler_x.pkl")
        self.scaler_y = joblib.load("scaler_y.pkl")
        self.input_cols = joblib.load("input_cols.pkl")
        self.output_cols = joblib.load("output_cols.pkl")
        
        # Load Model
        self.model = TrafficModel(len(self.input_cols), len(self.output_cols))
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model.eval()

    def get_input_size(self):
        return len(self.input_cols)

    def simulate(self, od_matrix_flat):
        """
        Mimics SUMO. 
        Input: List/Array of OD values.
        Output: Dictionary of detector readings.
        """
        # Prepare input
        x = np.array(od_matrix_flat).reshape(1, -1)
        x_scaled = self.scaler_x.transform(x)
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32)

        # Predict
        with torch.no_grad():
            y_scaled = self.model(x_tensor).numpy()

        # Scale back to real units
        y_real = self.scaler_y.inverse_transform(y_scaled)[0]
        
        # Format as dict (ensure no negative flows)
        return {col: max(0.0, val) for col, val in zip(self.output_cols, y_real)}