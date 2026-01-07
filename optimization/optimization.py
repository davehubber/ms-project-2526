import numpy as np
import torch
import torch.nn as nn

# ==============================================================================
# NOTE FOR STUDENT 2 (Data Scientist):
# This optimizer expects your model to be a standard PyTorch Class (nn.Module).
# It relies on your model having a standard .forward() method.
# ==============================================================================

class MSPSAOptimizer:
    def __init__(self, model, loss_func_numpy, a=0.001, c=0.1, A=100, alpha=0.602, gamma=0.101):
        """
        MSPSA Optimizer: Combines SPSA gradient with Analytical Neural Net gradient.
        
        Args:
            model (torch.nn.Module): The trained Neural Network provided by Student 2.
                                     IMPORTANT: This model must handle input/output shapes 
                                     matching the OD matrix and Detector counts.
            loss_func_numpy (callable): Function to calc RMSE (for logging/SPSA step).
            a, c, A, alpha, gamma: SPSA Hyperparameters. 
                                   'a' should be small (e.g., 0.001) because NN gradients are strong.
        """
        self.model = model
        self.loss_func = loss_func_numpy
        self.a = a
        self.c = c
        self.A = A
        self.alpha = alpha
        self.gamma = gamma

    def get_analytical_gradient(self, theta_numpy, target_numpy):
        """
        Calculates the EXACT gradient of the Loss w.r.t Inputs using PyTorch Autograd.
        
        STUDENT 2 NOTE: 
        This function uses your model's computational graph. 
        Ensure your model does NOT detach gradients in its forward pass!
        """
        # 1. Prepare Data
        theta_tensor = torch.from_numpy(theta_numpy).float().clone().detach()
        target_tensor = torch.from_numpy(target_numpy).float().clone().detach()
        
        # 2. Enable Gradient Tracking on Input (OD Matrix)
        # This allows us to see how changing the OD Matrix affects the Loss
        theta_tensor.requires_grad = True
        
        # 3. Forward Pass
        self.model.eval() # Eval mode is crucial (disables dropout/batchnorm randomness)
        pred_tensor = self.model(theta_tensor)
        
        # 4. Compute Loss (MSE is standard for gradient calculation)
        loss = nn.MSELoss()(pred_tensor, target_tensor)
        
        # 5. Backward Pass (Analytical Gradient)
        loss.backward()
        
        # 6. Extract Gradient
        return theta_tensor.grad.detach().numpy()

    def predict_numpy(self, theta_numpy):
        """Helper to run inference without gradients."""
        input_tensor = torch.from_numpy(theta_numpy).float()
        self.model.eval()
        with torch.no_grad():
            return self.model(input_tensor).numpy()

    def optimize(self, initial_theta, target_data, max_iterations=200):
        """
        Main MSPSA Optimization Loop.
        """
        theta = np.array(initial_theta, dtype=float)
        p = len(theta)
        loss_history = []
        
        # [Min et al. 2024] Scale Vector 's' calculation
        # Adjusts step size based on magnitude of parameters to handle high dimensions
        scale_vector = np.maximum(1.0, np.abs(theta))
        
        print(f"--- Starting MSPSA Optimization ({max_iterations} iterations) ---")
        
        for k in range(max_iterations):
            # 1. Update Decay Rates
            ak = self.a / (k + 1 + self.A) ** self.alpha
            ck = self.c / (k + 1) ** self.gamma
            
            # 2. SPSA Gradient Step (Perturbation)
            delta = 2 * np.round(np.random.rand(p)) - 1 # Bernoulli +/- 1
            
            theta_plus = np.maximum(0, theta + ck * delta)
            theta_minus = np.maximum(0, theta - ck * delta)
            
            y_plus = self.predict_numpy(theta_plus)
            y_minus = self.predict_numpy(theta_minus)
            
            loss_plus = self.loss_func(y_plus, target_data)
            loss_minus = self.loss_func(y_minus, target_data)
            
            # Approx Gradient (SPSA)
            g_spsa = (loss_plus - loss_minus) / (2 * ck * delta + 1e-8)
            
            # 3. Analytical Gradient Step (Neural Network)
            # [Ho et al. 2023] Extract exact gradient knowledge from Student 2's model
            g_nn = self.get_analytical_gradient(theta, target_data)
            
            # 4. Combine Gradients (MSPSA)
            g_final = (g_spsa + g_nn) / 2.0
            
            # 5. Update Theta
            # [Min et al. 2024] Apply scale vector to the step size
            step = ak * scale_vector * g_final
            theta = theta - step
            
            # Projection Constraint (OD flows >= 0)
            theta = np.maximum(0, theta)
            
            # Logging
            curr_loss = self.loss_func(self.predict_numpy(theta), target_data)
            loss_history.append(curr_loss)
            
            if k % 10 == 0:
                print(f"Iter {k}: Loss = {curr_loss:.6f}")

        return theta, loss_history

# --- INTEGRATION SECTION FOR STUDENT 2 ---
def load_student2_model():
    """
    TODO STUDENT 2: Populate this function to load your trained PyTorch model.
    """
    print("\n[INFO] Loading Traffic Metamodel...")
    
    # --- INSTRUCTIONS ---
    # 1. Initialize your model architecture
    #    e.g. model = TrafficModel(input_dim=50, output_dim=50)
    # 2. Load weights
    #    e.g. model.load_state_dict(torch.load("model_weights.pth"))
    # 3. Set to eval mode
    #    model.eval()
    # 4. Return model
    
    # --- FOR NOW: Returning Mock Model so the code runs ---
    # DELETE THIS MOCK WHEN REAL MODEL IS READY
    input_dim = 50
    output_dim = 50
    model = nn.Sequential(
        nn.Linear(input_dim, 128),
        nn.ReLU(),
        nn.Linear(128, output_dim)
    )
    print("[WARN] Using DUMMY MOCK MODEL. Replace in 'load_student2_model()'.")
    return model

if __name__ == "__main__":
    # 1. Configuration
    INPUT_DIM = 50   # TODO: Update to match real OD Matrix size
    ITERATIONS = 500 # Sufficient for convergence
    LEARNING_RATE = 0.5 # 'a' parameter - tune if convergence is too slow/unstable

    # 2. Load Model
    model = load_student2_model()
    
    # 3. Setup Data (Simulation placeholder)
    # In real deployment, 'target_data' comes from the Real World sensors
    print(f"[INFO] Generating dummy target data (Ground Truth)...")
    true_od = np.random.randint(50, 200, size=INPUT_DIM)
    
    # We use the model itself to generate 'true' data for this test
    # (Assuming the model perfectly captures reality for the sake of optimizer testing)
    with torch.no_grad():
        target_tensor = model(torch.from_numpy(true_od).float())
        target_data = target_tensor.numpy()
    
    # 4. Initial Guess (Perturbed OD)
    initial_guess = true_od * np.random.uniform(0.5, 1.5, size=INPUT_DIM)
    
    # 5. Run Optimizer
    print(f"[INFO] Initializing MSPSA Optimizer (alpha={LEARNING_RATE})...")
    def rmse(pred, target): return np.sqrt(np.mean((pred - target)**2))
    
    optimizer = MSPSAOptimizer(model, rmse, a=LEARNING_RATE)
    
    final_od, history = optimizer.optimize(initial_guess, target_data, max_iterations=ITERATIONS)
    
    # 6. Report
    print(f"\n--- Results ---")
    print(f"Final RMSE: {history[-1]:.4f}")
    
    # Verification
    print(f"True OD (First 5): {true_od[:5]}")
    print(f"Calc OD (First 5): {np.round(final_od[:5]).astype(int)}")
