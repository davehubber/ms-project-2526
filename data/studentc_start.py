from metamodel_inference import DigitalTwin
import numpy as np

# Initialize the "Virtual Simulator"
sim = DigitalTwin()

# Check how many OD pairs you need to optimize
num_vars = sim.get_input_size()
print(f"Optimizing {num_vars} OD pairs.")

# --- INSIDE YOUR SPSA LOOP ---

# 1. Create your OD Matrix (the variable you are changing)
# Example: random numbers
current_od = np.random.uniform(10, 100, num_vars)

# 2. Run the simulation (Takes milliseconds!)
results = sim.simulate(current_od)

# 3. Read the results to calculate your Error/Loss
# 'results' is a dictionary: {'Detector1_flow': 120.5, 'Detector1_speed': 45.2, ...}
flow_det_1 = results['det_0_i0_flow'] 
print(f"Flow at detector 0: {flow_det_1}")