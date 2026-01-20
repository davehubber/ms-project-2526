# Deep Neural Network OD Estimation (SPSA/M-SPSA)

This project implements a Deep Neural Network (DNN) as a metamodel (surrogate model) for SUMO traffic simulations. The goal is to estimate Origin-Destination (OD) matrices from traffic count data using optimization algorithms: **SPSA (Simultaneous Perturbation Stochastic Approximation)** and **M-SPSA (Modified SPSA)**.

By approximating the complex and computationally expensive SUMO simulation with a differentiable neural network, we can perform OD estimation orders of magnitude faster than traditional simulation-in-the-loop approaches.

## Project Structure

The project is organized as follows:

- **`data/`**: detailed data storage.
    - **`networks/`**: SUMO network files (`.net.xml`, `.taz.xml`, detectors).
    - **`sensor_data/`**: Real-world traffic data (processed and raw).
    - **`routes/`**: Route files and calibrated trip data.
    - **`generated/`**: Intermediate files (baseline matrices, datasets).
    - **`models/`**: Saved PyTorch models (`.pth`) and scalers (`.pkl`).
- **`scripts/`**: Python scripts for every stage of the pipeline.
    - `config.py`: Central configuration for all file paths.
    - `generate_*.py`: Data processing and preparation.
    - `create_*.py`: Dataset generation and matrix formatted.
    - `train_metamodel.py`: DNN training script.
    - `run_*.py`: Optimization execution files.
- **`results/`**: Output directory for estimation results (CSVs, plots).

## Installation & Setup

### 1. Prerequisites
- **Python 3.12** (Required for test compatibility).
- **SUMO (Simulation of Urban MObility)**: Ensure SUMO is installed and `SUMO_HOME` environment variable is set.

### 2. Environment
Create a virtual environment and install dependencies:

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

## Replication Guide

Follow these steps to replicate the experiments from scratch.

### Step 1: Data Preparation (`ground_truth.csv`)
**Note:** The raw sensor data is too large for the repository. You must place your raw CSV files (e.g., `2013AEDL.csv`, etc.) into `data/sensor_data/raw_years/`.

Run the processing script to generate the clean ground truth:
```bash
python scripts/generate_ground_truth.py
```
This filters for valid sensors, removes holidays/weekends, and aggregates data.

### Step 2: Baseline Matrix Creation
We generate a prior OD matrix to serve as a starting point or baseline for sampling.

1. **Agregate Counts**:
   ```bash
   python scripts/generate_counts.py
   ```
   *Generates `data/generated/counts.xml` from sensor data.*

2. **Generate Dummy Trips**:
   ```bash
   python scripts/generate_taz_trips.py
   ```
   *Generates `data/routes/taz_dummy_trips.xml` (all-to-all TAZ connectivity).*

3. **External SUMO Tools**:
   Run the following commands (ensure your paths match your system or run from the project root adjusting paths):
   
   **A. Generate Route Pool (`duarouter`)**:
   ```bash
   duarouter --net-file data/networks/net.net.xml \
             --additional-files data/networks/taz.xml \
             --route-files data/routes/taz_dummy_trips.xml \
             --output-file data/routes/routes_pool.rou.xml \
             --ignore-errors
   ```
   
   **B. Calibrate Trips (`routeSampler.py`)**:
   ```bash
   # Note: PATH to routeSampler depends on your SUMO installation
   python "path/to/sumo/tools/routeSampler.py" \
          --route-files data/routes/routes_pool.rou.xml \
          --edgedata-files data/generated/counts.xml \
          --output-file data/routes/calibrated_daily_trips.rou.xml \
          --optimize full
   ```

4. **Create Final Baseline Matrix**:
   ```bash
   python scripts/create_matrix.py
   ```
   *Converts the calibrated trips into a SUMO OD Matrix XML (`data/generated/baseline_matrix.xml`).*

### Step 3: Dataset Generation
Generate (X, y) pairs for training the Neural Network. This runs thousands of short SUMO simulations to map OD Counts (Inputs) to Detector Volumes (Labels).

```bash
python scripts/create_dataset.py
```
*Output: `data/generated/dataset.csv`*

### Step 4: Train Metamodel
Train the Deep Neural Network to approximate the traffic simulator.

```bash
python scripts/train_metamodel.py
```
- Trains the model.
- Saves model weights to `data/models/metamodel.pth`.
- Saves scalers to `data/models/scaler_x.pkl` and `scaler_y.pkl`.

### Step 5: Run Estimation (SPSA / M-SPSA)
Now you can perform OD Matrix Estimation using the trained metamodel.

- **Option A: SPSA (Stochastic)**
  ```bash
  python scripts/run_spsa.py
  ```
- **Option B: M-SPSA (Gradient-Based Surrogate)**
  ```bash
  python scripts/run_mspsa.py
  ```

Results (CSVs and Plots) will be saved to the `results/` directory.

### Step 6: Generate Route Files
To visualize the estimated OD matrix in SUMO:
```bash
python scripts/generate_routes.py
```
this converts the estimated matrix CSV into `.rou.xml` files for simulation.

## Usage & Customization

### Testing the Model
If you only want to test the model architecture or checking predictions without full estimation, you can use `scripts/train_metamodel.py` which includes a test split evaluation and produces "Reality Gap" plots.

### Customizing Estimation
- **Time Intervals**: The current setup is optimized for specific time snapshots (e.g., midnight or peak hours depending on dataset configuration). To estimate for different times, modify `load_target_volumes` in `run_spsa.py` / `run_mspsa.py` and ensure your `dataset.csv` reflects the appropriate traffic dynamics.
- **Hyperparameters**: Check `scripts/config.py` (paths) or the headers of individual scripts for learning rates, perturbation sizes (c, a), and network architecture.