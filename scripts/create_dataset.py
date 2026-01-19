import os
import sys
import subprocess
import numpy as np
import xml.etree.ElementTree as ET
import multiprocessing
from functools import partial
import tempfile
import shutil
import libsumo
import pandas as pd
from tqdm import tqdm  # pip install tqdm

import config

# --- Configuration ---
# Ensure these files are in the same directory or provide full paths
NET_FILE = config.NET_FILE
BASELINE_OD_FILE = config.BASELINE_MATRIX_FILE
DETECTORS_FILE = config.DETECTORS_FILE
TAZ_FILE = config.TAZ_FILE
OUTPUT_DATASET = config.DATASET_FILE

NUM_SAMPLES = 7500
REPLICATIONS_PER_SAMPLE = 4
SIMULATION_STEPS = 900  # 15 minutes total
WARMUP_STEPS = 600      # First 10 minutes are warmup

# --- Helper Functions ---

def parse_detectors(detector_file):
    """
    Parses the detector XML to create a mapping from lane-id to sensor-id.
    Example: 121725_C_0 -> 121725_C
    """
    tree = ET.parse(detector_file)
    root = tree.getroot()
    
    detector_map = {}
    parent_sensors = set()
    
    for loop in root.findall('inductionLoop'):
        lane_sensor_id = loop.get('id')
        # Logic: Parent ID is the ID without the last underscore and digit (e.g. _0)
        parent_id = "_".join(lane_sensor_id.split('_')[:-1])
        detector_map[lane_sensor_id] = parent_id
        parent_sensors.add(parent_id)
        
    return detector_map, sorted(list(parent_sensors))

def parse_baseline_matrix(matrix_file):
    """
    Parses the baseline OD matrix to extract OD pairs and their counts.
    """
    tree = ET.parse(matrix_file)
    root = tree.getroot()
    interval = root.find('interval')
    
    od_data = []
    for rel in interval.findall('tazRelation'):
        entry = {
            'from': rel.get('from'),
            'to': rel.get('to'),
            'count': int(rel.get('count'))
        }
        od_data.append(entry)
    
    return od_data

def run_single_simulation(run_seed, route_file, detector_map, unique_sensors):
    """
    Runs a single SUMO simulation using libsumo.
    Returns a dictionary of {sensor_id: volume} for the 600s-900s interval.
    """
    # Start libsumo without GUI for speed
    sumo_cmd = [
        "sumo",
        "-n", NET_FILE,
        "-r", route_file,
        "-a", DETECTORS_FILE,
        "--seed", str(run_seed),
        "--begin", "0",
        "--end", str(SIMULATION_STEPS),
        "--no-step-log", "true",
        "--no-warnings", "true"
    ]
    
    try:
        libsumo.start(sumo_cmd)
        
        # Fast forward simulation to the end (Step 900)
        while libsumo.simulation.getTime() < SIMULATION_STEPS:
            libsumo.simulationStep()
            
        # At t=900, the last interval (600-900) data is available.
        sensor_volumes = {s_id: 0 for s_id in unique_sensors}
        
        for lane_sensor_id, parent_sensor_id in detector_map.items():
            try:
                # getLastIntervalVehicleNumber returns the count for the last completed period (the 300s window)
                vol = libsumo.inductionloop.getLastIntervalVehicleNumber(lane_sensor_id)
                if vol != -1: 
                    sensor_volumes[parent_sensor_id] += vol
            except:
                pass # Handle cases where edges might be culled or sensors invalid

        libsumo.close()
        return sensor_volumes

    except Exception as e:
        # print(f"Error in simulation: {e}") # Optional: suppress noisy errors on cancel
        try:
            libsumo.close()
        except:
            pass
        return {s_id: 0 for s_id in unique_sensors}

def worker_process_scenario(args):
    """
    Worker function to process a single sample (1 scenario, N replications).
    """
    # Unpack arguments
    sample_idx, baseline_od, detector_map, unique_sensors = args
    
    # Create unique temp directory for this process
    temp_dir = tempfile.mkdtemp(prefix=f"proc_{sample_idx}_")
    
    try:
        # 1. Sample Demand (Poisson)
        current_counts = []
        sampled_od_pairs = []
        
        TAU = 3.0 

        for entry in baseline_od:
            # 1. Calculate the specific upper limit for this OD pair
            # The paper implies the baseline is the "base" demand, 
            # and the max demand is scaled by tau.
            max_demand = int(entry['count'] * TAU)
            
            # 2. Sample Uniformly (randint) as implied by "upper limit" usage (Source 384)
            # randint is inclusive of low, exclusive of high, so use max_demand + 1
            if max_demand > 0:
                new_count = np.random.randint(0, max_demand + 1)
            else:
                new_count = 0
                
            current_counts.append(new_count)
            
            sampled_od_pairs.append({
                'from': entry['from'],
                'to': entry['to'],
                'count': new_count 
            })

        # 2. Create Temporary OD Matrix XML
        temp_od_path = os.path.join(temp_dir, "od_matrix.xml")
        root = ET.Element("data")
        interval = ET.SubElement(root, "interval", id="15min", begin="0", end=str(SIMULATION_STEPS))
        
        # Scale 5-min baseline count to 15-min duration for od2trips
        for entry in sampled_od_pairs:
            ET.SubElement(interval, "tazRelation", 
                          **{
                              "from": entry['from'], 
                              "to": entry['to'], 
                              "count": str(entry['count'] * 3) 
                          })
            
        tree = ET.ElementTree(root)
        tree.write(temp_od_path)

        # 3. OD2TRIPS -> DUAROUTER
        trips_file = os.path.join(temp_dir, "trips.trips.xml")
        
        try:
            subprocess.run([
                "od2trips", 
                "-n", TAZ_FILE, 
                "-z", temp_od_path, 
                "-o", trips_file,
                "--vtype", "DEFAULT_VEHTYPE"
            ], check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            # print(f"SUMO Error: {e.stderr}") 
            raise e

        # Route trips using duarouter
        route_file = os.path.join(temp_dir, "routes.rou.xml")
        subprocess.run([
            "duarouter", 
            "-n", NET_FILE, 
            "-r", trips_file, 
            "-o", route_file,
            "--ignore-errors", "true",
            "--no-warnings", "true"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

        # 4. Run Replications
        accumulated_volumes = {s_id: 0 for s_id in unique_sensors}
        
        for i in range(REPLICATIONS_PER_SAMPLE):
            seed = (sample_idx * 100) + i
            volumes = run_single_simulation(seed, route_file, detector_map, unique_sensors)
            
            for s_id, vol in volumes.items():
                accumulated_volumes[s_id] += vol

        # 5. Average Results
        averaged_volumes = [accumulated_volumes[s_id] / REPLICATIONS_PER_SAMPLE for s_id in unique_sensors]
        
        # 6. Construct Row: [OD_Counts... , Sensor_Volumes...]
        row = current_counts + averaged_volumes
        return row

    except KeyboardInterrupt:
        # Allow workers to exit cleanly on interrupt
        return None
    except Exception as e:
        print(f"Error processing sample {sample_idx}: {e}")
        return None
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

# --- Main Execution ---

if __name__ == "__main__":
    # Validate required files
    required_files = [NET_FILE, BASELINE_OD_FILE, DETECTORS_FILE, TAZ_FILE]
    for f in required_files:
        if not os.path.exists(f):
            print(f"Error: Required file '{f}' not found.")
            sys.exit(1)

    print("Parsing configuration files...")
    detector_map, unique_sensors = parse_detectors(DETECTORS_FILE)
    baseline_od = parse_baseline_matrix(BASELINE_OD_FILE)
    
    print(f"Found {len(baseline_od)} OD pairs and {len(unique_sensors)} unique aggregated sensors.")
    print(f"Starting generation of {NUM_SAMPLES} samples with {REPLICATIONS_PER_SAMPLE} replications each.")
    print("Press Ctrl+C at any time to stop and save current progress.\n")
    
    tasks = [(i, baseline_od, detector_map, unique_sensors) for i in range(NUM_SAMPLES)]
    
    # Define DataFrame columns
    od_cols = [f"od_{x['from']}_{x['to']}" for x in baseline_od]
    sensor_cols = unique_sensors
    columns = od_cols + sensor_cols
    
    results = []
    
    num_workers = max(1, multiprocessing.cpu_count() - 2)
    print(f"Using {num_workers} parallel workers.")

    # Create the pool outside the try/except block so we can manage it manually if needed
    pool = multiprocessing.Pool(processes=num_workers)

    try:
        # Use imap to get results as they complete
        # tqdm wraps the iterator to show a progress bar
        iterator = pool.imap(worker_process_scenario, tasks)
        
        with tqdm(total=NUM_SAMPLES, unit="sample", desc="Generating Data") as pbar:
            for row in iterator:
                if row:
                    results.append(row)
                pbar.update(1)

    except KeyboardInterrupt:
        print("\n\n!!! Interrupted by User (Ctrl+C) !!!")
        print("Terminating worker processes...")
        pool.terminate()  # Immediately stop worker processes
        pool.join()       # Wait for them to clean up
        print("Workers terminated.")
        
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        pool.terminate()
        pool.join()

    else:
        # If loop finishes normally
        pool.close()
        pool.join()

    # --- Save Logic (Runs on both success and interrupt) ---
    if len(results) > 0:
        print(f"\nCollected {len(results)} valid samples.")
        print("Saving to CSV...")
        df = pd.DataFrame(results, columns=columns)
        df.to_csv(OUTPUT_DATASET, index=False)
        print(f"Dataset successfully saved to {OUTPUT_DATASET}")
    else:
        print("\nNo samples were generated before the process stopped.")