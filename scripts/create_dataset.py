import os
import sys
import subprocess
import numpy as np
import xml.etree.ElementTree as ET
import multiprocessing
import tempfile
import shutil
import libsumo
import pandas as pd
from tqdm import tqdm

import config

NET_FILE = config.NET_FILE
BASELINE_OD_FILE = config.BASELINE_MATRIX_FILE
DETECTORS_FILE = config.DETECTORS_FILE
TAZ_FILE = config.TAZ_FILE
OUTPUT_DATASET = config.DATASET_FILE

NUM_SAMPLES = 7500
REPLICATIONS_PER_SAMPLE = 4
SIMULATION_STEPS = 900
WARMUP_STEPS = 600

def parse_detectors(detector_file):
    tree = ET.parse(detector_file)
    root = tree.getroot()
    
    detector_map = {}
    parent_sensors = set()
    
    for loop in root.findall('inductionLoop'):
        lane_sensor_id = loop.get('id')
        parent_id = "_".join(lane_sensor_id.split('_')[:-1])
        detector_map[lane_sensor_id] = parent_id
        parent_sensors.add(parent_id)
        
    return detector_map, sorted(list(parent_sensors))

def parse_baseline_matrix(matrix_file):
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
        
        while libsumo.simulation.getTime() < SIMULATION_STEPS:
            libsumo.simulationStep()
            
        sensor_volumes = {s_id: 0 for s_id in unique_sensors}
        
        for lane_sensor_id, parent_sensor_id in detector_map.items():
            try:
                vol = libsumo.inductionloop.getLastIntervalVehicleNumber(lane_sensor_id)
                if vol != -1: 
                    sensor_volumes[parent_sensor_id] += vol
            except:
                pass

        libsumo.close()
        return sensor_volumes

    except Exception as e:
        try:
            libsumo.close()
        except:
            pass
        return {s_id: 0 for s_id in unique_sensors}

def worker_process_scenario(args):
    sample_idx, baseline_od, detector_map, unique_sensors = args
    
    temp_dir = tempfile.mkdtemp(prefix=f"proc_{sample_idx}_")
    
    try:
        current_counts = []
        sampled_od_pairs = []
        
        TAU = 3.0 

        for entry in baseline_od:
            max_demand = int(entry['count'] * TAU)
            
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

        temp_od_path = os.path.join(temp_dir, "od_matrix.xml")
        root = ET.Element("data")
        interval = ET.SubElement(root, "interval", id="15min", begin="0", end=str(SIMULATION_STEPS))
        
        for entry in sampled_od_pairs:
            ET.SubElement(interval, "tazRelation", 
                          **{
                              "from": entry['from'], 
                              "to": entry['to'], 
                              "count": str(entry['count'] * 3) 
                          })
            
        tree = ET.ElementTree(root)
        tree.write(temp_od_path)

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
            raise e

        route_file = os.path.join(temp_dir, "routes.rou.xml")
        subprocess.run([
            "duarouter", 
            "-n", NET_FILE, 
            "-r", trips_file, 
            "-o", route_file,
            "--ignore-errors", "true",
            "--no-warnings", "true"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

        accumulated_volumes = {s_id: 0 for s_id in unique_sensors}
        
        for i in range(REPLICATIONS_PER_SAMPLE):
            seed = (sample_idx * 100) + i
            volumes = run_single_simulation(seed, route_file, detector_map, unique_sensors)
            
            for s_id, vol in volumes.items():
                accumulated_volumes[s_id] += vol

        averaged_volumes = [accumulated_volumes[s_id] / REPLICATIONS_PER_SAMPLE for s_id in unique_sensors]
        
        row = current_counts + averaged_volumes
        return row

    except KeyboardInterrupt:
        return None
    except Exception as e:
        print(f"Error processing sample {sample_idx}: {e}")
        return None
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
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
    
    od_cols = [f"od_{x['from']}_{x['to']}" for x in baseline_od]
    sensor_cols = unique_sensors
    columns = od_cols + sensor_cols
    
    results = []
    
    num_workers = max(1, multiprocessing.cpu_count() - 2)
    print(f"Using {num_workers} parallel workers.")

    pool = multiprocessing.Pool(processes=num_workers)

    try:
        iterator = pool.imap(worker_process_scenario, tasks)
        
        with tqdm(total=NUM_SAMPLES, unit="sample", desc="Generating Data") as pbar:
            for row in iterator:
                if row:
                    results.append(row)
                pbar.update(1)

    except KeyboardInterrupt:
        print("\n\n!!! Interrupted by User (Ctrl+C) !!!")
        print("Terminating worker processes...")
        pool.terminate()
        pool.join()
        print("Workers terminated.")
        
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        pool.terminate()
        pool.join()

    else:
        pool.close()
        pool.join()

    if len(results) > 0:
        print(f"\nCollected {len(results)} valid samples.")
        print("Saving to CSV...")
        df = pd.DataFrame(results, columns=columns)
        df.to_csv(OUTPUT_DATASET, index=False)
        print(f"Dataset successfully saved to {OUTPUT_DATASET}")
    else:
        print("\nNo samples were generated before the process stopped.")