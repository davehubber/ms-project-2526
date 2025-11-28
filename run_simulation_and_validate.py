import os
import sys
import traci
import sumolib
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import datetime

# --- CONFIGURATION ---
SUMO_BINARY = "sumo-gui"  # Use "sumo" for command line only, "sumo-gui" to see it
CONFIG_FILE = "data/sumo/planner.sumo.cfg.xml"
DETECTOR_OUTPUT_FILE = "data/sumo/output_detectors.xml"
GROUND_TRUTH_FILE = "data/Representative_Workday_Profile.csv"

def run_simulation():
    """Starts SUMO and runs until completion or interruption."""
    print("--- Starting SUMO Simulation ---")
    
    # Locate SUMO binary
    sumo_cmd = [sumolib.checkBinary(SUMO_BINARY), "-c", CONFIG_FILE]
    
    try:
        traci.start(sumo_cmd)
        step = 0
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            step += 1
        print("Simulation finished naturally.")
    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")
    except Exception as e:
        print(f"Simulation error: {e}")
    finally:
        # close() forces SUMO to write the final XML tags
        try:
            traci.close()
        except:
            pass
        print("TraCI closed. Output files should be generated.")

def seconds_to_time_str(seconds):
    """
    Converts seconds (e.g., 32400.0) to 'HH:MM:SS' string.
    Ensures leading zeros (e.g., '09:00:00') to match Pandas CSV format.
    """
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return "{:02d}:{:02d}:{:02d}".format(h, m, s)

def parse_sumo_outputs(xml_file):
    """
    Parses the SUMO detector XML.
    Aggregates lanes (e.g., 121725_C_0 + 121725_C_1) into one ID (121725_C).
    """
    if not os.path.exists(xml_file):
        print(f"Error: Output file {xml_file} not found.")
        return pd.DataFrame()

    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"XML Parse Error: {e}")
        print("Tip: If you force-quit SUMO, the XML might be missing the closing </detectors> tag.")
        print("Try adding </detectors> to the end of the file manually.")
        return pd.DataFrame()

    data = []

    for interval in root.findall('interval'):
        begin_time = float(interval.get('begin'))
        det_id = interval.get('id')
        
        # Parse attributes
        n_veh = float(interval.get('nVehContrib', 0))
        speed = float(interval.get('speed', -1))
        
        # Logic: Detector ID is "121725_C_0". We want "121725_C".
        # We split by the last underscore to remove the lane number.
        sensor_base_id = det_id.rsplit('_', 1)[0]
        
        data.append({
            'interval_begin': begin_time,
            'raw_id': det_id,
            'sensor_id': sensor_base_id,
            'volume': n_veh,
            'speed': speed
        })

    df = pd.DataFrame(data)
    
    if df.empty:
        print("No data found in XML output.")
        return df

    # --- AGGREGATION ---
    # We sum the volume across lanes.
    # For speed, we calculate a weighted average based on volume.
    
    def weighted_avg_speed(x):
        total_vol = x['volume'].sum()
        if total_vol == 0:
            # If no cars, return mean of observed speeds (ignoring -1) or 0
            valid_speeds = x[x['speed'] != -1]['speed']
            return valid_speeds.mean() if not valid_speeds.empty else 0.0
        
        # Weighted average: sum(speed * volume) / sum(volume)
        # Filter out invalid speeds (-1)
        valid = x[x['speed'] != -1]
        if valid.empty:
            return 0.0
        return np.average(valid['speed'], weights=valid['volume'])

    aggregated = df.groupby(['sensor_id', 'interval_begin']).apply(
        lambda x: pd.Series({
            'sim_volume': x['volume'].sum(),
            'sim_speed': weighted_avg_speed(x)
        })
    ).reset_index()

    return aggregated

def load_ground_truth(csv_file):
    """Loads the representative workday CSV."""
    if not os.path.exists(csv_file):
        print("Ground truth file not found.")
        return pd.DataFrame()
    return pd.read_csv(csv_file)

def compare_data(sim_df, gt_df):
    """Compares Simulation vs Ground Truth."""
    if sim_df.empty or gt_df.empty:
        print("Insufficient data for comparison.")
        return

    # Convert SUMO seconds to HH:MM:SS string to match Ground Truth
    sim_df['TIME_OF_DAY'] = sim_df['interval_begin'].apply(seconds_to_time_str)

    # Rename GT columns for merging
    gt_df = gt_df.rename(columns={
        'EQUIPMENTID': 'sensor_id',
        'TOTAL_VOLUME': 'gt_volume',
        'AVG_SPEED_ARITHMETIC': 'gt_speed'
    })
    
    # Ensure TIME_OF_DAY in GT is string
    gt_df['TIME_OF_DAY'] = gt_df['TIME_OF_DAY'].astype(str)

    print(f"\nMerging data...")
    print(f"Sample SIM times: {sim_df['TIME_OF_DAY'].unique()[:5]}")
    print(f"Sample GT times:  {gt_df['TIME_OF_DAY'].unique()[:5]}")

    # Merge on Sensor ID and Time
    merged = pd.merge(sim_df, gt_df, on=['sensor_id', 'TIME_OF_DAY'], how='inner')
    
    if merged.empty:
        print("\nNo matching intervals found.")
        print("Debug Tips:")
        print("1. Check if 'sensor_id' format matches (e.g. '121725_C').")
        print("2. Check if timestamps overlap.")
        return

    # --- CALCULATE ERROR ---
    # RMSE for Volume
    merged['vol_diff'] = merged['sim_volume'] - merged['gt_volume']
    rmse_vol = np.sqrt((merged['vol_diff'] ** 2).mean())
    mae_vol = merged['vol_diff'].abs().mean()
    
    # RMSE for Speed (only where volume > 0 to avoid noise)
    valid_speed_comp = merged[merged['gt_volume'] > 0]
    merged['speed_diff'] = valid_speed_comp['sim_speed'] - valid_speed_comp['gt_speed']
    rmse_speed = np.sqrt((merged['speed_diff'] ** 2).mean())

    print("\n" + "="*40)
    print(f"Comparison Results (Matched {len(merged)} intervals)")
    print("="*40)
    print(f"Volume RMSE: {rmse_vol:.2f} vehicles")
    print(f"Volume MAE:  {mae_vol:.2f} vehicles")
    print(f"Speed RMSE:  {rmse_speed:.2f} km/h")
    
    # Show sample rows
    print("\nSample Discrepancies (Top 5 by Volume Diff):")
    merged['abs_vol_diff'] = merged['vol_diff'].abs()
    print(merged.sort_values('abs_vol_diff', ascending=False)[
        ['sensor_id', 'TIME_OF_DAY', 'sim_volume', 'gt_volume', 'vol_diff']
    ].head(5))

if __name__ == "__main__":
    # 1. Run Simulation
    run_simulation()
    
    # 2. Process Outputs
    print("\nProcessing output files...")
    sim_data = parse_sumo_outputs(DETECTOR_OUTPUT_FILE)
    
    # 3. Load Ground Truth
    gt_data = load_ground_truth(GROUND_TRUTH_FILE)
    
    # 4. Compare
    compare_data(sim_data, gt_data)