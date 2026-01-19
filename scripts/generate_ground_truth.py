import pandas as pd
import os
import holidays

import config

# Use the config directory to construct file paths
# The files are now in data/sensor_data/raw_years
FILE_LIST = [
    str(config.SENSOR_YEARS_DIR / "2013AEDL.csv"),
    str(config.SENSOR_YEARS_DIR / "1S2014AEDL.csv"), 
    str(config.SENSOR_YEARS_DIR / "2S2014AEDL.csv"), 
    str(config.SENSOR_YEARS_DIR / "1P2015AEDL.csv"), 
    str(config.SENSOR_YEARS_DIR / "2P2015AEDL.csv")
]

OUTPUT_FILE = config.GROUND_TRUTH_FILE
SENSORS_FILE = config.SENSORS_LOCATION_FILE

# IDs to strictly exclude, even if they are in the sensors file
EXCLUDED_SENSORS = {121744, 121752}

NUMERIC_COLS = [
    "TOTAL_VOLUME", "AVG_SPEED_ARITHMETIC", "AVG_SPEED_HARMONIC",
    "AVG_LENGTH", "AVG_SPACING", "OCCUPANCY"
]

def get_portuguese_holidays(years):
    pt_holidays = holidays.country_holidays("PT", years=years)
    return pt_holidays

def clean_and_filter_chunk(df, holiday_dict, valid_sensors):
    # 1. Filter by Valid Sensors
    if 'EQUIPMENTID' in df.columns:
        # FIX 1: Added .copy() here to prevent SettingWithCopyWarning
        df = df[df['EQUIPMENTID'].isin(valid_sensors)].copy()
    
    # Ensure LANE_BUNDLE_DIRECTION is present and clean
    if 'LANE_BUNDLE_DIRECTION' in df.columns:
        df['LANE_BUNDLE_DIRECTION'] = df['LANE_BUNDLE_DIRECTION'].astype(str).str.strip()
    
    df['AGG_PERIOD_START'] = pd.to_datetime(df['AGG_PERIOD_START'], errors='coerce')

    # Drop rows where data parsing failed
    df = df.dropna(subset='AGG_PERIOD_START')

    df['date_only'] = df['AGG_PERIOD_START'].dt.date
    df['day_of_week'] = df['AGG_PERIOD_START'].dt.dayofweek

    # Keep only work days (Mon-Fri are 0-4)
    df = df[df['day_of_week'] < 5]

    # Remove holidays
    df = df[~df['date_only'].isin(holiday_dict)]

    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

def main():
    print("--- Starting Traffic Data Processing ---")

    # Load valid sensors
    if os.path.exists(SENSORS_FILE):
        print(f"Loading sensor list from {SENSORS_FILE}...")
        try:
            sensors_df = pd.read_csv(SENSORS_FILE, encoding='latin-1')
            valid_sensors = set(sensors_df['EQUIPMENTID'].unique())
            
            # FIX 2: Explicitly remove the unwanted sensors
            # We discard both int and string versions to be safe against data type differences
            initial_count = len(valid_sensors)
            for sensor_id in EXCLUDED_SENSORS:
                valid_sensors.discard(sensor_id)       # Discard if integer
                valid_sensors.discard(str(sensor_id))  # Discard if string
            
            print(f"Loaded {initial_count} sensors. Removed {initial_count - len(valid_sensors)} excluded IDs.")
            print(f"Final valid sensor count: {len(valid_sensors)}")
            
        except Exception as e:
            print(f"Error reading {SENSORS_FILE}: {e}")
            return
    else:
        print(f"Error: {SENSORS_FILE} not found. Cannot filter by location.")
        return

    pt_holidays = get_portuguese_holidays([2013, 2014, 2015])
    print(f"Loaded {len(pt_holidays)} Portuguese holidays.")

    processed_frames = []

    for file_name in FILE_LIST:
        if not os.path.exists(file_name):
            print(f"Warning: File {file_name} not found. Skipping.")
            continue

        print(f"Processing {file_name}...")

        try:
            df = pd.read_csv(file_name, low_memory=False, encoding='latin-1')

            df_workdays = clean_and_filter_chunk(df, pt_holidays, valid_sensors)

            required_cols = ['EQUIPMENTID', 'LANE_BUNDLE_DIRECTION', 'AGG_PERIOD_START']
            
            cols_to_keep = [c for c in required_cols if c in df_workdays.columns] + \
                           [c for c in NUMERIC_COLS if c in df_workdays.columns]
            
            subset = df_workdays[cols_to_keep].copy()

            subset['TIME_OF_DAY'] = subset['AGG_PERIOD_START'].dt.time

            processed_frames.append(subset)
            print(f"  -> Kept {len(subset)} rows (Workdays & Valid Sensors only).")

        except Exception as e:
            print(f"  -> Error processing {file_name}: {e}")

    print("Combining dataframes...")
    if not processed_frames:
        print("No data processed. Exiting")
        return
    
    master_df = pd.concat(processed_frames, ignore_index=True)

    print("Calculating representative day aggregates...")

    # Group by EQUIPMENTID, LANE_BUNDLE_DIRECTION, and TIME_OF_DAY
    group_cols = ['EQUIPMENTID', 'LANE_BUNDLE_DIRECTION', 'TIME_OF_DAY']
    representative_day = master_df.groupby(group_cols)[NUMERIC_COLS].mean().reset_index()

    print("Applying direction suffixes to Sensor IDs...")
    representative_day['EQUIPMENTID'] = (
        representative_day['EQUIPMENTID'].astype(str) + "_" + representative_day['LANE_BUNDLE_DIRECTION']
    )

    representative_day.drop(columns=['LANE_BUNDLE_DIRECTION'], inplace=True)

    representative_day.to_csv(OUTPUT_FILE, index=False)
    print(f"--- Success! Representative day saved to {OUTPUT_FILE} ---")
    print(f"Total rows in output: {len(representative_day)}")

if __name__ == "__main__":
    main()