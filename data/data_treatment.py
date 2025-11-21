import pandas as pd
import glob
import os
import holidays
from datetime import time

FILE_LIST = [
    "2013AEDL.csv",
    "1S2014AEDL.csv", 
    "2S2014AEDL.csv", 
    "1P2015AEDL.csv", 
    "2P2015AEDL.csv"
]

OUTPUT_FILE = "Representative_Workday_Profile.csv"

NUMERIC_COLS = [
    "TOTAL_VOLUME", "AVG_SPEED_ARITHMETIC", "AVG_SPEED_HARMONIC",
    "AVG_LENGTH", "AVG_SPACING", "OCCUPANCY", "LIGHT_VEHICLE_RATE"
]


def get_portuguese_holidays(years):
    pt_holidays = holidays.country_holidays("PT", years=years)
    return pt_holidays

def clean_and_filter_chunk(df, holiday_dict):
    df['AGG_PERIOD_START'] = pd.to_datetime(df['AGG_PERIOD_START'], errors='coerce')

    # Drop rows where data parsing failed
    df = df.dropna(subset='AGG_PERIOD_START')

    df['date_only'] = df['AGG_PERIOD_START'].dt.date
    df['day_of_week'] = df['AGG_PERIOD_START'].dt.dayofweek

    # Keep only work days (mon-fri)
    df = df[df['day_of_week'] < 5]

    # Remove holidays

    df = df[~df['date_only'].isin(holiday_dict)]

    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

def main():
    print("--- Starting Traffic Data Processing ---")

    pt_holidays = get_portuguese_holidays([2013, 2014, 2015])
    print(f"Loaded {len(pt_holidays)} Portuguese holidays.")

    processed_frames = []

    for file_name in FILE_LIST:
        if not os.path.exists(file_name):
            print(f"Warning: File {file_name} not found. Skipping.")
            continue

        print(f"Processing {file_name}...")

        try:
            df = pd.read_csv(file_name, low_memory=False)

            df_workdays = clean_and_filter_chunk(df, pt_holidays)

            cols_to_keep = ['EQUIPMENTID', 'AGG_PERIOD_START'] + \
                           [c for c in NUMERIC_COLS if c in df_workdays.columns]
            
            subset = df_workdays[cols_to_keep].copy()

            subset['TIME_OF_DAY'] = subset['AGG_PERIOD_START'].dt.time

            processed_frames.append(subset)
            print(f"  -> Kept {len(subset)} rows (Workdays only).")

        except Exception as e:
            print(f"  -> Error processing {file_name}: {e}")

    print("Combining dataframes...")
    if not processed_frames:
        print("No data processed. Exiting")
        return
    
    master_df = pd.concat(processed_frames, ignore_index=True)

    print("Calculating representative day aggregates...")

    representative_day = master_df.groupby(['EQUIPMENTID', 'TIME_OF_DAY'])[NUMERIC_COLS].mean().reset_index()

    representative_day.to_csv(OUTPUT_FILE, index=False)
    print(f"--- Success! Representative day saved to {OUTPUT_FILE} ---")
    print(f"Total rows in output: {len(representative_day)}")

if __name__ == "__main__":
    main()