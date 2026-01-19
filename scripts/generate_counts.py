import pandas as pd
import xml.etree.ElementTree as ET

import config

# --- CONFIGURATION ---
xml_file = config.DETECTORS_FILE
csv_file = config.GROUND_TRUTH_FILE
output_file = config.COUNTS_FILE

# 1. LOAD MAP: Sensor ID -> Edge ID
# We assume XML IDs are like "121725_C_0" and CSV IDs are "121725_C"
# We map the CSV ID prefix to the SUMO Edge ID (lane ID without _0)
sensor_to_edge = {}
tree = ET.parse(xml_file)
root = tree.getroot()

for det in root.findall('inductionLoop'):
    full_id = det.get('id')         # e.g., "121725_C_0"
    lane_id = det.get('lane')       # e.g., "1181571_0"
    
    # Strip the lane suffix to get the pure ID keys
    # CSV ID is likely the part before the last underscore if the XML adds index
    # But based on your files: CSV="121725_C", XML="121725_C_0"
    csv_key = full_id.rsplit('_', 1)[0] 
    
    # Edge ID is the lane ID without the lane index
    edge_key = lane_id.rsplit('_', 1)[0]
    
    sensor_to_edge[csv_key] = edge_key

print(f"Mapped {len(sensor_to_edge)} sensors to edges.")

# 2. AGGREGATE CSV: Sum volumes for the whole day (Daily Total)
df = pd.read_csv(csv_file)

# Group by EQUIPMENTID and sum the TOTAL_VOLUME
# We drop TIME_OF_DAY to get a single 24h scalar per sensor
daily_counts = df.groupby('EQUIPMENTID')['TOTAL_VOLUME'].sum().reset_index()

# 3. MERGE & FORMAT
# Map the sensor volumes to edges.
# Note: Multiple sensors might map to the same edge. We sum them.
edge_volumes = {}

for index, row in daily_counts.iterrows():
    sensor_id = row['EQUIPMENTID']
    volume = row['TOTAL_VOLUME']
    
    if sensor_id in sensor_to_edge:
        edge_id = sensor_to_edge[sensor_id]
        if edge_id not in edge_volumes:
            edge_volumes[edge_id] = 0
        edge_volumes[edge_id] += volume
    else:
        # Optional: Print warning if CSV has sensors not in XML
        pass

# 4. WRITE XML
# We create a single interval spanning 24 hours (86400 seconds)
with open(output_file, 'w') as f:
    f.write('<data>\n')
    f.write('    <interval id="daily_total" begin="0" end="86400">\n')
    
    for edge, vol in edge_volumes.items():
        # Round to integer for SUMO tools
        f.write(f'        <edge id="{edge}" entered="{int(vol)}"/>\n')
        
    f.write('    </interval>\n')
    f.write('</data>')

print(f"Successfully generated {output_file} with {len(edge_volumes)} edges.")