import pandas as pd
import xml.etree.ElementTree as ET

import config

xml_file = config.DETECTORS_FILE
csv_file = config.GROUND_TRUTH_FILE
output_file = config.COUNTS_FILE

sensor_to_edge = {}
tree = ET.parse(xml_file)
root = tree.getroot()

for det in root.findall('inductionLoop'):
    full_id = det.get('id')
    lane_id = det.get('lane')
    
    csv_key = full_id.rsplit('_', 1)[0] 
    
    edge_key = lane_id.rsplit('_', 1)[0]
    
    sensor_to_edge[csv_key] = edge_key

print(f"Mapped {len(sensor_to_edge)} sensors to edges.")

df = pd.read_csv(csv_file)

daily_counts = df.groupby('EQUIPMENTID')['TOTAL_VOLUME'].sum().reset_index()

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
        pass

with open(output_file, 'w') as f:
    f.write('<data>\n')
    f.write('    <interval id="daily_total" begin="0" end="86400">\n')
    
    for edge, vol in edge_volumes.items():
        f.write(f'        <edge id="{edge}" entered="{int(vol)}"/>\n')
        
    f.write('    </interval>\n')
    f.write('</data>')

print(f"Successfully generated {output_file} with {len(edge_volumes)} edges.")