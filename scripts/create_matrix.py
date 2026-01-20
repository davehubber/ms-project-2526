import xml.etree.ElementTree as ET
import collections

import config

taz_file = config.TAZ_FILE
trips_file = config.DAILY_TRIPS_FILE
output_file = config.BASELINE_MATRIX_FILE

print("--- STEP 1: INDEXING TAZs ---")
taz_ids = []
tree = ET.parse(taz_file)
for taz in tree.findall('.//taz'):
    tid = taz.get('id')
    if tid:
        taz_ids.append(tid)

try:
    taz_ids.sort(key=int)
except ValueError:
    taz_ids.sort()

print(f"Found {len(taz_ids)} TAZs: {taz_ids}")

edge_to_taz = {}
for taz in tree.findall('.//taz'):
    tid = taz.get('id')
    for src in taz.findall('tazSource'):
        edge_to_taz[src.get('id')] = tid
    for sink in taz.findall('tazSink'):
        edge_to_taz[sink.get('id')] = tid

print("--- STEP 2: PARSING TRIPS ---")
daily_counts = collections.defaultdict(int)

context = ET.iterparse(trips_file, events=('end',))
for event, elem in context:
    if elem.tag == 'vehicle':
        route_elem = elem.find('route')
        if route_elem is not None:
            edges = route_elem.get('edges').split()
            if not edges: 
                continue
                
            first_edge = edges[0]
            last_edge = edges[-1]
            
            origin = edge_to_taz.get(first_edge)
            dest = edge_to_taz.get(last_edge)
            
            if origin and dest:
                daily_counts[(origin, dest)] += 1
        
        elem.clear()

print("--- STEP 3: GENERATING MATRIX ---")
with open(output_file, 'w') as f:
    f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    f.write('<data xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/datamode_file.xsd">\n')
    
    f.write('    <interval id="baseline_5min" begin="0" end="300">\n')
    
    count_pairs = 0
    total_volume = 0
    
    for o in taz_ids:
        for d in taz_ids:
            
            daily_vol = daily_counts.get((o, d), 0)
            
            val_float = daily_vol / 288.0
            
            val_int = int(round(val_float))
            
            final_demand = max(1, val_int)
            
            f.write(f'        <tazRelation from="{o}" to="{d}" count="{final_demand}"/>\n')
            
            count_pairs += 1
            total_volume += final_demand

    f.write('    </interval>\n')
    f.write('</data>')

print(f"--- DONE ---")
print(f"Generated {output_file}")
print(f"Total OD Pairs: {count_pairs}")
print(f"Total 5-min Volume: {total_volume}")