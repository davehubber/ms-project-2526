import xml.etree.ElementTree as ET

import config

# --- CONFIGURATION ---
taz_file = config.TAZ_FILE
output_trips_file = config.TAZ_DUMMY_TRIPS_FILE

# 1. Parse TAZ IDs
tree = ET.parse(taz_file)
root = tree.getroot()

# Find all 'taz' elements (handling the 'tazs' wrapper if present)
# Your file structure: <additional><tazs><taz ...>
tazs = []
for taz in root.findall('.//taz'):
    tazs.append(taz.get('id'))

print(f"Found {len(tazs)} TAZs: {tazs}")

# 2. Generate All-Pairs Trips
# We create a trip from every TAZ to every other TAZ
with open(output_trips_file, 'w') as f:
    f.write('<routes>\n')
    
    # Define a generic vehicle type
    f.write('    <vType id="taz_vehicle" vClass="passenger"/>\n')
    
    count = 0
    for origin in tazs:
        for dest in tazs:
            if origin == dest:
                continue # Skip intra-zonal trips if desired
            
            # We use the 'fromTaz' and 'toTaz' attributes which duarouter understands
            f.write(f'    <trip id="t_{origin}_to_{dest}" type="taz_vehicle" '
                    f'fromTaz="{origin}" toTaz="{dest}" depart="0" />\n')
            count += 1
            
    f.write('</routes>')

print(f"Generated {count} dummy trips in {output_trips_file}")