import xml.etree.ElementTree as ET

import config

taz_file = config.TAZ_FILE
output_trips_file = config.TAZ_DUMMY_TRIPS_FILE

tree = ET.parse(taz_file)
root = tree.getroot()

tazs = []
for taz in root.findall('.//taz'):
    tazs.append(taz.get('id'))

print(f"Found {len(tazs)} TAZs: {tazs}")

with open(output_trips_file, 'w') as f:
    f.write('<routes>\n')
    
    f.write('    <vType id="taz_vehicle" vClass="passenger"/>\n')
    
    count = 0
    for origin in tazs:
        for dest in tazs:
            if origin == dest:
                continue 
            
            f.write(f'    <trip id="t_{origin}_to_{dest}" type="taz_vehicle" '
                    f'fromTaz="{origin}" toTaz="{dest}" depart="0" />\n')
            count += 1
            
    f.write('</routes>')

print(f"Generated {count} dummy trips in {output_trips_file}")