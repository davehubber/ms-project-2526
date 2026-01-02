import os
import sys
import csv

# 1. Import sumolib
# If SUMO_HOME is set, this pulls the tools automatically
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

import sumolib

# 2. Load your Network
# Replace with your actual .net.xml file path
NET_FILE = "net.net.xml" 
net = sumolib.net.readNet(NET_FILE)

# 3. Process the CSV
input_csv = "sensors_location.csv"
radius = 10  # Search radius in meters

print(f"{'ID':<10} | {'Mapped Edge':<15} | {'Lane':<15} | {'Pos':<10}")
print("-" * 60)

with open(input_csv, mode='r', encoding='latin1') as csvfile:
    # Using DictReader to handle headers automatically
    reader = csv.DictReader(csvfile)
    
    for row in reader:
        sensor_id = row['EQUIPMENTID']
        
        # Clean the data: Remove quotes if present and replace comma with dot
        try:
            # Based on, columns are 'latitude' and 'longitude'
            lat_str = row['latitude'].replace(',', '.')
            lon_str = row['longitude'].replace(',', '.')
            
            lat = float(lat_str)
            lon = float(lon_str)
        except ValueError:
            print(f"Skipping ID {sensor_id}: Invalid coordinates")
            continue

        # 4. Convert Lat/Lon to SUMO X/Y
        # sumolib handles the projection defined in your .net.xml automatically
        x, y = net.convertLonLat2XY(lon, lat)

        # 5. Find the closest Lane
        # getNeighboringLanes returns a list of (lane, dist) tuples
        # We take the closest one (index 0)
        nearby_lanes = net.getNeighboringLanes(x, y, radius)

        if not nearby_lanes:
            print(f"{sensor_id:<10} | No lane found within {radius}m")
        else:
            # best_lane_obj is the Lane object, dist is distance to the lane shape
            best_lane_obj, dist = nearby_lanes[0]
            
            # Calculate the exact position (0 to length) along the lane
            # This gives you the 'pos' attribute needed for induction loops
            lane_pos = best_lane_obj.getClosestLanePosAndDist((x, y))[0]
            
            print(f"{sensor_id:<10} | {best_lane_obj.getEdge().getID():<15} | {best_lane_obj.getID():<15} | {lane_pos:.2f}")