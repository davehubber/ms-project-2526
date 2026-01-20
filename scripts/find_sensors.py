import os
import sys
import csv

import config

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

import sumolib

NET_FILE = config.NET_FILE
net = sumolib.net.readNet(NET_FILE)

input_csv = config.SENSORS_LOCATION_FILE
radius = 10

print(f"{'ID':<10} | {'Mapped Edge':<15} | {'Lane':<15} | {'Pos':<10}")
print("-" * 60)

with open(input_csv, mode='r', encoding='latin1') as csvfile:
    reader = csv.DictReader(csvfile)
    
    for row in reader:
        sensor_id = row['EQUIPMENTID']
        
        try:
            lat_str = row['latitude'].replace(',', '.')
            lon_str = row['longitude'].replace(',', '.')
            
            lat = float(lat_str)
            lon = float(lon_str)
        except ValueError:
            print(f"Skipping ID {sensor_id}: Invalid coordinates")
            continue

        x, y = net.convertLonLat2XY(lon, lat)

        nearby_lanes = net.getNeighboringLanes(x, y, radius)

        if not nearby_lanes:
            print(f"{sensor_id:<10} | No lane found within {radius}m")
        else:
            best_lane_obj, dist = nearby_lanes[0]
            
            lane_pos = best_lane_obj.getClosestLanePosAndDist((x, y))[0]
            
            print(f"{sensor_id:<10} | {best_lane_obj.getEdge().getID():<15} | {best_lane_obj.getID():<15} | {lane_pos:.2f}")