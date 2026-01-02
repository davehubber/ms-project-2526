import os
import sys
import xml.etree.ElementTree as ET

# --- SUMO CONFIG ---
try:
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        os.environ["SUMO_HOME"] = "/usr/share/sumo" 
        sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
    
    import sumolib
except ImportError:
    sys.exit("ERROR: Could not import 'sumolib'. Check SUMO_HOME.")


NET_FILE = "net.net.xml"
INPUT_TAZ = "taz.xml"
OUTPUT_TAZ = "taz_clean.xml"

def clean_taz_file():
    print(f">> Loading road network '{NET_FILE}'... (this may take a few seconds)")
    try:
        net = sumolib.net.readNet(NET_FILE)
    except Exception as e:
        sys.exit(f"CRITICAL ERROR reading the map: {e}")
        
    print(">> Network loaded. Starting cleanup...")

    try:
        tree = ET.parse(INPUT_TAZ)
        root = tree.getroot()
    except Exception as e:
        sys.exit(f"ERROR reading XML taz.xml: {e}")
    
    new_root = ET.Element("additional")
    
    cleaned_count = 0
    removed_edges = 0
    
    all_tazs = root.findall('.//taz')
    
    print(f">> Found {len(all_tazs)} zones (TAZs) in the file. Processing...")
    
    if len(all_tazs) == 0:
        print("[WARNING] Still seeing 0 zones. Check if the taz.xml file has <taz> tags!")
    
    for taz in all_tazs:
        taz_id = taz.get('id')
        
        # Sets to avoid duplicates
        sources = set()
        sinks = set()
        
        for child in taz:
            edge_id = child.get('id')
            weight = child.get('weight')
            
            # If no ID (malformed XML), skip
            if not edge_id: continue
            
            # Check if the edge exists in the network
            if not net.hasEdge(edge_id):
                # print(f"   [Clean] Non-existent edge {edge_id} removed.")
                removed_edges += 1
                continue
            
            edge_obj = net.getEdge(edge_id)
            
            # Connectivity Logic
            has_incoming = len(edge_obj.getIncoming()) > 0
            has_outgoing = len(edge_obj.getOutgoing()) > 0
            
            if child.tag == 'tazSource':
                # Only a valid Source if it has outgoing edges
                if has_outgoing:
                    sources.add((edge_id, weight))
                else:
                    removed_edges += 1

            elif child.tag == 'tazSink':
                # Only a valid Sink if it has incoming edges
                if has_incoming:
                    sinks.add((edge_id, weight))
                else:
                    removed_edges += 1

        # Rebuild the Clean TAZ
        # Note: If a zone ends up empty (no sources or sinks), we keep it empty
        # to avoid "Missing Zone" errors later in od2trips.
        new_taz = ET.SubElement(new_root, "taz", id=taz_id)
        
        for edge_id, weight in sources:
            ET.SubElement(new_taz, "tazSource", id=edge_id, weight=weight)
            
        for edge_id, weight in sinks:
            ET.SubElement(new_taz, "tazSink", id=edge_id, weight=weight)
            
        cleaned_count += 1
        
    tree_out = ET.ElementTree(new_root)
    ET.indent(tree_out, space="\t", level=0)
    tree_out.write(OUTPUT_TAZ, encoding="utf-8", xml_declaration=True)
    
    print(f"\n>> SUCCESS!")
    print(f"   TAZs processed and saved: {cleaned_count}")
    print(f"   Garbage removed (duplicate or invalid edges): {removed_edges}")
    print(f"   Clean file: {OUTPUT_TAZ}")

if __name__ == "__main__":
    if not os.path.exists(NET_FILE):
        print(f"ERROR: Missing file {NET_FILE}")
    elif not os.path.exists(INPUT_TAZ):
        print(f"ERROR: Missing file {INPUT_TAZ}")
    else:
        clean_taz_file()