import os

def overwrite_matrix_with_correct_order():
    
    reference_file = 'od-matrixes/matrix.1.0.2.0.1.xml'
    target_file = 'od-matrixes/matrix.7.0.8.0.1.xml'
    
    # 1. Read Reference Structure
    template_order = []
    with open(reference_file, 'r') as f:
        for line in f:
            if line.startswith('*') or line.startswith('$') or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 3:
                template_order.append((parts[0], parts[1]))

    # 2. Read Target File Data
    target_data = {}
    header_lines = []
    
    try:
        with open(target_file, 'r') as f:
            for line in f:
                if line.startswith('*') or line.startswith('$') or not line.strip():
                    header_lines.append(line)
                    continue
                parts = line.split()
                if len(parts) >= 3:
                    target_data[(parts[0], parts[1])] = parts[2]
    except FileNotFoundError:
        print(f"ERROR: Could not find {target_file}")
        return

    # 3. OVERWRITE the Target File
    print(f">> Overwriting file in-place: {target_file}...")
    
    used_pairs = set()
    written_count = 0
    filled_zeros = 0

    with open(target_file, 'w') as f:
        # Write headers
        for line in header_lines:
            f.write(line)
        
        # Write data following the template order
        for (org, dest) in template_order:
            pair = (org, dest)
            
            if pair in target_data and pair not in used_pairs:
                count = target_data[pair]
                used_pairs.add(pair)
            else:
                count = "0"
                if pair not in target_data:
                    filled_zeros += 1

            # Write formatted line
            f.write(f"\t\t{org}\t\t{dest}\t\t{count}\n")
            written_count += 1

    print(f"\n>> DONE! '{target_file}' has been replaced.")
    print(f"   Total rows: {written_count}")
    print(f"   Missing data filled with 0: {filled_zeros}")

if __name__ == "__main__":
    overwrite_matrix_with_correct_order()