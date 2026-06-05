
import os 
from collections import defaultdict
import re 
from pathlib import Path


def check_inchikey_in_file(filepath, pdb_complex):
    """
    Matches the specific ligand ID (e.g., out_lig1) and returns the last key.
    """
    found_lines = []
    last_keys = []
    
    # strip suffix to get base ID (avoids partial matches, e.g. lig1 matching lig10)
    if '_complex' in pdb_complex:
        target_id = pdb_complex.split('_complex')[0]
    else:
        # Fallback if the naming convention varies slightly
        target_id = pdb_complex.split('.pdb')[0]

    try:
        with open(filepath, 'r') as f:
            header = f.readline() 
            
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                parts = line.split(',')
                if len(parts) >= 1 and target_id in parts[0]:
                    found_lines.append(line)
                    last_keys.append(parts[-1])

    except FileNotFoundError:
        print(f"Error: {filepath} not found.")
        return False, [], None

    # Return True/False, the full lines, and the specific keys found
    return len(found_lines) > 0, found_lines, last_keys



#Parses IFP output file and returns interactions as a dict
def parse_single_ifp_file(file_path):
    path = Path(file_path)
    
    data = {
        "file_name": path.name,
        "num_interactions": 0,
        "interactions": []
    }

    # Regex to split: (ResType)(ResNum).(Chain).(InteractionType)
    # Example: TYR202.B.PiStacking -> ('TYR', '202', 'B', 'PiStacking')
    tag_regex = re.compile(r"([A-Z]+)(\d+)\.([A-Z0-9]+)\.(.+)")

    try:
        with open(path, "r") as f:
            lines = [line.strip() for line in f.readlines()]

        # find num_interactions for checksum
        for i, line in enumerate(lines):
            if "<num_interactions>" in line:
                data["num_interactions"] = int(lines[i+1])
                break

        # collect active interactions (value == 1)
        for i, line in enumerate(lines):
            # skip score metadata lines
            if line.startswith(">") and "m_score__max__" not in line:
                
                # Extract the tag name between the brackets < >
                tag_match = re.search(r"<(.*?)>", line)
                if not tag_match:
                    continue
                
                tag_name = tag_match.group(1)
                
                # Check if the value on the NEXT line is "1"
                if i + 1 < len(lines) and lines[i+1] == "1":
                    
                    # Parse the tag_name into components
                    parts = tag_regex.match(tag_name)
                    if parts:
                        res_type, res_num, chain_id, int_type = parts.groups()
                        
                        data["interactions"].append({
                            "residue_number": int(res_num),
                            "residue_type": res_type,
                            "chain_id": chain_id,
                            "interaction_type": int_type
                        })

        # Final Checksum Verification
        if len(data["interactions"]) != data["num_interactions"]:
            print(f"Warning: Checksum mismatch in {path.name}. "
                  f"Expected {data['num_interactions']}, found {len(data['interactions'])}.")

        return data

    except Exception as e:
        print(f"Error parsing {path.name}: {e}")
        return None


#Converts a complex-name to its corresponding ifp-output name
def convert_complex_name_to_ifp_name(complex_filename):
    """
    Converts 'ABGXADJDTPFFSZ-NMEKOBJZNA-O_out_lig1_complex.pdb' 
    to 'ABGXADJDTPFFSZ-NMEKOBJZNA-O_lig1_ifp.sdf'
    """
    # Regex breakdown:
    # ^(.+?)_out   -> Capture everything from start up to '_out' (non-greedy)
    # .*?_lig(\d+) -> Find '_lig' followed by the number
    pattern = re.compile(r"^(.+?)_out.*?_lig(\d+)")
    
    match = pattern.search(complex_filename)
    
    if match:
        hash_str = match.group(1)
        lig_num = match.group(2)
        return f"{hash_str}_lig{lig_num}_ifp.sdf"
    
    return None






