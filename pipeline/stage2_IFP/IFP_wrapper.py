import os
import re
import subprocess
from pathlib import Path

_pipeline_dir = Path(os.getcwd()).parent

input_root = str(_pipeline_dir / 'processed_files')



# --- Configuration ---
SDF_DIR = Path(os.path.join(input_root, "ligand_processed/ligands_split_sdf"))
PDB_DIR = Path(os.path.join(input_root, "receptor_processed/receptor_4_relaxed_reshifted"))
OUTPUT_DIR = Path(os.path.join(input_root, "IFP_output"))
YAML_TEMPLATE = Path("IFP_f2f_wrapper.yaml") 

OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
def main():
    sdf_files = list(SDF_DIR.glob("*.sdf"))
    print(f"Found {len(sdf_files)} SDF files.")

    # Regex to match your naming convention
    pattern = re.compile(r"(.+)_out_lig(\d+)\.(pdb|sdf)")

    for i, sdf_file in enumerate(sdf_files, 1):
        match = pattern.match(sdf_file.name)
        
        if match:
            prefix = match.group(1)
            lig_num = match.group(2)
            
            # Construct the matching PDB name
            pdb_name = f"{prefix}_out_lig{lig_num}.pdb"
            pdb_path = PDB_DIR / pdb_name
            output_name = f"{prefix}_lig{lig_num}_ifp.sdf"
            output_path = OUTPUT_DIR / output_name

            if pdb_path.exists():
                print(f"[{i}/{len(sdf_files)}] Processing: {prefix} (Lig {lig_num})")
                
                # maize ifp_template.yaml --input path/to/sdf --receptor path/to/pdb --output path/to/out
                command = [
                    "maize", str(YAML_TEMPLATE),
                    "--input", str(sdf_file),
                    "--receptor", str(pdb_path),
                    "--output", str(output_path)
                ]
                try:
                    subprocess.run(command, check=True, capture_output=True, text=True)
                except subprocess.CalledProcessError as e:
                    print(f"Error on {sdf_file.name}: {e.stderr}")
            else:
                print(f"Skipping {sdf_file.name}: PDB not found.")

if __name__ == "__main__":
    main()