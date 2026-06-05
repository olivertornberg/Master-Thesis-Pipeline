import os
import subprocess
import sys 
from pathlib import Path


_pipeline_dir = Path(os.getcwd()).parent
input_root = str(_pipeline_dir / 'processed_files')


def run_pipeline_batch(target_folder, output_folder):

    complex_files = [f for f in os.listdir(target_folder) if f.endswith('_complex.pdb')]
    
    for i, filename in enumerate(complex_files):
        base_name = filename.replace('_out', '').replace('_complex.pdb', '')
        expected_output = f"{base_name}_results.csv"
        
        output_path = os.path.join(output_folder, expected_output)

        
        if os.path.exists(output_path):
            print(f'Results for complex: {base_name} already in results-folder, skipping...')
            continue
                
        try:
            print(f'Processing {base_name}...')
            result = subprocess.run(
                [sys.executable, "pipeline_main.py", "--complex", filename],
                env=os.environ.copy(),
                stdout=sys.stdout,
                stderr=sys.stderr,
                check=True
            )

            
        except subprocess.CalledProcessError as e:
            print(f"!!! Crashed on {filename} !!!")
            print(e.stderr)
            continue 
        print(f'COMPLETED: {base_name}...')


if __name__ == "__main__":
    
    OUTPUT_DIR = os.path.join(_pipeline_dir, 'results')
    TARGET_DIR = os.path.join(input_root, 'final_complexes/') 
    
    run_pipeline_batch(TARGET_DIR, OUTPUT_DIR)