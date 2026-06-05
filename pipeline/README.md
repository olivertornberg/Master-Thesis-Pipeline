# Final Pipeline

3-stage workflow for structure preparation, interaction fingerprinting, and QM/MM descriptor generation.

## What this repository contains

| Stage | Folder | Purpose |
|---|---|---|
| 1 | `stage1_PDBprocessing/` | Prepare receptor/ligand/cofactor structures and build merged complexes |
| 2 | `stage2_IFP/` | Compute interaction fingerprints (IFP) from ligand poses vs receptors |
| 3 | `stage3_scripts/` | Run VeloxChem QM/MM descriptor and interaction-energy calculations |



## Expected top-level layout

```text
final_pipeline/
  raw_files/                        # raw unprocessed files
  processed_files/                  # created/filled by stages
  results/                          # stage 3 CSV result outputs
  stage1_PDBprocessing/
  stage2_IFP/
  stage3_scripts/
```

## Quick start

1. Set up environments:
- Stage 1: use `stage1_PDBprocessing/pyproject.toml` (plus `gromacs` via conda)
- Stage 2: use `stage2_IFP/pyproject.toml`
- Stage 3: use `stage3_scripts/pyproject.toml` (VeloxChem env)

2. Run Stage 1 notebook (highly dependent on raw file format):
- Open `stage1_PDBprocessing/pdb_processor.ipynb`
- Set `_dataset` in Cell 2 (for example `raw_files` or mutant folder)
- Run top-to-bottom


3. Run Stage 2 IFP:

```bash
cd stage2_IFP
python IFP_wrapper.py
```

4. Run Stage 3 QM/MM:

```bash
cd stage3_scripts
python pipeline_run_batch_wrapper.py
```

## Outputs

- `processed_files/`
Contains all intermediate stage outputs (processed receptors/ligands, final complexes, IFP files).

- `results/`
Contains stage 3 per-complex descriptor CSV files.

## Stage 3 engine choice

Two engine variants are available:
- `pipeline_engine.py`
Use when a cofactor should be included in the MM region.
- `pipeline_engine_nocofactors.py`
Use when no cofactor should be in the MM region.

For the cofactor engine, cofactor RESP point charges must be pre-calculated (recommended with VeloxChem `RespChargesDriver` at HF/6-31G* for consistency with ff99SB-style charge derivation) and inserted into the pc-array within the engine-script.



Associated GitHub:
https://github.com/olivertornberg/Master-Thesis-Pipeline

Questions? 
Feel free to reach out to me at oliver.toernberg@gmail.com 

