# Stage 1 — PDB Structure Processing

## Purpose
Prepares raw receptor and ligand structures for the QM/MM pipeline.
Starting from multi-model PDB files and docked ligand SDF files, this stage produces
clean, paired protein–ligand–cofactor complexes ready for VeloxChem/GROMACS calculations. This stage is highly dependent on the format of the structural input and may need to be reworked based on user needs. It is kept as a notebook as it is recommended to visually inspect files after each operation the first its used.  

---

## Processing steps (in order)

| Step | Description |
|---|---|
| 1. Split receptor | Splits multi-MODEL PDB into one file per pose |
| 2. Fix LIG label | Standardises residue names (UNL → LIG), corrects element columns |
| 3. Sort atom blocks | Uses MDAnalysis to re-order atoms into contiguous residue blocks |
| 4. Slice protein-only | Removes LIG and SAH from receptor; re-indexes atom serials |
| *(GROMACS)* | **External step** — hydrogen addition + energy minimisation (see below) |
| 5. Fix protonation names | Renames ASP/GLU/HIS/SER to protonation-state-specific residue names (ASH, GLH, HIP, etc.) |
| 6. Fix cofactor | Cleans the SAH cofactor PDB extracted from the reference structure |
| 7. Ligand SDF → PDB | Converts docked SDF poses to PDB using InChIKey matching to verify ligand identity |
| 8. Ligand cleanup | Standardises ligand chain/residue names; extracts formal charge metadata |
| 9. Merge complexes | Joins protein + SAH + ligand into a single complex PDB per pose |
| 10. Batch split | Divides complexes into N equal subsets for parallel HPC submission |
| 11. SDF split + verify | Splits multi-pose SDFs into per-pose SDF files with InChIKey verification |

---

## Prerequisites

### Python environment
Install the Python dependencies using the provided `pyproject.toml`:
```bash
conda create -n gromacs-env python=3.10
conda activate gromacs-env
conda install -c conda-forge gromacs=2026.0 mdanalysis rdkit
pip install biopython numpy
```

Or install exactly from the full `conda list` supplied with this project.

### GROMACS
GROMACS (`gmx`) is required between steps 4 and 5 for hydrogen addition and energy minimisation.


### Input files
Place your dataset folder inside `final_pipeline/`. The folder name is set by the `_dataset`
variable in Cell 2 of the notebook — change that one line to switch between datasets:

```python
_dataset = 'raw_files'          # e.g. default WT
# _dataset = 'N249G_raw_files'  # mutant example
```

The notebook expects this structure inside your dataset folder:

| Path (relative to `final_pipeline/<_dataset>/`) | Contents |
|---|---|
| `raw_files/receptor/` | Multi-MODEL receptor `.pdb` files |
| `raw_files/ligands/` | Multi-pose docked ligand `.sdf` files |
| `raw_files/NNMT_SAH/` | SAH cofactor reference `.pdb` |

---

## Running

Open `pdb_processor.ipynb` and run cells top-to-bottom.
All outputs are written automatically to `final_pipeline/processed_files/` — no paths need editing.

After step 4, pause and run the GROMACS notebook (`pdb_processor_gromacs.ipynb`) to add
hydrogens and minimise structures, then continue from step 5.

---

## Outputs

Everything is written under `final_pipeline/processed_files/`:

| Folder | Contents |
|---|---|
| `receptor_processed/receptor_0_split/` | Split single-pose receptor PDBs |
| `receptor_processed/receptor_1_label/` | Residue-labelled PDBs |
| `receptor_processed/receptor_2_atomblocks/` | Atom-block-sorted PDBs |
| `receptor_processed/receptor_3_onlyprotein/` | Protein-only PDBs (LIG/SAH removed) |
| `receptor_processed/receptor_4_relaxed_reshifted/` | *(Written by GROMACS stage)* |
| `receptor_processed/receptor_5_resiude_names/` | Protonation-corrected PDBs |
| `cofactor_processed/` | Cleaned SAH cofactor PDB |
| `ligand_processed/ligands_0_pdb/` | Per-pose ligand PDBs |
| `ligand_processed/ligands_1_cleanup_n_charge/` | Cleaned ligand PDBs |
| `ligand_processed/ligands_split_sdf/` | Per-pose ligand SDFs (for IFP stage) |
| `ligand_processed/ligand_charges.csv` | Formal charge metadata per atom |
| `final_complexes/` | Merged protein + SAH + ligand complexes |

