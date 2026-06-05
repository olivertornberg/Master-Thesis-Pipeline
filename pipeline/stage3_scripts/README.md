# Stage 3 — QM/MM Descriptor Calculation

## Purpose
Computes quantum-mechanical descriptors for each ligand inside its relaxed protein–ligand
complex using a QM/MM (Electrostatic Embedding) framework via VeloxChem.

For each complex the pipeline:
1. Builds a NPE embedding from the surrounding protein environment (Amber ff99SB point charges)
2. Runs a B3LYP/def2-SVPD SCF calculation on the ligand in the embedding field
3. Computes per-atom and molecular QM descriptors from the converged wavefunction
4. Records pairwise interaction energies with every IFP-detected interacting residue
5. Saves all results to a per-complex CSV in `final_pipeline/results/`

---

## Two engine variants

| File | Use when |
|---|---|
| `pipeline_engine.py` | Enzyme has a **cofactor** (e.g. SAH) in the active site that must be part of the MM region |
| `pipeline_engine_nocofactors.py` | No cofactor present — protein atoms are the only MM embedding |

`pipeline_main.py` imports `pipeline_engine` by default. To switch, change the import at the
top of `pipeline_main.py`:
```python
import pipeline_engine as engine          # with cofactor (default)
# import pipeline_engine_nocofactors as engine  # without cofactor
```

### Cofactor RESP charges (engine with cofactor only)

When the cofactor version is used, the MM region includes the cofactor (SAH) represented as
point charges. These charges are **pre-calculated RESP charges** computed with VeloxChem's
`RespChargesDriver` at the HF/6-31G* level — the same level of theory used by AMBER for
residue charge derivation (consistent with ff99SB). The resulting charges are hardcoded
in `create_cofactor_pc_array()` inside `pipeline_engine.py`.

If you are adapting this pipeline for a different cofactor, you must first calculate RESP
charges for it using VeloxChem:
```python
import veloxchem as vlx
mol = vlx.Molecule.read_pdb_file("cofactor.pdb")
basis = vlx.MolecularBasis.read(mol, "6-31G*")
resp_drv = vlx.RespChargesDriver()
charges = resp_drv.compute(mol, basis)
```
Then replace the hardcoded values in `create_cofactor_pc_array()`.

---

## Prerequisites

### Environment
This stage requires **VeloxChem** and runs in the `vlx-master` branch. Depending on the current VeloxChem release, it might be necessary to clone and compile a pre-release VeloxChem version from the github: 
https://github.com/VeloxChem/VeloxChem


Install dependencies via the provided `pyproject.toml`

### Inputs (from Stage 2 outputs)
All inputs are read from `final_pipeline/processed_files/`:

| Path | Contents |
|---|---|
| `final_complexes/` | Merged protein+cofactor+ligand `.pdb` complexes |
| `IFP_output/` | Per-pose IFP `.sdf` files from Stage 2 |
| `ligand_processed/ligand_charges.csv` | Formal charge metadata from Stage 1 |

---

## Running

From the `stage3_scripts/` directory:
```bash
conda activate vlx-master

# Single complex (for testing):
python pipeline_main.py --complex <filename>_complex.pdb

# Full batch (restartable — skips already-completed complexes):
python pipeline_run_batch_wrapper.py
```

All scripts resolve paths relative to their location automatically — no path editing needed.

---

## Outputs

Results are written to `final_pipeline/results/`, one CSV per complex:

| File | Contents |
|---|---|
| `<ID>_lig<N>_results.csv` | Ligand descriptors + per-residue interaction energies |

### Descriptors computed per ligand

| Descriptor | Description |
|---|---|
| `RESP_charges` | Per-atom RESP charges (HF/6-31G*) |
| `IE` | Per-atom ionisation energies (Koopmans' theorem) |
| `EA` | Per-atom electron affinities |
| `IE_Surface_Avg` | Surface-weighted average ionisation energy |
| `EA_Surface_Avg` | Surface-weighted average electron affinity |
| `SASA` | Solvent-accessible surface area (Å²) |
| `Log_P` | Calculated lipophilicity |

---

## Configuration

Radial cutoffs for the PE/NPE embedding are set in `pipeline_main.py`:
```python
npe_cutoff = 50   # non-polarisable MM shell radius (Å)
pe_cutoff  = 0    # polarisable MM shell radius (Å)
```
