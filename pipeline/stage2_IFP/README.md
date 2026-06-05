# Stage 2 — Interaction Fingerprint (IFP) Calculation

## Purpose
This stage computes **Interaction Fingerprints (IFPs)** for every ligand pose against its
paired receptor structure. An IFP is a binary or count-based description of protein–ligand
contacts (hydrogen bonds, pi-stacking, cation–pi, etc.) encoded per residue. The results
are saved as annotated SDF files and used downstream for ranking and descriptor analysis.

---

## How it works
`IFP_wrapper.py` is a batch driver that loops over all ligand SDF files produced in Stage 1
and calls the `maize` workflow engine for each one:

```
ligand SDF  ──►  LoadLibrary  ──►  IFPidv  ──►  SaveSingleLibrary  ──►  output SDF
receptor PDB ──────────────────────────►
```

The workflow is defined in `IFP_f2f_wrapper.yaml`. Maize handles the actual IFP calculation
via the `IFPidv` node.

Interaction types computed (configurable in the YAML):
| Type | Description |
|---|---|
| `HBAcceptor` | Ligand acts as H-bond acceptor |
| `HBDonor` | Ligand acts as H-bond donor |
| `Cationic` | Cationic (ionic) interaction |
| `PiCation` | Pi–cation interaction |
| `PiStacking` | Aromatic pi–pi stacking |

Default distance cutoff: **4.0 Å**

---

## Prerequisites

### Environment
Install via the provided `pyproject.toml`:
```bash
conda create -n maize python=3.10
conda activate maize
conda install -c conda-forge rdkit openbabel mdanalysis prolif
pip install -e ".[dev]"   # installs maize, maize-contrib, and remaining deps
```

Or recreate the exact environment from the full conda list supplied with this project.

### Inputs (from Stage 1 outputs)
These folders are expected under `final_pipeline/processed_files/`:

| Path | Contents |
|---|---|
| `ligand_processed/ligands_split_sdf/` | Single-pose `.sdf` files, named `<ID>_out_lig<N>.sdf` |
| `receptor_processed/receptor_4_relaxed_reshifted/` | GROMACS-relaxed receptor `.pdb` files, named `<ID>_out_lig<N>.pdb` |

---

## Running

From the `stage2_IFP/` directory:
```bash
conda activate maize
python IFP_wrapper.py
```

The script automatically resolves all paths relative to its location — no configuration needed
as long as Stage 1 has been run and `processed_files/` exists one level up.

---

## Outputs

All output is written to `final_pipeline/processed_files/IFP_output/`:

| File | Description |
|---|---|
| `<ID>_lig<N>_ifp.sdf` | Ligand SDF with IFP metadata tags added per pose |

---

## Configuration

To change interaction types or the distance cutoff, edit `IFP_f2f_wrapper.yaml`:

```yaml
- name: interaction
  value: ['HBAcceptor', 'HBDonor', 'Cationic', 'PiCation', 'PiStacking']

- name: cutoff
  value: 4.0
```

To switch the receptor dataset, change `_dataset` in `Stage 1` (pdb_processor.ipynb) and
re-run Stage 1 — Stage 2 will automatically pick up the new `processed_files/` contents.
