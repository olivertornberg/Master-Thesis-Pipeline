import argparse
import os
import pipeline_utils as utils
import pipeline_engine as engine
from pathlib import Path

_pipeline_dir = Path(os.getcwd()).parent
input_root = str(_pipeline_dir / 'processed_files')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--complex", required=True)
    args = parser.parse_args()
    
    # radial cutoff for MM-embedding (Angstrom)
    npe_cutoff = 50
    pe_cutoff = 0

    # paths
    complexes_folder = os.path.join(input_root, 'final_complexes/')
    ifp_outputs_folder = os.path.join(input_root, 'IFP_output/')
    ligand_charges_file_path = os.path.join(input_root, 'ligand_processed/ligand_charges.txt')
    output_directory = os.path.join(_pipeline_dir, 'results')



    ifp_result_filename = utils.convert_complex_name_to_ifp_name(args.complex)
    ifp_output_path = f"{ifp_outputs_folder}{ifp_result_filename}"
    
    # Parse the IFP data to know which residues to calculate
    ifp_interaction_data = utils.parse_single_ifp_file(ifp_output_path)
    
    # build ensemble
    complex_path = f"{complexes_folder}{args.complex}"
    ensemble, total_charge = engine.ensemble_parser_chargesensitive(ligand_charges_file_path, complex_path, npe_cutoff, pe_cutoff)
    

    # Pass interaction to script for exracting capped residues, returns nested list [[mol object, res id, xyz-string], []]
    residue_fragments, complex_name, ligand_total_charge, ligand_molecule = engine.interacting_ensemble_to_mols(ensemble, total_charge, ifp_interaction_data)
    
    # QM/MM SCF on ligand
    results = engine.ensemble_driver_caller(ensemble, ligand_molecule)

    if results == None:
        print(f"\n[!] SCF convergence failure for {args.complex}, skipping.\n")
        return

    # Pass ligand electronic structure to descriptor driver for properties-calculation 
    desc_drv = engine.DescriptorDriver()
    descriptor_result = desc_drv.compute_descriptors(ligand_molecule, scf_results = results )

    # compute residue interaction energies
    interaction_results = engine.calculate_interaction_strengths(residue_fragments, ligand_molecule )
    
    # save results
    engine.save_to_csv(complex_name, descriptor_result, interaction_results, ligand_molecule, output_directory)






if __name__ == "__main__":
    main()