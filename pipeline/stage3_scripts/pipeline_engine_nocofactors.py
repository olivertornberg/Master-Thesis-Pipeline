import veloxchem as vlx
import os
from collections import defaultdict
import numpy as np
import pipeline_utils as utils
import csv



class DescriptorDriver:
    def __init__(self):
        self.molecule = None       #molecule for descriptor calculation
        self.basis = None          #basis set 
        self.scf_drv = None        #scf_drv
        self.scf_results = None    #results from geometry optimization prior to descriptor calculations
        
        self.molgrid = None                   #tuple containing 5 lists  
        self.surface_points = None            #surface points (n x [x,y,z]) extracted from molgrid based on an electron density range
        self.point_atom_indices = None        #list of atom indices for every point n

        self.occ_mo_points_amplitude = []      #MO amplitude for occupied orbitals in every point
        
        self.unocc_mo_points_amplitude = []    #MO amplitude for occupied orbitals in every point


        self.ea_values = []
        self.ie_values = []

        self.atomtypes = []                    #List containing atomtypes of all atoms according to amber atomtypes in GAFF (atom indices 1 - n)
        self.SASA = None                       #Solvent-accessible surface area (Å^2) Could also get per atom?
        self.log_p = None

        self.resp_charges = None
        self.atom_indices = None
        

        self.ea_ie_results = None

        self.results_dict = {}





    def compute_respcharges(self):
                # RESP computation
        resp_drv = vlx.RespChargesDriver()
        resp_basis = vlx.MolecularBasis.read(self.molecule, "6-31G*")
        self.resp_charges = resp_drv.compute(self.molecule, resp_basis)
        
        
    
   
    

#Work-around for vlx.XCIntegrator to evaluate GTO values on a custom grid.
    def compute_gto_values_at_custom_points(self, custom_points: np.ndarray):
        """
        custom_points: numpy array of shape (N, 3) in bohr
        """
       
        # 1. Create your weights
        weights = np.ones(len(custom_points))

        # 2. Stack them into a (4, N) array
        # Use np.vstack to stack the x, y, z, and weights vertically
        molgrid_points = np.vstack((
        custom_points[:, 0],  # row 0: x
        custom_points[:, 1],  # row 1: y
        custom_points[:, 2],  # row 2: z
        weights               # row 3: weights
        ))
        from veloxchem.veloxchemlib import DenseMatrix, MolecularGrid

       


        coords_mat = DenseMatrix(molgrid_points)
        mol_grid = MolecularGrid(coords_mat)


        mol_grid.partition_grid_points()
        mol_grid.distribute_counts_and_displacements(0, 1) 

        xc_drv = vlx.XCIntegrator()
        chi_g = xc_drv.compute_gto_values(self.molecule, self.basis, mol_grid)
        
        return chi_g
    



    def create_surface_points(self):
        
        cpcm_drv = vlx.CpcmDriver()
        raw_cpcm_points = cpcm_drv.generate_cpcm_grid(self.molecule)
        self.molgrid = raw_cpcm_points[0][:, :3]
        chi_g = self.compute_gto_values_at_custom_points(self.molgrid)

        D = self.scf_results['D_alpha'] + self.scf_results['D_beta']
        G = np.einsum("ab,bg->ag", D, chi_g)
        n_g = np.einsum("ag,ag->g", chi_g, G)
    
        self.surface_points = np.column_stack((self.molgrid[:, 0], self.molgrid[:, 1], self.molgrid[:, 2]))
        print(f'No. of surface points created: {len(self.surface_points)}')
        





    def assign_points_to_atoms(self):
        atom_positions = self.molecule.get_coordinates_in_bohr()
        atom_vdw_radii = self.molecule.vdw_radii_to_numpy()  # Get van der Waals radii for each atom in bohr

        point_atom_indices = []
        for point in self.surface_points:
            distances = np.linalg.norm(atom_positions - point, axis=1)     # Calculate distances from this point to all atoms
            adjusted_distances = distances - atom_vdw_radii      # Adjust distances by subtracting van der Waals radii
            closest_atom_index = np.argmin(adjusted_distances)       # Find the index of the closest atom considering van der Waals radii
            point_atom_indices.append(closest_atom_index)

        self.point_atom_indices = point_atom_indices
        
        #check that amount of surface points equals the no. of point-to-atom indices 
        assert len(self.surface_points) == len(self.point_atom_indices), \
        f"Mismatch! Created {len(self.surface_points)} surface points " \
        f"but generated {len(self.point_atom_indices)} point-to-atom indices"



    def molecular_orbital_amplitude_per_point(self):
        
        chi_g = self.compute_gto_values_at_custom_points(self.molgrid)
        mol_orb_coeff = self.scf_results["C_alpha"]

        occ_orb = sum(self.scf_results['occ_alpha'])   
        tot_orb = len(self.scf_results['occ_alpha'])

        for i in range(int(occ_orb)):                              #for occupied orbitals
            mo_val = np.dot(chi_g.T, mol_orb_coeff[:, i])   # Atomic orbitals amplitude (per point in gid) * Coeffients for linnearily combining AO into MO = MO amplitude in each point
            self.occ_mo_points_amplitude.append(mo_val)

        for i in range(int(occ_orb), int(tot_orb)):                     #same
            mo_val = np.dot(chi_g.T, mol_orb_coeff[:, i])
            self.unocc_mo_points_amplitude.append(mo_val)
                

    def compute_IE_and_EA(self):
        energy_occ = self.scf_results['E_alpha'][:int(sum(self.scf_results['occ_alpha']))]    # Energies (Hartree) of occupied alpha MOs as calculated by scfdrver    
        energy_unocc = self.scf_results['E_alpha'][int(sum(self.scf_results['occ_alpha'])):] # Energies (Hartree) of virtual (empty) alpha MOs as calculated by scfdrver
         
        for idx, point in enumerate(self.surface_points):     #loop over all the surface points, keeping idx to find the correct surface point for every MO in "self.unocc_mo_points_amplitude"
            
            ea_point_numerator, ea_point_denominator  = 0, 0 # Initialize 

            for j in range(len(self.unocc_mo_points_amplitude)):  #Loop over  all unoccupied MOs; LUMO, LUMO+1, etc and their amplitude in every point [[MO1: p1, p2 ... pn], [MO2 p1, p2 ... pn]]

                unocc_mo_ampl_squared = (self.unocc_mo_points_amplitude[j][idx])**2         #squared cumulative amplitude for LUMO1 point1, LUMO2 point 1, LUMO3 point1 until LUMOn, point n

                ea_point_numerator += (unocc_mo_ampl_squared * energy_unocc[j])        #sum of squared amplitudes x virtual orbital energy
                ea_point_denominator += unocc_mo_ampl_squared                          #sum of squared amplitudes

            ea = (-ea_point_numerator / ea_point_denominator )* 27.211407953              #Hartree to eV
            self.ea_values.append(ea)  


            ie_point_numerator, ie_point_denominator  = 0, 0 # Initialize 

            for j in range(len(self.occ_mo_points_amplitude)):  #Loop over  all occupied MOs and their amplitude in every point [[MO1: p1, p2 ... pn], [MO2 p1, p2 ... pn]]

                occ_mo_ampl_squared = (self.occ_mo_points_amplitude[j][idx])**2         #squared cumulative amplitude 

                ie_point_numerator += (occ_mo_ampl_squared * abs(energy_occ[j]))        #sum of squared amplitudes x orbital energy
                ie_point_denominator += occ_mo_ampl_squared                          #sum of squared amplitudes

                 
            ie = (ie_point_numerator/ie_point_denominator)* 27.211407953            #Hartree to eV
            self.ie_values.append(ie)

        print(f'{len(self.ea_values)}, {len(self.ie_values)} values of electron affinity and ionization energy calculated for {len(self.surface_points)} surface points ')




    def compute_atom_statistics(self):

        # Create point-level data
        point_data = []
        for i in range(len(self.surface_points)):
            point_data.append({
                'point_index': i + 1,
                'atom_label': self.molecule.get_label(self.point_atom_indices[i]),
                'atom_index': int(self.point_atom_indices[i]),
                'IE': self.ie_values[i],
                'EA': self.ea_values[i]
            })
        
        # Group by atom for statistics
        ea_data = defaultdict(list)
        ie_data = defaultdict(list)
        
        for point in point_data:
            atom_idx = point['atom_index']
            ea_data[atom_idx].append(point['EA'])
            ie_data[atom_idx].append(point['IE'])
        
        # Calculate per-atom statistics
        atom_statistics = {}
        
        for atom in range(len(self.molecule.get_labels())):
            print(f'Atom : {atom}')
            atom_statistics[atom] = {
                'min_EA': min(ea_data[atom]),
                'max_EA': max(ea_data[atom]),
                'mean_EA': np.mean(ea_data[atom]),
                'std_EA': np.std(ea_data[atom]),
                'n_points': len(ea_data[atom]),
                'min_IE': min(ie_data[atom]),
                'max_IE': max(ie_data[atom]),
                'mean_IE': np.mean(ie_data[atom]),
                'std_IE': np.std(ie_data[atom])
            }
        
        # Store in self.results
        self.ea_ie_results = atom_statistics
        
        

        
    def define_atom_types(self):                           #Uses AtomTypeIdentifier to categorize each atom according to GAFF
        atomtypeidentifier = vlx.AtomTypeIdentifier()
        self.atomtypes = atomtypeidentifier.generate_gaff_atomtypes(self.molecule)




    def compute_log_p_and_SASA(self): 
    

        #SASA calculation
        smd = vlx.SmdDriver()
        smd.solute = self.molecule
        sasa_list = smd._get_SASA()
        self.sasa = sum(sasa_list) # Å^2

        #Calc scf energy for water solvation
        basis = vlx.MolecularBasis.read(self.molecule, 'def2-svp')

        scf_drv = vlx.ScfRestrictedDriver()
        scf_drv.xcfun = 'b3lyp'
        scf_drv.solvation_model = 'smd'
        scf_drv.smd_solvent = 'water'

        scf_results_water = scf_drv.compute(self.molecule, basis)

        basis = vlx.MolecularBasis.read(self.molecule, 'def2-svp')

        scf_drv = vlx.ScfRestrictedDriver()
        scf_drv.xcfun = 'b3lyp'
        scf_drv.solvation_model = 'smd'
        scf_drv.smd_solvent = '1-octanol'

        scf_results_octanol = scf_drv.compute(self.molecule, basis)
        

        #calculate log p
        hartree_to_j_mol = 2625500.2
        R = 8.3144626  # J / (mol * K)
        T = 298.15     # Kelvin

 
        ddg_solv_hartree  = scf_results_octanol['scf_energy'] - scf_results_water['scf_energy']
        ddg_solv_j_mol = ddg_solv_hartree * hartree_to_j_mol

        self.log_p = -ddg_solv_j_mol / (np.log(10) * R * T)



    def compile_results(self):
        
        self.results_dict['IE_EA'] = self.ea_ie_results
        self.results_dict['atomtypes'] = self.atomtypes
        self.results_dict['log_p'] = self.log_p
        self.results_dict['sasa'] = self.sasa
        self.results_dict['RESP_charges'] = self.resp_charges
        self.results_dict['ie_surface_average'] = np.mean(self.ie_values)
        self.results_dict['ea_surface_average'] = np.mean(self.ea_values)
        self.results_dict['scf_results'] = self.scf_results


        









    def compute_descriptors(self, molecule, basis=None, scf_drv=None, scf_results = None):
        self.molecule = molecule


        # Auto-create basis if not provided
        if basis is None:
            basis = vlx.MolecularBasis.read(molecule, '6-31G**')
        self.basis = basis
    
        # Auto-create SCF driver if not provided  
        if scf_drv is None:
            scf_drv = vlx.ScfRestrictedDriver()
            scf_drv.xcfun = 'b3lyp'
        self.scf_drv = scf_drv
        
        if scf_results is None:
            scf_results = scf_drv.compute(self.molecule, self.basis)  #electronic structure needed to find surface grid
        self.scf_results = scf_results
 
      
        # Compute RESP charges for all geometries
        print(f"computing RESP-charges...")
        self.compute_respcharges()    

        #compute surface points for molecule
        print(f"Creating surface grid...")
        self.create_surface_points()
        
        #assign each point to closest atom
        print(f"Assigning grid points to atoms...")
        self.assign_points_to_atoms()

        #calculate molecular orbital amplitude at each point
        print(f"Computing atomic IE and EA enegies...")
        self.molecular_orbital_amplitude_per_point()

        self.compute_IE_and_EA()    #eV

        # Compute atom statistics and store in self.results
        
        self.compute_atom_statistics()



        # Define atomtypes in molecule
        self.define_atom_types()


       # self.compute_log_p_and_SASA()
        self.compute_log_p_and_SASA() 
        self.compile_results()


        return self.results_dict 






#Wrapper for ensembleparser that takes a pdb complex, checks ligand_charges.txt for charges and creates the ensemble object

def ensemble_parser_chargesensitive(ligand_charges_file_path, pdb_complex, npe_cutoff, pe_cutoff):
    """
    creates ensemble object, takes arguments:
    ligand_charges_path - path to file containing chareg info
    pdb_complex - the name of the complex to create an ensemble of
    npe_cutoff - radial cutoff for non-polarizable embedding  
    pe_cutoff - radial cutoff for polarizable embedding 
    """

    filename = os.path.basename(pdb_complex)  #filename without specified path for the ensembleparser
    
    found, matching_entries, charges = utils.check_inchikey_in_file(f'{ligand_charges_file_path}', filename) # lig_charge_path: e.g. /home/oliverto/Degree_projectVM/ligand_charges.txt
    
    
    total_charge = 0
    
    if found:
        # Loop through each detected charge key (e.g., ["N1+", "O1-"])      LOOK AT THE INT BEFORE +-, not just first occurence
        for c_str in charges:
            if c_str.endswith('+'):
                total_charge += 1
                
            elif c_str.endswith('-'):
                total_charge -= 1
    
    #print(f"PDB: {filename}")
    print(f"Detected Charge Keys: {charges}")
    print(f"Calculated Net Charge: {total_charge}")
    
    ens_parser = vlx.EnsembleParser()
    
    ensemble = ens_parser.structures(
        trajectory_file=pdb_complex,
        qm_region='resname LIG',
        env_region='protein',
        npe_cutoff = npe_cutoff,
        pe_cutoff = pe_cutoff,
        qm_charge = total_charge
        )
    
    print(f"Number of npe residues: {ensemble[0]['number_residues_npe']}")
    print(f"Number of pe residues: {ensemble[0]['number_residues_pe']}")
    print(f"QM region charge: {ensemble[0]['qm_charge']}")
    print(f"QM region multiplicity: {ensemble[0]['qm_multiplicity']}")


    return ensemble, total_charge



#Function  that performs QM-MM scf-calculations on ensemble object 
def ensemble_driver_caller(ensemble, xfunc = None, basis= None):
    
    ens_drv = vlx.EnsembleDriver()
    
    ens_drv.set_env_models(
    pe_model=["SEP", "CP3"],
    npe_model=["tip3p", "ff19sb"],
    )

    if not xfunc:
        ens_drv.xcfun = "b3lyp"
    else: 
        ens_drv.xcfun = xfunc

    if not basis:
        basis =  "6-31G**"

    results = ens_drv.compute(
    ensemble,
    basis_set = basis
    )

    return results 



#Function that creates a ligand and vlx.Mol, ligand xyz-string,  object from ensemble
def ligand_ensemble_to_mol(ensemble):
    ligand_coords= ensemble[0]['qm_coords']
    ligand_elements= ensemble[0]['qm_elements']
    num_atoms = len(ligand_elements)
    
    ligand_xyz_string = ''
    ligand_xyz_string += f'{num_atoms}\n'
    ligand_xyz_string += f'ligand\n'

    for i in range(num_atoms):
        ligand_xyz_string += f'{ligand_elements[i]}    '
        ligand_xyz_string += f'{ligand_coords[i][0]:.7f}    '
        ligand_xyz_string+= f'{ligand_coords[i][1]:.7f}    '
        ligand_xyz_string+= f'{ligand_coords[i][2]:.7f}    \n'

    ligand_mol = vlx.Molecule.read_xyz_string(ligand_xyz_string)
    return ligand_mol, ligand_xyz_string

#joins ligand and residue strings into single xyz-string
def join_ligand_residue_xyz(ligand_string, residue_string):
    lig = ligand_string.split('\n')
    res = residue_string.split('\n')

    num_atoms = int(lig[0]) + int(res[0])
    res_lig_xyz_string = ''
    res_lig_xyz_string += f'{num_atoms}\n'
    res_lig_xyz_string += f'ligand + Residue\n'

    for i in range(int(lig[0])):
        res_lig_xyz_string += f' {lig[i+2]}    \n'
    for i in range(int(res[0])):
        res_lig_xyz_string += f' {res[i+2]}    \n'

    return res_lig_xyz_string

#Function that takes carbon-coords, a the carbon's bond-vector and generates hydrogens to create a methyl-group

def generate_methyl_hydrogens(c_coord, v_nc, bond_length=1.089):
    """
    Calculates coordinates for 3 Hydrogens to form a methyl group.
    :param c_coord: np.array([x, y, z]) of the terminal Carbon
    :param v_nc: np.array([x, y, z]) vector pointing from N to C
    :param bond_length: C-H bond length in Angstroms 
    """
    # 1. Normalize the main axis (N->C)
    axis = v_nc / np.linalg.norm(v_nc)
    
    # 2. Find an arbitrary perpendicular vector (w)
    # We pick a non-parallel vector to cross with
    pick_vec = np.array([1, 0, 0]) if abs(axis[0]) < 0.9 else np.array([0, 1, 0])
    w = np.cross(axis, pick_vec)
    w /= np.linalg.norm(w)
    
    # 3. Geometry constants for tetrahedral shape
    # The angle between C-N and C-H is ~70.53 degrees (180 - 109.47)
    theta = np.deg2rad(70.53) 
    phi_steps = [0, np.deg2rad(120), np.deg2rad(240)]
    
    h_coords = []
    
    for phi in phi_steps:
        # Rodrigues' Rotation Formula to rotate 'w' around 'axis' by 'phi'
        w_rotated = (w * np.cos(phi) + 
                     np.cross(axis, w) * np.sin(phi) + 
                     axis * np.dot(axis, w) * (1 - np.cos(phi)))
        
        # Combine the component along the axis and the component perpendicular
        # to get the final C-H direction
        direction = (axis * np.cos(theta)) + (w_rotated * np.sin(theta))
        
        # Place the hydrogen
        h_pos = c_coord + (direction * bond_length)
        h_coords.append(h_pos)
        
    return np.array(h_coords)


def interacting_ensemble_to_mols(ensemble, ligand_total_charge, interactions):     
    '''
    Takes the ensemble output of the ensemble driver, and the interaction 
    dict from the parse_single_ifp_file and returns molecule object of each 
    of the interaction residues with backbone-caps, with and without ligand 
    as well as the single ligand mol. 

    returns: 
    residue_fragments - dict containing ({
                'res_molecule': temp_res_mol,
                'res_lig_molecule': temp_res_lig_mol,
                'res_id': interacting_residue_id,
                'res_xyz_string': interacting_xyz_string,
                'res_lig_xyz_string': lig_res_string,
                'interaction_type': interaction_type
                })
    complex_name - the file name of the complex of which the interaction originates
    ligand_total_charge - ligand charge (for scf calculations)
    ligand_molecule - molecule object of ligand 

    '''


    residue_fragments =[]

    #Extract all information from passed variables
    complex_name = interactions['file_name'].strip('_ifp.sdf')
    total_interactions = interactions['num_interactions']
    unique_residue_ids = []

    resnames =  np.concatenate(( ensemble[0]['npe_resnames'],ensemble[0]['pe_resnames']))
    resids = np.concatenate(( ensemble[0]['npe_resids'],ensemble[0]['pe_resids']))
    atom_names = np.concatenate(( ensemble[0]['npe_atom_names'], ensemble[0]['pe_atom_names']))
    coords = np.concatenate(( ensemble[0]['npe_coords'],ensemble[0]['pe_coords'] ))
    elements = np.concatenate((ensemble[0]['npe_elements'], ensemble[0]['pe_elements'])) 


    #Ligand molecule object and xyz file. 
    ligand_molecule, lig_xyz = ligand_ensemble_to_mol(ensemble)
    ligand_molecule.set_charge(ligand_total_charge)




    
    #Loop over all the interactions
    for interaction in interactions['interactions']:    #for every interaction identified by IFP
        
        interaction_type = interaction['interaction_type']
        interacting_residue_id = interaction['residue_number'] 
        downstream_interacting_residue_id = interacting_residue_id +1 
        upstream_interacting_residue_id = interacting_residue_id -1
        interacting_residue_name = None 

        if int(interacting_residue_id) in unique_residue_ids: #If the residue interacting has not already been handled and extracted.
            for residue in residue_fragments:
                if residue['res_id'] == interacting_residue_id:
                    previous_interactions = residue['interaction_type']
                    residue['interaction_type'] = f'{previous_interactions}, {interaction_type}'
        
        else:
            #Extract geometry of interacting residue from npe or pe region.
            interacting_res_idx = np.where(np.isin(resids, [interacting_residue_id]))[0]

            interacting_res_coords = resnames[interacting_res_idx]
            interacting_residue_name = interacting_res_coords[0]
            interacting_res_coords = coords[interacting_res_idx]
            interacting_res_elements = elements[interacting_res_idx]
            downstream_interacting_residue_id

            interacting_xyz_string = ''
            interacting_xyz_string += f'{len(interacting_res_elements)+12}\n'
            interacting_xyz_string += f' \n'


            for i in range(len(interacting_res_elements)):
                interacting_xyz_string += f'{interacting_res_elements[i]}    '
                interacting_xyz_string += f'{interacting_res_coords[i][0]:.7f}    '
                interacting_xyz_string += f'{interacting_res_coords[i][1]:.7f}    '
                interacting_xyz_string += f'{interacting_res_coords[i][2]:.7f}    \n'


            # downstream backbone cap
            downstream_interacting_res_idx = np.where(np.isin(resids, [downstream_interacting_residue_id]))[0]
            
                #Nitrogen-bound hydrogen extracted from downstream backbone
            for atom_id in downstream_interacting_res_idx:
                if atom_names[atom_id] == 'H':                                     
                    h_coords = [coords[atom_id][0],coords[atom_id][1],coords[atom_id][2]]
                    interacting_xyz_string += f"H    {h_coords[0]:.7f}    {h_coords[1]:.7f}    {h_coords[2]:.7f}\n"   #Add coords to xyz-tring for molecule object
                
            d_n_coords = []
            for atom_id in downstream_interacting_res_idx:
                    if atom_names[atom_id] == 'N':
                        d_n_coords.append(coords[atom_id][0])         
                        d_n_coords.append(coords[atom_id][1])
                        d_n_coords.append(coords[atom_id][2]) 
                        interacting_xyz_string += f"N    {d_n_coords[0]:.7f}    {d_n_coords[1]:.7f}    {d_n_coords[2]:.7f}\n" #Add coords to xyz-tring for molecule object

                #Alpha-carbon from downstream backbone
            d_ca_coords = []   #save coords of nitrogen for N-->CA bond vector
            for atom_id in downstream_interacting_res_idx:
                if atom_names[atom_id] == 'CA':                                     
                    d_ca_coords.append(coords[atom_id][0])
                    d_ca_coords.append(coords[atom_id][1])
                    d_ca_coords.append(coords[atom_id][2])
                    interacting_xyz_string += f"C    {d_ca_coords[0]:.7f}    {d_ca_coords[1]:.7f}    {d_ca_coords[2]:.7f}\n"  #Add coords to xyz-tring for molecule object


                #create v_nitrogen_alphacarbon vector 
            v_nca = np.array(d_ca_coords)- np.array(d_n_coords) 
            
            d_h_coords = generate_methyl_hydrogens(d_ca_coords, v_nca, bond_length=1.089)   #generate tetraheral hydrogens on "alpha-carbon methyl group"
            
            for i, pos in enumerate(d_h_coords):
                interacting_xyz_string += f"H    {pos[0]:.7f}    {pos[1]:.7f}    {pos[2]:.7f}\n"

                
            # upstream backbone cap
            upstream_interacting_res_idx = np.where(np.isin(resids, [upstream_interacting_residue_id]))[0]

                #Carbonyl carbon extracted from upstream stream backbone
            u_c_coords = []     #save coords of nitrogen for N-->CA bond vector
            for atom_id in upstream_interacting_res_idx:
                if atom_names[atom_id] == 'C':     
                    u_c_coords.append(coords[atom_id][0])
                    u_c_coords.append(coords[atom_id][1])
                    u_c_coords.append(coords[atom_id][2])                                
                    interacting_xyz_string += f"C    {u_c_coords[0]:.7f}    {u_c_coords[1]:.7f}    {u_c_coords[2]:.7f}\n"   #Add coords to xyz-tring for molecule object
                
                #Backbone oxygen extracted from upstream backbone
            for atom_id in upstream_interacting_res_idx:
                if atom_names[atom_id] == 'O':                                     
                    temp_coords = [coords[atom_id][0],coords[atom_id][1],coords[atom_id][2]]
                    interacting_xyz_string += f"O    {temp_coords[0]:.7f}    {temp_coords[1]:.7f}    {temp_coords[2]:.7f}\n"   #Add coords to xyz-tring for molecule object
                
                #alpha-carbon extracted from upstream backbone
            u_ca_coords = []
            for atom_id in upstream_interacting_res_idx:
                if atom_names[atom_id] == 'CA':   
                    u_ca_coords.append(coords[atom_id][0])
                    u_ca_coords.append(coords[atom_id][1])
                    u_ca_coords.append(coords[atom_id][2])
                    interacting_xyz_string += f"C    {u_ca_coords[0]:.7f}    {u_ca_coords[1]:.7f}    {u_ca_coords[2]:.7f}\n"   #Add coords to xyz-tring for molecule object
                

                #create v_carbonyl-carbon_alphacarbon vector 
            v_cca = np.array(u_ca_coords)- np.array(u_c_coords) 
                

            d_h_coords = generate_methyl_hydrogens(u_ca_coords, v_cca, bond_length=1.089)   #generate tetraheral hydrogens on "alpha-carbon methyl group"
            
            for i, pos in enumerate(d_h_coords):
                interacting_xyz_string += f"H    {pos[0]:.7f}    {pos[1]:.7f}    {pos[2]:.7f}\n"
            
                
            temp_res_mol = vlx.Molecule.read_xyz_string(interacting_xyz_string)

            lig_res_string = join_ligand_residue_xyz(interacting_xyz_string,lig_xyz )
            temp_res_lig_mol = vlx.Molecule.read_xyz_string(lig_res_string)
            temp_res_lig_mol.set_charge(ligand_total_charge)


            residue_fragments.append({
                'res_molecule': temp_res_mol,
                'res_lig_molecule': temp_res_lig_mol,
                'res_id': interacting_residue_id,
                'res_name': interacting_residue_name,
                'res_xyz_string': interacting_xyz_string,
                'res_lig_xyz_string': lig_res_string,
                'interaction_type': interaction_type

                })
        
        unique_residue_ids.append(int(interacting_residue_id))
        

        

            
    

    print(f'{total_interactions} total interactions found in {len(residue_fragments)} unique residues.')
    return residue_fragments, complex_name, ligand_total_charge, ligand_molecule





def calculate_interaction_strengths(residue_fragments, ligand_molecule ):
    '''
    Takes the molecule-objects of residue, ligand and resiude+ligand and computes interaction energy.  

    returns: 
    result list containing dict ({
            'interaction_energy': interaction_energy, (kcals/mol)
            'scf_results_res':scf_results_res,
            'scf_results_res_lig' : scf_results_res_lig,
            'scf_results_lig': scf_results_lig,
            'interaction_type': interaction_type
        })
    '''
    
    number_of_residues = len(residue_fragments)
    print(f'Performing scf calcualtions on {number_of_residues} ligand-residue interactions...')

    results = []

    for i in range(len(residue_fragments)):
        interaction_type = residue_fragments[i]['interaction_type']
        residue_id = residue_fragments[i]['res_id']

        res_basis = vlx.MolecularBasis.read(residue_fragments[i]['res_molecule'], 'def2-svpd')
        res_scf_drv = vlx.ScfRestrictedDriver()
        res_scf_drv.xcfun = 'b3lyp'
        res_scf_results = res_scf_drv.compute(residue_fragments[i]['res_molecule'],res_basis)

        res_lig_basis = vlx.MolecularBasis.read(residue_fragments[i]['res_lig_molecule'], 'def2-svpd')
        res_lig_scf_drv = vlx.ScfRestrictedDriver()
        res_lig_scf_drv.xcfun = 'b3lyp'
        res_lig_scf_results_res = res_lig_scf_drv.compute(residue_fragments[i]['res_lig_molecule'],res_lig_basis)

        lig_basis = vlx.MolecularBasis.read(ligand_molecule, 'def2-svpd')
        lig_scf_drv = vlx.ScfRestrictedDriver()
        lig_scf_drv.xcfun = 'b3lyp'
        lig_scf_results = lig_scf_drv.compute(ligand_molecule,lig_basis)

        interaction_energy = (res_lig_scf_results_res['scf_energy'] - (lig_scf_results['scf_energy'] +  res_scf_results['scf_energy'])) * 627.5095

        results.append({
            'interaction_energy': interaction_energy,
            #'scf_results_res':scf_results_res,
            #'scf_results_res_lig' : scf_results_res_lig,
            #'scf_results_lig': scf_results_lig,
            'interaction_type': interaction_type,
            'residue_id': residue_id,
            'residue_name': residue_fragments[i]['res_name']
        })
        
    total_interaction_energy = 0
    for i in results: 
        total_interaction_energy += i['interaction_energy']

    results.append({
        'total_interaction_energy': total_interaction_energy
    })
    return results

def save_to_csv(complex_name, descdrv_results, interaction_results, ligand_molecule, directory):
    # 1. Prepare global metadata
    avg_ea = descdrv_results.get('ea_surface_average', 0.0)
    avg_ie = descdrv_results.get('ie_surface_average', 0.0)
    
    # Get XYZ as a NumPy array and Labels as a list
    xyz_coords = ligand_molecule.get_coordinates_in_angstrom() # Array of shape (N, 3)
    atom_labels = ligand_molecule.get_labels()
    
    # Access nested descriptor data
    ie_ea_data = descdrv_results['IE_EA']
    atom_types = descdrv_results.get('atomtypes', [])
    resp_charges = descdrv_results.get('RESP_charges', [])

    # 2. Handle Interaction Results (splitting per-residue from total)
    if interaction_results and 'total_interaction_energy' in interaction_results[-1]:
        total_data = interaction_results[-1]
        per_residue_interactions = interaction_results[:-1]
        total_energy = total_data.get('total_interaction_energy', 0.0)
    else:
        per_residue_interactions = interaction_results
        total_energy = "N/A"  


    
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    filename = f'{directory}/{complex_name}_results.csv'


    with open(filename, 'w', newline='') as f:
        # 3. Header Comments
        f.write(f"# Complex Name: {complex_name}\n")
        f.write("# --- Interaction Summary ---\n")
        for i, inter in enumerate(per_residue_interactions):
            res_id = inter.get('residue_id', 'N/A')
            en = inter.get('interaction_energy', 0.0)
            itype = inter.get('interaction_type', 'N/A')
            f.write(f"# Interaction {i}: Residue {res_id} | Type: {itype} | Energy: {en:.4f}\n")
        
        f.write(f"# TOTAL INTERACTION ENERGY: {total_energy:.4f}\n")
        
        f.write("# --- Surface Averages ---\n")
        f.write(f"# SASA: {descdrv_results.get('sasa', 'N/A')}\n")
        f.write(f"# Log_P: {descdrv_results.get('log_p', 'N/A')}\n")
        f.write(f"# EA_Surface_Avg: {avg_ea:.4f}\n")
        f.write(f"# IE_Surface_Avg: {avg_ie:.4f}\n")
        f.write("# ------------------------\n")
        
        # 4. CSV Table Setup
        writer = csv.writer(f)
        # Added x, y, z columns here
        writer.writerow([
            'atom_id', 'element', 'atom_type', 'x', 'y', 'z', 
            'resp_charge', 'mean_ea', 'std_ea', 'mean_ie', 'std_ie', 'n_points'
        ])
        
        # 5. Write atom-by-atom data
        for atom_idx in sorted(ie_ea_data.keys()):
            stats = ie_ea_data[atom_idx]
            
            # Map indices safely
            element = atom_labels[atom_idx] if atom_idx < len(atom_labels) else "N/A"
            a_type  = atom_types[atom_idx] if atom_idx < len(atom_types) else "N/A"
            charge  = resp_charges[atom_idx] if atom_idx < len(resp_charges) else 0.0
            
            # Extract coordinates for this specific atom index
            # ligand_molecule.get_coordinates_in_angstrom() returns [ [x,y,z], [x,y,z]... ]
            if atom_idx < len(xyz_coords):
                x, y, z = xyz_coords[atom_idx]
            else:
                x, y, z = (0.0, 0.0, 0.0)
            
            row = [
                atom_idx,
                element,
                a_type,
                f"{x:.4f}",
                f"{y:.4f}",
                f"{z:.4f}",
                charge,
                stats.get('mean_EA'),
                stats.get('std_EA'),
                stats.get('mean_IE'),
                stats.get('std_IE'),
                stats.get('n_points')
            ]
            writer.writerow(row)

    print(f"Successfully saved consolidated results to: {filename}")


