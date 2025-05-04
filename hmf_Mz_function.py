def hmf_Mz (m_dmeff,sigma0,haloMass,z):
    import numpy
    import os
    import subprocess
    path_CLASS = '/home/prakharb16/Downloads/class_public-dmeff'   #Path to the directory where Class is installed
    path_galacticus_master = '/home/prakharb16/galacticus-master'  #Path to the directory where galacticus-master is present
    path_datasets_master = '/home/prakharb16/datasets-master'  #Path to the directory where datasets-master is present
    
    
    os.chdir(path_CLASS) #Move to the directory where Class is installed
    
    
    """
    1) Function which makes a parameter file for CLASS using m_dmeff and sigma0
    """
    input_file = 'explanatory_IDM.ini'  #This file should be used for Columb-like models
    
    def modify_ini_file(input_file, m_dmeff,sigma0,z):
        
        """Modifies the 'm_dmeff' line in an INI file and saves the changes to a new file.

        Args:
            input_file (str): Path to the original INI file.
            new_value (float): The new value to replace 'm_dmeff = 0.0' with.

        Raises:
            FileNotFoundError: If the input file is not found.
        """
        try:
            filename, _ = os.path.splitext(input_file)
            print(filename)

            with open(input_file, 'r') as f:
                lines = f.readlines()

            modified_lines = []
            for line in lines:
                if line.strip() == 'm_dmeff = 1e-3':
                    modified_lines.append(f"m_dmeff = {m_dmeff}\n")
                elif line.strip() == 'sigma_dmeff = 1e-40':
                    modified_lines.append(f"sigma_dmeff = {sigma0}\n")
                elif line.strip() == 'z_pk = 0.0':
                    modified_lines.append(f"z_pk = 0.0, {z}\n")   #redshift values at which power spectrum from class is calculated
                else:
                    modified_lines.append(line)
            output_filename = f"{filename}_z{z}_{m_dmeff}MeV_{sigma0}cm2.ini"
            with open(output_filename, 'w') as f:
                f.writelines(modified_lines)

            print(f"Successfully modified '{input_file}' and saved as '{output_filename}'.")


        except FileNotFoundError as e:
            print(f"Error: Input file '{input_file}' not found.")
        return output_filename


   
    """
    2) Running the CLASS for the prepared parameter file
    """
    
    class_params_file = modify_ini_file(input_file, m_dmeff, sigma0,z)
    os.chdir(path_CLASS)
    result = subprocess.run(['./class', f'{class_params_file}'], capture_output=True, text=True) 
    if result.returncode == 0:
        print(result.stdout)  # Access the standard output
    else:
        print("Error:", result.stderr)  # Access the standard error
    output_text = result.stdout
    import re
    sigma8_match = re.search(r"-> sigma8=(\d+\.\d+)", output_text)
    if sigma8_match:
        sigma8_value = float(sigma8_match.group(1))
        print("Extracted sigma8 value:", sigma8_value)
    else:
        print("Sigma8 value not found in the output.")
    
    
    """
    3) Generating the hdf file for the Galacticus Parameter File
    """
    
    
    import h5py
    import numpy as np  # Assuming you'll use NumPy to load data
    import os
    h = 0.67556
    
    os.chdir(path_CLASS)
    filename, _ = os.path.splitext(class_params_file)
    power1_data = np.loadtxt(f"./output/{filename}00_z1_pk.dat")
    power2_data = np.loadtxt(f"./output/{filename}00_z2_pk.dat")
    wavenumber_data = power1_data[:,0]*h
    #file_id = H5Fcreate ("SampleFile.h5", H5F_ACC_EXCL, H5P_DEFAULT, H5P_DEFAULT)
    os.chdir(f'{path_galacticus_master}/parameters/tutorials') #Move to the tutorials directory in Galacticus
    file = h5py.File("powerSpectrum_main.hdf5", "w")
    # Create the HDF5 file
    with file as f:

        # Create groups
        f.create_group("extrapolation")
        f.create_group("extrapolation/wavenumber")
        f.create_group("parameters")
        #f.create_group("darkMatter")

        # Write root attributes
        f.attrs["description"] = "Transfer Function data generated using CLASS for IDM"  # Replace with your actual description
        file_format_version = 1  # Store as a float
        f.attrs["fileFormat"] = file_format_version
        #f.attrs["redshift"] = 0.0  # Adjust redshift value as needed
        f.attrs['extrapolationWavenumber'] = "abort"
        f.attrs['extrapolationRedshift'] = "abort"

        # Write extrapolation attributes
        extrapolation_group = f["extrapolation"]
        extrapolation_group["wavenumber"].attrs["high"] = (max(wavenumber_data))
        extrapolation_group["wavenumber"].attrs["low"] = (min(wavenumber_data))
        #extrapolation_group["Wavenumber"].attrs["extrapolationWavenumber"] = 'linear'
        # Optionally add "method" attribute if applicable

        # Write parameter attributes
        parameters_group = f["parameters"]
        parameters_group.attrs["HubbleConstant"] = 67.66  # Replace with your value
        parameters_group.attrs["OmegaBaryon"] = 0.04893  # Replace with your value
        parameters_group.attrs["OmegaDarkEnergy"] = 0.6911  # Replace with your value
        parameters_group.attrs["OmegaMatter"] = 0.3088  # Replace with your value
        parameters_group.attrs["temperatureCMB"] = 2.7255  # Replace with your value

        # Write datasets
        f.create_dataset("wavenumber", data=wavenumber_data)
        #f.create_dataset("power", data=np.array([power1_data[:,1]/(h**3),power2_data[:,1]/(h**3), power3_data[:,1]/(h**3)]))
        f.create_dataset("power", data=np.array([power1_data[:,1]/(h**3),power2_data[:,1]/(h**3)]))
        f.create_dataset("redshift", data=np.array([0.,z]))
    
    
    
    """
    4) Preparing the galacticus parameter file
    """
    
    
    def modify_xml_file(input_file, haloMass,z):
    
    
        try:
            filename, _ = os.path.splitext(input_file)
            print(filename)

            with open(input_file, 'r') as f:
                lines = f.readlines()

            modified_lines = []
            for line in lines:
                if line.strip() == f'<fileName value = "{path_galacticus_master}/parameters/tutorials/Driskell_CDM_fig6_2.hdf5"/>':
                    modified_lines.append(f'<fileName value = "{path_galacticus_master}/parameters/tutorials/powerSpectrum_main.hdf5"/>\n')
                elif line.strip() == '<outputFileName value="hmf_Driskell_CDM_fig6_2.hdf5"/>':
                    modified_lines.append('<outputFileName value="haloMassFunction_main.hdf5"/>\n')
                elif line.strip() == '<redshifts value = "0.0"/>':
                    modified_lines.append(f'<redshifts value = "{z}"/>\n')   #redshift values at which power spectrum from class is calculated
                elif line.strip() == '<haloMassMinimum value="1.0e05"/>':
                    modified_lines.append(f'<haloMassMinimum value="{haloMass}"/>\n')   #redshift values at which power spectrum from class is calculated
                elif line.strip() == '<haloMassMaximum value="1.0e15"/>':
                    modified_lines.append(f'<haloMassMaximum value="{haloMass*10}"/>\n')   #redshift values at which power spectrum from class is calculated
                elif line.strip() == '<pointsPerDecade value="20"    />':
                    modified_lines.append(f'<pointsPerDecade value="1"    />\n')   #redshift values at which power spectrum from clas
                elif line.strip() == '<sigma_8 value="0.840308" />':
                    modified_lines.append(f'<sigma_8 value="{sigma8_value}" />\n')   #redshift values at which power spectrum from clas
                else:
                    modified_lines.append(line)
            output_filename = 'haloMassFunction_main.xml'
            with open(output_filename, 'w') as f:
                f.writelines(modified_lines)

            print(f"Successfully modified '{input_file}' and saved as '{output_filename}'.")


        except FileNotFoundError as e:
            print(f"Error: Input file '{input_file}' not found.")
        return output_filename

    
    """
    5) Running the galacticus for the prepared parameter file
    """
    os.chdir(f'{path_galacticus_master}/parameters/tutorials')
    input_file = 'haloMassFunction.xml'
    galacticus_params_file = modify_xml_file(input_file, haloMass,z)
    os.chdir('..')
    os.chdir('..')
    command = f"./galacticus.exe parameters/tutorials/{galacticus_params_file}"
    #print(command)
    os.system(f'rm -rf {path_datasets_master}/dynamic')
    os.environ['GALACTICUS_EXEC_PATH'] = path_galacticus_master
    os.environ['GALACTICUS_DATA_PATH'] = path_datasets_master
    os.system(command)
    
    """
    6) Obtain the haloMassFractionCummulative from the galacticus output
    """
    
    with h5py.File('haloMassFunction_main.hdf5', "r") as f:

         # Access the "Outputs" group
        outputs_group = f["Outputs"]

        # Get all subgroup names within the "Outputs" group
        subgroup_names = list(outputs_group)
        output1 = outputs_group['Output1']
        hmf_cumm = output1['haloMassFractionCumulative'][0]
        print(f"The cummulative halo mass fraction for m_dmeff:{m_dmeff},sigma0:{sigma0},haloMass:{haloMass} at z:{z} is {hmf_cumm} ")
        return hmf_cumm
        
  
  
  
import h5py 
import numpy as np

def hmf_Mz_2d(filename):
    with h5py.File(filename, "r") as f:
        # Access the "Outputs" group
        outputs_group = f["Outputs"]

        # Get all subgroup names
        subgroup_names = list(outputs_group)

        # Sort subgroup names in reverse order based on the numerical value after "Output"
        subgroup_names.sort(key=lambda name: int(name.split("Output")[-1]), reverse=True)
        #print(subgroup_names)
        # Get the number of data points per redshift from the first subgroup
        first_subgroup = outputs_group[subgroup_names[0]]
        redshifts = np.array([5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,16.0,17.0,18.0,19.0,20.0,21.0,22.0,23.0,24.0,25.0,26.0,27.0,28.0,29.0,30.0,31.0,32.0,33.0,34.0,35.0,36.0,37.0,38.0,39.0,40.0,41.0,42.0,43.0,44.0,45.0,46.0,47.0,48.0,49.0,50.0,51.0,52.0,53.0,54.0,55.0,56.0,57.0,58.0,59.0,60.0])
        #redshifts = np.arange(0.0,61.0,1)
        #print(redshifts)
        halo_mass = first_subgroup["haloMass"][:]
        #data_points = len(first_subgroup["haloMassFractionCumulative"][:])  # Assuming all subgroups have the same size
        data_points = len(first_subgroup["haloMassFunctionLnM"][:])

        # Create the 2D array
        halo_mass_functions = np.zeros((len(subgroup_names), data_points))

        # Loop through sorted subgroups and populate the array
        for i, subgroup_name in enumerate(subgroup_names):
            subgroup = outputs_group[subgroup_name]
            halo_mass_functions[i] = subgroup["haloMassFractionCumulative"][:]
    return redshifts, halo_mass, halo_mass_functions
