###### Authors: I. Moulanier, F. Massimo

###### This file contains the initialization and iteration loop of the GSA-MD
###### i.e. the Gerchberg-Saxton Algorithm with Mode Decomposition 
###### It works both in cartesian and cylindrical geometry

import numpy as np
from ..mode_basis import *
from .reconstruction_utilities   import correct_field_amplitude
from .reconstruction_diagnostics import *

def GSA_MD(dict_image_preprocessing,dict_mesh,dict_GSA):

    number_of_planes                 = np.size(dict_mesh["plane_x_coordinates"])

    ##### Extract the energy at the reference plane that was computed during preprocessing
    energy_reference_plane           = dict_image_preprocessing["energy_reference_plane"]

    ##### Check inputs for the construction of the mode basis
    dict_mode_basis                  = check_mode_basis_inputs(dict_image_preprocessing,dict_GSA["lambda_0"],dict_GSA["dict_mode_basis"]) 
    
    ##### Store the fields of the mode basis
    dict_GSA["dict_mode_basis"]      = store_mode_basis_fields(dict_GSA["lambda_0"],dict_image_preprocessing,dict_mesh,dict_GSA["dict_mode_basis"])

    ##### Extract the experimentally measured field amplitude that the GSA-LG-MD will try to reconstruct/fit
    field_amplitude_exp              = dict_GSA["field_amplitude_exp "]

    #### Start the GSA-MD loop
    print("# GSA-MD loop started\n")

    for iter in range(dict_GSA["N_iterations"]):
        print("Iteration = ",iter)

        for i_plane in range(0,np.size(dict_mesh["plane_x_coordinates"])):
            print("- plane = ",i_plane)

            if (iter==0) and (i_plane==0):
                # Compute the initial mode decomposition (MD) Coefficients as the scalar product between the 
                # sqrt(Fluence)*exp(1j*initial_gaussian_beam_phase) and each of the modes in the dict_mode_basis
                # if dict_GSA["use_initial_Gaussian_phase"] = False, this phase is identically zero
                Coeffs_MD             = project_field_on_mode_basis(
                                                                    field_amplitude_exp[i_plane, :, :] 
                                                                    * np.exp(1j*dict_GSA["initial_gaussian_beam_phase"]),
                                                                    i_plane,
                                                                    dict_image_preprocessing,
                                                                    dict_mesh,
                                                                    dict_GSA["dict_mode_basis"]
                                                                    )

            else:

                # Reconstruct the field using the previous estimate of the MD coefficients
                E_reconstructed       = reconstruct_field_at_plane(
                                                                   Coeffs_MD,
                                                                   i_plane,
                                                                   dict_image_preprocessing,
                                                                   dict_mesh,
                                                                   dict_GSA["dict_mode_basis"]
                                                                   )
                E_corrected           = correct_field_amplitude(E_reconstructed[:,:],field_amplitude_exp[i_plane, :, :],dict_image_preprocessing)


                # Compute the MD Coefficients at this plane as the scalar product between the 
                # E_corrected and each of the modes in the dict_mode_basis
                Coeffs_MD_this_plane  = project_field_on_mode_basis(
                                                                    E_corrected[:,:], 
                                                                    i_plane,
                                                                    dict_image_preprocessing,
                                                                    dict_mesh,
                                                                    dict_GSA["dict_mode_basis"]
                                                                    )

                # new Coeffs_MD = Linear combination of the coefficients at this plane and their previous estimate
                Coeffs_MD             = 0.5 * ( Coeffs_MD + Coeffs_MD_this_plane )
                # Normalize and update the Coeffs_MD
                Coeffs_MD             = Coeffs_MD * np.sqrt(energy_reference_plane) / np.sqrt(np.sum(np.absolute(Coeffs_MD)**2))
                dict_GSA              = save_Coeffs_MD(Coeffs_MD,dict_image_preprocessing,dict_GSA)

            ####  Diagnostics
        if (iter%dict_GSA["iterations_between_outputs"]==0) or (iter==dict_GSA["N_iterations"]-1):
            dict_GSA                  = compute_error_and_dump_output( 
                                                                     iter,
                                                                     dict_image_preprocessing,
                                                                     dict_mesh,
                                                                     dict_GSA
                                                                     )                                          
        print()


    print("# END of GSA-MD reconstruction\n")
    print()

    return dict_GSA