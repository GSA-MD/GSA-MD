###### Authors: I. Moulanier, F. Massimo

###### This file contains the initialization and iteration loop of the GSA
###### i.e. the Gerchberg-Saxton Algorithm without Mode Decomposition,
###### with Fresnel propagator in the frequency domain
###### It works only in cartesian geometry
###### If you are aware of the existence of a cylindrical version, let us know!

from .reconstruction_utilities   import initialize_Fresnel_phase_ov_distance,correct_field_amplitude
from .reconstruction_diagnostics import *
from scipy.fft                   import fft2, ifft2, fftshift, ifftshift
from scipy.ndimage               import median_filter
import numpy as np

def GSA(dict_image_preprocessing,dict_mesh,dict_GSA):
    
    plane_x_coordinates              = dict_mesh["plane_x_coordinates"]
    number_of_planes                 = np.size(plane_x_coordinates)
    
    ##### Extract the experimentally measured field amplitude that the GSA-LG-MD will try to reconstruct/fit
    field_amplitude_exp              = dict_GSA["field_amplitude_exp "]
    ref_plane_index                  = dict_image_preprocessing["index_reference_plane"]
    E_field_ref_plane                = field_amplitude_exp[ref_plane_index,:,:]
    
    ##### Create the frequency parabola mesh used in the Fresnel propagator
    dict_GSA                         = initialize_Fresnel_phase_ov_distance(dict_mesh,dict_GSA)
    
    ##### Create list of planes where the loop will iterate (i.e. all planes except the reference plane)
    
    plane_indices_GSA_loop           = [i_plane for i_plane in range(0,number_of_planes)]
    plane_indices_GSA_loop.remove(ref_plane_index)
    
    
    #### Start the GSA-RMD loop
    print("# GSA loop started\n")
    
    for iter in range(dict_GSA["N_iterations"]):
        print("Iteration = ",iter)
    
        for i_plane in plane_indices_GSA_loop:
            print("- plane = ",i_plane)
            
            if (iter==0) and (i_plane==plane_indices_GSA_loop[0]):
                # Compute the initial mode decomposition (MD) Coefficients as the scalar product between the 
                # sqrt(Fluence)*exp(1j*initial_gaussian_beam_phase) and each of the modes in the dict_mode_basis
                E_field_ref_plane = E_field_ref_plane * np.exp(1j*dict_GSA["initial_gaussian_beam_phase"])
            
            # Reconstruct the field using the IFFT[FFT[E_field_ref]*FresnelPropagator]
            E_reconstructed       = np.fft.ifft2(np.fft.ifftshift(
                                        np.fft.fftshift(np.fft.fft2(E_field_ref_plane))
                                            * np.exp(1j*dict_GSA["Fresnel_phase_ov_distance"]*(plane_x_coordinates[i_plane]-plane_x_coordinates[ref_plane_index]))
                                            ))
            E_reconstructed[~dict_image_preprocessing["inside_r_max"]] = 0.
            
            # amplitude correction bases on how far the field amplitude is from the experimental one
            E_corrected           = correct_field_amplitude(E_reconstructed[:,:],field_amplitude_exp[i_plane, :, :],dict_image_preprocessing)
            
            # Back-propagate the field to the first plane using the IFFT[FFT[E_corrected]*InverseFresnelPropagator]                                                       
            E_field_ref_plane     = np.fft.ifft2(np.fft.ifftshift(
                                        np.fft.fftshift(np.fft.fft2(E_corrected))
                                            * np.exp(-1j*dict_GSA["Fresnel_phase_ov_distance"]*(plane_x_coordinates[i_plane]-plane_x_coordinates[ref_plane_index]))
                                                                     ))
            E_field_ref_plane[~dict_image_preprocessing["inside_r_max"]] = 0.
            
            # amplitude correction bases on how far the field amplitude is from the experimental one                                                 
            E_corrected           = correct_field_amplitude(E_field_ref_plane[:,:],field_amplitude_exp[ref_plane_index, :, :],dict_image_preprocessing)
            
            # retrieve the phase
            phase_ref_plane       = np.angle(E_corrected)
            
            # update the field at the first plane                                                                                            
            E_field_ref_plane     = field_amplitude_exp[ref_plane_index,:,:] * np.exp(1j*phase_ref_plane)
            
        ####  Diagnostics
        if (iter%dict_GSA["iterations_between_outputs"]==0) or (iter==dict_GSA["N_iterations"]-1):
            dict_GSA["phase_ref_plane"]   = phase_ref_plane
            dict_GSA                      = compute_error_and_dump_output( 
                                                                         iter,
                                                                         dict_image_preprocessing,
                                                                         dict_mesh,
                                                                         dict_GSA
                                                                         )                                          
        print()
        
    print("# END of GSA reconstruction\n")
    print()
        
    return dict_GSA
