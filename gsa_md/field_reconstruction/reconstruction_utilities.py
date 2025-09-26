###### Authors: I. Moulanier, F. Massimo

###### This module contains some utilities for the GSA reconstruction

import numpy as np
import os,sys
from scipy import signal
from numba import njit


def check_GSA_inputs(dict_image_preprocessing,dict_GSA):
    
    # Check the inputs of the field reconstruction
    if "use_Mode_Decomposition" not in dict_GSA.keys():
        print("Error: the boolean flag dict_GSA['use_Mode_Decomposition'] must be set.")
        sys.exit()
    
    if (dict_GSA["use_Mode_Decomposition"] == False) and (dict_image_preprocessing["geometry"]=="cylindrical"):
        print("Error: GSA without mode decomposition is implemented only in cartesian geometry")
        sys.exit()
        
    if "lambda_0" not in dict_GSA.keys():
        print("Error: the boolean flag dict_GSA['lambda_0'] must be set.")
        sys.exit()
    
    if "N_iterations" not in dict_GSA.keys():
        print("Error: the boolean flag dict_GSA['N_iterations'] must be set.")
        sys.exit()
        
    if "iterations_between_outputs" not in dict_GSA.keys():
        print("Error: the boolean flag dict_GSA['iterations_between_outputs'] must be set.")
        sys.exit()
        
    if "use_initial_Gaussian_phase" not in dict_GSA.keys():
        dict_GSA["use_initial_Gaussian_phase"] = False
        
    # Print the inputs of the field reconstruction
    
    print("Parameters for field reconstruction:")
    print("- lambda_0                           : ",dict_GSA["lambda_0"                  ]," m")
    print("- use_Mode_Decomposition             : ",dict_GSA["use_Mode_Decomposition"    ])
    print("- N_iterations                       : ",dict_GSA["N_iterations"              ])
    print("- iterations_between_outputs         : ",dict_GSA["iterations_between_outputs"])
    print("- use_initial_Gaussian_phase         : ",dict_GSA["use_initial_Gaussian_phase"])
    print("\n")
        
    return dict_GSA
    
def compute_experimental_field_amplitudes(dict_image_preprocessing,dict_GSA):
    
    if dict_image_preprocessing["geometry"] == "cylindrical":
        field_amplitude_exp           = np.sqrt(dict_image_preprocessing["fluence_exp_images_circular"])
    else:
        field_amplitude_exp           = np.sqrt(dict_image_preprocessing["fluence_exp_images_cartesian"])
    
    dict_GSA["field_amplitude_exp "]  = field_amplitude_exp 
    
    return dict_GSA

def gaussian_phase(dict_image_preprocessing,dict_mesh,dict_GSA):
    
    if dict_GSA["use_initial_Gaussian_phase"] == True:
        
        print("Generating Gaussian phase for the first iteration at the first plane")
        # defines the parabolic phase [in radians] of a gaussian beam at x = plane_x_coordinates[0]
        # using the same lambda_0, waist_0 and mesh used to define the mode basis

        plane_x_coordinates     = dict_mesh      ["plane_x_coordinates"]
        lambda_0                = dict_GSA       ["lambda_0"           ]
    
        if dict_GSA["use_Mode_Decomposition"] == True: 
            # use the same waist and focal plane of the modes
            dict_mode_basis     = dict_GSA       ["dict_mode_basis"    ]
            waist_0             = dict_mode_basis["waist_0"            ]
            x_focus             = dict_mode_basis["x_focus"            ]
            first_plane_index   = 0
            print("  using the same waist_0 and x_focus of the mode basis")
        else:
            number_of_planes    = np.size(plane_x_coordinates)
            ref_plane_index     = dict_image_preprocessing["index_reference_plane"]
            plane_indices_GSA_loop  = [i_plane for i_plane in range(0,number_of_planes)]
            plane_indices_GSA_loop.remove(ref_plane_index)
            first_plane_index   = plane_indices_GSA_loop[0]
            x_focus             = plane_x_coordinates[ref_plane_index]
            
            # Create the frequency parabola mesh used in the Fresnel propagator
            dict_GSA            = initialize_Fresnel_phase_ov_distance(dict_mesh,dict_GSA)
        
        # use the waist provided by the user and use the reference plane as focal plane
            if "waist_gaussian_beam_phase" not in dict_GSA.keys():
                print("Error: you need to provide a dict_GSA['waist_gaussian_beam_phase']")
                sys.exit()
            else:
                waist_0         = dict_GSA       ["waist_gaussian_beam_phase"]
                x_focus         = plane_x_coordinates[dict_image_preprocessing["index_reference_plane"]]
                print("using the dict_GSA['waist_gaussian_beam_phase'] and the reference plane position as x_focus")
    
        k0                      = 2.*np.pi/lambda_0
        x_R                     = np.pi/lambda_0 * (waist_0)**2                           # Rayleigh length [m]
        Delta_x_ov_x_R          = (plane_x_coordinates[first_plane_index]-x_focus)/x_R    # (x-x_focus) / x_R             
        RadCurl                 = x_R*(Delta_x_ov_x_R + 1/Delta_x_ov_x_R)                 # curvature radius of a Gaussian wavefront [m]
    
        if dict_image_preprocessing["geometry"] == "cylindrical":
            r_mesh                      = dict_mesh["r_mesh"    ]
            theta_mesh                  = dict_mesh["theta_mesh"]
            initial_gaussian_beam_phase = np.zeros((r_mesh.size, theta_mesh.size))
            for itheta in range(theta_mesh.size):
                initial_gaussian_beam_phase[:, itheta] = k0*np.square(r_mesh)/(2*RadCurl)
        else:
            y_mesh                      = dict_mesh["y_mesh"]
            z_mesh                      = dict_mesh["z_mesh"]
            ny                          = np.size(y_mesh)
            nz                          = np.size(z_mesh)         
            y2_mesh                     = np.square(y_mesh[:,np.newaxis]*np.ones(shape=(ny,nz)))
            z2_mesh                     = np.square(z_mesh[np.newaxis,:]*np.ones(shape=(ny,nz)))
            initial_gaussian_beam_phase = k0*(y2_mesh+z2_mesh)/(2*RadCurl)  
    
    else:
        
        # just a phase equal to zero
        if dict_image_preprocessing["geometry"] == "cylindrical":
            initial_gaussian_beam_phase = np.zeros_like(dict_image_preprocessing["fluence_exp_images_circular"][0,:,:])
        else:
            initial_gaussian_beam_phase = np.zeros_like(dict_image_preprocessing["fluence_exp_images_cartesian"][0,:,:])
    
    print()        
    dict_GSA["initial_gaussian_beam_phase"] = initial_gaussian_beam_phase
    
    return dict_GSA  
    
def correct_field_amplitude(E_reconstructed_this_plane,field_amplitude_exp_this_plane,dict_image_preprocessing):
    
    # Correct the reconstructed field depending on how far its amplitude is from the experimental one
    # This step significantly accelerates the convergence
    # See Y. Wu et al, Optics Express 29, 2, 1412-1427 (2021) https://doi.org/10.1364/OE.413723
    
    # To compute errors, a boolean mask called inside_r_max will be used, 
    # which is False where the distance from origin is larger than r_max 
    delta                                            = np.square(field_amplitude_exp_this_plane) - np.square(np.abs(E_reconstructed_this_plane))
    # delta                                            = np.zeros_like(field_amplitude_exp_this_plane)
    # delta                                            = compute_delta_numba(delta,field_amplitude_exp_this_plane,E_reconstructed_this_plane)
    delta                                           /= np.amax(field_amplitude_exp_this_plane)**2  
    delta[~dict_image_preprocessing["inside_r_max"]] = 0.  # how should we treat these points out of rmax?    #-np.inf # this will give np.exp(delta)=0 in these points
    
    # this step is different from the corresponding one in I. Moulanier's paper on JOSAB (that was done to improve convergence too)
    # it is done on the reconstructed field 
    # and not on the amplitude of the experimental field multiplied by the phase of the reconstructed field
    E_corrected                                      = E_reconstructed_this_plane * np.exp(delta) 
    # E_corrected                                      = np.zeros_like(E_reconstructed_this_plane)
    # E_corrected                                      = correct_field_numba(E_corrected,E_reconstructed_this_plane,delta)
    
    # this line is the same approach as I. Moulanier's paper on JOSAB
    #E_corrected  = field_amplitude_exp_this_plane * np.exp(1j*np.angle(E_reconstructed_this_plane)) * np.exp(delta)
    
    return E_corrected

# @njit(parallel=False)    
# def compute_delta_numba(delta,field_amplitude_exp_this_plane,E_reconstructed_this_plane):
#     for index_0 in range(field_amplitude_exp_this_plane.shape[0]):
#         for index_1 in range(field_amplitude_exp_this_plane.shape[1]):
#             delta[index_0,index_1] = field_amplitude_exp_this_plane[index_0,index_1]-np.abs(E_reconstructed_this_plane[index_0,index_1])
#     return delta
# 
# @njit(parallel=False)    
# def correct_field_numba(E_corrected,E_reconstructed_this_plane,delta):
#     for index_0 in range(E_corrected.shape[0]):
#         for index_1 in range(E_corrected.shape[1]):
#             E_corrected[index_0,index_1] = E_reconstructed_this_plane[index_0,index_1] * np.exp(delta[index_0,index_1])
#     return E_corrected


def initialize_Fresnel_phase_ov_distance(dict_mesh,dict_GSA):
    
    # Define a matrix Fresnel_phase_ov_distance contaning a part of the Fresnel propagator phase.
    # The full Fresnel propagator will be np.exp(1j*Fresnel_phase_ov_distance*distance),
    # where distance is the difference between the destination and origin plane.
    # The back-propagator is just the complex conjugate of this propagator (assuming that distance is positive).
    # Alternatively, the same propagator can be used for the back-propagator, with a negative distance.
    
    # This function is an adaptation of 
    # https://github.com/openUC2/UC2-GIT/blob/master/WORKSHOP/INLINE-HOLOGRAMM/PYTHON/FresnelPropagator.py
    # from the UC2 - Open and Modular Optical Toolbox (https://github.com/openUC2/UC2-GIT/tree/master)
    # https://doi.org/10.5281/zenodo.4041339

    y_mesh                                = dict_mesh["y_mesh"]
    ny                                    = np.size(y_mesh)
    Ly                                    = 2*np.amax(y_mesh)

    z_mesh                                = dict_mesh["z_mesh"]
    nz                                    = np.size(z_mesh)
    Lz                                    = 2*np.amax(z_mesh)
    
    fy                                    = (1./2/Ly)*np.linspace(-(ny-1), (ny-1), ny)
    fz                                    = (1./2/Lz)*np.linspace(-(nz-1), (nz-1), nz)
    
    fy2_plus_fz2                          = (np.square(fy[:,np.newaxis])+np.square(fz[np.newaxis,:])) * np.ones(shape=(ny,nz))
    fy2_plus_fz2[fy2_plus_fz2>(1. / dict_GSA["lambda_0"])**2]  = 0.

    # The Fresnel propagator is a phase given by the following exponential, 
    # i.e. exp[-i*(k_y**2+k_z**2)/(2*k_0) + i*k0]*x, where x is propagation distance,
    # which corresponds to the Fourier Transform of the Fresnel impulse response.
    Fresnel_phase_ov_distance             =   - np.pi * dict_GSA["lambda_0"] * fy2_plus_fz2
    Fresnel_phase_ov_distance            += 2 * np.pi / dict_GSA["lambda_0"]

    dict_GSA["Fresnel_phase_ov_distance"] = Fresnel_phase_ov_distance
    
    return dict_GSA