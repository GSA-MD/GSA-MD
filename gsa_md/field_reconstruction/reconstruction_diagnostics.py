###### Authors: I. Moulanier, A. Guerente, F. Massimo

###### This module contains functions used as diagnostics 
###### during the GSA field reconstruction

from ..mode_basis    import *
from scipy.fft import fft2, ifft2, fftshift, ifftshift
import numpy               as np
import scipy.constants     as sc
import os,sys,time,psutil
import matplotlib.pyplot as plt


def initialize_diagnostics(dict_GSA,dict_mesh):
    
    number_of_planes                = np.size(dict_mesh["plane_x_coordinates"])
    
    #### define the error fit at each output iteration
    dict_GSA["error_fit"          ] = np.ones(number_of_planes)
    
    #### define the time at which every output is computed
    dict_GSA["time_output"]         = np.zeros(1)
    
    return dict_GSA
    
def compute_error_and_dump_output(iter,dict_image_preprocessing,dict_mesh,dict_GSA):
    
    plane_x_coordinates                    = dict_mesh["plane_x_coordinates"]
    number_of_planes                       = np.size(plane_x_coordinates )
    
    # To compute errors, a boolean mask will be used, False where the distance from origin is larger than r_max 
    inside_r_max                           = dict_image_preprocessing["inside_r_max"    ]
    
    
    error_fit                              = dict_GSA             ["error_fit"       ]
    time_output                            = dict_GSA             ["time_output"     ]
    error_fit_at_plane                     = np.ones(number_of_planes)
    
    # if dict_image_preprocessing["geometry"]=="cylindrical":
    #     # to allow comparison between cylindrical and cartesian geometries,
    #     # the error is computed on the energy of its pixel
    #     # (because the total energy at a given plane is independent on the coordinates).
    #     # Therefore a factor r in the energy integral is necessary in cylindrical geometry
    #     weights_geometry       = np.ones_like(inside_r_max) #dict_mesh["r_mesh"][:,np.newaxis]*np.ones_like(inside_r_max) 
    # else:
    #     weights_geometry       = np.ones_like(inside_r_max) 
    
    weights_geometry       = np.ones_like(inside_r_max) 
    
    if dict_image_preprocessing["geometry"]=="cylindrical":
        fluence_exp_images = dict_image_preprocessing["fluence_exp_images_circular"]
    else: 
        fluence_exp_images = dict_image_preprocessing["fluence_exp_images_cartesian"]
    
    
    if dict_GSA["use_Mode_Decomposition"] == True:
        if dict_image_preprocessing["geometry"]=="cylindrical":
            Coeffs_MD = dict_GSA["dict_mode_basis"]["Coeffs_LG_pl"]
        else:
            Coeffs_MD = dict_GSA["dict_mode_basis"]["Coeffs_HG_mn"]
        np.save("outputs/Coeffs_MD_iteration_"+str(iter).zfill(5)+".npy",Coeffs_MD)
    else:
        np.save("outputs/Phase_ref_plane_iteration_"+str(iter).zfill(5)+".npy" ,dict_GSA["phase_ref_plane"])
        ref_plane_index        = dict_image_preprocessing["index_reference_plane"]
        phase_ref_plane        = dict_GSA["phase_ref_plane"]
        E_field_ref_plane      = dict_GSA["field_amplitude_exp "][ref_plane_index,:,:] * np.exp(1j*phase_ref_plane)
        
    for i_plane in range(number_of_planes):
        
        if dict_GSA["use_Mode_Decomposition"] == True:
            E_reconstruction       = reconstruct_field_at_plane(
                                                             Coeffs_MD,
                                                             i_plane,
                                                             dict_image_preprocessing,
                                                             dict_mesh,
                                                             dict_GSA["dict_mode_basis"],
                                                             )
        else:
            if i_plane == ref_plane_index:
                E_reconstruction   = E_field_ref_plane
            else:
                E_reconstruction   = np.fft.ifft2(np.fft.ifftshift(
                                         np.fft.fftshift(np.fft.fft2(E_field_ref_plane)) 
                                             * np.exp(1j*dict_GSA["Fresnel_phase_ov_distance"]*(plane_x_coordinates[i_plane]-plane_x_coordinates[ref_plane_index]))
                                                                 ))
                                                                 
        fluence_reconstruction             = np.abs(E_reconstruction)**2
        # error_fit_at_plane[i_plane]        = np.sum((weights_geometry*fluence_exp_images[i_plane,:,:])[inside_r_max])
        # error_fit_at_plane[i_plane]       -= np.sum((weights_geometry*fluence_reconstruction)[inside_r_max])
        # error_fit_at_plane[i_plane]        = np.abs(error_fit_at_plane[i_plane])
        # error_fit_at_plane[i_plane]       /= np.sum((weights_geometry*fluence_exp_images[i_plane,:,:])[inside_r_max])
        
        pointwise_error                    = np.abs((fluence_reconstruction-fluence_exp_images[i_plane,:,:])*weights_geometry)
        error_fit_at_plane[i_plane]        = np.linalg.norm(pointwise_error[inside_r_max])
        error_fit_at_plane[i_plane]       /= np.linalg.norm((weights_geometry*fluence_exp_images[i_plane,:,:])[inside_r_max])
                                                        
    if (iter==0):
        error_fit                          = error_fit_at_plane
        time_output                        = np.array([time.time()])
        time_diff_with_first_output        = 0.
    else:
        error_fit                          = np.vstack([error_fit,error_fit_at_plane])
        time_diff_with_first_output        = time.time()-time_output[0]
        time_output                        = np.hstack([time_output,time_diff_with_first_output])
    
    print()    
    print("Reconstruction Error    =", [float(f"{float(x):.3f}") for x in error_fit_at_plane])
    print("Time from first output  = ",time_diff_with_first_output," s")
    print_memory_usage()
    
    dict_GSA["error_fit"       ]           = error_fit
    dict_GSA["time_output"     ]           = time_output
    
    
    np.save("outputs/error_fit_iteration_"  +str(iter).zfill(5)+".npy",np.array(error_fit)   )
    np.save("outputs/time_output_iteration_"+str(iter).zfill(5)+".npy",np.array(time_output) )
    
    return dict_GSA
    
def final_output_dump(dict_image_preprocessing,dict_mesh,dict_GSA):
    
    print()
    print("# Final dump of outputs\n")
    
    # since all the time_output entries are the differences with the time 
    # of the first output, now we can set the time of the first output to zero
    dict_GSA["time_output"][0]   = 0. 
    
    # Save the reconstructed fluence and lineouts along y and z
    dict_GSA = save_reconstructed_fluence_and_lineouts_axis_y_and_z(dict_image_preprocessing,dict_mesh,dict_GSA)
        
    # Save the updated GSA dictionary with the outputs
    np.save('outputs/dict_GSA.npy', dict_GSA    ) 
    
    return dict_GSA 
    
def save_reconstructed_fluence_and_lineouts_axis_y_and_z(dict_image_preprocessing,dict_mesh,dict_GSA):
    
    ### This function computes various lineouts along the y and z axis for all planes,
    ### the calculation depends on the geometry
    ### It is supposed to be called only at the final iteration
    
    if dict_image_preprocessing["geometry"] == "cylindrical":
        dict_GSA     = cylindrical_geometry_save_reconstructed_fluence_and_lineouts_axis_y_and_z(dict_image_preprocessing,dict_mesh,dict_GSA)
    else:
        dict_GSA     = cartesian_geometry_save_reconstructed_fluence_and_lineouts_axis_y_and_z  (dict_image_preprocessing,dict_mesh,dict_GSA)
    
    return dict_GSA

def cylindrical_geometry_save_reconstructed_fluence_and_lineouts_axis_y_and_z(dict_image_preprocessing,dict_mesh,dict_GSA):
    
    ### This function computes fluence lineouts along the y and z axis for all planes
    ### in case the reconstruction is performed with a cylindrical grid.
    ### Lineouts along the y and z axis are computed:
    ### - on the image used for the reconstruction using the cylindrical grid, at theta=0,pi and theta=pi/2,3pi/2;
    ### - on the image used for the reconstruction before change to cylindrical coordinates, using a cartesian grid. 
    ### Computing both allows to check that the results are consistent with the two coordinate systems.
    ### It also stores the reconstructed fluence on the whole grid used for the reconstruction.
    
    plane_x_coordinates     = dict_mesh["plane_x_coordinates"]
    
    Coeffs_LG_pl            = dict_GSA ["dict_mode_basis"    ]["Coeffs_LG_pl"       ]
    r_mesh                  = dict_mesh["r_mesh"             ]
    theta_mesh              = dict_mesh["theta_mesh"         ]
    ntheta                  = np.size(theta_mesh)
    
    if (ntheta%4!=0):
        print("Error: choose a ntheta multiple of 4 to have an accurate calculation of the laser diffraction")
        sys.exit()
    
    fluence_reconstruction_circular = np.zeros(shape=(np.size(plane_x_coordinates),np.size(r_mesh),np.size(theta_mesh)))
    fluence_reconstruction_axis_y   = np.zeros(shape=(np.size(plane_x_coordinates),2*np.size(r_mesh)))
    fluence_reconstruction_axis_z   = np.zeros(shape=(np.size(plane_x_coordinates),2*np.size(r_mesh)))
    
    fluence_exp_circular            = dict_image_preprocessing["fluence_exp_images_circular"]
    fluence_exp_circular_axis_y     = np.zeros(shape=(np.size(plane_x_coordinates),2*np.size(r_mesh)))
    fluence_exp_circular_axis_z     = np.zeros(shape=(np.size(plane_x_coordinates),2*np.size(r_mesh)))
    
    fluence_exp_cartesian           = dict_image_preprocessing["fluence_exp_images_cartesian"]
    ny                              = np.size(fluence_exp_cartesian[0,:,0])
    nz                              = np.size(fluence_exp_cartesian[0,0,:])
    fluence_exp_cartesian_axis_y    = np.zeros(shape=(np.size(plane_x_coordinates),ny))
    fluence_exp_cartesian_axis_z    = np.zeros(shape=(np.size(plane_x_coordinates),nz))
    
    for i_plane in range(np.size(plane_x_coordinates)):
        
        fluence_reconstruction_circular[i_plane,:,:]      = np.square(
                                                                np.absolute(
                                                                   reconstruct_field_at_plane(
                                                                         Coeffs_LG_pl,
                                                                         i_plane,
                                                                         dict_image_preprocessing,
                                                                         dict_mesh,
                                                                         dict_GSA ["dict_mode_basis"]
                                                                             )
                                                                          )
                                                                      )
                                                                  
        fluence_reconstruction_axis_y[i_plane,:]          = np.hstack(
                                                                [np.flipud(fluence_reconstruction_circular[i_plane,:,ntheta//2]),
                                                                fluence_reconstruction_circular[i_plane,:,0]]
                                                                    )
        fluence_reconstruction_axis_z[i_plane,:]          = np.hstack(
                                                                [np.flipud(fluence_reconstruction_circular[i_plane,:,3*ntheta//4]),
                                                                fluence_reconstruction_circular[i_plane,:,ntheta//4]]
                                                                    )
                                                                
        fluence_exp_circular_axis_y[i_plane]              = np.hstack(
                                                                [np.flipud(fluence_exp_circular[i_plane,:,ntheta//2]),
                                                                fluence_exp_circular[i_plane,:,0]]
                                                                    )
        
        fluence_exp_circular_axis_z[i_plane]              = np.hstack(
                                                                [np.flipud(fluence_exp_circular[i_plane,:,3*ntheta//4]),
                                                                fluence_exp_circular[i_plane,:,ntheta//4]]
                                                                    )
                                                                
        if nz%2==0:
            fluence_exp_cartesian_axis_y [i_plane,:]      = 0.5*(fluence_exp_cartesian           [i_plane,:,nz//2-1]+fluence_exp_cartesian           [i_plane,:,nz//2])
        else:
            fluence_exp_cartesian_axis_y [i_plane,:]      =      fluence_exp_cartesian           [i_plane,:,nz//2]
            
        if ny%2==0:                      
            fluence_exp_cartesian_axis_z [i_plane,:]      = 0.5*(fluence_exp_cartesian           [i_plane,ny//2-1,:]+fluence_exp_cartesian           [i_plane,ny//2,:])
        else:
            fluence_exp_cartesian_axis_y [i_plane,:]      =      fluence_exp_cartesian           [i_plane,ny//2,:]
            
            
    dict_GSA["fluence_reconstruction_circular"       ] = fluence_reconstruction_circular
    dict_GSA["fluence_reconstruction_circular_axis_y"] = fluence_reconstruction_axis_y
    dict_GSA["fluence_reconstruction_circular_axis_z"] = fluence_reconstruction_axis_z 
    dict_GSA["fluence_exp_circular_axis_y"           ] = fluence_exp_circular_axis_y
    dict_GSA["fluence_exp_circular_axis_z"           ] = fluence_exp_circular_axis_z
    dict_GSA["fluence_exp_cartesian_axis_y"          ] = fluence_exp_cartesian_axis_y
    dict_GSA["fluence_exp_cartesian_axis_z"          ] = fluence_exp_cartesian_axis_z
    
    return dict_GSA
    
def cartesian_geometry_save_reconstructed_fluence_and_lineouts_axis_y_and_z(dict_image_preprocessing,dict_mesh,dict_GSA):
    
    ### This function computes fluence lineouts along the y and z axis for all planes
    ### in case the reconstruction is performed with a cartesuab grid grid.
    ### Lineouts along the y and z axis are computed:
    ### - on the image used for the reconstruction using the cartesian grid used for the reconstruction.
    ### It also stores the reconstructed fluence on the whole grid used for the reconstruction.
    
    plane_x_coordinates              = dict_mesh  ["plane_x_coordinates"]
    
    if (dict_GSA["use_Mode_Decomposition"]==True):
        Coeffs_HG_mn                 = dict_GSA   ["dict_mode_basis"    ]["Coeffs_HG_mn"     ]
    else:
        ref_plane_index              = dict_image_preprocessing["index_reference_plane"]
        phase_ref_plane              = dict_GSA["phase_ref_plane"]
        E_field_ref_plane            = dict_GSA["field_amplitude_exp "][ref_plane_index,:,:] * np.exp(1j*phase_ref_plane)
    
    fluence_exp_cartesian            = dict_image_preprocessing["fluence_exp_images_cartesian"]
    ny                               = np.size(fluence_exp_cartesian[0,:,0])
    nz                               = np.size(fluence_exp_cartesian[0,0,:])
    
    fluence_reconstruction_cartesian = np.zeros(shape=(np.size(plane_x_coordinates),ny,nz))
    fluence_reconstruction_axis_y    = np.zeros(shape=(np.size(plane_x_coordinates),ny))
    fluence_reconstruction_axis_z    = np.zeros(shape=(np.size(plane_x_coordinates),nz))
    fluence_exp_cartesian_axis_y     = np.zeros(shape=(np.size(plane_x_coordinates),ny))
    fluence_exp_cartesian_axis_z     = np.zeros(shape=(np.size(plane_x_coordinates),nz))
    
    for i_plane in range(np.size(plane_x_coordinates)):
        
        if (dict_GSA["use_Mode_Decomposition"]==True):
            fluence_reconstruction_cartesian[i_plane,:,:]  = np.square(
                                                                np.absolute(
                                                                   reconstruct_field_at_plane(
                                                                         Coeffs_HG_mn,
                                                                         i_plane,
                                                                         dict_image_preprocessing,
                                                                         dict_mesh,
                                                                         dict_GSA ["dict_mode_basis"]
                                                                               )
                                                                           )
                                                                       )
        else:
            if (i_plane==ref_plane_index):
                E_reconstruction = E_field_ref_plane
            else:                
                E_reconstruction = np.fft.ifft2(np.fft.ifftshift(
                                       np.fft.fftshift(np.fft.fft2(E_field_ref_plane)) 
                                           * np.exp(1j*dict_GSA["Fresnel_phase_ov_distance"]*(plane_x_coordinates[i_plane]-plane_x_coordinates[ref_plane_index]))
                                                                ))
                                                                
            fluence_reconstruction_cartesian[i_plane,:,:]  = np.square(np.absolute(E_reconstruction))
            
        if nz%2==0:
            fluence_reconstruction_axis_y[i_plane,:]       = 0.5*(fluence_reconstruction_cartesian[i_plane,:,nz//2-1]+fluence_reconstruction_cartesian[i_plane,:,nz//2])
            fluence_exp_cartesian_axis_y [i_plane,:]       = 0.5*(fluence_exp_cartesian           [i_plane,:,nz//2-1]+fluence_exp_cartesian           [i_plane,:,nz//2])
        else:
            fluence_reconstruction_axis_y[i_plane,:]       =      fluence_reconstruction_cartesian[i_plane,:,nz//2]
            fluence_exp_cartesian_axis_y [i_plane,:]       =      fluence_exp_cartesian           [i_plane,:,nz//2]
            
        if ny%2==0:    
            fluence_reconstruction_axis_z[i_plane,:]       = 0.5*(fluence_reconstruction_cartesian[i_plane,ny//2-1,:]+fluence_reconstruction_cartesian[i_plane,ny//2,:])                                                    
            fluence_exp_cartesian_axis_z [i_plane,:]       = 0.5*(fluence_exp_cartesian           [i_plane,ny//2-1,:]+fluence_exp_cartesian           [i_plane,ny//2,:])
        else:
            fluence_reconstruction_axis_z[i_plane,:]       =      fluence_reconstruction_cartesian[i_plane,ny//2,:]
            fluence_exp_cartesian_axis_z [i_plane,:]       =      fluence_exp_cartesian           [i_plane,ny//2,:]
    
    dict_GSA["fluence_reconstruction_cartesian"       ] = fluence_reconstruction_cartesian
    dict_GSA["fluence_reconstruction_cartesian_axis_y"] = fluence_reconstruction_axis_y
    dict_GSA["fluence_exp_cartesian_axis_y"           ] = fluence_exp_cartesian_axis_y
    dict_GSA["fluence_reconstruction_cartesian_axis_z"] = fluence_reconstruction_axis_z
    dict_GSA["fluence_exp_cartesian_axis_z"           ] = fluence_exp_cartesian_axis_z
    
    
    return dict_GSA
    
def save_Coeffs_MD(Coeffs_MD,dict_image_preprocessing,dict_GSA):
    
    # Save the current coefficients of the mode decomposition
    
    dict_mode_basis = dict_GSA["dict_mode_basis"]
    # Save the coefficients for the Mode Decomposition
    if dict_image_preprocessing["geometry"] == "cylindrical":
        dict_mode_basis["Coeffs_LG_pl"] = Coeffs_MD
    else: 
        dict_mode_basis["Coeffs_HG_mn"] = Coeffs_MD
        
    dict_GSA["dict_mode_basis"] = dict_mode_basis
    
    return dict_GSA
    
def print_memory_usage(): 
    pid = os.getpid()

    # Get the current process object
    current_process = psutil.Process(pid)

    # Get the memory usage in bytes (RSS - Resident Set Size)
    memory_usage = current_process.memory_info().rss

    # Convert memory usage to megabytes (MB)
    memory_usage_mb = memory_usage / (1024 ** 2)
    
    print(f"Memory usage            = {memory_usage_mb:.2f} MB")
    

    