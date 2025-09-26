###### Authors: I. Moulanier, F. Massimo

###### This file provides creates and saves these dictionaries:
###### - dict_preprocessing,
###### - dict_mesh,
###### - dict_GSA (which contains the dictionary dict_mode_basis if Mode Decomposition is used).
###### These dictionaries can be used as inputs for the reconstruction functions or just the plot functions
###### If some elemenst are missing, default ones will be provided. 

import numpy as np
import os,sys

########################### Load the images and provide their x position ###########################

# Define the file paths of the input fluence images, they must be square and defined on a Cartesian mesh

# Import the fluence images, provide their resolution and position on the propagation axis
# They must be square images
# The order of the list is important

gsa_md_dir          = os.path.dirname(__import__('gsa_md').__file__)
root_dir            = os.path.dirname(gsa_md_dir)
path_fluence_images = root_dir+"/examples/reconstruct_laser_HZDR_dataset/HZDR_dataset" 
image_file_paths    = [
   path_fluence_images+"/average_fluence_profile_at_plane_500.0_um.npy",
   path_fluence_images+"/average_fluence_profile_at_plane_1500.0_um.npy",
   path_fluence_images+"/average_fluence_profile_at_plane_2500.0_um.npy",
   path_fluence_images+"/average_fluence_profile_at_plane_3500.0_um.npy",
   path_fluence_images+"/average_fluence_profile_at_plane_4500.0_um.npy"
]

# Load the images into a numpy array with shape (number_of_images,size_axis_0,size_axis_1)
fluence_images           = [np.load(file) for file in image_file_paths]
input_fluence_images     = np.array(fluence_images)

# Define plane_x_coordinates and number_of_planes representing the x coordinates and number of measurement planes of the fluence images
plane_x_coordinates      = np.array([-2e-3, -1e-3, 0e-3, 1e-3, 2e-3]) # [m] x coordinates of the image planes

# physical size of each pixel
length_per_pixel         = 0.126e-6 # m

# A fluence image center must be defined at each transverse plane.
# This is used in postprocessing to center the Cartesian/cylindrical grid of each plane.

# If Mode Decomposition is used, these centers are also the mode centers.
# In that case, their choice can be particularly critical for the field reconstruction.

# If the mode centers of the images for the LG/HG decomposition 
# (i.e. yielding the minimum reconstruction error) are not known,
# just leave the following lists empty. 

# Otherwise, use an array with the same length of plane_x_coordinates, 
# e.g. np.array([0,0,0]) for 3 planes.

y0_at_planes             = [] # m
z0_at_planes             = [] # m

# Index of the reference plane for the energy calculation
# The total fluence of each image will be normalized to the total fluence 
# of the image corresponding to this index.
index_reference_plane    = 2 

########################### Other Inputs ###########################
# "cartesian" or "cylindrical"
# If Mode decomposition will be used,
# in "cartesian" coordinates a HG mode reconstruction will be performed in cartesian coordinates;
# in "cylindrical" coordinates a LG mode reconstruction will be performed.
# A reconstruction not using the Mode decomposition is available only for "cartesian".
geometry                 = "cartesian" #"cylindrical" 

# Dictionary for preprocessing
dict_image_preprocessing = {
                            "geometry"                 : geometry,
                            "input_fluence_images"     : input_fluence_images,
                            "length_per_pixel"         : length_per_pixel,
                            "y0_at_planes"             : y0_at_planes,
                            "z0_at_planes"             : z0_at_planes,
                            "threshold"                : 5e-4,          # noise threshold for postprocessing, [fluence units] 1e-4
                            "r_max_coordinate_change"  : 150e-6,        # [m] Radius above which fluence is set to zero in the preprocessing 
                            "median_filter_size"       : 10,            # size in pixel of the 2D filter applied to the input images
                            "index_reference_plane"    : index_reference_plane,
                           }       


# Dictionary for the mesh of the preprocessed images used by the GSA-MD.
# The mode basis is defined on this mesh
dict_mesh                = {
                            "plane_x_coordinates"      : plane_x_coordinates
                           }
if geometry=="cylindrical":
    dict_mesh              ["nr_converted_image"]      = 1600 
    dict_mesh              ["ntheta"            ]      = 720 # Must be a multiple of 4! These points will sample the angle [0,2pi)
else:
    dict_mesh              ["ny_converted_image"]      = 1600 
    dict_mesh              ["nz_converted_image"]      = 1600 


# Parameters for the GSA reconstruction

dict_GSA                 = {
                           "N_iterations"               : 200,    # number of iterations of the GSA reconstruction
                           "iterations_between_outputs" : 30, 
                           "use_Mode_Decomposition"     : True,
                           "lambda_0"                   : 0.8e-6, # [m] Carrier wavelength
                           "use_initial_Gaussian_phase" : False,  # if True, the first iteration will use a Gaussian phase at the first plane
                           } 

# When using the classic GSA without mode decomposition, using an initial gaussian phase can make a difference                       
if dict_GSA["use_Mode_Decomposition"]==False:
    dict_GSA["use_initial_Gaussian_phase"] = True


# waist of the fundamental mode
waist_0                  = 30e-6 # [m]

# in case a Mode Decompostion is used for the GSA, define a dict_mode_basis inside dict_GSA
if dict_GSA["use_Mode_Decomposition"] == True:
    
    # Dictionary for the LG/HG mode basis
    dict_mode_basis      = {
                            "waist_0"                   : waist_0, # [m] fundamental mode waist at focal plane 
                            "x_focus"                   : 0.       # [m] Position of the focal plane    
                            }    

    if geometry=="cylindrical":
        #### Parameters for the Laguerre-Gauss modes
        # The total number of radial indices will be Max_LG_index_p+1
        dict_mode_basis       ["Max_LG_index_p"]        = 14 
        # The total number of azimuthal indices will be 2*Max_LG_index_l+1
        dict_mode_basis       ["Max_LG_index_l"]        = 3 
        
        # "helical" LG modes have a theta dependence of type exp(i*l*theta) [l=-"Max_LG_index_l",0,"Max_LG_index_l"]
        # "sinusoidal" LG modes have a theta dependence of type cos(l*theta) and sin(l*theta) [l=0,"Max_LG_index_l"]
        # No differences in the results have been found so far changing this choice,
        # it is just for the user's preference depending on the LG type used 
        # when importing the results of the reconstruction run.
        dict_mode_basis       ["LG_mode_type"  ]        = "helical" # "sinusoidal"
    else:
        #### Parameters for the Hermite-Gauss modes
        # the total number of transverse indices on the y direction will be Max_HG_index_m+1
        dict_mode_basis       ["Max_HG_index_m"]        = 4 
        # the total number of transverse indices on the z direction will be Max_HG_index_n+1
        dict_mode_basis       ["Max_HG_index_n"]        = 4
    
    dict_GSA["dict_mode_basis"]                         = dict_mode_basis

else:
    dict_GSA["waist_gaussian_beam_phase"]               = waist_0        # [m] waist of an initial Gaussian phase



########################### Save the inputs provided by the user ###########################
os.mkdir("inputs_by_the_user")
np.save ("inputs_by_the_user/inputs_image_preprocessing.npy", dict_image_preprocessing )
np.save ("inputs_by_the_user/inputs_mesh.npy"               , dict_mesh                )
np.save ("inputs_by_the_user/inputs_GSA.npy"                , dict_GSA                 )

