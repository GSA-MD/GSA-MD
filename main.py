###### GSA field reconstruction from fluence images at transverse planes
###### Authors: Francesco Massimo, Ioaquin Moulanier
from gsa_md                      import *
from gsa_md.image_preprocessing  import *
from gsa_md.plot_utilities       import *
from gsa_md.field_reconstruction import *

import os,sys
import matplotlib.pyplot as plt
import numpy as np
plt.ion()

########################## Create input dictionaries ##############################

# You can either 
# 1) call a file that generates the input dictionaries as in this case (use your file path)
# or
# 2) just define here the input dictionaries without needing an additional file

# Find the installation directory of the gsa_md package
gsa_md_dir                  = os.path.dirname(__import__('gsa_md').__file__)
root_dir                    = os.path.dirname(gsa_md_dir)
input_script_path           = root_dir+'/examples/reconstruct_laser_HZDR_dataset/generate_input_files.py'

with open(input_script_path) as file:
    exec(file.read())

dict_image_preprocessing    = np.load('inputs_by_the_user/inputs_image_preprocessing.npy', allow_pickle=True).item()
dict_mesh                   = np.load('inputs_by_the_user/inputs_mesh.npy'               , allow_pickle=True).item()
dict_GSA                    = np.load('inputs_by_the_user/inputs_GSA.npy'                , allow_pickle=True).item()

######################### Image preprocessing ##################################

dict_image_preprocessing,   \
dict_mesh                   = process_input_fluence_images(
                              dict_image_preprocessing, dict_mesh)

################## Perform the GSA-LG-MD reconstruction ########################

dict_GSA                   = field_reconstruction_GSA( 
                              dict_image_preprocessing, dict_mesh, dict_GSA)

#################### Read data from a previous run #############################
# dict_image_preprocessing   = np.load('outputs/dict_image_preprocessing.npy', allow_pickle=True).item()
# dict_mesh                  = np.load('outputs/dict_mesh.npy', allow_pickle=True).item()
# dict_GSA                   = np.load('outputs/dict_GSA.npy', allow_pickle=True).item()
                                                              
############## Plot the evolution of the reconstruction error ##################

fig_error_vs_iteration,     \
fig_error_vs_time           = plot_error_evolution(
                                dict_mesh, dict_GSA)

######################### Plot the diffraction #################################

fig_1e2_width_y, fig_FWHM_y, \
fig_1e2_width_z, fig_FWHM_z = plot_diffraction(
                                dict_image_preprocessing,dict_mesh,dict_GSA)

##################### Plot lineouts of the y and z axis ########################
max_radius_for_plot         = 100e-6
fig_lineouts, axes_lineouts = plot_lineouts(
                                dict_image_preprocessing, dict_mesh,dict_GSA, \
                                max_radius_for_plot,orientation="horizontal")
                                                                
#################### Plot the fluence reconstruction ###########################

# Note that with polar_plot_shading="gouraud" the polar plots for cylindrical coordinates
# become much smoother when a lower number of sampling points is used along theta.
# However, this is only a graphical effect that masks the reduced sampling frequency.
# For quantitative comparisons, it's better to use polar_plot_shading="nearest"
# In Cartesian coordinates, this variable is not used.
plots_reconstructed_fluence(
                           dict_image_preprocessing,dict_mesh,dict_GSA,
                           max_radius_for_plot,relative_vmin=0,relative_vmax=0.7,
                           polar_plot_shading="nearest")
