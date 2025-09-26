###### Authors: M. Masckala, F. Massimo 

###### Compute the fluence at custom Cartesian planes 
###### using the Laguerre-Gauss coefficients obtained with the GSA-MD.
###### The number of modes where the reconstruction is truncated can be chosen. 
###### It is assumed that LG_type="helical" 
###### (i.e. LG modes with an angular variation exp(i*l*theta))
######
###### Note: if the distance from the focal plane is too long, the reconstruction will not be accurate,
######       since the paraxial hypothesis is not satisfied.

import numpy as np
import os,sys

############################# Inputs ###########################################

## Parameters to define the plane(s) where the reconstruction is performed
## The following is just an example with two planes
my_plane_x_minus_xfocus  = [-700e-6, 800e-6]  # m, distance of each plane from focal plane
dr                       = 0.1e-6             # m, plane resolution along r       (same at all planes)

nr                       = 1000               # number of grid points along r     (same at all planes)
ntheta                   = 360                # number of grid points along theta (same at all planes)

path_GSA_MD_outputs      = './outputs_cylindrical_GSA_MD'

########### Load the results of the GSA-MD laser field reconstruction ##########

## Load the results of the GSA-MD
## in particular the modes used for the field reconstruction
## they will be defined in our custom planes 

dict_image_preprocessing = np.load(path_GSA_MD_outputs+'/dict_image_preprocessing.npy',allow_pickle=True).item()
dict_mode_basis          = np.load(path_GSA_MD_outputs+'/dict_mode_basis.npy',allow_pickle=True).item()
dict_GSA                 = np.load(path_GSA_MD_outputs+'/dict_GSA.npy',allow_pickle=True).item()
Coeffs_LG_pl             = np.load('./outputs_cylindrical_GSA_MD/Coeffs_MD_iteration_00199.npy')


############# Specify the LG mode base for the reconstruction ##################
lambda_0                 = dict_GSA["lambda_0"] # m, the laser wavelength used in the field reconstruction

# in principle we can retain only a subset of modes for some analysis,
# but in this example we use all the ones used during the mode reconstruction.
# Change the value of these variables in case.
Max_LG_index_p           = dict_mode_basis["Max_LG_index_p"] # maximum radial index
Max_LG_index_l           = dict_mode_basis["Max_LG_index_l"] # maximum azimuthal index

# A sanity check in case the reconstruction is truncated
if (Max_LG_index_p>dict_mode_basis["Max_LG_index_p"]):
    print("ERROR: You cannot use a p index higher than the one used in the reconstruction")
    print("dict_mode_basis['Max_LG_index_p'] = ",dict_mode_basis["Max_LG_index_p"] )
    sys.exit()
if (Max_LG_index_l>dict_mode_basis["Max_LG_index_l"]):
    print("ERROR: You cannot use a p index higher than the one used in the reconstruction")
    print("dict_mode_basis['Max_LG_index_p'] = ",dict_mode_basis["Max_LG_index_p"] )
    sys.exit()

# we define our dictionary for the mode basis, using the same mode parameters
# used by the GSA-MD
my_dict_mode_basis                   = {}
x_focus                              = dict_mode_basis["x_focus" ] # m, focal plane coordinate used in the reconstruction
my_dict_mode_basis["waist_0"       ] = dict_mode_basis["waist_0"]
my_dict_mode_basis["x_focus"       ] = dict_mode_basis["x_focus"]
my_dict_mode_basis["LG_mode_type"  ] = dict_mode_basis["LG_mode_type"  ]

# but we can choose the maximum number of modes that we want to retain 
my_dict_mode_basis["Max_LG_index_p"] = Max_LG_index_p
my_dict_mode_basis["Max_LG_index_l"] = Max_LG_index_l

############# Define custom Cartesian planes for the reconstruction ############

# the mesh will be defined in cylindrical coordinates
my_plane_x_coordinates   = (np.array(my_plane_x_minus_xfocus)+x_focus) # meters
my_r_mesh                = np.arange(nr)*dr+dr/2.                      # meters
my_theta_mesh            = np.arange(ntheta)*2*np.pi/ntheta            # radians

my_dict_mesh             = {"plane_x_coordinates": my_plane_x_coordinates,"r_mesh": my_r_mesh, "theta_mesh": my_theta_mesh} 

##################  Reconstruction of E_field and fluence ######################

# define an array containing the complex field at the custom planes
E_field_cyl              = np.zeros(shape=(np.size(my_plane_x_coordinates),nr,ntheta),dtype=complex) # field unit used in the GSA-MD

# Store the sum_p LG^{pl} field for each p,l value on the custom mesh
# of all transverse planes

#### Note: if the number of requested planes is high,
####       you may want instead to store instead the fields on a plane at time
####       and loop the reconstruction over the planes

from gsa_md.mode_basis.laguerre_gauss_modes import *
my_dict_mode_basis       = store_LG_mode_basis_fields(lambda_0,my_dict_mesh,my_dict_mode_basis,check_total_power_integral=False)

#### Reconstruction of the field using the LG mode decomposition,
#### using only the modes where this series it is truncated

# For the helical LG, as in a FFT, the negative l harmonics are stored after the ones for l>=0, with increasingly negative l.
# l >= 0
E_field_cyl[:,:,:]       = np.sum(
                         Coeffs_LG_pl                            [0:my_dict_mode_basis["Max_LG_index_p"]+1,0:(my_dict_mode_basis["Max_LG_index_l"]+1),np.newaxis,np.newaxis,np.newaxis]
        *                my_dict_mode_basis["LG_fields_at_x_r"]  [0:my_dict_mode_basis["Max_LG_index_p"]+1,0:(my_dict_mode_basis["Max_LG_index_l"]+1),:         ,:         ,np.newaxis]
        *                my_dict_mode_basis["LG_fields_at_theta"][np.newaxis                              ,0:(my_dict_mode_basis["Max_LG_index_l"]+1),np.newaxis,np.newaxis,:         ],
                         axis=(0,1))
# l < 0, Remember that the LG_fields_at_theta for -l is the conjugate of LG_fields_at_theta of l, 
# and that LG_fields_at_x_r only depends on p, |l|
# Use the same l index used when writing the outputs! 
l_min_coeffs             = dict_mode_basis["Max_LG_index_l"]+1
l_max_coeffs             = dict_mode_basis["Max_LG_index_l"]+1+Max_LG_index_l
E_field_cyl[:,:,:]      += np.sum(
                         Coeffs_LG_pl                            [0:my_dict_mode_basis["Max_LG_index_p"]+1,l_min_coeffs:(l_max_coeffs+1)             ,np.newaxis,np.newaxis,np.newaxis]
        *                my_dict_mode_basis["LG_fields_at_x_r"]  [0:my_dict_mode_basis["Max_LG_index_p"]+1,1:(my_dict_mode_basis["Max_LG_index_l"]+1),:         ,:         ,np.newaxis]
        *   np.conjugate(my_dict_mode_basis["LG_fields_at_theta"][np.newaxis                              ,1:(my_dict_mode_basis["Max_LG_index_l"]+1),np.newaxis,np.newaxis,:         ]),
                         axis=(0,1))

# Free some memory
del my_dict_mode_basis["LG_fields_at_x_r"],my_dict_mode_basis["LG_fields_at_theta"]

# Compute fluence at the same planes
fluence_cyl              = np.real(np.square(np.abs(E_field_cyl)))

################## Compute the fluence on a Cartesian grid #####################

# array storing the fluence at the Cartesian coordinates
fluence_cartesian        = np.zeros(shape=(np.size(my_plane_x_coordinates),2*nr,2*nr)) # fluence units

# the transformation cylindrical->Cartesian coordinates 
# will be performed through an interpolation
my_r_mesh                = np.arange(nr)*dr+dr/2.                      # meters
my_theta_mesh            = np.arange(ntheta)*2*np.pi/ntheta            # radians

my_y_mesh                = np.hstack([np.flip(-my_r_mesh),my_r_mesh])  # meters
my_z_mesh                = np.hstack([np.flip(-my_r_mesh),my_r_mesh])  # meters

my_y_meshgrid, my_z_meshgrid = np.meshgrid(my_y_mesh, my_z_mesh, indexing="ij")

# Definition of the radial axis r and theta axis (each has the same shape as my_y_meshgrid and my_z_meshgrid)
my_r_meshgrid            = np.sqrt(np.square(my_y_meshgrid)+np.square(my_z_meshgrid))   
my_theta_meshgrid        = np.arctan2(my_z_meshgrid,my_y_meshgrid)

# Convert (r,theta) to float "indices" for the interpolation 
# and clip them to the array range
r_index                  = (my_r_meshgrid.flatten() - my_r_mesh[0]) / dr
r_index                  = np.clip(r_index, 0, len(my_r_mesh) - 1)

# This way of computing the theta_index avoids discontinuities at theta=0 or theta=pi 
dtheta                   = 2.*np.pi/ntheta
theta_effective          = np.mod(my_theta_meshgrid.flatten() - my_theta_mesh[0],
                                  my_theta_mesh[-1] - my_theta_mesh[0]) + my_theta_mesh[0]
theta_index              = (theta_effective - my_theta_mesh[0]) / (my_theta_mesh[1] - my_theta_mesh[0])

# Now the coordinate indices for interpolation can be defined
coordinate_indices       = np.array([r_index, theta_index])

from scipy.ndimage       import map_coordinates

# Interpolation at each plane
for i_plane in range(np.size(my_plane_x_coordinates)):
    fluence_cartesian[i_plane, :, :] = map_coordinates(
                                           fluence_cyl[i_plane, :, :],
                                           coordinate_indices,
                                           order=2,mode='constant',cval=0.0
                                                      ).reshape(my_r_meshgrid.shape)

# Free some memory
del my_y_meshgrid,my_z_meshgrid,
del my_r_meshgrid,my_theta_meshgrid,
del r_index,theta_index

###############################   Plot  ########################################
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
plt.ion()

# Define the colors for the colormap
colors      = [(1.        , 1.        , 1.       ),   # white
               (0.        , 0.21875   , 0.65625  ),   # blue
               (0.60546875, 0.30859375, 0.5859375),   # purple
               (0.8359375 , 0.0078125 , 0.4375   )]   # pink

# Create the colormap
cmap_name   = 'my_cmap'
my_cmap     = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors, N=256)

# Extract the maximum fluence from Exp results 
vmax        = 0.7*np.amax(fluence_cartesian)

# Limits of the mesh
extent      = [ my_y_mesh.min()/1e-6, my_y_mesh.max()/1e-6, my_z_mesh.min()/1e-6, my_z_mesh.max()/1e-6]

# Plot the fluence at each plane
for i_plane in range(np.size(my_plane_x_coordinates)):
    plt.figure()
    plt.title("x = "+str(my_plane_x_coordinates[i_plane]/1e-6)+" um")
    plt.imshow(fluence_cartesian[i_plane,:,:].T,extent=extent,aspect="equal",origin="lower",vmin=0, vmax=vmax,cmap=my_cmap)
    plt.xlabel("y [um]")
    plt.ylabel("z [um]")
    plt.colorbar(label='|E|^2 [a.u.]')
    