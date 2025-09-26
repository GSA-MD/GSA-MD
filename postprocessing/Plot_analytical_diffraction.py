###### Authors: O. Khomyshyn, F. Massimo

###### This script plots the diffraction expected 
###### from the analytical propagation of the LG/HG modes 
###### The experimental and GSA-MD-reconstructed width evolution 
####### of the laser are plotted as well

import os,sys
from gsa_md.plot_utilities.plot_functions  import *
import matplotlib.pyplot as plt
import numpy as np
plt.ion()

################################ Inputs ########################################

# path of the outputs of the GSA-MD reconstruction
path_GSA_MD_outputs      = 'outputs_cartesian_GSA_MD' #'outputs_cylindrical_GSA_MD'

# name of the file with the coefficients to use for the mode reconstruction
mode_coefficients_file   = '/Coeffs_MD_iteration_00199.npy'

# number of planes for the analytical propagation, 
# between the minimum and maximum coordinates of the planes 
# of the images used for the GSA-MD reconstruction 
n_planes                 = 100

###################### Read data from a previous run ###########################
dict_image_preprocessing = np.load(path_GSA_MD_outputs+'/dict_image_preprocessing.npy', allow_pickle=True).item()
dict_mesh                = np.load(path_GSA_MD_outputs+'/dict_mesh.npy', allow_pickle=True).item()
dict_mode_basis          = np.load(path_GSA_MD_outputs+'/dict_mode_basis.npy', allow_pickle=True).item()
dict_GSA                 = np.load(path_GSA_MD_outputs+'/dict_GSA.npy', allow_pickle=True).item()

                                                             
### Plot the experimental and reconstructed diffraction at the input planes ####
fig_1e2_width_y, fig_FWHM_y, \
fig_1e2_width_z, fig_FWHM_z     = plot_diffraction(
                                           dict_image_preprocessing,
                                           dict_mesh,
                                           dict_GSA,
                                           plot_cylindrical_experimental_fluence = False,
                                           only_markers = True
                                           )

plt.close(fig_FWHM_y.number)
plt.close(fig_FWHM_z.number)

############################ Auxiliary function ################################
def compute_width_at_level_times_peak_value(x,y,level=0.5):
    
    ##### Given a curve y=f(x) with a peak y_peak, this function computes the 
    ##### width (in x units) of the peak, defined as the difference between the first x positions around the peak which
    ##### have y = level*y_peak.
    ##### when level = 0.5 you are computing the FWHM;
    ##### when level = 1 / math.e ** 2, you are computing the 1/e^2 width
    print()
    x_array_midpoints = (x[1:]+x[:-1])/2. # array containing the midpoints in the x array
    index_y_peak      = np.where(y==y.max())[0][0]
    x_peak            = x_array_midpoints[index_y_peak]
    y_peak            = y[index_y_peak]
    
    # Iterative determination of FWHM : from the x_peak,
    # the x where y = level*y_peak are searched, on the left and on the right
    #print("Peak at ", x_peak, ", max =",y_peak)
    left_peak_index   = index_y_peak
    right_peak_index  = index_y_peak
    while (y[left_peak_index]>y_peak*level and left_peak_index > 1):
        left_peak_index  = left_peak_index - 1 
    while (y[right_peak_index]>y_peak*level and right_peak_index < np.size(y)-2):
        right_peak_index = right_peak_index +1
        
    width             = x_array_midpoints[right_peak_index] - x_array_midpoints[left_peak_index]

    #print("Peak at ", x_peak, ", max y =",y_peak,", width ", width )
    #print("level edges = ",x_array_midpoints[left_peak_index],", ",x_array_midpoints[right_peak_index])
    
    # returns the left and right borders of the peak, the peak width and the position of the peak
    return width
    
######### Define the mesh for the analytical propagation of the  modes #########

# define the positions of the planes 
# where the analytical reconstruction will be performed
x_propagation_planes      = np.linspace(
                                  np.amin(dict_mesh["plane_x_coordinates"]),
                                  np.amax(dict_mesh["plane_x_coordinates"]),
                                  num=n_planes,
                                  )
# start defining the 1D mesh for the propagation of the modes,
# for both the y and z axis, and the 1D arrays that store the field on the axis

if dict_image_preprocessing["geometry"] == "cylindrical":
    # LG coefficients for the reconstruction from a previous run
    Coeffs_LG_pl          = np.load(path_GSA_MD_outputs+mode_coefficients_file)
    # 1D meshes for the propagation of the modes
    r_mesh                = np.arange(dict_mesh["nr_converted_image"])*dict_image_preprocessing["length_per_pixel"  ]
    y_mesh                = np.hstack([-np.flip(r_mesh),r_mesh])
    z_mesh                = np.hstack([-np.flip(r_mesh),r_mesh])
    r_mesh_axis_y         = np.abs(y_mesh)
    r_mesh_axis_z         = np.abs(z_mesh)
    theta_mesh_axis_y     = np.array([np.pi    if y>0 else         0 for y in y_mesh])
    theta_mesh_axis_z     = np.array([np.pi/2. if z>0 else -np.pi/2. for z in z_mesh])
    dict_mesh_axis_y      = {"r_mesh"     : r_mesh_axis_y,   \
                             "theta_mesh" : theta_mesh_axis_y}
    dict_mesh_axis_z      = {"r_mesh"     : r_mesh_axis_z,   \
                             "theta_mesh" : theta_mesh_axis_z}                     
else: # "cartesian"
    # HG coefficients for the reconstruction from a previous run
    Coeffs_HG_mn          = np.load(path_GSA_MD_outputs+mode_coefficients_file)
    # 1D meshes for the propagation of the modes
    y_mesh                = np.arange(dict_mesh["ny_converted_image"])*dict_image_preprocessing["length_per_pixel"]
    y_mesh               -= y_mesh.max()/2.
    z_mesh                = np.arange(dict_mesh["nz_converted_image"])*dict_image_preprocessing["length_per_pixel"]
    z_mesh               -= z_mesh.max()/2.
    dict_mesh_propagation = {"y_mesh"     : y_mesh, \
                             "z_mesh"     : z_mesh  }
                             
# 1D arrays that store the field on the axis
Ey_axis_y             = np.zeros_like(y_mesh,dtype=complex)
Ey_axis_z             = np.zeros_like(z_mesh,dtype=complex)  

##################### Analytical propagation of the  modes #####################

# the 1/e2 widths along the axes y and z
one_ov_e2_width_y_theory  = []       
one_ov_e2_width_z_theory  = []

# For each transverse plane, reconstruct the fluence on the y and z axis and compute the 1/e2 width
for i_plane in range(np.size(x_propagation_planes)):
    
    print("Computing 1/e2 widths at plane ",i_plane+1," of ",np.size(x_propagation_planes))
    
    # reset the fields
    Ey_axis_y             = np.zeros(shape=(np.size(y_mesh)),dtype=complex)
    Ey_axis_z             = np.zeros(shape=(np.size(z_mesh)),dtype=complex) 
    
    if dict_image_preprocessing["geometry"] == "cylindrical":
        
        # change x coordinate to the one of the plane to regenerate the mode basis
        dict_mesh_axis_y["plane_x_coordinates"] = np.array([x_propagation_planes[i_plane]])
        dict_mesh_axis_z["plane_x_coordinates"] = np.array([x_propagation_planes[i_plane]]) 
        
        from gsa_md.mode_basis.laguerre_gauss_modes import *

        # it is assumed that helical LG modes are used,
        # i.e. with an azimuthal variation exp(i*l*theta)
        for l in range(-dict_mode_basis["Max_LG_index_l"], dict_mode_basis["Max_LG_index_l"] + 1):
            # negative l indices are stored with the FFT convention for negative frequencies
            l_index = l if l >= 0 else dict_mode_basis["Max_LG_index_l"] + abs(l)
            # sum the x,r part of the modes with the same l index and different radial p indices
            Ey_axis_y_radial = np.zeros(shape=(1,np.size(y_mesh)),dtype=complex)
            Ey_axis_z_radial = np.zeros(shape=(1,np.size(z_mesh)),dtype=complex)
            for p in range(0,dict_mode_basis["Max_LG_index_p"]+1):
                Ey_axis_y_radial += (Coeffs_LG_pl[p,l_index] 
                                     * LG_pl_field_x_r(
                                                       dict_GSA["lambda_0"],
                                                       p,l,dict_mesh_axis_y,
                                                       dict_mode_basis,
                                                       check_total_power_integral=False
                                                       )) 
                Ey_axis_z_radial += (Coeffs_LG_pl[p,l_index] 
                                     * LG_pl_field_x_r(
                                                       dict_GSA["lambda_0"],
                                                       p,l,dict_mesh_axis_z,
                                                       dict_mode_basis,
                                                       check_total_power_integral=False
                                                       )) 
            # then multiply this radial contribution by the azimuthal variation 
            # and sum to the total field
            Ey_axis_y += (Ey_axis_y_radial * LG_pl_field_theta(l,dict_mesh_axis_y,dict_mode_basis,check_total_power_integral=False)).flatten()
            Ey_axis_z += (Ey_axis_z_radial * LG_pl_field_theta(l,dict_mesh_axis_z,dict_mode_basis,check_total_power_integral=False)).flatten()
            
    else: # "cartesian"
    
        # change x coordinate to the one of the plane to regenerate the mode basis
        dict_mesh_propagation["plane_x_coordinates"] = np.array([x_propagation_planes[i_plane]])
        
        from gsa_md.mode_basis.hermite_gauss_modes import *
        
        # recreate the HG fields on the selected mesh
        
        HG_m_y = np.zeros(shape=((dict_mode_basis["Max_HG_index_m"]+1),np.size(y_mesh)),dtype=complex)
        for m in range(dict_mode_basis["Max_HG_index_m"] + 1):
            HG_m_y[m,:] = HG_m_field_x(dict_GSA["lambda_0"],m,y_mesh,dict_mesh_propagation,dict_mode_basis,check_total_power_integral=False)[:,:]
            
        HG_n_z = np.zeros(shape=((dict_mode_basis["Max_HG_index_n"]+1),np.size(z_mesh)),dtype=complex)
        for n in range(dict_mode_basis["Max_HG_index_n"] + 1):
            HG_n_z[n,:] = HG_m_field_x(dict_GSA["lambda_0"],n,z_mesh,dict_mesh_propagation,dict_mode_basis,check_total_power_integral=False)[:,:]
        
        # reconstruct the entire field on the y and z axis
        for m in range(dict_mode_basis["Max_HG_index_m"] + 1):
            for n in range(dict_mode_basis["Max_HG_index_n"] + 1):
                
                # axis y
                if np.size(z_mesh)%2!=0: # the z_mesh passes through zero
                    Ey_axis_y += (Coeffs_HG_mn[m,n]*HG_m_y[m,:]*HG_n_z[n,np.size(z_mesh)//2]).T
                else: # the z_mesh does not pass through zero and an interpolation is done
                    Ey_axis_y += (Coeffs_HG_mn[m,n]*HG_m_y[m,:]*0.5*(HG_n_z[n,np.size(z_mesh)//2]+HG_n_z[n,np.size(z_mesh)//2+1])).T
                # axis z
                if np.size(z_mesh)%2!=0: # the y_mesh passes through zero
                    Ey_axis_z += (Coeffs_HG_mn[m,n]*HG_n_z[n,:]*HG_m_y[m,np.size(y_mesh)//2]).T
                else: # the y_mesh does not pass through zero and an interpolation is done
                    Ey_axis_z += (Coeffs_HG_mn[m,n]*HG_n_z[n,:]*0.5*(HG_m_y[m,np.size(y_mesh)//2]+HG_m_y[m,np.size(y_mesh)//2+1])).T     

    # compute 1/e2 width along y and z of the fluence
    one_ov_e2_width_y_theory.append(compute_width_at_level_times_peak_value(y_mesh,np.square(np.abs(Ey_axis_y)),level=1./np.e**2))
    one_ov_e2_width_z_theory.append(compute_width_at_level_times_peak_value(z_mesh,np.square(np.abs(Ey_axis_z)),level=1./np.e**2))

one_ov_e2_width_y_theory = np.array(one_ov_e2_width_y_theory)
one_ov_e2_width_z_theory = np.array(one_ov_e2_width_z_theory)

######################## Plot the analytical propagation #######################

if dict_image_preprocessing["geometry"] == "cylindrical":
    color = "green"
else: # "cartesian"
    color = "red"

# y axis
plt.figure(fig_1e2_width_y.number)
plt.plot((x_propagation_planes-dict_mode_basis["x_focus"])/1e-6,one_ov_e2_width_y_theory/1e-6,label="y, LG analytical propagation",c=color,linestyle="--")
plt.legend()
plt.ylim(0,1.2*np.maximum(np.amax(one_ov_e2_width_y_theory),np.amax(one_ov_e2_width_z_theory))/1e-6)


# z axis
plt.figure(fig_1e2_width_z.number)
plt.plot((x_propagation_planes-dict_mode_basis["x_focus"])/1e-6,one_ov_e2_width_z_theory/1e-6,label="z, LG analytical propagation",c=color,linestyle="--")
plt.legend()
plt.ylim(0,1.2*np.maximum(np.amax(one_ov_e2_width_y_theory),np.amax(one_ov_e2_width_z_theory))/1e-6)





