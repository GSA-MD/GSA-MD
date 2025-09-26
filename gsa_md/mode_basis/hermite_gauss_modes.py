###### Authors: I. Moulanier, Francesco Massimo

###### This module contains the definitions of the HG modes 
###### and the related functions, e.g. mode projection 
###### and reconstruction through mode decomposition

import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
import os, sys
from numba import njit,prange
os.environ['NUMBA_WARNINGS'] = '1' # this can print warnings for race conditions

def HG_m_field_x(lambda_0,m,transverse_mesh,dict_mesh,dict_mode_basis,check_total_power_integral=True):
    
    # This function returns the field of the partial HG mode, i.e. HG^{m}, at plane x on one transverse direction
    # The field of a full HG^{mn} mode at x,y, is HG^{mn}(x,y)=HG^{m}(x)*HG^{n}(y)
    # m                   : generic index of the partial HG mode
    # lambda_0            : carrier wavelength                                         [m]
    # waist_0             : waist at the focal plane of the LG mode                    [m]
    # x_focus             : x coordinate of the focal plane                            [m]
    # plane_x_coordinates : x coordinate of the planes where the field is defined      [m]
    # transverse_mesh     : radial mesh where the field is defined, size n_transverse  [m]
    # check_total_power_integral: when True a check on the total power 
    #    on the reconstruction mesh is done. If the total HG integral
    #    over the mesh is not 1 (e.g. the mesh boundaries are too small), 
    #    the base is not orthonormal and the function computing the scalar
    #    product to find the HG coefficients in the GSA-MD is incorrect.
    #
    # The reference system assumes the HG mode centered in y,z=0,0
    
    
    waist_0               = dict_mode_basis["waist_0"            ]
    x_focus               = dict_mode_basis["x_focus"            ]
    plane_x_coordinates   = dict_mesh      ["plane_x_coordinates"]
    
    if m<0:
        print("Error, negative index for a HG mode")
    
    # the field will be in the same units of the fluence in the fluence images
    HG_field_at_x         = np.zeros((np.size(plane_x_coordinates), np.size(transverse_mesh)), dtype=complex)
    
    # Rayleigh length [m]
    x_R                   = np.pi*waist_0**2/lambda_0 
    
    # Radial resolution [m]
    dtransverse           = np.abs(transverse_mesh[1] - transverse_mesh[0]) 
    
    for i_plane in range(0,np.size(plane_x_coordinates)):
        
        x                 = plane_x_coordinates [i_plane] - x_focus  # Axial distance from the focal plane
        
        # Avoid division by zero in the computation
        if x == 0:
            w                 = waist_0
            R                 = np.inf
            Gouy_phase        = np.exp(-1j*0.)
            curved_phase      = np.exp(1j*0.)
        else:
            w                 = waist_0 * np.sqrt(1 + (x /x_R)**2)
            R                 = x * (1 + (x_R/x)**2) 
            Gouy_phase        = np.exp(-1j   * (m+1/2) * np.arctan2(x,x_R) )
            curved_phase      = np.exp(1j*(2.*np.pi/lambda_0)*transverse_mesh**2 / (2*R))
        
        # Compute the complex field using the partial HG mode formula 
        prefactor             = (2/np.pi)**(1/4.)*1./np.sqrt( 2**m * sp.gamma(m+1) * w )  # Check this factor!
        
        # remember that gamma(n+1) = n! when n is an integer
        transverse_component  = sp.eval_hermite(m, np.sqrt(2) * transverse_mesh / w )     \
                              * np.exp(-transverse_mesh**2 / w**2)                        \
                              
        field                 = prefactor * transverse_component * Gouy_phase * curved_phase
        
        integral_at_plane     = np.sum( np.absolute(field)**2 ) * dtransverse 
        
        # the total integral in circular coordinates of the |field|^2 at each plane should be equal to 1,
        # i.e. int (-infty->infty) int (-infty->infty) |field|^2 dx dy = 1.
        # The HG^m{y} and HG^m{z} parts are identical, so their integral must be 1.
        # This check can be skipped if check_total_power_integral = False, 
        # to enable the definition of HG on arbitrary meshes
        if check_total_power_integral==True:
            if (integral_at_plane>1.05) or (integral_at_plane<0.95):
                print("ERROR: total power at plane ",i_plane," in the x,y (or x,z) part is ",integral_at_plane, ", too far from 1.")
                print("If it is lower than 1, try to choose a larger mesh or a smaller waist")
                sys.exit()
        
        # normalize to have the total power equal to 1
        HG_field_at_x[i_plane, :] = field/np.sqrt(integral_at_plane)
        
    return HG_field_at_x
 
def store_HG_mode_basis_fields(lambda_0,dict_mesh,dict_mode_basis,check_total_power_integral=True):

    #### Create and store the arrays needed to define the field of the partial HG modes HG^m(x,...).
    #### The field of a full HG mode HG^{mn}(x,y,z) will be given by HG^m(x,y)*HG^n(x,z).
    
    # the total number of transverse indices on the y direction will be Max_HG_index_m+1
    # the total number of transverse indices on the z direction will be Max_HG_index_n+1
    Max_HG_index_m     = dict_mode_basis  ["Max_HG_index_m"     ]
    Max_HG_index_n     = dict_mode_basis  ["Max_HG_index_n"     ]
    y_mesh             = dict_mesh        ["y_mesh"             ]
    z_mesh             = dict_mesh        ["z_mesh"             ]
    
    number_of_planes   = np.size(dict_mesh["plane_x_coordinates"])
    ny                 = np.size(y_mesh                          )
    nz                 = np.size(z_mesh                          )


    # this is the part dependent on x and y
    HG_m_fields_at_x     = np.zeros((Max_HG_index_m+1,number_of_planes,ny), dtype=complex)
    HG_n_fields_at_x     = np.zeros((Max_HG_index_n+1,number_of_planes,nz), dtype=complex)
    
    for m in range(0, Max_HG_index_m+1):
        print("Storing HG_m_fields_at_x for m = ",m)
        HG_m_fields_at_x[m,:,:] = HG_m_field_x(lambda_0,m,y_mesh,dict_mesh,dict_mode_basis,check_total_power_integral=check_total_power_integral)
    print()
    for n in range(0, Max_HG_index_n+1):
        print("Storing HG_n_fields_at_x for n = ",n)
        HG_n_fields_at_x[n,:,:] = HG_m_field_x(lambda_0,n,z_mesh,dict_mesh,dict_mode_basis,check_total_power_integral=check_total_power_integral)
    print()
    
    
    # Store the fields used by the GSA-MD
    dict_mode_basis["HG_m_fields_at_x"] = HG_m_fields_at_x
    dict_mode_basis["HG_n_fields_at_x"] = HG_n_fields_at_x
    
    return dict_mode_basis

def plot_real_part_HG_mode(i_plane,m,n,dict_mesh,dict_mode_basis):
    # Plot the real part of the field of the HG^mn mode,
    # for a given plane of index i_plane, in Cartesian coordinates.

    if i_plane < 0 or i_plane >= np.size(dict_mesh["plane_x_coordinates"]):
        raise ValueError("Plane index out of range.")
        
    field_HG_m            = dict_mode_basis["HG_m_fields_at_x"][m,i_plane,:]
    field_HG_n            = dict_mode_basis["HG_n_fields_at_x"][n,i_plane,:]
    field                 = field_HG_m[:,np.newaxis]*field_HG_n[np.newaxis,:]
    
    y_meshgrid,z_meshgrid = np.meshgrid(dict_mesh["y_mesh"],dict_mesh["z_mesh"],indexing='ij')

    # Cartesian plot
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.grid(False)  # Disable the grid explicitly
    plt.pcolormesh(y_meshgrid,z_meshgrid,np.real(field[0,:,:]), shading='auto', cmap='afmhot_r')
    plt.title('Real part- Plane '+str(i_plane)+', m,n = '+str(m)+', '+str(n))
    plt.xlabel('y (um)')
    plt.ylabel('z (um)')
    plt.tight_layout()
    plt.show()
 
def project_field_on_HG_modes(field,i_plane,dict_image_preprocessing,dict_mesh,dict_mode_basis):
    
    # This function computes C_mn as the integral C_mn = < field(x,y), HG^mn(i_plane,y,z) > at one plane
    # where field is a complex field defined on x,y,z, on a transverse_mesh with resolution dy,dz
    
    energy_reference_plane = dict_image_preprocessing["energy_reference_plane"]
    Max_HG_index_m         = dict_mode_basis         ["Max_HG_index_m"        ]
    Max_HG_index_n         = dict_mode_basis         ["Max_HG_index_n"        ]
    HG_m_fields_at_x       = dict_mode_basis         ["HG_m_fields_at_x"      ][:,i_plane,:]
    HG_n_fields_at_x       = dict_mode_basis         ["HG_n_fields_at_x"      ][:,i_plane,:]
    y_mesh                 = dict_mesh               ["y_mesh"                ]
    z_mesh                 = dict_mesh               ["z_mesh"                ]
    
    dy                     = y_mesh[1]-y_mesh[0]
    dz                     = z_mesh[1]-z_mesh[0]
    
    Coeffs_HG_mn           = np.zeros(shape=(Max_HG_index_m+1,Max_HG_index_n+1),dtype=complex)
    
    #Coeffs_HG_mn           = compute_HG_coefficients_no_numba(Coeffs_HG_mn,energy_reference_plane,field,HG_m_fields_at_x,HG_n_fields_at_x,dy,dz)
    Coeffs_HG_mn           = compute_HG_coefficients_numba(Coeffs_HG_mn,energy_reference_plane,field,HG_m_fields_at_x,HG_n_fields_at_x,dy,dz)
    
    return Coeffs_HG_mn


def compute_HG_coefficients_no_numba(Coeffs_HG_mn,energy_reference_plane,field,HG_m_fields_at_x,HG_n_fields_at_x,dy,dz):
    integrand              = field[np.newaxis,np.newaxis,:,:] * np.conjugate(HG_m_fields_at_x[:,np.newaxis,:,np.newaxis])  
    integrand              = integrand * np.conjugate(HG_n_fields_at_x[np.newaxis,:,np.newaxis,:])       
    Coeffs_HG_mn[:,:]      = np.sum( integrand[:,:,:,:] * dy * dz ,axis=(2,3)) 
    Coeffs_HG_mn           = Coeffs_HG_mn * np.sqrt(energy_reference_plane) / np.sqrt(np.sum(np.absolute(Coeffs_HG_mn)**2))  
    return Coeffs_HG_mn

@njit(parallel=True)
def compute_HG_coefficients_numba(Coeffs_HG_mn,energy_reference_plane,field,HG_m_fields_at_x,HG_n_fields_at_x,dy,dz):   
    # using prange only for independent loop iterations to avoid race conditions
    for m in prange(HG_m_fields_at_x.shape[0]):
        for n in prange(HG_n_fields_at_x.shape[0]):
            total_integral_m_n = 0.0+0.0j
            for i_y in range(HG_m_fields_at_x.shape[1]):
                for i_z  in range(HG_n_fields_at_x.shape[1]):
                    integral_sum_at_y_z = field[i_y,i_z]*(HG_m_fields_at_x[m,i_y]*HG_m_fields_at_x[n,i_z]).conjugate()*dy*dz
                    total_integral_m_n += integral_sum_at_y_z
            Coeffs_HG_mn[m,n] = total_integral_m_n
    
    # Normalize the coefficients
    mode_energy_sum  = np.sum(np.square(np.abs(Coeffs_HG_mn)))
    Coeffs_HG_mn *=np.sqrt(energy_reference_plane/mode_energy_sum)
    
    return Coeffs_HG_mn

def HG_reconstruct_field_at_plane(Coeffs_HG_mn,i_plane,dict_mode_basis):
    
    # This function computes E(i_plane,y,z) = sum_{mn} [ C_mn * HG^{mn}(i_plane,y,z) ]
    
    HG_m_fields_at_x       = dict_mode_basis["HG_m_fields_at_x"][:,i_plane,:]
    HG_n_fields_at_x       = dict_mode_basis["HG_n_fields_at_x"][:,i_plane,:]
    
    E = np.zeros(shape=(HG_m_fields_at_x.shape[1],HG_n_fields_at_x.shape[1]), dtype=complex)
    #E = reconstruct_HG_E_no_numba(E,Coeffs_HG_mn,HG_m_fields_at_x,HG_n_fields_at_x)
    E = reconstruct_HG_E_numba(E,Coeffs_HG_mn,HG_m_fields_at_x,HG_n_fields_at_x)
    
    return E

def reconstruct_HG_E_no_numba(E,Coeffs_HG_mn,HG_m_fields_at_x,HG_n_fields_at_x): 
    E = np.sum(Coeffs_HG_mn[:,:,np.newaxis,np.newaxis]*HG_m_fields_at_x[:,np.newaxis,:,np.newaxis]*HG_n_fields_at_x[np.newaxis,:,np.newaxis,:],axis=(0,1))
    return E
    
@njit(parallel=True)
def reconstruct_HG_E_numba(E,Coeffs_HG_mn,HG_m_fields_at_x,HG_n_fields_at_x): 
    # using prange only for independent loop iterations to avoid race conditions
    for i_y in prange(HG_m_fields_at_x.shape[1]):
        for i_z  in prange(HG_n_fields_at_x.shape[1]):
                sum_E = 0.0+0.0j
                for m in range(HG_m_fields_at_x.shape[0]):
                    for n in range(HG_n_fields_at_x.shape[0]):
                        sum_modes = Coeffs_HG_mn[m,n]*HG_m_fields_at_x[m,i_y]*HG_n_fields_at_x[n,i_z]
                        sum_E    += sum_modes    
                E[i_y,i_z] = sum_E 
                
    return E

