###### Authors: Francesco Massimo, I. Moulanier

###### This module contains the definitions of the LG modes 
###### and the related functions, e.g. mode projection 
###### and reconstruction through mode decomposition

import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
import os, sys
import numba
from numba import njit,prange
os.environ['NUMBA_WARNINGS'] = '1' # this can print warnings for race conditions

plt.ion()

def LG_pl_field_x_r(lambda_0,p,l,dict_mesh,dict_mode_basis,check_total_power_integral=True):
    
    # This function returns the partial LG^{pl} field at plane x, distance r (i.e. only the part not dependent on theta)
    # Note that the result is identical for l and -l
    # p , l    : radial and azimuthal index of the LG mode
    # lambda_0 : carrier wavelength                                    [m]
    # waist_0  : waist at the focal plane of the LG mode               [m]
    # x_focus  : x coordinate of the focal plane                       [m]
    # plane_x_coordinates : x coordinate of the planes where the field is defined [m]
    # r_mesh   : radial mesh where the field is defined, size nr       [m]
    # check_total_power_integral: when True a check on the total power 
    #    on the reconstruction mesh is done. If the total LG integral
    #    over the mesh is not 1 (e.g. the mesh r boundary is too small), 
    #    the base is not orthonormal and the function computing the scalar
    #    product to find the LG coefficients is incorrect.
    
    waist_0               = dict_mode_basis["waist_0"            ]
    x_focus               = dict_mode_basis["x_focus"            ]
    plane_x_coordinates   = dict_mesh      ["plane_x_coordinates"]
    r_mesh                = dict_mesh      ["r_mesh"             ]
    
    # the field will be in the same units of the fluence in the fluence images
    LG_field_at_x_r       = np.zeros((np.size(plane_x_coordinates), np.size(r_mesh)), dtype=complex)
    
    # Rayleigh length [m]
    x_R                   = np.pi*waist_0**2/lambda_0 
    
    # Radial resolution [m]
    dr                    = np.abs(r_mesh[1] - r_mesh[0]) 
    
    for i_plane in range(0,np.size(plane_x_coordinates)):
        
        x                 = plane_x_coordinates [i_plane] - x_focus  # Axial distance from the focal plane
        
        # Avoid division by zero in the computation
        if x == 0:
            w             = waist_0
            R             = np.inf
            Gouy_phase    = np.exp(-1j*0.)
            curved_phase  = np.exp(1j*0.)
        else:
            w             = waist_0 * np.sqrt(1. + (x /x_R)**2)
            R             = x * (1. + (x_R/x)**2)
            Gouy_phase    = np.exp(-1j   * (2*p+abs(l)+1) * np.arctan2(x,x_R) )
            curved_phase  = np.exp(1j*(2.*np.pi/lambda_0)*r_mesh**2 / (2*R)) 
        
        # Compute the complex field using the LG mode formula (only the part dependent on x and r)
        prefactor         = 1./w* np.sqrt( 2 * sp.gamma(p+1) / (np.pi * (sp.gamma(p+ abs(l)+1)))   )
        
        if (dict_mode_basis["LG_mode_type"] == "sinusoidal"):
            if (l==0):
                prefactor *= 1.
            else:
                prefactor *= np.sqrt(2.)
        
        # remember that gamma(n+1) = n! when n is an integer
        radial_component  = sp.eval_genlaguerre(p, abs(l), (2 * r_mesh**2 / w**2)) \
                            * (np.sqrt(2) * r_mesh / w)**abs(l)                    \
                            * np.exp(-r_mesh**2 / w**2)                            
                            
        field             = prefactor * radial_component * Gouy_phase * curved_phase
        
        integral_at_plane = np.sum( np.absolute(field)**2 * r_mesh ) * dr 
        
        # the total integral in circular coordinates of the |field|^2 at each plane should be equal to 1,
        # i.e. int (0->infty) int (0->2pi) |field|^2 dtheta dr = 1
        # the azimuthal integral is equal to 2pi (or pi for sinusoidal LG with |l|>0), so the radial integral must be equal to 1/(2pi)
        # (or 1/pi for sinusoidal LG with |l|>0)
        # This check can be skipped if check_total_power_integral = False, 
        # to enable the definition of LG on arbitrary meshes
        if (check_total_power_integral==True):
            if (dict_mode_basis["LG_mode_type"]=="helical"):
                if (integral_at_plane>1.05/(2*np.pi)) or (integral_at_plane<0.95/(2*np.pi)):
                    print("ERROR: total power at plane ",i_plane," in the x,r part is ",integral_at_plane, ", too far from 1/(2pi)")
                    print("If it is lower than 1, try to choose a larger radius for the mesh or a smaller waist")
                    sys.exit()
            elif (dict_mode_basis["LG_mode_type"]=="sinusoidal"):
                if l==0:
                    if (integral_at_plane>1.05/(2*np.pi)) or (integral_at_plane<0.95/(2*np.pi)):
                        print("ERROR: total power at plane ",i_plane," in the x,r part is ",integral_at_plane, ", too far from 1/(pi/2)")
                        print("If it is lower than 1, try to choose a larger radius for the mesh or a smaller waist")
                        sys.exit()
                else:
                    if (integral_at_plane>1.05/(np.pi)) or (integral_at_plane<0.95/(np.pi)):
                        print("ERROR: total power at plane ",i_plane," in the x,r part is ",integral_at_plane, ", too far from 1/(pi)")
                        print("If it is lower than 1, try to choose a larger radius for the mesh or a smaller waist")
                        sys.exit()    
        
        # Store the field at this plane
        LG_field_at_x_r[i_plane, :] = field
    
    return LG_field_at_x_r

def LG_pl_field_theta(l,dict_mesh,dict_mode_basis,check_total_power_integral="True"):
    
    # This function returns the azimuthal part of the LG^{pl} field, at the angle theta.
    # For -l the result is the complex conjugate of the result for l.
    # l          : azimuthal index of the LG mode
    # theta_mesh : theta mesh where the field is defined, size ntheta [rad]
    
    theta_mesh        = dict_mesh["theta_mesh"]
    
    # azimuthal resolution [theta]
    dtheta            = theta_mesh[1] - theta_mesh[0] 
    
    # Store the exp(i*l*theta) part of the LG mode.
    # For the helical LG modes it can be used for both positive and negative l (with conjugation)
    # For the sinusoidal LG modes it can be used to find the cos(l*theta) and sin(l*theta) through the real and imaginary part
    LG_field_at_theta = np.exp(1j * l * theta_mesh)
    
    # the total integral in circular coordinates of the |field|^2 at each plane should be equal to 1,
    # i.e. int (0->infty) int (0->2pi) |field|^2 dtheta dr = 1
    # the azimuthal integral must be equal to 2pi
    if (check_total_power_integral==True):
        integral_at_plane = np.sum( np.absolute(LG_field_at_theta)**2 ) * dtheta
        if (integral_at_plane/(2*np.pi)>1.05) or (integral_at_plane/(2*np.pi)<0.95):
            print("ERROR: total power in the theta part is ",integral_at_plane, ": too far from 2*pi")
            print("Use more sampling points along theta")
            sys.exit()
    
            
    
    return LG_field_at_theta
    
def store_LG_mode_basis_fields(lambda_0,dict_mesh,dict_mode_basis,check_total_power_integral="True"):
                                        
    #### Create and store the arrays needed to define the field of the LG modes.
    #### One part depends on p,l,x,r, it will be stored in an array of shape (Max_LG_index_p+1,Max_LG_index_l+1,number_of_planes,nr)
    #### The other part depends on l,theta, it will be stored in an array of shape ((Max_LG_index_l+1, ntheta), dtype=complex)
    #### The full field of the LG^pl mode will be the multiplication of the two parts (the second one conjugated if l<0)

    Max_LG_index_p     = dict_mode_basis["Max_LG_index_p"     ]
    Max_LG_index_l     = dict_mode_basis["Max_LG_index_l"     ]
    plane_x_coordinates= dict_mesh      ["plane_x_coordinates"]
    r_mesh             = dict_mesh      ["r_mesh"             ]
    theta_mesh         = dict_mesh      ["theta_mesh"         ]
    
    nr                 = np.size(r_mesh                       )
    ntheta             = np.size(theta_mesh                   )
    number_of_planes   = np.size(plane_x_coordinates          )
    
    # this is the part dependent on x and r (identical for l and -l, so only the l>=0 are stored)
    LG_fields_at_x_r   = np.zeros((Max_LG_index_p+1,Max_LG_index_l+1,number_of_planes,nr), dtype=complex)
    # this is the part dependent on theta (the equivalent field for -l is the conjugate of the field for l )
    LG_fields_at_theta = np.zeros((Max_LG_index_l+1, ntheta), dtype=complex)
    
    for l in range(0, Max_LG_index_l+1):
        print("Storing LG_fields_at_theta for l = ",l)
        LG_fields_at_theta[l,:] = LG_pl_field_theta(l,dict_mesh,dict_mode_basis,check_total_power_integral=check_total_power_integral)
        for p in range(0, Max_LG_index_p+1):
            print("Storing LG_fields_at_x_r for p,l = ",p,", ",l)
            LG_fields_at_x_r[p,l,:,:] = LG_pl_field_x_r(lambda_0,p,l,dict_mesh,dict_mode_basis,check_total_power_integral=check_total_power_integral)
    print()
    
    # Store the fields used by the GSA-MD
    dict_mode_basis["LG_fields_at_x_r"  ] = LG_fields_at_x_r
    dict_mode_basis["LG_fields_at_theta"] = LG_fields_at_theta  
    
    return dict_mode_basis

def plot_real_part_LG_mode(i_plane,p,l,dict_mesh,dict_mode_basis):
    
    # Plot the real part of the field of the LG^pl mode,
    # for a given plane of index i_plane, in a cartesian and a polar plot.
    
    r_mesh     = dict_mesh["r_mesh"]
    theta_mesh = dict_mesh["theta_mesh"]
    
    if i_plane < 0 or i_plane >= np.size(dict_mesh["plane_x_coordinates"]):
        raise ValueError("Plane index out of range.")
        
    LG_fields_at_x_r      = dict_mode_basis["LG_fields_at_x_r"  ][p,l,i_plane,:]
    field                 = LG_fields_at_x_r[:]
    
    if (dict_mode_basis["LG_mode_type"]=="helical"):
        LG_fields_at_theta    = dict_mode_basis["LG_fields_at_theta"][abs(l),:]
        if (l >=0 ):
            field             = field[:,np.newaxis] *              LG_fields_at_theta[np.newaxis,:]
        else:
            field             = field[:,np.newaxis] * np.conjugate(LG_fields_at_theta[np.newaxis,:])
    elif (dict_mode_basis["LG_mode_type"]=="sinusoidal"):
        LG_fields_at_theta    = dict_mode_basis["LG_fields_at_theta"][abs(l),:]
        if (l >=0 ):
            field             = field[:,np.newaxis] *      np.real(LG_fields_at_theta[np.newaxis,:])
        else:
            field             = field[:,np.newaxis] *      np.imag(LG_fields_at_theta[np.newaxis,:])

    r_meshgrid,theta_meshgrid = np.meshgrid(dict_mesh["r_mesh"],dict_mesh["theta_mesh"],indexing='ij')
    
    # Cartesian plot
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.grid(False)  # Disable the grid explicitly
    #plt.pcolormesh(theta_mesh, r_mesh, np.abs(self.field_at_plane[i_plane, :, :]), shading='auto', cmap='afmhot_r')
    plt.pcolormesh(theta_meshgrid,r_meshgrid,np.real(field), shading='auto', cmap='afmhot_r')
    plt.title('Real part (Polar) - Plane '+str(i_plane)+', p,l = '+str(p)+', '+str(l))
    plt.xlabel('Theta (radians)')
    plt.ylabel('Radius (pixels)')
        
    # Polar plot
    plt.subplot(1, 2, 2, projection='polar')
    plt.grid(False)  # Disable the grid explicitly
    #c = plt.pcolormesh(theta_mesh, r_mesh, np.abs(self.field_at_plane[i_plane, :, :]), shading='auto', cmap='afmhot_r')
    c = plt.pcolormesh(theta_meshgrid,r_meshgrid,np.real(field), shading='auto', cmap='afmhot_r')
    plt.title('Real part (Polar) - Plane '+str(i_plane)+', p,l = '+str(p)+', '+str(l))
        
    plt.tight_layout()
    plt.show()

def project_field_on_LG_modes(field,i_plane,dict_image_preprocessing,dict_mesh,dict_mode_basis): 
    
    # This function computes C_pl as the integral C_pl = < field(r,theta), LG^pl(i_plane,r,theta) > 
    # where field is a complex field defined on r,theta, on a r_mesh with resolution dr, azimuthal resolution dtheta
    
    energy_reference_plane = dict_image_preprocessing["energy_reference_plane"]
    Max_LG_index_p         = dict_mode_basis         ["Max_LG_index_p"        ]
    Max_LG_index_l         = dict_mode_basis         ["Max_LG_index_l"        ]
    LG_fields_at_x_r       = dict_mode_basis         ["LG_fields_at_x_r"      ][:,:,i_plane,:]
    LG_fields_at_theta     = dict_mode_basis         ["LG_fields_at_theta"    ][:,:]
    r_mesh                 = dict_mesh               ["r_mesh"                ]
    theta_mesh             = dict_mesh               ["theta_mesh"            ]
    
    dr                     = r_mesh    [1]-r_mesh    [0]
    dtheta                 = theta_mesh[1]-theta_mesh[0]
    
    Coeffs_LG_pl           = np.zeros(shape=(Max_LG_index_p+1,int(2*Max_LG_index_l+1)),dtype=complex)
    
    # Compute the coefficients as scalar product with the LG modes
    if (dict_mode_basis["LG_mode_type"]=="helical"):
        #Coeffs_LG_pl = compute_LG_coefficients_no_numba(Coeffs_LG_pl,Max_LG_index_l,energy_reference_plane,field,LG_fields_at_x_r,LG_fields_at_theta,dtheta,r_mesh,dr) 
        Coeffs_LG_pl = compute_helical_LG_coefficients_numba(Coeffs_LG_pl,Max_LG_index_l,energy_reference_plane,field,LG_fields_at_x_r,LG_fields_at_theta,dtheta,r_mesh,dr)
    elif (dict_mode_basis["LG_mode_type"]=="sinusoidal"):
        #Coeffs_LG_pl = compute_sinusoidal_LG_coefficients_no_numba(Coeffs_LG_pl,Max_LG_index_l,energy_reference_plane,field,LG_fields_at_x_r,LG_fields_at_theta,dtheta,r_mesh,dr) 
        Coeffs_LG_pl = compute_sinusoidal_LG_coefficients_numba(Coeffs_LG_pl,Max_LG_index_l,energy_reference_plane,field,LG_fields_at_x_r,LG_fields_at_theta,dtheta,r_mesh,dr)
    
    return Coeffs_LG_pl

def compute_helical_LG_coefficients_no_numba(Coeffs_LG_pl, Max_LG_index_l, energy_reference_plane, field, LG_fields_at_x_r, LG_fields_at_theta, dtheta, r_mesh, dr):
    # For the helical LG, as in a FFT, the negative l harmonics are stored after the ones for l>=0, with increasingly negative l.
    integrand                            = field[np.newaxis,np.newaxis,:,:]              * np.conjugate(LG_fields_at_x_r  [:,0:(Max_LG_index_l+1),:,np.newaxis]) 
    integrand                            = integrand                                     * r_mesh[np.newaxis, np.newaxis,:,np.newaxis] * dr
    # l >= 0
    Coeffs_LG_pl[:,0:(Max_LG_index_l+1)] = np.sum( integrand[:,0:(Max_LG_index_l+1),:,:] * np.conjugate(LG_fields_at_theta[np.newaxis,0:(Max_LG_index_l+1),np.newaxis,:]) ,axis=(2,3)) * dtheta 
    # l < 0, Rember that the LG_fields_at_theta for -l is the conjugate of LG_fields_at_theta of l, 
    # and that LG_fields_at_x_r only depends on p, |l| 
    Coeffs_LG_pl[:,(Max_LG_index_l+1)::] = np.sum( integrand[:,1:(Max_LG_index_l+1),:,:] *             (LG_fields_at_theta[np.newaxis,1:(Max_LG_index_l+1),np.newaxis,:]) ,axis=(2,3)) * dtheta 

    # As in a FFT, the negative l harmonics are stored after the ones for l>=0, with increasingly negative l.
    # Normalize the new Coeff_LG_pl
    Coeffs_LG_pl = Coeffs_LG_pl * np.sqrt(energy_reference_plane) / np.sqrt(np.sum(np.absolute(Coeffs_LG_pl)**2)) 

    return Coeffs_LG_pl

def compute_sinusoidal_LG_coefficients_no_numba(Coeffs_LG_pl, Max_LG_index_l, energy_reference_plane, field, LG_fields_at_x_r, LG_fields_at_theta, dtheta, r_mesh, dr):
    # As in a FFT, the negative l harmonics are stored after the ones for l>=0, with increasingly negative l.
    integrand                            = field[np.newaxis,np.newaxis,:,:]              * np.conjugate(LG_fields_at_x_r  [:,0:(Max_LG_index_l+1),:,np.newaxis]) 
    integrand                            = integrand                                     * r_mesh[np.newaxis, np.newaxis,:,np.newaxis] * dr
    # l >= 0
    Coeffs_LG_pl[:,0:(Max_LG_index_l+1)] = np.sum( integrand[:,0:(Max_LG_index_l+1),:,:] * np.real(LG_fields_at_theta[np.newaxis,0:(Max_LG_index_l+1),np.newaxis,:]) ,axis=(2,3)) * dtheta 
    # l < 0, Rember that the LG_fields_at_theta for -l is the conjugate of LG_fields_at_theta of l, 
    # and that LG_fields_at_x_r only depends on p, |l| 
    Coeffs_LG_pl[:,(Max_LG_index_l+1)::] = np.sum( integrand[:,1:(Max_LG_index_l+1),:,:] * np.imag(LG_fields_at_theta[np.newaxis,1:(Max_LG_index_l+1),np.newaxis,:]) ,axis=(2,3)) * dtheta 

    # As in a FFT, the negative l harmonics are stored after the ones for l>=0, with increasingly negative l.
    # Normalize the new Coeff_LG_pl
    Coeffs_LG_pl = Coeffs_LG_pl * np.sqrt(energy_reference_plane) / np.sqrt(np.sum(np.absolute(Coeffs_LG_pl)**2)) 

    return Coeffs_LG_pl

@njit(parallel=True)
def compute_helical_LG_coefficients_numba(Coeffs_LG_pl, Max_LG_index_l, energy_reference_plane, field, LG_fields_at_x_r, LG_fields_at_theta, dtheta, r_mesh, dr):

    # Precompute common multipliers
    r_mesh_dr_dtheta = r_mesh * dr * dtheta
    
    # For the helical LG, as in a FFT, the negative l harmonics are stored after the ones for l>=0, with increasingly negative l.

    # using prange only for independent loop iterations to avoid race conditions
    for p in prange(LG_fields_at_x_r.shape[0]):
        # Positive l coefficients
        for l in prange(Max_LG_index_l + 1):
            sum_pos_l = 0.0 + 0.0j  # Local accumulator for positive l
            for i_r in range(LG_fields_at_x_r.shape[2]):
                for i_theta in range(LG_fields_at_theta.shape[1]):
                    sum_pos_l += field[i_r, i_theta] * (LG_fields_at_x_r[p, l, i_r] * LG_fields_at_theta[l, i_theta]).conjugate() * r_mesh_dr_dtheta[i_r]
            Coeffs_LG_pl[p, l] = sum_pos_l  # Update after accumulation

    # using prange only for independent loop iterations to avoid race conditions
    for p in prange(LG_fields_at_x_r.shape[0]):
        # Negative l coefficients
        for l in prange(1, Max_LG_index_l + 1):
            neg_l_index = Max_LG_index_l + l
            sum_neg_l = 0.0 + 0.0j  # Local accumulator for negative l
            for i_r in range(LG_fields_at_x_r.shape[2]):
                for i_theta in range(LG_fields_at_theta.shape[1]):
                    sum_neg_l += field[i_r, i_theta] * LG_fields_at_x_r[p, l, i_r].conjugate() * LG_fields_at_theta[l, i_theta] * r_mesh_dr_dtheta[i_r]
            Coeffs_LG_pl[p, neg_l_index] = sum_neg_l  # Update after accumulation
    
    # Normalize the coefficients
    mode_energy_sum  = np.sum(np.square(np.abs(Coeffs_LG_pl)))
    Coeffs_LG_pl    *=np.sqrt(energy_reference_plane/mode_energy_sum)

    return Coeffs_LG_pl

@njit(parallel=True)
def compute_sinusoidal_LG_coefficients_numba(Coeffs_LG_pl, Max_LG_index_l, energy_reference_plane, field, LG_fields_at_x_r, LG_fields_at_theta, dtheta, r_mesh, dr,inside):

    # Precompute common multipliers
    r_mesh_dr_dtheta = r_mesh * dr * dtheta
    
    # As in a FFT, the negative l harmonics are stored after the ones for l>=0, with increasingly negative l.

    # using prange only for independent loop iterations to avoid race conditions
    for p in prange(LG_fields_at_x_r.shape[0]):
        # Positive l coefficients, in the sinusoidal LG they store the factor cos(l*theta)
        for l in prange(Max_LG_index_l + 1):
            sum_pos_l = 0.0 + 0.0j  # Local accumulator for positive l
            for i_r in range(LG_fields_at_x_r.shape[2]):
                for i_theta in range(LG_fields_at_theta.shape[1]):
                    sum_pos_l += field[i_r, i_theta] * (LG_fields_at_x_r[p, l, i_r] * np.real(LG_fields_at_theta[l, i_theta])).conjugate() * r_mesh_dr_dtheta[i_r]
            Coeffs_LG_pl[p, l] = sum_pos_l  # Update after accumulation

    # using prange only for independent loop iterations to avoid race conditions
    for p in prange(LG_fields_at_x_r.shape[0]):
        # Negative l coefficients, in the sinusoidal LG they store the factor sin(l*theta)
        for l in prange(1, Max_LG_index_l + 1):
            neg_l_index = Max_LG_index_l + l
            sum_neg_l = 0.0 + 0.0j  # Local accumulator for negative l
            for i_r in range(LG_fields_at_x_r.shape[2]):
                for i_theta in range(LG_fields_at_theta.shape[1]):
                    sum_neg_l += field[i_r, i_theta] * LG_fields_at_x_r[p, l, i_r].conjugate() * np.imag(LG_fields_at_theta[l, i_theta]) * r_mesh_dr_dtheta[i_r]
            Coeffs_LG_pl[p, neg_l_index] = sum_neg_l  # Update after accumulation
    
    # Normalize the coefficients
    mode_energy_sum  = np.sum(np.square(np.abs(Coeffs_LG_pl)))
    Coeffs_LG_pl    *=np.sqrt(energy_reference_plane/mode_energy_sum)

    return Coeffs_LG_pl


def LG_reconstruct_field_at_plane(Coeffs_LG_pl,i_plane,dict_mode_basis):
    
    # This function computes E(i_plane,r,theta) = sum_{pl} [ C_pl * LG^{pl}(i_plane,r,theta) ]
    Max_LG_index_p         = dict_mode_basis["Max_LG_index_p"    ]
    Max_LG_index_l         = dict_mode_basis["Max_LG_index_l"    ]
    LG_fields_at_x_r       = dict_mode_basis["LG_fields_at_x_r"  ][:,:,i_plane,:]
    LG_fields_at_theta     = dict_mode_basis["LG_fields_at_theta"][:,:]
    
    
    E = np.zeros(shape=(LG_fields_at_x_r.shape[2],LG_fields_at_theta.shape[1]), dtype=complex)
    # use reconstruction function based on numba
    if (dict_mode_basis["LG_mode_type"]=="helical"):
        #E = reconstruct_helical_LG_E_no_numba(E,Max_LG_index_l,Coeffs_LG_pl,LG_fields_at_x_r,LG_fields_at_theta)
        E = reconstruct_helical_LG_E_numba(E,Max_LG_index_l,Coeffs_LG_pl,LG_fields_at_x_r,LG_fields_at_theta)
    elif (dict_mode_basis["LG_mode_type"]=="sinusoidal"):
        #E = reconstruct_sinusoidal_LG_E_no_numba(E,Max_LG_index_l,Coeffs_LG_pl,LG_fields_at_x_r,LG_fields_at_theta)
        E = reconstruct_sinusoidal_LG_E_numba(E,Max_LG_index_l,Coeffs_LG_pl,LG_fields_at_x_r,LG_fields_at_theta)
    return E

def reconstruct_helical_LG_E_no_numba(E, Max_LG_index_l, Coeffs_LG_pl, LG_fields_at_x_r, LG_fields_at_theta):
    # For the helical LG, as in a FFT, the negative l harmonics are stored after the ones for l>=0, with increasingly negative l.
    # l >= 0
    E  = np.sum(Coeffs_LG_pl[:,0:(Max_LG_index_l+1),np.newaxis,np.newaxis]*LG_fields_at_x_r  [:,0:(Max_LG_index_l+1),:,np.newaxis]*LG_fields_at_theta[np.newaxis,0:(Max_LG_index_l+1),np.newaxis,:],axis=(0,1))
    # l < 0, Remember that the LG_fields_at_theta for -l is the conjugate of LG_fields_at_theta of l, 
    # and that LG_fields_at_x_r only depends on p, |l| 
    E += np.sum(Coeffs_LG_pl[:,(Max_LG_index_l+1)::,np.newaxis,np.newaxis]*LG_fields_at_x_r  [:,1:(Max_LG_index_l+1),:,np.newaxis]*np.conjugate(LG_fields_at_theta[np.newaxis,1:(Max_LG_index_l+1),np.newaxis,:]),axis=(0,1))
    return E

def reconstruct_sinusoidal_LG_E_no_numba(E, Max_LG_index_l, Coeffs_LG_pl, LG_fields_at_x_r, LG_fields_at_theta):
    # As in a FFT, the negative l harmonics are stored after the ones for l>=0, with increasingly negative l.
    # Positive l coefficients, for the sinusoidal LG modes they store the factor cos(l*theta)
    # l >= 0
    E  = np.sum(Coeffs_LG_pl[:,0:(Max_LG_index_l+1),np.newaxis,np.newaxis]*LG_fields_at_x_r  [:,0:(Max_LG_index_l+1),:,np.newaxis]*np.real(LG_fields_at_theta[np.newaxis,0:(Max_LG_index_l+1),np.newaxis,:]),axis=(0,1))
    # Negative l coefficients, for the sinusoidal LG modes they store the factor sin(l*theta)
    # l < 0, remember that LG_fields_at_x_r only depends on p, |l| 
    E += np.sum(Coeffs_LG_pl[:,(Max_LG_index_l+1)::,np.newaxis,np.newaxis]*LG_fields_at_x_r  [:,1:(Max_LG_index_l+1),:,np.newaxis]*np.imag(LG_fields_at_theta[np.newaxis,1:(Max_LG_index_l+1),np.newaxis,:]),axis=(0,1))
    return E
    
@njit(parallel=True)
def reconstruct_helical_LG_E_numba(E, Max_LG_index_l, Coeffs_LG_pl, LG_fields_at_x_r, LG_fields_at_theta):
    # using prange only for independent loop iterations to avoid race conditions
    for i_r in prange(LG_fields_at_x_r.shape[2]):
        for i_theta in prange(LG_fields_at_theta.shape[1]):
            sum_E = 0.0 + 0.0j 
            for p in range(LG_fields_at_x_r.shape[0]):
                # For the helical LG, as in a FFT, the negative l harmonics are stored after the ones for l>=0, with increasingly negative l.
                # Positive l coefficients
                for l in range(Max_LG_index_l + 1):
                    sum_E += Coeffs_LG_pl[p, l] * LG_fields_at_x_r[p, l, i_r] * LG_fields_at_theta[l, i_theta]
                # Negative l coefficients
                # # l < 0, Remember that the LG_fields_at_theta for -l is the conjugate of LG_fields_at_theta of l, 
                # # and that LG_fields_at_x_r only depends on p, |l| 
                for l in range(1, Max_LG_index_l + 1):
                    neg_l_index = Max_LG_index_l + l
                    sum_E += Coeffs_LG_pl[p, neg_l_index] * LG_fields_at_x_r[p, l, i_r] * np.conjugate(LG_fields_at_theta[l, i_theta])
            E[i_r, i_theta] = sum_E  

    return E

@njit(parallel=True)
def reconstruct_sinusoidal_LG_E_numba(E, Max_LG_index_l, Coeffs_LG_pl, LG_fields_at_x_r, LG_fields_at_theta):
    # using prange only for independent loop iterations to avoid race conditions
    for i_r in prange(LG_fields_at_x_r.shape[2]):
        for i_theta in prange(LG_fields_at_theta.shape[1]):
            sum_E = 0.0 + 0.0j 
            for p in range(LG_fields_at_x_r.shape[0]):
                # As in a FFT, the negative l harmonics are stored after the ones for l>=0, with increasingly negative l.
                # Positive l coefficients, for the sinusoidal LG modes they store the factor cos(l*theta)
                for l in range(Max_LG_index_l + 1):
                    sum_E += Coeffs_LG_pl[p, l] * LG_fields_at_x_r[p, l, i_r] * np.real(LG_fields_at_theta[l, i_theta])
                # Negative l coefficients, for the sinusoidal LG modes they store the factor sin(l*theta)
                # Remember that the LG_fields_at_x_r only depend on p, |l| 
                for l in range(1, Max_LG_index_l + 1):
                    neg_l_index = Max_LG_index_l + l
                    sum_E += Coeffs_LG_pl[p, neg_l_index] * LG_fields_at_x_r[p, l, i_r] * np.imag(LG_fields_at_theta[l, i_theta])
            E[i_r, i_theta] = sum_E  

    return E
    
