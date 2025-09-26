###### Authors: I. Moulanier, F. Massimo

###### This file contains general basic operations with the mode basis, 
###### e.g. mode projection and field reconstruction through the sum of modes.
###### The functions inside have if branches for the chosen mode basis (HG or LG)


from .hermite_gauss_modes  import *
from .laguerre_gauss_modes import *
import os,sys

def check_mode_basis_inputs(dict_image_preprocessing,lambda_0,dict_mode_basis):
    
    if ( lambda_0 <=0 ):
        print("ERROR: lambda_0 is <= 0." )
        sys.exit()
    if ( dict_mode_basis["waist_0"] <= 0):
        print("ERROR: waist_0 is <= 0." )
        sys.exit()
    if "x_focus" not in dict_mode_basis.keys():
        print("ERROR: an x_focus must be specified for the mode basis" )
        sys.exit()
        
    if dict_image_preprocessing["geometry"] == "cartesian":
        if "Max_HG_index_m" not in dict_mode_basis.keys():
            print("ERROR: an Max_HG_index_m (y direction) must be specified for the mode basis" )
            sys.exit()
        if "Max_HG_index_n" not in dict_mode_basis.keys():
            print("ERROR: an Max_HG_index_n (z direction) must be specified for the mode basis" )
            sys.exit()
        
        print("Chosen mode basis: Hermite-Gauss")
        print("- Mode m index (y direction) from 0 to ",dict_mode_basis["Max_HG_index_m"])
        print("- Mode n index (z direction) from 0 to ",dict_mode_basis["Max_HG_index_n"])
        print()
        
    else:
        if ("LG_mode_type" not in dict_mode_basis.keys()):
            # by default, helical LG modes are used
            dict_mode_basis["LG_mode_type"] = "helical"
        else:
            if (dict_mode_basis["LG_mode_type"] != "helical") & (dict_mode_basis["LG_mode_type"] != "sinusoidal" ):
                print("ERROR: the LG_mode_type must be either 'helical' or 'sinusoidal' ")
                sys.exit()
            
        if ("Max_LG_index_l" not in dict_mode_basis.keys()):
            print("ERROR: an Max_LG_index_p (r direction) must be specified for the mode basis" )
            sys.exit()
        if ("Max_LG_index_l" not in dict_mode_basis.keys()):
            print("ERROR: an Max_LG_index_l (theta direction) must be specified for the mode basis" )
            sys.exit()
        
        print("Chosen mode basis: Laguerre-Gauss")
        print("- Mode p index (r direction    ) from 0 to ",dict_mode_basis["Max_LG_index_p"])
        print("- Mode l index (theta direction) from 0 to ",dict_mode_basis["Max_LG_index_l"])
        print()
            
    return dict_mode_basis

### These functions use dict_mode_basis as input because it can be part of a dict_GSA
### or just defined by the user independently 

def store_mode_basis_fields(lambda_0,dict_image_preprocessing,dict_mesh,dict_mode_basis):
    
    print("# Storing the mode basis fields\n")
    
    if dict_image_preprocessing["geometry"] == "cylindrical": # Laguerre-Gauss modes
        dict_mode_basis = store_LG_mode_basis_fields(lambda_0,dict_mesh,dict_mode_basis)
    else:                                                     # Hermite-Gauss modes
        dict_mode_basis = store_HG_mode_basis_fields(lambda_0,dict_mesh,dict_mode_basis)
        
    # save the updated mode basis dictionary with the partial mode fields
    np.save('outputs/dict_mode_basis.npy', dict_mode_basis)  
    
    return dict_mode_basis
    
def project_field_on_mode_basis(field,i_plane,dict_image_preprocessing,dict_mesh,dict_mode_basis):
    
    if (dict_image_preprocessing["geometry"]=="cylindrical"):
        # C_pl = <field(r,theta), LG^{pl}(i_plane,r,theta)>
        Coeffs_LG_pl = project_field_on_LG_modes(field,i_plane,dict_image_preprocessing,dict_mesh,dict_mode_basis)
        return Coeffs_LG_pl
    else:
        # C_mn = <field(y,z), HG^{mn}(i_plane,y,z)>
        Coeffs_HG_mn = project_field_on_HG_modes(field,i_plane,dict_image_preprocessing,dict_mesh,dict_mode_basis)
        return Coeffs_HG_mn

def reconstruct_field_at_plane(Coeffs_MD,i_plane,dict_image_preprocessing,dict_mesh,dict_mode_basis):
    
    if (dict_image_preprocessing["geometry"]=="cylindrical"):
        # E(r,theta) = sum_{pl} [ C_pl * LG^{pl}(i_plane,r,theta) ]
        E = LG_reconstruct_field_at_plane(Coeffs_MD,i_plane,dict_mode_basis)
        return E
    else:
        # E(y,z)     = sum_{mn} [ C_mn * HG^{mn}(i_plane,y,z   ) ]
        E = HG_reconstruct_field_at_plane(Coeffs_MD,i_plane,dict_mode_basis)
        return E