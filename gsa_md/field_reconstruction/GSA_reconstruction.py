###### Authors: Ioaquin Moulanier, F. Massimo

###### This module contains the field reconstruction main function,
###### which calls the GSA or GSA_MD reconstruction loop 
###### depending on the user initial choice

from .reconstruction_utilities   import check_GSA_inputs,compute_experimental_field_amplitudes,gaussian_phase
from .reconstruction_diagnostics import final_output_dump,initialize_diagnostics
from .GSA_loop                   import GSA
from .GSA_MD_loop                import GSA_MD
import os,sys

def field_reconstruction_GSA(dict_image_preprocessing,dict_mesh,dict_GSA):
    
    print("### Field reconstruction\n")
    
    #### Check the inputs
    dict_GSA                        = check_GSA_inputs(dict_image_preprocessing,dict_GSA)
    
    #### Initialize the vectors for the diagnostic
    dict_GSA                        = initialize_diagnostics(dict_GSA,dict_mesh)
    
    #### Extract field amplitude
    dict_GSA                        = compute_experimental_field_amplitudes(dict_image_preprocessing,dict_GSA)
    
    #### Create an initial Gaussian beam phase if needed
    dict_GSA                        = gaussian_phase(dict_image_preprocessing,dict_mesh,dict_GSA) 
    
    #### Reconstruct the field, with or without Mode Decomposition (MD)
    if dict_GSA["use_Mode_Decomposition"] == True:
        dict_GSA                    = GSA_MD(dict_image_preprocessing,dict_mesh,dict_GSA)
    else:
        dict_GSA                    = GSA   (dict_image_preprocessing,dict_mesh,dict_GSA)
    
    #### Final dump
    dict_GSA                        = final_output_dump(dict_image_preprocessing,dict_mesh,dict_GSA)
                                            
    return dict_GSA
               
