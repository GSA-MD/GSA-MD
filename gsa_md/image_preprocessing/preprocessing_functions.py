###### Authors: Francesco Massimo, Ioaquin Moulanier

###### This module contains the pre-processing operations for the GSA

import numpy             as np
import matplotlib.pyplot as plt
import scipy.signal      as signal
import scipy.ndimage,os,sys,time
from scipy.ndimage       import median_filter
from scipy.ndimage       import map_coordinates

def process_input_fluence_images(dict_image_preprocessing,dict_mesh):
    
    ### This is the main preprocessing function,
    ### which calls the following functions in sequence: 
    ### - check_preprocessing_and_mesh_inputs
    ### - filter_and_remove_background_from_input_images
    ### - find_image_centers
    ### - compute_reference_plane_energy
    ### - convert_images_from_cartesian_to_circular_coordinates (if geometry=="cylindrical")
    ### - convert_images_cartesian_coordinates                  (if geometry=="cartesian")
    ### - create_mask_inside_r_max
    ### In the process, if not already present, the following data are added 
    ### to the dictionaries provided as input
    ### - image_center_coordinates
    ### - preprocessed_images (i.e. the preprocessed input images and their conversion to the cylindrical/cartesian grid)
    ### - transverse_mesh (r_mesh,theta_mesh or y_mesh,z_mesh)
    
    print("\n### Preprocessing of the input fluence images \n")
    time_image_preprocessing = time.time()
    
    ##### Create folder that will contain the dictionaries and the outputs
    ##### from this execution of the program
    os.mkdir("outputs")
    
    
    ##### Check coherence of inputs and add the missing entries in the input dictionaries
    dict_image_preprocessing, dict_mesh = check_preprocessing_and_mesh_inputs(dict_image_preprocessing,dict_mesh)
    
    ##### Preliminary image preprocessing: filering and background removal
    dict_image_preprocessing            = filter_and_remove_background_from_input_images(dict_image_preprocessing)
    
    ##### Find image centers if not provided by user and store them
    dict_image_preprocessing, dict_mesh = find_image_centers(dict_image_preprocessing,dict_mesh)
    
    ##### Compute energy in the reference plane
    dict_image_preprocessing            = compute_reference_plane_energy(dict_image_preprocessing)
    
    ##### Use the image center coordinates to either:
    ##### - convert the images from cartesian to cylindrical coordinates (geometry="cylindrical")
    ##### - center the images in cartesian coordinates                   (geometry="cartesian"  )
    
    if dict_image_preprocessing["geometry"]=="cylindrical":
        # Create the mesh, convert and store the images in cylindrical coordinates
        # and save also a centered image in a cartesian grid to check the consistency of the images in the two grids
        dict_image_preprocessing, dict_mesh = transfer_fluence_images_to_cylindrical_grid(dict_image_preprocessing,dict_mesh)
    else:
        # Create the mesh, center the images in a cartesian grid
        ny = dict_mesh["ny_converted_image"]
        nz = dict_mesh["nz_converted_image"]
        dict_image_preprocessing, dict_mesh = transfer_fluence_images_to_cartesian_grid(dict_image_preprocessing,dict_mesh,ny=ny,nz=nz)
    
    ## define a mask equal to False for r > r_max_coordinate_change 
    dict_image_preprocessing                = create_mask_inside_r_max(dict_image_preprocessing,dict_mesh)
    
    ## save time for preprocessing
    time_image_preprocessing                = time.time() - time_image_preprocessing
    dict_image_preprocessing["time_image_preprocessing"] = time_image_preprocessing
    
    # Save dictionaries after the preprocessing
    np.save ("outputs/dict_image_preprocessing.npy", dict_image_preprocessing )
    np.save ("outputs/dict_mesh.npy"               , dict_mesh                )
        
    return dict_image_preprocessing, dict_mesh


def check_preprocessing_and_mesh_inputs(dict_image_preprocessing,dict_mesh):
    
    input_fluence_images            = dict_image_preprocessing["input_fluence_images"]
    plane_x_coordinates             = dict_mesh               ["plane_x_coordinates" ]
    
    if np.size(plane_x_coordinates)!=np.size(input_fluence_images[:,0,0]):
        print("Error: array plane_x_coordinates must have a number of elements equal to the number of input fluence images")
        sys.exit()
    
    ##### Preprocess the Inputs
    
    # reference plane used for energy calculation
    if "index_reference_plane" not in dict_image_preprocessing.keys():
        dict_image_preprocessing["index_reference_plane"] = -1
    else:
        if dict_image_preprocessing["index_reference_plane"]>(np.size(plane_x_coordinates)-1):
            print("Error: index_reference_plane must be smaller than the number of images")
    
    # noise threshold, if fluence (after subtraction of its minimum value) is below this value the corresponding point in the converted image will be put to zero [fluence units]
    if "threshold" not in dict_image_preprocessing.keys():                              
        dict_image_preprocessing["threshold"] = 0.
    
    # radius around the y0 and z0, beyond which the corresponding point in the converted image will be put to zero [m]
    if "r_max_coordinate_change" not in dict_image_preprocessing.keys():  
        dict_image_preprocessing["r_max_coordinate_change"] = np.inf
    
    if "median_filter_size" not in dict_image_preprocessing.keys(): 
        dict_image_preprocessing["median_filter_size"] = 4
    
    # the size of the converted image
    if dict_image_preprocessing["geometry"]=="cylindrical":
        if "ntheta" not in dict_mesh.keys(): 
            dict_mesh["ntheta"] = 400
        else:
            if dict_mesh["ntheta"]%4 != 0:
                print("Error: choose a ntheta multiple of 4")
                sys.exit()
        if "nr_converted_image" not in dict_mesh.keys(): 
            dict_mesh["nr_converted_image"] = np.maximum(np.size(input_fluence_images[0,:,0])//2,np.size(input_fluence_images[0,0,:])//2)
            
    elif dict_image_preprocessing["geometry"]=="cartesian":
        if "ny_converted_image" not in dict_mesh.keys(): 
            dict_mesh["ny_converted_image"] = np.maximum(np.size(input_fluence_images[0,:,0])//2,np.size(input_fluence_images[0,0,:])//2)
        else:
            if (dict_mesh["ny_converted_image"] < np.size(input_fluence_images[0,:,0])):
                print("Error: ny_converted_image = ",dict_mesh["ny_converted_image"]," is too small, it must be larger than")
                print("the image size on axis 0 : ",np.size(input_fluence_images[0,:,0]))
        if "nz_converted_image" not in dict_mesh.keys(): 
            dict_mesh["nz_converted_image"] = np.maximum(np.size(input_fluence_images[0,:,0])//2,np.size(input_fluence_images[0,0,:])//2)
        else:
            if (dict_mesh["nz_converted_image"] < np.size(input_fluence_images[0,0,:])):
                print("Error: nz_converted_image = ",dict_mesh["nz_converted_image"]," is too small, it must be larger than")
                print("the image size on axis 1 : ",np.size(input_fluence_images[0,0,:]))    
    else:
        print("Error: geometry must be either 'cylindrical' or 'cartesian'. ")
        sys.exit()
    
    # Print the inputs
    print("Parameters for fluence image preprocessing:")
    print("- geometry                           : ",dict_image_preprocessing["geometry"               ])
    print("- index_reference_plane              : ",dict_image_preprocessing["index_reference_plane"  ])
    print("- threshold                          : ",dict_image_preprocessing["threshold"              ])
    print("- median_filter_size                 : ",dict_image_preprocessing["median_filter_size"     ])
    print("- r_max_coordinate_change (m)        : ",dict_image_preprocessing["r_max_coordinate_change"])
    if dict_image_preprocessing["geometry"]=="cylindrical":
        print("- nr_converted_image                 : ",dict_mesh           ["nr_converted_image"     ])
        print("- ntheta                             : ",dict_mesh           ["ntheta"                 ])
    else: 
        print("- ny_converted_image                 : ",dict_mesh           ["ny_converted_image"     ])
        print("- nz_converted_image                 : ",dict_mesh           ["nz_converted_image"     ])
    print("\n")
    
    return dict_image_preprocessing,dict_mesh

def filter_and_remove_background_from_input_images(dict_image_preprocessing):
    
    input_fluence_images = dict_image_preprocessing["input_fluence_images"]
    threshold             = dict_image_preprocessing["threshold"          ]
    
    # filter and subtract the minimum from input fluence image
    # Normalize the images and apply threshold
    filtered_images_no_background_noise = np.zeros_like(input_fluence_images)
    for i_plane in range(np.size(input_fluence_images[:,0,0])):
        if dict_image_preprocessing["median_filter_size"]>1:
            filtered_images_no_background_noise[i_plane,:,:] = median_filter(input_fluence_images[i_plane,:,:], size = dict_image_preprocessing["median_filter_size"])
        elif (dict_image_preprocessing["median_filter_size"]<=1):
            print("No median filter is used for fluence image preprocessing.\n")
            filtered_images_no_background_noise[i_plane,:,:] = input_fluence_images[i_plane,:,:]
        filtered_images_no_background_noise[i_plane,:,:]-= np.amin(filtered_images_no_background_noise[i_plane,:,:])

    filtered_images_no_background_noise[filtered_images_no_background_noise < threshold] = 0.0
    
    dict_image_preprocessing["filtered_images_no_background_noise"] = filtered_images_no_background_noise
    
    return dict_image_preprocessing
    

def find_image_centers(dict_image_preprocessing,dict_mesh):
    ### The image center will be found if not provided by the user.
    ### In this case, it will be defined as the peak point of the image 
    ### (it seems to work better than choosing the center of mass)
    
    filtered_images_no_background_noise = dict_image_preprocessing["filtered_images_no_background_noise"]
    y0_at_planes                        = dict_image_preprocessing["y0_at_planes"                       ]
    z0_at_planes                        = dict_image_preprocessing["z0_at_planes"                       ]
    length_per_pixel                          = dict_image_preprocessing["length_per_pixel"                         ]
    plane_x_coordinates                 = dict_mesh               ["plane_x_coordinates"]
    
    # find the center of the laser fluence
    find_y0_and_z0 = True
    if (type(y0_at_planes)!=list) and (type(z0_at_planes)!=list):
        find_y0_and_z0 = False

    if find_y0_and_z0==True:
        for i_plane in range(np.size(filtered_images_no_background_noise[:,0,0])):
            maximum_indices = np.unravel_index(np.argmax(filtered_images_no_background_noise[i_plane,:,:]), filtered_images_no_background_noise[i_plane,:,:].shape)
            y0 = maximum_indices[0]*length_per_pixel
            z0 = maximum_indices[1]*length_per_pixel;
            
            y0_at_planes.append(y0)
            z0_at_planes.append(z0)
            
    y0_at_planes = np.asarray(y0_at_planes)
    z0_at_planes = np.asarray(z0_at_planes)
    
    # Print the centers of the image
    if find_y0_and_z0:
        string = "defined from maximum of each image"
    else:
        string = "provided by user"
    print("Center of the fluence images "+string)
    for i_plane in range(0,np.size(plane_x_coordinates)):
        print(f"- x = {plane_x_coordinates[i_plane]:>12.3e} m : \t y0 = {y0_at_planes[i_plane]:>12.3e} m, z0 = {z0_at_planes[i_plane]:>12.3e} m")
    print("\n")   
    
    # Save the image centers
    dict_image_preprocessing["y0_at_planes"] = y0_at_planes
    dict_image_preprocessing["z0_at_planes"] = z0_at_planes
    
    return dict_image_preprocessing,dict_mesh

def compute_reference_plane_energy(dict_image_preprocessing):
    index_reference_plane                              = dict_image_preprocessing["index_reference_plane" ]
    fluence_exp_images_cartesian                       = dict_image_preprocessing["filtered_images_no_background_noise"]
    length_per_pixel                                         = dict_image_preprocessing["length_per_pixel"                         ]
    energy_reference_plane                             = np.sum(fluence_exp_images_cartesian[index_reference_plane,:,:])*length_per_pixel*length_per_pixel
    dict_image_preprocessing["energy_reference_plane"] = energy_reference_plane
    return dict_image_preprocessing
    

def transfer_fluence_images_to_cylindrical_grid(dict_image_preprocessing,dict_mesh):
    
    fluence_exp_images_cartesian       = dict_image_preprocessing["filtered_images_no_background_noise"]
    index_reference_plane              = dict_image_preprocessing["index_reference_plane" ]
    length_per_pixel                         = dict_image_preprocessing["length_per_pixel"                         ]
    y0_at_planes                       = dict_image_preprocessing["y0_at_planes"                       ]
    z0_at_planes                       = dict_image_preprocessing["z0_at_planes"                       ]
    r_max_coordinate_change            = dict_image_preprocessing["r_max_coordinate_change"            ]
    nr_converted_image                 = dict_mesh               ["nr_converted_image"                 ]
    ntheta                             = dict_mesh               ["ntheta"                             ]
        
    # Image dimensions
    number_of_planes,width,height = fluence_exp_images_cartesian.shape
    
    # Generate the theta values
    theta_mesh = (2 * np.pi / ntheta) * np.linspace(0, ntheta - 1., num=ntheta)
    # Define an array for the r_mesh, keeping the same resolution of the initial image along r (i.e. the length_per_pixel)
    # in general the maximum radius should be larger than the radius of the image
    r_mesh     = length_per_pixel*np.arange(0,nr_converted_image)
    # note that the first cell center is at dr/2
    r_mesh     = r_mesh+length_per_pixel/2.
    # This array will store images converted in polar coordinates
    fluence_exp_images_circular = np.zeros(shape=(number_of_planes,nr_converted_image,ntheta))

    for i_plane in range(0,number_of_planes):
        
        # Compute the maximum radius
        this_image_max_radius = np.minimum(y0_at_planes[i_plane],((width-1)  * length_per_pixel - y0_at_planes[i_plane]))
        this_image_max_radius = np.minimum(this_image_max_radius,z0_at_planes[i_plane])
        this_image_max_radius = np.minimum(this_image_max_radius,((height-1) * length_per_pixel - z0_at_planes[i_plane]) )
        
        # Create a grid of radius and theta
        this_image_r_mesh     = length_per_pixel*np.arange(int(this_image_max_radius/length_per_pixel))
        this_image_r_mesh     = this_image_r_mesh+length_per_pixel/2
        this_image_r_meshgrid, theta_meshgrid = np.meshgrid(this_image_r_mesh,theta_mesh,indexing='ij')
        
        # Convert polar coordinates to Cartesian coordinates
        y = (y0_at_planes[i_plane] + this_image_r_meshgrid * np.cos(theta_meshgrid)) / length_per_pixel
        z = (z0_at_planes[i_plane] + this_image_r_meshgrid * np.sin(theta_meshgrid)) / length_per_pixel;
        
        # Ensure coordinates are within image bounds
        y = np.clip(y, 0, width - 1)
        z = np.clip(z, 0, height - 1)
        #plt.ion();plt.figure();plt.scatter(coordinates[0,:],coordinates[1,:]);
        
        # Interpolate the image values at the Cartesian coordinates
        coordinates = np.array([y.flatten(), z.flatten()])
        image_array_circular = map_coordinates(fluence_exp_images_cartesian[i_plane,:,:], coordinates, order=2, mode='nearest').reshape(this_image_r_meshgrid.shape)
        mask_radius = ((y*length_per_pixel-y0_at_planes[i_plane])**2+(z*length_per_pixel-z0_at_planes[i_plane])**2) > r_max_coordinate_change**2
        image_array_circular[mask_radius] = 0.
        
        # place the image in the larger array, the rest of the array will be zero
        fluence_exp_images_circular[i_plane,:this_image_r_mesh.size,:] = image_array_circular
        
        # try to make the transition smoother at theta=0=2pi
        fluence_exp_images_circular[i_plane,:,-1] = 0.5*(fluence_exp_images_circular[i_plane,:,0]+fluence_exp_images_circular[i_plane,:,-2])

    ### The total energy of the cartesian images must be the same of the total energy of the circular images,
    ### for each plane

    # Find the integration steps
    dr            = length_per_pixel
    dtheta        = theta_mesh[1]-theta_mesh[0]

    # ensure that the total energy of all the circular images remains constant during the propagation along the planes
    # the total energy value will be the one at the plane i_plane=index_reference_plane
    # this energy will be in fluence * m * m units
    energy_reference_plane = dict_image_preprocessing["energy_reference_plane"]
    for i_plane in range(number_of_planes):
        integrand         = fluence_exp_images_circular[i_plane, :, :]
        # Integrate using sum over r and theta, multiply by dr and dtheta
        energy_this_plane = np.sum(integrand * r_mesh[:,np.newaxis]) * dr * dtheta 
        fluence_exp_images_circular[i_plane,:,:] *=energy_reference_plane/energy_this_plane

    # ensure that the interpolation did not create negative values
    # if this happens, NaN values are created very soon in the GSA-MD
    fluence_exp_images_circular[fluence_exp_images_circular < 0.] = 0.
    
    # save a the experimental images also in a centered cartesian grid, to check the accuracy of the coordinate change
    dict_image_preprocessing,dict_mesh = transfer_fluence_images_to_cartesian_grid(dict_image_preprocessing,dict_mesh,ny=2*np.size(r_mesh),nz=2*np.size(r_mesh))
    
    # Save the new variables
    dict_image_preprocessing["fluence_exp_images_circular"] = fluence_exp_images_circular 
    dict_image_preprocessing["energy_reference_plane"     ] = energy_reference_plane
    dict_mesh               ["r_mesh"                     ] = r_mesh
    dict_mesh               ["theta_mesh"                 ] = theta_mesh
    
    return dict_image_preprocessing,dict_mesh
    
def transfer_fluence_images_to_cartesian_grid(dict_image_preprocessing,dict_mesh,ny,nz):
    
    fluence_exp_images_cartesian       = dict_image_preprocessing["filtered_images_no_background_noise"]
    index_reference_plane              = dict_image_preprocessing["index_reference_plane"              ]
    length_per_pixel                   = dict_image_preprocessing["length_per_pixel"                   ]
    y0_at_planes                       = dict_image_preprocessing["y0_at_planes"                       ]
    z0_at_planes                       = dict_image_preprocessing["z0_at_planes"                       ]
    r_max_coordinate_change            = dict_image_preprocessing["r_max_coordinate_change"            ]
    
    # Image dimensions
    number_of_planes,width,height      = fluence_exp_images_cartesian.shape
    
    # Ensure that the image can be placed exactly at the center of a larger image with the requested size
    if ((width%2) != (ny%2)):
        print("If the number of pixels of the input image along axis 0 is even (odd), then ny_converted_image should be even (odd)")
        print("Here ",width," pixel points are present on axis 0, but ny_converted_image = ",ny)
        sys.exit()
    if ((height%2) != (nz%2)):
        print("If the number of pixels of the input image along axis 1 is even (odd), then nz_converted_image should be even (odd)")
        print("Here ",height," pixel points are present on axis 1, but nz_converted_image = ",nz)
        sys.exit()
    
    # Generate the y mesh and z mesh, keeping the same resolution of the initial image (i.e. the length_per_pixel)
    # the center of the mesh will be always the origin
    # with an even number of points, it will be (-(N/2-1/2),...,-1,0,+1,...,(N/2-1/2))*length_per_pixel
    # with an odd number of points, it will be  (-(N/2-1/2),...,-1,0,+1,...,(N/2-1/2))*length_per_pixel
    # this way the Hermite Gauss can be always centered in the origin
    y_mesh = length_per_pixel * np.arange(0, ny)
    y_mesh = y_mesh-y_mesh.max()/2.
    z_mesh = length_per_pixel * np.arange(0, nz)
    z_mesh = z_mesh-z_mesh.max()/2.
    
    # This array will store images converted in polar coordinates
    new_fluence_exp_images_cartesian = np.zeros(shape=(number_of_planes,ny,nz))

    for i_plane in range(0,number_of_planes):
        
        # Define the local mesh
        this_image_y_mesh     = length_per_pixel*np.arange(0,width )
        this_image_y_mesh     = this_image_y_mesh-this_image_y_mesh.max()/2.
        this_image_z_mesh     = length_per_pixel*np.arange(0,height)
        this_image_z_mesh     = this_image_z_mesh-this_image_z_mesh.max()/2.
        
        # Create a grid of radius and theta
        this_image_y_meshgrid, this_image_z_meshgrid = np.meshgrid(this_image_y_mesh,this_image_z_mesh,indexing='ij')
        
        # Find the local coordinates of each pixel point, shifting by the image center coordinates
        y = ( y0_at_planes[i_plane]+this_image_y_meshgrid ) / length_per_pixel
        z = ( z0_at_planes[i_plane]+this_image_z_meshgrid ) / length_per_pixel;
        
        
        # Ensure coordinates are within image bounds
        y = np.clip(y, 0, width - 1)
        z = np.clip(z, 0, height - 1)
        #plt.ion();plt.figure();plt.scatter(coordinates[0,:],coordinates[1,:]);
        
        # Interpolate the image values at the Cartesian coordinates
        coordinates = np.array([y.flatten(), z.flatten()])
        interpolated_image_array = map_coordinates(fluence_exp_images_cartesian[i_plane,:,:], coordinates, order=2, mode='nearest').reshape(this_image_y_meshgrid.shape)
        
        mask_radius = ((y*length_per_pixel-y0_at_planes[i_plane])**2+(z*length_per_pixel-z0_at_planes[i_plane])**2) > r_max_coordinate_change**2
        interpolated_image_array[mask_radius] = 0.
        
        # Get the dimensions of the arrays
        small_height, small_width = interpolated_image_array.shape
        large_height, large_width = new_fluence_exp_images_cartesian[i_plane,:,:].shape
        if (small_height>large_height) or (small_width>large_width):
            print("Error: choose a larger size for the converted image")
            sys.exit()

        # Calculate the starting indices for centering the small array
        start_y = (large_height - small_height) // 2
        start_x = (large_width  - small_width ) // 2

        # Place the small array in the center of the large array
        new_fluence_exp_images_cartesian[i_plane,start_y:start_y + small_height, start_x:start_x + small_width] = interpolated_image_array

    ### The total energy of the cartesian images must be the same of the total energy of the circular images,
    ### for each plane

    # Find the integration steps
    dy            = length_per_pixel
    dz            = length_per_pixel

    # ensure that the total energy of all the circular images remains constant during the propagation along the planes
    # the total energy value will be the one at the plane i_plane=index_reference_plane
    # this energy will be in fluence * m * m units
    energy_reference_plane = energy_reference_plane = dict_image_preprocessing["energy_reference_plane"]
    
    for i_plane in range(number_of_planes):
        # Integrate using sum over y and z, multiply by dy and dz
        energy_this_plane = np.sum(new_fluence_exp_images_cartesian[i_plane, :, :]) * dy * dz
        new_fluence_exp_images_cartesian[i_plane,:,:] *=energy_reference_plane/energy_this_plane

    # ensure that the interpolation did not create negative values
    # if this happens, NaN values are created very soon in the GSA-MD
    new_fluence_exp_images_cartesian[new_fluence_exp_images_cartesian < 0] = 0.0
    
    # Save the new variables
    dict_image_preprocessing["fluence_exp_images_cartesian"] = new_fluence_exp_images_cartesian
    dict_mesh               ["y_mesh"                      ] = y_mesh
    dict_mesh               ["z_mesh"                      ] = z_mesh
    
    return dict_image_preprocessing,dict_mesh
    
def create_mask_inside_r_max(dict_image_preprocessing,dict_mesh):
    
    ## define a mask equal to False for r > r_max_coordinate_change 
    
    if dict_image_preprocessing["geometry"] == "cylindrical":
        
        r_mesh       = dict_mesh["r_mesh"    ]
        theta_mesh   = dict_mesh["theta_mesh"]
        
        # create a boolean mask to keep only the data inside the max radius (e.g. for error calculation)
        inside_r_max = np.zeros((r_mesh.size, theta_mesh.size), dtype=bool)
        
        for itheta in range(theta_mesh.size):
            inside_r_max[:, itheta] = r_mesh < dict_image_preprocessing["r_max_coordinate_change"]
        
    else:
        
        y_mesh       = dict_mesh["y_mesh"]
        z_mesh       = dict_mesh["z_mesh"]
        y2_mesh      = np.square(y_mesh[:,np.newaxis]*np.ones(shape=(y_mesh.size,z_mesh.size)))
        z2_mesh      = np.square(z_mesh[np.newaxis,:]*np.ones(shape=(y_mesh.size,z_mesh.size)))
        inside_r_max = (y2_mesh+z2_mesh) < dict_image_preprocessing["r_max_coordinate_change"]**2 
    
    # create a boolean mask to keep only the data inside the max radius (e.g. for error calculation)
    dict_image_preprocessing["inside_r_max"] = inside_r_max
    
    return dict_image_preprocessing
    