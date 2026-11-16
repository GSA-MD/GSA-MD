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
    ### - compute_effective_r_max
    ### - convert_images_from_cartesian_to_circular_coordinates (if geometry=="cylindrical")
    ### - convert_images_cartesian_coordinates                  (if geometry=="cartesian")
    ### - create_mask_inside_r_max
    ### In the process, if not already present, the following data are added 
    ### to the dictionaries provided as input
    ### - image_center_coordinates
    ### - r_max_effective (radius actually used to cut the images and to define the mask inside_r_max)
    ### - measured_region_circular and/or measured_region_cartesian (points of the converted images that were measured)
    ### - inside_r_max_and_measured (for each plane, points inside r_max_effective that were measured)
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
    
    ##### Check if the disc of radius r_max_coordinate_change around the center of each image
    ##### is fully contained in that image, and define r_max_effective
    dict_image_preprocessing            = compute_effective_r_max(dict_image_preprocessing)

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
    
    ## define a mask equal to False for r > r_max_effective,
    ## and a mask for each plane equal to False for r > r_max_effective or where the image was not measured
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
    
    # if True, the disc of radius r_max_coordinate_change can be partially outside some images
    if "allow_partially_measured_planes" not in dict_image_preprocessing.keys():
        dict_image_preprocessing["allow_partially_measured_planes"] = False

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
    print("- allow_partially_measured_planes    : ",dict_image_preprocessing["allow_partially_measured_planes"])
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
    
def compute_effective_r_max(dict_image_preprocessing):

    ### The radius r_max_effective is used to cut the images (and to define the mask inside_r_max).
    ### The distance between the center of an image and the nearest edge of that image
    ### is measured from the center to the center of the last pixel of the image.
    ### If allow_partially_measured_planes = False, all the points with distance from the center
    ### smaller than r_max_effective must be points measured in all the images. Therefore:
    ### - if the user provided r_max_coordinate_change larger than the distance between
    ###   the center of an image and the nearest edge of that image, the program stops with an error;
    ### - if the user did not provide r_max_coordinate_change (default = np.inf), r_max_effective is the
    ###   minimum over all images of the distance between the image center and the nearest image edge;
    ### - otherwise r_max_effective = r_max_coordinate_change.
    ### Otherwise, points outside the input image would be filled with zeros or
    ### with the values at the image edges, and treated as measured data.
    ### If allow_partially_measured_planes = True, r_max_coordinate_change must be provided and
    ### r_max_effective = r_max_coordinate_change. The disc of radius r_max_effective can be partially
    ### outside some images: the points of the disc outside an image are treated as not measured
    ### (no amplitude correction is applied there during the reconstruction) and a warning is printed.

    fluence_exp_images_cartesian       = dict_image_preprocessing["filtered_images_no_background_noise"]
    length_per_pixel                   = dict_image_preprocessing["length_per_pixel"                   ]
    y0_at_planes                       = dict_image_preprocessing["y0_at_planes"                       ]
    z0_at_planes                       = dict_image_preprocessing["z0_at_planes"                       ]
    r_max_coordinate_change            = dict_image_preprocessing["r_max_coordinate_change"            ]
    index_reference_plane              = dict_image_preprocessing["index_reference_plane"              ]
    allow_partially_measured_planes    = dict_image_preprocessing["allow_partially_measured_planes"    ]

    number_of_planes,width,height      = fluence_exp_images_cartesian.shape

    # distance between the center of each image and the nearest edge of that image
    distance_center_to_nearest_edge    = np.zeros(number_of_planes)
    for i_plane in range(0,number_of_planes):
        distance_center_to_nearest_edge[i_plane] = np.amin([ y0_at_planes[i_plane],
                                                             (width -1)*length_per_pixel - y0_at_planes[i_plane],
                                                             z0_at_planes[i_plane],
                                                             (height-1)*length_per_pixel - z0_at_planes[i_plane] ])

    index_limiting_plane               = np.argmin(distance_center_to_nearest_edge)
    max_radius_inside_all_images       = distance_center_to_nearest_edge[index_limiting_plane]

    if max_radius_inside_all_images <= 0.:
        print("Error: the center of the image at plane index ",index_limiting_plane," is on the image edge or outside the image")
        print("y0 = ",y0_at_planes[index_limiting_plane]," m, z0 = ",z0_at_planes[index_limiting_plane]," m")
        sys.exit()

    if allow_partially_measured_planes:
        if not np.isfinite(r_max_coordinate_change):
            print("Error: with allow_partially_measured_planes = True, a finite r_max_coordinate_change must be provided")
            sys.exit()
        r_max_effective                = r_max_coordinate_change
        print("r_max_effective = r_max_coordinate_change = ",r_max_effective," m")
        partially_measured_planes      = np.where(distance_center_to_nearest_edge < r_max_effective)[0]
        if partially_measured_planes.size > 0:
            print("WARNING: the disc of radius r_max_effective around the image center is partially outside the image at the planes:")
            for i_plane in partially_measured_planes:
                print(f"         - plane index {i_plane}: distance between the center and the nearest image edge = {distance_center_to_nearest_edge[i_plane]:.3e} m")
            print("         In these planes, the points outside the image are treated as not measured.")
            print("         The fluence of each image is still normalized to the total fluence of the reference plane,")
            print("         i.e. the fluence of these images is overestimated by the fraction of energy outside the image.")
            if (index_reference_plane % number_of_planes) in partially_measured_planes:
                print("WARNING: the reference plane is partially measured: its total fluence, used for the normalization,")
                print("         does not include the energy outside its image.")
    else:
        if r_max_coordinate_change > max_radius_inside_all_images:
            if np.isfinite(r_max_coordinate_change):
                # a radius provided by the user that goes beyond the edge of at least one image is not accepted
                print("Error: r_max_coordinate_change = ",r_max_coordinate_change," m is larger than the distance between")
                print("       the center and the nearest edge of the image at plane index ",index_limiting_plane,",")
                print("       i.e. part of the disc of radius r_max_coordinate_change is outside that image.")
                print("       Choose r_max_coordinate_change <= ",max_radius_inside_all_images," m,")
                print("       or improve the centering of the beam in the images (or use larger images),")
                print("       or set allow_partially_measured_planes = True.")
                sys.exit()
            else:
                # r_max_coordinate_change not provided by the user (default = np.inf):
                # use the largest radius around the image centers contained in all the images
                r_max_effective        = max_radius_inside_all_images
                print("r_max_coordinate_change not provided:")
                print("r_max_effective set to ",r_max_effective," m, i.e. the largest radius around the image centers")
                print("contained in all the images (limited by the image at plane index ",index_limiting_plane,")")
        else:
            r_max_effective            = r_max_coordinate_change
            print("r_max_effective = r_max_coordinate_change = ",r_max_effective," m")
    print("\n")

    dict_image_preprocessing["r_max_effective"] = r_max_effective

    return dict_image_preprocessing


def transfer_fluence_images_to_cylindrical_grid(dict_image_preprocessing,dict_mesh):
    
    fluence_exp_images_cartesian  = dict_image_preprocessing["filtered_images_no_background_noise"]
    index_reference_plane         = dict_image_preprocessing["index_reference_plane"              ]
    length_per_pixel              = dict_image_preprocessing["length_per_pixel"                   ]
    y0_at_planes                  = dict_image_preprocessing["y0_at_planes"                       ]
    z0_at_planes                  = dict_image_preprocessing["z0_at_planes"                       ]
    r_max_effective               = dict_image_preprocessing["r_max_effective"                    ]
    nr_converted_image            = dict_mesh               ["nr_converted_image"                 ]
    ntheta                        = dict_mesh               ["ntheta"                             ]

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

    # This array will store, for each plane, the points of the polar grid that were measured, i.e. inside the input image
    measured_region_circular    = np.zeros(shape=(number_of_planes,nr_converted_image,ntheta),dtype=bool)

    # Create a grid of radius and theta
    r_meshgrid, theta_meshgrid  = np.meshgrid(r_mesh,theta_mesh,indexing='ij')

    for i_plane in range(0,number_of_planes):
        
        # Convert polar coordinates to Cartesian coordinates (in pixel units)
        y_pixel_unclipped = (y0_at_planes[i_plane] + r_meshgrid * np.cos(theta_meshgrid)) / length_per_pixel
        z_pixel_unclipped = (z0_at_planes[i_plane] + r_meshgrid * np.sin(theta_meshgrid)) / length_per_pixel
        
        # Points of the polar grid inside the input image
        measured_region_circular[i_plane,:,:] = (y_pixel_unclipped >= 0) & (y_pixel_unclipped <= width -1) \
                                              & (z_pixel_unclipped >= 0) & (z_pixel_unclipped <= height-1)
        
        # Ensure coordinates are within image bounds
        y = np.clip(y_pixel_unclipped, 0, width - 1 )
        z = np.clip(z_pixel_unclipped, 0, height - 1)
        
        # Interpolate the image values at the Cartesian coordinates
        coordinates = np.array([y.flatten(), z.flatten()])
        image_array_circular = map_coordinates(fluence_exp_images_cartesian[i_plane,:,:], coordinates, order=2, mode='nearest').reshape(r_meshgrid.shape)
        
        # The fluence is set to zero beyond r_max_effective and outside the input image
        mask_radius = r_meshgrid > r_max_effective
        image_array_circular[mask_radius | ~measured_region_circular[i_plane,:,:]] = 0.
        fluence_exp_images_circular[i_plane,:,:] = image_array_circular
        
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
    dict_image_preprocessing["measured_region_circular"   ] = measured_region_circular
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
    r_max_effective                    = dict_image_preprocessing["r_max_effective"                    ]
    
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
    # This array will store, for each plane, the points of the new grid that were measured, i.e. inside the input image
    measured_region_cartesian        = np.zeros(shape=(number_of_planes,ny,nz),dtype=bool)

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
        
        
        # Points of the new grid inside the input image
        this_image_measured_region = (y >= 0) & (y <= width -1) & (z >= 0) & (z <= height-1)

        # Ensure coordinates are within image bounds
        y = np.clip(y, 0, width - 1)
        z = np.clip(z, 0, height - 1)
        #plt.ion();plt.figure();plt.scatter(coordinates[0,:],coordinates[1,:]);
        
        # Interpolate the image values at the Cartesian coordinates
        coordinates = np.array([y.flatten(), z.flatten()])
        interpolated_image_array = map_coordinates(fluence_exp_images_cartesian[i_plane,:,:], coordinates, order=2, mode='nearest').reshape(this_image_y_meshgrid.shape)
        
        # The fluence is set to zero beyond r_max_effective and outside the input image.
        # The mask must use the distance from the center of the point on the new grid,
        # i.e. not the clipped coordinates y,z. Otherwise the points outside the input image
        # would be clipped on its edge, considered inside r_max and filled with the values at the image edge.
        mask_radius = (this_image_y_meshgrid**2+this_image_z_meshgrid**2) > r_max_effective**2
        interpolated_image_array[mask_radius | ~this_image_measured_region] = 0.
        
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
        measured_region_cartesian       [i_plane,start_y:start_y + small_height, start_x:start_x + small_width] = this_image_measured_region

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
    dict_image_preprocessing["measured_region_cartesian"   ] = measured_region_cartesian
    dict_mesh               ["y_mesh"                      ] = y_mesh
    dict_mesh               ["z_mesh"                      ] = z_mesh
    
    return dict_image_preprocessing,dict_mesh
    
def create_mask_inside_r_max(dict_image_preprocessing,dict_mesh):
    
    ## define a mask equal to False for r > r_max_effective (inside_r_max),
    ## and for each plane a mask equal to False for r > r_max_effective or where the image was not measured (inside_r_max_and_measured)

    r_max_effective  = dict_image_preprocessing["r_max_effective"]
    
    if dict_image_preprocessing["geometry"] == "cylindrical":
        
        r_mesh       = dict_mesh["r_mesh"    ]
        theta_mesh   = dict_mesh["theta_mesh"]
        
        # create a boolean mask to keep only the data inside the max radius (e.g. for error calculation)
        inside_r_max = np.zeros((r_mesh.size, theta_mesh.size), dtype=bool)
        
        for itheta in range(theta_mesh.size):
            inside_r_max[:, itheta] = r_mesh < r_max_effective

        measured_region        = dict_image_preprocessing["measured_region_circular"]
        # weights for the area of each grid point
        weights_area           = r_mesh[:,np.newaxis]*np.ones(shape=(r_mesh.size,theta_mesh.size))
        max_radius_of_the_grid = r_mesh.max()
        
    else:
        
        y_mesh       = dict_mesh["y_mesh"]
        z_mesh       = dict_mesh["z_mesh"]
        y2_mesh      = np.square(y_mesh[:,np.newaxis]*np.ones(shape=(y_mesh.size,z_mesh.size)))
        z2_mesh      = np.square(z_mesh[np.newaxis,:]*np.ones(shape=(y_mesh.size,z_mesh.size)))
        inside_r_max = (y2_mesh+z2_mesh) < r_max_effective**2

        measured_region        = dict_image_preprocessing["measured_region_cartesian"]
        weights_area           = np.ones(shape=(y_mesh.size,z_mesh.size))
        max_radius_of_the_grid = np.minimum(y_mesh.max(),z_mesh.max())

    if dict_image_preprocessing["allow_partially_measured_planes"] and (r_max_effective > max_radius_of_the_grid):
        print("Error: r_max_coordinate_change = ",r_max_effective," m is larger than the grid used for the reconstruction")
        print("       (maximum radius = ",max_radius_of_the_grid," m). Choose a smaller r_max_coordinate_change or a larger grid.")
        sys.exit()

    inside_r_max_and_measured  = inside_r_max[np.newaxis,:,:] & measured_region

    # print the fraction of the disc of radius r_max_effective inside the image, for the partially measured planes
    for i_plane in range(inside_r_max_and_measured.shape[0]):
        measured_fraction      = np.sum(weights_area[inside_r_max_and_measured[i_plane,:,:]])/np.sum(weights_area[inside_r_max])
        if measured_fraction < 1.:
            print(f"Plane index {i_plane}: fraction of the disc of radius r_max_effective inside the image = {measured_fraction:.3f}")

    dict_image_preprocessing["inside_r_max_and_measured"] = inside_r_max_and_measured
    
    # create a boolean mask to keep only the data inside the max radius (e.g. for error calculation)
    dict_image_preprocessing["inside_r_max"] = inside_r_max
    
    return dict_image_preprocessing
    