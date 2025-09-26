###### Authors: I. Moulanier, F. Massimo, A. Guerente

###### This module contains plot functions for the GSA, 
###### working both with and without Mode Decomposition 
###### All the plotted fields should already be in the dictionaries 
###### used as inputs of these functions




import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.interpolate import griddata
import os,sys
import matplotlib.colors as mcolors

#import scienceplots

#plt.style.use('science')

# Define the colors for the colormap
colors = [(1.        , 1.        , 1.       ),   # white
          (0.        , 0.21875   , 0.65625  ),   # blue
          (0.60546875, 0.30859375, 0.5859375),   # purple
          (0.8359375 , 0.0078125 , 0.4375   )]   # pink

# Create the colormap
cmap_name = 'my_cmap'
my_cmap   = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors, N=256)


def plot_error_evolution(dict_mesh,dict_GSA,fig_error_vs_iteration=None,fig_error_vs_time=None,label="",linestyle="-",colors=["c","b"]):
    
    # you can provide an existing plot figure in the arguments to superpose the lines
    # the label is added to the usual label, this helps creating comparison plots
    
    ### make a figure of error vs iteration
    if fig_error_vs_iteration == None:
        fig_error_vs_iteration = plt.figure()
    else:
        plt.figure(fig_error_vs_iteration.number);
    
    
    N_iterations               = dict_GSA["N_iterations"              ]
    iterations_between_outputs = dict_GSA["iterations_between_outputs"]
    
    # Calculate mean and std across the planes for each iteration
    error_mean_iter = np.mean(dict_GSA["error_fit"], axis=1)    # Mean across planes
    error_std_iter  = np.std (dict_GSA["error_fit"], axis=1)    # Std across planes
    
    iters_outputs              = []
    for iter in range(0,N_iterations):
        if (iter%dict_GSA["iterations_between_outputs"]==0) or (iter==dict_GSA["N_iterations"]-1):
            iters_outputs.append(iter)
    
    for i_plane in range(np.size(dict_mesh["plane_x_coordinates"])):
        plt.plot(np.array(iters_outputs),dict_GSA["error_fit"][:,i_plane],linestyle=linestyle,c=colors[0])
    
    # Plot mean error vs iteration with shaded area for std
    plt.plot(np.array(iters_outputs), error_mean_iter, label=label, linestyle=linestyle,linewidth=2,c=colors[1])
    plt.fill_between(np.array(iters_outputs), 
                     error_mean_iter - error_std_iter, 
                     error_mean_iter + error_std_iter, 
                     alpha=0.2,color=colors[0])
    
    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("Reconstruction Error")
    plt.ylim(0.,1.01)
    
    
    ### make a figure of error vs time from the first output
    if fig_error_vs_time == None:
        fig_error_vs_time = plt.figure()
    else:
        plt.figure(fig_error_vs_time.number);
    
    for i_plane in range(np.size(dict_mesh["plane_x_coordinates"])):
        plt.plot(dict_GSA["time_output"],dict_GSA["error_fit"][:,i_plane],linestyle=linestyle,c=colors[0])
    
    # Plot mean error vs iteration with shaded area for std
    plt.plot(dict_GSA["time_output"], error_mean_iter, label=label, linestyle=linestyle,c=colors[1],linewidth=2)
    plt.fill_between(dict_GSA["time_output"], 
                     error_mean_iter - error_std_iter, 
                     error_mean_iter + error_std_iter, 
                     alpha=0.2,color=colors[0])
    
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Reconstruction Error")
    plt.ylim(0.,1.01)
    
    
    return fig_error_vs_iteration,fig_error_vs_time

def compute_width_at_level_times_peak_value(x,y,level=0.5):
    
    ##### Given a curve y=f(x) with a peak y_peak, this function computes the 
    ##### width (in x units) of the peak, defined as the difference between the first x positions around the peak which
    ##### have y = level*y_peak.
    ##### when level = 0.5 you are computing the FWHM;
    ##### when level = 1 / math.e ** 2, you are computing the 1/e^2 width
    
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

    
def plot_diffraction(dict_image_preprocessing,dict_mesh,dict_GSA,fig_1e2_width_y=None,fig_FWHM_y=None,fig_1e2_width_z=None,fig_FWHM_z=None,plot_cartesian_experimental_fluence=True,plot_cylindrical_experimental_fluence=True,only_markers=False):

    plane_x_coordinates         = dict_mesh["plane_x_coordinates"]

    ### Extract the data depending on the geometry

    if dict_image_preprocessing["geometry"] == "cylindrical":

        axis_y_circular               = np.hstack([-np.flipud(dict_mesh["r_mesh"]),dict_mesh["r_mesh"]])
        axis_z_circular               = np.hstack([-np.flipud(dict_mesh["r_mesh"]),dict_mesh["r_mesh"]])
        axis_z_cartesian              = dict_mesh  ["y_mesh"]
        axis_y_cartesian              = dict_mesh  ["z_mesh"] 

        fluence_reconstruction_axis_y = dict_GSA["fluence_reconstruction_circular_axis_y"]
        fluence_reconstruction_axis_z = dict_GSA["fluence_reconstruction_circular_axis_z"]  
        fluence_exp_circular_axis_y   = dict_GSA["fluence_exp_circular_axis_y"           ] 
        fluence_exp_circular_axis_z   = dict_GSA["fluence_exp_circular_axis_z"           ] 
        fluence_exp_cartesian_axis_y  = dict_GSA["fluence_exp_cartesian_axis_y"          ] 
        fluence_exp_cartesian_axis_z  = dict_GSA["fluence_exp_cartesian_axis_z"          ] 

    else:

        axis_y_cartesian              = dict_mesh  ["y_mesh"]  
        axis_z_cartesian              = dict_mesh  ["z_mesh"] 

        fluence_reconstruction_axis_y = dict_GSA["fluence_reconstruction_cartesian_axis_y"] 
        fluence_reconstruction_axis_z = dict_GSA["fluence_reconstruction_cartesian_axis_z"] 
        fluence_exp_cartesian_axis_y  = dict_GSA["fluence_exp_cartesian_axis_y"           ] 
        fluence_exp_cartesian_axis_z  = dict_GSA["fluence_exp_cartesian_axis_z"           ] 

    # Compute the 1/e^2 amplitude and FWHM
    one_ov_e2_EXP_circular_axis_y     = np.zeros_like(plane_x_coordinates)
    one_ov_e2_EXP_cartesian_axis_y    = np.zeros_like(plane_x_coordinates)
    one_ov_e2_GSA_axis_y              = np.zeros_like(plane_x_coordinates)
    one_ov_e2_EXP_circular_axis_z     = np.zeros_like(plane_x_coordinates) 
    one_ov_e2_EXP_cartesian_axis_z    = np.zeros_like(plane_x_coordinates) 
    one_ov_e2_GSA_axis_z              = np.zeros_like(plane_x_coordinates) 

    FWHM_EXP_circular_axis_y          = np.zeros_like(plane_x_coordinates)
    FWHM_EXP_cartesian_axis_y         = np.zeros_like(plane_x_coordinates)
    FWHM_GSA_axis_y                   = np.zeros_like(plane_x_coordinates)
    FWHM_EXP_circular_axis_z          = np.zeros_like(plane_x_coordinates) 
    FWHM_EXP_cartesian_axis_z         = np.zeros_like(plane_x_coordinates) 
    FWHM_GSA_axis_z                   = np.zeros_like(plane_x_coordinates) 

    if dict_image_preprocessing["geometry"] == "cylindrical":

        for i_plane in range(np.size(plane_x_coordinates)):

            # compute 1/e^2 width
            one_ov_e2_EXP_circular_axis_y [i_plane] = compute_width_at_level_times_peak_value(axis_y_circular , fluence_exp_circular_axis_y  [i_plane,:],level=(1/math.e**2))
            one_ov_e2_EXP_cartesian_axis_y[i_plane] = compute_width_at_level_times_peak_value(axis_y_cartesian, fluence_exp_cartesian_axis_y [i_plane,:],level=(1/math.e**2))
            one_ov_e2_GSA_axis_y          [i_plane] = compute_width_at_level_times_peak_value(axis_y_circular , fluence_reconstruction_axis_y[i_plane,:],level=(1/math.e**2))
            one_ov_e2_EXP_circular_axis_z [i_plane] = compute_width_at_level_times_peak_value(axis_z_circular , fluence_exp_circular_axis_z  [i_plane,:],level=(1/math.e**2))
            one_ov_e2_EXP_cartesian_axis_z[i_plane] = compute_width_at_level_times_peak_value(axis_z_cartesian, fluence_exp_cartesian_axis_z [i_plane,:],level=(1/math.e**2))
            one_ov_e2_GSA_axis_z          [i_plane] = compute_width_at_level_times_peak_value(axis_z_circular , fluence_reconstruction_axis_z[i_plane,:],level=(1/math.e**2))

            # compute FWHM
            FWHM_EXP_circular_axis_y      [i_plane] = compute_width_at_level_times_peak_value(axis_y_circular , fluence_exp_circular_axis_y  [i_plane,:],level=(1/2.       ))
            FWHM_EXP_cartesian_axis_y     [i_plane] = compute_width_at_level_times_peak_value(axis_y_cartesian, fluence_exp_cartesian_axis_y [i_plane,:],level=(1/2.       ))
            FWHM_GSA_axis_y               [i_plane] = compute_width_at_level_times_peak_value(axis_y_circular , fluence_reconstruction_axis_y[i_plane,:],level=(1/2.       ))
            FWHM_EXP_circular_axis_z      [i_plane] = compute_width_at_level_times_peak_value(axis_z_circular , fluence_exp_circular_axis_z  [i_plane,:],level=(1/2.       ))
            FWHM_EXP_cartesian_axis_z     [i_plane] = compute_width_at_level_times_peak_value(axis_z_cartesian, fluence_exp_cartesian_axis_z [i_plane,:],level=(1/2.       ))
            FWHM_GSA_axis_z               [i_plane] = compute_width_at_level_times_peak_value(axis_z_circular , fluence_reconstruction_axis_z[i_plane,:],level=(1/2.       ))

    else:

        for i_plane in range(np.size(plane_x_coordinates)):

            # compute 1/e^2 width
            one_ov_e2_EXP_cartesian_axis_y[i_plane] = compute_width_at_level_times_peak_value(axis_y_cartesian, fluence_exp_cartesian_axis_y [i_plane,:],level=(1/math.e**2))
            one_ov_e2_GSA_axis_y          [i_plane] = compute_width_at_level_times_peak_value(axis_y_cartesian, fluence_reconstruction_axis_y[i_plane,:],level=(1/math.e**2))
            one_ov_e2_EXP_cartesian_axis_z[i_plane] = compute_width_at_level_times_peak_value(axis_z_cartesian, fluence_exp_cartesian_axis_z [i_plane,:],level=(1/math.e**2))
            one_ov_e2_GSA_axis_z          [i_plane] = compute_width_at_level_times_peak_value(axis_z_cartesian, fluence_reconstruction_axis_z[i_plane,:],level=(1/math.e**2))

            # compute FWHM
            FWHM_EXP_cartesian_axis_y     [i_plane] = compute_width_at_level_times_peak_value(axis_y_cartesian, fluence_exp_cartesian_axis_y [i_plane,:],level=(1/2.       ))
            FWHM_GSA_axis_y               [i_plane] = compute_width_at_level_times_peak_value(axis_y_cartesian, fluence_reconstruction_axis_y[i_plane,:],level=(1/2.       ))
            FWHM_EXP_cartesian_axis_z     [i_plane] = compute_width_at_level_times_peak_value(axis_z_cartesian, fluence_exp_cartesian_axis_z [i_plane,:],level=(1/2.       ))
            FWHM_GSA_axis_z               [i_plane] = compute_width_at_level_times_peak_value(axis_z_cartesian, fluence_reconstruction_axis_z[i_plane,:],level=(1/2.       ))


    # Plot 1/e^2 and FWHM evolution during propagation
    if dict_image_preprocessing["geometry"] == "cylindrical":
        grid_label = "cyl."
        color_GSA  = "limegreen"
    else:
        grid_label = "cart."
        color_GSA = "fuchsia"

    if fig_1e2_width_y == None:
        fig_1e2_width_y = plt.figure()
    else:
        plt.figure(fig_1e2_width_y.number);

    if only_markers:
        alpha_lines = 0.
    else:
        alpha_lines = 1.
    # y axis
    plt.scatter    (plane_x_coordinates/1e-6,one_ov_e2_GSA_axis_y/1e-6          ,label="GSA, y, "+grid_label+" grid" ,c=color_GSA,linestyle="--")
    plt.plot       (plane_x_coordinates/1e-6,one_ov_e2_GSA_axis_y/1e-6          ,label="_GSA, y, "+grid_label+" grid",c=color_GSA,lw=2,alpha=alpha_lines,linestyle="--")
    if plot_cartesian_experimental_fluence:
        plt.scatter(plane_x_coordinates/1e-6,one_ov_e2_EXP_cartesian_axis_y/1e-6,label="EXP, y, cart. grid",c="k",linestyle="--")
        plt.plot   (plane_x_coordinates/1e-6,one_ov_e2_EXP_cartesian_axis_y/1e-6,label="_EXP, y, cart. grid",linestyle="--",dashes=(2, 2),c="k",lw=2,alpha=alpha_lines)
    if (dict_image_preprocessing["geometry"] == "cylindrical") & (plot_cylindrical_experimental_fluence):
        plt.scatter(plane_x_coordinates/1e-6,one_ov_e2_EXP_circular_axis_y/1e-6 ,label="EXP, y, cyl. grid",c="c",linestyle="--")
        plt.plot   (plane_x_coordinates/1e-6,one_ov_e2_EXP_circular_axis_y/1e-6 ,label="_EXP, y, cyl. grid",linestyle="--",dashes=(2, 4),c="c",lw=2,alpha=alpha_lines)

    plt.xlabel("x (um)")
    plt.ylabel("1/e^2 width_y (um)")
    plt.legend()
    plt.ylim(0,)
    
    if fig_1e2_width_z == None:
        fig_1e2_width_z = plt.figure()
    else:
        plt.figure(fig_1e2_width_z.number);
        
    # z axis
    color_GSA = "fuchsia"
    if dict_image_preprocessing["geometry"] == "cylindrical":
        color_GSA = "limegreen" 
    plt.scatter    (plane_x_coordinates/1e-6,one_ov_e2_GSA_axis_z/1e-6          ,label="GSA, z, "+grid_label+ "grid" ,c=color_GSA,linestyle="--")
    plt.plot       (plane_x_coordinates/1e-6,one_ov_e2_GSA_axis_z/1e-6          ,label="_GSA, z, "+grid_label+ "grid",c=color_GSA,lw=2,alpha=alpha_lines,linestyle="--")
    if plot_cartesian_experimental_fluence:
        plt.scatter(plane_x_coordinates/1e-6,one_ov_e2_EXP_cartesian_axis_z/1e-6,label="EXP, z, cart. grid",c="k",linestyle="--")
        plt.plot   (plane_x_coordinates/1e-6,one_ov_e2_EXP_cartesian_axis_z/1e-6,label="_EXP, z, cart. grid",linestyle="--",dashes=(2, 2),c="k",lw=2,alpha=alpha_lines)
    if (dict_image_preprocessing["geometry"] == "cylindrical") & (plot_cylindrical_experimental_fluence):
        plt.scatter(plane_x_coordinates/1e-6,one_ov_e2_EXP_circular_axis_z/1e-6 ,label="EXP, z, cyl. grid",c="c",linestyle="--")
        plt.plot   (plane_x_coordinates/1e-6,one_ov_e2_EXP_circular_axis_z/1e-6 ,label="_EXP, z, cyl. grid",linestyle="--",dashes=(2, 4),c="c",lw=2,alpha=alpha_lines)

    plt.xlabel("x (um)")
    plt.ylabel("1/e^2 width_z (um)")
    plt.legend()
    plt.ylim(0,)

    if fig_FWHM_y == None:
        fig_FWHM_y = plt.figure()
    else:
        plt.figure(fig_FWHM_y.number);

    # y axis
    color_GSA = "fuchsia"
    if dict_image_preprocessing["geometry"] == "cylindrical":
        color_GSA = "limegreen"
    plt.scatter    (plane_x_coordinates/1e-6,FWHM_GSA_axis_y/1e-6          ,label="GSA, y, "+grid_label+" grid" ,c=color_GSA,linestyle="--")
    plt.plot       (plane_x_coordinates/1e-6,FWHM_GSA_axis_y/1e-6          ,label="_GSA, y, "+grid_label+" grid",c=color_GSA,lw=2,alpha=alpha_lines,linestyle="--")
    if plot_cartesian_experimental_fluence:
        plt.scatter(plane_x_coordinates/1e-6,FWHM_EXP_cartesian_axis_y/1e-6,label="EXP, y, cart. grid",c="k",linestyle="--")
        plt.plot   (plane_x_coordinates/1e-6,FWHM_EXP_cartesian_axis_y/1e-6,label="_EXP, y, cart. grid",linestyle="--",dashes=(2, 2),c="k",lw=2,alpha=alpha_lines)
    if (dict_image_preprocessing["geometry"] == "cylindrical") & (plot_cylindrical_experimental_fluence):
        plt.scatter(plane_x_coordinates/1e-6,FWHM_EXP_circular_axis_y/1e-6 ,label="EXP, y, cyl. grid",c="c",linestyle="--")
        plt.plot   (plane_x_coordinates/1e-6,FWHM_EXP_circular_axis_y/1e-6 ,label="_EXP, y, cyl. grid",linestyle="--",dashes=(2, 4),c="c",lw=2,alpha=alpha_lines)

    plt.xlabel("x (um)")
    plt.ylabel("FWHM_y (um)")
    plt.legend()
    plt.ylim(0,)
    
    if fig_FWHM_z == None:
        fig_FWHM_z = plt.figure()
    else:
        plt.figure(fig_FWHM_z.number);
        
    # z axis
    plt.scatter    (plane_x_coordinates/1e-6,FWHM_GSA_axis_z/1e-6,label="GSA, z, "+grid_label+" grid",c=color_GSA)
    plt.plot       (plane_x_coordinates/1e-6,FWHM_GSA_axis_z/1e-6,label="_GSA, z, "+grid_label+" grid",c=color_GSA,lw=2,alpha=alpha_lines)
    if plot_cartesian_experimental_fluence:
        plt.scatter(plane_x_coordinates/1e-6,FWHM_EXP_cartesian_axis_z/1e-6,label="EXP, z, cart. grid",c="k")
        plt.plot   (plane_x_coordinates/1e-6,FWHM_EXP_cartesian_axis_z/1e-6,label="_EXP, z, cart. grid",linestyle="--",dashes=(2, 2),c="k",lw=2,alpha=alpha_lines)
    if (dict_image_preprocessing["geometry"] == "cylindrical") & (plot_cylindrical_experimental_fluence):
        plt.scatter(plane_x_coordinates/1e-6,FWHM_EXP_circular_axis_z/1e-6,label="EXP, z, cyl. grid",c="c")
        plt.plot   (plane_x_coordinates/1e-6,FWHM_EXP_circular_axis_z/1e-6,label="_EXP, z, cyl. grid",linestyle="--",dashes=(2, 4),c="c",lw=2,alpha=alpha_lines)

    plt.xlabel("x (um)")
    plt.ylabel("FWHM_z (um)")
    plt.legend()
    plt.ylim(0,)

    return fig_1e2_width_y, fig_FWHM_y, fig_1e2_width_z, fig_FWHM_z


def plot_lineouts(dict_image_preprocessing, dict_mesh,dict_GSA,max_radius_for_plot,orientation = "horizontal",fig_lineouts=None,axes_lineouts=None,plot_cartesian_experimental_fluence=True,plot_cylindrical_experimental_fluence=True):
    
    plane_x_coordinates         = dict_mesh["plane_x_coordinates"]
    number_of_planes            = np.size(plane_x_coordinates  )
    
    # Determine the subplot layout based on the orientation
    if fig_lineouts is None or axes_lineouts is None:
        if orientation == 'horizontal':
            fig_lineouts, axes_lineouts = plt.subplots(2, number_of_planes, figsize=(15, 6))
        elif orientation == 'vertical':
            fig_lineouts, axes_lineouts = plt.subplots(number_of_planes, 2, figsize=(6, 15))
        else:
            raise ValueError("Orientation must be either 'horizontal' or 'vertical'")
    else:
        # Ensure axes_lineouts is a numpy array for consistent indexing
        axes_lineouts = np.array(axes_lineouts)
    
    max_value = np.maximum(np.amax(dict_GSA["fluence_exp_cartesian_axis_y"]),np.amax(dict_GSA["fluence_exp_cartesian_axis_z"]))
    
    if dict_image_preprocessing["geometry"] == "cylindrical":
        color_GSA = "limegreen"
    else:
        color_GSA = "fuchsia"
    # Plot the data
    for i_plane in range(number_of_planes):
        
        if orientation == 'horizontal':
            ax_y = axes_lineouts[0, i_plane]  # First row for y
            ax_z = axes_lineouts[1, i_plane]  # Second row for z
        else:
            ax_y = axes_lineouts[i_plane, 0]  # First column for y
            ax_z = axes_lineouts[i_plane, 1]  # Second column for z
            
        if dict_image_preprocessing["geometry"] == "cylindrical":
            if plot_cartesian_experimental_fluence:
                ax_y.plot(dict_mesh["y_mesh"]/1e-6, dict_GSA["fluence_exp_cartesian_axis_y"       ][i_plane,:],c="k",lw=2)
            if plot_cylindrical_experimental_fluence:
                ax_y.plot(dict_mesh["y_mesh"]/1e-6, dict_GSA["fluence_exp_circular_axis_y"        ][i_plane,:],c="c",lw=2,linestyle="--",dashes=(4, 4))
            ax_y.plot(dict_mesh["y_mesh"]/1e-6, dict_GSA["fluence_reconstruction_circular_axis_y" ][i_plane,:],c=color_GSA,lw=2,linestyle="--",dashes=(2, 4))
        else:
            if plot_cartesian_experimental_fluence:
                ax_y.plot(dict_mesh["y_mesh"]/1e-6, dict_GSA["fluence_exp_cartesian_axis_y"       ][i_plane,:],c="k",lw=2)
            ax_y.plot(dict_mesh["y_mesh"]/1e-6, dict_GSA["fluence_reconstruction_cartesian_axis_y"][i_plane,:],c=color_GSA,lw=2,linestyle="--",dashes=(2,2))
            
        ax_y.set_xlim(-max_radius_for_plot/1e-6,max_radius_for_plot/1e-6)
        ax_y.set_ylim(-0.01,1.2*max_value)
        ax_y.set_title("plane at x = "+str(plane_x_coordinates[i_plane])+" m")
        ax_y.set_xlabel('y (um)')
        ax_y.set_ylabel('fluence (arb. units)')

        if dict_image_preprocessing["geometry"] == "cylindrical":
            
            if (i_plane==(number_of_planes-1)):
                if plot_cartesian_experimental_fluence:
                    ax_z.plot(dict_mesh["z_mesh"]/1e-6, dict_GSA["fluence_exp_cartesian_axis_z"       ][i_plane,:],c="k",lw=2,label="EXP, cart. grid")
                if plot_cylindrical_experimental_fluence:
                    ax_z.plot(dict_mesh["z_mesh"]/1e-6, dict_GSA["fluence_exp_circular_axis_z"        ][i_plane,:],c="c",lw=2,linestyle="--",dashes=(4, 4),label="EXP, cyl. grid")
                ax_z.plot(dict_mesh["z_mesh"]/1e-6, dict_GSA["fluence_reconstruction_circular_axis_z" ][i_plane,:],c=color_GSA,lw=2,linestyle="--",dashes=(2, 4),label="GSA-MD, cyl. grid")
            else:
                if plot_cartesian_experimental_fluence:
                    ax_z.plot(dict_mesh["z_mesh"]/1e-6, dict_GSA["fluence_exp_cartesian_axis_z"       ][i_plane,:],c="k",lw=2)
                if plot_cylindrical_experimental_fluence:
                    ax_z.plot(dict_mesh["z_mesh"]/1e-6, dict_GSA["fluence_exp_circular_axis_z"        ][i_plane,:],c="c",lw=2,linestyle="--",dashes=(4, 4))
                ax_z.plot(dict_mesh["z_mesh"]/1e-6, dict_GSA["fluence_reconstruction_circular_axis_z" ][i_plane,:],c=color_GSA,lw=2,linestyle="--",dashes=(2, 4))
        
        else:
            
            if (i_plane==(number_of_planes-1)):
                if plot_cartesian_experimental_fluence:
                    ax_z.plot(dict_mesh["z_mesh"]/1e-6, dict_GSA["fluence_exp_cartesian_axis_z"       ][i_plane,:],c="k",lw=2,label="EXP, cart. grid")
                ax_z.plot(dict_mesh["z_mesh"]/1e-6, dict_GSA["fluence_reconstruction_cartesian_axis_z"][i_plane,:],c=color_GSA,lw=2,linestyle="--",dashes=(2,2),label="GSA-MD, cart. grid")
            else:
                if plot_cartesian_experimental_fluence:
                    ax_z.plot(dict_mesh["z_mesh"]/1e-6, dict_GSA["fluence_exp_cartesian_axis_z"       ][i_plane,:],c="k",lw=2)
                ax_z.plot(dict_mesh["z_mesh"]/1e-6, dict_GSA["fluence_reconstruction_cartesian_axis_z"][i_plane,:],c=color_GSA,lw=2,linestyle="--",dashes=(2,2))
                
        ax_z.set_xlim(-max_radius_for_plot/1e-6,max_radius_for_plot/1e-6)
        ax_z.set_ylim(-0.01,1.2*max_value)
        ax_z.set_title("plane at x = "+str(plane_x_coordinates[i_plane])+" m")
        ax_z.set_xlabel('z (um)')
        ax_z.set_ylabel('fluence (arb. units)')
        
        if (i_plane==(number_of_planes-1)):
            ax_z.legend()
    
    # Adjust layout for better readability
    plt.tight_layout()
    plt.show()
    
    return fig_lineouts, axes_lineouts

##### This function was not updated since a while    
def subplots_reconstructed_fluence(dict_image_preprocessing,dict_mesh,dict_GSA,max_radius_for_plot,cmap=None,relative_vmin=0,relative_vmax=0.7,orientation="horizontal",polar_plot_shading="nearest"):
    
    plane_x_coordinates         = dict_mesh["plane_x_coordinates"]
    number_of_planes            = np.size(plane_x_coordinates  )
    
    if cmap==None:
        cmap=my_cmap
    
    if dict_image_preprocessing["geometry"]=="cylindrical":
        
        # Determine the subplot layout based on the orientation
        if orientation == 'horizontal':
            fig, axes = plt.subplots(2, number_of_planes, figsize=(14, 5.),subplot_kw={'projection': 'polar'})
        elif orientation == 'vertical':
            fig, axes = plt.subplots(number_of_planes, 2, figsize=(7, 14),subplot_kw={'projection': 'polar'})
        else:
            raise ValueError("Orientation must be either 'horizontal' or 'vertical'")
            
        
        r_mesh                       = dict_mesh               ["r_mesh"                         ]
        y_mesh                       = dict_mesh               ["y_mesh"                         ]
        z_mesh                       = dict_mesh               ["z_mesh"                         ]
        theta_mesh                   = dict_mesh               ["theta_mesh"                     ]
        fluence_exp_images_circular  = dict_image_preprocessing["fluence_exp_images_circular"    ]
        fluence_exp_images_cartesian = dict_image_preprocessing["fluence_exp_images_cartesian"   ]
        fluence_gsa_images_circular  = dict_GSA                ["fluence_reconstruction_circular"]
        
        r_meshgrid, theta_meshgrid = np.meshgrid(r_mesh/1e-6,theta_mesh,indexing='ij')
                                                 
        vmin   = relative_vmin*np.amax(fluence_exp_images_cartesian)
        vmax   = relative_vmax*np.amax(fluence_exp_images_cartesian)
        
        
        for i_plane in range(number_of_planes):
            
            if orientation == 'horizontal':
                ax_exp = axes[0, i_plane]  # First row for experiment
                ax_GSA = axes[1, i_plane]  # Second row for reconstruction
            else:
                ax_exp = axes[i_plane, 0]  # First column for experiment
                ax_GSA = axes[i_plane, 1]  # Second column for reconstruction
            
            
            im_exp=ax_exp.pcolormesh(theta_meshgrid, r_meshgrid, fluence_exp_images_circular[i_plane,:,:], shading='nearest', cmap=cmap,vmin=vmin,vmax=vmax)
            plt.colorbar(im_exp)
            im_GSA=ax_GSA.pcolormesh(theta_meshgrid, r_meshgrid, fluence_gsa_images_circular[i_plane,:,:], shading='nearest', cmap=cmap,vmin=vmin,vmax=vmax)
            plt.colorbar(im_GSA)
            
            ax_exp.grid("False")
            ax_exp.set_title("plane at x = "+str(plane_x_coordinates[i_plane])+" m")
            
            ax_exp.set_ylabel('r (um)')
            ax_GSA.set_ylabel('r (um)')
            
            ax_exp.grid(alpha=0.1)   
            ax_GSA.grid(alpha=0.1)   
            
            # ax_exp.grid("False")
            # ax_GSA.grid("False")
            # 
            # # Remove angular ticks and labels
            # ax_exp.set_xticks([])
            # ax_exp.set_yticks([])
            # 
            # ax_GSA.set_xticks([])
            # ax_GSA.set_yticks([])
            # 
            # # Remove radial grid and angular grid
            # ax_exp.grid(False)
            # ax_GSA.grid(False)
            # 
            # # Remove the radial labels
            # ax_exp.set_yticklabels([])
            # ax_GSA.set_yticklabels([])
            
        plt.tight_layout()
        
        # make also a cartesian plot to check the conversion to cylindrical coordinates
        # Determine the subplot layout based on the orientation
        if orientation == 'horizontal':
            fig_cart, axes = plt.subplots(1, number_of_planes, figsize=(15.3, 2.5))
        elif orientation == 'vertical':
            fig_cart, axes = plt.subplots(number_of_planes, 1, figsize=(4.5, 14))
        else:
            raise ValueError("Orientation must be either 'horizontal' or 'vertical'")
                                                 
        vmin   = relative_vmin*np.amax(fluence_exp_images_cartesian)
        vmax   = relative_vmax*np.amax(fluence_exp_images_cartesian)
        
        extent = [y_mesh.min()/1e-6,y_mesh.max()/1e-6,z_mesh.min()/1e-6,z_mesh.max()/1e-6]
        
        for i_plane in range(number_of_planes):
            
            ax_exp = axes[i_plane]  # First row for experiment

            im_exp=ax_exp.imshow(fluence_exp_images_cartesian[i_plane,:,:].T,extent=extent,cmap=cmap,aspect="auto",vmin=vmin,vmax=vmax,origin="lower")
            plt.colorbar(im_exp)
            
            ax_exp.set_title("plane at x = "+str(plane_x_coordinates[i_plane])+" m")
            
            ax_exp.set_xlabel('y (um)')
            ax_exp.set_ylabel('z (um)')
            
            ax_exp.set_xlim(-max_radius_for_plot/1e-6,max_radius_for_plot/1e-6)
            ax_exp.set_ylim(-max_radius_for_plot/1e-6,max_radius_for_plot/1e-6)
            
        plt.tight_layout()
            
    else:
        
        # Determine the subplot layout based on the orientation
        if orientation == 'horizontal':
            fig, axes = plt.subplots(2, number_of_planes, figsize=(14, 4.3))
        elif orientation == 'vertical':
            fig, axes = plt.subplots(number_of_planes, 2, figsize=(7, 14))
        else:
            raise ValueError("Orientation must be either 'horizontal' or 'vertical'")
            
        y_mesh                       = dict_mesh               ["y_mesh"                          ]
        z_mesh                       = dict_mesh               ["z_mesh"                          ]
        fluence_exp_images_cartesian = dict_image_preprocessing["fluence_exp_images_cartesian"    ]
        fluence_gsa_images_cartesian = dict_GSA                ["fluence_reconstruction_cartesian"]
                                                 
        vmin   = relative_vmin*np.amax(fluence_exp_images_cartesian)
        vmax   = relative_vmax*np.amax(fluence_exp_images_cartesian)
        
        extent = [y_mesh.min()/1e-6,y_mesh.max()/1e-6,z_mesh.min()/1e-6,z_mesh.max()/1e-6]
        
        for i_plane in range(number_of_planes):
            
            if orientation == 'horizontal':
                ax_exp = axes[0, i_plane]  # First row for experiment
                ax_GSA = axes[1, i_plane]  # Second row for reconstruction
            else:
                ax_exp = axes[i_plane, 0]  # First column for experiment
                ax_GSA = axes[i_plane, 1]  # Second column for reconstruction
            
            im_exp=ax_exp.imshow(fluence_exp_images_cartesian[i_plane,:,:].T,extent=extent,cmap=cmap,aspect="auto",vmin=vmin,vmax=vmax,origin="lower")
            plt.colorbar(im_exp)
            im_GSA=ax_GSA.imshow(fluence_gsa_images_cartesian[i_plane,:,:].T,extent=extent,cmap=cmap,aspect="auto",vmin=vmin,vmax=vmax,origin="lower")
            plt.colorbar(im_GSA)
            
            ax_exp.set_title("plane at x = "+str(plane_x_coordinates[i_plane])+" m")
            
            #ax_exp.set_xlabel('y (um)')
            ax_exp.set_ylabel('z (um)')
            
            ax_exp.set_xlim(-max_radius_for_plot/1e-6,max_radius_for_plot/1e-6)
            ax_exp.set_ylim(-max_radius_for_plot/1e-6,max_radius_for_plot/1e-6)
            
            ax_GSA.set_xlabel('y (um)')
            ax_GSA.set_ylabel('z (um)')
            
            ax_GSA.set_xlim(-max_radius_for_plot/1e-6,max_radius_for_plot/1e-6)
            ax_GSA.set_ylim(-max_radius_for_plot/1e-6,max_radius_for_plot/1e-6)
            
    plt.tight_layout()
    plt.show()
    
    return fig, axes
    
    
def plots_reconstructed_fluence(dict_image_preprocessing,dict_mesh,dict_GSA,max_radius_for_plot,cmap=None,relative_vmin=0,relative_vmax=0.7,polar_plot_shading="nearest"):
    
    plane_x_coordinates         = dict_mesh["plane_x_coordinates"]
    number_of_planes            = np.size(plane_x_coordinates  )
    
    if cmap==None:
        cmap=my_cmap
    
    if dict_image_preprocessing["geometry"]=="cylindrical":    
        
        r_mesh                       = dict_mesh               ["r_mesh"                         ]
        y_mesh                       = dict_mesh               ["y_mesh"                         ]
        z_mesh                       = dict_mesh               ["z_mesh"                         ]
        theta_mesh                   = dict_mesh               ["theta_mesh"                     ]
        fluence_exp_images_circular  = dict_image_preprocessing["fluence_exp_images_circular"    ]
        fluence_exp_images_cartesian = dict_image_preprocessing["fluence_exp_images_cartesian"   ]
        fluence_gsa_images_circular  = dict_GSA                ["fluence_reconstruction_circular"]
        
        # Set the vmin and vmax
        # Note that for a fair comparison the vmax is chosen from the images 
        # in Cartesian geometry before their conversion to cylindrical coordinates.
        vmin                         = relative_vmin*np.amax(fluence_exp_images_cartesian)
        vmax                         = relative_vmax*np.amax(fluence_exp_images_cartesian)
        
        # Define a meshgrid along r, theta
        r_meshgrid, theta_meshgrid   = np.meshgrid(r_mesh/1e-6,theta_mesh,indexing='ij')
        
        # Create a theta mesh where the theta=0 point is doubled, to have theta=2*pi.
        # When this is not done, an artefact appears in a polar plot, 
        # especially when few sampling points along theta are used.
        theta_mesh                   = np.append(theta_mesh, 2*np.pi)
        r_meshgrid, theta_meshgrid   = np.meshgrid(r_mesh/1e-6, theta_mesh, indexing='ij')
        
        # Coherently, the fluence at theta=0 is equal to the one at theta=2*pi
        fluence_exp_images_circular  = np.concatenate( \
                                              (fluence_exp_images_circular, \
                                               fluence_exp_images_circular[:, :, 0:1]), \
                                               axis=2)
        fluence_gsa_images_circular  = np.concatenate( \
                                              (fluence_gsa_images_circular, \
                                               fluence_gsa_images_circular[:, :, 0:1]), \
                                               axis=2)
        
        
        for i_plane in range(number_of_planes):
            
            # experiment, cartesian grid
            plt.figure()
            extent = [y_mesh.min()/1e-6,y_mesh.max()/1e-6,z_mesh.min()/1e-6,z_mesh.max()/1e-6]
            im_exp=plt.imshow(fluence_exp_images_cartesian[i_plane,:,:].T,extent=extent,cmap=cmap,aspect="equal",vmin=vmin,vmax=vmax,origin="lower")
            plt.colorbar(im_exp)
            plt.title("EXP, cart. grid., plane at x = "+str(plane_x_coordinates[i_plane])+" m")
            plt.xlabel('y (um)')
            plt.ylabel('z (um)')
            plt.xlim(-max_radius_for_plot/1e-6,max_radius_for_plot/1e-6)
            plt.ylim(-max_radius_for_plot/1e-6,max_radius_for_plot/1e-6)
            plt.tight_layout()
            
            # experiment, cylindrical grid
            fig, ax_exp = plt.subplots(subplot_kw={'projection': 'polar'})
            ax_exp.grid(False) 
            im_exp=ax_exp.pcolormesh(theta_meshgrid, r_meshgrid, fluence_exp_images_circular[i_plane,:,:], shading=polar_plot_shading, cmap=cmap,vmin=vmin,vmax=vmax)
            plt.colorbar(im_exp)
            ax_exp.set_ylabel('r (um)')
            ax_exp.set_rlabel_position(-22.5)
            ax_exp.set_title("EXP, cyl. grid, plane at x = "+str(plane_x_coordinates[i_plane])+" m")
            ax_exp.set_rmax(max_radius_for_plot/1e-6)
            ax_exp.grid(alpha=0.1)   
            plt.tight_layout()
            
            # reconstruction, cylindrical grid
            fig, ax_GSA = plt.subplots(subplot_kw={'projection': 'polar'})
            ax_GSA.grid(False) 
            im_GSA=ax_GSA.pcolormesh(theta_meshgrid, r_meshgrid, fluence_gsa_images_circular[i_plane,:,:], shading=polar_plot_shading, cmap=cmap,vmin=vmin,vmax=vmax)
            plt.colorbar(im_GSA)
            ax_GSA.set_ylabel('r (um)')
            ax_GSA.set_rlabel_position(-22.5)
            ax_GSA.set_title("GSA, cyl. grid, plane at x = "+str(plane_x_coordinates[i_plane])+" m")
            ax_GSA.set_rmax(max_radius_for_plot/1e-6)
            ax_GSA.grid(alpha=0.1)   
            plt.tight_layout()
            
            # Remove angular ticks and labels
            ax_exp.set_xticks([])
            ax_exp.set_yticks([])
            ax_GSA.set_xticks([])
            ax_GSA.set_yticks([])
            # Remove radial grid and angular grid
            ax_exp.grid(False)
            ax_GSA.grid(False)
            # Remove the radial labels
            ax_exp.set_yticklabels([])
            ax_GSA.set_yticklabels([])
    else:
            
        y_mesh                       = dict_mesh               ["y_mesh"                          ]
        z_mesh                       = dict_mesh               ["z_mesh"                          ]
        fluence_exp_images_cartesian = dict_image_preprocessing["fluence_exp_images_cartesian"    ]
        fluence_gsa_images_cartesian = dict_GSA                ["fluence_reconstruction_cartesian"]
                                                 
        vmin   = relative_vmin*np.amax(fluence_exp_images_cartesian)
        vmax   = relative_vmax*np.amax(fluence_exp_images_cartesian)
        
        extent = [y_mesh.min()/1e-6,y_mesh.max()/1e-6,z_mesh.min()/1e-6,z_mesh.max()/1e-6]
        
        for i_plane in range(number_of_planes):

            plt.figure()
            im_exp=plt.imshow(fluence_exp_images_cartesian[i_plane,:,:].T,extent=extent,cmap=cmap,aspect="equal",vmin=vmin,vmax=vmax,origin="lower")
            plt.colorbar(im_exp)
            plt.title("EXP, cart. grid., plane at x = "+str(plane_x_coordinates[i_plane])+" m")
            plt.xlabel('y (um)')
            plt.ylabel('z (um)')
            plt.xlim(-max_radius_for_plot/1e-6,max_radius_for_plot/1e-6)
            plt.ylim(-max_radius_for_plot/1e-6,max_radius_for_plot/1e-6)
            plt.tight_layout()
            
            plt.figure()
            im_GSA=plt.imshow(fluence_gsa_images_cartesian[i_plane,:,:].T,extent=extent,cmap=cmap,aspect="equal",vmin=vmin,vmax=vmax,origin="lower")
            plt.colorbar(im_GSA)
            plt.title("GSA, cart. grid., plane at x = "+str(plane_x_coordinates[i_plane])+" m")
            plt.xlabel('y (um)')
            plt.ylabel('z (um)')
            plt.xlim(-max_radius_for_plot/1e-6,max_radius_for_plot/1e-6)
            plt.ylim(-max_radius_for_plot/1e-6,max_radius_for_plot/1e-6)
            plt.tight_layout()

    plt.show()
                
        
        
        
        
        
        




