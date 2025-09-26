# A more detailed introduction

Fluence images measured at different transverse planes along a laser pulse propagation axis
give information on the spatial distribution of the laser complex field amplitude,
but not on the spatial distribution of its phase. To retrieve this complex field 
(including amplitude and phase) when only fluence images are available, usually 
phase retrieval algorithms are required.
One of the most common is the Gerchberg-Saxton Algorithm (GSA) [GSA1972].

The Gerchberg-Saxton Algorithm with Mode decomposition (GSA-MD) [Moulanier2023a]
introduces a paraxial mode decomposition of the unknown complex field. 
Instead of reconstructing the complex field 
directly on a spatial grid, the GSA-MD yields a set of mode coefficients to
represent the field with a chosen basis, e.g. Hermite-Gauss (HG) or Laguerre Gauss (LG) modes.
This allows the user to easily reconstruct the laser field at an arbitrary location 
using the mode coefficients and the mode analytical expressions.

The implementation in this repository supports both Hermite-Gauss (HG) 
and Laguerre-Gauss (LG) decomposition.
The latter, performed in cylindrical coordinates, is particularly suited for
lasers with a high degree of cylindrical symmetry.
For the sake of comparisons, a version of the GSA without mode decomposition, which uses
the Fresnel propagator method, is provided as well.

# Assumptions for the field reconstruction

- The laser propagates along the `x` axis.
- The laser field is assumed to have negligible space-temporal couplings. This way, the laser field distribution is assumed to be approximately equal to a transverse distribution 
multiplied by the pulse temporal profile, modulated by the laser carrier frequency.
- A paraxial propagation is assumed.

# Units

- All lengths variables are in meters ("length unit").
- All fluence images are in so-called "fluence units", which are the square of "field units".
- All electric fields are in so-called "field units".

The conversion to physical units (e.g. from arbitrary field units to V/m) requires multiplying by appropriate constants, based on pulse energy and temporal profile.

# An overview of the laser field reconstruction

### Field reconstruction: how it starts
Running the provided `main.py` executes `generate_input_files.py` to prepare and save the inputs. 
This script generates three dictionaries, with different purviews:
  - `dict_image_preprocessing`: input images and their preprocessing;
  - `dict_mesh`: mesh of the preprocessed images, where the mode basis will be defined;
  - `dict_GSA`: the GSA reconstruction (with or without mode decomposition).
  
For example, the input fluence images provided by the user are stored in the `dict_image_preprocessing["input_fluence_images"]`.

During the execution of the program these dictionaries will be updated and other 
keys will be added (e.g. preprocessed images), also with default entries in case 
some inputs are missing.

To keep in memory the inputs provided by the user (before the input sanity checks and corrections), 
files called e.g. `inputs_image_preprocessing.npy` will be created in the folder `inputs_by_the_user`. 
The final version of these dictionaries will be saved instead in a file e.g. `dict_image_preprocessing.npy`, 
in the `outputs` folder.

If an `inputs_by_the_user` or an `outputs` folder already exist, the program will stop. 
This helps avoiding to overwrite previous data.
To reproduce a previous run, just read the dictionaries from the folders created 
by the previous run and use them as inputs for the field reconstruction 
or plot functions in the repository.

### Image preprocessing 
Implemented in `image_preprocessing/preprocessing_functions.py`.

After the generation of the input dictionaries, the provided `main.py` file calls the
function `process_input_fluence_images`.

The images are read as `y`,`z` arrays, therefore `dict_image_preprocessing["input_fluence_images"]`
must have a shape ``(nx,ny,nz)``, where `nx` is the number of transverse planes 
where the fluence images are provided. 
Each image must have the same shape `(ny,nz)`.

The user must provide a `dict_image_preprocessing["length_per_pixel"]` in meters, 
i.e. the minimum resolution of of the images. This value must be the same 
for each fluence image.

In the preprocessing, the background noise is reduced with a 2D median filter 
with size `dict_image_preprocessing["median_filter_size"]`, then values 
below `dict_image_preprocessing["threshold"]` are set to zero, 
then the minimum of the images is set to zero.

The position of the planes on the laser propagation axis is provided by the user in `dict_mesh[""plane_x_coordinates"].`
Depending on the `dict_image_preprocessing["geometry"]`, the fluence images will be 
transferred to a cylindrical (if `dict_image_preprocessing["geometry"]="cylindrical"`)
or to cartesian grid (if `dict_image_preprocessing["geometry"]="cartesian"`).

At each plane, a grid is defined for the fluence images. In these grids the images 
will be centered, i.e. their center will be at `y=0`,`z=0` (`r=0`) on each plane. 
The user can provide a center for each image, in the form of arrays with shape `nx` called `dict_image_preprocessing["y0_at_planes"]`
and `dict_image_preprocessing["z0_at_planes"]`. If the user does not provide them,
they will be set by default as the coordinates of the maximum of each image.
This feature allows the user also to perform an optimization on the image center coordinates
to minimize the reconstruction error, as described in [Moulanier2023a]. 

In cylindrical coordinates, the cylindrical grids will have a shape `dict_mesh["nr_converted_image"],dict_mesh["ntheta"]`. 
In cartesian coordinates, the grids will have a shape `dict_mesh["ny_converted_image"],dict_mesh["nz_converted_image"]`. 
The mode basis is defined and the field reconstruction is performed on these transverse grids, one for each transverse plane.

The total sum of fluence values of each image
will be normalized and set equal to the total fluence in the plane whose index is 
`dict_image_preprocessing["index_reference_plane"]`. This ensures 
that the total fluence is conserved during the propagation. This value will
be stored in `dict_image_preprocessing["energy_reference_plane"]`
It is recommended to choose the index of the plane closest to the focal plane. 
Note that this integration is performed in cylindrical coordinates when `dict_image_preprocessing["geometry"]="cylindrical"`. 
In both geometries, the reference total fluence is computed from the Cartesian 
version of the image at the chosen plane.

The resulting preprocessed images are stored then 
in `dict_image_preprocessing["fluence_exp_images_cartesian"]` 
(and `dict_image_preprocessing["fluence_exp_images_circular"]` 
if `dict_image_preprocessing["geometry"]="cylindrical"`).

The user can specify a radius around the image center, called `dict_preprocessing["r_max_coordinate_change"]`, 
beyond which the fluence is set to zero after the transfer to the grid 
for the reconstruction.

These fluence arrays have three indices:
- plane index
- r     index (cylindrical) or y index (cartesian)
- theta index (cylindrical) or z index (cartesian)

Note that in cylindrical geometry the field reconstruction is performed only 
on the cylindrical grid. In this geometry, the preprocessed image is transferred 
also to a centered cartesian grid in `dict_image_preprocessing["fluence_exp_images_cartesian"]` 
only to be able to check that the coordinate transformation 
(made through interpolation) is accurate. Normally this interpolation should 
work well with a sufficient number of sampling points.

In the following, the grid (either cartesian or cylindrical) where the images 
are transferred by the preprocessing operations will be referred to as the field 
reconstruction grid, since all the operations of the field reconstruction 
will be performed on that grid.

### GSA reconstruction 
Implemented in `field_reconstruction/GSA_reconstruction.py`.

Two algorithms are available:
- GSA-MD,
- GSA.

After pre-processing, the `main.py` file calls the function `field_reconstruction_GSA`
to perform the field reconstruction with the GSA with or without mode decomposition
(this can be set with the boolean `dict_GSA["use_Mode_Decomposition"]`).
It is assumed that the carrier wavelength of the field to reconstruct is equal 
to `dict_GSA["lambda_0"]`.

Note that the GSA without mode decomposition is only available in cartesian geometry.

Following is a description of the two GSA algorithms included in the repository, 
with and without mode decomposition.

### Mode basis operations 
Implemented in `mode_basis/mode_basis_operations.py`.

For the GSA with mode decomposition (GSA-MD) loop, the fields of the Laguerre-Gauss 
(LG, file `mode_basis/laguerre_gauss_modes.py`) and Hermite-Gauss (HG, file `mode_basis/hermite_gauss_modes.py`) modes are defined on the field reconstruction grid. 
For the LG mode basis, a `dict_mode_basis["LG_mode_type"]` can be selected 
(`helical` or `sinusoidal`, default: `helical`).

The field arrays of a full mode would have five indices:
- `p`       index (cylindrical) or m index (cartesian)
- `l`       index (cylindrical) or n index (cartesian)
- `i_plane` index
- `r`       index (cylindrical) or y index (cartesian)
- `theta`   index (cylindrical) or z index (cartesian)

The mode basis parameters (wavelength, waist, focal plane position) are set by 
the user in a subdictionary of the `dict_GSA` dictionary, called `dict_GSA["dict_mode_basis"]`.
The mentioned parameters would be stored inside this subdictionary, namely in
`dict_mode_basis["lambda_0"]`, `dict_mode_basis["waist_0"]`, `dict_mode_basis["x_focus"]`. 

The mode maximum indices are set by the user in 
`dict_mode_basis["Max_LG_index_p"]` (radial index) and `dict_mode_basis["Max_LG_index_l"]` 
(azimuthal index) for the LG modes and 
`dict_mode_basis["Max_HG_index_m"]` (`y` index) and `dict_mode_basis["Max_HG_index_n"]` 
(`z` index) for the HG modes. 

The helical LG modes azimuthal indices span from `-dict_mode_basis["Max_LG_index_l"]` 
to `dict_mode_basis["Max_LG_index_l"]`
included, but they are stored with the same convention of a FFT (i.e. from `0` to `dict_mode_basis["Max_LG_index_l"]`, 
and then for increasingly negative index). 
The sinusoidal LG modes use the `cos(l * theta)` for positive `l` indices 
and the `sin(l * theta)` for negative  `l` indices.

Consequently, the total number of LG modes used for the reconstruction in 
cylindrical geometry is 
`(dict_mode_basis["Max_LG_index_p"]+1) * (2 * dict_mode_basis["Max_LG_index_l"] +1)`

The HG mode indices start from 0 and end at `dict_mode_basis["Max_HG_index_m"]` (`dict_mode_basis["Max_HG_index_n"]`) included.

Consequently, the total number of HG modes used for the reconstruction in cartesian geometry is `(dict_mode_basis["Max_HG_index_m"]+ 1) * (dict_mode_basis["Max_HG_index_n"] + 1)`

Operations for the projection of a field on the LG/HG mode basis and the reconstruction 
of a field with the LG/HG mode basis are defined in 
`mode_basis_operations.py`. These operations are performed at one plane `i_plane` 
and involve integrals (using a basic rectangle rule) or sums over the field 
reconstruction grid where the modes are defined.

Note that the results of the field reconstruction may change with the choice 
of waist and focal plane position. For a first run, it is recommended to choose 
a focal plane position equal to the one of the plane with maximum intensity and 
a waist large enough to avoid quick diffraction but small enough to allow contain
the total energy across all the planes. Note that the best waist is not necessarily 
the waist of the gaussian fit (or any fit) of the laser fluence distribution.

Note that the operations of field reconstruction involve only the `dict_mesh` and 
the `dict_mode_basis` subdictionaty. Therefore, to reconstruct the fields on another 
code, e.g. as input of a PIC simulation, you will need to use the MD coefficients 
computed by the GSA-MD and define your own mesh in `dict_mesh`.

### GSA-MD 
Implemented in file `field_reconstruction/GSA_MD_loop.py`.

As the original GSA [GSA1972], the GSA-MD is an iterative algorithm that passes 
through the planes where an image is available. Both try to reconstruct the phase 
(or equivalently the complex field) that gives fluence images as close as possible 
to those given in input during the propagation. The GSA-MD assumes that this 
complex field can be decomposed as a sum of modes in the chosen mode basis (LG/HG). 

After `dict_GSA["N_iterations"]`, the result provided by the GSA-MD is the 
estimate of the coefficients (`dict_GSA["Coeffs_LG_pl"]`/`dict_GSA["Coeffs_HG_mn"]`) 
of the decomposition at the last iteration. 
These coefficients allow to easily reconstruct the field at any position different 
from those of the input planes using the analytical expressions of the LG/HG modes.
However, since the field reconstruction is a fit, this reconstruction out of the 
input planes is an extrapolation and its physical accuracy is not guaranteed,
especially in presence of overfitting (e.g. using many modes or in presence of 
shot-to-shot fluctuations) or bias errors (e.g. using too few modes or input planes).

Every `dict_GSA["iterations_between_outputs"]` iterations, the current 
coefficients are dumped as output, as well as the error of the fit at each plane 
(the L2 norm of the difference between the reconstructed and experimental fluence 
normalized by the L2 norm of the experimental fluence) between the reconstructed 
and experimental image.

The error is computed for the grid points where the distance from 
the image center is smaller than `dict_preprocessing["r_max_coordinate_change"]`.

After the last iteration, the reconstructed field and its lineouts on the `y` 
and `z` axis are dumped as well in the dictionary `dict_GSA`.

To improve the convergence of the reconstruction, an initial phase can be used
for the first plane, corresponding to the one of a Gaussian beam with the 
same `dict_mode_basis["waist_0"]` and `dict_mode_basis["x_focus"]` of the mode basis. 
This option is activated with the boolean `dict_GSA["use_initial_Gaussian_phase"]` 
(default value: `False`). 


### GSA 
Implemented in `field_reconstruction/GSA_loop.py`.

The version of the GSA without mode decomposition included in the repository 
uses a Fresnel propagator to compute the estimate of the field at each plane
in the GSA loop. 

No `dict_mode_basis` is created for the execution of this algorithm.

The optional initial Gaussian beam phase, activated by `dict_GSA["use_initial_Gaussian_phase"]` 
(default value: `False`), has a waist equal to `dict_GSA["waist_gaussian_beam_phase"]`
and a focal plane equal to the position of the `dict_image_preprocessing["index_reference_plane"]`.

The plane at index `dict_image_preprocessing["index_reference_plane"]` will be 
the starting point for all the propagation functions.

The output of the algorithm 
will include a `dict_GSA["phase_ref_plane"]`, i.e. the phase at that plane.

The error on the fluence at that plane by definition will be zero,
because the reconstructed field at that plane will be just the experimental field 
amplitude multiplied by a phase equal to `dict_GSA["phase_ref_plane"]`.

Note that the GSA without mode decomposition is implemented only in cartesian geometry.


### Postprocessing 
Implemented in `plot_utilities/plot_functions.py`.

The example file `main.py` contains a complete workflow with the field  reconstruction:
image preprocessing, field reconstruction and plots of the outputs.

The last ones are produced through some functions defined in `plot_utilities/plot_functions.py`:
- `plot_error_evolution`: evolution of the reconstruction error vs iteration and vs time;
- `plot_diffraction`: evolution of the FWHM and 1.e^2 width of the laser at the image planes, experimental and reconstructed.
- `plot_lineouts`: lineouts on the `y` and `z` axis of the experimental and reconstructed fluences.
- `plots_reconstructed_fluence`: plots of the experimental and reconstructed fluences at each input plane.

For the cylindrical geometry, the experimental data is provided both from the 
cartesian grid before the coordinate change and from the cylindrical grid.
This allows to check that the coordinate transformation towards the cylindrical 
grid is consistent with the preprocessed data.


### References

.. [GSA1972] 
    R. W. Gerchberg, and W. O. Saxton, A practical algorithm for the determination of the phase from image and diffraction plane pictures, Optik 35, 237 (1972).

.. [Moulanier2023a] 
    I. Moulanier et al, Fast laser field reconstruction method based on a Gerchberg–Saxton algorithm with mode decomposition, Journal of the Optical Society of America B, 40, 9, 2450 (2022), https://doi.org/10.1364/JOSAB.489884`

.. [Moulanier2023b] 
    I. Moulanier et al., Modeling of the driver transverse profile for laser wakefield electron acceleration at APOLLON research facility, Physics of Plasmas 30, 053109 (2023) https://doi.org/10.1063/5.0142894



