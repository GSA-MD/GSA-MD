# GSA-MD: a laser field reconstruction algorithm

### About
GSA-MD (Gerchberg-Saxton Algorithm with Mode Decomposition) is an open source 
code to reconstruct a laser pulse electromagnetic field (amplitude and phase) 
from fluence images measured at different transverse planes 
along the laser propagation axis. This operation, particularly relevant for 
high accuracy modeling of high intensity laser-plasma interaction, 
is necessary when no measurements are available for the laser wavefront.

The code does it by finding the coefficients of the decomposition of the field 
in paraxial modes (Hermite Gauss or Laguerre Gauss), allowing to easily propagate
the reconstructed laser field at an arbitrary point, e.g. for numerical simulations. 
Its description was first published in [Moulanier2023a]. 
In [Moulanier2023b] its use in high-accuracy Particle in Cell simulations is described.

An implementation of the Gerchberg-Saxton Algorithm without mode decomposition [GSA1972] 
is provided as well.

### Contributors
Ioaquin Moulanier implemented the GSA-MD in Python for his PhD thesis at the Laboratoire de Physique des Gaz et des Plasmas (LPGP), building on a first implementation in Fortran by Gilles Maynard. In that first Python implementation, described in [Moulanier2023a,Moulanier2023b], the algorithm speed was significantly improved through a `numba` parallelization, an improvement of the laser reconstruction technique and an efficient data storage strategy for the Hermite Gauss modes.

Francesco Massimo at LPGP worked on the implementation shared in this GitHub repository and 
on its formulation in cylindrical geometry with the Laguerre-Gauss modes.

The code benefitted also from contributions to the diagnostic and postprocessing scripts by Adrian Guerente, Mohamad Masckala, Oleksandra Khomyshyn and Steyn Theunis Lodewyk.

### Installation 
The GSA-MD implementation in this repository is written in Python.
The Python libraries required to install the code are listed in the file `requirements.txt`.

To install the code, first install Python 3. 
Then, `git clone` the code repository from GitHub.
Afterwards, open a Linux-like command line window, navigate to the downloaded code directory `gsa_md`
and run the installation command 

`python3 -m pip install .`

If you prefer to use a virtual environment, activate it before the installation through `pip`.

If the installation was successful, you should be able to run 

`import gsa_md`

and use all the functions included in the library, e.g. for image preprocessing and plotting.

### How to run the code
In a directory, copy and paste, or create a symbolic link to the `main.py` file, which contains an example of complete workflow of laser reconstruction, including input image preprocessing, field reconstruction and result postprocessing.

The `main.py` file first executes a script, `examples/reconstruct_laser_HZDR_dataset/generate_input_files.py`, which creates and saves dictionaries that are then used by the field reconstruction and postprocessing functions of the `gsa_md` library. 

This example workflow reconstructs the field of a laser pulse from 5 fluence images contained in `examples/reconstruct_laser_HZDR_dataset/HZDR_dataset`. 

You can adapt these scripts (or merge them into one file) to use the `gsa_md` library functions to reconstruct the field of a laser pulse from your own dataset. 

The file `examples/reconstruct_laser_HZDR_dataset/generate_input_files.py` contains useful comments 
to guide the user in the definition of the reconstruction inputs.


### How to cite the code

When publishing simulation results involving the GSA-MD, please cite the following article and repositories:

I. Moulanier et al, Fast laser field reconstruction method based on a Gerchberg–Saxton algorithm 
with mode decomposition, Journal of the Optical Society of America B, 40, 9, 2450 (2022), 
https://doi.org/10.1364/JOSAB.489884`

...

...

If help or changes in the code were obtained from GSA-MD developers, please acknowledge 
their participation in any subsequent publication or or presentation.

### References

[GSA1972] R. W. Gerchberg, and W. O. Saxton, A practical algorithm for the determination of the phase from image and diffraction plane pictures, Optik 35, 237 (1972).

[Moulanier2023a] I. Moulanier et al, Fast laser field reconstruction method based on a Gerchberg–Saxton algorithm with mode decomposition, Journal of the Optical Society of America B, 40, 9, 2450 (2022), https://doi.org/10.1364/JOSAB.489884`

[Moulanier2023b] I. Moulanier et al., Modeling of the driver transverse profile for laser wakefield electron acceleration at APOLLON research facility, Physics of Plasmas 30, 053109 (2023) https://doi.org/10.1063/5.0142894




