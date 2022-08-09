# EUREC4A Radiative profiles

##Steps to run the script

### 0. Add output/ in .gitignore

### 1. Compile the radiation code

In rte-rrtmgp/build create a Makefile.conf following the template in the folder.
Alternatively, you can set environment variables FC and FCFLAGS to be the name of the
Fortran compiler and the compilation flags.
Invoke make.

### 2. Compile main script

In Makefile edit the flags NCHOME and NFHOME for your platform. These point to the
root of the netCDF C and Fortran installations on your platform. 
Call make.

### 3. Run the script

In script, edit compute_radiation_from_soundings.sh with the path to your dropsonde file.
Run compute_radiation_from_soundings.sh

### 4. Use T,p-dependent gas optics or fixed gas optics

In ./code/, choose the main script "mv sonde_radiation_default.F90 sonde_radiation.F90" for standard case with vertically-varying extinction coefficients, and "mv sonde_radiation_uniform_gas_optics.F90 sonde_radiation.F90" for vertically uniform extinction coefficients.

### 5. Compute radiative transfer with moist intrusions of various sorts

In the correct conda environment, execute:
- compute_radiation_from_idealized_moist_intrusions.sh for intrusions of various shapes heights and water paths based on the lower intrusion occurring on 2020-02-13
- compute_radiation_from_idealized_moist_intrusions_fix_kappa.sh, same calculation, but with fixed extinction to its lowest tropospheric values
- compute_radiation_from_idealized_warming.sh for the surface warming calculation


