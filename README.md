
# Fix Rigid/Abrade README

## Overview
The following README was created to support the publication of "Abradable DEM: A Novel Framework to Capture the Mechanistic Evolution of Particle Shape". 
## Repository Structure
The source code for Fix Rigid/Abrade is located in *./src/RIGID* as *fix_rigid_abrade.cpp* and *fix_rigid_abrade.h*. The accompanying compute is defined in *compute_rigid_local_abrade.cpp* and *compute_rigid_local_abrade.h* in the same directory. 

## Getting Running
The code can be compiled with CMake, or the traditional MAKE, with the GRANULAR, MOLECULE, and RIGID packages. For the latter, on MacOS, the following commands can be used in a terminal directed at *./src*:

-   make yes-granular
-   make yes-rigid
-   make yes-molecule
-   make mpi
-   mv the lmp_mpi executable to your <project_directory>

For other operating systems, users are directed to the LAMMPS manual (https://docs.lammps.org/Manual.html).

Example input scripts for Fix Rigid/Abrade are located in *./examples/rigid/Fix_Rigid_Abrade* and the associated documentation can be made through *./doc/src*.

To run an example script:
-   Open the relevant ./Example_Scripts/ directory in terminal
-   "mpirun --oversubscribe -np N lmp_mpi -in in.<example_script>" where N is the number of processors you wish to run

## Additional Resources

Simulation outputs, such as dump files, can be inspected in Ovito (https://www.ovito.org/) or other visualiser of your choice. Additional simulation scripts can be found in the publication's Edinburgh DataShare repository (https://datashare.ed.ac.uk/handle/10283/9099?show=full).
