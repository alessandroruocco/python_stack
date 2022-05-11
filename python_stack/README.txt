LPI Scripts
===========

This is a collection of tools for running and analysing EPOCH simulations of LPI

Requirements
------------

 - Python3
 - EPOCH
Installation
------------

1. Install virtualenv if you don't already have it:

pip install --user virtualenv

2. Create a python virtual environment for this repository. You will need to
   activate this each time you want to use the scripts. Assuming you want your
   virtual environment in the current directory:

virtualenv venv

3. You can activate the environment using

source venv/bin/activate

4. Install required python modules to the new virtual environment:

pip install scipy numpy matplotlib sympy pyfftw numba pathos subprocess32 daemon scikit-image

5. Install the python sdf module. This is included in the EPOCH repository and
   allows python to read EPOCH output files. Assuming you are in the root
   directory of your epoch repository:

rm ./SDF/utilities/build
export C_INCLUDE_PATH="`pwd`/SDF/C/include:$C_INCLUDE_PATH"
export LIBRARY_PATH="`pwd`/SDF/C/lib:$LIBRARY_PATH"
pip install ./SDF/utilities/

6. Install the sdfUtils, srsUtils, pbsUtils and pyEPOCH modules. The '-e' flag
   makes these 'editable', so you can modify them without having to reinstall.
   Assuming you are in the same directory as the README:

pip install -e ./sdfUtils
pip install -e ./srsUtils
pip install -e ./pbsUtils
pip install -e ./pyEPOCH

************* From this on, it is outdated ********











(If I want to fix a certain interval, --Intervallo 'True' --IntervalInitial .... --IntervalFinal

/home/vol08/scarf928/python_from_alex/plotFieldEnergy.py ./  --prefix 'regular_' --Lambda 0.35  --space --densitySpecies Electron  --densProfFile regular_00000.sdf --Snapshots 80 

--log --minF 1e-6 --coloreMap inferno --minNeTick 0.05 --neTickInterval 0.05

Derived_Poynting_Flux_x  --LaserIntensity 2e15 -o sx.jpg
Derived_Number_Density_Electron -o ne.jpg
Derived_Number_Density_a1_ion -o ni.jpg
Electric_Field_Ex -o ex.jpg
Magnetic_Field_Bz -o bz.jpg

Derived_Temperature_Electron

--Tempora_cut True --Intervallo 'True' --IntervalInitial .... --IntervalFinal

iiiiiiiiiiii
ii Mio.    iiii
iiiiiiiiiiii


/home/vol08/scarf928/python_from_alex/python_boundaries_ref_trans.py


 

f.Derived_Number_Density_Electron

   It is recommended that you install the modules in the order listed.

Description
-----------

The following scripts are provided alongside this README:

 - singleSpeckle.py: Generates 2D EPOCH input decks for LPI problems.

   Decks include multiple output blocks to implement diagnostics. The data
   files produced by these diagnostics are used by the scripts listed below.

   The output files are prefixed:
    - regular: Snapshots of fields and moments of the distribution function.
	- boundary: High-frequency (ω_Νyquist > ω_laser) snapshots of the fields in
	    a strip of cells surrounding the domain. Used to calculate SRS & SBS
		reflectivity and laser transmission.
	- strip: Same as boundary data, but for a strip of cells along the center
	    of the domain (y=0). Used to track propagation of the laser and
		as a diagnostic of SRS and laser absorption.
    - probe: Data from particles leaving the domain. This is used to measure 
		hot electron fluxes.
	- particles: Data from particles in the bulk domain. This can be used to
		investigate kinetic effects within the domain.

 - plotFieldSnapshots.py: Plots field snapshots (regular*.sdf)

 - plotFieldEnergy.py: Produces space-time maps of electron plasma wave energy

	Averages the electrostatic (E_x) field component over the transverse (y)
	dimension which primarily captures electron plasma waves, and plots a
	space-time map of this.

 - reflectivity.py: Reflectivity & transmission (boundary*.sdf)

 - plotStripEnergy.py: Generates space-time plots of laser intensity and SRS
	emission (strip*.sdf)

 - probeElectrons.py: Generates a histogram of outgoing electron data that may
    be plotted by plotProbeElectrons (probe*.sdf)

 - plotProbeElectrons.py: Plots the histogrammed hot-electron data (probe*.sdf)

The scripts make use of the four supplied python modules:

 - srsUtils: Contains functions for performing useful LPI calculations.
	
	The module is split into a series of submodules:
	 - langmuir: Electron plasma wave frequency and damping
	 - srs: Calculations relevant to SRS. E.g. frequency/wavenumber matching,
	     growth rates, inhomogeneous gain.
	 - tpd: Two-plasmon decay matching, growth rates, inhomogeneous gain.
	 - filter: Digital filtering functions

 - sdfUtils: Contains helper functions for handling SDF files

	Two utilities are supplied that are added to the user's PATH on
	installation:
	 - lsPrefix: Supply a folder name as an argument to list all SDF files,
	     grouped according to their prefix.
	 - sdfUtils: Supply a SDF file as an argument to list the contents of the
	     file.

 - pyEPOCH: Library for reading and generating EPOCH input decks.

 - pbsUtils: Library for generating HPC job scripts. I haven't used this in a
     while so might not work very well.

Usage:

  All scripts are controlled via command line arguments. To print a list of the
  arguments use the --help argument. Unfortunately, documentation is
  lacking/non-existant so the best way to learn what the scripts do is to read
  the code.
  
  A good starting point would be to run a very small 2D simulation. To generate
  an input deck, use this command:

./singleSpeckle.py -n 0.10 --nCrit -L 300e-6 -T 4.5 --keV -I 1e16 --IUnit wcm2 -F 6.7 --planeWave -t 10e-12 --ppce 128 -b 0.0 -o ./test

  This will create a folder called "test" containing an EPOCH input deck. The
  parameters would be:
  
   - Central density: 0.10 n_crit
   - Density scale-length ( L_n == dn_e/dx / n_e ): 300μm
   - Electron temperature: 4.5keV
   - Laser Intensity: 10^16 W/cm^2
   - Laser profile: plane wave
   - Simulation duration: 2ps
   - Particles per cell: 128
   - Mobile ions: No

  These are just made up numbers to get started so I've no idea if they do
  anything interesting!

  As the name suggests the script was originally written to simulate individual
  speckles, so the domain is sized according to the anticipated speckle size,
  which is determined by the lens F-number. This should really be changed, but
  as is there are a few hacky arguments you can use to change the domain size.
  The command above generates a plane wave beam, but the domain is sized to fit
  a F/6.7 speckle. 

  The command also prints a bunch of information that can be useful for setting
  up a simulation. To play around with the parameters without writing the input
  deck to disk, replace '-o ./test' with '--dummy'. Probably the most helpful
  information to begin with is the estimate of required HPC time to complete the
  simulation, which is a rough estimate valid for the HPCs I've used.


