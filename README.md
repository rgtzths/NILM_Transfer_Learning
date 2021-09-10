# ICMLC_ICWAPR_code_base

## Python Requirements
### Create a conda environment
Create a conda environment
- conda create --name nilmtk python

Activate the environment
- conda activate nilmtk

Add conda-forge to conda
- conda config --add channels conda-forge

### Install Packages (in the environment)
- Install NILMTK
  - conda install -c nilmtk nilmtk=0.4.3
- Install tensorflow
  - pip3 install tensorflow==2.5.0
- Install PyWavellets
  - pip3 install PyWavelets==1.1.1
- Install cvxpy
  - conda install cvxpy=1.1.13
- Download UKDale H5 - https://data.ukedc.rl.ac.uk/browse/edc/efficiency/residential/EnergyConsumption/Domestic/UK-DALE-2017/UK-DALE-FULL-disaggregated/ukdale.h5.zip
- Download UKDale Full Dataset - https://data.ukedc.rl.ac.uk/browse/edc/efficiency/residential/EnergyConsumption/Domestic/UK-DALE-2017/UK-DALE-FULL-disaggregated/ukdale.zip
