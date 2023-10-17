# ICMLC_ICWAPR_code_base

## Python Requirements
### Create a conda environment

Create a conda environment
- conda env create -f environment.yml

Activate the environment
- conda activate nilmtk-env

Install NILMTK with the changes made
- conda install nilmtk-3.5-py_0.tar.bz2 (might need to build package if using not using liinux - the dir with everything you need is the nilmtk folder)

### Install Packages (in the environment)
- Install tensorflow
  - pip3 install tensorflow==2.5.0
- Install PyWavellets
  - pip3 install PyWavelets==1.1.1

## Datasets

The datasets should be placed outside the repository in a folder called datasets.
The folder structure should be:

- datasets
  - ukdale
    - ukdale.h5
  - refit
    - refit.h5

- Download UKDale H5 - https://data.ukedc.rl.ac.uk/browse/edc/efficiency/residential/EnergyConsumption/Domestic/UK-DALE-2017/UK-DALE-FULL-disaggregated/ukdale.h5.zip

- Download REFIT CSV - 

- Convert REFIT to H5 using the NILMTK Converter

### Side Note

In the transfer learning process you need to change the fridge_frezzer name in the refit base_results to fridge.

## Authors

* **Rafael Teixeira** - [rgtzths](https://github.com/rgtzths)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Citation

If you use this code please site our work:
Teixeira, Rafael & Antunes, Mário & Gomes, Diogo. (2021). Using Deep Learning and Knowledge Transfer to Disaggregate Energy Consumption. 1-7. 10.1109/ICWAPR54887.2021.9736149. 
