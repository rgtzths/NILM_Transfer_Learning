## To install this package run in this folder the following commands

- conda create -n builder conda-build anaconda-client
- conda activate builder
- bash ci/build_conda_package.sh

### The package for instalation should be in ../artifacts/noarch/ with the name nilmtk-3.5-...