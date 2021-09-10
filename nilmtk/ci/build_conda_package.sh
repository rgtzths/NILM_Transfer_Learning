echo Building conda package...

# Build conda packages
mkdir ../artifacts
export ARTIFACTS_FOLDER=`readlink -f ../artifacts`

conda config --set anaconda_upload no
conda config --add channels conda-forge
conda config --add channels nilmtk

# Replace version with the tag
sed -i "s/0\.4\.0\.dev1/3.5/g" conda/meta.yaml

conda-build --quiet --no-test --output-folder "$ARTIFACTS_FOLDER" conda 