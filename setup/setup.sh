#!/bin/bash


## SETUP VARS
ENV_FILE="setup/environment.yml"
# Extract the environment name from the .yml file
ENV_NAME=$(grep "name:" $ENV_FILE | cut -d " " -f 2)
SETUP_SCRIPT="setup/setup_handshapes.py"

# Check conda
if ! command -v conda &> /dev/null
then
    echo "conda could not be found, please ensure it is installed and added to your PATH."
    exit 1
fi

echo "Creating Conda environment from $ENV_FILE..."
conda env create -f $ENV_FILE

if [ $? -eq 0 ]; then
    echo "Conda environment '$ENV_NAME' created successfully."
else
    echo "Failed to create Conda environment from $ENV_FILE."
    exit 1
fi

echo "Activating Conda environment '$ENV_NAME'..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

if [ $? -eq 0 ]; then
    echo "Conda environment '$ENV_NAME' activated."
else
    echo "Failed to activate Conda environment '$ENV_NAME'."
    exit 1
fi

echo "Running Python script/command..."
python $SETUP_SCRIPT


echo "Setup Completed."