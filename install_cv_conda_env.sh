#!/bin/bash
# Salva il nome dell'environment in una variabile
ENV_NAME=comvis

# Chiama questo script dalla cartella che lo contiene con il comando: bash install_ppi_conda_env.sh

# Crea un nuovo ambiente conda chiamato ENV_NAME con Python 3.11.3
conda create -n $ENV_NAME python=3.11.3 -y

# Attiva l'ambiente
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

# Installa pip tramite conda-forge
conda install -c conda-forge pip -y

# Cerca il pip installato da conda e lo usa per installare i pacchetti Python
PIP_PATH=$(which pip)
$PIP_PATH install --upgrade pip

# Installa torch e torchvision con supporto GPU per cuda 11.8 usando  pip dell environment
$PIP_PATH install torch==2.2.0+cu118 torchvision==0.17.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html

$PIP_PATH install tqdm ripser ipykernel kaleido pytorch-fid datasets gdown

$PIP_PATH install --upgrade nbformat

# Installa i pacchetti Python di base per computer vision con conda senza specificare la versione
conda install -c conda-forge numpy matplotlib scikit-learn scikit-image pandas h5py ripser -y