#!/bin/bash
set -e

# Install Miniconda
wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p /opt/miniconda
rm /tmp/miniconda.sh

# Init conda
eval "$(/opt/miniconda/bin/conda shell.bash hook)"
echo 'eval "$(/opt/miniconda/bin/conda shell.bash hook)"' >> ~/.bashrc
echo "conda activate pyg" >> ~/.bashrc

# Accept Terms of Service
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
apt-get update
apt-get install -y cuda-toolkit-12-6

export CPATH=/usr/local/cuda/include:$CPATH
export PATH=/usr/local/cuda/bin:$PATH
apt-get install -y ninja-build build-essential cmake

# Create env with Python 3.11
conda create -n pyg python=3.11 -y
conda activate pyg

# Install PyTorch 2.12 + CUDA 12.6 via pip (conda packages lag behind)
pip install torch==2.12.0 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu126

# Install PyG + extensions
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse \
    -f https://data.pyg.org/whl/torch-2.12.0+cu126.html

# Other packages
pip install torchmetrics jupyterlab numpy matplotlib pandas ogb

# Verify
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
python -c "import torch_geometric; print('PyG:', torch_geometric.__version__)"

echo "Done! Run: conda activate pyg"
