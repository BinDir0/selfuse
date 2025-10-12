#!/bin/bash
set -e

# Install system dependencies
echo "Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y build-essential cmake git wget curl unzip software-properties-common apt-transport-https ca-certificates gnupg lsb-release
sudo apt-get install -y libjpeg-dev libpng-dev libtiff-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev libgtk-3-dev libatlas-base-dev gfortran
sudo apt-get install -y python3-dev python3-pip python3-venv libhdf5-dev pkg-config

# Add NVIDIA repository and install NCCL
echo "Adding NVIDIA repository and installing NCCL..."
# make sure the right version of the system is used
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get install -y libnccl2 libnccl-dev
rm -f cuda-keyring_1.0-1_all.deb

# Install pdsh for deepspeed
sudo apt-get install -y pdsh

conda create -n legendvla python=3.10
conda activate legendvla

# install VILA
git clone https://github.com/Ivan-Zhong/VILA.git
cd VILA
./environment_setup.sh
pip install -e .

# install manopth
cd ..
git clone https://github.com/hassony2/manopth
cd manopth
conda env update -n legendvla -f environment.yml 
pip install -e .

# install legendvla
cd ..
pip install -r requirements.txt
