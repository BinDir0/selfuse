#!/bin/bash
set -e

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

# install pdsh for deepspeed
sudo apt-get install pdsh

# install nccl for deepspeed
sudo apt-get install libnccl2 libnccl-dev
