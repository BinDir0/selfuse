#!/bin/bash
set -e

conda create -n egovla python=3.10
conda activate egovla

# install VILA
git clone https://github.com/NVlabs/VILA
cd VILA
./environment_setup.sh
pip install -e .

# install manopth
cd ..
git clone https://github.com/hassony2/manopth
cd manopth
conda env update -n egovla -f environment.yml 
pip install -e .

# install egovla
cd ..
pip install -r requirements.txt

# install pdsh for deepspeed
sudo apt-get install pdsh
