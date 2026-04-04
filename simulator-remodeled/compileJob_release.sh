#!/bin/bash

#SBATCH --output=compile_release_std-%j.out
#SBATCH --error=compile_release_error-%j.out
#SBATCH -q all

source ./gpu-simulator/setup_environment_no_git.sh && make clean -C ./gpu-simulator/ && make -C ./gpu-simulator/
