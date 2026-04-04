#!/bin/bash

#SBATCH --output=compile_debug_std-%j.out
#SBATCH --error=compile_debug_error-%j.out
#SBATCH -q all

source ./gpu-simulator/setup_environment_no_git.sh debug && make clean -C ./gpu-simulator/ && make -C ./gpu-simulator/
