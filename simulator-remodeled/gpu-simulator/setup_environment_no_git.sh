#!/bin/bash
# Copyright (c) 2020 Timothy Rogers, Purdue University
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice, this
# list of conditions and the following disclaimer in the documentation and/or
# other materials provided with the distribution.
# Neither the name of The University of British Columbia nor the names of its
# contributors may be used to endorse or promote products derived from this
# software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# MOD. to pick the right compiler
if [ $IS_SERT = '1' ] ; then
    # VERSION=9.3.0
    # DIR=/Soft/gcc/$VERSION
    # export CC=$DIR/bin/gcc
    # export CXX=$DIR/bin/g++
    # export F77=$DIR/bin/gfortran
    # export F90=$DIR/bin/gfortran
    # export PATH=$DIR/bin:$PATH
    echo "Warning: IS_SERT is set to 1, but no specific compiler settings are applied in this script."
fi



export ACCELSIM_SETUP_ENVIRONMENT_WAS_RUN=
export ACCELSIM_ROOT="$( cd "$( dirname "$BASH_SOURCE" )" && pwd )"
export CUDA_VERSION=`nvcc --version | grep release | sed -re 's/.*release ([0-9]+\.[0-9]+).*/\1/'`;

if [ $# -ge '1' ] ;
then
    export ACCELSIM_CONFIG=$1
else
    export ACCELSIM_CONFIG=release
fi

if [ $# -ge '2' ] ;
then
    export SANITIZER_COM_FLAG=$2
else
    export SANITIZER_COM_FLAG=sanitizer_disabled
fi

source $ACCELSIM_ROOT/gpgpu-sim/setup_environment $ACCELSIM_CONFIG

export ACCELSIM_SETUP_ENVIRONMENT_WAS_RUN=1
