#!/bin/bash
set -x
export BASH_ROOT="$( cd "$( dirname "$BASH_SOURCE" )" && pwd )"
DATA_SUBDIR="/data_dirs/"
DATA_ROOT=$BASH_ROOT$DATA_SUBDIR

if [ ! -d $DATA_ROOT ]; then
    wget https://github.com/upc-arco/gpu-app-collection-simulation-dataset/releases/download/v1.0/other-apps-part1.tar.gz
    wget https://github.com/upc-arco/gpu-app-collection-simulation-dataset/releases/download/v1.0/other-apps-part2.tar.gz
    wget https://github.com/upc-arco/gpu-app-collection-simulation-dataset/releases/download/v1.0/other-apps-part3.tar.gz
    wget https://github.com/upc-arco/gpu-app-collection-simulation-dataset/releases/download/v1.0/other-apps-part4.tar.gz
    wget https://github.com/upc-arco/gpu-app-collection-simulation-dataset/releases/download/v1.0/other-apps-part5.tar.gz
    wget https://github.com/upc-arco/gpu-app-collection-simulation-dataset/releases/download/v1.0/other-apps-part6.tar.gz
    wget https://github.com/upc-arco/gpu-app-collection-simulation-dataset/releases/download/v1.0/other-apps-part7.tar.gz
    wget https://github.com/upc-arco/gpu-app-collection-simulation-dataset/releases/download/v1.0/other-apps-part8.tar.gz
    wget https://github.com/upc-arco/gpu-app-collection-simulation-dataset/releases/download/v1.0/other-apps-part9.tar.gz
    wget https://github.com/upc-arco/gpu-app-collection-simulation-dataset/releases/download/v1.0/other-apps-part10.tar.gz
    wget https://github.com/upc-arco/gpu-app-collection-simulation-dataset/releases/download/v1.0/tango-data-part1.tar.gz
    wget https://github.com/upc-arco/gpu-app-collection-simulation-dataset/releases/download/v1.0/tango-data-part2.tar.gz
    mkdir -p $DATA_ROOT
    tar xzvf other-apps-part1.tar.gz
    tar xzvf other-apps-part2.tar.gz
    tar xzvf other-apps-part3.tar.gz
    tar xzvf other-apps-part4.tar.gz
    tar xzvf other-apps-part5.tar.gz
    tar xzvf other-apps-part6.tar.gz
    tar xzvf other-apps-part7.tar.gz
    tar xzvf other-apps-part8.tar.gz
    tar xzvf other-apps-part9.tar.gz
    tar xzvf other-apps-part10.tar.gz
    tar xzvf tango-data-part1.tar.gz -C $DATA_ROOT
    tar xzvf tango-data-part2.tar.gz -C $DATA_ROOT
    rm other-apps-part1.tar.gz
    rm other-apps-part2.tar.gz
    rm other-apps-part3.tar.gz
    rm other-apps-part4.tar.gz
    rm other-apps-part5.tar.gz
    rm other-apps-part6.tar.gz
    rm other-apps-part7.tar.gz
    rm other-apps-part8.tar.gz
    rm other-apps-part9.tar.gz
    rm other-apps-part10.tar.gz
    rm tango-data-part1.tar.gz
    rm tango-data-part2.tar.gz
    mv data_dirs/tango/CifarNet data_dirs/tango/Tango-CN
    mv data_dirs/tango/ResNet data_dirs/tango/Tango-RN
    mv data_dirs/tango/LSTM data_dirs/tango/Tango-LSTM
    mv data_dirs/tango/GRU data_dirs/tango/Tango-GRU
    mv data_dirs/tango/AlexNet data_dirs/tango/Tango-AN
    mv data_dirs/tango/SqueezeNet data_dirs/tango/Tango-SN
    mkdir -p $DATA_ROOT/cuda/lonestargpu-2.0/lonestar-bfs-wla
    ln -s $DATA_ROOT/cuda/lonestargpu-2.0/inputs $DATA_ROOT/cuda/lonestargpu-2.0/lonestar-bfs-wla/data
fi
