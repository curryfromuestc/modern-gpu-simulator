#!/bin/bash
set -x
export BASH_ROOT="$( cd "$( dirname "$BASH_SOURCE" )" && pwd )"
DATA_SUBDIR="/data_dirs/"
DATA_ROOT=$BASH_ROOT$DATA_SUBDIR

usage() {
    echo "Usage: $0 -d <dataset>"
    echo ""
    echo "Required:"
    echo "  -d <dataset>   Dataset to download"
    echo ""
    echo "Available datasets:"
    echo "  other-apps     GPU application benchmarks (10 parts)"
    echo "  tango          Tango DNN benchmarks (2 parts)"
    echo ""
    echo "Examples:"
    echo "  $0 -d other-apps"
    echo "  $0 -d tango"
    exit 1
}

DATASET=""
while getopts "d:h" opt; do
    case $opt in
        d) DATASET="$OPTARG" ;;
        h) usage ;;
        *) usage ;;
    esac
done

if [ -z "$DATASET" ]; then
    echo "Error: You must specify a dataset with -d"
    echo ""
    usage
fi

BASE_URL="https://github.com/upc-arco/gpu-app-collection-simulation-dataset/releases/download/v1.0"

download_other_apps() {
    mkdir -p "$DATA_ROOT"
    for i in $(seq 1 10); do
        TARFILE="other-apps-part${i}.tar.gz"
        wget "${BASE_URL}/${TARFILE}"
        tar xzvf "$TARFILE"
        rm "$TARFILE"
    done
    mkdir -p "$DATA_ROOT/cuda/lonestargpu-2.0/lonestar-bfs-wla"
    ln -s "$DATA_ROOT/cuda/lonestargpu-2.0/inputs" \
          "$DATA_ROOT/cuda/lonestargpu-2.0/lonestar-bfs-wla/data"
}

download_tango() {
    mkdir -p "$DATA_ROOT"
    for i in 1 2; do
        TARFILE="tango-data-part${i}.tar.gz"
        wget "${BASE_URL}/${TARFILE}"
        tar xzvf "$TARFILE" -C "$DATA_ROOT"
        rm "$TARFILE"
    done
    mv "$DATA_ROOT/tango/CifarNet"   "$DATA_ROOT/tango/Tango-CN"
    mv "$DATA_ROOT/tango/ResNet"     "$DATA_ROOT/tango/Tango-RN"
    mv "$DATA_ROOT/tango/LSTM"       "$DATA_ROOT/tango/Tango-LSTM"
    mv "$DATA_ROOT/tango/GRU"        "$DATA_ROOT/tango/Tango-GRU"
    mv "$DATA_ROOT/tango/AlexNet"    "$DATA_ROOT/tango/Tango-AN"
    mv "$DATA_ROOT/tango/SqueezeNet" "$DATA_ROOT/tango/Tango-SN"
}

case "$DATASET" in
    other-apps)
        download_other_apps
        ;;
    tango)
        download_tango
        ;;
    *)
        echo "Error: Unknown dataset '$DATASET'"
        echo "Available datasets: other-apps, tango"
        exit 1
        ;;
esac
