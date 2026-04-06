#!/bin/bash
export DATA_ROOT="$( cd "$( dirname "$BASH_SOURCE" )" && pwd )"
export DATA_ROOT=$DATA_ROOT/../../hw_run

usage() {
    echo "Usage: $0 -c <card>"
    echo ""
    echo "Required:"
    echo "  -c <card>   GPU card to download hw data for"
    echo ""
    echo "Available cards:"
    echo "  ampere-rtx3070"
    echo "  turing-rtx2060"
    echo "  pascal-titanx"
    echo "  pascal-p100"
    echo "  pascal-1080ti"
    echo "  fermi-gtx480"
    echo "  volta-titanv"
    echo "  volta-quadro-v100"
    echo "  volta-tesla-v100"
    echo "  kepler-titan"
    echo ""
    echo "Examples:"
    echo "  $0 -c ampere-rtx3070"
    echo "  $0 -c volta-titanv"
    exit 1
}

CARD=""
while getopts "c:h" opt; do
    case $opt in
        c) CARD="$OPTARG" ;;
        h) usage ;;
        *) usage ;;
    esac
done

if [ -z "$CARD" ]; then
    echo "Error: You must specify a GPU card with -c"
    echo ""
    usage
fi

get_tarfile() {
    case "$1" in
        ampere-rtx3070)    echo "ampere.rtx3070.tgz" ;;
        turing-rtx2060)    echo "turing.rtx2060.cycle.tgz" ;;
        pascal-titanx)     echo "pascal.titanx.cycle.tgz" ;;
        pascal-p100)       echo "pascal.tesla.p100.cycles.tgz" ;;
        pascal-1080ti)     echo "pascal.1080ti.cycle.tgz" ;;
        fermi-gtx480)      echo "fermi.gtx480.cycle.tgz" ;;
        volta-titanv)      echo "volta.titanv.tgz" ;;
        volta-quadro-v100) echo "quadro.v100.cycle.tgz" ;;
        volta-tesla-v100)  echo "tesla.v100.tgz" ;;
        kepler-titan)      echo "kepler.titan.cycle.tgz" ;;
    esac
}

get_datadir() {
    case "$1" in
        ampere-rtx3070)    echo "AMPERE-RTX3070" ;;
        turing-rtx2060)    echo "TURING-RTX2060" ;;
        pascal-titanx)     echo "TITAN-X-PASCAL" ;;
        pascal-p100)       echo "TESLA-P100" ;;
        pascal-1080ti)     echo "1080TI_PASCAL" ;;
        fermi-gtx480)      echo "GTX480" ;;
        volta-titanv)      echo "TITANV" ;;
        volta-quadro-v100) echo "QUADRO-V100" ;;
        volta-tesla-v100)  echo "TESLA-V100" ;;
        kepler-titan)      echo "KEPLER-TITAN" ;;
    esac
}

TARFILE=$(get_tarfile "$CARD")
DATADIR=$(get_datadir "$CARD")

if [ -z "$TARFILE" ]; then
    echo "Error: Unknown card '$CARD'"
    echo ""
    usage
fi

mkdir -p "$DATA_ROOT"

FULL_DATA_PATH="$DATA_ROOT/$DATADIR"
if [ -d "$FULL_DATA_PATH" ]; then
    echo "Data already exists: $FULL_DATA_PATH"
    echo "Skipping download."
    exit 0
fi

BASE_URL="https://engineering.purdue.edu/tgrogers/gpgpu-sim/hw_data"
wget "$BASE_URL/$TARFILE"
tar -xzvf "$TARFILE" -C "$DATA_ROOT"
rm "$TARFILE"
