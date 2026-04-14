#!/bin/bash
set -uo pipefail

SIMROOT="/home/yanggl/code/modern-gpu-simulator/simulator-remodeled"

export LD_LIBRARY_PATH="$SIMROOT/gpu-simulator/gpgpu-sim/lib/gcc-/cuda-12060/release:${LD_LIBRARY_PATH:-}"

BINARY="$SIMROOT/gpu-simulator/bin/release/accel-sim.out"
TRACES_ROOT="$SIMROOT/hw_run/traces/device-0/12.6"
GPGPU_CFG_DIR="$SIMROOT/gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM87_ORIN_AGX"
TRACE_CFG_DIR="$SIMROOT/gpu-simulator/configs/tested-cfgs/SM87_ORIN_AGX"
RUN_DIR="$SIMROOT/sim_run_12.6"
CFG_NAME="SM87_ORIN_AGX_MAXN-SASS"

if [ ! -f "$BINARY" ]; then
    echo "ERROR: simulator binary missing: $BINARY"
    exit 1
fi

declare -A BENCHMARKS
for bench_dir in "$TRACES_ROOT"/*/; do
    bench=$(basename "$bench_dir")
    args=$(ls "$bench_dir" | head -1)
    if [ -d "$bench_dir/$args/traces" ]; then
        BENCHMARKS["$bench"]="$args"
    fi
done

TOTAL=${#BENCHMARKS[@]}
N=0
PASS=0
FAIL=0

echo "============================================"
echo "  Orin MAXN run: $TOTAL benchmarks"
echo "  Config: $CFG_NAME"
echo "  Started: $(date)"
echo "============================================"

for bench in $(echo "${!BENCHMARKS[@]}" | tr ' ' '\n' | sort); do
    args="${BENCHMARKS[$bench]}"
    N=$((N + 1))

    work="$RUN_DIR/$bench/$args/$CFG_NAME"
    mkdir -p "$work"

    cp "$GPGPU_CFG_DIR/gpgpusim.config" "$work/"
    cp "$GPGPU_CFG_DIR/config_ampere_islip.icnt" "$work/"
    cp "$GPGPU_CFG_DIR/accelwattch_sass_sim.xml" "$work/"
    cp "$TRACE_CFG_DIR/trace.config" "$work/"

    ln -sfn "$TRACES_ROOT/$bench/$args/traces" "$work/traces"

    echo "[$N/$TOTAL] $bench  ($(date '+%H:%M:%S'))"
    cd "$work"
    if "$BINARY" -config gpgpusim.config -config trace.config \
        -trace ./traces/dynamic_trace.pb > sim.out 2>&1; then
        cycles=$(grep "gpu_tot_sim_cycle" sim.out | tail -1 | awk -F'=' '{print $2}' | tr -d ' ')
        echo "  DONE  cycles=$cycles  ($(date '+%H:%M:%S'))"
        PASS=$((PASS + 1))
    else
        rc=$?
        echo "  FAILED rc=$rc"
        FAIL=$((FAIL + 1))
    fi
    cd "$SIMROOT"
done

echo "============================================"
echo "  Done: $PASS pass / $FAIL fail"
echo "  Finished: $(date)"
echo "============================================"
