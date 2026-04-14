#!/bin/bash
set -euo pipefail

# ============================================================
# Ablation Study Runner: C0-C4 on SM87 Orin AGX
# Runs all 5 configs x 10 benchmarks serially
# ============================================================

SIMROOT="/home/yanggl/code/modern-gpu-simulator/simulator-remodeled"

# Setup environment (LD_LIBRARY_PATH for simulator shared libs)
export LD_LIBRARY_PATH="$SIMROOT/gpu-simulator/gpgpu-sim/lib/gcc-/cuda-12060/release:${LD_LIBRARY_PATH:-}"

BINARY="$SIMROOT/gpu-simulator/bin/release/accel-sim.out"
TRACES_ROOT="$SIMROOT/hw_run/traces/device-0/12.6"
GPGPU_CFGS="$SIMROOT/gpu-simulator/gpgpu-sim/configs/tested-cfgs"
TRACE_CFGS="$SIMROOT/gpu-simulator/configs/tested-cfgs"
RUN_DIR="$SIMROOT/sim_run_ablation"

CONFIGS=(
    "SM87_ORIN_AGX_ablation_C0_baseline"
    "SM87_ORIN_AGX_ablation_C1_L1"
    "SM87_ORIN_AGX_ablation_C2_L1L2"
)

if [ ! -f "$BINARY" ]; then
    echo "ERROR: Simulator binary not found at $BINARY"
    exit 1
fi

# Discover benchmarks and their args from traces directory
declare -A BENCHMARKS
for bench_dir in "$TRACES_ROOT"/*/; do
    bench=$(basename "$bench_dir")
    args=$(ls "$bench_dir" | head -1)
    if [ -d "$bench_dir/$args/traces" ]; then
        BENCHMARKS["$bench"]="$args"
    fi
done

TOTAL_CONFIGS=${#CONFIGS[@]}
TOTAL_BENCHMARKS=${#BENCHMARKS[@]}
TOTAL_RUNS=$((TOTAL_CONFIGS * TOTAL_BENCHMARKS))
CURRENT_RUN=0
PASS=0
FAIL=0

echo "============================================"
echo "  Ablation Study: $TOTAL_CONFIGS configs x $TOTAL_BENCHMARKS benchmarks = $TOTAL_RUNS runs"
echo "  Output: $RUN_DIR"
echo "  Started: $(date)"
echo "============================================"
echo ""

for cfg in "${CONFIGS[@]}"; do
    for bench in $(echo "${!BENCHMARKS[@]}" | tr ' ' '\n' | sort); do
        args="${BENCHMARKS[$bench]}"
        CURRENT_RUN=$((CURRENT_RUN + 1))

        # Create run directory
        work="$RUN_DIR/$bench/$args/$cfg"
        mkdir -p "$work"

        # Copy config files
        cp "$GPGPU_CFGS/$cfg/gpgpusim.config" "$work/"
        cp "$GPGPU_CFGS/$cfg/config_ampere_islip.icnt" "$work/"
        cp "$GPGPU_CFGS/$cfg/accelwattch_sass_sim.xml" "$work/"
        cp "$TRACE_CFGS/$cfg/trace.config" "$work/"

        # Symlink traces
        ln -sfn "$TRACES_ROOT/$bench/$args/traces" "$work/traces"

        echo "[$CURRENT_RUN/$TOTAL_RUNS] $cfg / $bench"
        echo "  Started: $(date '+%H:%M:%S')"

        # Run simulation
        cd "$work"
        if "$BINARY" -config gpgpusim.config -config trace.config \
            -trace ./traces/dynamic_trace.pb > sim.out 2>&1; then
            echo "  DONE ($(date '+%H:%M:%S'))"
            PASS=$((PASS + 1))
        else
            echo "  FAILED (exit code $?)"
            FAIL=$((FAIL + 1))
        fi
        cd "$SIMROOT"
    done
done

echo ""
echo "============================================"
echo "  Ablation Study Complete"
echo "  Finished: $(date)"
echo "  Pass: $PASS / $TOTAL_RUNS    Fail: $FAIL / $TOTAL_RUNS"
echo "  Results in: $RUN_DIR"
echo "============================================"
