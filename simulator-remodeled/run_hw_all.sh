#!/bin/bash
# Run HW profiling for all rodinia benchmarks with sudo
set -uo pipefail

HWROOT="/home/yanggl/code/modern-gpu-simulator/simulator-remodeled/hw_run/device-0/12.6"
export PATH=/usr/local/cuda-12.6/bin:$PATH
export SUDO_ASKPASS=/tmp/askpass.sh
cat > /tmp/askpass.sh << 'EOF'
#!/bin/bash
echo "BICS22ygl"
EOF
chmod +x /tmp/askpass.sh

# Refresh sudo credentials
SUDO_ASKPASS=/tmp/askpass.sh sudo -A -v

TOTAL=0
PASS=0
FAIL=0
for bench_dir in "$HWROOT"/*/; do
    bench=$(basename "$bench_dir")
    for args_dir in "$bench_dir"*/; do
        args=$(basename "$args_dir")
        TOTAL=$((TOTAL + 1))
        run_sh="$args_dir/run.sh"
        if [ ! -f "$run_sh" ]; then
            echo "SKIP $bench/$args (no run.sh)"
            continue
        fi
        echo "[$TOTAL] $bench/$args"
        pushd "$args_dir" > /dev/null
        if SUDO_ASKPASS=/tmp/askpass.sh sudo -A bash run.sh > run_output.log 2>&1; then
            echo "  DONE"
            PASS=$((PASS + 1))
        else
            echo "  FAIL"
            FAIL=$((FAIL + 1))
        fi
        popd > /dev/null
        # Keep sudo alive
        SUDO_ASKPASS=/tmp/askpass.sh sudo -A -v 2>/dev/null
    done
done

echo ""
echo "HW profiling complete: $PASS/$TOTAL passed, $FAIL failed"
rm -f /tmp/askpass.sh
