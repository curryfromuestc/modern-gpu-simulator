#!/usr/bin/env python3
"""
Compute MAPE (Mean Absolute Percentage Error) between hardware and simulator cycles.

Usage:
    python3 compute_mape.py \
        --hw-dir hw_run/device-0/12.9/ \
        --sim-dir sim_run_12.9/ \
        --config RTXPRO6000-SASS \
        --clock-mhz 2430 \
        --suite rodinia_2.0-ft
"""

import os
import sys
import re
import csv
import argparse
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "util", "job_launching"))
import common


def parse_nsys_cycles(hw_dir, benchmark_name, args_folder):
    """Parse cycles from nsys cycles_processed.csv or cycles.csv."""
    bench_dir = os.path.join(hw_dir, benchmark_name, args_folder)
    for fname in ["cycles_processed.csv", "cycles.csv"]:
        fpath = os.path.join(bench_dir, fname)
        if os.path.isfile(fpath):
            total_duration_ns = 0.0
            kernel_count = 0
            with open(fpath, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('"'):
                        # Try parsing CSV header or data
                        pass
                    # nsys gputrace CSV format has Duration column
                    # Lines look like: Start,Duration,...,Name
                    parts = line.split(",")
                    if len(parts) >= 2:
                        try:
                            duration = float(parts[1].strip().strip('"'))
                            total_duration_ns += duration
                            kernel_count += 1
                        except ValueError:
                            continue
            return total_duration_ns, kernel_count
    return None, 0


def parse_ncu_cycles(hw_dir, benchmark_name, args_folder):
    """Parse cycles from ncu gpc__cycles_elapsed files."""
    bench_dir = os.path.join(hw_dir, benchmark_name, args_folder)
    if not os.path.isdir(bench_dir):
        return None, 0
    total_cycles = 0.0
    kernel_count = 0
    for fname in os.listdir(bench_dir):
        if "gpc__cycles_elapsed" in fname:
            fpath = os.path.join(bench_dir, fname)
            # Collect CSV lines after the "==PROF== Disconnected" marker
            csv_lines = []
            collecting = False
            with open(fpath, "r") as f:
                for line in f:
                    if "==PROF== Disconnected from process" in line:
                        collecting = True
                        continue
                    if collecting:
                        csv_lines.append(line)
            if not csv_lines:
                return None, 0
            # Parse using csv.reader to handle quoted fields with commas
            reader = csv.reader(csv_lines)
            rows = list(reader)
            if not rows:
                return None, 0
            # Skip header
            for row in rows[1:]:
                if not row:
                    continue
                try:
                    val_str = row[-1].strip().replace(",", "")
                    cycles = float(val_str)
                    total_cycles += cycles
                    kernel_count += 1
                except (ValueError, IndexError):
                    pass
            return total_cycles, kernel_count
    return None, 0


def parse_sim_cycles(sim_dir, benchmark_name, args_folder, config_name):
    """Parse gpu_tot_sim_cycle from simulator output."""
    run_dir = os.path.join(sim_dir, benchmark_name, args_folder, config_name)
    if not os.path.isdir(run_dir):
        return None

    # Look for the simulator output file
    for fname in os.listdir(run_dir):
        if fname.endswith(".o") or fname.endswith(".out") or fname == "accel-sim-out.log":
            continue
        fpath = os.path.join(run_dir, fname)
        if os.path.isfile(fpath):
            try:
                with open(fpath, "r") as f:
                    content = f.read()
                    matches = re.findall(r"gpu_tot_sim_cycle\s*=\s*(\d+)", content)
                    if matches:
                        return int(matches[-1])
            except (UnicodeDecodeError, PermissionError):
                continue

    # Also check for slurm output or stderr files
    for fname in sorted(os.listdir(run_dir)):
        fpath = os.path.join(run_dir, fname)
        if os.path.isfile(fpath):
            try:
                with open(fpath, "r") as f:
                    content = f.read()
                    matches = re.findall(r"gpu_tot_sim_cycle\s*=\s*(\d+)", content)
                    if matches:
                        return int(matches[-1])
            except (UnicodeDecodeError, PermissionError):
                continue
    return None


def main():
    parser = argparse.ArgumentParser(description="Compute MAPE between HW and simulator")
    parser.add_argument("--hw-dir", required=True, help="HW profiling output directory")
    parser.add_argument("--sim-dir", required=True, help="Simulation output directory")
    parser.add_argument("--config", required=True, help="Simulator config name (e.g. RTXPRO6000-SASS)")
    parser.add_argument("--clock-mhz", type=float, default=2430.0,
                        help="GPU core clock in MHz for nsys duration->cycles conversion")
    parser.add_argument("--suite", default="rodinia_2.0-ft", help="Benchmark suite name")
    parser.add_argument("--use-ncu", action="store_true", help="Use ncu output instead of nsys")
    parser.add_argument("--output", default=None, help="Output markdown file path")
    args = parser.parse_args()

    common.load_defined_yamls()
    benchmarks = common.gen_apps_from_suite_list([args.suite])

    results = []
    for bench in benchmarks:
        edir, ddir, exe, argslist = bench
        for argpair in argslist:
            bench_args = argpair["args"]
            args_folder = common.get_argfoldername(bench_args)
            full_name = os.path.join(exe, args_folder)

            # Get HW cycles
            if args.use_ncu:
                hw_cycles, kcount = parse_ncu_cycles(args.hw_dir, exe, args_folder)
            else:
                duration_ns, kcount = parse_nsys_cycles(args.hw_dir, exe, args_folder)
                if duration_ns is not None:
                    # Convert ns to cycles: cycles = duration_ns * clock_MHz / 1000
                    hw_cycles = duration_ns * args.clock_mhz / 1000.0
                else:
                    hw_cycles = None

            # Get sim cycles
            sim_cycles = parse_sim_cycles(args.sim_dir, exe, args_folder, args.config)

            if hw_cycles is not None and sim_cycles is not None and hw_cycles > 0:
                error_pct = (sim_cycles - hw_cycles) / hw_cycles * 100.0
                abs_error_pct = abs(error_pct)
            else:
                error_pct = None
                abs_error_pct = None

            results.append({
                "name": exe,
                "args": args_folder,
                "full": full_name,
                "hw_cycles": hw_cycles,
                "sim_cycles": sim_cycles,
                "error": error_pct,
                "abs_error": abs_error_pct,
            })

    # Print results
    print(f"{'Benchmark':<45} {'HW Cycles':>15} {'Sim Cycles':>15} {'Error%':>10} {'AbsError%':>10}")
    print("-" * 100)

    errors = []
    abs_errors = []
    hw_list = []
    sim_list = []

    for r in results:
        hw_str = f"{r['hw_cycles']:.0f}" if r["hw_cycles"] is not None else "N/A"
        sim_str = f"{r['sim_cycles']}" if r["sim_cycles"] is not None else "N/A"
        err_str = f"{r['error']:.2f}" if r["error"] is not None else "N/A"
        abs_str = f"{r['abs_error']:.2f}" if r["abs_error"] is not None else "N/A"
        print(f"{r['name']:<45} {hw_str:>15} {sim_str:>15} {err_str:>10} {abs_str:>10}")

        if r["error"] is not None:
            errors.append(r["error"])
            abs_errors.append(r["abs_error"])
            hw_list.append(r["hw_cycles"])
            sim_list.append(r["sim_cycles"])

    print("-" * 100)

    if errors:
        avg_error = sum(errors) / len(errors)
        avg_abs_error = sum(abs_errors) / len(abs_errors)
        median_abs = sorted(abs_errors)[len(abs_errors) // 2]

        # Pearson correlation
        n = len(hw_list)
        if n > 1:
            mean_hw = sum(hw_list) / n
            mean_sim = sum(sim_list) / n
            cov = sum((h - mean_hw) * (s - mean_sim) for h, s in zip(hw_list, sim_list)) / n
            std_hw = math.sqrt(sum((h - mean_hw) ** 2 for h in hw_list) / n)
            std_sim = math.sqrt(sum((s - mean_sim) ** 2 for s in sim_list) / n)
            pearson = cov / (std_hw * std_sim) if std_hw > 0 and std_sim > 0 else 0
        else:
            pearson = 0

        print(f"{'Average (MAPE)':<45} {'':>15} {'':>15} {avg_error:>10.2f} {avg_abs_error:>10.2f}")
        print(f"{'Median AbsError':<45} {'':>15} {'':>15} {'':>10} {median_abs:>10.2f}")
        print(f"{'Pearson Correlation':<45} {'':>15} {'':>15} {pearson:>10.4f} {'':>10}")
        print(f"\nTotal benchmarks: {len(results)}, with valid data: {len(errors)}")

    # Write markdown if requested
    if args.output and errors:
        with open(args.output, "w") as f:
            f.write(f"| Benchmark | Arguments | Error_Cycles % | AbsError_Cycles % |\n")
            f.write(f"|:--|:--|--:|--:|\n")
            for r in results:
                if r["error"] is not None:
                    f.write(f"| {r['name']} | {r['args']} | {r['error']:.4f} | {r['abs_error']:.4f} |\n")
            f.write(f"| Average | | {avg_error:.4f} | {avg_abs_error:.4f} |\n")
            f.write(f"| Median | | | {median_abs:.4f} |\n")
            f.write(f"| Pearson_Corr | | {pearson:.6f} | |\n")
        print(f"\nMarkdown written to {args.output}")


if __name__ == "__main__":
    main()
