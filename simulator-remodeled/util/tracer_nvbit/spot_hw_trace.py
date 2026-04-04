#!/usr/bin/env python3
# Script to run benchmarks using the NVBit instr_count tool and capture output.

from optparse import OptionParser
import os
import subprocess
import sys
this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"
sys.path.insert(0,os.path.join(this_directory,"..","job_launching"))
import common
import re
import shutil
import glob
import datetime
import yaml

# Determine the root directory for spot runs, using TRACES_ROOT_DIR if set.
traces_root_dir = os.getenv('TRACES_ROOT_DIR')
if not traces_root_dir:
    # Default if TRACES_ROOT_DIR is not set.
    # Go up two levels from the script's directory.
    traces_root_dir = os.path.abspath(os.path.join(this_directory, "..", ".."))

# Path to the instr_count tool
instr_count_so_path = os.path.join(this_directory, "nvbit_release", "tools", "instr_count", "instr_count.so")
if not os.path.exists(instr_count_so_path):
    print(f"Error: instr_count tool not found at {instr_count_so_path}")
    print("Please build the NVBit tools in util/tracer_nvbit/nvbit_release")
    sys.exit(1)

# We will look for the benchmarks
parser = OptionParser()
parser.add_option("-B", "--benchmark_list", dest="benchmark_list",
                 help="a comma seperated list of benchmark suites to run. See apps/define-*.yml for " +\
                       "the benchmark suite names.",
                 default="rodinia_2.0-ft")
parser.add_option("-D", "--device_num", dest="device_num",
                 help="CUDA device number",
                 default="0")
parser.add_option("-n", "--norun", dest="norun", action="store_true",
                 help="Do not actually run the apps, just create the dir structure and launch files")
parser.add_option("-l", "--limit_kernel_number", dest='kernel_number', type='int', default=-99,
                 help="Sets a hard limit to the number of kernels instrumented by instr_count.")
parser.add_option("-t", "--terminate_upon_limit", dest='terminate_upon_limit', action="store_true",
                 help="Once the kernel limit is reached, terminate the application process.")

(options, args) = parser.parse_args()

common.load_defined_yamls()

benchmarks = []
benchmarks = common.gen_apps_from_suite_list(options.benchmark_list.split(","))

cuda_version = common.get_cuda_version( this_directory )
now_time = datetime.datetime.now()
day_string = now_time.strftime("%y.%m.%d-%A")
time_string = now_time.strftime("%H:%M:%S")
logfile = day_string + "--" + time_string + ".csv" # Note: logfile is defined but not used

print(f"Using TRACES_ROOT_DIR: {traces_root_dir}")
print(f"Using instr_count tool: {instr_count_so_path}")

for bench in benchmarks:
    edir, ddir, exe, argslist = bench
    for argpair in argslist:
        args_str = argpair["args"]
        run_name = os.path.join( exe, common.get_argfoldername( args_str ) )

        # Construct the path including "spot" and using "instr_count_data"
        this_run_base = os.path.abspath(os.path.expandvars(
            os.path.join(traces_root_dir, "spot", "hw_run", "instr_count_data")))

        this_run_dir = os.path.join(this_run_base, "device-" + options.device_num, cuda_version, run_name)

        if not os.path.exists(this_run_dir):
            os.makedirs(this_run_dir)
        # Removed creation of this_trace_folder as instr_count uses stdout/stderr

        # link the data directory
        try:
            benchmark_data_dir = common.dir_option_test(os.path.join(ddir,exe,"data"),"",this_directory)
            data_link_path = os.path.join(this_run_dir, "data")
            if os.path.lexists(data_link_path):
                os.remove(data_link_path)
            os.symlink(benchmark_data_dir, data_link_path, target_is_directory=True)
            print(f"Linked data directory: {benchmark_data_dir} -> {data_link_path}")
        except common.PathMissing:
            print(f"No specific data directory found for {exe} at {os.path.join(ddir,exe,'data')}")
        except Exception as e:
            print(f"Error linking data directory for {exe}: {e}")

        # Link the top-level data directory
        try:
            all_data_link = os.path.join(this_run_dir,"data_dirs")
            if os.path.lexists(all_data_link):
                os.remove(all_data_link)
            top_data_dir_path = common.dir_option_test(ddir, "", this_directory)
            os.symlink(top_data_dir_path, all_data_link, target_is_directory=True)
            print(f"Linked top-level data directory: {top_data_dir_path} -> {all_data_link}")
        except common.PathMissing:
             print(f"Top-level data directory not found at {ddir}")
        except Exception as e:
            print(f"Error linking top-level data directory for {exe}: {e}")


        if args_str is None:
            args_str = ""

        try:
            exec_path = common.file_option_test(os.path.join(edir, exe),"",this_directory)
        except common.PathMissing as e:
            print(f"Executable not found for {exe}: {e}. Skipping benchmark.")
            continue # Skip this benchmark if executable is missing

        sh_contents = "#!/bin/bash\n"
        sh_contents += "set -e\n" # Exit immediately if a command exits with a non-zero status.

        # Set environment variables for NVBit tool
        sh_contents += f"export CUDA_VISIBLE_DEVICES=\"{options.device_num}\"\n"
        # The instr_count tool might read these env vars (optional, depends on tool implementation)
        if options.terminate_upon_limit:
            sh_contents += "export TERMINATE_UPON_LIMIT=1\n"
        if options.kernel_number > 0:
             sh_contents += f"export DYNAMIC_KERNEL_LIMIT_END={options.kernel_number}\n"
        elif 'mlperf' in exec_path:
             sh_contents += "export DYNAMIC_KERNEL_LIMIT_END=50\n" # Default for mlperf if no limit set
             if not exec_path.startswith('.'): # Ensure mlperf scripts are sourced if needed
                exec_path = '. ' + exec_path
        else:
             sh_contents += "export DYNAMIC_KERNEL_LIMIT_END=0\n" # Default for others (no limit)

        # Command to run the application with NVBit injection
        # Redirect stdout and stderr to files
        sh_contents += f"export CUDA_INJECTION64_PATH=\"{instr_count_so_path}\"\n"
        sh_contents += f"export LD_PRELOAD=\"{instr_count_so_path}\"\n"
        sh_contents += f"{exec_path} {args_str} > stdout.log 2> stderr.log\n"
        # Unset LD_PRELOAD and CUDA_INJECTION64_PATH after execution (good practice)
        sh_contents += "unset LD_PRELOAD\n"
        sh_contents += "unset CUDA_INJECTION64_PATH\n"


        run_script_path = os.path.join(this_run_dir,"run.sh")
        try:
            with open(run_script_path, "w") as f:
                f.write(sh_contents)
            subprocess.check_call(['chmod', 'u+x', run_script_path])
        except Exception as e:
            print(f"Error creating or setting permissions for run script {run_script_path}: {e}")
            print("Skipping benchmark.")
            continue # Skip if script creation fails

        if not options.norun:
            saved_dir = os.getcwd()
            print(f"\n=== Running {run_name} in {this_run_dir} ===")
            os.chdir(this_run_dir)

            print(f"Executing: bash {run_script_path}")
            try:
                # Use subprocess.run for better control and error checking
                completed_process = subprocess.run(["bash", "run.sh"], check=True, capture_output=True, text=True)
                # Output is already redirected to files by run.sh, but we can log success
                print(f"Finished running {exe}. Check stdout.log and stderr.log in {this_run_dir}")
                # print("Captured stdout:\n", completed_process.stdout) # Optional: print captured output
                # print("Captured stderr:\n", completed_process.stderr) # Optional: print captured output
            except subprocess.CalledProcessError as e:
                # Error handling: print message and continue to next benchmark
                print(f"!!! Error invoking NVBit instr_count on {run_name} in {this_run_dir} !!!")
                print(f"Return code: {e.returncode}")
                # Output from the failed command is already in stdout.log/stderr.log
                print(f"Check stdout.log and stderr.log in {this_run_dir} for details.")
                # Continue to the next iteration instead of exiting
            except Exception as e:
                print(f"An unexpected error occurred while running {run_name}: {e}")
            finally:
                # Always change back to the original directory
                os.chdir(saved_dir)
        else:
            print(f"Skipping execution for {run_name} (--norun specified). Script generated at {run_script_path}")

print("\n=== Script finished ===") 