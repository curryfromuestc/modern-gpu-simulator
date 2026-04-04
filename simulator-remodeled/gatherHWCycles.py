#!/usr/bin/env python3

import sys
import os
import yaml
from yaml.loader import SafeLoader

class Execution:
    def __init__(self, benchmarkName, argumentsExecution, kernelSimulatedList):
        self.name = benchmarkName
        self.config = argumentsExecution
        self.kernelSimulatedList = kernelSimulatedList

    def set_cycles(self, cyclesPerKernelList):
        self.cyclesPerKernelList = cyclesPerKernelList
        self.totalCycles = sum(cyclesPerKernelList)
         


pathOfAppsYml = "util/job_launching/apps/define-all-apps.yml"
appsDumpedExecutionRoot = "hw_run/RealHW/11.4/"
tracesRoot = "hw_run/Turing/pannotia/11.4/"
# tracesRoot = "hw_run/Turing/GPUMicrobenchmarks/11.4/"
# tracesRoot = "hw_run/Turing/parboil/11.4/"
# tracesRoot = "hw_run/Turing/ispass/11.4/"
# tracesRoot = "hw_run/Turing/rodinia3_pred_com/11.4/"
# tracesRoot = "hw_run/Turing/deepbench_pred_com/11.4/"

# function that opens a .yml and parses its content
def parse_yml_file(yml_file):
    with open(yml_file, 'r') as f:
        yml_data = yaml.load(f, Loader=SafeLoader)
        return yml_data
    
# function to read file from a path and only keeps lines with the word kernel- in them
def read_kernelslist_file(kernelslistPath):
    kernelslist = []
    with open(kernelslistPath, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if "kernel-" in line:
                kernelslist.append(int(line.strip().replace("kernel-", "").replace(".traceg","").replace(".gz","")))
    return kernelslist

def main():
    global pathOfAppsYml
    allExecutions = []
    if len(sys.argv) != 3:
        print("Error, the program should be executed as: python3 gatherHWCycles.py name-of-benchmark-suit outfileName")
        sys.exit(1)
    suite = sys.argv[1]
    benchmarksToGather = parse_yml_file(pathOfAppsYml)[suite]['execs']
    for benchmark in benchmarksToGather:
        for name in benchmark:
            for config in benchmark[name]:
                args = str(config['args']).split(" ")
                joinedArgs = "_".join(args).replace("/", "_").replace(".", "_").replace("-", "_").replace("=", "_").replace(">", "_").replace("<", "_").replace(",","_")
                if joinedArgs == "None":
                    joinedArgs = "NO_ARGS"
                kernelslistPath = tracesRoot + name + "/" + joinedArgs + "/traces/kernelslist.g"
                kernelsLines = read_kernelslist_file(kernelslistPath)
                allExecutions.append(Execution(name, joinedArgs, kernelsLines))
    for execution in allExecutions:
        cyclesPerKernelList = []
        hwrunPath = appsDumpedExecutionRoot + execution.name + "/" + execution.config
        for file in os.listdir(hwrunPath):
            if file.endswith(".csv.gpc__cycles_elapsed.0"):
                cyclesFile = os.path.join(hwrunPath, file)
                with open(cyclesFile, 'r') as f:
                    lines = f.readlines()
                    startedKernelProfiling = False
                    readThisLine = False
                    for line in lines:
                        if readThisLine:
                            splitedLine1 = line.split(",")
                            splitedLine2 = line.split('"cycle",')
                            kernelID = int(splitedLine1[0].replace('"',"")) + 1
                            cycles = float(splitedLine2[-1].strip().replace('"',"").replace(",",""))
                            if kernelID in execution.kernelSimulatedList:
                                cyclesPerKernelList.append(cycles)
                        if startedKernelProfiling:
                            readThisLine = True
                        if "==PROF== Disconnected from process" in line:
                            startedKernelProfiling = True
        execution.set_cycles(cyclesPerKernelList)
    
    # Dump total cycles per kernel in a csv file
    outputName = sys.argv[2]
    with open(outputName, "w") as f:
        for execution in allExecutions:
            f.write(execution.name + "," + execution.config + "," + str(execution.totalCycles) + "\n")



main()