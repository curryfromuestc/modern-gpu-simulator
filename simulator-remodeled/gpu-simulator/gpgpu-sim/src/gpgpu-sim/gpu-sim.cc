// Copyright (c) 2023-2025, Rodrigo Huerta, Mojtaba Abaie Shoushtary, Josep-Llorenç Cruz, Antonio González
// Universitat Politecnica de Catalunya
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution. Neither the name of
// The Universitat Politecnica de Catalunya nor the names of its contributors may be
// used to endorse or promote products derived from this software without
// specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

// Copyright (c) 2009-2021, Tor M. Aamodt, Wilson W.L. Fung, George L. Yuan,
// Ali Bakhoda, Andrew Turner, Ivan Sham, Vijay Kandiah, Nikos Hardavellas, 
// Mahmoud Khairy, Junrui Pan, Timothy G. Rogers
// The University of British Columbia, Northwestern University, Purdue University
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer;
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution;
// 3. Neither the names of The University of British Columbia, Northwestern 
//    University nor the names of their contributors may be used to
//    endorse or promote products derived from this software without specific
//    prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "gpu-sim.h"

#include <math.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include "zlib.h"

#include "dram.h"
#include "mem_fetch.h"
#include "shader.h"
#include "shader_trace.h"

#include <time.h>
#include "addrdec.h"
#include "delayqueue.h"
#include "dram.h"
#include "gpu-cache.h"
#include "gpu-misc.h"
#include "icnt_wrapper.h"
#include "l2cache.h"
#include "shader.h"
#include "stat-tool.h"

#include "../../libcuda/gpgpu_context.h"
#include "../abstract_hardware_model.h"
#include "../cuda-sim/cuda-sim.h"
#include "../cuda-sim/cuda_device_runtime.h"
#include "../cuda-sim/ptx-stats.h"
#include "../cuda-sim/ptx_ir.h"
#include "../debug.h"
#include "../gpgpusim_entrypoint.h"
#include "../statwrapper.h"
#include "../trace.h"
#include "mem_latency_stat.h"
#include "power_stat.h"
#include "stats.h"
#include "visualizer.h"

#include "../constants.h" // MOD. Added to do not duplicate some constant declarations in many files

#ifdef GPGPUSIM_POWER_MODEL
#include "power_interface.h"
#else
class gpgpu_sim_wrapper {};
#endif

#include <stdio.h>
#include <string.h>
#include <iostream>
#include <sstream>
#include <string>

bool g_interactive_debugger_enabled = false;

tr1_hash_map<new_addr_type, unsigned> address_random_interleaving;

/* Clock Domains */

#define CORE 0x01
#define L2 0x02
#define DRAM 0x04
#define ICNT 0x08

#define MEM_LATENCY_STAT_IMPL

#include "mem_latency_stat.h"

#include "remodeling/gmmu.h"

// MOD. Begin. Improved tracer
void gpgpu_sim::parse_extra_trace_info(std::string filepath, bool is_extra_trace_enabled) {
  if(is_extra_trace_enabled) {
    m_extra_trace_info.DeserializeFromFile(filepath.c_str());
  }
}
// MOD. End. Improved tracer

void power_config::reg_options(class OptionParser *opp) {
  option_parser_register(opp, "-accelwattch_xml_file", OPT_CSTR,
                         &g_power_config_name, "AccelWattch XML file",
                         "accelwattch_sass_sim.xml");

  option_parser_register(opp, "-power_simulation_enabled", OPT_BOOL,
                         &g_power_simulation_enabled,
                         "Turn on power simulator (1=On, 0=Off)", "0");

  option_parser_register(opp, "-power_per_cycle_dump", OPT_BOOL,
                         &g_power_per_cycle_dump,
                         "Dump detailed power output each cycle", "0");




  option_parser_register(opp, "-hw_perf_file_name", OPT_CSTR,
                         &g_hw_perf_file_name, "Hardware Performance Statistics file",
                         "hw_perf.csv");

  option_parser_register(opp, "-hw_perf_bench_name", OPT_CSTR,
                         &g_hw_perf_bench_name, "Kernel Name in Hardware Performance Statistics file",
                         "");

  option_parser_register(opp, "-power_simulation_mode", OPT_INT32,
                         &g_power_simulation_mode,
                         "Switch performance counter input for power simulation (0=Sim, 1=HW, 2=HW-Sim Hybrid)", "0");

  option_parser_register(opp, "-dvfs_enabled", OPT_BOOL,
                         &g_dvfs_enabled,
                         "Turn on DVFS for power model", "0");
  option_parser_register(opp, "-aggregate_power_stats", OPT_BOOL,
                         &g_aggregate_power_stats,
                         "Accumulate power across all kernels", "0");

  //Accelwattch Hyrbid Configuration

  option_parser_register(opp, "-accelwattch_hybrid_perfsim_L1_RH", OPT_BOOL,
                         &accelwattch_hybrid_configuration[HW_L1_RH],
                         "Get L1 Read Hits for Accelwattch-Hybrid from Accel-Sim", "0");
  option_parser_register(opp, "-accelwattch_hybrid_perfsim_L1_RM", OPT_BOOL,
                         &accelwattch_hybrid_configuration[HW_L1_RM],
                         "Get L1 Read Misses for Accelwattch-Hybrid from Accel-Sim", "0");
  option_parser_register(opp, "-accelwattch_hybrid_perfsim_L1_WH", OPT_BOOL,
                         &accelwattch_hybrid_configuration[HW_L1_WH],
                         "Get L1 Write Hits for Accelwattch-Hybrid from Accel-Sim", "0");
  option_parser_register(opp, "-accelwattch_hybrid_perfsim_L1_WM", OPT_BOOL,
                         &accelwattch_hybrid_configuration[HW_L1_WM],
                         "Get L1 Write Misses for Accelwattch-Hybrid from Accel-Sim", "0");

  option_parser_register(opp, "-accelwattch_hybrid_perfsim_L2_RH", OPT_BOOL,
                         &accelwattch_hybrid_configuration[HW_L2_RH],
                         "Get L2 Read Hits for Accelwattch-Hybrid from Accel-Sim", "0");
  option_parser_register(opp, "-accelwattch_hybrid_perfsim_L2_RM", OPT_BOOL,
                         &accelwattch_hybrid_configuration[HW_L2_RM],
                         "Get L2 Read Misses for Accelwattch-Hybrid from Accel-Sim", "0");
  option_parser_register(opp, "-accelwattch_hybrid_perfsim_L2_WH", OPT_BOOL,
                         &accelwattch_hybrid_configuration[HW_L2_WH],
                         "Get L2 Write Hits for Accelwattch-Hybrid from Accel-Sim", "0");
  option_parser_register(opp, "-accelwattch_hybrid_perfsim_L2_WM", OPT_BOOL,
                         &accelwattch_hybrid_configuration[HW_L2_WM],
                         "Get L2 Write Misses for Accelwattch-Hybrid from Accel-Sim", "0");

  option_parser_register(opp, "-accelwattch_hybrid_perfsim_CC_ACC", OPT_BOOL,
                         &accelwattch_hybrid_configuration[HW_CC_ACC],
                         "Get Constant Cache Acesses for Accelwattch-Hybrid from Accel-Sim", "0");

  option_parser_register(opp, "-accelwattch_hybrid_perfsim_SHARED_ACC", OPT_BOOL,
                         &accelwattch_hybrid_configuration[HW_SHRD_ACC],
                         "Get Shared Memory Acesses for Accelwattch-Hybrid from Accel-Sim", "0");

  option_parser_register(opp, "-accelwattch_hybrid_perfsim_DRAM_RD", OPT_BOOL,
                         &accelwattch_hybrid_configuration[HW_DRAM_RD],
                         "Get DRAM Reads for Accelwattch-Hybrid from Accel-Sim", "0");
  option_parser_register(opp, "-accelwattch_hybrid_perfsim_DRAM_WR", OPT_BOOL,
                         &accelwattch_hybrid_configuration[HW_DRAM_WR],
                         "Get DRAM Writes for Accelwattch-Hybrid from Accel-Sim", "0");

  option_parser_register(opp, "-accelwattch_hybrid_perfsim_NOC", OPT_BOOL,
                         &accelwattch_hybrid_configuration[HW_NOC],
                         "Get Interconnect Acesses for Accelwattch-Hybrid from Accel-Sim", "0");

  option_parser_register(opp, "-accelwattch_hybrid_perfsim_PIPE_DUTY", OPT_BOOL,
                         &accelwattch_hybrid_configuration[HW_PIPE_DUTY],
                         "Get Pipeline Duty Cycle Acesses for Accelwattch-Hybrid from Accel-Sim", "0");

  option_parser_register(opp, "-accelwattch_hybrid_perfsim_NUM_SM_IDLE", OPT_BOOL,
                         &accelwattch_hybrid_configuration[HW_NUM_SM_IDLE],
                         "Get Number of Idle SMs for Accelwattch-Hybrid from Accel-Sim", "0");

  option_parser_register(opp, "-accelwattch_hybrid_perfsim_CYCLES", OPT_BOOL,
                         &accelwattch_hybrid_configuration[HW_CYCLES],
                         "Get Executed Cycles for Accelwattch-Hybrid from Accel-Sim", "0");

  option_parser_register(opp, "-accelwattch_hybrid_perfsim_VOLTAGE", OPT_BOOL,
                         &accelwattch_hybrid_configuration[HW_VOLTAGE],
                         "Get Chip Voltage for Accelwattch-Hybrid from Accel-Sim", "0");


  // Output Data Formats
  option_parser_register(
      opp, "-power_trace_enabled", OPT_BOOL, &g_power_trace_enabled,
      "produce a file for the power trace (1=On, 0=Off)", "0");

  option_parser_register(
      opp, "-power_trace_zlevel", OPT_INT32, &g_power_trace_zlevel,
      "Compression level of the power trace output log (0=no comp, 9=highest)",
      "6");

  option_parser_register(
      opp, "-steady_power_levels_enabled", OPT_BOOL,
      &g_steady_power_levels_enabled,
      "produce a file for the steady power levels (1=On, 0=Off)", "0");

  option_parser_register(opp, "-steady_state_definition", OPT_CSTR,
                         &gpu_steady_state_definition,
                         "allowed deviation:number of samples", "8:4");
}

void memory_config::reg_options(class OptionParser *opp) {
  option_parser_register(opp, "-gpgpu_perf_sim_memcpy", OPT_BOOL,
                         &m_perf_sim_memcpy, "Fill the L2 cache on memcpy",
                         "1");
  option_parser_register(opp, "-gpgpu_simple_dram_model", OPT_BOOL,
                         &simple_dram_model,
                         "simple_dram_model with fixed latency and BW", "0");
  option_parser_register(opp, "-gpgpu_dram_scheduler", OPT_INT32,
                         &scheduler_type, "0 = fifo, 1 = FR-FCFS (defaul)",
                         "1");
  option_parser_register(opp, "-gpgpu_dram_partition_queues", OPT_CSTR,
                         &gpgpu_L2_queue_config, "i2$:$2d:d2$:$2i", "8:8:8:8");

  option_parser_register(opp, "-l2_ideal", OPT_BOOL, &l2_ideal,
                         "Use a ideal L2 cache that always hit", "0");
  option_parser_register(opp, "-gpgpu_cache:dl2", OPT_CSTR,
                         &m_L2_config.m_config_string,
                         "unified banked L2 data cache config "
                         " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_"
                         "alloc>,<mshr>:<N>:<merge>,<mq>}",
                         "64:128:8,L:B:m:N,A:16:4,4");
  option_parser_register(opp, "-gpgpu_cache:dl2_texture_only", OPT_BOOL,
                         &m_L2_texure_only, "L2 cache used for texture only",
                         "1");
  option_parser_register(
      opp, "-gpgpu_n_mem", OPT_UINT32, &m_n_mem,
      "number of memory modules (e.g. memory controllers) in gpu", "8");
  option_parser_register(opp, "-gpgpu_n_sub_partition_per_mchannel", OPT_UINT32,
                         &m_n_sub_partition_per_memory_channel,
                         "number of memory subpartition in each memory module",
                         "1");
  option_parser_register(opp, "-gpgpu_n_mem_per_ctrlr", OPT_UINT32,
                         &gpu_n_mem_per_ctrlr,
                         "number of memory chips per memory controller", "1");
  option_parser_register(opp, "-gpgpu_memlatency_stat", OPT_INT32,
                         &gpgpu_memlatency_stat,
                         "track and display latency statistics 0x2 enables MC, "
                         "0x4 enables queue logs",
                         "0");
  option_parser_register(opp, "-gpgpu_frfcfs_dram_sched_queue_size", OPT_INT32,
                         &gpgpu_frfcfs_dram_sched_queue_size,
                         "0 = unlimited (default); # entries per chip", "0");
  option_parser_register(opp, "-gpgpu_dram_return_queue_size", OPT_INT32,
                         &gpgpu_dram_return_queue_size,
                         "0 = unlimited (default); # entries per chip", "0");
  option_parser_register(opp, "-gpgpu_dram_buswidth", OPT_UINT32, &busW,
                         "default = 4 bytes (8 bytes per cycle at DDR)", "4");
  option_parser_register(
      opp, "-gpgpu_dram_burst_length", OPT_UINT32, &BL,
      "Burst length of each DRAM request (default = 4 data bus cycle)", "4");
  option_parser_register(opp, "-dram_data_command_freq_ratio", OPT_UINT32,
                         &data_command_freq_ratio,
                         "Frequency ratio between DRAM data bus and command "
                         "bus (default = 2 times, i.e. DDR)",
                         "2");
  option_parser_register(
      opp, "-gpgpu_dram_timing_opt", OPT_CSTR, &gpgpu_dram_timing_opt,
      "DRAM timing parameters = "
      "{nbk:tCCD:tRRD:tRCD:tRAS:tRP:tRC:CL:WL:tCDLR:tWR:nbkgrp:tCCDL:tRTPL}",
      "4:2:8:12:21:13:34:9:4:5:13:1:0:0");
  option_parser_register(opp, "-gpgpu_l2_rop_latency", OPT_UINT32, &rop_latency,
                         "ROP queue latency (default 85)", "85");
  option_parser_register(opp, "-dram_latency", OPT_UINT32, &dram_latency,
                         "DRAM latency (default 30)", "30");
  option_parser_register(opp, "-dram_dual_bus_interface", OPT_UINT32,
                         &dual_bus_interface,
                         "dual_bus_interface (default = 0) ", "0");
  option_parser_register(opp, "-dram_bnk_indexing_policy", OPT_UINT32,
                         &dram_bnk_indexing_policy,
                         "dram_bnk_indexing_policy (0 = normal indexing, 1 = "
                         "Xoring with the higher bits) (Default = 0)",
                         "0");
  option_parser_register(opp, "-dram_bnkgrp_indexing_policy", OPT_UINT32,
                         &dram_bnkgrp_indexing_policy,
                         "dram_bnkgrp_indexing_policy (0 = take higher bits, 1 "
                         "= take lower bits) (Default = 0)",
                         "0");
  option_parser_register(opp, "-dram_seperate_write_queue_enable", OPT_BOOL,
                         &seperate_write_queue_enabled,
                         "Seperate_Write_Queue_Enable", "0");
  option_parser_register(opp, "-dram_write_queue_size", OPT_CSTR,
                         &write_queue_size_opt, "Write_Queue_Size", "32:28:16");
  option_parser_register(
      opp, "-dram_elimnate_rw_turnaround", OPT_BOOL, &elimnate_rw_turnaround,
      "elimnate_rw_turnaround i.e set tWTR and tRTW = 0", "0");
  option_parser_register(opp, "-icnt_flit_size", OPT_UINT32, &icnt_flit_size,
                         "icnt_flit_size", "32");
  m_address_mapping.addrdec_setoption(opp);
}

void shader_core_config::reg_options(class OptionParser *opp) {
  option_parser_register(opp, "-gpgpu_simd_model", OPT_INT32, &model,
                         "1 = post-dominator", "1");
  option_parser_register(
      opp, "-gpgpu_shader_core_pipeline", OPT_CSTR,
      &gpgpu_shader_core_pipeline_opt,
      "shader core pipeline config, i.e., {<nthread>:<warpsize>}", "1024:32");
  option_parser_register(opp, "-gpgpu_tex_cache:l1", OPT_CSTR,
                         &m_L1T_config.m_config_string,
                         "per-shader L1 texture cache  (READ-ONLY) config "
                         " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_"
                         "alloc>,<mshr>:<N>:<merge>,<mq>:<rf>}",
                         "8:128:5,L:R:m:N,F:128:4,128:2");
  option_parser_register(
      opp, "-gpgpu_const_cache:l1", OPT_CSTR, &m_L1C_config.m_config_string,
      "per-shader L1 constant memory cache  (READ-ONLY) config "
      " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_alloc>,<mshr>:<N>:<"
      "merge>,<mq>} ",
      "64:64:2,L:R:f:N,A:2:32,4");
  option_parser_register(opp, "-gpgpu_cache:il1", OPT_CSTR,
                         &m_L1I_L1_half_C_cache_config.m_config_string,
                         "shader L1 instruction cache config "
                         " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_"
                         "alloc>,<mshr>:<N>:<merge>,<mq>} ",
                         "4:256:4,L:R:f:N,A:2:32,4");
  // MOD. Begin. Added L0I
  option_parser_register(opp, "-gpgpu_cache:il0", OPT_CSTR,
                         &m_L0I_config.m_config_string,
                         "shader L0 instruction cache config "
                         " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_"
                         "alloc>,<mshr>:<N>:<merge>,<mq>} ",
                         "N:64:128:16,L:R:f:N:L,S:2:48,4");
  // MOD. Begin. Added L0I
  option_parser_register(opp, "-gpgpu_cache:dl1", OPT_CSTR,
                         &m_L1D_config.m_config_string,
                         "per-shader L1 data cache config "
                         " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_"
                         "alloc>,<mshr>:<N>:<merge>,<mq> | none}",
                         "none");
  option_parser_register(opp, "-gpgpu_l1_cache_write_ratio", OPT_UINT32,
                         &m_L1D_config.m_wr_percent, "L1D write ratio", "0");
  option_parser_register(opp, "-gpgpu_l1_banks", OPT_UINT32,
                         &m_L1D_config.l1_banks, "The number of L1 cache banks",
                         "1");
  option_parser_register(opp, "-gpgpu_l1_banks_byte_interleaving", OPT_UINT32,
                         &m_L1D_config.l1_banks_byte_interleaving,
                         "l1 banks byte interleaving granularity", "32");
  option_parser_register(opp, "-gpgpu_l1_banks_hashing_function", OPT_UINT32,
                         &m_L1D_config.l1_banks_hashing_function,
                         "l1 banks hashing function", "0");
  option_parser_register(opp, "-gpgpu_l1_latency", OPT_UINT32,
                         &m_L1D_config.l1_latency, "L1 Hit Latency", "1");
  option_parser_register(opp, "-gpgpu_smem_latency", OPT_UINT32, &smem_latency,
                         "smem Latency", "3");
  option_parser_register(opp, "-gpgpu_cache:dl1PrefL1", OPT_CSTR,
                         &m_L1D_config.m_config_stringPrefL1,
                         "per-shader L1 data cache config "
                         " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_"
                         "alloc>,<mshr>:<N>:<merge>,<mq> | none}",
                         "none");
  option_parser_register(opp, "-gpgpu_cache:dl1PrefShared", OPT_CSTR,
                         &m_L1D_config.m_config_stringPrefShared,
                         "per-shader L1 data cache config "
                         " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_"
                         "alloc>,<mshr>:<N>:<merge>,<mq> | none}",
                         "none");
  option_parser_register(opp, "-gpgpu_gmem_skip_L1D", OPT_BOOL, &gmem_skip_L1D,
                         "global memory access skip L1D cache (implements "
                         "-Xptxas -dlcm=cg, default=no skip)",
                         "0");

  option_parser_register(opp, "-gpgpu_perfect_mem", OPT_BOOL,
                         &gpgpu_perfect_mem,
                         "enable perfect memory mode (no cache miss)", "0");
  option_parser_register(
      opp, "-n_regfile_gating_group", OPT_UINT32, &n_regfile_gating_group,
      "group of lanes that should be read/written together)", "4");
  option_parser_register(
      opp, "-gpgpu_clock_gated_reg_file", OPT_BOOL, &gpgpu_clock_gated_reg_file,
      "enable clock gated reg file for power calculations", "0");
  option_parser_register(
      opp, "-gpgpu_clock_gated_lanes", OPT_BOOL, &gpgpu_clock_gated_lanes,
      "enable clock gated lanes for power calculations", "0");
  option_parser_register(opp, "-gpgpu_shader_registers", OPT_UINT32,
                         &gpgpu_shader_registers,
                         "Number of registers per shader core. Limits number "
                         "of concurrent CTAs. (default 8192)",
                         "8192");
  option_parser_register(
      opp, "-gpgpu_registers_per_block", OPT_UINT32, &gpgpu_registers_per_block,
      "Maximum number of registers per CTA. (default 8192)", "8192");
  option_parser_register(opp, "-gpgpu_ignore_resources_limitation", OPT_BOOL,
                         &gpgpu_ignore_resources_limitation,
                         "gpgpu_ignore_resources_limitation (default 0)", "0");
  option_parser_register(
      opp, "-gpgpu_shader_cta", OPT_UINT32, &max_cta_per_core,
      "Maximum number of concurrent CTAs in shader (default 32)", "32");
  option_parser_register(
      opp, "-gpgpu_num_cta_barriers", OPT_UINT32, &max_barriers_per_cta,
      "Maximum number of named barriers per CTA (default 16)", "16");
  option_parser_register(opp, "-gpgpu_n_clusters", OPT_UINT32, &n_simt_clusters,
                         "number of processing clusters", "10");
  option_parser_register(opp, "-gpgpu_n_cores_per_cluster", OPT_UINT32,
                         &n_simt_cores_per_cluster,
                         "number of simd cores per cluster", "3");
  option_parser_register(opp, "-gpgpu_n_cluster_ejection_buffer_size",
                         OPT_UINT32, &n_simt_ejection_buffer_size,
                         "number of packets in ejection buffer", "8");
  option_parser_register(
      opp, "-gpgpu_n_ldst_response_buffer_size", OPT_UINT32,
      &ldst_unit_response_queue_size,
      "number of response packets in ld/st unit ejection buffer", "2");
  option_parser_register(
      opp, "-gpgpu_shmem_per_block", OPT_UINT32, &gpgpu_shmem_per_block,
      "Size of shared memory per thread block or CTA (default 48kB)", "49152");
  option_parser_register(
      opp, "-gpgpu_shmem_size", OPT_UINT32, &gpgpu_shmem_size,
      "Size of shared memory per shader core (default 16kB)", "16384");
  option_parser_register(opp, "-gpgpu_shmem_option", OPT_CSTR,
                         &gpgpu_shmem_option,
                         "Option list of shared memory sizes", "0");
  option_parser_register(
      opp, "-gpgpu_unified_l1d_size", OPT_UINT32,
      &m_L1D_config.m_unified_cache_size,
      "Size of unified data cache(L1D + shared memory) in KB", "0");
  option_parser_register(opp, "-gpgpu_adaptive_cache_config", OPT_BOOL,
                         &adaptive_cache_config, "adaptive_cache_config", "0");
  option_parser_register(
      opp, "-gpgpu_shmem_sizeDefault", OPT_UINT32, &gpgpu_shmem_sizeDefault,
      "Size of shared memory per shader core (default 16kB)", "16384");
  option_parser_register(
      opp, "-gpgpu_shmem_size_PrefL1", OPT_UINT32, &gpgpu_shmem_sizePrefL1,
      "Size of shared memory per shader core (default 16kB)", "16384");
  option_parser_register(opp, "-gpgpu_shmem_size_PrefShared", OPT_UINT32,
                         &gpgpu_shmem_sizePrefShared,
                         "Size of shared memory per shader core (default 16kB)",
                         "16384");
  option_parser_register(
      opp, "-gpgpu_shmem_num_banks", OPT_UINT32, &num_shmem_bank,
      "Number of banks in the shared memory in each shader core (default 16)",
      "16");
  option_parser_register(
      opp, "-gpgpu_shmem_limited_broadcast", OPT_BOOL, &shmem_limited_broadcast,
      "Limit shared memory to do one broadcast per cycle (default on)", "1");
  option_parser_register(opp, "-gpgpu_shmem_warp_parts", OPT_INT32,
                         &mem_warp_parts,
                         "Number of portions a warp is divided into for shared "
                         "memory bank conflict check ",
                         "2");
  option_parser_register(
      opp, "-gpgpu_mem_unit_ports", OPT_INT32, &mem_unit_ports,
      "The number of memory transactions allowed per core cycle", "1");
  option_parser_register(opp, "-gpgpu_shmem_warp_parts", OPT_INT32,
                         &mem_warp_parts,
                         "Number of portions a warp is divided into for shared "
                         "memory bank conflict check ",
                         "2");
  option_parser_register(
      opp, "-gpgpu_warpdistro_shader", OPT_INT32, &gpgpu_warpdistro_shader,
      "Specify which shader core to collect the warp size distribution from",
      "-1");
  option_parser_register(
      opp, "-gpgpu_warp_issue_shader", OPT_INT32, &gpgpu_warp_issue_shader,
      "Specify which shader core to collect the warp issue distribution from",
      "0");
  option_parser_register(opp, "-gpgpu_local_mem_map", OPT_BOOL,
                         &gpgpu_local_mem_map,
                         "Mapping from local memory space address to simulated "
                         "GPU physical address space (default = enabled)",
                         "1");
  option_parser_register(opp, "-gpgpu_num_reg_banks", OPT_INT32,
                         &gpgpu_num_reg_banks,
                         "Number of register banks (default = 8)", "8");
  option_parser_register(
      opp, "-gpgpu_reg_bank_use_warp_id", OPT_BOOL, &gpgpu_reg_bank_use_warp_id,
      "Use warp ID in mapping registers to banks (default = off)", "0");
  option_parser_register(opp, "-gpgpu_sub_core_model", OPT_BOOL,
                         &sub_core_model,
                         "Sub Core Volta/Pascal model (default = off)", "0");
  option_parser_register(opp, "-gpgpu_enable_specialized_operand_collector",
                         OPT_BOOL, &enable_specialized_operand_collector,
                         "enable_specialized_operand_collector", "1");
  option_parser_register(opp, "-gpgpu_operand_collector_num_units_sp",
                         OPT_INT32, &gpgpu_operand_collector_num_units_sp,
                         "number of collector units (default = 4)", "4");
  option_parser_register(opp, "-gpgpu_operand_collector_num_units_dp",
                         OPT_INT32, &gpgpu_operand_collector_num_units_dp,
                         "number of collector units (default = 0)", "0");
  option_parser_register(opp, "-gpgpu_operand_collector_num_units_sfu",
                         OPT_INT32, &gpgpu_operand_collector_num_units_sfu,
                         "number of collector units (default = 4)", "4");
  option_parser_register(opp, "-gpgpu_operand_collector_num_units_int",
                         OPT_INT32, &gpgpu_operand_collector_num_units_int,
                         "number of collector units (default = 0)", "0");
  option_parser_register(opp, "-gpgpu_operand_collector_num_units_tensor_core",
                         OPT_INT32,
                         &gpgpu_operand_collector_num_units_tensor_core,
                         "number of collector units (default = 4)", "4");
  option_parser_register(opp, "-gpgpu_operand_collector_num_units_mem",
                         OPT_INT32, &gpgpu_operand_collector_num_units_mem,
                         "number of collector units (default = 2)", "2");
  option_parser_register(opp, "-gpgpu_operand_collector_num_units_gen",
                         OPT_UINT32, &gpgpu_operand_collector_num_units_gen,
                         "number of collector units (default = 0)", "0");
  option_parser_register(opp, "-gpgpu_operand_collector_num_in_ports_sp",
                         OPT_INT32, &gpgpu_operand_collector_num_in_ports_sp,
                         "number of collector unit in ports (default = 1)",
                         "1");
  option_parser_register(opp, "-gpgpu_operand_collector_num_in_ports_dp",
                         OPT_INT32, &gpgpu_operand_collector_num_in_ports_dp,
                         "number of collector unit in ports (default = 0)",
                         "0");
  option_parser_register(opp, "-gpgpu_operand_collector_num_in_ports_sfu",
                         OPT_INT32, &gpgpu_operand_collector_num_in_ports_sfu,
                         "number of collector unit in ports (default = 1)",
                         "1");
  option_parser_register(opp, "-gpgpu_operand_collector_num_in_ports_int",
                         OPT_INT32, &gpgpu_operand_collector_num_in_ports_int,
                         "number of collector unit in ports (default = 0)",
                         "0");
  option_parser_register(
      opp, "-gpgpu_operand_collector_num_in_ports_tensor_core", OPT_INT32,
      &gpgpu_operand_collector_num_in_ports_tensor_core,
      "number of collector unit in ports (default = 1)", "1");
  option_parser_register(opp, "-gpgpu_operand_collector_num_in_ports_mem",
                         OPT_INT32, &gpgpu_operand_collector_num_in_ports_mem,
                         "number of collector unit in ports (default = 1)",
                         "1");
  option_parser_register(opp, "-gpgpu_operand_collector_num_in_ports_gen",
                         OPT_INT32, &gpgpu_operand_collector_num_in_ports_gen,
                         "number of collector unit in ports (default = 0)",
                         "0");
  option_parser_register(opp, "-gpgpu_operand_collector_num_out_ports_sp",
                         OPT_INT32, &gpgpu_operand_collector_num_out_ports_sp,
                         "number of collector unit in ports (default = 1)",
                         "1");
  option_parser_register(opp, "-gpgpu_operand_collector_num_out_ports_dp",
                         OPT_INT32, &gpgpu_operand_collector_num_out_ports_dp,
                         "number of collector unit in ports (default = 0)",
                         "0");
  option_parser_register(opp, "-gpgpu_operand_collector_num_out_ports_sfu",
                         OPT_INT32, &gpgpu_operand_collector_num_out_ports_sfu,
                         "number of collector unit in ports (default = 1)",
                         "1");
  option_parser_register(opp, "-gpgpu_operand_collector_num_out_ports_int",
                         OPT_INT32, &gpgpu_operand_collector_num_out_ports_int,
                         "number of collector unit in ports (default = 0)",
                         "0");
  option_parser_register(
      opp, "-gpgpu_operand_collector_num_out_ports_tensor_core", OPT_INT32,
      &gpgpu_operand_collector_num_out_ports_tensor_core,
      "number of collector unit in ports (default = 1)", "1");
  option_parser_register(opp, "-gpgpu_operand_collector_num_out_ports_mem",
                         OPT_INT32, &gpgpu_operand_collector_num_out_ports_mem,
                         "number of collector unit in ports (default = 1)",
                         "1");
  option_parser_register(opp, "-gpgpu_operand_collector_num_out_ports_gen",
                         OPT_INT32, &gpgpu_operand_collector_num_out_ports_gen,
                         "number of collector unit in ports (default = 0)",
                         "0");
  option_parser_register(opp, "-gpgpu_coalesce_arch", OPT_INT32,
                         &gpgpu_coalesce_arch,
                         "Coalescing arch (GT200 = 13, Fermi = 20)", "13");
  option_parser_register(opp, "-gpgpu_num_sched_per_core", OPT_INT32,
                         &gpgpu_num_sched_per_core,
                         "Number of warp schedulers per core", "1");
  option_parser_register(opp, "-gpgpu_max_insn_issue_per_warp", OPT_INT32,
                         &gpgpu_max_insn_issue_per_warp,
                         "Max number of instructions that can be issued per "
                         "warp in one cycle by scheduler (either 1 or 2)",
                         "2");
  option_parser_register(opp, "-gpgpu_dual_issue_diff_exec_units", OPT_BOOL,
                         &gpgpu_dual_issue_diff_exec_units,
                         "should dual issue use two different execution unit "
                         "resources (Default = 1)",
                         "1");
  option_parser_register(opp, "-gpgpu_simt_core_sim_order", OPT_INT32,
                         &simt_core_sim_order,
                         "Select the simulation order of cores in a cluster "
                         "(0=Fix, 1=Round-Robin)",
                         "1");
  option_parser_register(
      opp, "-gpgpu_pipeline_widths", OPT_CSTR, &pipeline_widths_string,
      "Pipeline widths "
      "ID_OC_SP,ID_OC_DP,ID_OC_INT,ID_OC_SFU,ID_OC_MEM,OC_EX_SP,OC_EX_DP,OC_EX_"
      "INT,OC_EX_SFU,OC_EX_MEM,EX_WB,ID_OC_TENSOR_CORE,OC_EX_TENSOR_CORE",
      "1,1,1,1,1,1,1,1,1,1,1,1,1");
  option_parser_register(opp, "-gpgpu_tensor_core_avail", OPT_UINT32,
                         &gpgpu_tensor_core_avail,
                         "Tensor Core Available (default=0)", "0");
  option_parser_register(opp, "-gpgpu_num_sp_units", OPT_UINT32,
                         &gpgpu_num_sp_units, "Number of SP units (default=1)",
                         "1");
  option_parser_register(opp, "-gpgpu_num_dp_units", OPT_UINT32,
                         &gpgpu_num_dp_units, "Number of DP units (default=0)",
                         "0");
  option_parser_register(opp, "-gpgpu_num_int_units", OPT_UINT32,
                         &gpgpu_num_int_units,
                         "Number of INT units (default=0)", "0");
  option_parser_register(opp, "-gpgpu_num_sfu_units", OPT_UINT32,
                         &gpgpu_num_sfu_units, "Number of SF units (default=1)",
                         "1");
  option_parser_register(opp, "-gpgpu_num_tensor_core_units", OPT_UINT32,
                         &gpgpu_num_tensor_core_units,
                         "Number of tensor_core units (default=1)", "0");
  option_parser_register(
      opp, "-gpgpu_num_mem_units", OPT_UINT32, &gpgpu_num_mem_units,
      "Number if ldst units (default=1) WARNING: not hooked up to anything",
      "1");
  option_parser_register(
      opp, "-gpgpu_scheduler", OPT_CSTR, &gpgpu_scheduler_string,
      "Scheduler configuration: < lrr | gto | two_level_active > "
      "If "
      "two_level_active:<num_active_warps>:<inner_prioritization>:<outer_"
      "prioritization>"
      "For complete list of prioritization values see shader.h enum "
      "scheduler_prioritization_type"
      "Default: gto",
      "gto");

  option_parser_register(
      opp, "-gpgpu_concurrent_kernel_sm", OPT_BOOL, &gpgpu_concurrent_kernel_sm,
      "Support concurrent kernels on a SM (default = disabled)", "0");
  option_parser_register(opp, "-gpgpu_perfect_inst_const_cache", OPT_BOOL,
                         &perfect_inst_const_cache,
                         "perfect inst and const cache mode, so all inst and "
                         "const hits in the cache(default = disabled)",
                         "0");
  option_parser_register(
      opp, "-gpgpu_inst_fetch_throughput", OPT_INT32, &inst_fetch_throughput,
      "the number of fetched intruction per warp each cycle", "1");
  option_parser_register(opp, "-gpgpu_reg_file_port_throughput", OPT_INT32,
                         &reg_file_port_throughput,
                         "the number ports of the register file", "1");

  for (unsigned j = 0; j < SPECIALIZED_UNIT_NUM; ++j) {
    std::stringstream ss;
    ss << "-specialized_unit_" << j + 1;
    option_parser_register(opp, ss.str().c_str(), OPT_CSTR,
                           &specialized_unit_string[j],
                           "specialized unit config"
                           " {<enabled>,<num_units>:<latency>:<initiation>,<ID_"
                           "OC_SPEC>:<OC_EX_SPEC>,<NAME>}",
                           "0,4,4,4,4,BRA");
  }

  // MOD. Predication
  option_parser_register(opp, "-is_trace_predication_enabled", OPT_BOOL,
                         &is_trace_predication_enabled,
                         "If enabled, the predication information of the traces is read and treated as regular registers beggining with the identifier 257 (264 in execution for PT which is P7 in traces)."
                         "(default = disabled)",
                         "0");

  // MOD. Fix WAR at baseline. Enable or disable scoreboard war in order to solve war hazards at baseline with different modes
  option_parser_register(
      opp, "-scoreboard_war_mode", OPT_CSTR, &scoreboard_war_mode,
      "Scoreboard war mode: < disabled | wb | opc > "
      "It enables a scoreboard_reads to solve potential WAR issues due to operand collector OoO issue."
      "wb: releases source register when the instruction is at WB stage."
      "opc: releases source register when the instruction leaves the operand collector stage"
      "Default: opc",
      "opc");
  // MOD.  Fix WAR at baseline. Configuration of the Scoreboard_reads
  option_parser_register(opp, "-scoreboard_war_max_uses_per_reg", OPT_UINT32,
                         &scoreboard_war_max_uses_per_reg, "Number of maximum uses per register allowed"
                         " in the scoreboard_reads in order to prevent WAR hazards in the baseline. (default=9999)",
                         "9999");
  
  option_parser_register(opp, "-scoreboard_war_static_power", OPT_DOUBLE,
                         &scoreboard_war_static_power, "Static power consumption of each scoreboard war that has more than one bit."
                         "Configure to any positive number (default=0)",
                         "0");

  option_parser_register(opp, "-scoreboard_war_dynamic_power", OPT_DOUBLE,
                         &scoreboard_war_dynamic_power, "Dynamic power consumption of each scoreboard war that has more than one bitr."
                         "Configure to any positive number (default=0)",
                         "0");

  // MOD. Begin. Fix loads after store
  option_parser_register(opp, "-is_fix_memory_reordering_enabled_baseline", OPT_BOOL,
                         &is_fix_memory_reordering_enabled_baseline,
                         "If enabled, the baseline fixes loads after issuing a store. The warp won't be allowed to issue loads until the store reaches WB."
                         "Fix loads after store (default = disabled)",
                         "0");
  // MOD. End

  // MOD. Begin. Improving fetch and decode
  option_parser_register(opp, "-is_fetch_and_decode_improved", OPT_BOOL,
                         &is_fetch_and_decode_improved,
                         "If enabled, the fetch and decode stages are separeted for each sub-core and gpgpu_inst_fetch_throughput is disabled."
                         "Fix the stages fetch and decode (default = disabled)",
                         "0");
  // MOD. End

  // MOD. Begin. Added L0I
  option_parser_register(opp, "-is_L0I_enabled", OPT_BOOL,
                         &is_L0I_enabled,
                         "If enabled, each sub-core has its own L0I. Only works if is_fetch_and_decode_improved is enabled too."
                         "Adds a L0I to each sub-core (default = disabled)",
                         "0");
  option_parser_register(opp, "-max_request_allowed_to_L1I", OPT_INT32,
                         &max_request_allowed_to_L1I, "Number of maximum request allowed to L1I in a given cycle. Also known as maximum number of ports for requests"
                         "Configure to any positive number (default=1)",
                         "1");
  option_parser_register(opp, "-max_reply_allowed_from_L1I", OPT_INT32,
                         &max_reply_allowed_from_L1I, "Number of maximum reply allowed from L1I in a given cycle. Also known as maximum number of ports for replies"
                         "Configure to any positive number (default=1)",
                         "1"); 
  option_parser_register(opp, "-latency_L0_to_L1", OPT_INT32,
                         &latency_L0_to_L1, "Latency of requests from L0 and L1. L1 if it is instruction cahce, L1.5 constant cache."
                         "Configure to any positive number (default=40)",
                         "40"); 
  option_parser_register(opp, "-latency_L1_to_L0", OPT_INT32,
                         &latency_L1_to_L0, "Latency of replies from L1 to L0. L1 if it is instruction cahce, L1.5 constant cache.."
                         "Configure to any positive number (default=40)",
                         "40"); 
  // MOD. End

  // MOD. Begin. Fix misaligned fetched instructions
  option_parser_register(opp, "-is_fix_instruction_fetch_misalignment", OPT_BOOL,
                         &is_fix_instruction_fetch_misalignment,
                         "If enabled, instructions that are stored in different blocks of the Instruction cache won't be decoded and placed in the Instruction Buffer in the same fetch request. Only works if is_fetch_and_decode_improved is enabled too."
                         "Fix misaligment when fetching instructions (default = disabled)",
                         "0");
  // MOD. End. Fix misaligned fetched instructions

  // MOD. Begin. Instruction addresses of different kernels have a different address request in memory
  option_parser_register(opp, "-is_fix_different_kernels_pc_addresses", OPT_BOOL,
                         &is_fix_different_kernels_pc_addresses,
                         "If enabled, instruction addresses of different kernels to have a different address request in memory. It is mandatory to enable -is_fetch_and_decode_improved too."
                         "Fix instruction pc addresses requests (default = disabled)",
                         "0");
  // MOD. End

  // MOD. Begin. Instruction decodign when the following PC is far from the previous PC
  option_parser_register(opp, "-is_fix_not_decoding_not_contiguos_instructions", OPT_BOOL,
                         &is_fix_not_decoding_not_contiguos_instructions,
                         "If enabled, instructions with a separated PC are prevented from being decoding because it is impossible that have been fetch in the same request. It is mandatory to enable -is_fetch_and_decode_improved too."
                         "Fix instruction decodign when the following PC is far from the previous PC (default = disabled)",
                         "0");
  // MOD. End

  // MOD. Begin. Fixed LDST_Unit model
  option_parser_register(opp, "-is_improved_ldst_unit_enabled", OPT_BOOL,
                         &is_improved_ldst_unit_enabled,
                         "If enabled, the modeling of LDST_Unit is improved. Supporting to dispatch instructions and do different actions from different sub-cores."
                         "Improves the modeling of the LDST_Unit (default = disabled)",
                         "0");
  // MOD. End

  // MOD. Begin. Improved Result bus to take into account conflicts with RF banks
  option_parser_register(opp, "-is_improved_result_bus", OPT_BOOL,
                         &is_improved_result_bus,
                         "If enabled, the modeling of result bus is improved. It takes into account the number of ports in each RF bank to do not schedule instructions that are going to finish in the same cycle for more than the ports that are in that RF bank."
                         "Improves the modeling of the result bus (default = disabled)",
                         "0");
  // MOD. End

  // MOD. Begin. Improving OPC
  option_parser_register(opp, "-is_opc_improved", OPT_BOOL,
                         &is_opc_improved,
                         "If enabled, the OPC stage does not used the gpgpu_reg_file_port_throughput to make a loop of the OPC and recreates OPC stage more faithfully."
                         "gpgpu_reg_file_port_throughput will be used as the number of ports for each register file bank."
                         "Fix the OPC stage (default = disabled)",
                         "0");
  option_parser_register(opp, "-cu_num_ports", OPT_INT32,
                         &cu_num_ports, "Number of ports for each Collector Unit when they read operands from the Register File. Only enabled if is_opc_improved is enabled."
                         "Configure to any positive number (default=2)",
                         "2");

  // MOD. End. Improving OPC

  // MOD. Begin. Skip RF limitation
  option_parser_register(opp, "-is_skip_rf_limit_enabled", OPT_BOOL,
                         &is_skip_rf_limit_enabled,
                         "If enabled, the limitation of registers for a RF is not done."
                         "Skips checking the limitation of registers for a CTA (default = disabled)",
                         "0");
  // MOD. End

  // MOD. Begin. Relax barriers in baseline
  option_parser_register(opp, "-is_relax_barriers_baseline", OPT_BOOL,
                         &is_relax_barriers_baseline,
                         "If enabled, after a barrier only memory operations will stall a warp when this is waiting in a barrier (default = disabled)",
                         "0");
  // MOD. End

  // MOD. Begin. Extended IBuffer
  option_parser_register(opp, "-is_extended_ibuffer_enabled", OPT_BOOL,
                         &is_extended_ibuffer_enabled,
                         "If enabled, the extended buffer is used. Also if LOOG is enabled."
                         "(default = disabled)",
                         "0");
                         
  option_parser_register(opp, "-extended_ibuffer_size", OPT_INT32,
                         &extended_ibuffer_size, "Size of the extended Instruction Buffer. If LOOG is enabled, loog_frontend_size is used instead of this variable."
                         "Configure to any positive number (default=8)",
                         "8");

  option_parser_register(opp, "-fetch_decode_width", OPT_INT32,
                         &fetch_decode_width, "Size of the fetch and decode width. Cannot be bigger than extended_ibuffer_size or ibuffer_remodeled_size. If LOOG is enabled, loog_frontend_size is used instead of this variable."
                         "Configure to any positive number (default=2)",
                         "1");

  option_parser_register(opp, "-extended_ibuffer_static_power", OPT_DOUBLE,
                         &extended_ibuffer_static_power, "Static power consumption of each Extended Ibuffer."
                         "Configure to any positive number (default=0)",
                         "0");

  option_parser_register(opp, "-extended_ibuffer_dynamic_power", OPT_DOUBLE,
                         &extended_ibuffer_dynamic_power, "Dynamic power consumption of each Extended Ibuffer."
                         "Configure to any positive number (default=0)",
                         "0");
  // MOD. End. Extended IBuffer


  // MOD. Begin VPREG
  option_parser_register(
      opp, "-vpreg_mode", OPT_CSTR, &vpreg_mode_string,
      "Virtual Physical Registers mode: < disabled | reissue | reissue_informed > "
      "It enables a VPREG improvement in order to use renaming and reduce the size of register file. It also needs that -ibuffer_ooo_mode is set to wb."
      "reissue: In case of not having free physical registers at WB, the bits is_issued and is_reissued of the instruction in IBuffer_ooo are set to 1 and the instruction will be reissued."
      "reissue_informed: It works equally to reissue mode. However, the SOCGPU module knows and prevents to select instructions that are not the oldest when there aren't free physical registers."
      "disabled: The improvement is disabled."
      "Default: disabled",
      "disabled");

  option_parser_register(opp, "-vpreg_num_virtual_regs_per_sm", OPT_INT32,
                         &vpreg_num_virtual_regs_per_sm, "Number of virtual register inside each SM. Later on this number will be divided in 4 to assign equally to each sub-core. Recommended to be a power of 2"
                         "It is only used if vpreg is enabled. It should be bigger than the number of physical + maximum of destination registers allowed in an instruction."
                         "Configure to any number (default=2052)",
                         "2052");

  option_parser_register(opp, "-vpreg_num_physical_regs_per_sm", OPT_INT32,
                         &vpreg_num_physical_regs_per_sm, "Number of physical register inside each SM. Later on this number will be divided in 4 to assign equally to each sub-core. Recommended to be a power of 2"
                         "It is only used if vpreg is enabled."
                         "Configure to any positive number (default=2048)",
                         "2048");

  option_parser_register(opp, "-vpreg_balanced_banks_mode", OPT_BOOL,
                         &is_vpreg_balanced_banks_mode_enabled,
                         "If enabled, the allocation of physical registers will be balanced across the different banks."
                         "Virtual physical registers balanced banks mode (default = disabled)",
                         "0");

  option_parser_register(opp, "-vpreg_reissue_informed_socgpu_threshold", OPT_INT32,
                         &vpreg_reissue_informed_socgpu_threshold, "Number of physical registers that when it reaches this number or less than this number, only warps with the oldest instruction ready will be consider to be issued ."
                         "It is only used if vpreg is set to reissue_informed. It should be greater or equal than 0. This number will be for each sub-core free pool."
                         "Configure to any number greater or equal than 0 (default=0)",
                         "0");

  option_parser_register(opp, "-vpreg_max_rollback_entries_done_in_a_cycle", OPT_INT32,
                         &vpreg_max_rollback_entries_done_in_a_cycle, "Number of maximum entries that can do a rollback from the Instruction Buffer to the decode VMT in a cycle. Until the rollback is finished, the decode for that warp is stalled."
                         "If is set to 0 is treated like an ideal scenario where all the entries of the Instruction Bufffer do the rollback the same cycle as the flush is detected. Therefore, there isn't any stall. If it is configured equal to the size of the Instruction Buffer, it will have only one cycle stall."
                         "Configure to any number greater or equal than 0 and smaller or equal than the size of the Instruction Buffer(default=0)",
                         "0");

  option_parser_register(opp, "-is_vpreg_predicated_war_waw_dependencies_ignored", OPT_BOOL,
                         &is_vpreg_predicated_war_waw_dependencies_ignored,
                         "If enabled, the war and waw hazards because there is a predication register involded are ignored (default = disabled)",
                         "0");
  
  option_parser_register(opp, "-is_vpreg_predicated_dest_reg_dependencies_ignored", OPT_BOOL,
                         &is_vpreg_predicated_dest_reg_dependencies_ignored,
                         "If enabled, the dependencies when two instructions have the same logical destination register and different predication is ignored (default = disabled)",
                         "0");
  
  option_parser_register(opp, "-vpreg_merge_module_static_power", OPT_DOUBLE,
                         &vpreg_merge_module_static_power, "Static power consumption of each VPREG merge module."
                         "Configure to any positive number (default=0)",
                         "0");

  option_parser_register(opp, "-vpreg_merge_module_dynamic_power", OPT_DOUBLE,
                         &vpreg_merge_module_dynamic_power, "Dynamic power consumption of each VPREG merge module."
                         "Configure to any positive number (default=0)",
                         "0");

  option_parser_register(opp, "-vpreg_collector_unit_extra_static_power", OPT_DOUBLE,
                         &vpreg_collector_unit_extra_static_power, "Static power consumption of each collector unit module."
                         "Configure to any positive number (default=0)",
                         "0");

  option_parser_register(opp, "-vpreg_collector_unit_extra_dynamic_power", OPT_DOUBLE,
                         &vpreg_collector_unit_extra_dynamic_power, "Dynamic power consumption of each collector unit module."
                         "Configure to any positive number (default=0)",
                         "0");
  // MOD. End VPREG

  // MOD. Begin. Remodeling
  option_parser_register(
      opp, "-is_SM_remodeling_enabled", OPT_BOOL, &is_SM_remodeling_enabled,
      "If enabled, the simulator will use a more accurate model for the SMs "
      "based on NVIDIA Volta/Turing/Ampere."
      "is_SM_remodeling_enabled (default = disabled)",
      "0");
  option_parser_register(opp, "-num_subcores_in_SM", OPT_INT32,
                         &num_subcores_in_SM,
                         "Configures the number of subcores in the SM. Usually "
                         "4 in the latests NVIDIA architectures since Volta."
                         "num_subcores_in_SM (default = 4)",
                         "4");
  option_parser_register(
      opp, "-is_remodeling_scoreboarding_enabled", OPT_BOOL,
      &is_remodeling_scoreboarding_enabled,
      "If enabled, the simulator will use scoreboards in the accurate model "
      "for the SMs based on NVIDIA Volta/Turing/Ampere. Otherwise, it will use "
      "the model based on hints in the control bits written by the compiler. "
      "If it uses PTX mode, Scoreboards will be also enabled due to the "
      "imposibility of using the control bits. "
      "is_remodeling_scoreboarding_enabled (default = disabled)",
      "0");
  option_parser_register(opp, "-is_ibuffer_remodeled_enabled", OPT_BOOL,
                         &is_ibuffer_remodeled_enabled,
                         "If enabled, the extended buffer is used. Also if LOOG is enabled."
                         "(default = disabled)",
                         "0");
                         
  option_parser_register(opp, "-ibuffer_remodeled_size", OPT_INT32,
                         &ibuffer_remodeled_size, "Size of the extended Instruction Buffer. If LOOG is enabled, loog_frontend_size is used instead of this variable."
                         "Configure to any positive number (default=3)",
                         "3");
  option_parser_register(opp, "-num_wait_barriers_per_warp", OPT_UINT32,
                         &num_wait_barriers_per_warp, "Number of wait barriers that each warp has. Current architectures like Volta/Turing/Ampere have 6 barriers."
                         "Configure to any positive number (default=6)",
                         "6");
  option_parser_register(opp, "-sfu_latency", OPT_INT32,
                         &sfu_latency, "Latency of the SFU instructions."
                         "Configure to any positive number (default=21)",
                         "21");
  option_parser_register(opp, "-tensor_latency", OPT_INT32,
                         &tensor_latency, "Maximum latency of the Tensor instructions."
                         "Configure to any positive number (default=32)",
                         "32");
  option_parser_register(opp, "-tensor_extra_latency_16816_fp32_1688_fp32", OPT_INT32,
                         &tensor_extra_latency_16816_fp32_1688_fp32, "Maximum latency of the Tensor instructions."
                         "Configure to any positive number (default=0)",
                         "0");
  option_parser_register(opp, "-tensor_rate_per_cycle", OPT_INT32,
                         &tensor_rate_per_cycle, "Rate of processing of the tensor cores per cycle."
                         "Configure to any positive number (default=2048)",
                         "2048");
  option_parser_register(opp, "-branch_latency", OPT_INT32,
                         &branch_latency, "Latency of the branch instructions."
                         "Configure to any positive number (default=1)",
                         "1");
  option_parser_register(opp, "-half_latency", OPT_INT32,
                         &half_latency, "Latency of the half instructions."
                         "Configure to any positive number (default=2)",
                         "2");
  option_parser_register(opp, "-uniform_latency", OPT_INT32,
                         &uniform_latency, "Latency of the uniform instructions."
                         "Configure to any positive number (default=2)",
                         "2");
  option_parser_register(opp, "-predicate_latency", OPT_UINT32,
                         &predicate_latency, "Latency of the predicate instructions."
                         "Configure to any positive number (default=2)",
                         "2");
  option_parser_register(opp, "-miscellaneous_queue_latency", OPT_INT32,
                         &miscellaneous_queue_latency, "Latency of the miscellaneous queue instructions."
                         "Configure to any positive number (default=2)",
                         "2");
  option_parser_register(opp, "-miscellaneous_no_queue_latency", OPT_INT32,
                         &miscellaneous_no_queue_latency, "Latency of the miscellaneous no queue instructions."
                         "Configure to any positive number (default=1)",
                         "1");
  option_parser_register(opp, "-sfu_initiation", OPT_INT32,
                         &sfu_initiation, "Initiation interval of the SFU instructions in the dispatch latch. The number of cycles-1 that an instruction can not be sent to the same execution pipeline."
                         "Configure to any positive number (default=8)",
                         "8");
  option_parser_register(opp, "-tensor_initiation", OPT_INT32,
                         &tensor_initiation, "Initiation interval of the Tensor instructions in the dispatch latch. The number of cycles-1 that an instruction can not be sent to the same execution pipeline."
                         "Configure to any positive number (default=16)",
                         "16");
  option_parser_register(opp, "-branch_initiation", OPT_INT32,
                         &branch_initiation, "Initiation interval of the branch instructions in the dispatch latch. The number of cycles-1 that an instruction can not be sent to the same execution pipeline."
                         "Configure to any positive number (default=1)",
                         "1");
  option_parser_register(opp, "-half_initiation", OPT_INT32,
                         &half_initiation, "Initiation interval of the half instructions in the dispatch latch. The number of cycles-1 that an instruction can not be sent to the same execution pipeline."
                         "Configure to any positive number (default=2)",
                         "2");
  option_parser_register(opp, "-uniform_initiation", OPT_INT32,
                         &uniform_initiation, "Initiation interval of the uniform instructions in the dispatch latch. The number of cycles-1 that an instruction can not be sent to the same execution pipeline."
                         "Configure to any positive number (default=2)",
                         "2");
  option_parser_register(opp, "-predicate_initiation", OPT_INT32,
                         &predicate_initiation, "Initiation interval of the predicate instructions in the dispatch latch. The number of cycles-1 that an instruction can not be sent to the same execution pipeline."
                         "Configure to any positive number (default=2)",
                         "2");
  option_parser_register(opp, "-miscellaneous_queue_initiation", OPT_INT32,
                         &miscellaneous_queue_initiation, "Initiation interval of the miscellaneous queue instructions in the dispatch latch. The number of cycles-1 that an instruction can not be sent to the same execution pipeline."
                         "Configure to any positive number (default=2)",
                         "2");
  option_parser_register(opp, "-miscellaneous_no_queue_initiation", OPT_INT32,
                         &miscellaneous_no_queue_initiation, "Initiation interval of the miscellaneous no queue instructions in the dispatch latch. The number of cycles-1 that an instruction can not be sent to the same execution pipeline."
                         "Configure to any positive number (default=1)",
                         "1");
  option_parser_register(opp, "-miscellaneous_queue_size", OPT_UINT32,
                         &miscellaneous_queue_size, "Size of the queue inside each subcore for the miscellaneous queue instructions."
                         "Configure to any positive number (default=2)",
                         "2");
  option_parser_register(opp, "-memory_subcore_queue_size", OPT_UINT32,
                         &memory_subcore_queue_size, "Size of the queue inside each subcore for the LD/ST/TEXT instructions before being sent to the shared unit of the SM."
                         "Configure to any positive number (default=4)",
                         "4");
  option_parser_register(opp, "-memory_intermidiate_stages_subcore_unit", OPT_UINT32,
                         &memory_intermidiate_stages_subcore_unit, "Number of intermediate stages in the memory pipeline of the subcore unit."
                         "Configure to any positive number (default=3)",
                         "3");
  option_parser_register(opp, "-memory_sm_prt_size", OPT_UINT32,
                         &memory_sm_prt_size, "Size of the PRT inside the shared memory unit in each SM. It allows to have several instructions solving their accesses to the different types of memories."
                         "Configure to any positive number (default=64)",
                         "64");
  option_parser_register(opp, "-num_cycles_to_wait_to_dispatch_another_inst_from_subcore_to_sm_shared_pipeline_when_is_mem_inst", OPT_UINT32,
                         &num_cycles_to_wait_to_dispatch_another_inst_from_subcore_to_sm_shared_pipeline_when_is_mem_inst, "Number of cycles that the shared pipelines by sub-cores in the SM needs to wait until it is allowed to do the next issue when dispatched instruction is memory. Current cycle is included in the counter. For example, 1 allows to issue the next cycle."
                         "Configure to any positive number (default=2)",
                         "2");
  option_parser_register(opp, "-num_cycles_to_wait_to_dispatch_another_inst_from_subcore_to_sm_shared_pipeline_when_is_dp_inst", OPT_UINT32,
                         &num_cycles_to_wait_to_dispatch_another_inst_from_subcore_to_sm_shared_pipeline_when_is_dp_inst, "Number of cycles that the shared pipelines by sub-cores in the SM needs to wait until it is allowed to do the next issue when dispatched instruction is DP. Only effective if -dp_sm_shared_queue_size is 1. Current cycle is included in the counter. For example, 1 allows to issue the next cycle."
                         "Configure to any positive number (default=1)",
                         "1");
  option_parser_register(opp, "-memory_subcore_extra_latency_load_shared_mem", OPT_UINT32,
                         &memory_subcore_extra_latency_load_shared_mem, "Offset latency of the shared memory instruction when is being processed (calculating address and other stuff) inside each subcore."
                         "Configure to any positive number (default=2)",
                         "2");
  option_parser_register(opp, "-memory_shared_memory_minimum_latency", OPT_UINT32,
                         &memory_shared_memory_minimum_latency, "Minimum latency of the shared memory instruction when is accessing to the shared memory of the SM in the memory structures."
                         "Configure to any positive number (default=16)",
                         "16");
  option_parser_register(opp, "-memory_shared_memory_extra_latency_ldsm_multiple_matrix", OPT_UINT32,
                         &memory_shared_memory_extra_latency_ldsm_multiple_matrix, "Extra latency required at SM shared memory structures when LDSM loads multiple matrices."
                         "Configure to any positive number (default=2)",
                         "2");
  option_parser_register(opp, "-memmory_max_concurrent_requests_shmem_per_sm", OPT_UINT32,
                         &memmory_max_concurrent_requests_shmem_per_sm, "Maximum number of shared memory instructions that can be concurrent in the memory SM structure."
                         "Configure to any positive number (default=4)",
                         "4");
  option_parser_register(opp, "-memmory_max_concurrent_requests_standard_per_sm", OPT_UINT32,
                         &memmory_max_concurrent_requests_standard_per_sm, "Maximum number of standard memory instructions (no shared) that can be concurrent in the memory SM structure."
                         "Configure to any positive number (default=8)",
                         "8");
  option_parser_register(opp, "-sm_memory_unit_l1c_access_queue_size", OPT_UINT32,
                         &sm_memory_unit_l1c_access_queue_size, "Size of the access queue to the L1 constant cache at the memory SM structure."
                         "Configure to any positive number (default=1)",
                         "1");
  option_parser_register(opp, "-sm_memory_unit_l1t_access_queue_size", OPT_UINT32,
                          &sm_memory_unit_l1t_access_queue_size, "Size of the access queue to the L1 texture cache at the memory SM structure."
                          "Configure to any positive number (default=1)",
                          "1");
  option_parser_register(opp, "-sm_memory_unit_l1d_access_queue_size", OPT_UINT32,
                          &sm_memory_unit_l1d_access_queue_size, "Size of the access queue to the L1 constant cache at the memory SM structure. There is one queue per bank."
                          "Configure to any positive number (default=1)",
                          "1");
  option_parser_register(opp, "-sm_memory_unit_shmem_access_queue_size", OPT_UINT32,
                          &sm_memory_unit_shmem_access_queue_size, "Size of the access queue to the Shared memory (SHMEM) at the memory SM structure."
                          "Configure to any positive number (default=1)",
                          "1");
  option_parser_register(opp, "-sm_memory_unit_bypass_l1d_directly_go_to_l2_access_queue_size", OPT_UINT32,
                          &sm_memory_unit_bypass_l1d_directly_go_to_l2_access_queue_size, "Size of the access queue to directly to L2 because is bypassin l1d at the memory SM structure. There is one queue per bank."
                          "Configure to any positive number (default=1)",
                          "1");
  option_parser_register(opp, "-sm_memory_unit_miscellaneous_access_queue_size", OPT_UINT32,
                          &sm_memory_unit_miscellaneous_access_queue_size, "Size of the access queue to the miscellaneous at the memory SM structure."
                          "Configure to any positive number (default=1)",
                          "1");
  option_parser_register(opp, "-constant_cache_latency_at_sm_structure", OPT_UINT32,
                         &constant_cache_latency_at_sm_structure, "Minimum latency of the constant  memory instruction when is accessing to the constant memory (l1 constant cache) of the SM in the memory structures."
                         "Configure to any positive number (default=20)",
                         "20");
  option_parser_register(opp, "-memory_l1d_minimum_latency", OPT_UINT32,
                         &memory_l1d_minimum_latency, "Minimum latency of the global memory instruction when is accessing to the L1D cache of the SM in the memory structures."
                         "Configure to any positive number (default=11)",
                         "11");
  option_parser_register(opp, "-memory_global_shared_latency_for_ldgsts", OPT_UINT32,
                         &memory_global_shared_latency_for_ldgsts, "Latency of the ICNT network between global and shared memory for the the LDGSTS."
                         "Configure to any positive number (default=11)",
                         "11");
  option_parser_register(opp, "-memory_l1d_max_lookups_per_cycle_per_bank", OPT_UINT32,
                         &memory_l1d_max_lookups_per_cycle_per_bank, "Maximum number of look ups allowed per cycle in each bank of the L1D. If there are 4 banks and this parameter is set to 4, it will perform a maximum of 16 look ups in total.."
                         "Configure to any positive number (default=4)",
                         "4");
  option_parser_register(opp, "-memory_maximum_coalescing_cycles", OPT_UINT32,
                         &memory_maximum_coalescing_cycles, "Minimum latency of the memory instruction when it needs to perform a coalescing operation for a memory instruction."
                         "Configure to any positive number (default=1)",
                         "1");                                                    
  option_parser_register(opp, "-offset_latency_firts_stage_memory_subcore", OPT_INT32,
                         &offset_latency_firts_stage_memory_subcore, "Number of cycles difference at the first stage of memory subcore unit respect to the baseline architecture Ampere."
                         "Configure to any positive number (default=0)",
                         "0");   
  option_parser_register(opp, "-memory_num_scalar_units_per_subcore", OPT_UINT32,
                         &memory_num_scalar_units_per_subcore, "Number of scalar units inside each subcore for the memory instructions. Used for calculating addresses"
                         "Configure to any positive number (default=4)",
                         "8");
  option_parser_register(opp, "-memory_subcore_link_to_sm_byte_size", OPT_UINT32,
                         &memory_subcore_link_to_sm_byte_size, "Byte size between the link of a subcore and the SM for transfering data of memory instructions to SM memory structures."
                         "Configure to any positive number (default=4)",
                         "4");
  option_parser_register(opp, "-is_load_half_bandwidth_in_the_subcore_link_to_sm_enabled", OPT_BOOL,
                         &is_load_half_bandwidth_in_the_subcore_link_to_sm_enabled, "If enabled, laods have half of the bandwidth specified at -memory_subcore_link_to_sm_byte_size."
                         "Configure to any positive number (default=1)",
                         "1");
  option_parser_register(opp, "-is_store_half_bandwidth_in_the_subcore_link_to_sm_enabled", OPT_BOOL,
                         &is_store_half_bandwidth_in_the_subcore_link_to_sm_enabled, "If enabled, stores have half of the bandwidth specified at -memory_subcore_link_to_sm_byte_size."
                         "Configure to any positive number (default=0)",
                         "0");
  option_parser_register(opp, "-dp_subcore_queue_size", OPT_UINT32,
                         &dp_subcore_queue_size, "Size of the queue inside each subcore for the DP instructions before being sent to the shared unit of the SM. Only used if there is not dedicated DP unit in each sub-core"
                         "Configure to any positive number (default=4)",
                         "4");
  option_parser_register(opp, "-dp_subcore_max_latency", OPT_UINT32,
                         &dp_subcore_max_latency, "Number of maximum cycles that the DP instructions takes for transferring the data from the subcore to the shared unit of the SM. Only used if there is not dedicated DP unit in each sub-core. While is active, it is not possible to transfer other DP instruction."
                         "Configure to any positive number (default=8)",
                         "8");
  option_parser_register(opp, "-is_dp_pipeline_shared_for_subcores", OPT_BOOL,
                         &is_dp_pipeline_shared_for_subcores,
                         "If enabled, Subcores have a shared unit for executing the DP instructions. The -dp_subcore_queue_size is taken into account ."
                         "(default = enabled)",
                         "1");
  option_parser_register(opp, "-dp_sm_shared_queue_size", OPT_UINT32,
                         &dp_sm_shared_queue_size, "Size of the queue inside each SM shared execution unit for the DP instructions. Only used if there is not dedicated DP unit in each sub-core"
                         "Configure to any positive number (default=1)",
                         "1");
  option_parser_register(opp, "-dp_shared_intermidiate_stages", OPT_UINT32,
                         &dp_shared_intermidiate_stages, "Number of intermediate stages in the shared DP pipeline."
                         "Configure to any positive number (default=1)",
                         "1");
  option_parser_register(opp, "-is_fp32ops_allowed_in_int_pipeline", OPT_BOOL,
                         &is_fp32ops_allowed_in_int_pipeline,
                         "If enabled, FFMA instructions can be dispatched also to the INT execution pipeline."
                         "(default = disabled)",
                         "0");
  option_parser_register(opp, "-is_fp32_and_int_unified_pipeline", OPT_BOOL,
                         &is_fp32_and_int_unified_pipeline,
                         "If enabled, FP32 and INT instructions are dispatched to the same execution pipeline."
                         "(default = disabled)",
                         "0");
  option_parser_register(opp, "-invalidate_instruction_caches_at_kernel_end", OPT_BOOL,
                         &invalidate_instruction_caches_at_kernel_end,
                         "Invalidate instructions cache at the end of each kernel call", "0");
  option_parser_register(opp, "-ibuffer_coalescing", OPT_BOOL,
                         &ibuffer_coalescing,
                         "If enabled, IBuffers will snoop for instructions even though the served request is not for them."
                         "(default = disabled)",
                         "0");
  option_parser_register(opp, "-perfect_constant_cache", OPT_BOOL,
                         &perfect_constant_cache,
                         "If enabled, constant cache."
                         "(default = disabled)",
                         "0");
  option_parser_register(opp, "-perfect_instruction_cache", OPT_BOOL,
                         &perfect_instruction_cache,
                         "If enabled, Perfect_instruction cache."
                         "(default = disabled)",
                         "0");
  option_parser_register(opp, "-is_instruction_prefetching_enabled", OPT_BOOL,
                         &is_instruction_prefetching_enabled,
                         "If enabled, Instruction cache has prefetching enabled. The prefetcher just tries to prefetc subsequent blocks of the current one. That number of blocks can be configured with -prefetch_per_stream_buffer_size."
                         "(default = disabled)",
                         "0");
  option_parser_register(opp, "-prefetch_per_stream_buffer_size", OPT_UINT32,
                         &prefetch_per_stream_buffer_size, "Number of blocks that the instruction cache prefetcher (stream buffer)can hold. Only effective if -is_instruction_prefetching_enabled is enabled."
                         "Configure to any positive number (default=1)",
                         "1");
  option_parser_register(opp, "-prefetch_num_stream_buffers", OPT_UINT32,
                         &prefetch_num_stream_buffers, "Number of stream buffers in the first level instruction cache. Only effective if -is_instruction_prefetching_enabled is enabled."
                         "Configure to any positive number (default=1)",
                         "1");
  option_parser_register(opp, "-num_instruction_prefetches_per_cycle", OPT_UINT32,
                         &num_instruction_prefetches_per_cycle, "Number of blocks that the instruction cache prefetcher tries to prefetch every cycle. Only effective if -is_instruction_prefetching_enabled is enabled."
                         "Configure to any positive number (default=1)",
                         "1");
  option_parser_register(opp, "-is_rf_cache_enabled", OPT_BOOL,
                         &is_rf_cache_enabled,
                         "If enabled, Regular register file has the register file feature enabled."
                         "(default = enabled)",
                         "1");
  option_parser_register(opp, "-max_operands_regular_register_file", OPT_INT32,
                         &max_operands_regular_register_file, "Number of operands with regular register allowed per instruction. Only used when -is_rf_cache_enabled is enabled."
                         "Configure to any positive number (default=4)",
                         "4");
  option_parser_register(opp, "-max_latency_regular_register_file_latency", OPT_INT32,
                         &max_latency_regular_register_file_latency, "Maximum supported latency (cycles) for reading operands in the regular register file."
                         "Configure to any positive number (default=4)",
                         "4");
  option_parser_register(opp, "-num_regular_register_file_read_ports_per_bank", OPT_INT32,
                         &num_regular_register_file_read_ports_per_bank, "Number of read ports per bank in the regular register file available per cycle."
                         "Configure to any positive number (default=6)",
                         "6");
  option_parser_register(opp, "-num_regular_register_file_write_ports_per_bank", OPT_INT32,
                         &num_regular_register_file_write_ports_per_bank, "Number of write ports per bank in the regular register file available per cycle."
                         "Configure to any positive number (default=1)",
                         "1");
  option_parser_register(opp, "-max_size_register_file_write_queue_for_fixed_latency_instructions", OPT_INT32,
                         &max_size_register_file_write_queue_for_fixed_latency_instructions, "Maximum size of the register file write_queue for fixed latency instructions."
                         "Configure to any positive number (default=1)",
                         "1");
  option_parser_register(opp, "-max_pops_per_cycle_register_file_write_queue_for_fixed_latency_instructions", OPT_INT32,
                         &max_pops_per_cycle_register_file_write_queue_for_fixed_latency_instructions, "Maximum number of pops per cycle of the register file write_queue for fixed latency instructions."
                         "Configure to any positive number (default=1)",
                         "1");
  option_parser_register(opp, "-num_threads_granularity_read_regular_register_file_dp_inst", OPT_INT32,
                         &num_threads_granularity_read_regular_register_file_dp_inst, "Granularity of theads inside a warp for reading RF for DP instrucions. For example, 8 means that it needs 4 cycles because the width is 256 bits and takes 4 cycles to read the 32 threads (1024 width)."
                         "Configure to any positive number (default=8)",
                         "8");
  option_parser_register(opp, "-num_threads_granularity_read_regular_register_file_mem_inst", OPT_INT32,
                         &num_threads_granularity_read_regular_register_file_mem_inst, "Granularity of theads inside a warp for reading RF for memory instrucions. For example, 8 means that it needs 4 cycles because the width is 256 bits and takes 4 cycles to read the 32 threads (1024 width)."
                         "Configure to any positive number (default=8)",
                         "8");
  option_parser_register(opp, "-num_threads_granularity_read_regular_register_file_sfu_inst", OPT_INT32,
                         &num_threads_granularity_read_regular_register_file_sfu_inst, "Granularity of theads inside a warp for reading RF for special function instrucions. For example, 8 means that it needs 4 cycles because the width is 256 bits and takes 4 cycles to read the 32 threads (1024 width)."
                         "Configure to any positive number (default=8)",
                         "8");
  option_parser_register(opp, "-num_threads_granularity_read_regular_register_file_other_inst", OPT_INT32,
                         &num_threads_granularity_read_regular_register_file_other_inst, "Granularity of theads inside a warp for reading RF for DP instrucions. For example, 8 means that it needs 4 cycles because the width is 256 bits and takes 4 cycles to read the 32 threads (1024 width)."
                         "Configure to any positive number (default=8)",
                         "8");
  option_parser_register(opp, "-num_cycles_needed_to_write_a_reg_from_sm_struct_to_subcore", OPT_INT32,
                         &num_cycles_needed_to_write_a_reg_from_sm_struct_to_subcore, "Number of cycles needed to write a register from the SM structure (memory or shared dp if enabled) to the subcore."
                         "Configure to any positive number (default=2)",
                         "2");
  option_parser_register(opp, "-is_const_cache_accessed_blocks_tracking_enabled", OPT_BOOL,
                         &is_const_cache_accessed_blocks_tracking_enabled,
                         "If enabled, a set will track all the address accessed by the constant cache and then it will report the different number of accesses blocks at the end of the execution."
                         "(default = disabled)",
                         "0");
  option_parser_register(opp, "-is_global_memory_accesses_blocks_tracking_enabled", OPT_BOOL,
                         &is_global_memory_accesses_blocks_tracking_enabled,
                         "If enabled, a set will track all the address accessed by the global memory and then it will report the different number of accessed blocks at the end of the execution."
                         "(default = disabled)",
                         "0");
  option_parser_register(opp, "-num_const_cache_cycle_misses_before_switch_to_other_warp", OPT_UINT32,
                         &num_const_cache_cycle_misses_before_switch_to_other_warp, "Number of cycles the issue stage can not swap to a different warp due to constant cache keeps trying a miss."
                         "Configure to any positive number (default=4)",
                         "4");
  option_parser_register(opp, "-num_cycles_issue_port_busy_after_imadwide", OPT_UINT32,
                         &num_cycles_issue_port_busy_after_imadwide, "Number of cycles the issue stage can not issue any instruction because the previous one issued (an IMAD.WIDE) is keeping the issue port busy."
                         "Configure to any positive number (default=4)",
                         "4");
  option_parser_register(opp, "-num_stall_cycles_wait_after_bits_stall_0_and_yield", OPT_UINT32,
                         &num_stall_cycles_wait_after_bits_stall_0_and_yield, "Number of cycles that the stall counter is set when the instruction has the combination of Stall bits set to 0 and yield set on."
                         "Configure to any positive number (default=0)",
                         "0");
  option_parser_register(opp, "-num_cycles_to_stall_SM_at_gpu_memory_barrier", OPT_UINT32,
                         &num_cycles_to_stall_SM_at_gpu_memory_barrier, "Number of cycles that the SM is stalled after all the warps of the SM have reached the MEMBAR.GPU."
                         "Configure to any positive number (default=0)",
                         "186");
  option_parser_register(opp, "-num_cycles_to_stall_SM_at_system_memory_barrier", OPT_UINT32,
                         &num_cycles_to_stall_SM_at_system_memory_barrier, "Number of cycles that the SM is stalled after all the warps of the SM have reached the MEMBAR.SYS."
                         "Configure to any positive number (default=0)",
                         "2900");
  option_parser_register(opp, "-num_cycles_to_stall_SM_at_cta_memory_barrier", OPT_UINT32,
                         &num_cycles_to_stall_SM_at_cta_memory_barrier, "Number of cycles that the SM is stalled after all the warps of the SM have reached the MEMBAR.CTA."
                         "Configure to any positive number (default=0)",
                         "53");
  option_parser_register(
      opp, "-gpgpu_subcore_const_cache:l0", OPT_CSTR, &m_L0C_config.m_config_string,
      "per-subcore L0 constant memory cache  (READ-ONLY) config "
      " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_alloc>,<mshr>:<N>:<"
      "merge>,<mq>} ",
      "2:128:2,L:R:f:N,A:2:32,4");
  // MOD. End. Remodeling

  // MOD. Begin. Parallelism
  option_parser_register(opp, "-is_custom_omp_scheduler_enabled", OPT_BOOL,
                         &is_custom_omp_scheduler_enabled,
                         "If enabled, a set will track all the address accessed by the constant cache and then it will report the different number of accesses at the end of the execution."
                         "(default = enabled)",
                         "1");
  option_parser_register(opp, "-custom_omp_scheduler_ratio_to_dynamic", OPT_FLOAT,
                         &custom_omp_scheduler_ratio_to_dynamic,
                         "Ratio which below that number the OMP for-loop scheduler is set to dynamic. Range from 0 to 1."
                         "(default = 0.3)",
                         "0.3");
  // MOD. End. Parallelism

  option_parser_register(opp, "-filter_first_kernel_id", OPT_UINT32,
                          &filter_first_kernel_id, "If enabled, the simulation will start from the first kernel id configured in this parameter. If it has a value of 1 or 0 it is disabled."
                          "Configure to any positive number (default=0)",
                          "0");
  option_parser_register(opp, "-filter_last_kernel_id", OPT_UINT32,
                          &filter_last_kernel_id, "If enabled, the simulation will end at the last kernel id configured in this parameter. If it has a value of 1 or 0 it is disabled."
                          "Configure to any positive number (default=0)",
                          "0");
  // MOD. Begin. InterWarp coalescing
  option_parser_register(opp, "-measure_coalescing_potential_stats", OPT_BOOL,
    &measure_coalescing_potential_stats,
    "If enabled, We track the distance and how much intra and inter warp coalescing exists It consumes more RAM."
    "(default = 0 (disabled))",
    "0");
  option_parser_register(opp, "-is_interwarp_coalescing_enabled", OPT_BOOL,
    &is_interwarp_coalescing_enabled,
    "If enabled, We enable the Interwarp coalescing techniques."
    "(default = 0 (disabled))",
    "0");
  option_parser_register(opp, "-num_interwarp_coalescing_tables", OPT_UINT32,
    &num_interwarp_coalescing_tables,
    "Number of interwarp coalescing tables."
    "(default = 1)",
    "1");
  option_parser_register(opp, "-max_size_interwarp_coalescing_per_table", OPT_UINT32,
    &max_size_interwarp_coalescing_per_table,
    "Size of each interwarp coalescing tables."
    "(default = 64)",
    "64");
  option_parser_register(opp, "-interwarp_coalescing_quanta", OPT_UINT32,
    &interwarp_coalescing_quanta,
    "Number of cycles used for the quanta in interwarp coalescing in case of being need."
    "(default = 100000)",
    "100000");
  option_parser_register(opp, "-interwarp_coalescing_quanta_warppool_policy_miss_ratio_threshold", OPT_DOUBLE,
    &interwarp_coalescing_quanta_warppool_policy_miss_ratio_threshold,
    "Threshold where beyond that miss ratio in that quanta, GTL_WARPID policy is applied if we use WARPPOOL_HYBRID. Otherwise, OLDEST policy is applied."
    "(default = 0.99)",
    "0.99");
  option_parser_register(opp, "-number_of_coalescers", OPT_UINT32, &number_of_coalescers,
    "Number of coalescers applaying intra-warp coalescing to different instruction in each of the coalescers."
    "(default = 1)",
    "1");
  option_parser_register(opp, "-number_of_clusters_for_prt_selection", OPT_UINT32, &number_of_clusters_for_prt_selection,
    "Number of cluster that are going to be used for PRT selection when WARPID_N_CLUSTERS_WITH_OLDEST is chosen ."
    "(default = 16)", "16");
  option_parser_register(
    opp, "-interwarp_coalescing_selection_policy_string", OPT_CSTR, &interwarp_coalescing_selection_policy_string,
    "interwarp_coalescing_selection_policy_string mode: < OLDEST | GTL_WARPID | SAME_LAST_LEADER_INST_PC_THEN_OLDEST | WARPPOOL_HYBRID | DEP_COUNT_WAIT_OLDEST_INST_IBUFFER_GENERIC | DEP_COUNT_WAIT_OLDEST_INST_IBUFFER_CHECKING_WARP_ID | DEP_COUNT_WAIT_DETECTED_AT_DECODE_GENERIC | DEP_COUNT_WAIT_DETECTED_AT_DECODE_CHECKING_WARP_ID > "
    "Default: OLDEST",
    "OLDEST");
  option_parser_register(
    opp, "-prt_selection_policy_string", OPT_CSTR, &prt_selection_policy_string,
    "prt_selection_policy_string mode: < OLDEST | SAME_LAST_WARP_ID_THEN_OLDEST | SAME_LAST_INST_PC_THEN_OLDEST | WARPID_N_CLUSTERS_WITH_OLDEST | DEP_COUNT_WAIT_GENERIC_THEN_OLDEST | DEP_COUNT_WAIT_CHECKING_WARP_ID_THEN_OLDEST > "
    "Default: OLDEST",
    "OLDEST");
  
  // MOD. End. InterWarp coalescing

  // Virtual memory parameters
  option_parser_register(opp, "-is_num_virtual_pages_tracking_enabled", OPT_BOOL,
                         &is_num_virtual_pages_tracking_enabled,
                         "If enabled, a set will track all the different virtual pages. Then it will report the different number of virtual pages accessed at the end of the execution."
                         "(default = disabled)",
                         "0");
  option_parser_register(opp, "-virtual_page_size_in_bytes", OPT_UINT32,
                         &virtual_page_size_in_bytes, "Size of a virtual page in bytes. Default 2MiB == 2097152."
                         "Configure to any positive number (default=2097152)",
                         "2097152");
}

void gpgpu_sim_config::reg_options(option_parser_t opp) {
  gpgpu_functional_sim_config::reg_options(opp);
  m_shader_config.reg_options(opp);
  m_memory_config.reg_options(opp);
  power_config::reg_options(opp);
  option_parser_register(opp, "-gpgpu_max_cycle", OPT_INT64, &gpu_max_cycle_opt,
                         "terminates gpu simulation early (0 = no limit)", "0");
  option_parser_register(opp, "-gpgpu_max_insn", OPT_INT64, &gpu_max_insn_opt,
                         "terminates gpu simulation early (0 = no limit)", "0");
  option_parser_register(opp, "-gpgpu_max_cta", OPT_INT32, &gpu_max_cta_opt,
                         "terminates gpu simulation early (0 = no limit)", "0");
  option_parser_register(opp, "-gpgpu_max_completed_cta", OPT_INT32,
                         &gpu_max_completed_cta_opt,
                         "terminates gpu simulation early (0 = no limit)", "0");
  option_parser_register(
      opp, "-gpgpu_runtime_stat", OPT_CSTR, &gpgpu_runtime_stat,
      "display runtime statistics such as dram utilization {<freq>:<flag>}",
      "10000:0");
  option_parser_register(opp, "-liveness_message_freq", OPT_INT64,
                         &liveness_message_freq,
                         "Minimum number of seconds between simulation "
                         "liveness messages (0 = always print)",
                         "1");
  option_parser_register(opp, "-gpgpu_compute_capability_major", OPT_UINT32,
                         &gpgpu_compute_capability_major,
                         "Major compute capability version number", "7");
  option_parser_register(opp, "-gpgpu_compute_capability_minor", OPT_UINT32,
                         &gpgpu_compute_capability_minor,
                         "Minor compute capability version number", "0");
  option_parser_register(opp, "-gpgpu_flush_l1_cache", OPT_BOOL,
                         &gpgpu_flush_l1_cache,
                         "Flush L1 cache at the end of each kernel call", "0");
  option_parser_register(opp, "-gpgpu_flush_l2_cache", OPT_BOOL,
                         &gpgpu_flush_l2_cache,
                         "Flush L2 cache at the end of each kernel call", "0");
  option_parser_register(
      opp, "-gpgpu_deadlock_detect", OPT_BOOL, &gpu_deadlock_detect,
      "Stop the simulation at deadlock (1=on (default), 0=off)", "1");
  option_parser_register(
      opp, "-gpgpu_ptx_instruction_classification", OPT_INT32,
      &(gpgpu_ctx->func_sim->gpgpu_ptx_instruction_classification),
      "if enabled will classify ptx instruction types per kernel (Max 255 "
      "kernels now)",
      "0");
  option_parser_register(
      opp, "-gpgpu_ptx_sim_mode", OPT_INT32,
      &(gpgpu_ctx->func_sim->g_ptx_sim_mode),
      "Select between Performance (default) or Functional simulation (1)", "0");
  option_parser_register(opp, "-gpgpu_clock_domains", OPT_CSTR,
                         &gpgpu_clock_domains,
                         "Clock Domain Frequencies in MhZ {<Core Clock>:<ICNT "
                         "Clock>:<L2 Clock>:<DRAM Clock>}",
                         "500.0:2000.0:2000.0:2000.0");
  option_parser_register(
      opp, "-gpgpu_max_concurrent_kernel", OPT_INT32, &max_concurrent_kernel,
      "maximum kernels that can run concurrently on GPU, set this value "
      "according to max resident grids for your compute capability", "32");
  option_parser_register(
      opp, "-gpgpu_cflog_interval", OPT_INT32, &gpgpu_cflog_interval,
      "Interval between each snapshot in control flow logger", "0");
  option_parser_register(opp, "-visualizer_enabled", OPT_BOOL,
                         &g_visualizer_enabled,
                         "Turn on visualizer output (1=On, 0=Off)", "1");
  option_parser_register(opp, "-visualizer_outputfile", OPT_CSTR,
                         &g_visualizer_filename,
                         "Specifies the output log file for visualizer", NULL);
  option_parser_register(
      opp, "-visualizer_zlevel", OPT_INT32, &g_visualizer_zlevel,
      "Compression level of the visualizer output log (0=no comp, 9=highest)",
      "6");
  option_parser_register(opp, "-gpgpu_stack_size_limit", OPT_INT32,
                         &stack_size_limit, "GPU thread stack size", "1024");
  option_parser_register(opp, "-gpgpu_heap_size_limit", OPT_INT32,
                         &heap_size_limit, "GPU malloc heap size ", "8388608");
  option_parser_register(opp, "-gpgpu_runtime_sync_depth_limit", OPT_INT32,
                         &runtime_sync_depth_limit,
                         "GPU device runtime synchronize depth", "2");
  option_parser_register(opp, "-gpgpu_runtime_pending_launch_count_limit",
                         OPT_INT32, &runtime_pending_launch_count_limit,
                         "GPU device runtime pending launch count", "2048");
  option_parser_register(opp, "-trace_enabled", OPT_BOOL, &Trace::enabled,
                         "Turn on traces", "0");
  option_parser_register(opp, "-trace_components", OPT_CSTR, &Trace::config_str,
                         "comma seperated list of traces to enable. "
                         "Complete list found in trace_streams.tup. "
                         "Default none",
                         "none");
  option_parser_register(
      opp, "-trace_sampling_core", OPT_INT32, &Trace::sampling_core,
      "The core which is printed using CORE_DPRINTF. Default 0", "0");
  option_parser_register(opp, "-trace_sampling_memory_partition", OPT_INT32,
                         &Trace::sampling_memory_partition,
                         "The memory partition which is printed using "
                         "MEMPART_DPRINTF. Default -1 (i.e. all)",
                         "-1");
  gpgpu_ctx->stats->ptx_file_line_stats_options(opp);

  // Jin: kernel launch latency
  option_parser_register(opp, "-gpgpu_kernel_launch_latency", OPT_INT32,
                         &(gpgpu_ctx->device_runtime->g_kernel_launch_latency),
                         "Kernel launch latency in cycles. Default: 0", "0");
  option_parser_register(opp, "-gpgpu_cdp_enabled", OPT_BOOL,
                         &(gpgpu_ctx->device_runtime->g_cdp_enabled),
                         "Turn on CDP", "0");

  option_parser_register(opp, "-gpgpu_TB_launch_latency", OPT_INT32,
                         &(gpgpu_ctx->device_runtime->g_TB_launch_latency),
                         "thread block launch latency in cycles. Default: 0",
                         "0");
}

/////////////////////////////////////////////////////////////////////////////

void increment_x_then_y_then_z(dim3 &i, const dim3 &bound) {
  i.x++;
  if (i.x >= bound.x) {
    i.x = 0;
    i.y++;
    if (i.y >= bound.y) {
      i.y = 0;
      if (i.z < bound.z) i.z++;
    }
  }
}

void gpgpu_sim::launch(kernel_info_t *kinfo) {
  unsigned cta_size = kinfo->threads_per_cta();
  if (cta_size > m_shader_config->n_thread_per_shader) {
    printf(
        "Execution error: Shader kernel CTA (block) size is too large for "
        "microarch config.\n");
    printf("                 CTA size (x*y*z) = %u, max supported = %u\n",
           cta_size, m_shader_config->n_thread_per_shader);
    printf(
        "                 => either change -gpgpu_shader argument in "
        "gpgpusim.config file or\n");
    printf(
        "                 modify the CUDA source to decrease the kernel block "
        "size.\n");
    abort();
  }
  unsigned n = 0;
  for (n = 0; n < m_running_kernels.size(); n++) {
    if ((NULL == m_running_kernels[n]) || m_running_kernels[n]->done()) {
      m_running_kernels[n] = kinfo;
      break;
    }
  }
  m_shader_stats->allocate_for_a_new_kernel(); // MOD. Custom Stats
  m_shader_stats->num_kernel_not_in_binary += !kinfo->is_captured_from_binary;
  m_grid_barrier_status[kinfo->get_uid()] = grid_barrier_status(kinfo->get_uid(), 0);
  assert(n < m_running_kernels.size());
}

bool gpgpu_sim::can_start_kernel() {
  for (unsigned n = 0; n < m_running_kernels.size(); n++) {
    if ((NULL == m_running_kernels[n]) || m_running_kernels[n]->done())
      return true;
  }
  return false;
}

bool gpgpu_sim::hit_max_cta_count() const {
  if (m_config.gpu_max_cta_opt != 0) {
    if ((gpu_tot_issued_cta + m_total_cta_launched) >= m_config.gpu_max_cta_opt)
      return true;
  }
  return false;
}

bool gpgpu_sim::kernel_more_cta_left(kernel_info_t *kernel) const {
  if (hit_max_cta_count()) return false;

  if (kernel && !kernel->no_more_ctas_to_run()) return true;

  return false;
}

bool gpgpu_sim::get_more_cta_left() const {
  if (hit_max_cta_count()) return false;

  for (unsigned n = 0; n < m_running_kernels.size(); n++) {
    if (m_running_kernels[n] && !m_running_kernels[n]->no_more_ctas_to_run())
      return true;
  }
  return false;
}

void gpgpu_sim::decrement_kernel_latency() {
  for (unsigned n = 0; n < m_running_kernels.size(); n++) {
    if (m_running_kernels[n] && m_running_kernels[n]->m_kernel_TB_latency)
      m_running_kernels[n]->m_kernel_TB_latency--;
  }
}

kernel_info_t *gpgpu_sim::select_kernel() {
  if (m_running_kernels[m_last_issued_kernel] &&
      !m_running_kernels[m_last_issued_kernel]->no_more_ctas_to_run() &&
      !m_running_kernels[m_last_issued_kernel]->m_kernel_TB_latency) {
    unsigned launch_uid = m_running_kernels[m_last_issued_kernel]->get_uid();
    if (std::find(m_executed_kernel_uids.begin(), m_executed_kernel_uids.end(),
                  launch_uid) == m_executed_kernel_uids.end()) {
      m_running_kernels[m_last_issued_kernel]->start_cycle =
          gpu_sim_cycle + gpu_tot_sim_cycle;
      m_executed_kernel_uids.push_back(launch_uid);
      m_executed_kernel_names.push_back(
          m_running_kernels[m_last_issued_kernel]->name());
    }
    return m_running_kernels[m_last_issued_kernel];
  }

  for (unsigned n = 0; n < m_running_kernels.size(); n++) {
    unsigned idx =
        (n + m_last_issued_kernel + 1) % m_config.max_concurrent_kernel;
    if (kernel_more_cta_left(m_running_kernels[idx]) &&
        !m_running_kernels[idx]->m_kernel_TB_latency) {
      m_last_issued_kernel = idx;
      m_running_kernels[idx]->start_cycle = gpu_sim_cycle + gpu_tot_sim_cycle;
      // record this kernel for stat print if it is the first time this kernel
      // is selected for execution
      unsigned launch_uid = m_running_kernels[idx]->get_uid();
      assert(std::find(m_executed_kernel_uids.begin(),
                       m_executed_kernel_uids.end(),
                       launch_uid) == m_executed_kernel_uids.end());
      m_executed_kernel_uids.push_back(launch_uid);
      m_executed_kernel_names.push_back(m_running_kernels[idx]->name());

      return m_running_kernels[idx];
    }
  }
  return NULL;
}

unsigned gpgpu_sim::finished_kernel() {
  if (m_finished_kernel.empty()) return 0;
  unsigned result = m_finished_kernel.front();
  m_finished_kernel.pop_front();
  return result;
}

void gpgpu_sim::set_kernel_done(kernel_info_t *kernel) {
  unsigned uid = kernel->get_uid();
  m_grid_barrier_status.erase(uid);
  m_finished_kernel.push_back(uid);
  std::vector<kernel_info_t *>::iterator k;
  for (k = m_running_kernels.begin(); k != m_running_kernels.end(); k++) {
    if (*k == kernel) {
      kernel->end_cycle = gpu_sim_cycle + gpu_tot_sim_cycle;
      *k = NULL;
      break;
    }
  }
  assert(k != m_running_kernels.end());
}

void gpgpu_sim::stop_all_running_kernels() {
  std::vector<kernel_info_t *>::iterator k;
  for (k = m_running_kernels.begin(); k != m_running_kernels.end(); ++k) {
    if (*k != NULL) {       // If a kernel is active
      set_kernel_done(*k);  // Stop the kernel
      assert(*k == NULL);
    }
  }
}

void exec_gpgpu_sim::createSIMTCluster() {
  m_cluster = new simt_core_cluster *[m_shader_config->n_simt_clusters];
  for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++)
    m_cluster[i] =
        new exec_simt_core_cluster(this, i, m_shader_config, m_memory_config,
                                   m_shader_stats, m_memory_stats);
}

gpgpu_sim::gpgpu_sim(const gpgpu_sim_config &config, gpgpu_context *ctx)
    : gpgpu_t(config, ctx), m_gpu_per_sm_stats("GPU_per_SM_stats"),
    m_coalescing_stats_across_sms_l1d("l1d", _memory_space_t::global_space),
    m_coalescing_stats_across_sms_const("const", _memory_space_t::const_space),
    m_coalescing_stats_across_sms_sharedmem("sharedMem", _memory_space_t::shared_space),
    m_config(config)  {
  gpgpu_ctx = ctx;
  m_shader_config = &m_config.m_shader_config;
  m_memory_config = &m_config.m_memory_config;
  ctx->ptx_parser->set_ptx_warp_size(m_shader_config);
  ptx_file_line_stats_create_exposed_latency_tracker(m_config.num_shader());

  m_shader_stats = new shader_core_stats(m_shader_config, this);

  // MOD. Begin. Custom powermodel stats. Changed place to allow pass shader stats as parameter
  #ifdef GPGPUSIM_POWER_MODEL
    m_gpgpusim_wrapper = new gpgpu_sim_wrapper(config.g_power_simulation_enabled,
                                             config.g_power_config_name, config.g_power_simulation_mode, config.g_dvfs_enabled,
                                             &config, m_shader_config, m_shader_stats); // MOD. Custom stats powermodel
  #endif
  // MOD. Begin.

  m_memory_stats = new memory_stats_t(m_config.num_shader(), m_shader_config,
                                      m_memory_config, this);
  average_pipeline_duty_cycle = (float *)malloc(sizeof(float));
  active_sms = (float *)malloc(sizeof(float));
  total_sms_accumulated_across_cycles = 0; 
  m_power_stats =
      new power_stat_t(m_shader_config, average_pipeline_duty_cycle, active_sms,
                       m_shader_stats, m_memory_config, m_memory_stats);

  gpu_sim_insn = 0;
  gpu_tot_sim_insn = 0;
  gpu_tot_issued_cta = 0;
  gpu_completed_cta = 0;
  m_total_cta_launched = 0;
  gpu_deadlock = false;

  gpu_stall_dramfull = 0;
  gpu_stall_icnt2sh = 0;
  partiton_reqs_in_parallel = 0;
  partiton_reqs_in_parallel_total = 0;
  partiton_reqs_in_parallel_util = 0;
  partiton_reqs_in_parallel_util_total = 0;
  gpu_sim_cycle_parition_util = 0;
  gpu_tot_sim_cycle_parition_util = 0;
  partiton_replys_in_parallel = 0;
  partiton_replys_in_parallel_total = 0;

  m_memory_partition_unit =
      new memory_partition_unit *[m_memory_config->m_n_mem];
  m_memory_sub_partition =
      new memory_sub_partition *[m_memory_config->m_n_mem_sub_partition];
  for (unsigned i = 0; i < m_memory_config->m_n_mem; i++) {
    memory_stats_t *subpartition_mem_stats = new memory_stats_t(m_config.num_shader(), m_shader_config,
                                      m_memory_config, this);
    m_memory_partition_unit[i] =
        new memory_partition_unit(i, m_memory_config, subpartition_mem_stats, this);
        // new memory_partition_unit(i, m_memory_config, m_memory_stats, this);
    for (unsigned p = 0;
         p < m_memory_config->m_n_sub_partition_per_memory_channel; p++) {
      unsigned submpid =
          i * m_memory_config->m_n_sub_partition_per_memory_channel + p;
      m_memory_sub_partition[submpid] =
          m_memory_partition_unit[i]->get_sub_partition(p);
    }
  }

  icnt_wrapper_init();
  icnt_create(m_shader_config->n_simt_clusters,
              m_memory_config->m_n_mem_sub_partition, 0);

  time_vector_create(NUM_MEM_REQ_STAT);
  fprintf(stdout,
          "GPGPU-Sim uArch: performance model initialization complete.\n");

  m_running_kernels.resize(config.max_concurrent_kernel, NULL);
  m_last_issued_kernel = 0;
  m_last_cluster_issue = m_shader_config->n_simt_clusters -
                         1;  // this causes first launch to use simt cluster 0
  *average_pipeline_duty_cycle = 0;
  *active_sms = 0;

  last_liveness_message_time = 0;

  // Jin: functional simulation for CDP
  m_functional_sim = false;
  m_functional_sim_kernel = NULL;

  create_gpu_per_sm_stats();
}

gpgpu_sim::~gpgpu_sim() {
  delete m_shader_stats;
  delete m_memory_stats;
  delete m_power_stats;
  delete m_gpgpusim_wrapper; // MOD. Custom stats powermodel
  for(unsigned i = 0; i < m_memory_config->m_n_mem; i++) {
    delete m_memory_partition_unit[i];
  }
  delete[] m_memory_partition_unit;
  delete[] m_memory_sub_partition;
  for(unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
    delete m_cluster[i];
  }
  delete[] m_cluster;
  free(average_pipeline_duty_cycle);
  free(active_sms);
  icnt_delete(0);
  shader_CTA_count_destroy();
  time_vector_destroy();
  ptx_file_line_stats_destroy_exposed_latency_tracker();
}

void gpgpu_sim::create_gpu_per_sm_stats() {
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("gpu_sim_insn", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, "= ", "", true, true, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("gpgpu_n_mem_read_local", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, "= ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("gpgpu_n_mem_write_local", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, "= ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("gpgpu_n_mem_read_global", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, "= ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("gpgpu_n_mem_write_global", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, "= ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("gpgpu_n_mem_texture", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, "= ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("gpgpu_n_mem_const", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, "= ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("gpgpu_n_mem_read_inst", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, "= ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("gpgpu_n_mem_l2_writeback", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, "= ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("gpgpu_n_mem_l1_write_allocate", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, "= ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("gpgpu_n_mem_l2_write_allocate", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, "= ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("gpgpu_n_mem_grid_barrier", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, "= ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("gpgpu_n_mem_tlb_miss_data", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, "= ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("gpgpu_n_mem_tlb_miss_inst", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, "= ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("gpgpu_n_load_insn", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, " = ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("gpgpu_n_store_insn", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, "= ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("gpgpu_n_shmem_insn", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, "= ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("gpgpu_n_sstarr_insn", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, "= ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("gpgpu_n_tex_insn", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, "= ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("gpgpu_n_const_mem_insn", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, "= ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("gpgpu_n_param_mem_insn", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, "= ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("gpgpu_n_shmem_bkconflict", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, "= ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("gpgpu_n_shmem_port_conflict", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, "= ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("gpgpu_n_stall_dispatch_to_subpipeline_mem", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, "= ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("gpgpu_n_l1cache_bkconflict", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, "= ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("gpgpu_n_l1cache_coalescing_conflicts", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, "= ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("gpgpu_n_directly_to_l2_coalescing_conflicts", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, "= ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("gpgpu_n_cmem_portconflict", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, "= ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("gpgpu_n_cmem_coalescing_conflicts", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, "= ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("total_num_warp_instructions", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, "= ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("Total_effective_incomplete_warps", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, ": ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("total_accesses_l1d_instructions", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, ": ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("total_l1d_instructions", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, ": ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("total_conflicts_shared_instructions", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, ": ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("total_shared_instructions", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, ": ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("total_shared_mem_accesses", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, ": ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("total_num_ldst_unit_instructions", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, ": ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("total_num_dp_instructions", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, ": ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("ctas_completed", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, ": ", "", true, false, false);

  m_gpu_per_sm_stats.add_unsigned_long_long_stat("total_num_times_wb_port_conflict", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, ": ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("total_num_times_wb_evaluated", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, ": ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("total_num_cycles_issue_stage_issuing", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, ": ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("total_num_evals_rf", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, ": ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("total_num_evals_rf_with_conflict", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, ": ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("total_num_cycles_issue_stage_stall_next_stage_not_available", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, ": ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("total_num_cycles_issue_stage_stall_issue_port_busy", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, ": ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("total_num_cycles_issue_stage_stall_no_valid_instruction", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, ": ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("total_num_cycles_issue_stage_stall_no_warps_ready", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, ": ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("total_num_cycles_issue_stage_evaluated", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, ": ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("total_num_register_file_cache_hits", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, ": ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("total_num_register_file_cache_allocations", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, ": ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("total_num_regular_regfile_reads", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, ": ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("total_num_regular_regfile_writes", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, ": ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("total_num_constant_cache_reads", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, ": ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("total_num_uniform_predicate_regfile_writes", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, ": ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("total_num_uniform_predicate_regfile_reads", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, ": ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("total_num_uniform_regfile_writes", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, ": ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("total_num_uniform_regfile_reads", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, ": ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("total_num_predicate_regfile_writes", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, ": ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("total_num_predicate_regfile_reads", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, ": ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("total_accesses", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, ": ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("total_accesses_coalesced", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, ": ", "", true, false, false);
  m_gpu_per_sm_stats.add_unsigned_long_long_stat("total_accesses_not_coalesced", AllowedTypesStats::UNSIGNED_LONG_LONG, 0, ": ", "", true, false, false);
  
  for(unsigned int i = 0; i < N_MEM_STAGE_ACCESS_TYPE; i++) {
    for(unsigned int j = 0; j < N_MEM_STAGE_STALL_TYPE; j++) {
      std::string stat_name = "gpgpu_stall_shd_mem[" + mem_stage_access_type_to_string(static_cast<mem_stage_access_type>(i)) + "][" + mem_stage_stall_type_to_string(static_cast<mem_stage_stall_type>(j)) + "]";
      m_gpu_per_sm_stats.add_unsigned_long_long_stat(stat_name, AllowedTypesStats::UNSIGNED_LONG_LONG, 0, ": ", "", true, false, false);
    }
  }

  for(unsigned int i = 0; i < m_shader_config->warp_size + 1; i++) {
    m_gpu_per_sm_stats.add_unsigned_long_long_stat("warp_occ_dist" + std::to_string(i), AllowedTypesStats::UNSIGNED_LONG_LONG, 0, "= ", "", true, true, false);
  }

  
}

void gpgpu_sim::gather_gpu_per_sm_stats() {
  for(unsigned int i = 0; i < m_shader_config->n_simt_clusters; i++) {
    for(unsigned int j = 0; j < m_shader_config->n_simt_cores_per_cluster; j++) {
      m_cluster[i]->gather_stats(m_gpu_per_sm_stats, m_coalescing_stats_across_sms_l1d, m_coalescing_stats_across_sms_const, m_coalescing_stats_across_sms_sharedmem);
      m_shader_stats->m_incoming_traffic_stats->join_stats(m_cluster[i]->get_incomming_traffic_stats());
      m_shader_stats->m_outgoing_traffic_stats->join_stats(m_cluster[i]->get_outgoing_traffic_stats());
    }
  }
}

void gpgpu_sim::reset_cycless_access_history() {
  for(unsigned int i = 0; i < m_shader_config->n_simt_clusters; i++) {
    for(unsigned int j = 0; j < m_shader_config->n_simt_cores_per_cluster; j++) {
      m_cluster[i]->reset_cycless_access_history();
    }
  }
}

void gpgpu_sim::gather_gpu_per_sm_single_stat(std::string stat_name) {
  for(unsigned int i = 0; i < m_shader_config->n_simt_clusters; i++) {
    for(unsigned int j = 0; j < m_shader_config->n_simt_cores_per_cluster; j++) {
      m_cluster[i]->gather_single_stat(m_gpu_per_sm_stats, stat_name);
    }
  }
}

void gpgpu_sim::reset_gpu_per_sm_stats() {
  for(auto stat_name : m_gpu_per_sm_stats.m_stats_name) {
    m_gpu_per_sm_stats.reset_stat(stat_name);
  }
}

int gpgpu_sim::shared_mem_size() const {
  return m_shader_config->gpgpu_shmem_size;
}

int gpgpu_sim::shared_mem_per_block() const {
  return m_shader_config->gpgpu_shmem_per_block;
}

int gpgpu_sim::num_registers_per_core() const {
  // MOD. Begin. VPREG
  if(m_shader_config->is_vpreg_enabled) {
    return m_shader_config->vpreg_num_physical_regs_per_sm * 32; // Translate from warp registers to thread registers
  }else {
    return m_shader_config->gpgpu_shader_registers;
  }
  // MOD. End. VPREG
}

int gpgpu_sim::num_registers_per_block() const {
  return m_shader_config->gpgpu_registers_per_block;
}

int gpgpu_sim::wrp_size() const { return m_shader_config->warp_size; }

int gpgpu_sim::shader_clock() const { return m_config.core_freq / 1000; }

int gpgpu_sim::max_cta_per_core() const {
  return m_shader_config->max_cta_per_core;
}

int gpgpu_sim::get_max_cta(const kernel_info_t &k) const {
  return m_shader_config->max_cta(k);
}

void gpgpu_sim::set_prop(cudaDeviceProp *prop) { m_cuda_properties = prop; }

int gpgpu_sim::compute_capability_major() const {
  return m_config.gpgpu_compute_capability_major;
}

int gpgpu_sim::compute_capability_minor() const {
  return m_config.gpgpu_compute_capability_minor;
}

const struct cudaDeviceProp *gpgpu_sim::get_prop() const {
  return m_cuda_properties;
}

enum divergence_support_t gpgpu_sim::simd_model() const {
  return m_shader_config->model;
}

void gpgpu_sim_config::init_clock_domains(void) {
  sscanf(gpgpu_clock_domains, "%lf:%lf:%lf:%lf", &core_freq, &icnt_freq,
         &l2_freq, &dram_freq);
  core_freq = core_freq MhZ;
  icnt_freq = icnt_freq MhZ;
  l2_freq = l2_freq MhZ;
  dram_freq = dram_freq MhZ;
  core_period = 1 / core_freq;
  icnt_period = 1 / icnt_freq;
  dram_period = 1 / dram_freq;
  l2_period = 1 / l2_freq;
  printf("GPGPU-Sim uArch: clock freqs: %lf:%lf:%lf:%lf\n", core_freq,
         icnt_freq, l2_freq, dram_freq);
  printf("GPGPU-Sim uArch: clock periods: %.20lf:%.20lf:%.20lf:%.20lf\n",
         core_period, icnt_period, l2_period, dram_period);
}

void gpgpu_sim::reinit_clock_domains(void) {
  core_time = 0;
  dram_time = 0;
  icnt_time = 0;
  l2_time = 0;
}

bool gpgpu_sim::active() {
  if (m_config.gpu_max_cycle_opt &&
      (gpu_tot_sim_cycle + gpu_sim_cycle) >= m_config.gpu_max_cycle_opt)
    return false;
  if (m_config.gpu_max_insn_opt &&
      (gpu_tot_sim_insn + gpu_sim_insn) >= m_config.gpu_max_insn_opt)
    return false;
  if (m_config.gpu_max_cta_opt &&
      (gpu_tot_issued_cta >= m_config.gpu_max_cta_opt))
    return false;
  if (m_config.gpu_max_completed_cta_opt &&
      (gpu_completed_cta >= m_config.gpu_max_completed_cta_opt))
    return false;
  if (m_config.gpu_deadlock_detect && gpu_deadlock) return false;
  for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++)
    if (m_cluster[i]->get_not_completed() > 0) return true;
  ;
  for (unsigned i = 0; i < m_memory_config->m_n_mem; i++)
    if (m_memory_partition_unit[i]->busy() > 0) return true;
  ;
  if (icnt_busy(0)) return true;
  if (get_more_cta_left()) return true;
  return false;
}

void gpgpu_sim::init() {
  // run a CUDA grid on the GPU microarchitecture simulator
  gpu_sim_cycle = 0;
  dram_sim_cycle = 0;
  gpu_sim_insn = 0;
  last_gpu_sim_insn = 0;
  m_total_cta_launched = 0;
  gpu_completed_cta = 0;
  partiton_reqs_in_parallel = 0;
  partiton_replys_in_parallel = 0;
  partiton_reqs_in_parallel_util = 0;
  gpu_sim_cycle_parition_util = 0;

// McPAT initialization function. Called on first launch of GPU
#ifdef GPGPUSIM_POWER_MODEL
  if (m_config.g_power_simulation_enabled) {
    init_mcpat(m_config, m_gpgpusim_wrapper, m_config.gpu_stat_sample_freq,
               gpu_tot_sim_insn, gpu_sim_insn, 0);
  }
#endif

  reinit_clock_domains();
  gpgpu_ctx->func_sim->set_param_gpgpu_num_shaders(m_config.num_shader());
  for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++)
    m_cluster[i]->reinit();
  m_shader_stats->new_grid();
  // initialize the control-flow, memory access, memory latency logger
  if (m_config.g_visualizer_enabled) {
    create_thread_CFlogger(gpgpu_ctx, m_config.num_shader(),
                           m_shader_config->n_thread_per_shader, 0,
                           m_config.gpgpu_cflog_interval);
  }
  shader_CTA_count_create(m_config.num_shader(), m_config.gpgpu_cflog_interval);
  if (m_config.gpgpu_cflog_interval != 0) {
    insn_warp_occ_create(m_config.num_shader(), m_shader_config->warp_size);
    shader_warp_occ_create(m_config.num_shader(), m_shader_config->warp_size,
                           m_config.gpgpu_cflog_interval);
    shader_mem_acc_create(m_config.num_shader(), m_memory_config->m_n_mem, 4,
                          m_config.gpgpu_cflog_interval);
    shader_mem_lat_create(m_config.num_shader(), m_config.gpgpu_cflog_interval);
    shader_cache_access_create(m_config.num_shader(), 3,
                               m_config.gpgpu_cflog_interval);
    set_spill_interval(m_config.gpgpu_cflog_interval * 40);
  }

  if (g_network_mode) icnt_init(0);
}

void gpgpu_sim::update_stats() {
  m_memory_stats->memlatstat_lat_pw();
  gpu_tot_sim_cycle += gpu_sim_cycle;
  gpu_tot_sim_insn += gpu_sim_insn;
  gpu_tot_issued_cta += m_total_cta_launched;
  partiton_reqs_in_parallel_total += partiton_reqs_in_parallel;
  partiton_replys_in_parallel_total += partiton_replys_in_parallel;
  partiton_reqs_in_parallel_util_total += partiton_reqs_in_parallel_util;
  gpu_tot_sim_cycle_parition_util += gpu_sim_cycle_parition_util;
  gpu_tot_occupancy += gpu_occupancy;

  gpu_sim_cycle = 0;
  dram_sim_cycle = 0;
  partiton_reqs_in_parallel = 0;
  partiton_replys_in_parallel = 0;
  partiton_reqs_in_parallel_util = 0;
  gpu_sim_cycle_parition_util = 0;
  gpu_sim_insn = 0;
  m_total_cta_launched = 0;
  gpu_completed_cta = 0;
  gpu_occupancy = occupancy_stats();
}

PowerscalingCoefficients *gpgpu_sim::get_scaling_coeffs()
{
  return m_gpgpusim_wrapper->get_scaling_coeffs();
}

void gpgpu_sim::print_stats() {
  m_shader_stats->gpu_cycles_per_kernel[m_shader_stats->m_current_kernel_pos]=gpu_sim_cycle; // MOD. Custom Stats

  gpgpu_ctx->stats->ptx_file_line_stats_write_file();
  gpu_print_stat();

  if (g_network_mode) {
    printf(
        "----------------------------Interconnect-DETAILS----------------------"
        "----------\n");
    icnt_display_stats(0);
    icnt_display_overall_stats(0);
    printf(
        "----------------------------END-of-Interconnect-DETAILS---------------"
        "----------\n");
  }
}

void gpgpu_sim::deadlock_check() {
  if (m_config.gpu_deadlock_detect && gpu_deadlock) {
    fflush(stdout);
    printf(
        "\n\nGPGPU-Sim uArch: ERROR ** deadlock detected: last writeback core "
        "%u @ gpu_sim_cycle %u (+ gpu_tot_sim_cycle %u) (%u cycles ago)\n",
        gpu_sim_insn_last_update_sid, (unsigned)gpu_sim_insn_last_update,
        (unsigned)(gpu_tot_sim_cycle - gpu_sim_cycle),
        (unsigned)(gpu_sim_cycle - gpu_sim_insn_last_update));
    unsigned num_cores = 0;
    for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
      unsigned not_completed = m_cluster[i]->get_not_completed();
      if (not_completed) {
        if (!num_cores) {
          printf(
              "GPGPU-Sim uArch: DEADLOCK  shader cores no longer committing "
              "instructions [core(# threads)]:\n");
          printf("GPGPU-Sim uArch: DEADLOCK  ");
          m_cluster[i]->print_not_completed(stdout);
        } else if (num_cores < 8) {
          m_cluster[i]->print_not_completed(stdout);
        } else if (num_cores >= 8) {
          printf(" + others ... ");
        }
        num_cores += m_shader_config->n_simt_cores_per_cluster;
      }
    }
    printf("\n");
    for (unsigned i = 0; i < m_memory_config->m_n_mem; i++) {
      bool busy = m_memory_partition_unit[i]->busy();
      if (busy)
        printf("GPGPU-Sim uArch DEADLOCK:  memory partition %u busy\n", i);
    }
    if (icnt_busy(0)) {
      printf("GPGPU-Sim uArch DEADLOCK:  iterconnect contains traffic\n");
      icnt_display_state(stdout, 0);
    }
    printf(
        "\nRe-run the simulator in gdb and use debug routines in .gdbinit to "
        "debug this\n");
    fflush(stdout);
    abort();
  }
}

/// printing the names and uids of a set of executed kernels (usually there is
/// only one)
std::string gpgpu_sim::executed_kernel_info_string() {
  std::stringstream statout;

  statout << "kernel_name = ";
  for (unsigned int k = 0; k < m_executed_kernel_names.size(); k++) {
    statout << m_executed_kernel_names[k] << " ";
  }
  statout << std::endl;
  statout << "kernel_launch_uid = ";
  for (unsigned int k = 0; k < m_executed_kernel_uids.size(); k++) {
    statout << m_executed_kernel_uids[k] << " ";
  }
  statout << std::endl;

  return statout.str();
}

std::string gpgpu_sim::executed_kernel_name() {
  std::stringstream statout;  
  if( m_executed_kernel_names.size() == 1)
     statout << m_executed_kernel_names[0];
  else{
    for (unsigned int k = 0; k < m_executed_kernel_names.size(); k++) {
      statout << m_executed_kernel_names[k] << " ";
    }
  }
  return statout.str();
}
void gpgpu_sim::set_cache_config(std::string kernel_name,
                                 FuncCache cacheConfig) {
  m_special_cache_config[kernel_name] = cacheConfig;
}

FuncCache gpgpu_sim::get_cache_config(std::string kernel_name) {
  for (std::map<std::string, FuncCache>::iterator iter =
           m_special_cache_config.begin();
       iter != m_special_cache_config.end(); iter++) {
    std::string kernel = iter->first;
    if (kernel_name.compare(kernel) == 0) {
      return iter->second;
    }
  }
  return (FuncCache)0;
}

bool gpgpu_sim::has_special_cache_config(std::string kernel_name) {
  for (std::map<std::string, FuncCache>::iterator iter =
           m_special_cache_config.begin();
       iter != m_special_cache_config.end(); iter++) {
    std::string kernel = iter->first;
    if (kernel_name.compare(kernel) == 0) {
      return true;
    }
  }
  return false;
}

void gpgpu_sim::set_cache_config(std::string kernel_name) {
  if (has_special_cache_config(kernel_name)) {
    change_cache_config(get_cache_config(kernel_name));
  } else {
    change_cache_config(FuncCachePreferNone);
  }
}

void gpgpu_sim::change_cache_config(FuncCache cache_config) {
  if (cache_config != m_shader_config->m_L1D_config.get_cache_status()) {
    printf("FLUSH L1 Cache at configuration change between kernels\n");
    for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
      m_cluster[i]->cache_invalidate();
    }
  }

  switch (cache_config) {
    case FuncCachePreferNone:
      m_shader_config->m_L1D_config.init(
          m_shader_config->m_L1D_config.m_config_string, FuncCachePreferNone);
      m_shader_config->gpgpu_shmem_size =
          m_shader_config->gpgpu_shmem_sizeDefault;
      break;
    case FuncCachePreferL1:
      if ((m_shader_config->m_L1D_config.m_config_stringPrefL1 == NULL) ||
          (m_shader_config->gpgpu_shmem_sizePrefL1 == (unsigned)-1)) {
        printf("WARNING: missing Preferred L1 configuration\n");
        m_shader_config->m_L1D_config.init(
            m_shader_config->m_L1D_config.m_config_string, FuncCachePreferNone);
        m_shader_config->gpgpu_shmem_size =
            m_shader_config->gpgpu_shmem_sizeDefault;

      } else {
        m_shader_config->m_L1D_config.init(
            m_shader_config->m_L1D_config.m_config_stringPrefL1,
            FuncCachePreferL1);
        m_shader_config->gpgpu_shmem_size =
            m_shader_config->gpgpu_shmem_sizePrefL1;
      }
      break;
    case FuncCachePreferShared:
      if ((m_shader_config->m_L1D_config.m_config_stringPrefShared == NULL) ||
          (m_shader_config->gpgpu_shmem_sizePrefShared == (unsigned)-1)) {
        printf("WARNING: missing Preferred L1 configuration\n");
        m_shader_config->m_L1D_config.init(
            m_shader_config->m_L1D_config.m_config_string, FuncCachePreferNone);
        m_shader_config->gpgpu_shmem_size =
            m_shader_config->gpgpu_shmem_sizeDefault;
      } else {
        m_shader_config->m_L1D_config.init(
            m_shader_config->m_L1D_config.m_config_stringPrefShared,
            FuncCachePreferShared);
        m_shader_config->gpgpu_shmem_size =
            m_shader_config->gpgpu_shmem_sizePrefShared;
      }
      break;
    default:
      break;
  }
}

void gpgpu_sim::clear_executed_kernel_info() {
  m_executed_kernel_names.clear();
  m_executed_kernel_uids.clear();
}
void gpgpu_sim::gpu_print_stat() {
  FILE *statfout = stdout;

  std::string kernel_info_str = executed_kernel_info_string();
  gather_gpu_per_sm_stats();
  reset_cycless_access_history();
  
  gpu_sim_insn = m_gpu_per_sm_stats.m_stats_map["gpu_sim_insn"]->get_value();
  
  fprintf(statfout, "%s", kernel_info_str.c_str());

  printf("gpu_sim_cycle = %lld\n", gpu_sim_cycle);
  m_gpu_per_sm_stats.m_stats_map["gpu_sim_insn"]->print(statfout);
  printf("gpu_ipc = %12.4f\n", (float)gpu_sim_insn / gpu_sim_cycle);
  printf("gpu_tot_sim_cycle = %lld\n", gpu_tot_sim_cycle + gpu_sim_cycle);
  printf("gpu_tot_sim_insn = %lld\n", gpu_tot_sim_insn + gpu_sim_insn);
  printf("gpu_tot_ipc = %12.4f\n", (float)(gpu_tot_sim_insn + gpu_sim_insn) /
                                       (gpu_tot_sim_cycle + gpu_sim_cycle));
  printf("gpu_tot_issued_cta = %lld\n",
         gpu_tot_issued_cta + m_total_cta_launched);
  printf("gpu_occupancy = %.4f%% \n", gpu_occupancy.get_occ_fraction() * 100);
  printf("gpu_tot_occupancy = %.4f%% \n",
         (gpu_occupancy + gpu_tot_occupancy).get_occ_fraction() * 100);
  printf("gpu_tot_sms_occupancy = %.4f%%\n", ((*active_sms )/ total_sms_accumulated_across_cycles)*100);
  fprintf(statfout, "max_total_param_size = %llu\n",
          gpgpu_ctx->device_runtime->g_max_total_param_size);

  // performance counter for stalls due to congestion.
  printf("gpu_stall_dramfull = %d\n", gpu_stall_dramfull);
  printf("gpu_stall_icnt2sh    = %d\n", gpu_stall_icnt2sh);

  // printf("partiton_reqs_in_parallel = %lld\n", partiton_reqs_in_parallel);
  // printf("partiton_reqs_in_parallel_total    = %lld\n",
  // partiton_reqs_in_parallel_total );
  printf("partiton_level_parallism = %12.4f\n",
         (float)partiton_reqs_in_parallel / gpu_sim_cycle);
  printf("partiton_level_parallism_total  = %12.4f\n",
         (float)(partiton_reqs_in_parallel + partiton_reqs_in_parallel_total) /
             (gpu_tot_sim_cycle + gpu_sim_cycle));
  // printf("partiton_reqs_in_parallel_util = %lld\n",
  // partiton_reqs_in_parallel_util);
  // printf("partiton_reqs_in_parallel_util_total    = %lld\n",
  // partiton_reqs_in_parallel_util_total ); printf("gpu_sim_cycle_parition_util
  // = %lld\n", gpu_sim_cycle_parition_util);
  // printf("gpu_tot_sim_cycle_parition_util    = %lld\n",
  // gpu_tot_sim_cycle_parition_util );
  printf("partiton_level_parallism_util = %12.4f\n",
         (float)partiton_reqs_in_parallel_util / gpu_sim_cycle_parition_util);
  printf("partiton_level_parallism_util_total  = %12.4f\n",
         (float)(partiton_reqs_in_parallel_util +
                 partiton_reqs_in_parallel_util_total) /
             (gpu_sim_cycle_parition_util + gpu_tot_sim_cycle_parition_util));
  // printf("partiton_replys_in_parallel = %lld\n",
  // partiton_replys_in_parallel); printf("partiton_replys_in_parallel_total =
  // %lld\n", partiton_replys_in_parallel_total );
  printf("L2_BW  = %12.4f GB/Sec\n",
         ((float)(partiton_replys_in_parallel * 32) /
          (gpu_sim_cycle * m_config.icnt_period)) /
             1000000000);
  printf("L2_BW_total  = %12.4f GB/Sec\n",
         ((float)((partiton_replys_in_parallel +
                   partiton_replys_in_parallel_total) *
                  32) /
          ((gpu_tot_sim_cycle + gpu_sim_cycle) * m_config.icnt_period)) /
             1000000000);

  time_t curr_time;
  time(&curr_time);
  unsigned long long elapsed_time =
      MAX(curr_time - gpgpu_ctx->the_gpgpusim->g_simulation_starttime, 1);
  printf("gpu_total_sim_rate=%u\n",
         (unsigned)((gpu_tot_sim_insn + gpu_sim_insn) / elapsed_time));

  // shader_print_l1_miss_stat( stdout );
  shader_print_cache_stats(stdout);

  cache_stats core_cache_stats;
  core_cache_stats.clear();
  for (unsigned i = 0; i < m_config.num_cluster(); i++) {
    m_cluster[i]->get_cache_stats(core_cache_stats);
  }
  core_cache_stats.compute_total_write_and_read_accesses();
  double total_cache_accesses = core_cache_stats.get_total_write_and_read_accesses();
  double l1d_bw = ((total_cache_accesses * 32) / ((gpu_tot_sim_cycle + gpu_sim_cycle) * m_config.core_period)) / 1000000000;
  printf("L1D_BW_total  = %12.4lf GB/Sec\n", l1d_bw);
  printf("\nTotal_core_cache_stats:\n");
  core_cache_stats.print_stats(stdout, "Total_core_cache_stats_breakdown");
  printf("\nTotal_core_cache_fail_stats:\n");
  core_cache_stats.print_fail_stats(stdout,
                                    "Total_core_cache_fail_stats_breakdown");
  shader_print_scheduler_stat(stdout, false);

  m_shader_stats->compute_derived_custom_stats(); // MOD. Custom Stats
  m_shader_stats->compute_ibuffer_ooo_stats(); // MOD. IBuffer_ooo custom Stats
  m_shader_stats->print(stdout);
#ifdef GPGPUSIM_POWER_MODEL
  if (m_config.g_power_simulation_enabled) {
    if(m_config.g_power_simulation_mode > 0){
        //if(!m_config.g_aggregate_power_stats)
          mcpat_reset_perf_count(m_gpgpusim_wrapper);
        calculate_hw_mcpat(m_config, getShaderCoreConfig(), m_gpgpusim_wrapper,
                  m_power_stats, m_config.gpu_stat_sample_freq,
                  gpu_tot_sim_cycle, gpu_sim_cycle, gpu_tot_sim_insn,
                  gpu_sim_insn, m_config.g_power_simulation_mode, m_config.g_dvfs_enabled, 
                  m_config.g_hw_perf_file_name, m_config.g_hw_perf_bench_name, executed_kernel_name(), m_config.accelwattch_hybrid_configuration, m_config.g_aggregate_power_stats);
    }
    m_gpgpusim_wrapper->print_power_kernel_stats(
        gpu_sim_cycle, gpu_tot_sim_cycle, gpu_tot_sim_insn + gpu_sim_insn,
        kernel_info_str, true);

    // MOD. Begin. Energy
    double gpu_tot_energy_avg, rf_energy_average, execution_time;
    execution_time = m_config.get_core_period() * (gpu_tot_sim_cycle + gpu_sim_cycle);
    gpu_tot_energy_avg = m_gpgpusim_wrapper->get_gpu_tot_power_avg() * execution_time;
    rf_energy_average = m_gpgpusim_wrapper->get_register_file_power_avg() * execution_time;
    printf("gpu_tot_avg_energy = %.9f\n", gpu_tot_energy_avg);
    printf("gpu_register_file_avg_energy = %.9f\n", rf_energy_average);
    // MOD. End. Energy

    //if(!m_config.g_aggregate_power_stats)
      mcpat_reset_perf_count(m_gpgpusim_wrapper);
  }
#endif

  dram_tot_sim_cycle += dram_sim_cycle;
  // performance counter that are not local to one shader
  for(unsigned int i = 0; i < m_memory_config->m_n_mem; i++) {
    m_memory_stats->add(m_memory_partition_unit[i]->get_memory_partition_stats());
    m_memory_partition_unit[i]->get_memory_partition_stats()->reset();
  }
  m_memory_stats->memlatstat_print(m_memory_config->m_n_mem,
                                   m_memory_config->nbk);
  for (unsigned i = 0; i < m_memory_config->m_n_mem; i++)
    m_memory_partition_unit[i]->print(stdout);

  // L2 cache stats
  if (!m_memory_config->m_L2_config.disabled()) {
    cache_stats l2_stats;
    struct cache_sub_stats l2_css;
    struct cache_sub_stats total_l2_css;
    l2_stats.clear();
    l2_css.clear();
    total_l2_css.clear();

    printf("\n========= L2 cache stats =========\n");
    for (unsigned i = 0; i < m_memory_config->m_n_mem_sub_partition; i++) {
      m_memory_sub_partition[i]->accumulate_L2cache_stats(l2_stats);
      m_memory_sub_partition[i]->get_L2cache_sub_stats(l2_css);

      fprintf(stdout,
              "L2_cache_bank[%d]: Access = %llu, Miss = %llu, Miss_rate = "
              "%.3lf, Pending_hits = %llu, Reservation_fails = %llu\n",
              i, l2_css.accesses, l2_css.misses,
              (double)l2_css.misses / (double)l2_css.accesses,
              l2_css.pending_hits, l2_css.res_fails);

      total_l2_css += l2_css;
    }
    if (!m_memory_config->m_L2_config.disabled() &&
        m_memory_config->m_L2_config.get_num_lines()) {
      // L2c_print_cache_stat();
      printf("L2_total_cache_accesses = %llu\n", total_l2_css.accesses);
      printf("L2_total_cache_misses = %llu\n", total_l2_css.misses);
      if (total_l2_css.accesses > 0)
        printf("L2_total_cache_miss_rate = %.4lf\n",
               (double)total_l2_css.misses / (double)total_l2_css.accesses);
      printf("L2_total_cache_pending_hits = %llu\n", total_l2_css.pending_hits);
      printf("L2_total_cache_reservation_fails = %llu\n",
             total_l2_css.res_fails);
      printf("L2_total_cache_breakdown:\n");
      l2_stats.print_stats(stdout, "L2_cache_stats_breakdown");
      printf("L2_total_cache_reservation_fail_breakdown:\n");
      l2_stats.print_fail_stats(stdout, "L2_cache_stats_fail_breakdown");
      total_l2_css.print_port_stats(stdout, "L2_cache");
    }
  }

  if (m_config.gpgpu_cflog_interval != 0) {
    spill_log_to_file(stdout, 1, gpu_sim_cycle);
    insn_warp_occ_print(stdout);
  }
  if (gpgpu_ctx->func_sim->gpgpu_ptx_instruction_classification) {
    StatDisp(gpgpu_ctx->func_sim->g_inst_classification_stat
                 [gpgpu_ctx->func_sim->g_ptx_kernel_count]);
    StatDisp(gpgpu_ctx->func_sim->g_inst_op_classification_stat
                 [gpgpu_ctx->func_sim->g_ptx_kernel_count]);
  }

#ifdef GPGPUSIM_POWER_MODEL
  if (m_config.g_power_simulation_enabled) {
    m_gpgpusim_wrapper->detect_print_steady_state(
        1, gpu_tot_sim_insn + gpu_sim_insn);
  }
#endif

  // Interconnect power stat print
  long total_simt_to_mem = 0;
  long total_mem_to_simt = 0;
  long temp_stm = 0;
  long temp_mts = 0;
  for (unsigned i = 0; i < m_config.num_cluster(); i++) {
    m_cluster[i]->get_icnt_stats(temp_stm, temp_mts);
    total_simt_to_mem += temp_stm;
    total_mem_to_simt += temp_mts;
  }
  printf("\nicnt_total_pkts_mem_to_simt=%ld\n", total_mem_to_simt);
  printf("icnt_total_pkts_simt_to_mem=%ld\n", total_simt_to_mem);

  time_vector_print();
  fflush(stdout);

  clear_executed_kernel_info();
  reset_gpu_per_sm_stats();
}

// performance counter that are not local to one shader
unsigned gpgpu_sim::threads_per_core() const {
  return m_shader_config->n_thread_per_shader;
}

void shader_core_ctx::mem_instruction_stats(const warp_inst_t &inst) {
  unsigned active_count = inst.active_count();
  // this breaks some encapsulation: the is_[space] functions, if you change
  // those, change this.
  switch (inst.space.get_type()) {
    case undefined_space:
    case reg_space:
      break;
    case shared_space:
      m_stats->gpgpu_n_shmem_insn += active_count;
      break;
    case sstarr_space:
      m_stats->gpgpu_n_sstarr_insn += active_count;
      break;
    case const_space:
      m_stats->gpgpu_n_const_insn += active_count;
      break;
    case param_space_kernel:
    case param_space_local:
      m_stats->gpgpu_n_param_insn += active_count;
      break;
    case tex_space:
      m_stats->gpgpu_n_tex_insn += active_count;
      break;
    case global_space:
    case local_space:
      if (inst.is_store())
        m_stats->gpgpu_n_store_insn += active_count;
      else
        m_stats->gpgpu_n_load_insn += active_count;
      break;
    default:
      abort();
  }
}
bool shader_core_ctx::can_issue_1block(kernel_info_t &kernel) {
  // Jin: concurrent kernels on one SM
  if (m_config->gpgpu_concurrent_kernel_sm) {
    if (m_config->max_cta(kernel) < 1) return false;

    return occupy_shader_resource_1block(kernel, false);
  } else {
    return (get_n_active_cta() < m_config->max_cta(kernel));
  }
}

int shader_core_ctx::find_available_hwtid(unsigned int cta_size, bool occupy) {
  unsigned int step;
  for (step = 0; step < m_config->n_thread_per_shader; step += cta_size) {
    unsigned int hw_tid;
    for (hw_tid = step; hw_tid < step + cta_size; hw_tid++) {
      if (m_occupied_hwtid.test(hw_tid)) break;
    }
    if (hw_tid == step + cta_size)  // consecutive non-active
      break;
  }
  if (step >= m_config->n_thread_per_shader)  // didn't find
    return -1;
  else {
    if (occupy) {
      for (unsigned hw_tid = step; hw_tid < step + cta_size; hw_tid++)
        m_occupied_hwtid.set(hw_tid);
    }
    return step;
  }
}

bool shader_core_ctx::occupy_shader_resource_1block(kernel_info_t &k,
                                                    bool occupy) {
  unsigned threads_per_cta = k.threads_per_cta();
  const class function_info *kernel = k.entry();
  unsigned int padded_cta_size = threads_per_cta;
  unsigned int warp_size = m_config->warp_size;
  if (padded_cta_size % warp_size)
    padded_cta_size = ((padded_cta_size / warp_size) + 1) * (warp_size);

  if (m_occupied_n_threads + padded_cta_size > m_config->n_thread_per_shader)
    return false;

  if (find_available_hwtid(padded_cta_size, false) == -1) return false;

  const struct gpgpu_ptx_sim_info *kernel_info = ptx_sim_kernel_info(kernel);

  if (m_occupied_shmem + kernel_info->smem > m_config->gpgpu_shmem_size)
    return false;

  unsigned int used_regs = padded_cta_size * ((kernel_info->regs + 3) & ~3);
  if (m_occupied_regs + used_regs > m_config->gpgpu_shader_registers)
    return false;

  if (m_occupied_ctas + 1 > m_config->max_cta_per_core) return false;

  if (occupy) {
    m_occupied_n_threads += padded_cta_size;
    m_occupied_shmem += kernel_info->smem;
    m_occupied_regs += (padded_cta_size * ((kernel_info->regs + 3) & ~3));
    m_occupied_ctas++;

    SHADER_DPRINTF(LIVENESS,
                   "GPGPU-Sim uArch: Occupied %u threads, %u shared mem, %u "
                   "registers, %u ctas, on shader %d\n",
                   m_occupied_n_threads, m_occupied_shmem, m_occupied_regs,
                   m_occupied_ctas, m_sid);
  }

  return true;
}

void shader_core_ctx::release_shader_resource_1block(unsigned hw_ctaid,
                                                     kernel_info_t &k) {
  if (m_config->gpgpu_concurrent_kernel_sm) {
    unsigned threads_per_cta = k.threads_per_cta();
    const class function_info *kernel = k.entry();
    unsigned int padded_cta_size = threads_per_cta;
    unsigned int warp_size = m_config->warp_size;
    if (padded_cta_size % warp_size)
      padded_cta_size = ((padded_cta_size / warp_size) + 1) * (warp_size);

    assert(m_occupied_n_threads >= padded_cta_size);
    m_occupied_n_threads -= padded_cta_size;

    int start_thread = m_occupied_cta_to_hwtid[hw_ctaid];

    for (unsigned hwtid = start_thread; hwtid < start_thread + padded_cta_size;
         hwtid++)
      m_occupied_hwtid.reset(hwtid);
    m_occupied_cta_to_hwtid.erase(hw_ctaid);

    const struct gpgpu_ptx_sim_info *kernel_info = ptx_sim_kernel_info(kernel);

    assert(m_occupied_shmem >= (unsigned int)kernel_info->smem);
    m_occupied_shmem -= kernel_info->smem;

    unsigned int used_regs = padded_cta_size * ((kernel_info->regs + 3) & ~3);
    assert(m_occupied_regs >= used_regs);
    m_occupied_regs -= used_regs;

    assert(m_occupied_ctas >= 1);
    m_occupied_ctas--;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * Launches a cooperative thread array (CTA).
 *
 * @param kernel
 *    object that tells us which kernel to ask for a CTA from
 */

unsigned exec_shader_core_ctx::sim_init_thread(
    kernel_info_t &kernel, ptx_thread_info **thread_info, int sid, unsigned tid,
    unsigned threads_left, unsigned num_threads, core_t *core,
    unsigned hw_cta_id, unsigned hw_warp_id, gpgpu_t *gpu) {
  return ptx_sim_init_thread(kernel, thread_info, sid, tid, threads_left,
                             num_threads, core, hw_cta_id, hw_warp_id, gpu);
}

void shader_core_ctx::issue_block2core(kernel_info_t &kernel) {
  if (!m_config->gpgpu_concurrent_kernel_sm)
    set_max_cta(kernel);
  else
    assert(occupy_shader_resource_1block(kernel, true));

  kernel.inc_running();

  // find a free CTA context
  unsigned free_cta_hw_id = (unsigned)-1;

  unsigned max_cta_per_core;
  if (!m_config->gpgpu_concurrent_kernel_sm)
    max_cta_per_core = kernel_max_cta_per_shader;
  else
    max_cta_per_core = m_config->max_cta_per_core;
  for (unsigned i = 0; i < max_cta_per_core; i++) {
    if (m_cta_status[i] == 0) {
      free_cta_hw_id = i;
      break;
    }
  }
  assert(free_cta_hw_id != (unsigned)-1);

  // determine hardware threads and warps that will be used for this CTA
  int cta_size = kernel.threads_per_cta();

  // hw warp id = hw thread id mod warp size, so we need to find a range
  // of hardware thread ids corresponding to an integral number of hardware
  // thread ids
  int padded_cta_size = cta_size;
  if (cta_size % m_config->warp_size)
    padded_cta_size =
        ((cta_size / m_config->warp_size) + 1) * (m_config->warp_size);

  unsigned int start_thread, end_thread;

  if (!m_config->gpgpu_concurrent_kernel_sm) {
    start_thread = free_cta_hw_id * padded_cta_size;
    end_thread = start_thread + cta_size;
  } else {
    start_thread = find_available_hwtid(padded_cta_size, true);
    assert((int)start_thread != -1);
    end_thread = start_thread + cta_size;
    assert(m_occupied_cta_to_hwtid.find(free_cta_hw_id) ==
           m_occupied_cta_to_hwtid.end());
    m_occupied_cta_to_hwtid[free_cta_hw_id] = start_thread;
  }

  // reset the microarchitecture state of the selected hardware thread and warp
  // contexts
  reinit(start_thread, end_thread, false);

  // initalize scalar threads and determine which hardware warps they are
  // allocated to bind functional simulation state of threads to hardware
  // resources (simulation)
  warp_set_t warps;
  unsigned nthreads_in_block = 0;
  function_info *kernel_func_info = kernel.entry();
  symbol_table *symtab = kernel_func_info->get_symtab();
  unsigned ctaid = kernel.get_next_cta_id_single();
  checkpoint *g_checkpoint = new checkpoint();
  for (unsigned i = start_thread; i < end_thread; i++) {
    m_threadState[i].m_cta_id = free_cta_hw_id;
    unsigned warp_id = i / m_config->warp_size;
    nthreads_in_block += sim_init_thread(
        kernel, &m_thread[i], m_sid, i, cta_size - (i - start_thread),
        m_config->n_thread_per_shader, this, free_cta_hw_id, warp_id,
        m_cluster->get_gpu());
    m_threadState[i].m_active = true;
    // load thread local memory and register file
    if (m_gpu->resume_option == 1 && kernel.get_uid() == m_gpu->resume_kernel &&
        ctaid >= m_gpu->resume_CTA && ctaid < m_gpu->checkpoint_CTA_t) {
      char fname[2048];
      snprintf(fname, 2048, "checkpoint_files/thread_%d_%d_reg.txt",
               i % cta_size, ctaid);
      m_thread[i]->resume_reg_thread(fname, symtab);
      char f1name[2048];
      snprintf(f1name, 2048, "checkpoint_files/local_mem_thread_%d_%d_reg.txt",
               i % cta_size, ctaid);
      g_checkpoint->load_global_mem(m_thread[i]->m_local_mem, f1name);
    }
    //
    warps.set(warp_id);
  }
  assert(nthreads_in_block > 0 &&
         nthreads_in_block <=
             m_config->n_thread_per_shader);  // should be at least one, but
                                              // less than max
  m_cta_status[free_cta_hw_id] = nthreads_in_block;

  if (m_gpu->resume_option == 1 && kernel.get_uid() == m_gpu->resume_kernel &&
      ctaid >= m_gpu->resume_CTA && ctaid < m_gpu->checkpoint_CTA_t) {
    char f1name[2048];
    snprintf(f1name, 2048, "checkpoint_files/shared_mem_%d.txt", ctaid);

    g_checkpoint->load_global_mem(m_thread[start_thread]->m_shared_mem, f1name);
  }
  // now that we know which warps are used in this CTA, we can allocate
  // resources for use in CTA-wide barrier operations
  m_barriers.allocate_barrier(free_cta_hw_id, warps);

  // initialize the SIMT stacks and fetch hardware
  init_warps(free_cta_hw_id, start_thread, end_thread, ctaid, cta_size, kernel);
  m_n_active_cta++;

  shader_CTA_count_log(m_sid, 1);
  SHADER_DPRINTF(LIVENESS,
                 "GPGPU-Sim uArch: cta:%2u, start_tid:%4u, end_tid:%4u, "
                 "initialized @(%lld,%lld), kernel_uid:%u, kernel_name:%s\n",
                 free_cta_hw_id, start_thread, end_thread, m_gpu->gpu_sim_cycle,
                 m_gpu->gpu_tot_sim_cycle, kernel.get_uid(), kernel.get_name().c_str());
}

///////////////////////////////////////////////////////////////////////////////////////////

void dram_t::dram_log(int task) {
  if (task == SAMPLELOG) {
    StatAddSample(mrqq_Dist, que_length());
  } else if (task == DUMPLOG) {
    printf("Queue Length DRAM[%d] ", id);
    StatDisp(mrqq_Dist);
  }
}

// Find next clock domain and increment its time
int gpgpu_sim::next_clock_domain(void) {
  double smallest = min3(core_time, icnt_time, dram_time);
  int mask = 0x00;
  if (l2_time <= smallest) {
    smallest = l2_time;
    mask |= L2;
    l2_time += m_config.l2_period;
  }
  if (icnt_time <= smallest) {
    mask |= ICNT;
    icnt_time += m_config.icnt_period;
  }
  if (dram_time <= smallest) {
    mask |= DRAM;
    dram_time += m_config.dram_period;
  }
  if (core_time <= smallest) {
    mask |= CORE;
    core_time += m_config.core_period;
  }
  return mask;
}

void gpgpu_sim::issue_block2core() {
  unsigned last_issued = m_last_cluster_issue;
  for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
    unsigned idx = (i + last_issued + 1) % m_shader_config->n_simt_clusters;
    unsigned num = m_cluster[idx]->issue_block2core();
    if (num) {
      m_last_cluster_issue = idx;
      m_total_cta_launched += num;
    }
  }
}

unsigned long long g_single_step =
    0;  // set this in gdb to single step the pipeline



std::unique_ptr<grid_barrier_notify_info> gpgpu_sim::register_grid_barrier_arrivement(mem_fetch *mf) {
  std::unique_ptr<grid_barrier_notify_info> notifcation_res = nullptr;
  unsigned int kernel_id = mf->get_kernel_id();
  assert(m_grid_barrier_status.find(kernel_id) != m_grid_barrier_status.end());
  if(!m_grid_barrier_status[kernel_id].active) {
    m_grid_barrier_status[kernel_id].active = true;
  }
  m_grid_barrier_status[kernel_id].sm_ids_to_notify.insert(mf->get_sid());
  m_grid_barrier_status[kernel_id].num_threads_arrived += mf->get_inst().active_count();
  if(m_grid_barrier_status[kernel_id].barrier_completed()) {
    m_grid_barrier_status[kernel_id].active = false;
    m_grid_barrier_status[kernel_id].num_threads_arrived = 0;
    notifcation_res = std::make_unique<grid_barrier_notify_info>(kernel_id, m_grid_barrier_status[kernel_id].sm_ids_to_notify);
    m_grid_barrier_status[kernel_id].sm_ids_to_notify.clear();
  }
  delete mf;
  return notifcation_res;
}

void gpgpu_sim::increase_num_threads_kernel(unsigned kernel_id, unsigned num_threads) {
  assert(m_grid_barrier_status.find(kernel_id) != m_grid_barrier_status.end());
  m_grid_barrier_status[kernel_id].num_threads_kernel += num_threads;
}

void gpgpu_sim::decrease_num_threads_kernel(unsigned kernel_id, unsigned num_threads) {
  assert(m_grid_barrier_status.find(kernel_id) != m_grid_barrier_status.end());
  assert(m_grid_barrier_status[kernel_id].num_threads_kernel >= num_threads);
  m_grid_barrier_status[kernel_id].num_threads_kernel -= num_threads;
}


void gpgpu_sim::cycle() {
  m_active_sms_this_cycle = 0;
  m_current_cycle_clock_mask = next_clock_domain();
  if (m_current_cycle_clock_mask & CORE) {
    // shader core loading (pop from ICNT into core) follows CORE clock
    for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
      m_cluster[i]->icnt_cycle();
    }
    m_power_stats->pwr_mem_stat->core_cache_stats[CURRENT_STAT_IDX].clear();
  }
  unsigned partiton_replys_in_parallel_per_cycle = 0;
  if (m_current_cycle_clock_mask & ICNT) {
    // pop from grid barrier notify queue
    if(!m_grid_barrier_notify_queue.empty()) {
      auto it_sm_ids_to_notify = m_grid_barrier_notify_queue.front()->sm_ids_to_notify.begin();
      while(it_sm_ids_to_notify != m_grid_barrier_notify_queue.front()->sm_ids_to_notify.end()) {
        auto sm_id = *it_sm_ids_to_notify;
        mem_access_t res_acc(gpgpu_ctx);
        res_acc.set_space(miscellaneous_space);
        res_acc.set_write(false);
        res_acc.set_last_access(true);
        res_acc.set_size(32);
        res_acc.set_type(GRID_BARRIER_ACC);
        mem_fetch *mf = new mem_fetch(res_acc, nullptr, 0, 0, sm_id, sm_id, m_memory_config, gpu_sim_cycle + gpu_tot_sim_cycle);
        mf->set_reply();
        mf->set_kernel_id(m_grid_barrier_notify_queue.front()->kernel_id);
        if(::icnt_has_buffer(m_shader_config->mem2device(0), mf->size(), 0)) {
          mf->set_return_timestamp(gpu_sim_cycle + gpu_tot_sim_cycle);
          mf->set_status(IN_ICNT_TO_SHADER, gpu_sim_cycle + gpu_tot_sim_cycle);
          ::icnt_push(m_shader_config->mem2device(0), mf->get_tpc(), mf,
                      mf->size(), 0);
          m_memory_sub_partition[0]->pop();
          it_sm_ids_to_notify = m_grid_barrier_notify_queue.front()->sm_ids_to_notify.erase(it_sm_ids_to_notify);
        }else{
          gpu_stall_icnt2sh++;
          ++it_sm_ids_to_notify;
        }
      }
      if(m_grid_barrier_notify_queue.front()->sm_ids_to_notify.empty()) {
        m_grid_barrier_notify_queue.pop();
      }
    }
    for (unsigned i = 0; i < m_memory_config->m_n_mem_sub_partition; i++) {
      mem_fetch *mf = m_memory_sub_partition[i]->top();
      if (mf) {
        if(m_shader_config->is_const_cache_accessed_blocks_tracking_enabled && (mf->get_access_type() == CONST_ACC_R)) {
          m_shader_stats->all_const_cache_accessed_blocks.insert(mf->get_addr());
        }
        if(m_shader_config->is_num_virtual_pages_tracking_enabled) {
          unsigned int id_page = mf->get_addr() / m_shader_config->virtual_page_size_in_bytes;
          m_shader_stats->all_virtual_pages_accessed.insert(id_page);
        }
        unsigned response_size =
            mf->get_is_write() ? mf->get_ctrl_size() : mf->size();
        if (::icnt_has_buffer(m_shader_config->mem2device(i), response_size, 0)) {
          // if (!mf->get_is_write())
          mf->set_return_timestamp(gpu_sim_cycle + gpu_tot_sim_cycle);
          mf->set_status(IN_ICNT_TO_SHADER, gpu_sim_cycle + gpu_tot_sim_cycle);
          ::icnt_push(m_shader_config->mem2device(i), mf->get_tpc(), mf,
                      response_size, 0);
          m_memory_sub_partition[i]->pop();
          partiton_replys_in_parallel_per_cycle++;
        } else {
          gpu_stall_icnt2sh++;
        }
      } else {
        m_memory_sub_partition[i]->pop();
      }
    }
  }
  partiton_replys_in_parallel += partiton_replys_in_parallel_per_cycle;

  if (m_current_cycle_clock_mask & DRAM) {
    #pragma omp parallel for
    for (unsigned i = 0; i < m_memory_config->m_n_mem; i++) {
      if (m_memory_config->simple_dram_model)
        m_memory_partition_unit[i]->simple_dram_model_cycle();
      else
        m_memory_partition_unit[i]
            ->dram_cycle();  // Issue the dram command (scheduler + delay model)
      // Update performance counters for DRAM
      if(m_config.g_power_simulation_enabled && (((gpu_tot_sim_cycle + gpu_sim_cycle) + 1) % m_config.gpu_stat_sample_freq == 0)) {
        m_memory_partition_unit[i]->set_dram_power_stats(
          m_power_stats->pwr_mem_stat->n_cmd[CURRENT_STAT_IDX][i],
          m_power_stats->pwr_mem_stat->n_activity[CURRENT_STAT_IDX][i],
          m_power_stats->pwr_mem_stat->n_nop[CURRENT_STAT_IDX][i],
          m_power_stats->pwr_mem_stat->n_act[CURRENT_STAT_IDX][i],
          m_power_stats->pwr_mem_stat->n_pre[CURRENT_STAT_IDX][i],
          m_power_stats->pwr_mem_stat->n_rd[CURRENT_STAT_IDX][i],
          m_power_stats->pwr_mem_stat->n_wr[CURRENT_STAT_IDX][i],
          m_power_stats->pwr_mem_stat->n_wr_WB[CURRENT_STAT_IDX][i],
          m_power_stats->pwr_mem_stat->n_req[CURRENT_STAT_IDX][i]);
      }
    }
    dram_sim_cycle++;
  }

  // L2 operations follow L2 clock domain
  unsigned partiton_reqs_in_parallel_per_cycle = 0;
  if (m_current_cycle_clock_mask & L2) {
    m_power_stats->pwr_mem_stat->l2_cache_stats[CURRENT_STAT_IDX].clear();
    for (unsigned i = 0; i < m_memory_config->m_n_mem_sub_partition; i++) {
      // move memory request from interconnect into memory partition (if not
      // backed up) Note:This needs to be called in DRAM clock domain if there
      // is no L2 cache in the system In the worst case, we may need to push
      // SECTOR_CHUNCK_SIZE requests, so ensure you have enough buffer for them
      if (m_memory_sub_partition[i]->full(SECTOR_CHUNCK_SIZE)) {
        gpu_stall_dramfull++;
      } else {
        mem_fetch *mf = (mem_fetch *)icnt_pop(m_shader_config->mem2device(i), 0);
        if(mf) {
          if(mf->get_inst().op == GRID_BARRIER_OP) {
            std::unique_ptr<grid_barrier_notify_info> notifcation_res = register_grid_barrier_arrivement(mf);
            mf = nullptr;
            if(notifcation_res) {
              m_grid_barrier_notify_queue.push(std::move(notifcation_res));
            }
          }else {
            partiton_reqs_in_parallel_per_cycle++;
          }
        }
        m_memory_sub_partition[i]->push(mf, gpu_sim_cycle + gpu_tot_sim_cycle);
      }
      m_memory_sub_partition[i]->cache_cycle(gpu_sim_cycle + gpu_tot_sim_cycle);
      if(m_config.g_power_simulation_enabled && (((gpu_tot_sim_cycle + gpu_sim_cycle) + 1) % m_config.gpu_stat_sample_freq == 0)) {
        m_memory_sub_partition[i]->accumulate_L2cache_stats(
            m_power_stats->pwr_mem_stat->l2_cache_stats[CURRENT_STAT_IDX]);
      }
    }
  }
  partiton_reqs_in_parallel += partiton_reqs_in_parallel_per_cycle;
  if (partiton_reqs_in_parallel_per_cycle > 0) {
    partiton_reqs_in_parallel_util += partiton_reqs_in_parallel_per_cycle;
    gpu_sim_cycle_parition_util++;
  }

  if (m_current_cycle_clock_mask & ICNT) {
    icnt_transfer(0);
  }

  if (m_current_cycle_clock_mask & CORE) {
    // L1 cache + shader core pipeline stages
    #pragma omp parallel for schedule(runtime) reduction(+:m_active_sms_this_cycle)
    for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
      if (m_cluster[i]->get_not_completed() || get_more_cta_left()) {
       m_cluster[i]->core_cycle();
      }
      // Update core icnt/cache stats for AccelWattch
      if(m_config.g_power_simulation_enabled && (((gpu_tot_sim_cycle + gpu_sim_cycle) + 1) % m_config.gpu_stat_sample_freq == 0)) {
        m_cluster[i]->get_icnt_stats(
            m_power_stats->pwr_mem_stat->n_simt_to_mem[CURRENT_STAT_IDX][i],
            m_power_stats->pwr_mem_stat->n_mem_to_simt[CURRENT_STAT_IDX][i]);
      }
      m_active_sms_this_cycle += m_cluster[i]->get_n_active_sms();
    }
      float temp = 0;
      float previous_active_sms = *active_sms;
      for (unsigned i = 0; i < m_shader_config->num_shader(); i++) {
        m_cluster[i]->get_current_occupancy(
            gpu_occupancy.aggregate_warp_slot_filled,
            gpu_occupancy.aggregate_theoretical_warp_slots);
        temp += m_shader_stats->m_pipeline_duty_cycle[i];
        // active_sms_this_cycle += m_cluster[i]->get_n_active_sms();
        // Update core icnt/cache stats for AccelWattch
        if(m_config.g_power_simulation_enabled && (((gpu_tot_sim_cycle + gpu_sim_cycle) + 1) % m_config.gpu_stat_sample_freq == 0)) {
          m_cluster[i]->get_cache_stats(
              m_power_stats->pwr_mem_stat->core_cache_stats[CURRENT_STAT_IDX]);
        }
      }
      *active_sms += m_active_sms_this_cycle;
      if(previous_active_sms != *active_sms){
        total_sms_accumulated_across_cycles += m_shader_config->num_shader();
      }

      if(m_shader_config->is_custom_omp_scheduler_enabled) {/////////////////CAMBIAR A ARRIBA DESPUES DE CALCULAR
        omp_sched_t next_omp_scheduler = omp_sched_static;
        bool is_gpu_action = (m_active_sms_this_cycle > 0);
        float ratio_active = m_active_sms_this_cycle / m_shader_config->num_shader();
        if ((ratio_active < m_shader_config->custom_omp_scheduler_ratio_to_dynamic) && is_gpu_action) {
          next_omp_scheduler = omp_sched_dynamic;
        }
        if((m_current_omp_scheduler != next_omp_scheduler) && is_gpu_action) {
          m_current_omp_scheduler = next_omp_scheduler;
          omp_set_schedule(next_omp_scheduler, 1);
        }
      }
      
      temp = temp / m_shader_config->num_shader();
      *average_pipeline_duty_cycle = ((*average_pipeline_duty_cycle) + temp);
      // cout<<"Average pipeline duty cycle:
      // "<<*average_pipeline_duty_cycle<<endl;

      if (g_single_step &&
          ((gpu_sim_cycle + gpu_tot_sim_cycle) >= g_single_step)) {
        raise(SIGTRAP);  // Debug breakpoint
      }
      gpu_sim_cycle++;

      if (g_interactive_debugger_enabled) {
        gpgpu_debug();
      }

      // McPAT main cycle (interface with McPAT)
      #ifdef GPGPUSIM_POWER_MODEL
        if (m_config.g_power_simulation_enabled) {
          if(m_config.g_power_simulation_mode == 0){
          mcpat_cycle(m_config, getShaderCoreConfig(), m_gpgpusim_wrapper,
                      m_power_stats, m_config.gpu_stat_sample_freq,
                      gpu_tot_sim_cycle, gpu_sim_cycle, gpu_tot_sim_insn,
                      gpu_sim_insn, m_config.g_dvfs_enabled, 0);
          }
        }
      #endif
      
      issue_block2core();
      decrement_kernel_latency();

      // Depending on configuration, invalidate the caches once all of threads are
      // completed.
      int all_threads_complete = 1;
      if (m_config.gpgpu_flush_l1_cache) {
        for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
          if (m_cluster[i]->get_not_completed() == 0)
            m_cluster[i]->cache_invalidate();
          else
            all_threads_complete = 0;
        }
      }

      if (m_config.gpgpu_flush_l2_cache) {
        if (!m_config.gpgpu_flush_l1_cache) {
          for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
            if (m_cluster[i]->get_not_completed() != 0) {
              all_threads_complete = 0;
              break;
            }
          }
        }

        if (all_threads_complete && !m_memory_config->m_L2_config.disabled()) {
          // printf("Flushed L2 caches...\n");
          if (m_memory_config->m_L2_config.get_num_lines()) {
            int dlc = 0;
            for (unsigned i = 0; i < m_memory_config->m_n_mem; i++) {
              // dlc = m_memory_sub_partition[i]->flushL2();
              dlc = m_memory_sub_partition[i]->invalidateL2();
              assert(dlc == 0);  // TODO: need to model actual writes to DRAM here
              // printf("Dirty lines flushed from L2 %d is %d\n", i, dlc);
            }
          }
        }
      }

      if (!(gpu_sim_cycle % m_config.gpu_stat_sample_freq)) {
        time_t days, hrs, minutes, sec;
        time_t curr_time;
        time(&curr_time);
        unsigned long long elapsed_time =
            MAX(curr_time - gpgpu_ctx->the_gpgpusim->g_simulation_starttime, 1);
        if ((elapsed_time - last_liveness_message_time) >=
                m_config.liveness_message_freq &&
            DTRACE(LIVENESS)) {
          days = elapsed_time / (3600 * 24);
          hrs = elapsed_time / 3600 - 24 * days;
          minutes = elapsed_time / 60 - 60 * (hrs + 24 * days);
          sec = elapsed_time - 60 * (minutes + 60 * (hrs + 24 * days));

          unsigned long long active = 0, total = 0;
          for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
            m_cluster[i]->get_current_occupancy(active, total);
          }
          DPRINTFG(LIVENESS,
                  "uArch: inst.: %lld (ipc=%4.1f, occ=%0.4f%% [%llu / %llu]) "
                  "sim_rate=%u (inst/sec) elapsed = %u:%u:%02u:%02u / %s",
                  gpu_tot_sim_insn + gpu_sim_insn,
                  (double)gpu_sim_insn / (double)gpu_sim_cycle,
                  float(active) / float(total) * 100, active, total,
                  (unsigned)((gpu_tot_sim_insn + gpu_sim_insn) / elapsed_time),
                  (unsigned)days, (unsigned)hrs, (unsigned)minutes,
                  (unsigned)sec, ctime(&curr_time));
          fflush(stdout);
          last_liveness_message_time = elapsed_time;
        }
        visualizer_printstat();
        m_memory_stats->memlatstat_lat_pw();
        if (m_config.gpgpu_runtime_stat &&
            (m_config.gpu_runtime_stat_flag != 0)) {
          if (m_config.gpu_runtime_stat_flag & GPU_RSTAT_BW_STAT) {
            for (unsigned i = 0; i < m_memory_config->m_n_mem; i++)
              m_memory_partition_unit[i]->print_stat(stdout);
            printf("maxmrqlatency = %d \n", m_memory_stats->max_mrq_latency);
            printf("maxmflatency = %d \n", m_memory_stats->max_mf_latency);
          }
          if (m_config.gpu_runtime_stat_flag & GPU_RSTAT_SHD_INFO)
            shader_print_runtime_stat(stdout);
          if (m_config.gpu_runtime_stat_flag & GPU_RSTAT_L1MISS)
            shader_print_l1_miss_stat(stdout);
          if (m_config.gpu_runtime_stat_flag & GPU_RSTAT_SCHED)
            shader_print_scheduler_stat(stdout, false);
        }
      }

      if (!(gpu_sim_cycle % 100000)) {
        // deadlock detection
        gather_gpu_per_sm_single_stat("gpu_sim_insn");
        gpu_sim_insn = m_gpu_per_sm_stats.m_stats_map["gpu_sim_insn"]->get_value();
        if (m_config.gpu_deadlock_detect && ((gpu_tot_sim_insn + gpu_sim_insn) == last_gpu_sim_insn) ) { // MOD. More accurate deadlock detection
          gpu_deadlock = true;
        } else {
          last_gpu_sim_insn = gpu_sim_insn + gpu_tot_sim_insn; // MOD. More accurate deadlock detection
        }
      }
      try_snap_shot(gpu_sim_cycle);
      spill_log_to_file(stdout, 0, gpu_sim_cycle);

      #if (CUDART_VERSION >= 5000)
        // launch device kernel
        gpgpu_ctx->device_runtime->launch_one_device_kernel();
      #endif
  }
}

void shader_core_ctx::dump_warp_state(FILE *fout) const {
  fprintf(fout, "\n");
  fprintf(fout, "per warp functional simulation status:\n");
  for (unsigned w = 0; w < m_config->max_warps_per_shader; w++)
    m_warp[w]->print(fout);
}

void gpgpu_sim::perf_memcpy_to_gpu(size_t dst_start_addr, size_t count) {
  if (m_memory_config->m_perf_sim_memcpy) {
    // if(!m_config.trace_driven_mode)    //in trace-driven mode, CUDA runtime
    // can start nre data structure at any position 	assert (dst_start_addr %
    // 32
    //== 0);

    for (unsigned counter = 0; counter < count; counter += 32) {
      const unsigned wr_addr = dst_start_addr + counter;
      addrdec_t raw_addr;
      mem_access_sector_mask_t mask;
      mask.set(wr_addr % 128 / 32);
      m_memory_config->m_address_mapping.addrdec_tlx(wr_addr, &raw_addr);
      const unsigned partition_id =
          raw_addr.sub_partition /
          m_memory_config->m_n_sub_partition_per_memory_channel;
      m_memory_partition_unit[partition_id]->handle_memcpy_to_gpu(
          wr_addr, raw_addr.sub_partition, mask);
    }
  }
}

void gpgpu_sim::dump_pipeline(int mask, int s, int m) const {
  /*
     You may want to use this function while running GPGPU-Sim in gdb.
     One way to do that is add the following to your .gdbinit file:

        define dp
           call g_the_gpu.dump_pipeline_impl((0x40|0x4|0x1),$arg0,0)
        end

     Then, typing "dp 3" will show the contents of the pipeline for shader
     core 3.
  */

  printf("Dumping pipeline state...\n");
  if (!mask) mask = 0xFFFFFFFF;
  for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
    if (s != -1) {
      i = s;
    }
    if (mask & 1)
      m_cluster[m_shader_config->sid_to_cluster(i)]->display_pipeline(
          i, stdout, 1, mask & 0x2E);
    if (s != -1) {
      break;
    }
  }
  if (mask & 0x10000) {
    for (unsigned i = 0; i < m_memory_config->m_n_mem; i++) {
      if (m != -1) {
        i = m;
      }
      printf("DRAM / memory controller %u:\n", i);
      if (mask & 0x100000) m_memory_partition_unit[i]->print_stat(stdout);
      if (mask & 0x1000000) m_memory_partition_unit[i]->visualize();
      if (mask & 0x10000000) m_memory_partition_unit[i]->print(stdout);
      if (m != -1) {
        break;
      }
    }
  }
  fflush(stdout);
}

const shader_core_config *gpgpu_sim::getShaderCoreConfig() {
  return m_shader_config;
}

const memory_config *gpgpu_sim::getMemoryConfig() { return m_memory_config; }

simt_core_cluster *gpgpu_sim::getSIMTCluster() { return *m_cluster; }
