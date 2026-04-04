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



// Copyright (c) 2009-2021, Tor M. Aamodt, Wilson W.L. Fung, Vijay Kandiah, Nikos Hardavellas
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


#ifndef GPU_SIM_H
#define GPU_SIM_H

#include <stdio.h>
#include <fstream>
#include <iostream>
#include <list>
#include "../abstract_hardware_model.h"
#include "../option_parser.h"
#include "../trace.h"
#include "../constants.h" // MOD. Do not duplicate somo constants declarations
#include "addrdec.h"
#include "gpu-cache.h"
#include "shader.h"

#include "remodeling/new_stats.h"

#include "remodeling/fusedMemory/coalescingStats.h"

#include "../../../../util/traces_enhanced/src/traced_execution.h" // MOD. Improved tracer

#include <omp.h>

// constants for statistics printouts
#define GPU_RSTAT_SHD_INFO 0x1
#define GPU_RSTAT_BW_STAT 0x2
#define GPU_RSTAT_WARP_DIS 0x4
#define GPU_RSTAT_DWF_MAP 0x8
#define GPU_RSTAT_L1MISS 0x10
#define GPU_RSTAT_PDOM 0x20
#define GPU_RSTAT_SCHED 0x40
#define GPU_MEMLATSTAT_MC 0x2

// constants for configuring merging of coalesced scatter-gather requests
#define TEX_MSHR_MERGE 0x4
#define CONST_MSHR_MERGE 0x2
#define GLOBAL_MSHR_MERGE 0x1

// clock constants
#define MhZ *1000000

#define CREATELOG 111
#define SAMPLELOG 222
#define DUMPLOG 333

class gpgpu_context;

extern tr1_hash_map<new_addr_type, unsigned> address_random_interleaving;

enum dram_ctrl_t { DRAM_FIFO = 0, DRAM_FRFCFS = 1 };

enum hw_perf_t {
  HW_BENCH_NAME=0,
  HW_KERNEL_NAME,
  HW_L1_RH,
  HW_L1_RM,
  HW_L1_WH,
  HW_L1_WM,
  HW_CC_ACC,
  HW_SHRD_ACC,
  HW_DRAM_RD,
  HW_DRAM_WR,
  HW_L2_RH,
  HW_L2_RM,
  HW_L2_WH,
  HW_L2_WM,
  HW_NOC,
  HW_PIPE_DUTY,
  HW_NUM_SM_IDLE,
  HW_CYCLES,
  HW_VOLTAGE,
  HW_TOTAL_STATS
};

struct power_config {
  power_config() { 
    m_valid = true;
    g_use_nonlinear_model = false;
    g_steady_state_tracking_filename = nullptr;
    g_power_trace_filename = nullptr;
    g_power_filename = nullptr;
    g_metric_trace_filename = nullptr;
  }

  ~power_config() {
    if (g_power_filename) {
      free(g_power_filename);
      g_power_filename = nullptr;
    }
    if (g_power_trace_filename) {
      free(g_power_trace_filename);
      g_power_trace_filename = nullptr;
    }
    if (g_metric_trace_filename) {
      free(g_metric_trace_filename);
      g_metric_trace_filename = nullptr;
    }
    if (g_steady_state_tracking_filename) {
      free(g_steady_state_tracking_filename);
      g_steady_state_tracking_filename = nullptr;
    }
  }

  void init() {
    // initialize file name if it is not set
    time_t curr_time;
    time(&curr_time);
    char *date = ctime(&curr_time);
    char *s = date;
    while (*s) {
      if (*s == ' ' || *s == '\t' || *s == ':') *s = '-';
      if (*s == '\n' || *s == '\r') *s = 0;
      s++;
    }
    char buf1[1024];
    //snprintf(buf1, 1024, "accelwattch_power_report__%s.log", date);
    snprintf(buf1, 1024, "accelwattch_power_report.log");

    if(g_power_filename != nullptr) {
      free(g_power_filename);
    }
    g_power_filename = strdup(buf1);
    char buf2[1024];
    snprintf(buf2, 1024, "gpgpusim_power_trace_report__%s.log.gz", date);
    if(g_power_trace_filename != nullptr) {
      free(g_power_trace_filename);
    }
    g_power_trace_filename = strdup(buf2);
    char buf3[1024];
    snprintf(buf3, 1024, "gpgpusim_metric_trace_report__%s.log.gz", date);
    if(g_metric_trace_filename != nullptr) {
      free(g_metric_trace_filename);
    }
    g_metric_trace_filename = strdup(buf3);
    char buf4[1024];
    snprintf(buf4, 1024, "gpgpusim_steady_state_tracking_report__%s.log.gz",
             date);
    if(g_steady_state_tracking_filename != nullptr) {
      free(g_steady_state_tracking_filename);
    }
    g_steady_state_tracking_filename = strdup(buf4);
    
    // for(int i =0; i< hw_perf_t::HW_TOTAL_STATS; i++){
    //   accelwattch_hybrid_configuration[i] = 0;
    // }

    if (g_steady_power_levels_enabled) {
      sscanf(gpu_steady_state_definition, "%lf:%lf",
             &gpu_steady_power_deviation, &gpu_steady_min_period);
    }

    // NOTE: After changing the nonlinear model to only scaling idle core,
    // NOTE: The min_inc_per_active_sm is not used any more
    if (g_use_nonlinear_model) {
      sscanf(gpu_nonlinear_model_config, "%lf:%lf", &gpu_idle_core_power,
             &gpu_min_inc_per_active_sm);
    }
  }
  void reg_options(class OptionParser *opp);

  char *g_power_config_name;

  bool m_valid;
  bool g_power_simulation_enabled;
  bool g_power_trace_enabled;
  bool g_steady_power_levels_enabled;
  bool g_power_per_cycle_dump;
  bool g_power_simulator_debug;
  char *g_power_filename;
  char *g_power_trace_filename;
  char *g_metric_trace_filename;
  char *g_steady_state_tracking_filename;
  int g_power_trace_zlevel;
  char *gpu_steady_state_definition;
  double gpu_steady_power_deviation;
  double gpu_steady_min_period;


  char *g_hw_perf_file_name;
  char *g_hw_perf_bench_name;
  int g_power_simulation_mode;
  bool g_dvfs_enabled;
  bool g_aggregate_power_stats;
  bool accelwattch_hybrid_configuration[hw_perf_t::HW_TOTAL_STATS];

  // Nonlinear power model
  bool g_use_nonlinear_model;
  char *gpu_nonlinear_model_config;
  double gpu_idle_core_power;
  double gpu_min_inc_per_active_sm;
};

class memory_config {
 public:
  memory_config(gpgpu_context *ctx) {
    m_valid = false;
    gpgpu_dram_timing_opt = NULL;
    gpgpu_L2_queue_config = NULL;
    gpgpu_ctx = ctx;
  }
  void init() {
    assert(gpgpu_dram_timing_opt);
    if (strchr(gpgpu_dram_timing_opt, '=') == NULL) {
      // dram timing option in ordered variables (legacy)
      // Disabling bank groups if their values are not specified
      nbkgrp = 1;
      tCCDL = 0;
      tRTPL = 0;
      sscanf(gpgpu_dram_timing_opt, "%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d",
             &nbk, &tCCD, &tRRD, &tRCD, &tRAS, &tRP, &tRC, &CL, &WL, &tCDLR,
             &tWR, &nbkgrp, &tCCDL, &tRTPL);
    } else {
      // named dram timing options (unordered)
      option_parser_t dram_opp = option_parser_create();

      option_parser_register(dram_opp, "nbk", OPT_UINT32, &nbk,
                             "number of banks", "");
      option_parser_register(dram_opp, "CCD", OPT_UINT32, &tCCD,
                             "column to column delay", "");
      option_parser_register(
          dram_opp, "RRD", OPT_UINT32, &tRRD,
          "minimal delay between activation of rows in different banks", "");
      option_parser_register(dram_opp, "RCD", OPT_UINT32, &tRCD,
                             "row to column delay", "");
      option_parser_register(dram_opp, "RAS", OPT_UINT32, &tRAS,
                             "time needed to activate row", "");
      option_parser_register(dram_opp, "RP", OPT_UINT32, &tRP,
                             "time needed to precharge (deactivate) row", "");
      option_parser_register(dram_opp, "RC", OPT_UINT32, &tRC, "row cycle time",
                             "");
      option_parser_register(dram_opp, "CDLR", OPT_UINT32, &tCDLR,
                             "switching from write to read (changes tWTR)", "");
      option_parser_register(dram_opp, "WR", OPT_UINT32, &tWR,
                             "last data-in to row precharge", "");

      option_parser_register(dram_opp, "CL", OPT_UINT32, &CL, "CAS latency",
                             "");
      option_parser_register(dram_opp, "WL", OPT_UINT32, &WL, "Write latency",
                             "");

      // Disabling bank groups if their values are not specified
      option_parser_register(dram_opp, "nbkgrp", OPT_UINT32, &nbkgrp,
                             "number of bank groups", "1");
      option_parser_register(
          dram_opp, "CCDL", OPT_UINT32, &tCCDL,
          "column to column delay between accesses to different bank groups",
          "0");
      option_parser_register(
          dram_opp, "RTPL", OPT_UINT32, &tRTPL,
          "read to precharge delay between accesses to different bank groups",
          "0");

      option_parser_delimited_string(dram_opp, gpgpu_dram_timing_opt, "=:;");
      fprintf(stdout, "DRAM Timing Options:\n");
      option_parser_print(dram_opp, stdout);
      option_parser_destroy(dram_opp);
    }

    int nbkt = nbk / nbkgrp;
    unsigned i;
    for (i = 0; nbkt > 0; i++) {
      nbkt = nbkt >> 1;
    }
    bk_tag_length = i - 1;
    assert(nbkgrp > 0 && "Number of bank groups cannot be zero");
    tRCDWR = tRCD - (WL + 1);
    if (elimnate_rw_turnaround) {
      tRTW = 0;
      tWTR = 0;
    } else {
      tRTW = (CL + (BL / data_command_freq_ratio) + 2 - WL);
      tWTR = (WL + (BL / data_command_freq_ratio) + tCDLR);
    }
    tWTP = (WL + (BL / data_command_freq_ratio) + tWR);
    dram_atom_size =
        BL * busW * gpu_n_mem_per_ctrlr;  // burst length x bus width x # chips
                                          // per partition

    assert(m_n_sub_partition_per_memory_channel > 0);
    assert((nbk % m_n_sub_partition_per_memory_channel == 0) &&
           "Number of DRAM banks must be a perfect multiple of memory sub "
           "partition");
    m_n_mem_sub_partition = m_n_mem * m_n_sub_partition_per_memory_channel;
    fprintf(stdout, "Total number of memory sub partition = %u\n",
            m_n_mem_sub_partition);

    m_address_mapping.init(m_n_mem, m_n_sub_partition_per_memory_channel);
    m_L2_config.init(&m_address_mapping);

    m_valid = true;

    sscanf(write_queue_size_opt, "%d:%d:%d",
           &gpgpu_frfcfs_dram_write_queue_size, &write_high_watermark,
           &write_low_watermark);
  }
  void reg_options(class OptionParser *opp);

  bool m_valid;
  mutable l2_cache_config m_L2_config;
  bool m_L2_texure_only;

  char *gpgpu_dram_timing_opt;
  char *gpgpu_L2_queue_config;
  bool l2_ideal;
  unsigned gpgpu_frfcfs_dram_sched_queue_size;
  unsigned gpgpu_dram_return_queue_size;
  enum dram_ctrl_t scheduler_type;
  bool gpgpu_memlatency_stat;
  unsigned m_n_mem;
  unsigned m_n_sub_partition_per_memory_channel;
  unsigned m_n_mem_sub_partition;
  unsigned gpu_n_mem_per_ctrlr;

  unsigned rop_latency;
  unsigned dram_latency;

  // DRAM parameters

  unsigned tCCDL;  // column to column delay when bank groups are enabled
  unsigned tRTPL;  // read to precharge delay when bank groups are enabled for
                   // GDDR5 this is identical to RTPS, if for other DRAM this is
                   // different, you will need to split them in two

  unsigned tCCD;    // column to column delay
  unsigned tRRD;    // minimal time required between activation of rows in
                    // different banks
  unsigned tRCD;    // row to column delay - time required to activate a row
                    // before a read
  unsigned tRCDWR;  // row to column delay for a write command
  unsigned tRAS;    // time needed to activate row
  unsigned tRP;     // row precharge ie. deactivate row
  unsigned
      tRC;  // row cycle time ie. precharge current, then activate different row
  unsigned tCDLR;  // Last data-in to Read command (switching from write to
                   // read)
  unsigned tWR;    // Last data-in to Row precharge

  unsigned CL;    // CAS latency
  unsigned WL;    // WRITE latency
  unsigned BL;    // Burst Length in bytes (4 in GDDR3, 8 in GDDR5)
  unsigned tRTW;  // time to switch from read to write
  unsigned tWTR;  // time to switch from write to read
  unsigned tWTP;  // time to switch from write to precharge in the same bank
  unsigned busW;

  unsigned nbkgrp;  // number of bank groups (has to be power of 2)
  unsigned
      bk_tag_length;  // number of bits that define a bank inside a bank group

  unsigned nbk;

  bool elimnate_rw_turnaround;

  unsigned
      data_command_freq_ratio;  // frequency ratio between DRAM data bus and
                                // command bus (2 for GDDR3, 4 for GDDR5)
  unsigned
      dram_atom_size;  // number of bytes transferred per read or write command

  linear_to_raw_address_translation m_address_mapping;

  unsigned icnt_flit_size;

  unsigned dram_bnk_indexing_policy;
  unsigned dram_bnkgrp_indexing_policy;
  bool dual_bus_interface;

  bool seperate_write_queue_enabled;
  char *write_queue_size_opt;
  unsigned gpgpu_frfcfs_dram_write_queue_size;
  unsigned write_high_watermark;
  unsigned write_low_watermark;
  bool m_perf_sim_memcpy;
  bool simple_dram_model;

  gpgpu_context *gpgpu_ctx;
};

extern bool g_interactive_debugger_enabled;

class gpgpu_sim_config : public power_config,
                         public gpgpu_functional_sim_config {
 public:
  gpgpu_sim_config(gpgpu_context *ctx)
      : m_shader_config(ctx), m_memory_config(ctx) {
    m_valid = false;
    gpgpu_ctx = ctx;
    g_visualizer_filename = nullptr;
  }
  ~gpgpu_sim_config() {
    if (g_visualizer_filename) {
      free(g_visualizer_filename);
      g_visualizer_filename = nullptr;
    }
  }
  void reg_options(class OptionParser *opp);
  void init() {
    gpu_stat_sample_freq = 10000;
    gpu_runtime_stat_flag = 0;
    sscanf(gpgpu_runtime_stat, "%d:%x", &gpu_stat_sample_freq,
           &gpu_runtime_stat_flag);
    m_shader_config.init();
    ptx_set_tex_cache_linesize(m_shader_config.m_L1T_config.get_line_sz());
    m_memory_config.init();
    init_clock_domains();
    power_config::init();
    Trace::init();

    // initialize file name if it is not set
    time_t curr_time;
    time(&curr_time);
    char *date = ctime(&curr_time);
    char *s = date;
    while (*s) {
      if (*s == ' ' || *s == '\t' || *s == ':') *s = '-';
      if (*s == '\n' || *s == '\r') *s = 0;
      s++;
    }
    char buf[1024];
    snprintf(buf, 1024, "gpgpusim_visualizer__%s.log.gz", date);
    if(g_visualizer_filename != nullptr) {
      free(g_visualizer_filename);
    }
    g_visualizer_filename = strdup(buf);

    m_valid = true;
  }
  unsigned get_core_freq() const { return core_freq; }
  double get_core_period() const { return core_period; } // MOD. Energy
  double get_dram_period() const { return dram_period; } // MOD. Enegy
  int get_gpu_stat_sample_freq() const { return gpu_stat_sample_freq;}; // MOD. Energy
  unsigned num_shader() const { return m_shader_config.num_shader(); }
  unsigned num_cluster() const { return m_shader_config.n_simt_clusters; }
  unsigned get_max_concurrent_kernel() const { return max_concurrent_kernel; }
  unsigned checkpoint_option;

  size_t stack_limit() const { return stack_size_limit; }
  size_t heap_limit() const { return heap_size_limit; }
  size_t sync_depth_limit() const { return runtime_sync_depth_limit; }
  size_t pending_launch_count_limit() const {
    return runtime_pending_launch_count_limit;
  }

  const shader_core_config& get_gpgpu_sim_config() const { return m_shader_config; }

  bool flush_l1() const { return gpgpu_flush_l1_cache; }

  void set_custom_options(bool is_trace_mode) {
    m_shader_config.is_trace_mode = is_trace_mode; // MOD. General Config Helper

    // MOD. Begin. Fix WAR at baseline.
    std::string scb_r_mode_config = m_shader_config.scoreboard_war_mode;
    m_shader_config.scoreboard_war_reads_mode = scb_r_mode_config.find("wb") != std::string::npos ? scoreboard_reads_mode::RELEASE_AT_WB
                                  : scb_r_mode_config.find("opc") != std::string::npos ? scoreboard_reads_mode::RELEASE_AT_OPC
                                  : scoreboard_reads_mode::DISABLED;
    // MOD. End

    // MOD. Begin. Added L0I
    if(m_shader_config.is_L0I_enabled) {
      if(!m_shader_config.is_fetch_and_decode_improved) {
        std::cout << "Error. If -is_L0I_enabled is set to 1, -is_fetch_and_decode_improved must be set to 1 too." << std::endl;
        fflush(stdout);
        abort();
      }
    }
    // MOD. End. Added L0I

    // MOD. Begin. Fix misaligned fetched instructions
    if(m_shader_config.is_fix_instruction_fetch_misalignment) {
      if(!m_shader_config.is_fetch_and_decode_improved) {
        std::cout << "Error. If -is_fix_instruction_fetch_misalignment is set to 1, -is_fetch_and_decode_improved must be set to 1 too." << std::endl;
        fflush(stdout);
        abort();
      }
    }
    // MOD. End. Fix misaligned fetched instructions

    // MOD. Begin. Instruction addresses of different kernels have a different address request in memory
    if(m_shader_config.is_fix_different_kernels_pc_addresses) {
      if(!m_shader_config.is_fetch_and_decode_improved) {
        std::cout << "Error. If -is_fix_different_kernels_pc_addresses is set to 1, -is_fetch_and_decode_improved must be set to 1 too." << std::endl;
        fflush(stdout);
        abort();
      }
    }
    // MOD. End.

    // MOD. Begin. Fix not decoding not contiguos instructions
    if(m_shader_config.is_fix_not_decoding_not_contiguos_instructions) {
      if(!m_shader_config.is_fetch_and_decode_improved) {
        std::cout << "Error. If -is_fix_not_decoding_not_contiguos_instructions is set to 1, -is_fetch_and_decode_improved must be set to 1 too." << std::endl;
        fflush(stdout);
        abort();
      }
    }
    // MOD. End.

    if(m_shader_config.is_instruction_prefetching_enabled && (m_shader_config.prefetch_per_stream_buffer_size == 0)) {
      std::cout << "Error. If -is_instruction_prefetching_enabled is set to 1, -prefetch_per_stream_buffer_size must be set to a value greater than 0." << std::endl;
      fflush(stdout);
      abort();
    }

    // MOD. Begin. General parse options
    // scedulers
    // must currently occur after all inputs have been initialized.
    std::string sched_config = m_shader_config.gpgpu_scheduler_string;
    const concrete_scheduler scheduler =
        sched_config.find("lrr") != std::string::npos
            ? CONCRETE_SCHEDULER_LRR
            : sched_config.find("two_level_active") != std::string::npos
                  ? CONCRETE_SCHEDULER_TWO_LEVEL_ACTIVE
                  : sched_config.find("gto") != std::string::npos
                        ? CONCRETE_SCHEDULER_GTO
                        : sched_config.find("rrr") != std::string::npos
                              ? CONCRETE_SCHEDULER_RRR
                        : sched_config.find("old") != std::string::npos
                              ? CONCRETE_SCHEDULER_OLDEST_FIRST
                              : sched_config.find("warp_limiting") !=
                                        std::string::npos
                                    ? CONCRETE_SCHEDULER_WARP_LIMITING
                                    : NUM_CONCRETE_SCHEDULERS;
    assert(scheduler != NUM_CONCRETE_SCHEDULERS);
    m_shader_config.warp_scheduling_mode = scheduler;
    // MOD. End. General parse options

    m_shader_config.cycles_needed_for_address_calculation = ceil(m_shader_config.warp_size / m_shader_config.memory_num_scalar_units_per_subcore);
    m_shader_config.maximum_l1d_latency_at_sm_structure = m_shader_config.memory_l1d_minimum_latency + m_shader_config.memory_maximum_coalescing_cycles;
    m_shader_config.maximum_shared_memory_latency_at_sm_structure = m_shader_config.memory_shared_memory_minimum_latency + m_shader_config.memory_maximum_coalescing_cycles + m_shader_config.memory_shared_memory_extra_latency_ldsm_multiple_matrix;
  
    // Parse inter-warp coalescing selection policy
    std::string iwc_policy_str = m_shader_config.interwarp_coalescing_selection_policy_string;
    m_shader_config.interwarp_coalescing_selection_policy = 
        iwc_policy_str.find("GTL_WARPID") != std::string::npos ? InterWarpCoalescingSelectionPolicies::GTL_WARPID
        : iwc_policy_str.find("SAME_LAST_LEADER_INST_PC_THEN_OLDEST") != std::string::npos ? InterWarpCoalescingSelectionPolicies::SAME_LAST_LEADER_INST_PC_THEN_OLDEST
        : iwc_policy_str.find("WARPPOOL_HYBRID") != std::string::npos ? InterWarpCoalescingSelectionPolicies::WARPPOOL_HYBRID
        : iwc_policy_str.find("DEP_COUNT_WAIT_OLDEST_INST_IBUFFER_GENERIC") != std::string::npos ? InterWarpCoalescingSelectionPolicies::DEP_COUNT_WAIT_OLDEST_INST_IBUFFER_GENERIC
        : iwc_policy_str.find("DEP_COUNT_WAIT_OLDEST_INST_IBUFFER_CHECKING_WARP_ID") != std::string::npos ? InterWarpCoalescingSelectionPolicies::DEP_COUNT_WAIT_OLDEST_INST_IBUFFER_CHECKING_WARP_ID
        : iwc_policy_str.find("DEP_COUNT_WAIT_DETECTED_AT_DECODE_GENERIC") != std::string::npos ? InterWarpCoalescingSelectionPolicies::DEP_COUNT_WAIT_DETECTED_AT_DECODE_GENERIC
        : iwc_policy_str.find("DEP_COUNT_WAIT_DETECTED_AT_DECODE_CHECKING_WARP_ID") != std::string::npos ? InterWarpCoalescingSelectionPolicies::DEP_COUNT_WAIT_DETECTED_AT_DECODE_CHECKING_WARP_ID
        : InterWarpCoalescingSelectionPolicies::IWCOAL_OLDEST; // Default to OLDEST if not matching any other policy
    
    std::string prt_sel_policy_str = m_shader_config.prt_selection_policy_string;
    m_shader_config.prt_selection_policy = 
        prt_sel_policy_str.find("SAME_LAST_WARP_ID_THEN_OLDEST") != std::string::npos ? PRTSelectionPolicies::SAME_LAST_WARP_ID_THEN_OLDEST
        : prt_sel_policy_str.find("SAME_LAST_INST_PC_THEN_OLDEST") != std::string::npos ? PRTSelectionPolicies::SAME_LAST_INST_PC_THEN_OLDEST
        : prt_sel_policy_str.find("WARPID_N_CLUSTERS_WITH_OLDEST") != std::string::npos ? PRTSelectionPolicies::WARPID_N_CLUSTERS_WITH_OLDEST
        : prt_sel_policy_str.find("DEP_COUNT_WAIT_GENERIC_THEN_OLDEST") != std::string::npos ? PRTSelectionPolicies::DEP_COUNT_WAIT_GENERIC_THEN_OLDEST
        : prt_sel_policy_str.find("DEP_COUNT_WAIT_CHECKING_WARP_ID_THEN_OLDEST") != std::string::npos ? PRTSelectionPolicies::DEP_COUNT_WAIT_CHECKING_WARP_ID_THEN_OLDEST
        : PRTSelectionPolicies::OLDEST; // Default to OLDEST if not matching any other policy
    if( (m_shader_config.prt_selection_policy == PRTSelectionPolicies::DEP_COUNT_WAIT_GENERIC_THEN_OLDEST) || (m_shader_config.prt_selection_policy == PRTSelectionPolicies::DEP_COUNT_WAIT_CHECKING_WARP_ID_THEN_OLDEST) ) {
      assert(m_shader_config.is_interwarp_coalescing_enabled);
      assert((m_shader_config.interwarp_coalescing_selection_policy == InterWarpCoalescingSelectionPolicies::DEP_COUNT_WAIT_OLDEST_INST_IBUFFER_GENERIC) || (m_shader_config.interwarp_coalescing_selection_policy == InterWarpCoalescingSelectionPolicies::DEP_COUNT_WAIT_OLDEST_INST_IBUFFER_CHECKING_WARP_ID) || (m_shader_config.interwarp_coalescing_selection_policy == InterWarpCoalescingSelectionPolicies::DEP_COUNT_WAIT_DETECTED_AT_DECODE_GENERIC) || (m_shader_config.interwarp_coalescing_selection_policy == InterWarpCoalescingSelectionPolicies::DEP_COUNT_WAIT_DETECTED_AT_DECODE_CHECKING_WARP_ID));
    }
    assert(!m_shader_config.is_fp32_and_int_unified_pipeline || (m_shader_config.is_fp32_and_int_unified_pipeline && !m_shader_config.is_fp32ops_allowed_in_int_pipeline));
  }
  

 private:
  void init_clock_domains(void);

  // backward pointer
  class gpgpu_context *gpgpu_ctx;
  bool m_valid;
  shader_core_config m_shader_config;
  memory_config m_memory_config;
  // clock domains - frequency
  double core_freq;
  double icnt_freq;
  double dram_freq;
  double l2_freq;
  double core_period;
  double icnt_period;
  double dram_period;
  double l2_period;

  // GPGPU-Sim timing model options
  unsigned long long gpu_max_cycle_opt;
  unsigned long long gpu_max_insn_opt;
  unsigned gpu_max_cta_opt;
  unsigned gpu_max_completed_cta_opt;
  char *gpgpu_runtime_stat;
  bool gpgpu_flush_l1_cache;
  bool gpgpu_flush_l2_cache;
  bool gpu_deadlock_detect;
  int gpgpu_frfcfs_dram_sched_queue_size;
  int gpgpu_cflog_interval;
  char *gpgpu_clock_domains;
  unsigned max_concurrent_kernel;

  // visualizer
  bool g_visualizer_enabled;
  char *g_visualizer_filename;
  int g_visualizer_zlevel;

  // statistics collection
  int gpu_stat_sample_freq;
  int gpu_runtime_stat_flag;

  // Device Limits
  size_t stack_size_limit;
  size_t heap_size_limit;
  size_t runtime_sync_depth_limit;
  size_t runtime_pending_launch_count_limit;

  // gpu compute capability options
  unsigned int gpgpu_compute_capability_major;
  unsigned int gpgpu_compute_capability_minor;
  unsigned long long liveness_message_freq;

  friend class gpgpu_sim;
};

struct occupancy_stats {
  occupancy_stats()
      : aggregate_warp_slot_filled(0), aggregate_theoretical_warp_slots(0) {}
  occupancy_stats(unsigned long long wsf, unsigned long long tws)
      : aggregate_warp_slot_filled(wsf),
        aggregate_theoretical_warp_slots(tws) {}

  unsigned long long aggregate_warp_slot_filled;
  unsigned long long aggregate_theoretical_warp_slots;

  float get_occ_fraction() const {
    return float(aggregate_warp_slot_filled) /
           float(aggregate_theoretical_warp_slots);
  }

  occupancy_stats &operator+=(const occupancy_stats &rhs) {
    aggregate_warp_slot_filled += rhs.aggregate_warp_slot_filled;
    aggregate_theoretical_warp_slots += rhs.aggregate_theoretical_warp_slots;
    return *this;
  }

  occupancy_stats operator+(const occupancy_stats &rhs) const {
    return occupancy_stats(
        aggregate_warp_slot_filled + rhs.aggregate_warp_slot_filled,
        aggregate_theoretical_warp_slots +
            rhs.aggregate_theoretical_warp_slots);
  }
};

class gpgpu_context;
class ptx_instruction;

class watchpoint_event {
 public:
  watchpoint_event() {
    m_thread = NULL;
    m_inst = NULL;
  }
  watchpoint_event(const ptx_thread_info *thd, const ptx_instruction *pI) {
    m_thread = thd;
    m_inst = pI;
  }
  const ptx_thread_info *thread() const { return m_thread; }
  const ptx_instruction *inst() const { return m_inst; }

 private:
  const ptx_thread_info *m_thread;
  const ptx_instruction *m_inst;
};

struct grid_barrier_notify_info {
  grid_barrier_notify_info() : kernel_id(0), sm_ids_to_notify(std::set<unsigned int>()) {}
  grid_barrier_notify_info(unsigned int kernel_id, std::set<unsigned int> sm_ids_to_notify)
      : kernel_id(kernel_id), sm_ids_to_notify(sm_ids_to_notify) {}

  unsigned int kernel_id;
  std::set<unsigned int> sm_ids_to_notify;
};

struct grid_barrier_status {
  grid_barrier_status() : kernel_id(0), num_threads_kernel(0), num_threads_arrived(0), active(false) {}
  grid_barrier_status(unsigned int kernel_id, unsigned long long num_threads_kernel)
      : kernel_id(kernel_id), num_threads_kernel(num_threads_kernel), num_threads_arrived(0), active(false) {}

  bool barrier_completed() {
    return num_threads_arrived == num_threads_kernel;
  }

  unsigned int kernel_id;
  unsigned long long num_threads_kernel;
  unsigned long long num_threads_arrived;
  bool active;
  std::set<unsigned int> sm_ids_to_notify;
};

class gpgpu_sim : public gpgpu_t {
 public:
  gpgpu_sim(const gpgpu_sim_config &config, gpgpu_context *ctx);
  virtual ~gpgpu_sim();
  void set_prop(struct cudaDeviceProp *prop);

  unsigned long long get_current_gpu_cycle() {
    return gpu_sim_cycle + gpu_tot_sim_cycle;
  }

  std::map<std::string, address_type> *get_kernel_adresses_map() { return &m_first_pc_of_each_defined_kernel; } // MOD. Instruction addresses of different kernels have a different address request in memory
  void launch(kernel_info_t *kinfo);
  bool can_start_kernel();
  unsigned finished_kernel();
  void set_kernel_done(kernel_info_t *kernel);
  void stop_all_running_kernels();

  void init();
  std::unique_ptr<grid_barrier_notify_info> register_grid_barrier_arrivement(mem_fetch *mf);
  void increase_num_threads_kernel(unsigned kernel_id, unsigned num_threads);
  void decrease_num_threads_kernel(unsigned kernel_id, unsigned num_threads);
  void cycle();
  bool active();
  bool cycle_insn_cta_max_hit() {
    return (m_config.gpu_max_cycle_opt && (gpu_tot_sim_cycle + gpu_sim_cycle) >=
                                              m_config.gpu_max_cycle_opt) ||
           (m_config.gpu_max_insn_opt &&
            (gpu_tot_sim_insn + gpu_sim_insn) >= m_config.gpu_max_insn_opt) ||
           (m_config.gpu_max_cta_opt &&
            (gpu_tot_issued_cta >= m_config.gpu_max_cta_opt)) ||
           (m_config.gpu_max_completed_cta_opt &&
            (gpu_completed_cta >= m_config.gpu_max_completed_cta_opt));
  }
  void print_stats();
  void update_stats();
  void deadlock_check();
  void inc_completed_cta() { gpu_completed_cta++; }
  void get_pdom_stack_top_info(unsigned sid, unsigned tid, unsigned *pc,
                               unsigned *rpc);

  int shared_mem_size() const;
  int shared_mem_per_block() const;
  int compute_capability_major() const;
  int compute_capability_minor() const;
  int num_registers_per_core() const;
  int num_registers_per_block() const;
  int wrp_size() const;
  int shader_clock() const;
  int max_cta_per_core() const;
  int get_max_cta(const kernel_info_t &k) const;
  const struct cudaDeviceProp *get_prop() const;
  enum divergence_support_t simd_model() const;

  unsigned threads_per_core() const;
  bool get_more_cta_left() const;
  bool kernel_more_cta_left(kernel_info_t *kernel) const;
  bool hit_max_cta_count() const;
  kernel_info_t *select_kernel();
  PowerscalingCoefficients *get_scaling_coeffs();
  void decrement_kernel_latency();

  const gpgpu_sim_config &get_config() const { return m_config; }
  void gpu_print_stat();
  void dump_pipeline(int mask, int s, int m) const;

  void perf_memcpy_to_gpu(size_t dst_start_addr, size_t count);

  // The next three functions added to be used by the functional simulation
  // function

  //! Get shader core configuration
  /*!
   * Returning the configuration of the shader core, used by the functional
   * simulation only so far
   */
  const shader_core_config *getShaderCoreConfig();

  //! Get shader core Memory Configuration
  /*!
   * Returning the memory configuration of the shader core, used by the
   * functional simulation only so far
   */
  const memory_config *getMemoryConfig();

  //! Get shader core SIMT cluster
  /*!
   * Returning the cluster of of the shader core, used by the functional
   * simulation so far
   */
  simt_core_cluster *getSIMTCluster();

  void hit_watchpoint(unsigned watchpoint_num, ptx_thread_info *thd,
                      const ptx_instruction *pI);

  // backward pointer
  class gpgpu_context *gpgpu_ctx;
  
  shader_core_stats* get_shader_stats(){ return m_shader_stats;} // MOD. VPREG

  traced_execution& get_extra_trace_info() { return m_extra_trace_info; }

  void parse_extra_trace_info(std::string filepath, bool is_extra_trace_enabled); // MOD. Improved tracer

  Element_stats m_gpu_per_sm_stats;

  coalescingStatsAcrossSms m_coalescing_stats_across_sms_l1d;
  coalescingStatsAcrossSms m_coalescing_stats_across_sms_const;
  coalescingStatsAcrossSms m_coalescing_stats_across_sms_sharedmem;

  omp_sched_t m_current_omp_scheduler;
  float m_active_sms_this_cycle;


 private:
  void create_gpu_per_sm_stats();
  void gather_gpu_per_sm_stats();
  void reset_cycless_access_history();
  void gather_gpu_per_sm_single_stat(std::string stat_name);
  void reset_gpu_per_sm_stats();
  // clocks
  void reinit_clock_domains(void);
  int next_clock_domain(void);
  void issue_block2core();
  void print_dram_stats(FILE *fout) const;
  void shader_print_runtime_stat(FILE *fout);
  void shader_print_l1_miss_stat(FILE *fout) const;
  void shader_print_cache_stats(FILE *fout) const;
  void shader_print_scheduler_stat(FILE *fout, bool print_dynamic_info);
  void visualizer_printstat();
  void print_shader_cycle_distro(FILE *fout) const;

  void gpgpu_debug();

  unsigned int m_current_cycle_clock_mask;

 protected:
  ///// data /////
  class simt_core_cluster **m_cluster;
  class memory_partition_unit **m_memory_partition_unit;
  class memory_sub_partition **m_memory_sub_partition;

  std::map<std::string, address_type> m_first_pc_of_each_defined_kernel; // MOD. Fix requesting same address for different kernels
  traced_execution m_extra_trace_info;  // MOD. Improved tracer

  std::vector<kernel_info_t *> m_running_kernels;
  unsigned m_last_issued_kernel;

  std::list<unsigned> m_finished_kernel;
  std::map<unsigned int, grid_barrier_status> m_grid_barrier_status;
  std::queue<std::unique_ptr<grid_barrier_notify_info>> m_grid_barrier_notify_queue;
  // m_total_cta_launched == per-kernel count. gpu_tot_issued_cta == global
  // count.
  unsigned long long total_sms_accumulated_across_cycles;
  unsigned long long m_total_cta_launched;
  unsigned long long gpu_tot_issued_cta;
  unsigned gpu_completed_cta;

  unsigned m_last_cluster_issue;
  float *average_pipeline_duty_cycle;
  float *active_sms;
  // time of next rising edge
  double core_time;
  double icnt_time;
  double dram_time;
  double l2_time;

  // debug
  bool gpu_deadlock;

  //// configuration parameters ////
  const gpgpu_sim_config &m_config;

  const struct cudaDeviceProp *m_cuda_properties;
  const shader_core_config *m_shader_config;
  const memory_config *m_memory_config;

  // stats
  class shader_core_stats *m_shader_stats;
  class memory_stats_t *m_memory_stats;
  class power_stat_t *m_power_stats;
  class gpgpu_sim_wrapper *m_gpgpusim_wrapper;
  unsigned long long last_gpu_sim_insn;

  unsigned long long last_liveness_message_time;

  std::map<std::string, FuncCache> m_special_cache_config;

  std::vector<std::string>
      m_executed_kernel_names;  //< names of kernel for stat printout
  std::vector<unsigned>
      m_executed_kernel_uids;  //< uids of kernel launches for stat printout
  std::map<unsigned, watchpoint_event> g_watchpoint_hits;

  std::string executed_kernel_info_string();  //< format the kernel information
                                              // into a string for stat printout
  std::string executed_kernel_name();
  void clear_executed_kernel_info();  //< clear the kernel information after
                                      // stat printout
  virtual void createSIMTCluster() = 0;

 public:
  unsigned long long gpu_sim_insn;
  unsigned long long gpu_tot_sim_insn;
  unsigned long long gpu_sim_insn_last_update;
  unsigned gpu_sim_insn_last_update_sid;
  occupancy_stats gpu_occupancy;
  occupancy_stats gpu_tot_occupancy;

  // performance counter for stalls due to congestion.
  unsigned int gpu_stall_dramfull;
  unsigned int gpu_stall_icnt2sh;
  unsigned long long partiton_reqs_in_parallel;
  unsigned long long partiton_reqs_in_parallel_total;
  unsigned long long partiton_reqs_in_parallel_util;
  unsigned long long partiton_reqs_in_parallel_util_total;
  unsigned long long gpu_sim_cycle_parition_util;
  unsigned long long gpu_tot_sim_cycle_parition_util;
  unsigned long long partiton_replys_in_parallel;
  unsigned long long partiton_replys_in_parallel_total;

  FuncCache get_cache_config(std::string kernel_name);
  void set_cache_config(std::string kernel_name, FuncCache cacheConfig);
  bool has_special_cache_config(std::string kernel_name);
  void change_cache_config(FuncCache cache_config);
  void set_cache_config(std::string kernel_name);

  // Jin: functional simulation for CDP
 private:
  // set by stream operation every time a functoinal simulation is done
  bool m_functional_sim;
  kernel_info_t *m_functional_sim_kernel;

 public:
  bool is_functional_sim() { return m_functional_sim; }
  kernel_info_t *get_functional_kernel() { return m_functional_sim_kernel; }
  void functional_launch(kernel_info_t *k) {
    m_functional_sim = true;
    m_functional_sim_kernel = k;
  }
  void finish_functional_sim(kernel_info_t *k) {
    assert(m_functional_sim);
    assert(m_functional_sim_kernel == k);
    m_functional_sim = false;
    m_functional_sim_kernel = NULL;
  }
};

class exec_gpgpu_sim : public gpgpu_sim {
 public:
  exec_gpgpu_sim(const gpgpu_sim_config &config, gpgpu_context *ctx)
      : gpgpu_sim(config, ctx) {
    createSIMTCluster();
  }

  virtual void createSIMTCluster();
};

#endif
