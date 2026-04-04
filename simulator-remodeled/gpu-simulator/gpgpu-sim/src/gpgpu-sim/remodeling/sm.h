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

#pragma once

#include "../../abstract_hardware_model.h"

#include <stdio.h>
#include <unordered_map>
#include <vector>
#include <stack>
#include <memory>
#include "ldst_unit_sm.h"

#include "../shader_core_wrapper.h"
#include "../shader.h"
#include "../../constants.h"
// #include "../../../../trace-driven/trace_driven.h"

#include "subcore.h"
#include "new_stats.h"

#define NO_TENSOR_OP_4REG_PER_OP_LATENCY_READ_FIXED_LATENCY_INST 3
#define MULTIPLIER_LATENCY_READ_FIXED_LATENCY_INST_TENSOR_CORE_INSTS_WITH_4_REGS_PER_OP 2
#define MAXIMUM_LATENCY_READ_FIXED_LATENCY_INST NO_TENSOR_OP_4REG_PER_OP_LATENCY_READ_FIXED_LATENCY_INST*MULTIPLIER_LATENCY_READ_FIXED_LATENCY_INST_TENSOR_CORE_INSTS_WITH_4_REGS_PER_OP
#define NUM_INTERMEDIATE_CYCLES_UN_BETWEEN_ISSUE_AND_FU_EXECUTION_FOR_FIXED_LATENCY_INST NO_TENSOR_OP_4REG_PER_OP_LATENCY_READ_FIXED_LATENCY_INST+2 // 1 CONTROL, 1 ALLOCATE and 3 READ
#define NUM_INTERMEDIATE_CYCLES_UN_BETWEEN_ISSUE_AND_FU_EXECUTION_FOR_FIXED_LATENCY_INST_TENSOR_CORE_INSTS_WITH_4_REGS_PER_OP MAXIMUM_LATENCY_READ_FIXED_LATENCY_INST+2 // 1 CONTROL, 1 ALLOCATE and 3 READ


class shader_core_config;
class read_only_cache;
class gpgpu_sim;
class simt_core_cluster;
class shader_core_stats;
class memory_config;
class shader_core_mem_fetch_allocator;
class Scoreboard;
class Scoreboard_reads;
class functional_unit;
class coalescingStatsPerSm;
class coalescingStatsAcrossSms;

enum subcore_dispatch_latch_name_t {
  DISPATCH_SP = 0, // FP32
  DISPATCH_DP, // FP64
  DISPATCH_HP, // FP16
  DISPATCH_INT,
  DISPATCH_SFU,
  DISPATCH_TENSOR_CORE,
  DISPATCH_UNIFORM,
  DISPATCH_BRANCH,
  DISPATCH_MISCELLANEOUS, // NOPS go here
  N_DISPATCH_LATCHES
};

const char *const subcore_dispatch_latch_decode[] = {
  "DISPATCH_SP",
  "DISPATCH_DP",
  "DISPATCH_HP", 
  "DISPATCH_INT",
  "DISPATCH_SFU",
  "DISPATCH_TENSOR_CORE",
  "DISPATCH_UNIFORM",
  "DISPATCH_BRANCH",
  "DISPATCH_MISCELLANEOUS",
  "N_DISPATCH_LATCHES"
};

class Waiting_Dep_Counters_per_Warp {
  public:
    Waiting_Dep_Counters_per_Warp(unsigned int warp_id) : m_warp_id(warp_id) {}
    Waiting_Dep_Counters_per_Warp() : m_warp_id(0) {}
    void increase_dep_counter(unsigned int dep_counter_id) {
      if(m_waiting_dep_counters.find(dep_counter_id) == m_waiting_dep_counters.end()) {
        m_waiting_dep_counters[dep_counter_id] = 1;
      } else {
        m_waiting_dep_counters[dep_counter_id]++;
      }
    }
    void decrease_dep_counter(unsigned int dep_counter_id) {
      assert(m_waiting_dep_counters.find(dep_counter_id) != m_waiting_dep_counters.end());
      assert(m_waiting_dep_counters[dep_counter_id] > 0);
      m_waiting_dep_counters[dep_counter_id]--;
      if(m_waiting_dep_counters[dep_counter_id] == 0) {
        m_waiting_dep_counters.erase(dep_counter_id);
      }
    }
    void clear() {
      m_waiting_dep_counters.clear();
    }
    unsigned int m_warp_id;
    std::map<int, unsigned int> m_waiting_dep_counters;
};
class InterWarp_Coalescing_Waiting_Dep_Counters {
  public:
    InterWarp_Coalescing_Waiting_Dep_Counters(unsigned int num_warps) {
      m_waiting_dep_counters_per_warp.resize(num_warps);
      for(unsigned int i = 0; i < num_warps; i++) {
        m_waiting_dep_counters_per_warp[i].m_warp_id = i;
      }
    }
    void clear() {
      for(unsigned int i = 0; i < m_waiting_dep_counters_per_warp.size(); i++) {
        m_waiting_dep_counters_per_warp[i].clear();
      }
    }
    std::vector<Waiting_Dep_Counters_per_Warp> m_waiting_dep_counters_per_warp;
};


unsigned int translate_warp_id_of_sm_to_subcore(unsigned int warp_id, unsigned int num_subcores);

TraceEnhancedOperandType get_reg_type_eval(traced_operand& op);

bool check_is_reserved_regs_remodeling(int reg, TraceEnhancedOperandType reg_type, bool is_trace_mode);

unsigned int translate_reg_to_global_id(int reg, TraceEnhancedOperandType reg_type);

class SM : public core_t, public shader_core_ctx_wrapper {
 public:
  SM(unsigned int num_subcores, gpgpu_sim *gpu, simt_core_cluster *cluster,
     unsigned shader_id, unsigned tpc_id, const shader_core_config *config,
     const memory_config *mem_config, shader_core_stats *stats);

  ~SM() override;
  void init() override;

  void num_cycles_to_stall_SM(unsigned int num_cycles);

  void cycle() override;

  void consume_pending_wait_barrier_actions(std::stack<Wait_Barrier_Entry_Modifier> &actions);
  void add_pending_wait_barrier_decrement(warp_inst_t *inst, Wait_Barrier_Type barrier_type, unsigned int barrier_id);
  void add_pending_wait_barrier_increment(warp_inst_t *inst, Wait_Barrier_Type barrier_type, unsigned int barrier_id);
  void instruction_retirement(warp_inst_t *instruction);
  void issue_warp(register_set_uniptr &warp, warp_inst_t *pI,
                          const active_mask_t &active_mask, unsigned warp_id,
                          unsigned subcore_id, bool use_traditional_scoreboarding);
  virtual void func_exec_inst(warp_inst_t &inst);
  void check_if_warp_has_finished_executing_and_can_be_reclaim(shd_warp_t *warp);
  virtual void checkExecutionStatusAndUpdate(warp_inst_t &inst, unsigned t,
                                             unsigned tid);
  // Returns numbers of addresses in translated_addrs, each addr points to a 4B (32-bit) word                                           
  unsigned int translate_local_memaddr(address_type localaddr, unsigned tid, unsigned num_shader, unsigned datasize,
                                   new_addr_type *translated_addrs);
  void create_logical_structures();
  void create_memory_interfaces();
  virtual void create_shd_warp();
  virtual void init_warps(unsigned cta_id, unsigned start_thread,
                          unsigned end_thread, unsigned ctaid, int cta_size,
                          kernel_info_t &kernel);
  virtual unsigned int sim_init_thread(kernel_info_t &kernel,
                                   ptx_thread_info **thread_info, int sid,
                                   unsigned tid, unsigned threads_left,
                                   unsigned num_threads, core_t *core,
                                   unsigned hw_cta_id, unsigned hw_warp_id,
                                   gpgpu_t *gpu);
  void reinit(unsigned start_thread, unsigned end_thread,
              bool reset_not_completed) override;

  void register_cta_thread_exit(unsigned cta_num, kernel_info_t *kernel);
  address_type next_pc(int tid) const; // return the next pc of a thread

  void set_kernel(kernel_info_t *k) override;
  kernel_info_t *get_kernel() override;
  kernel_info_t *get_kernel_info() override;
  unsigned long long get_current_gpu_cycle() override;
  unsigned int get_num_subcores();
  unsigned int get_sid() const override;
  unsigned int get_tpc_id() const;
  unsigned int get_kernel_id(unsigned warp_id);
  gpgpu_sim *get_gpu() override;
  shader_core_mem_fetch_allocator &get_memf_fetch_allocator();
  read_only_cache *get_L1C();
  std::shared_ptr<Scoreboard_reads> get_scoreboard_WAR();
  std::shared_ptr<Scoreboard> get_scoreboard();
  const memory_config *get_memory_config() const;
  const shader_core_config *get_config() const override;
  shader_core_stats* get_stats() override;
  std::list<unsigned> get_regs_written(const inst_t &fvt) const;
  shd_warp_t *get_shd_warp(int id) override;
  int get_subcore_req_fetch_L1I_priority();
  void set_subcore_req_fetch_L1I_priority(int new_subcore_req_fetch_L1I_priority) override;
  void set_last_inst_gpu_sim_cycle(unsigned long long last_inst_gpu_sim_cycle);
  void set_last_inst_gpu_tot_sim_cycle(unsigned long long last_inst_gpu_tot_sim_cycle);
  bool is_any_subcore_problems_of_fordward_progress() const;
  bool get_is_loog_enabled() override;
  RRS* get_loog_rrs() override;

  void get_pdom_stack_top_info(unsigned tid, unsigned *pc, unsigned *rpc) const override;
  void get_pdom_stack_top_info(unsigned warp_id, const warp_inst_t *pI, unsigned *pc, unsigned *rpc) override;
  virtual const active_mask_t &get_active_mask(unsigned warp_id, const warp_inst_t *pI);

  virtual void warp_exit(unsigned warp_id);

  void cache_flush() override;
  void data_cache_invalidate();
  void cache_invalidate() override;

  void accept_fetch_response(mem_fetch *mf) override;
  void accept_ldst_unit_response(class mem_fetch *mf) override;
  bool fetch_unit_response_buffer_full() const override;
  bool ldst_unit_response_buffer_full() const override;
  bool are_all_wait_barrier_ready(unsigned int warp_id);
  bool warp_waiting_at_barrier(unsigned warp_id) const override;
  bool check_if_non_released_reduction_barrier(warp_inst_t &inst);
  bool warp_waiting_at_mem_barrier(unsigned warp_id) override;
  bool warp_waiting_grid_barrier(unsigned warp_id) override;
  void clear_gridbar(unsigned int kernel_id);
  void broadcast_barrier_reduction(unsigned cta_id, unsigned bar_id,
                                   warp_set_t warps);
  void decrement_atomic_count(unsigned wid, unsigned n);
  unsigned int get_atomic_count(unsigned wid);

  void set_max_cta(const kernel_info_t &kernel);
  unsigned int get_n_active_cta() const override;
  unsigned int get_not_completed() const override;
  unsigned int isactive() const override;
  float get_current_occupancy(unsigned long long &active,
                              unsigned long long &total) const override;

  void issue_block2core(class kernel_info_t &kernel) override;
  bool can_issue_1block(kernel_info_t &kernel) override;
  int find_available_hwtid(unsigned int cta_size, bool occupy);
  bool occupy_shader_resource_1block(kernel_info_t &k, bool occupy);
  void release_shader_resource_1block(unsigned hw_ctaid, kernel_info_t &k);

  void warp_inst_complete(const warp_inst_t &inst) override;
  void dec_inst_in_pipeline(unsigned warp_id) override;
  void store_ack(class mem_fetch *mf) override;
  void inc_store_req(unsigned warp_id) override;
  bool ptx_thread_done(unsigned hw_thread_id) const override;

  // debug:
  void display_simt_state(FILE *fout, int mask) const;
  void display_pipeline(FILE *fout, int print_mem, int mask3bit) const override;
  void display_SM(FILE *fout, int print_mem, int mask3bit) const;
  void dump_warp_state(FILE *fout) const;

  // Stats
  void print_cache_stats(FILE *fp, unsigned &dl1_accesses,
                         unsigned &dl1_misses) override;
  void get_cache_stats(cache_stats &cs) override;
  void get_L0I_sub_stats(struct cache_sub_stats &css) const override;
  void get_L1I_sub_stats(struct cache_sub_stats &css) const override;
  void get_L1D_sub_stats(struct cache_sub_stats &css) const override;
  void get_L1C_sub_stats(struct cache_sub_stats &css) const override;
  void get_L1T_sub_stats(struct cache_sub_stats &css) const override;

  unsigned int inactive_lanes_accesses_sfu(unsigned active_count, double latency);
  unsigned int inactive_lanes_accesses_nonsfu(unsigned active_count,
                                          double latency);
  void mem_instruction_stats(const warp_inst_t &inst) override;
  void incload_stat();
  void incstore_stat();
  void incialu_stat(unsigned active_count, double latency);
  void incimul_stat(unsigned active_count, double latency);
  void incimul24_stat(unsigned active_count, double latency);
  void incimul32_stat(unsigned active_count, double latency);
  void incidiv_stat(unsigned active_count, double latency);
  void incfpalu_stat(unsigned active_count, double latency);
  void incfpmul_stat(unsigned active_count, double latency);
  void incfpdiv_stat(unsigned active_count, double latency);
  void incdpalu_stat(unsigned active_count, double latency);
  void incdpmul_stat(unsigned active_count, double latency);
  void incdpdiv_stat(unsigned active_count, double latency);
  void incsqrt_stat(unsigned active_count, double latency);
  void inclog_stat(unsigned active_count, double latency);
  void incexp_stat(unsigned active_count, double latency);
  void incsin_stat(unsigned active_count, double latency);
  void inctensor_stat(unsigned active_count, double latency);
  void inctex_stat(unsigned active_count, double latency);
  void inc_const_accesses(unsigned active_count);
  void incsfu_stat(unsigned active_count, double latency);
  void incsp_stat(unsigned active_count, double latency);
  void incmem_stat(unsigned active_count, double latency);
  void incregfile_reads(unsigned active_count);
  void incregfile_writes(unsigned active_count);
  void incnon_rf_operands(unsigned active_count);
  void incspactivelanes_stat(unsigned active_count);
  void incsfuactivelanes_stat(unsigned active_count);
  void incfuactivelanes_stat(unsigned active_count);
  void incfumemactivelanes_stat(unsigned active_count);
  void inc_simt_to_mem(unsigned n_flits);
  void incexecstat(warp_inst_t *&inst);
  void get_icnt_power_stats(long &n_simt_to_mem, long &n_mem_to_simt) const override;

  address_type from_local_pc_to_global_pc_address(address_type local_pc, unsigned int unique_function_id) override;
  address_type from_global_pc_address_to_local_pc(address_type global_pc, unsigned int unique_function_id) override;
  
  bool can_send_inst_from_subcore_to_sm_shared_pipeline() const;

  void set_num_cycles_to_wait_to_dispatch_another_inst_from_subcore_to_sm_shared_pipeline(unsigned int num_cycles);

  Element_stats m_sm_stats;

  void create_gpu_per_sm_stats(Element_stats &all_stats) override;
  void reset_cycless_access_history() override;
  void gather_gpu_per_sm_stats(Element_stats &all_stats, coalescingStatsAcrossSms& coal_stats_l1d, coalescingStatsAcrossSms& coal_stats_const, coalescingStatsAcrossSms& coal_stats_sharedmem) override;
  void gather_gpu_per_sm_single_stat(Element_stats &all_stats, std::string stat_name) override;

  void increment_sm_stat_by_integer(std::string stat_name, int val_to_increment) override;
  
  bool is_using_interwarp_coal_warps_waiting_dep_counter();

  InterWarp_Coalescing_Waiting_Dep_Counters *m_interwarp_coal_warps_waiting_dep_counter;

 private:
  unsigned int m_sm_id;
  unsigned int m_tpc_id;
  unsigned int m_num_subcores;

  unsigned int m_num_cycles_to_wait_to_dispatch_another_inst_from_subcore_to_sm_shared_pipeline;

  unsigned long long m_last_inst_gpu_sim_cycle;
  unsigned long long m_last_inst_gpu_tot_sim_cycle;

  const shader_core_config *m_config;
  const memory_config *m_memory_config;
  simt_core_cluster *m_cluster;
  gpgpu_sim *m_gpu;
  shader_core_stats *m_stats;
  std::vector<Subcore*> m_subcores;

  std::vector<register_set_uniptr*> m_EX_WB_sm_shared_units_subcore_latches; 
  std::vector<register_set_uniptr*> m_EX_MEM_reception_latches_per_subcore;
  register_set_uniptr m_EX_DP_shared_sm_reception_latch = register_set_uniptr(1,"EX_DP_shared_sm_reception_latch");
  functional_unit *m_shared_dp_unit;

  mem_fetch_interface *m_icnt;
  std::shared_ptr<read_only_cache> m_L1I_L1_half_C_cache;
  mem_fetch_interface *m_icnt_L0s;
  unsigned int m_subcore_req_fetch_L1I_priority;  // MOD. Added L0I
  // In order to be perfect, another icnt from memory_unit_subcore to shared
  // structures of memory of the subcore needs to be places here.
  ldst_unit_sm *m_ldst_unit_shared_of_sm;
  std::shared_ptr<shader_core_mem_fetch_allocator> m_mem_fetch_allocator;

  std::vector<shd_warp_t *> m_physical_warp;  // per warp information array

  // Scoreboard to fully track data hazards in order to have retrocompatibilty with modes with no control codes
  std::shared_ptr<Scoreboard> m_scoreboard; // RAW and WAW Hazards
  std::shared_ptr<Scoreboard_reads> m_scoreboard_WAR; // WAR Hazards
  std::stack<Wait_Barrier_Entry_Modifier> m_pending_wait_barrier_decrements;
  std::stack<Wait_Barrier_Entry_Modifier> m_pending_wait_barrier_increments;

  barrier_set_t m_barriers;

  // thread contexts
  thread_ctx_t *m_threadState;

  // Used for handing out dynamic warp_ids to new warps.
  // the differnece between a warp_id and a dynamic_warp_id
  // is that the dynamic_warp_id is a running number unique to every warp
  // run on this shader, where the warp_id is the static warp slot.
  unsigned int m_dynamic_warp_id;

  // used for local address mapping with single kernel launch
  unsigned int kernel_max_cta_per_shader;
  unsigned int kernel_padded_threads_per_cta;

  // CTA scheduling / hardware thread allocation
  unsigned int m_n_active_cta;  // number of Cooperative Thread Arrays (blocks)
                                // currently running on this shader.
  unsigned int m_cta_status[MAX_CTA_PER_SHADER];  // CTAs status
  unsigned int m_not_completed;  // number of threads to be completed (==0 when all
                             // thread on this core completed)
  
  // SM occupancy structures
  int m_active_warps;
  std::bitset<MAX_THREAD_PER_SM> m_active_threads;
  unsigned int m_occupied_n_threads;
  unsigned int m_occupied_shmem;
  unsigned int m_occupied_regs;
  unsigned int m_occupied_ctas;
  std::bitset<MAX_THREAD_PER_SM> m_occupied_hwtid;
  std::map<unsigned int, unsigned int> m_occupied_cta_to_hwtid;

  // Power
  PowerscalingCoefficients *m_scaling_coeffs;
};