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

#include <vector>
#include <queue>
#include <memory>
#include <deque>

#include "../stats.h"
#include "../shader.h"
#include "functional_unit.h"

class SM;
class mem_fetch_interface;
class shader_core_stats;
class coalescingAddressStats;
class coalescingStatsPerSm;
class ldst_unit_sm;

uint64_t calculate_constant_address(uint64_t reg_offset_value, traced_operand& op_c);

struct l1d_queue_element {
  l1d_queue_element() {}
  std::deque<mem_fetch*> mfs;
};


class AccessQueue {
  public:
    AccessQueue(unsigned int max_size);
    ~AccessQueue();
    void push(mem_access_t *acc);
    void pop();
    mem_access_t* front();
    bool empty();
    bool full();
    unsigned int size();

  private:
    std::queue<mem_access_t*> m_accesses;
    unsigned int m_max_size;
};

class PendingRequestTableEntry {
  public:
    PendingRequestTableEntry();

    void assign_entry(std::shared_ptr<warp_inst_t> &inst);
    void set_id(unsigned int id);
    void decrement_num_pending_accesses_to_solve();
    void increment_num_pending_accesses_to_solve();
    void release();

    std::shared_ptr<warp_inst_t>& get_inst();
    unsigned int get_id() const;
    unsigned int get_num_pending_accesses_to_solve() const;
    unsigned long long get_assignation_cycle() const;
    unsigned int get_total_num_accesses_to_do() const;
    void set_assignation_cycle(unsigned long long cycle);
    bool is_free() const;
    bool is_pending_to_receive_requests() const;
    void print(FILE *fout) const;

  private:
    std::shared_ptr<warp_inst_t> m_inst;
    unsigned int m_id;
    unsigned int m_num_pending_accesses_to_solve;
    bool m_is_free;
    unsigned long long m_assignation_cycle;
    unsigned int m_total_num_accesses_to_do; // Only useful for global memory accesses with active threads
};

struct cluster_prt_candidate {

  cluster_prt_candidate() : m_id(std::numeric_limits<unsigned int>::max()), m_cycle(std::numeric_limits<unsigned int>::max()) {}
  unsigned int m_id;
  unsigned int m_cycle;
};

class PendingRequestTable {
  public:
    PendingRequestTable(unsigned int max_num_entries, ldst_unit_sm *ldst_unit_sm);
    
    void assign_entry(std::shared_ptr<warp_inst_t> &inst);
    void reactivate_entry(std::shared_ptr<warp_inst_t> &inst);
    void solve_access(unsigned int id);
    void get_accesses_to_coalescing(std::vector<mem_access_t*> &current_accs);
    void get_access_to_next_stage(std::queue<mem_access_t*> &current_accs);
    mem_access_t* get_next_processed_access(unsigned int id);
    std::shared_ptr<warp_inst_t> pop_entry(unsigned int icnt_id);
    std::shared_ptr<warp_inst_t> pop_entries(unsigned int icnt_id);

    bool is_full();
    bool is_empty();

    bool are_entries_to_pop_icnt_id(unsigned int icnt_id);
    bool are_entries_to_process_coalescing();

    unsigned int oldest_selection_policy();
    unsigned int same_last_warp_id();
    unsigned int same_last_pc();
    unsigned int warp_id_N_cluster_priority_and_oldest_inside_each_cluster();
    unsigned int dep_counters_waiting(bool checking_warp_id);

    void management_entries_to_process();

    bool is_entry_going_to_global_memory(unsigned int id);

    bool is_entry_going_to_l1d(unsigned int id);

    void print(FILE *fout) const;
  private:
    unsigned int m_max_num_entries;
    unsigned int m_max_num_entries_to_process_concurrently;
    std::vector<PendingRequestTableEntry> m_entries;
    std::queue<unsigned int> m_entries_id_free_list; 
    std::vector<unsigned int> m_entries_id_pending_list_to_process;
    // One queue per subcore and one extra for icnt of LDGST
    std::vector<std::queue<unsigned int>> m_entries_id_pending_list_to_free;
    std::vector<unsigned int> m_current_entries_id_being_processed; 
    std::vector<unsigned int> m_entries_id_finishing_processed; 
    ldst_unit_sm *m_ldst_unit_sm;
    PRTSelectionPolicies m_selection_policy;
    unsigned int m_last_warp_id;
    address_type m_last_pc;
};

struct pop_interwarp_result {
  pop_interwarp_result() : m_found(false), m_table_idx(0) {}
  bool m_found;
  unsigned int m_table_idx;
  std::map<new_addr_type, mem_access_t *>::iterator m_it_to_pop;
};

class InterWarpCoalescingUnit {
  public:
  InterWarpCoalescingUnit(ldst_unit_sm * mem_unit, unsigned int num_tables, unsigned int max_size_per_table);
  ~InterWarpCoalescingUnit();

  new_addr_type get_addr_signature(new_addr_type addr, memory_space_t space);
  bool insert_access(mem_access_t *acc);
  InterWarpCoalescingSelectionPolicies get_warppool_selection_policy();
  void change_warppool_current_policy(InterWarpCoalescingSelectionPolicies new_policy);
  pop_interwarp_result pop_policy_oldest();
  pop_interwarp_result pop_policy_gtl_warpid();
  pop_interwarp_result pop_policy_dep_counters(bool checking_warp_id);
  mem_access_t* pop_access(bool need_to_drain_intercoalescing_unit);
  bool can_pop_access();
  bool access_is_candidate_to_be_inserted(mem_access_t *acc);
  bool is_empty();
  private:
    ldst_unit_sm *m_ldst_unit_sm;
    std::vector<std::map<new_addr_type, mem_access_t*>> m_intercoalescing_tables;
    unsigned int m_num_tables;
    unsigned int m_max_size_per_table;
    InterWarpCoalescingSelectionPolicies m_selection_policy;
    InterWarpCoalescingSelectionPolicies m_warppool_current_policy;
    unsigned int m_last_greedy_warp_id;
};

class ldst_unit_sm : public functional_unit_shared_sm_part {
 public:
  ldst_unit_sm(
    std::vector<register_set_uniptr*> result_ports,
    std::vector<register_set_uniptr*> reception_ports,
    mem_fetch_interface *icnt,
    mem_fetch_interface *icnt_L1C_L1_half_C,
    std::shared_ptr<shader_core_mem_fetch_allocator> mf_allocator,
    SM *core,
    std::shared_ptr<Scoreboard> scoreboard,
    std::shared_ptr<Scoreboard_reads> scoreboard_reads,
    const shader_core_config *config,
    const memory_config *mem_config,
    shader_core_stats *stats,
    unsigned sid,
    unsigned tpc,
    unsigned int max_size_arbiter_to_subpipeline_reg_per_subcore);
  
  ~ldst_unit_sm() override;
  // modifiers
  void issue(register_set_uniptr &inst, unsigned int icnt_id);
  void cycle() override;

  read_only_cache *get_L1C();
  l1_cache *get_L1D();

  SM* get_SM();

  void fill(mem_fetch *mf);
  void flush();
  void invalidate();
  void writeback(unsigned int icnt_id);

  // accessors
  virtual unsigned clock_multiplier() const;

  bool can_issue(const warp_inst_t *inst) const override;

  bool is_dispatch_reg_empty(unsigned int icnt_id) const;

  virtual void active_lanes_in_pipeline();
  virtual bool stallable() const { return true; }
  bool response_buffer_full() const;
  void print(FILE *fout) const;
  void print_cache_stats(FILE *fp, unsigned &dl1_accesses,
                         unsigned &dl1_misses);
  void get_cache_stats(unsigned &read_accesses, unsigned &write_accesses,
                       unsigned &read_misses, unsigned &write_misses,
                       unsigned cache_type);
  void get_cache_stats(cache_stats &cs);

  void get_L1D_sub_stats(struct cache_sub_stats &css) const;
  void get_L1C_sub_stats(struct cache_sub_stats &css) const;
  void get_L1T_sub_stats(struct cache_sub_stats &css) const;


  coalescingStatsPerSm *get_coalescingStatPerSm_l1d();
  coalescingStatsPerSm *get_coalescingStatPerSm_const();
  coalescingStatsPerSm *get_coalescingStatPerSm_sharedmem();

  void reset_coalescingHistory();

  PendingRequestTable& get_prt();

  unsigned int get_reserved_idx_icnt_to_shmem();

  // All the logic related to this functions is because there can be the case that different entries are compiting for the L1D cache and there might be not enough associativity to process all of them without having reservation fails, which leads to a deadlock.
  bool can_entry_be_selected_for_processing(unsigned int value);
  void increment_num_reserved_associativity_currently_processing(unsigned int value);
  void decrement_num_reserved_associativity_currently_processing(unsigned int value);

  // for debugging
  unsigned long long m_last_inst_gpu_sim_cycle;
  unsigned long long m_last_inst_gpu_tot_sim_cycle;
  unsigned int m_current_num_shared_mem_inst;
  unsigned int m_current_num_normal_mem_inst;

 protected:
  ldst_unit_sm(std::vector<register_set_uniptr*> result_ports, std::vector<register_set_uniptr*> reception_ports, mem_fetch_interface *icnt,
            mem_fetch_interface *icnt_L1C_L1_half_C, std::shared_ptr<shader_core_mem_fetch_allocator> mf_allocator, SM *core,
            std::shared_ptr<Scoreboard> scoreboard, std::shared_ptr<Scoreboard_reads> scoreboard_reads, const shader_core_config *config, // MOD. Fix WAR at baseline.
            const memory_config *mem_config, shader_core_stats *stats,
            unsigned sid, unsigned tpc, l1_cache *new_l1d_cache, unsigned int max_size_arbiter_to_subpipeline_reg_per_subcore);

  void init(mem_fetch_interface *icnt, mem_fetch_interface *icnt_L1C_L1_half_C,
            std::shared_ptr<shader_core_mem_fetch_allocator> mf_allocator,
            SM *core,
            std::shared_ptr<Scoreboard> scoreboard, std::shared_ptr<Scoreboard_reads> scoreboard_reads, const shader_core_config *config, // MOD. Fix WAR at baseline.
            const memory_config *mem_config, shader_core_stats *stats, unsigned sid, unsigned tpc);

  virtual mem_stage_stall_type process_cache_access(
      cache_t &cache, new_addr_type address, warp_inst_t &inst,
      std::list<cache_event> &events, mem_fetch *mf,
      enum cache_request_status status);
  mem_stage_stall_type process_memory_access_queue(cache_t &cache, mem_access_t *acc, bool is_const_cache);

  unsigned get_first_key_pending_writes(warp_inst_t *inst); // MOD. LOOG
  long double get_second_key_pending_writes(warp_inst_t *inst, int idx); // MOD. VPREG

  void global_shared_latency_queue_cycle();

  const memory_config *m_memory_config;
  mem_fetch_interface *m_icnt;
  mem_fetch_interface *m_icnt_L1C_L1_half_C;
  std::shared_ptr<shader_core_mem_fetch_allocator> m_mf_allocator;
  SM *m_core;
  unsigned m_sid;
  unsigned m_tpc;

  tex_cache *m_L1T;        // texture cache
  read_only_cache *m_L1C;  // constant cache
  l1_cache *m_L1D;         // data cache

  std::list<mem_fetch *> m_response_fifo;
  std::shared_ptr<Scoreboard> m_scoreboard;
  std::shared_ptr<Scoreboard_reads> m_scoreboard_reads; // MOD. Fix WAR at baseline.
  mem_fetch *m_next_global;
  unsigned m_num_writeback_clients;

  std::vector<enum mem_stage_stall_type> m_mem_rc_icnt_and_subcores;

  shader_core_stats *m_stats;

  // std::vector<std::deque<mem_fetch *>> l1d_latency_queue;
  std::vector<std::deque<std::shared_ptr<l1d_queue_element>>> l1d_latency_queue;
  std::deque<mem_fetch *> constant_cache_l1_latency_queue;
  
  void L1_constant_cache_latency_queue_cycle();

  void print_L1_constant_latency_queue(FILE *f);


  void L1_latency_queue_cycle();

  void print_L1_latency_queue(FILE *f); // MOD. VPREG

  void reset_is_this_l1d_bank_allocated_this_cycle();

  void cache_cycles();
    
  void shared_dispatch();
  void execute_miscellaneous_dispatch();
  void execute_cache_dispatch(AccessQueue *qu, cache_t *cache, std::function<mem_stage_stall_type(cache_t&, mem_access_t*)> func_process);  
  mem_stage_stall_type dispatch_to_memory_access_queue_l1Dcache(cache_t &cache, mem_access_t *acc);
  mem_stage_stall_type dispatch_to_memory_access_queue_l1Ccache(cache_t &cache, mem_access_t *acc);
  mem_stage_stall_type dispatch_to_memory_access_queue_l1Tcache(cache_t &cache, mem_access_t *acc);
  void dispatch_access_directly_to_l2();

  unsigned long long get_instruction_id(warp_inst_t* inst, unsigned int idx);

  void solve_next_missed_access(cache_t *cache,  bool is_constant);
  void pending_access_logic(std::vector<unsigned int> &prt_list);
  bool is_possible_to_push_to_wb_icnt(unsigned int icnt_id, bool is_ldgsts);
  void push_to_wb_icnt(warp_inst_t inst, unsigned int icnt_id);
  
  // These two queues does not have a limitited size in order to simplify the programmings
  // A queue for the icnt that has the pending movements to shared memory. Only used for LDGSTS
  std::unique_ptr<warp_inst_t> m_ldgsts_icnt_between_ldg_and_sts_part1;
  std::unique_ptr<warp_inst_t> m_ldgsts_icnt_between_ldg_and_sts_part2;
  // A queue per subcore that has the pending writebacks from this unit.
  std::vector<std::unique_ptr<warp_inst_t>> m_pending_wbs_per_subcore;

  std::vector<warp_inst_t> m_evaluating_wb_icnt_and_subcores;
  unsigned int m_max_size_arbiter_to_subpipeline_reg_for_icnt_and_subcores;

  std::vector<unsigned> m_writeback_arb_icnt_and_subcores;  // round-robin arbiter for writeback contention between L1T, L1C, shared for each subcore
  unsigned int m_writeback_arb_between_icnt_and_subcores;  // round-robin arbiter for writeback contention subcores
  unsigned int m_dispatch_subpipeline_arb_between_icnt_and_subcores;  // round-robin arbiter for dispatching instructions to subpipelines (shared, constant, texture l1D) between subcores

  unsigned int m_num_icnt_and_subcores_clients;
  unsigned int m_reserved_idx_icnt_to_shmem; // IDX of vectors structures devoted to the icnt to the shared memory. Used for the LDGSTS

  bool is_already_dispatched_to_shared_mem_this_cycle;
  bool is_already_dispatched_to_texture_mem_this_cycle;
  bool is_already_dispatched_to_constant_mem_this_cycle;

  unsigned int m_num_cycles_to_wait_to_issue_another_mem_inst_from_the_subcores;

  int m_num_reserved_associativity_currently_processing;

  std::vector<std::unique_ptr<warp_inst_t>> m_global_shared_latency_queue_for_ldgsts;

  coalescingAddressStats* m_coalescing_stats_l1d;
  coalescingAddressStats* m_coalescing_stats_const;
  coalescingAddressStats* m_coalescing_stats_sharedmem;

  PendingRequestTable* m_prt;

  // One access queue per l1d bank
  AccessQueue m_access_queue_to_l1c;
  AccessQueue m_access_queue_to_l1t;
  std::vector<AccessQueue*> m_access_queue_to_l1d_preTLB;
  std::vector<AccessQueue*> m_access_queue_to_l1d_postTLB;
  AccessQueue m_access_queue_to_shmem;
  AccessQueue m_access_queue_to_bypass_to_l2;
  AccessQueue m_access_queue_to_miscellaneous;
  std::queue<mem_access_t*> m_next_access_to_queue;
  std::vector<mem_access_t*> m_next_access_to_intercoalescing;
  InterWarpCoalescingUnit *m_intercoalescing_unit;
  

  mem_access_t **m_shmem_pipeline;
  register_set_uniptr m_ldgsts_aux = register_set_uniptr(1, "ldgsts_aux");
};
