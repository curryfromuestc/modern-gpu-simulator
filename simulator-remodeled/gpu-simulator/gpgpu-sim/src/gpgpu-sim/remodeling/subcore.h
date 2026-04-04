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
#include <queue>
#include <memory>
#include "../../constants.h"
#include "../shader.h"
#include "register_file.h"

class read_only_cache;
class functional_unit;
class SM;
class first_level_instruction_cache;

class Subcore {
 public:
  Subcore(unsigned subcore_id, const shader_core_config *config,
          shader_core_stats *stats, SM * sm,
          register_set_uniptr *EX_DP_shared_sm_reception_latch,
          register_set_uniptr *EX_MEM_shared_sm_reception_latch);

  ~Subcore();
  void create_pipeline();

  void decrease_active_warp();
  void increase_active_warp();

  void num_cycles_to_stall(unsigned int num_cycles);
  
  void send_from_fixed_latency_unit_to_rf_write_queue(warp_inst_t *&src);

  Subcore* getptr();
  void cycle();

  int get_fixed_latency_result_queue_size();
  
  void issue_warp(SM *shared_sm, register_set_uniptr &dispatch_latch, warp_inst_t *pI,
                 const active_mask_t &active_mask, unsigned sm_warp_id,
                 functional_unit* fu, bool is_fixed_latency_inst,
                 bool use_traditional_scoreboarding,
                 bool has_dst_reg, TraceEnhancedOperandType dst_result_queue_type);
  void assign_warp_to_subcore(shd_warp_t *warp);
  void finilized_warps_assignation();

  void create_L0s(mem_fetch_interface *icnt_icache);
  first_level_instruction_cache* get_L0I();
  read_only_cache* get_L0C();
  register_set_uniptr *get_EX_WB_sm_shared_units_latch();
  unsigned int get_subcore_id();
  SM *get_sm();

  void display_pipeline(FILE *fout, int print_mem, int mask3bit) const;
  void print_stage(unsigned int stage, FILE *fout) const;
  void get_L0I_sub_stats(struct cache_sub_stats &css) const;

  void add_interwarp_coalescing_dep_counter_at_decode_tracking(warp_inst_t * pI, unsigned sm_warp_id);
  void remove_interwarp_coalescing_dep_counter_at_decode_tracking(warp_inst_t * pI, unsigned sm_warp_id);

  bool is_subcore_with_problems_of_fordward_progress() const;

 private:
  int m_num_active_warps_subcore;
  first_level_instruction_cache* m_L0I;
  read_only_cache *m_L0C_cache;
  unsigned int m_num_pending_cycles_constant_cache_misses_before_switch_to_other_warp;
  unsigned int m_num_pending_cycles_with_issue_port_busy;
  unsigned int m_subcore_id;
  unsigned int m_num_warps_per_subcore;
  unsigned int m_num_regular_rf_banks;
  const shader_core_config *m_config;
  shader_core_stats *m_stats;
  SM *m_sm;

  ifetch_buffer_t m_inst_fetch_decode_latch;

  std::vector<shd_warp_t *> m_warps_of_subcore;
  std::vector<shd_warp_t *> m_warps_of_subcore_priority_ordered_for_fetch;
  // This is the iterator pointer to the greedy warp that has been issued
  unsigned int m_greedy_pointer_issue;
  // This is the iterator pointer to the greedy warp that is going to fetch. At
  // the end of the subcore cycle m_greedy_pointer_fetch =
  // m_greedy_pointer_issue
  unsigned int m_greedy_pointer_fetch;

  bool m_is_next_stage_of_issue_busy;

  register_set_uniptr m_ISSUE_CONTROL_latch = register_set_uniptr(1, "ISSUE_CONTROL_latch");
  register_set_uniptr m_CONTROL_ALLOCATE_latch = register_set_uniptr(1, "CONTROL_ALLOCATE_latch");
  register_set_uniptr m_EX_WB_sm_shared_units_latch = register_set_uniptr(1, "EX_WB_sm_units_latch");
  register_set_uniptr m_read_stage_aux_latch = register_set_uniptr(1, "READ_stage_aux_latch");
  register_set_uniptr *m_EX_DP_shared_sm_reception_latch;
  register_set_uniptr *m_EX_MEM_shared_sm_reception_latch;
  std::vector<std::unique_ptr<warp_inst_t>> m_pipeline_read_stage_latency_reg;

  // Result queues
  // Due to simplification, PRED, UPRED and BREG destination registers go to the regular result queue
  register_set_uniptr m_regular_fixed_latency_rf_write_queue;
  register_set_uniptr m_uniform_fixed_latency_rf_write_queue;
  int m_reserved_slots_regular_fixed_latency_rf_write_queue;
  int m_reserved_slots_uniform_fixed_latency_rf_write_queue;
  
  register_set_uniptr m_EX_WB_sm_variable_latency_latch = register_set_uniptr(1, "EX_WB_sm_variable_latency_latch");

  std::vector<functional_unit*> m_all_subcore_ex_pipelines;
  functional_unit* m_int_pipeline;
  functional_unit* m_sp_pipeline;
  functional_unit* m_uniform_pipeline;
  functional_unit* m_tensor_pipeline;
  functional_unit* m_branch_pipeline;
  functional_unit* m_sfu_pipeline;
  functional_unit* m_miscellaneous_with_queue_pipeline;
  functional_unit* m_miscellaneous_no_queue_pipeline;
  functional_unit* m_memory_unit_subcore;
  functional_unit* m_dp_pipeline;

  Register_file *m_regular_rf;
  Register_file *m_uniform_rf;

  bool has_regular_fixed_latency_rf_result_queue_space();
  bool has_uniform_fixed_latency_rf_result_queue_space();
  void reserve_slot_regular_fixed_latency_rf_result_queue_space();
  void reserve_slot_uniform_fixed_latency_rf_result_queue_space();
  void free_slot_regular_fixed_latency_rf_result_queue_space();
  void free_slot_uniform_fixed_latency_rf_result_queue_space();

  bool writeback_latch_proccess(SM *shared_sm, register_set_uniptr &latch, bool is_from_shared_sm_structure);
  
  void writeback_process_fixed_latency_write_queue(register_set_uniptr &latch, SM *shared_sm, unsigned int max_num_pops, TraceEnhancedOperandType dst_result_queue_type);

  void single_decode(SM *shared_sm, warp_inst_t *pI,
                     IBuffer_Entry &ibuffer_entry, unsigned int sm_warp_id,
                     unsigned int subcore_warp_id, shd_warp_t *warp);

  void writeback(SM * shared_sm);
  void execute();
  void read_rf(SM *shared_sm);
  void allocate(SM *shared_sm);
  void control_stage(SM *shared_sm);
  void issue(SM *shared_sm);
  void decode(SM *shared_sm);
  void fetch(SM *shared_sm);
  
  void set_num_pending_cycles_with_issue_port_busy(const warp_inst_t *pI);
  void generate_fixed_latency_constant_accesses(warp_inst_t *pI);
  bool are_l1c_operands_ready(SM *shared_sm, const warp_inst_t *pI);
  void assign_instruction_warp_id(warp_inst_t *pI, unsigned int subcore_warp_id, unsigned int sm_warp_id);

  void allocate_reads(RF_requests rf_requests, const warp_inst_t *pI, unsigned int sm_warp_id, unsigned int regular_rf_num_read_cycles);
  void allocate_writes(warp_inst_t *inst, Register_file *dst_rf, unsigned int num_uses, unsigned int target_latency_execution_wb);
  bool is_possible_to_write(warp_inst_t *inst, Register_file *dst_rf, unsigned int target_latency_execution_wb, unsigned int &num_uses);
  warp_inst_t *get_next_inst(SM *shared_sm, unsigned int warp_id, address_type pc);
  bool is_waiting_ldgdepbar(const warp_inst_t *pI, unsigned int subcore_warp_id);
  bool is_wait_barriers_ready_entry_point(const warp_inst_t* inst, unsigned int subcore_warp_id);
  std::vector<Wait_Barrier_Checking> wait_barriers_to_check_generic(const warp_inst_t* inst, unsigned int subcore_warp_id);
  std::vector<Wait_Barrier_Checking> wait_barriers_to_check_depbar(const warp_inst_t* inst, unsigned int subcore_warp_id);
  bool is_wait_barriers_ready(std::vector<Wait_Barrier_Checking> &wait_barriers_checking, unsigned int subcore_warp_id);
  void modify_warp_state();
  std::vector<unsigned int> order_greedy_then_highest_id(SM *shared_sm, unsigned int greedy_pointer);
  static bool sort_warps_by_highest_id_dynamic_id(shd_warp_t *lhs,
                                                shd_warp_t *rhs);
  functional_unit* get_fu(const warp_inst_t *pI);
  void create_register_file(SM *shared_sm);
  void manage_instruction_operand_stats(SM *shared_sm, warp_inst_t *pI);
  void manage_operand_stat(SM *shared_sm, const warp_inst_t *pI, unsigned int num_accesses_per_operand, void (Subcore::*increase_stat)(unsigned int, SM *shared_sm));    
  void inc_regular_regfile_reads(unsigned int active_count, SM *shared_sm);
  void inc_regular_regfile_writes(unsigned int active_count, SM *shared_sm);
  void inc_uniform_regfile_reads(unsigned int active_count, SM *shared_sm);
  void inc_uniform_regfile_writes(unsigned int active_count, SM *shared_sm);
  void inc_predicate_regfile_reads(unsigned int active_count, SM *shared_sm);
  void inc_predicate_regfile_writes(unsigned int active_count, SM *shared_sm);
  void inc_uniform_predicate_regfile_reads(unsigned int active_count, SM *sass_op_type);
  void inc_uniform_predicate_regfile_writes(unsigned int active_count, SM *shared_sm);
  void inc_constant_cache_reads(unsigned int active_count, SM *shared_sm);
  void incnon_rf_operands(unsigned int active_count, SM *shared_sm);
};