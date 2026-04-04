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

#include <cassert>
#include <memory>

#include "subcore.h"
#include "functional_unit.h"
#include "sm.h"


#include "../../../../trace-driven/trace_driven.h"
#include "../../../libcuda/gpgpu_context.h"
#include "../gpu-sim.h"
#include "../shader_trace.h"
#include "../stat-tool.h"
#include "first_level_instruction_cache.h"
#include "ldst_unit_sm.h"
#include "register_file.h"

#include "../../../../../util/traces_enhanced/src/traced_instruction.h"



Subcore::Subcore(unsigned subcore_id, const shader_core_config *config,
                 shader_core_stats *stats, SM *sm,
                 register_set_uniptr *EX_DP_shared_sm_reception_latch,
                 register_set_uniptr *EX_MEM_shared_sm_reception_latch) :
                        m_regular_fixed_latency_rf_write_queue(config->max_size_register_file_write_queue_for_fixed_latency_instructions, "regular_fixed_latency_rf_write_queue"),
                        m_uniform_fixed_latency_rf_write_queue(config->max_size_register_file_write_queue_for_fixed_latency_instructions, "uniform_fixed_latency_rf_write_queue") {
  m_subcore_id = subcore_id;
  m_config = config;
  m_stats = stats;
  m_sm = sm;
  m_num_warps_per_subcore =
      m_config->max_warps_per_shader / sm->get_num_subcores();
  m_EX_DP_shared_sm_reception_latch = EX_DP_shared_sm_reception_latch;
  m_EX_MEM_shared_sm_reception_latch = EX_MEM_shared_sm_reception_latch;
  m_num_pending_cycles_constant_cache_misses_before_switch_to_other_warp = 0;
  m_num_pending_cycles_with_issue_port_busy = 0;
  m_pipeline_read_stage_latency_reg.resize(MAXIMUM_LATENCY_READ_FIXED_LATENCY_INST);
  for (unsigned i = 0; i < MAXIMUM_LATENCY_READ_FIXED_LATENCY_INST; i++) {
    m_pipeline_read_stage_latency_reg[i] = std::make_unique<warp_inst_t>(config);
  }
  m_reserved_slots_regular_fixed_latency_rf_write_queue = 0;
  m_reserved_slots_uniform_fixed_latency_rf_write_queue = 0;
  m_num_active_warps_subcore = 0;
  m_is_next_stage_of_issue_busy = false;
}

Subcore::~Subcore() {
  if(!m_config->is_fp32_and_int_unified_pipeline) {
    delete m_int_pipeline;
  }
  delete m_sp_pipeline;
  delete m_uniform_pipeline;
  delete m_tensor_pipeline;
  delete m_branch_pipeline;
  delete m_sfu_pipeline;
  delete m_miscellaneous_with_queue_pipeline;
  delete m_miscellaneous_no_queue_pipeline;
  delete m_memory_unit_subcore;
  delete m_dp_pipeline;
  delete m_uniform_rf;
  delete m_regular_rf;
  delete m_L0I;
  delete m_L0C_cache;
  m_all_subcore_ex_pipelines.clear();
  m_pipeline_read_stage_latency_reg.clear();
}

Subcore* Subcore::getptr() { return this; }

void Subcore::cycle() {
  if(m_num_active_warps_subcore > 0) {
    writeback(m_sm);
    execute();
    read_rf(m_sm);
    allocate(m_sm);
    control_stage(m_sm);
    issue(m_sm);
    decode(m_sm);
    fetch(m_sm);
    m_greedy_pointer_fetch = m_greedy_pointer_issue;
  }
  // if(m_sm->get_sid() == 0 && m_subcore_id == 0 && m_sm->get_current_gpu_cycle() == 3837) {
  //   fflush(stdout);
  // }
  m_L0C_cache->cycle();
  m_L0I->cycle();
}

void Subcore::decrease_active_warp() {
  m_num_active_warps_subcore--;
  assert(m_num_active_warps_subcore >= 0);
}

void Subcore::increase_active_warp() {
  m_num_active_warps_subcore++;
}


void Subcore::num_cycles_to_stall(unsigned int num_cycles) {
  m_num_pending_cycles_with_issue_port_busy += num_cycles;
}

int Subcore::get_fixed_latency_result_queue_size() {
  return m_reserved_slots_regular_fixed_latency_rf_write_queue;
}

bool Subcore::is_subcore_with_problems_of_fordward_progress() const {
  return m_is_next_stage_of_issue_busy;
}


bool Subcore::has_regular_fixed_latency_rf_result_queue_space() {
  bool res = m_reserved_slots_regular_fixed_latency_rf_write_queue < m_config->max_size_register_file_write_queue_for_fixed_latency_instructions;
  return res;
}

bool Subcore::has_uniform_fixed_latency_rf_result_queue_space() {
  bool res = m_reserved_slots_uniform_fixed_latency_rf_write_queue < m_config->max_size_register_file_write_queue_for_fixed_latency_instructions;
  return res;
}

void Subcore::reserve_slot_regular_fixed_latency_rf_result_queue_space() {
  assert(has_regular_fixed_latency_rf_result_queue_space());
  m_reserved_slots_regular_fixed_latency_rf_write_queue++;
}

void Subcore::reserve_slot_uniform_fixed_latency_rf_result_queue_space() {
  assert(has_uniform_fixed_latency_rf_result_queue_space());
  m_reserved_slots_uniform_fixed_latency_rf_write_queue++;
}

void Subcore::free_slot_regular_fixed_latency_rf_result_queue_space() {
  assert(m_reserved_slots_regular_fixed_latency_rf_write_queue > 0);
  m_reserved_slots_regular_fixed_latency_rf_write_queue--;
}

void Subcore::free_slot_uniform_fixed_latency_rf_result_queue_space() {
  assert(m_reserved_slots_uniform_fixed_latency_rf_write_queue > 0);
  m_reserved_slots_uniform_fixed_latency_rf_write_queue--;
}

bool Subcore::writeback_latch_proccess(SM *shared_sm, register_set_uniptr &latch, bool is_from_shared_sm_structure) {
  warp_inst_t *ready_reg = latch.get_ready();
  bool is_retirement_allowed = true;
  unsigned int num_uses = 0;
  unsigned int num_encoded_dsts = 0;
  bool conflict_wb_with_sm_shared_unit = false;
  Register_file *dst_rf = nullptr;
  bool has_rf_modeled = true;
  if (ready_reg && !ready_reg->empty()) {
    if(ready_reg->get_extra_trace_instruction_info().has_destination_registers()) {    
      num_encoded_dsts = ready_reg->get_extra_trace_instruction_info().get_num_destination_registers();
      num_uses = get_number_of_uses_per_operand(ready_reg->get_extra_trace_instruction_info(), ready_reg->get_extra_trace_instruction_info().get_operand(0).get_operand_reg_number(), 0, ready_reg->get_extra_trace_instruction_info().get_operand(0).get_operand_type());
      TraceEnhancedOperandType dst_type = TraceEnhancedOperandType::NONE;
      dst_type = get_reg_type_eval(ready_reg->get_extra_trace_instruction_info().get_operand(0));
      
      if(dst_type == TraceEnhancedOperandType::REG)  {
        dst_rf = m_regular_rf;
      }else if(dst_type == TraceEnhancedOperandType::UREG) {
        dst_rf = m_uniform_rf;
      }else {
        has_rf_modeled = false;
        assert((dst_type == TraceEnhancedOperandType::UPRED) || (dst_type == TraceEnhancedOperandType::PRED) || (dst_type == TraceEnhancedOperandType::BREG));
      }
      if(has_rf_modeled) {
        num_uses = std::min(num_uses, dst_rf->get_num_banks() * dst_rf->get_num_write_ports_per_bank());
      }
          
      if((num_encoded_dsts>0) && has_rf_modeled) {
        if(ready_reg->m_has_wb_from_sm_struct_to_subcore) {
          unsigned int target_reg_id = ready_reg->get_extra_trace_instruction_info().get_operand(0).get_operand_reg_number();
          target_reg_id = ready_reg->get_final_dst_reg(target_reg_id);
          unsigned int bank_id = dst_rf->calculate_target_bank(target_reg_id);
          bool can_write = dst_rf->is_rf_bank_write_port_available_this_cycle(bank_id);
          bool reserve_wb = ready_reg->sm_shared_wb_consumed(can_write, m_config->num_cycles_needed_to_write_a_reg_from_sm_struct_to_subcore, conflict_wb_with_sm_shared_unit);
          if(reserve_wb) {
            dst_rf->allocate_rf_bank_write_port_this_cycle(bank_id);
          }
          is_retirement_allowed = ready_reg->has_sm_shared_wb_finished();
        }else if(has_rf_modeled) {
          for(unsigned j = 0; (j < num_uses) && is_retirement_allowed; j++) {
            unsigned int current_reg_id = ready_reg->get_extra_trace_instruction_info().get_operand(0).get_operand_reg_number() + j;
            unsigned int bank_id = dst_rf->calculate_target_bank(current_reg_id);
            is_retirement_allowed = dst_rf->is_rf_bank_write_port_available_this_cycle(bank_id);
          }
        }
      }
    }
    
    if(is_retirement_allowed) {
      std::unique_ptr<warp_inst_t> &inst_to_retire = latch.get_ready_smartptr();
      assert(inst_to_retire.get() == ready_reg);
      for(unsigned j = 0; (j < num_uses) && !ready_reg->m_has_wb_from_sm_struct_to_subcore && has_rf_modeled && (num_encoded_dsts>0); j++) {
        unsigned int current_reg_id = ready_reg->get_extra_trace_instruction_info().get_operand(0).get_operand_reg_number() + j;
        unsigned int bank_id = m_regular_rf->calculate_target_bank(current_reg_id);
        dst_rf->allocate_rf_bank_write_port_this_cycle(bank_id);
      }
      shared_sm->instruction_retirement(inst_to_retire.get());
    }else {
      if(!ready_reg->m_has_wb_from_sm_struct_to_subcore || (ready_reg->m_has_wb_from_sm_struct_to_subcore && conflict_wb_with_sm_shared_unit)) {
        shared_sm->m_sm_stats.m_stats_map["total_num_times_wb_port_conflict"]->increment_with_integer(1);
      }
    }

    shared_sm->m_sm_stats.m_stats_map["total_num_times_wb_evaluated"]->increment_with_integer(1);
  }

  return is_retirement_allowed;
}

void Subcore::writeback_process_fixed_latency_write_queue(register_set_uniptr &latch, SM *shared_sm, unsigned int max_num_pops, TraceEnhancedOperandType dst_result_queue_type) {
  for(unsigned int i = 0; (i < max_num_pops) && latch.has_ready(); i++) {
    bool retired = writeback_latch_proccess(shared_sm, latch, false);
    if(retired) {
      if(dst_result_queue_type == TraceEnhancedOperandType::UREG) {
        free_slot_uniform_fixed_latency_rf_result_queue_space();
      }else {
        free_slot_regular_fixed_latency_rf_result_queue_space();
      }
    }
  }
}

void Subcore::writeback(SM *shared_sm) {
  writeback_process_fixed_latency_write_queue(m_regular_fixed_latency_rf_write_queue, shared_sm, m_config->max_pops_per_cycle_register_file_write_queue_for_fixed_latency_instructions, TraceEnhancedOperandType::REG);
  writeback_process_fixed_latency_write_queue(m_uniform_fixed_latency_rf_write_queue, shared_sm, m_config->max_pops_per_cycle_register_file_write_queue_for_fixed_latency_instructions, TraceEnhancedOperandType::UREG);
  writeback_latch_proccess(shared_sm, m_EX_WB_sm_variable_latency_latch, false);
  writeback_latch_proccess(shared_sm, m_EX_WB_sm_shared_units_latch, true);
}

void Subcore::execute() {
  for (auto fu : m_all_subcore_ex_pipelines) {
    fu->cycle();
  }
}

void Subcore::read_rf(SM *shared_sm) {
  if(!m_pipeline_read_stage_latency_reg[0]->empty()) {
    functional_unit* fu = m_pipeline_read_stage_latency_reg[0]->get_fu_assigned();
    assert(fu);
    assert(m_read_stage_aux_latch.has_free());
    fu->release_read_barrier(m_pipeline_read_stage_latency_reg[0]);
    m_read_stage_aux_latch.move_in(m_pipeline_read_stage_latency_reg[0]);
    fu->issue(m_read_stage_aux_latch);
  }
  for (unsigned stage = 0; (stage + 1) < MAXIMUM_LATENCY_READ_FIXED_LATENCY_INST; stage++) {
    if (m_pipeline_read_stage_latency_reg[stage]->empty()) {
      move_warp_uniptr(m_pipeline_read_stage_latency_reg[stage], m_pipeline_read_stage_latency_reg[stage + 1]);
    }
  }
  m_regular_rf->cycle();
  m_uniform_rf->cycle();
}

void Subcore::allocate(SM *shared_sm) {
  if(m_CONTROL_ALLOCATE_latch.has_ready()) {
    warp_inst_t *current_ins = m_CONTROL_ALLOCATE_latch.get_ready();
    functional_unit* fu = current_ins->get_fu_assigned();
    RF_requests rf_requests;
    assert(fu->is_fixed_latency_unit());
    unsigned int latency_read_fixed_latency_inst = current_ins->is_tensor_core_op_with_4_registers_per_op() ? MAXIMUM_LATENCY_READ_FIXED_LATENCY_INST : NO_TENSOR_OP_4REG_PER_OP_LATENCY_READ_FIXED_LATENCY_INST;
    if(m_pipeline_read_stage_latency_reg[latency_read_fixed_latency_inst - 1]->empty()) {
      unsigned int sm_warp_id = current_ins->warp_id();
      rf_requests.m_regular = m_regular_rf->is_possible_to_read_cacheable(current_ins, sm_warp_id, fu->get_rf_num_read_cycles());
      rf_requests.m_uniform = m_uniform_rf->is_possible_to_read_cacheable(current_ins, sm_warp_id, m_config->warp_size);
      bool is_read_available = rf_requests.is_possible_to_read();
      unsigned int target_latency_execution = latency_read_fixed_latency_inst + current_ins->latency  + current_ins->initiation_interval;
      bool is_fu_latency_available = fu->is_latency_available(target_latency_execution);
      bool is_rf_ready = is_read_available;
      shared_sm->m_sm_stats.m_stats_map["total_num_evals_rf"]->increment_with_integer(1);
      if(is_rf_ready && is_fu_latency_available) {
        allocate_reads(rf_requests, current_ins, sm_warp_id, fu->get_rf_num_read_cycles());
        shared_sm->m_sm_stats.m_stats_map["total_num_register_file_cache_hits"]->increment_with_integer(rf_requests.m_regular.m_rf_cache_read_requests.size());
        shared_sm->m_sm_stats.m_stats_map["total_num_register_file_cache_allocations"]->increment_with_integer(rf_requests.m_regular.m_rf_cache_allocate_requests.size());
        fu->reserve_latency(target_latency_execution);
        m_CONTROL_ALLOCATE_latch.move_out_to(m_pipeline_read_stage_latency_reg[latency_read_fixed_latency_inst - 1]);
      }else {
        fu->add_extra_cycle_initiation_interval();
        shared_sm->m_sm_stats.m_stats_map["total_num_evals_rf_with_conflict"]->increment_with_integer(1);
      }
    }else {
      fu->add_extra_cycle_initiation_interval();
      shared_sm->m_sm_stats.m_stats_map["total_num_evals_rf_with_conflict"]->increment_with_integer(1);
    }
  }
}

void Subcore::control_stage(SM *shared_sm) {
  if(m_ISSUE_CONTROL_latch.has_ready()) {
    warp_inst_t *current_ins = m_ISSUE_CONTROL_latch.get_ready();
    functional_unit* fu = current_ins->get_fu_assigned();
    bool is_fixed_latency_inst = fu->is_fixed_latency_unit();
    if(!current_ins->m_has_perform_control_stage) {
      if (m_sm->get_config()->is_trace_mode && !((!m_sm->get_shd_warp(current_ins->warp_id())->get_kernel_info()->is_captured_from_binary) || m_sm->get_config()->is_remodeling_scoreboarding_enabled)) {
        if (current_ins->get_extra_trace_instruction_info().get_control_bits().get_is_new_read_barrier()) {
          m_sm->add_pending_wait_barrier_increment(current_ins, READ_WAIT_BARRIER, current_ins->get_extra_trace_instruction_info().get_control_bits().get_id_new_read_barrier());
        }
        if (current_ins->get_extra_trace_instruction_info().get_control_bits().get_is_new_write_barrier()) {
          m_sm->add_pending_wait_barrier_increment(current_ins, WRITE_WAIT_BARRIER, current_ins->get_extra_trace_instruction_info().get_control_bits().get_id_new_write_barrier());
        }
      }
      current_ins->m_has_perform_control_stage = true;
    }
    if(m_CONTROL_ALLOCATE_latch.has_free() || !is_fixed_latency_inst) {
      if(is_fixed_latency_inst) {
        assert(m_CONTROL_ALLOCATE_latch.has_free());
        move_warp_between_reg_sets(m_CONTROL_ALLOCATE_latch, 0, m_ISSUE_CONTROL_latch, 0);
      }else {
        // If it is not a fixed_latency_instruction, it is direclty issued to its functional unit
        if(fu->can_issue(current_ins)) {
          fu->issue(m_ISSUE_CONTROL_latch);
        }
      }
    }else {
      if(is_fixed_latency_inst) {
        fu->add_extra_cycle_initiation_interval();
      }
    }
  }
}

void Subcore::issue(SM *shared_sm) {
  bool is_valid_inst =
      false;  // there was one warp with a valid instruction to issue
  bool is_issued_inst = false;  // Achieved to issue an instruction?
  bool is_issue_port_busy = true;
  bool is_next_stage_availabe = true;
  // bool has_been_possible_to_switch_warp = false;
  // bool is_any_waiting_in_inst_barrier = false;
  // bool is_any_waiting_in_stall_count = false;
  // bool is_any_waiting_in_wait_barrier = false;
  // bool is_any_waiting_in_yield = false;
  // bool is_any_waiting_in_fu_occupied = false;
  // bool is_any_waiting_l1c = false;

  modify_warp_state();
  if(m_num_pending_cycles_with_issue_port_busy > 0) {
    m_num_pending_cycles_with_issue_port_busy--;
  }else if(m_ISSUE_CONTROL_latch.has_free()) {
    is_issue_port_busy = false;
    is_next_stage_availabe = true;
    std::vector<unsigned int> priority_ordered_for_issue = order_greedy_then_highest_id(shared_sm, m_greedy_pointer_issue);
    for (auto c_warp_id : priority_ordered_for_issue) {
      shd_warp_t *c_warp = m_warps_of_subcore[c_warp_id];
      // Don't consider warps that are not yet valid
      if (c_warp == NULL || c_warp->done_exit()) {
        continue;
      }
      unsigned int sm_warp_id = c_warp->get_warp_id();
      unsigned int subcore_warp_id = translate_warp_id_of_sm_to_subcore(
          sm_warp_id, shared_sm->get_num_subcores());
      bool is_the_greedy_warp = (m_greedy_pointer_issue == subcore_warp_id);
      assert(c_warp_id == subcore_warp_id);
      bool is_valid_inst_in_the_warp =
          c_warp->get_IBuffer_remodeled()->is_next_valid();

      if (is_valid_inst_in_the_warp) {
        is_valid_inst = true;

        bool use_traditional_scoreboarding = !c_warp->get_kernel_info()->is_captured_from_binary || m_config->is_remodeling_scoreboarding_enabled || !m_config->is_trace_mode;

        warp_inst_t *pI = c_warp->get_IBuffer_remodeled()->next_inst();
        assert(pI != nullptr);

        bool are_traditional_scoreaboards_ready = true;
        bool is_stall_counter_0 = true;
        bool are_wait_barriers_ready = true;
        bool is_not_yield = true;
        
        if(use_traditional_scoreboarding) {
          are_traditional_scoreaboards_ready = !(shared_sm->get_scoreboard()->checkCollision_remodeling(sm_warp_id, pI) || shared_sm->get_scoreboard_WAR()->checkCollision_remodeling(sm_warp_id, pI));
        }else {
          is_stall_counter_0 =
            c_warp->get_dependency_state()->is_stall_counter_0();
          are_wait_barriers_ready =
            is_wait_barriers_ready_entry_point(pI, subcore_warp_id);
          is_not_yield = c_warp->get_dependency_state()->is_yield_ready(); 
        }

        bool is_not_warp_waiting_ldgdepbar = !is_waiting_ldgdepbar(pI, subcore_warp_id);
        bool is_not_warp_waiting_in_programmer_barrier = !c_warp->waiting();
        functional_unit* fu = get_fu(pI);
        bool is_fu_available = true;;
        bool is_fixed_latency_inst = fu->is_fixed_latency_unit();
        if(is_fixed_latency_inst) {
          is_fu_available = fu->can_issue(pI);
        }
        bool is_l1c_ready = are_l1c_operands_ready(shared_sm, pI);
        bool is_write_available_result_queue_for_fixed_latency_available = true;
        bool has_dst_regs = false;
        TraceEnhancedOperandType dst_type = TraceEnhancedOperandType::NONE;
        if(fu->is_fixed_latency_unit()) {
          if(pI->get_extra_trace_instruction_info().has_destination_registers()) {
            has_dst_regs = true;
            dst_type = fu->get_result_queue_type();
            if(dst_type == TraceEnhancedOperandType::UREG) {
              is_write_available_result_queue_for_fixed_latency_available = has_uniform_fixed_latency_rf_result_queue_space();           
            }else {
              is_write_available_result_queue_for_fixed_latency_available = has_regular_fixed_latency_rf_result_queue_space();
            }
          }
        }

        bool are_switch_warp_conditions_ready =
            is_not_yield && is_stall_counter_0 && are_wait_barriers_ready &&
            is_fu_available && is_not_warp_waiting_in_programmer_barrier &&
            is_not_warp_waiting_ldgdepbar && are_traditional_scoreaboards_ready && is_write_available_result_queue_for_fixed_latency_available;

        bool can_l1c_switch_warp = true;

        if(m_greedy_pointer_issue == subcore_warp_id ) {
          if(is_l1c_ready) {
            m_num_pending_cycles_constant_cache_misses_before_switch_to_other_warp = m_config->num_const_cache_cycle_misses_before_switch_to_other_warp;
          }else if(m_num_pending_cycles_constant_cache_misses_before_switch_to_other_warp > 0) {
            m_num_pending_cycles_constant_cache_misses_before_switch_to_other_warp--;
          }
          if(m_num_pending_cycles_constant_cache_misses_before_switch_to_other_warp > 0) {
            can_l1c_switch_warp = false;
          }
        }

        bool is_inst_ready_to_issue = are_switch_warp_conditions_ready && is_l1c_ready;
        if (is_inst_ready_to_issue) {
          const active_mask_t &active_mask =
              shared_sm->get_active_mask(sm_warp_id, pI);
          assert(c_warp->inst_in_pipeline());
          set_num_pending_cycles_with_issue_port_busy(pI);
          if(m_config->is_interwarp_coalescing_enabled && ((m_config->interwarp_coalescing_selection_policy == DEP_COUNT_WAIT_DETECTED_AT_DECODE_GENERIC) ||
              (m_config->interwarp_coalescing_selection_policy == DEP_COUNT_WAIT_DETECTED_AT_DECODE_CHECKING_WARP_ID)))  {
            remove_interwarp_coalescing_dep_counter_at_decode_tracking(pI, sm_warp_id);
          }
          issue_warp(shared_sm, m_ISSUE_CONTROL_latch, pI, active_mask, sm_warp_id, fu, is_fixed_latency_inst, use_traditional_scoreboarding, has_dst_regs, dst_type);
          is_issued_inst = true;
          m_greedy_pointer_issue = subcore_warp_id;
          m_num_pending_cycles_constant_cache_misses_before_switch_to_other_warp = m_config->num_const_cache_cycle_misses_before_switch_to_other_warp;
          break;
        }else {
          if(!are_switch_warp_conditions_ready) {
            // has_been_possible_to_switch_warp = true;
            // if(!is_fu_available) {
            //   is_any_waiting_in_fu_occupied = true;
            // }
            // if(!is_not_warp_waiting_in_programmer_barrier || !is_not_warp_waiting_ldgdepbar) {
            //   is_any_waiting_in_inst_barrier = true;
            // }
            // if(!is_not_yield) {
            //   is_any_waiting_in_yield = true;
            // }
            // if(!is_stall_counter_0) {
            //   is_any_waiting_in_stall_count = true;
            // }
            // if(!are_wait_barriers_ready) {
            //   is_any_waiting_in_wait_barrier = true;
            // }
            // if(!is_l1c_ready) {
            //   is_any_waiting_l1c = true;
            // }
          }else {
            if(!is_the_greedy_warp || (can_l1c_switch_warp)) {
              // has_been_possible_to_switch_warp = true;
              // if(!is_l1c_ready) {
              //   is_any_waiting_l1c = true;
              // }
            }else {
              // has_been_possible_to_switch_warp = false;
              break;
            }
          }
        
        }
      }
    }
  }else {
    is_next_stage_availabe = false;
  }

  // Stats
  if(is_issued_inst) {
    shared_sm->m_sm_stats.m_stats_map["total_num_cycles_issue_stage_issuing"]->increment_with_integer(1);
  }else if(!is_next_stage_availabe){
    shared_sm->m_sm_stats.m_stats_map["total_num_cycles_issue_stage_stall_next_stage_not_available"]->increment_with_integer(1);
  }else if(is_issue_port_busy) { // IMAD.WIDE scenario
    shared_sm->m_sm_stats.m_stats_map["total_num_cycles_issue_stage_stall_issue_port_busy"]->increment_with_integer(1);
  }else if(!is_valid_inst) {
    shared_sm->m_sm_stats.m_stats_map["total_num_cycles_issue_stage_stall_no_valid_instruction"]->increment_with_integer(1);
  }else { // It has been possible to switch to another warp, but none where ready to issue
    shared_sm->m_sm_stats.m_stats_map["total_num_cycles_issue_stage_stall_no_warps_ready"]->increment_with_integer(1);
    // m_stats->total_num_cycles_issue_stage_stall_at_least_one_warp_with_fu_occupied += is_any_waiting_in_fu_occupied;
    // m_stats->total_num_cycles_issue_stage_stall_at_least_one_warp_waiting_inst_barrier += is_any_waiting_in_inst_barrier;
    // m_stats->total_num_cycles_issue_stage_stall_at_least_one_warp_waiting_yield += is_any_waiting_in_yield;
    // m_stats->total_num_cycles_issue_stage_stall_at_least_one_warp_waiting_stall_count += is_any_waiting_in_stall_count;
    // m_stats->total_num_cycles_issue_stage_stall_at_least_one_warp_waiting_wait_barrier += is_any_waiting_in_wait_barrier;
    // m_stats->total_num_cycles_issue_stage_stall_at_least_one_warp_waiting_l1c += is_any_waiting_l1c;
  }
  shared_sm->m_sm_stats.m_stats_map["total_num_cycles_issue_stage_evaluated"]->increment_with_integer(1);

  m_is_next_stage_of_issue_busy = !is_next_stage_availabe;
}

void Subcore::modify_warp_state() {
  for (auto warp : m_warps_of_subcore) {
    warp->get_dependency_state()->cycle();
    if(m_config->is_interwarp_coalescing_enabled && ((m_config->interwarp_coalescing_selection_policy == DEP_COUNT_WAIT_OLDEST_INST_IBUFFER_GENERIC) ||
        (m_config->interwarp_coalescing_selection_policy == DEP_COUNT_WAIT_OLDEST_INST_IBUFFER_CHECKING_WARP_ID)))  {
      if (warp == NULL || warp->done_exit()) {
        continue;
      }
      unsigned int sm_warp_id = warp->get_warp_id();
      bool is_valid_inst_in_the_warp =
      warp->get_IBuffer_remodeled()->is_next_valid();
      if(is_valid_inst_in_the_warp) {
        warp_inst_t *pI = warp->get_IBuffer_remodeled()->next_inst();
        assert(pI != nullptr);
        add_interwarp_coalescing_dep_counter_at_decode_tracking(pI, sm_warp_id);
      }
    }
  }
}

void Subcore::add_interwarp_coalescing_dep_counter_at_decode_tracking(warp_inst_t * pI, unsigned sm_warp_id) {
  std::vector<Wait_Barrier_Checking> wait_barriers_checking_generic;
  if(pI->is_any_kind_of_barrier()) {
    for(unsigned int i = 0; i < m_config->num_wait_barriers_per_warp; i++) {
      wait_barriers_checking_generic.push_back(Wait_Barrier_Checking(i, 0));
    }
  }else{
    wait_barriers_checking_generic = wait_barriers_to_check_generic(pI, sm_warp_id);
    if(pI->op == DEPBAR_OP) {
      std::vector<Wait_Barrier_Checking> wait_barriers_checking_depbar = wait_barriers_to_check_depbar(pI, sm_warp_id);
        wait_barriers_checking_generic.insert(wait_barriers_checking_generic.end(), wait_barriers_checking_depbar.begin(), wait_barriers_checking_depbar.end());
    }
  }
  for(auto &wait_bar : wait_barriers_checking_generic) {
    m_sm->m_interwarp_coal_warps_waiting_dep_counter->m_waiting_dep_counters_per_warp[sm_warp_id].increase_dep_counter(wait_bar.barrier_id);
  }
}

void Subcore::remove_interwarp_coalescing_dep_counter_at_decode_tracking(warp_inst_t * pI, unsigned sm_warp_id) {
  std::vector<Wait_Barrier_Checking> wait_barriers_checking_generic;
  if(pI->is_any_kind_of_barrier()) {
    for(unsigned int i = 0; i < m_config->num_wait_barriers_per_warp; i++) {
      wait_barriers_checking_generic.push_back(Wait_Barrier_Checking(i, 0));
    }
  }else{
    wait_barriers_checking_generic = wait_barriers_to_check_generic(pI, sm_warp_id);
    if(pI->op == DEPBAR_OP) {
      std::vector<Wait_Barrier_Checking> wait_barriers_checking_depbar = wait_barriers_to_check_depbar(pI, sm_warp_id);
        wait_barriers_checking_generic.insert(wait_barriers_checking_generic.end(), wait_barriers_checking_depbar.begin(), wait_barriers_checking_depbar.end());
    }
  }
  for(auto &wait_bar : wait_barriers_checking_generic) {
    m_sm->m_interwarp_coal_warps_waiting_dep_counter->m_waiting_dep_counters_per_warp[sm_warp_id].decrease_dep_counter(wait_bar.barrier_id);
  }
}

void Subcore::set_num_pending_cycles_with_issue_port_busy(const warp_inst_t *pI) {
  if(m_config->is_trace_mode && pI->get_extra_trace_instruction_info().get_is_imad()) {
    m_num_pending_cycles_with_issue_port_busy = m_config->num_cycles_issue_port_busy_after_imadwide;
  }else {
    m_num_pending_cycles_with_issue_port_busy = 0;
  }
}

bool Subcore::is_waiting_ldgdepbar(const warp_inst_t *pI, unsigned int subcore_warp_id) {
  bool res = false;
  if(pI->op ==  LDGDEPBAR_OP) {
    res = m_warps_of_subcore[subcore_warp_id]->get_dependency_state()->are_ldgsts_pending();
  }
  return res;
}

bool Subcore::is_wait_barriers_ready_entry_point(const warp_inst_t *inst,
                                                 unsigned int subcore_warp_id) {
  std::vector<Wait_Barrier_Checking> wait_barriers_checking_generic =
      wait_barriers_to_check_generic(inst, subcore_warp_id);
                                              
  bool are_wait_barriers_ready =
    is_wait_barriers_ready(wait_barriers_checking_generic, subcore_warp_id);
  if (are_wait_barriers_ready && inst->op == DEPBAR_OP) {
    std::vector<Wait_Barrier_Checking> wait_barriers_checking_depbar = wait_barriers_to_check_depbar(inst, subcore_warp_id);
    are_wait_barriers_ready =
    is_wait_barriers_ready(wait_barriers_checking_depbar, subcore_warp_id);
  }
  return are_wait_barriers_ready;
}

std::vector<Wait_Barrier_Checking> Subcore::wait_barriers_to_check_generic(const warp_inst_t* inst, unsigned int subcore_warp_id) {
  int wait_barrier_mask_int = inst->get_extra_trace_instruction_info()
                                  .get_control_bits()
                                  .get_wait_barrier_bits();
  std::bitset<6> wait_barrier_mask(wait_barrier_mask_int);
  std::vector<Wait_Barrier_Checking> wait_barriers_checking;
  for (unsigned int i = 0; i < m_config->num_wait_barriers_per_warp; i++) {
    if (wait_barrier_mask[i]) {
      wait_barriers_checking.push_back(Wait_Barrier_Checking(i, 0));
    }
  }
  return wait_barriers_checking;
}

std::vector<Wait_Barrier_Checking> Subcore::wait_barriers_to_check_depbar(const warp_inst_t* inst, unsigned int subcore_warp_id) {
  std::size_t num_operands = inst->get_extra_trace_instruction_info().get_num_operands();
  assert( num_operands > 1);
  traced_operand& op_sb = inst->get_extra_trace_instruction_info().get_operand(0);
  traced_operand& op_val = inst->get_extra_trace_instruction_info().get_operand(1);
  assert(op_sb.get_operand_type() == TraceEnhancedOperandType::SB);
  assert(op_sb.get_has_reg());
  assert(op_val.get_operand_type() == TraceEnhancedOperandType::IMM_UINT64);
  assert(op_val.get_has_inmediate());
  unsigned int sb_reg = op_sb.get_operand_reg_number();
  unsigned int sb_max_allowed_val = op_val.get_operands_inmediates()[0];
  assert(sb_reg < m_config->num_wait_barriers_per_warp );
  std::vector<Wait_Barrier_Checking> wait_barriers_checking;
  wait_barriers_checking.push_back(Wait_Barrier_Checking(sb_reg, sb_max_allowed_val));
  for (unsigned int i = 2; i < num_operands; i++) {
    assert(inst->get_extra_trace_instruction_info().get_operand(i).get_has_inmediate());
    sb_reg = inst->get_extra_trace_instruction_info().get_operand(i).get_operands_inmediates()[0];
    assert(sb_reg < m_config->num_wait_barriers_per_warp );
    wait_barriers_checking.push_back(Wait_Barrier_Checking(sb_reg, 0));
  }
  return wait_barriers_checking;
}

bool Subcore::is_wait_barriers_ready(std::vector<Wait_Barrier_Checking> &wait_barriers_checking,
                                             unsigned int subcore_warp_id) {
  bool are_wait_barriers_ready =
      m_warps_of_subcore[subcore_warp_id]
          ->get_dependency_state()
          ->are_wait_barriers_ready(wait_barriers_checking);
  return are_wait_barriers_ready;
}

void Subcore::generate_fixed_latency_constant_accesses(warp_inst_t *pI) {
  if(!(pI->is_load() && (pI->space.get_type() == const_space)) && pI->has_extra_trace_instruction_info() && !pI->m_has_the_constant_addr_already_calculated) {
    for(unsigned int i = 0; !pI->get_generated_constant_accesses() && (i < pI->get_extra_trace_instruction_info().get_num_operands()); i++) {
      if(pI->get_extra_trace_instruction_info().get_operand(i).get_operand_type() == TraceEnhancedOperandType::CBANK) {
        new_addr_type addr = calculate_constant_address(0, pI->get_extra_trace_instruction_info().get_operand(i));
        pI->generate_fixed_latency_constant_accesses(addr);
        pI->set_generated_constant_accesses(true);
      }
    }
  }
}

bool Subcore::are_l1c_operands_ready(SM *shared_sm, const warp_inst_t *pI) {
  bool are_l1c_operands_ready = true;
  if(pI->get_generated_constant_accesses() && !pI->accessq_empty()) {
    for(const auto mem_acc : pI->get_mem_accesses()) {
      assert(mem_acc.get_type() == CONST_ACC_R);
      new_addr_type addr = mem_acc.get_addr();
      mem_fetch *mf = shared_sm->get_memf_fetch_allocator().alloc(*pI, mem_acc, shared_sm->get_current_gpu_cycle());
      mf->set_subcore(m_subcore_id);
      mf->set_is_fixed_latency_constant_access(true);
      enum cache_request_status status = HIT;
      if(!m_config->perfect_inst_const_cache && !m_config->perfect_constant_cache) {
        std::list<cache_event> events;
        bool useless = false;
        status = m_L0C_cache->access(addr, mf, shared_sm->get_current_gpu_cycle(), events, useless);
      }
      are_l1c_operands_ready = are_l1c_operands_ready ? (status == HIT) : false;
      if( (status == HIT) || (status == HIT_RESERVED) || (status == RESERVATION_FAIL) ) {
        delete mf;
      }
    }
  }
  return are_l1c_operands_ready;
}

void Subcore::assign_instruction_warp_id(warp_inst_t *pI, unsigned int subcore_warp_id, unsigned int sm_warp_id) {
  unsigned int dynamic_warp_id = m_warps_of_subcore[subcore_warp_id]->get_dynamic_warp_id();
  pI->set_some_warp_attributes(sm_warp_id, dynamic_warp_id);
}

void Subcore::allocate_reads(RF_requests rf_requests, const warp_inst_t *pI, unsigned int sm_warp_id, unsigned int regular_rf_num_read_cycles) {
  assert(rf_requests.m_regular.m_is_possible_to_read);
  m_regular_rf->allocate_reads_cacheable(rf_requests.m_regular, pI, sm_warp_id, regular_rf_num_read_cycles);
  assert(rf_requests.m_uniform.m_is_possible_to_read);
  m_uniform_rf->allocate_reads_cacheable(rf_requests.m_uniform, pI, sm_warp_id, m_config->warp_size);
}


bool Subcore::is_possible_to_write(warp_inst_t *inst, Register_file *dst_rf, unsigned int target_latency_execution_wb, unsigned int &num_uses) {
  bool is_write_available = true;
  num_uses = get_number_of_uses_per_operand(inst->get_extra_trace_instruction_info(), inst->get_extra_trace_instruction_info().get_operand(0).get_operand_reg_number(), 0, inst->get_extra_trace_instruction_info().get_operand(0).get_operand_type());
  num_uses = std::min(num_uses, dst_rf->get_num_banks() * dst_rf->get_num_write_ports_per_bank());
  for(unsigned j = 0; (j < num_uses) && is_write_available; j++) {
    unsigned int current_reg_id = inst->get_extra_trace_instruction_info().get_operand(0).get_operand_reg_number() + j;
    unsigned int bank_id = dst_rf->calculate_target_bank(current_reg_id);
    is_write_available = dst_rf->is_rf_bank_write_port_available_at_given_cycle(bank_id, target_latency_execution_wb);
  }
  return is_write_available;
}

void Subcore::allocate_writes(warp_inst_t *inst, Register_file *dst_rf, unsigned int num_uses, unsigned int target_latency_execution_wb) {
  for(unsigned j = 0; (j < num_uses) && !inst->m_has_wb_from_sm_struct_to_subcore; j++) {
    unsigned int current_reg_id = inst->get_extra_trace_instruction_info().get_operand(0).get_operand_reg_number() + j;
    unsigned int bank_id = dst_rf->calculate_target_bank(current_reg_id);
    dst_rf->allocate_rf_bank_write_port_at_given_cycle(bank_id, target_latency_execution_wb);
  }
}

void Subcore::issue_warp(SM *shared_sm, register_set_uniptr &dispatch_latch, warp_inst_t *pI,
                         const active_mask_t &active_mask, unsigned sm_warp_id,
                         functional_unit* fu, bool is_fixed_latency_inst,
                         bool use_traditional_scoreboarding, bool has_dst_reg, TraceEnhancedOperandType dst_result_queue_type) {
  pI->set_fu_assigned(fu);
  manage_instruction_operand_stats(shared_sm, pI);
  shared_sm->issue_warp(dispatch_latch, pI, active_mask, sm_warp_id, m_subcore_id, use_traditional_scoreboarding);
  if(has_dst_reg && fu->is_fixed_latency_unit()) {
    if(dst_result_queue_type == TraceEnhancedOperandType::UREG) {
      reserve_slot_uniform_fixed_latency_rf_result_queue_space();
    }else {
      reserve_slot_regular_fixed_latency_rf_result_queue_space();
    }
  }
  if(is_fixed_latency_inst) {
    fu->reserve_unit(dispatch_latch);
  }
}

std::vector<unsigned int> Subcore::order_greedy_then_highest_id(SM *shared_sm, unsigned int greedy_pointer) {
  std::vector<unsigned int> result_list;
  std::vector<shd_warp_t*> temp = m_warps_of_subcore;
  result_list.push_back(greedy_pointer);
  std::sort(temp.begin(), temp.end(), sort_warps_by_highest_id_dynamic_id);
  for(auto c_warp : temp) {
    unsigned int warp_subcore_id = translate_warp_id_of_sm_to_subcore( c_warp->get_warp_id(), shared_sm->get_num_subcores() );
    if(!c_warp->done_exit() && (warp_subcore_id != greedy_pointer)) {
      result_list.push_back(warp_subcore_id);
    }
  }
  return result_list;
}

bool Subcore::sort_warps_by_highest_id_dynamic_id(shd_warp_t *lhs,
                                                shd_warp_t *rhs) {
  if (rhs && lhs) {
    if (lhs->done_exit() || lhs->waiting()) {
      return false;
    } else if (rhs->done_exit() || rhs->waiting()) {
      return true;
    } else {
      return lhs->get_dynamic_warp_id() > rhs->get_dynamic_warp_id();
    }
  } else {
    return lhs > rhs;
  }
}

functional_unit* Subcore::get_fu(const warp_inst_t *pI) {
  functional_unit* fu = nullptr;
  switch (pI->op) {
    case PREDICATE_OP:
    case INTP_OP:
      if(m_config->is_fp32_and_int_unified_pipeline) {
        fu = m_sp_pipeline;
      }else {
        fu = m_int_pipeline;
      }
      break;
    case HALF_OP:
      fu = m_sp_pipeline;
    case SP_OP:
      fu = m_sp_pipeline;
      if(m_config->is_fp32ops_allowed_in_int_pipeline && m_int_pipeline->can_issue(pI) && !pI->get_extra_trace_instruction_info().get_is_imad()) { /// INCLUIR AQUI IMAD
        fu = m_int_pipeline;
      }
      break;
    case DP_OP:
      fu = m_dp_pipeline;
      break;
    case UNIFORM_OP:
      fu = m_uniform_pipeline;
      break;
    case TENSOR_CORE_OP:
      fu = m_tensor_pipeline;
      break;
    case CALL_OPS:
    case RET_OPS:
    case EXIT_OPS:
    case BRANCH_OP:
      fu = m_branch_pipeline;
      break;
    case SFU_OP:
      fu = m_sfu_pipeline;
      break;
    case MISCELLANEOUS_QUEUE_OP:
      fu = m_miscellaneous_with_queue_pipeline;
      break;
    case LDGDEPBAR_OP:
    case BARRIER_OP:
    case DEPBAR_OP:
    case MISCELLANEOUS_NO_QUEUE_OP:
      fu = m_miscellaneous_no_queue_pipeline;
      break;
    case TEXTURE_OP:
    case SURFACE_OP:
    case MEMORY_BARRIER_OP:
    case GRID_BARRIER_OP:
    case MEMORY_MISCELLANEOUS_OP:
    case LOAD_OP:
    case STORE_OP:
      fu = m_memory_unit_subcore;
      break;
    default:
      fflush(stdout);
      std::cout << "ERROR. EXECUTION PIPELINE FOR THIS INSTRUCTION NOT "
                   "IMPLEMENTED"
                << std::endl;
      abort();
  }
  return fu;
}

void Subcore::single_decode(SM *shared_sm, warp_inst_t *pI,
                            IBuffer_Entry &ibuffer_entry,
                            unsigned int sm_warp_id,
                            unsigned int subcore_warp_id, shd_warp_t *warp) {
  if (pI) {
    assign_instruction_warp_id(pI, subcore_warp_id, sm_warp_id);
    generate_fixed_latency_constant_accesses(pI);
    pI->assign_predicate_latencies_if_needed(m_sm->get_gpu());
    warp->inc_inst_in_pipeline();
    pI->set_unique_inst_id(warp->m_last_unique_inst_id);
    if(pI->is_tensor_core_op()) {
      pI->get_tensor_core_instruction_info();
    }
    warp->m_last_unique_inst_id++;
    if (pI->is_load() || pI->is_store()) {
      pI->generate_mem_latencies(m_sm->get_gpu());
    }else if(pI->is_memory_barrier() || pI->is_grid_barrier() || pI->is_memory_miscelanous()) {
      pI->generate_other_mem_ops_latencies(m_sm->get_gpu());
    }else if(pI->is_texture()) {
      pI->generate_texture_latencies(m_sm->get_gpu());
    } else if (pI->is_dp_op()) {
      pI->generate_dp_latencies(m_sm->get_gpu());      
    }else if(pI->is_tensor_core_op()) {
      pI->generate_tensor_core_latencies(m_sm->get_gpu());
    }else if(pI->is_sfu_useful()) {
      pI->m_num_cycles_to_wait_to_free_WAR = pI->latency + pI->initiation_interval - 3;
    }else if(pI->is_miscellaneous_queue()) {
      pI->generate_miscellaneous_queue_latencies(m_sm->get_gpu());
    }
    ibuffer_entry.m_valid = true;
    
    ibuffer_entry.m_inst = pI;
    assert(ibuffer_entry.m_inst->pc == ibuffer_entry.m_pc);
    if ((pI->oprnd_type == INT_OP) ||
        (pI->oprnd_type == UN_OP)) {  // these counters get added up in
                                      // mcPat to compute scheduler power
      m_stats->m_num_INTdecoded_insn[shared_sm->get_sid()]++;
    } else if (pI->oprnd_type == FP_OP) {
      m_stats->m_num_FPdecoded_insn[shared_sm->get_sid()]++;
    }

    if(m_config->is_interwarp_coalescing_enabled && ( (m_config->interwarp_coalescing_selection_policy == InterWarpCoalescingSelectionPolicies::DEP_COUNT_WAIT_DETECTED_AT_DECODE_GENERIC) || (m_config->interwarp_coalescing_selection_policy == InterWarpCoalescingSelectionPolicies::DEP_COUNT_WAIT_DETECTED_AT_DECODE_CHECKING_WARP_ID) ) ) {
      add_interwarp_coalescing_dep_counter_at_decode_tracking(pI, sm_warp_id);
    }
  }
}

void Subcore::decode(SM *shared_sm) {
  if (m_inst_fetch_decode_latch.m_valid) {
    address_type pc = m_inst_fetch_decode_latch.m_pc;
    m_stats->m_num_decoded_insn[shared_sm->get_sid()]++;
    if(m_config->ibuffer_coalescing) {
      for (auto warp : m_warps_of_subcore) {
        if (!warp->functional_done()) {
          unsigned int sm_warp_id = warp->get_warp_id();
          unsigned int subcore_warp_id = translate_warp_id_of_sm_to_subcore(
              sm_warp_id, shared_sm->get_num_subcores());
          for (auto &ibuffer_entry :
              warp->get_IBuffer_remodeled()->get_remodeled_ibuffer()) {
            if (!ibuffer_entry.m_valid && (ibuffer_entry.m_pc == pc)) {
              warp_inst_t *pI = get_next_inst(shared_sm, subcore_warp_id, pc);
              single_decode(shared_sm, pI, ibuffer_entry, sm_warp_id,
                            subcore_warp_id, warp);
            }
          }
        }
      }
    }else {
      
      unsigned int subcore_warp_id = m_inst_fetch_decode_latch.m_warp_id;
      shd_warp_t *warp = m_warps_of_subcore[subcore_warp_id];
      unsigned int sm_warp_id = warp->get_warp_id();
      for (auto &ibuffer_entry :
           warp->get_IBuffer_remodeled()->get_remodeled_ibuffer()) {
        if (!ibuffer_entry.m_valid && (ibuffer_entry.m_pc == pc)) {
          warp_inst_t *pI = get_next_inst(shared_sm, subcore_warp_id, pc);
          single_decode(shared_sm, pI, ibuffer_entry, sm_warp_id,
                        subcore_warp_id, warp);
        }
      }
    }
    m_inst_fetch_decode_latch.m_valid = false;
  }
}

warp_inst_t *Subcore::get_next_inst(SM *shared_sm, unsigned int warp_id, address_type pc) {
  assert(warp_id < m_warps_of_subcore.size());
  if (m_config->is_trace_mode) {
    // read the inst from the traces
    trace_shd_warp_t *m_trace_warp = static_cast<trace_shd_warp_t *>(
        m_warps_of_subcore[warp_id]);
    return m_trace_warp->get_next_trace_inst(pc);
  } else {
    return shared_sm->get_gpu()->gpgpu_ctx->ptx_fetch_inst(pc);
  }
}

void Subcore::fetch(SM *shared_sm) {
  if (!m_inst_fetch_decode_latch.m_valid) {
    if (m_L0I->is_first_access_ready()) {
      mem_fetch *mf = m_L0I->next_first_access();
      unsigned int unique_function_id = mf->get_unique_function_id();
      unsigned int subcore_warp_id = translate_warp_id_of_sm_to_subcore(
          mf->get_wid(), shared_sm->get_num_subcores());
      address_type local_pc_response =
          shared_sm->from_global_pc_address_to_local_pc(mf->get_addr(), unique_function_id);
      m_inst_fetch_decode_latch = ifetch_buffer_t(
          local_pc_response, mf->get_access_size(), subcore_warp_id);
      // if(shared_sm->get_sid() == 1 && m_subcore_id == 0) {
      //   std::cout << "Fetch Response. SM: " << shared_sm->get_sid() << ". Subcore: " << m_subcore_id << ". Warp_ID: " << mf->get_wid() << ". PC: " << std::hex << local_pc_response << std::dec << ". Cycle: " << shared_sm->get_current_gpu_cycle() << std::endl;
      //   fflush(stdout);
      // }
      m_inst_fetch_decode_latch.m_valid = true;
      m_warps_of_subcore[subcore_warp_id]->set_last_fetch(
          shared_sm->get_gpu()->gpu_sim_cycle);
      delete mf;
    }
    std::vector<unsigned int> priority_ordered_for_fetch = order_greedy_then_highest_id(shared_sm, m_greedy_pointer_fetch);
    for (auto c_warp_id : priority_ordered_for_fetch) {
      shd_warp_t *c_warp = m_warps_of_subcore[c_warp_id];
      shared_sm->check_if_warp_has_finished_executing_and_can_be_reclaim(c_warp);
      if (c_warp->functional_done()) {
        continue;
      }
      unsigned int sm_warp_id = c_warp->get_warp_id();
      unsigned int subcore_warp_id = translate_warp_id_of_sm_to_subcore(
          sm_warp_id, shared_sm->get_num_subcores());
      assert(subcore_warp_id < m_warps_of_subcore.size() &&
             sm_warp_id < shared_sm->get_config()->max_warps_per_shader);
      bool is_ibuffer_with_space = m_warps_of_subcore[subcore_warp_id]
                                       ->get_IBuffer_remodeled()
                                       ->can_fetch();
      if (is_ibuffer_with_space) {
        address_type local_pc_request = m_warps_of_subcore[subcore_warp_id]
                                            ->get_IBuffer_remodeled()
                                            ->get_next_pc_to_fetch_request();
        // Request different address for different kernels
        unsigned int unique_function_id = c_warp->get_current_unique_function_id_call();
        address_type global_pc_addr =
            shared_sm->from_local_pc_to_global_pc_address(local_pc_request, unique_function_id);
        unsigned int line_size = m_config->m_L0I_config.get_line_sz();
        unsigned int nbytes = num_bytes_cache_req(line_size, local_pc_request);

        // TODO: replace with use of allocator
        // mem_fetch *mf = m_mem_fetch_allocator->alloc()
        mem_access_t acc(INST_ACC_R, global_pc_addr, nbytes, false,
                         shared_sm->get_gpu()->gpgpu_ctx);
        mem_fetch *mf =
            new mem_fetch(acc, NULL /*we don't have an instruction yet*/,
                          READ_PACKET_SIZE, sm_warp_id, shared_sm->get_sid(),
                          shared_sm->get_tpc_id(), shared_sm->get_memory_config(),
                          shared_sm->get_gpu()->gpu_tot_sim_cycle +
                              shared_sm->get_gpu()->gpu_sim_cycle, NULL, NULL, unique_function_id);
        mf->set_subcore(m_subcore_id);
        std::list<cache_event> events;
        enum cache_request_status status;
        if (m_config->perfect_inst_const_cache || m_config->perfect_instruction_cache) {
          status = HIT;
          shader_cache_access_log(shared_sm->get_sid(), INSTRUCTION, 0);
          m_inst_fetch_decode_latch =
              ifetch_buffer_t(local_pc_request, nbytes, subcore_warp_id);
          delete mf;
        } else {
          status = m_L0I->access((new_addr_type)global_pc_addr, mf,
                                 shared_sm->get_current_gpu_cycle(),
                                 events);
        }
        
        // if(shared_sm->get_sid() == 1 && m_subcore_id == 0) {
        //   std::cout << "Fetch Request. SM: " << shared_sm->get_sid() << ". Subcore: " << m_subcore_id << ". Warp_ID: " << mf->get_wid() << ". PC: " << std::hex << local_pc_request << std::dec << ". Status: " << status << ". Cycle: " << shared_sm->get_current_gpu_cycle() << std::endl;
        //   fflush(stdout);
        // }

        if ((status == MISS) || (status == HIT) ||
            (status == IN_L0I_RESPONSE_QUEUE)) {
          m_warps_of_subcore[subcore_warp_id]->set_last_fetch(
              shared_sm->get_gpu()->gpu_sim_cycle);
        } else {
          assert(status == RESERVATION_FAIL);
          m_warps_of_subcore[subcore_warp_id]->get_IBuffer_remodeled()
                                            ->remove_entry(local_pc_request);
        }
        if( (status == RESERVATION_FAIL) ||  (status == IN_L0I_RESPONSE_QUEUE)) {
          delete mf;
        }

        break;
      }
    }
  }
}

void Subcore::assign_warp_to_subcore(shd_warp_t *warp) {
  m_warps_of_subcore.push_back(warp);
  warp->m_subcore = this;
}

void Subcore::finilized_warps_assignation() {
  m_greedy_pointer_issue = m_warps_of_subcore.size() - 1;
  m_greedy_pointer_fetch = m_warps_of_subcore.size() - 1;
}

void Subcore::create_pipeline() {  
  SM *shared_sm = get_sm();
  create_register_file(shared_sm);
  unsigned int num_intermediate_cycles_until_fu_execution = NUM_INTERMEDIATE_CYCLES_UN_BETWEEN_ISSUE_AND_FU_EXECUTION_FOR_FIXED_LATENCY_INST;
  if(!m_config->is_fp32_and_int_unified_pipeline) {
    m_int_pipeline = new functional_unit(nullptr, m_regular_rf, m_config, m_config->max_int_latency, "INT", shared_sm, INTP__OP, true, false, 1, num_intermediate_cycles_until_fu_execution,
      &m_regular_fixed_latency_rf_write_queue, m_config->max_size_register_file_write_queue_for_fixed_latency_instructions, false, TraceEnhancedOperandType::REG);
    m_all_subcore_ex_pipelines.push_back(m_int_pipeline);
  }
  m_sp_pipeline = new functional_unit(nullptr, m_regular_rf, m_config, m_config->max_sp_latency, "SP", shared_sm, SP__OP, true, false, 1, num_intermediate_cycles_until_fu_execution,
      &m_regular_fixed_latency_rf_write_queue, m_config->max_size_register_file_write_queue_for_fixed_latency_instructions, false, TraceEnhancedOperandType::REG);
  m_uniform_pipeline = new functional_unit(nullptr, m_regular_rf, m_config, m_config->uniform_latency, "UNIFORM", shared_sm, SPECIALIZED__OP, true, false, 1, num_intermediate_cycles_until_fu_execution,
      &m_uniform_fixed_latency_rf_write_queue, m_config->max_size_register_file_write_queue_for_fixed_latency_instructions, false, TraceEnhancedOperandType::UREG);
  m_tensor_pipeline = new functional_unit(nullptr, m_regular_rf, m_config, m_config->tensor_latency, "TENSOR", shared_sm, SPECIALIZED__OP, true, false, 1, NUM_INTERMEDIATE_CYCLES_UN_BETWEEN_ISSUE_AND_FU_EXECUTION_FOR_FIXED_LATENCY_INST_TENSOR_CORE_INSTS_WITH_4_REGS_PER_OP, 
      &m_regular_fixed_latency_rf_write_queue, m_config->max_size_register_file_write_queue_for_fixed_latency_instructions, false, TraceEnhancedOperandType::REG);
  m_branch_pipeline = new functional_unit(nullptr, m_regular_rf, m_config, m_config->branch_latency, "BRANCH", shared_sm, SPECIALIZED__OP, true, false, 1, num_intermediate_cycles_until_fu_execution,
    &m_regular_fixed_latency_rf_write_queue, m_config->max_size_register_file_write_queue_for_fixed_latency_instructions, false, TraceEnhancedOperandType::REG);
  m_sfu_pipeline = new functional_unit_sfu(nullptr, m_regular_rf, m_config, m_config->sfu_latency, "SFU", shared_sm, SFU__OP, true, false, 1, num_intermediate_cycles_until_fu_execution,
    &m_EX_WB_sm_variable_latency_latch, m_config->max_size_register_file_write_queue_for_fixed_latency_instructions, false, TraceEnhancedOperandType::REG);
  m_miscellaneous_with_queue_pipeline = new functional_unit_with_queue( //nullptr, 0////////////////////// ?
      nullptr, m_regular_rf, m_config, m_config->miscellaneous_queue_latency, "MISC_QUEUE", shared_sm, SPECIALIZED__OP, true, true, m_config->miscellaneous_queue_size,
      num_intermediate_cycles_until_fu_execution,  &m_EX_WB_sm_variable_latency_latch, m_config->max_size_register_file_write_queue_for_fixed_latency_instructions, false, 1, 0, TraceEnhancedOperandType::REG);
  m_miscellaneous_no_queue_pipeline = new functional_unit( nullptr, m_regular_rf, m_config, m_config->miscellaneous_no_queue_latency, "MISC_NO_QUEUE", shared_sm, SPECIALIZED__OP,
      true, false, 1, num_intermediate_cycles_until_fu_execution, &m_regular_fixed_latency_rf_write_queue,  m_config->max_size_register_file_write_queue_for_fixed_latency_instructions, false, TraceEnhancedOperandType::REG);
      
  m_memory_unit_subcore = new functional_unit_with_queue( m_EX_MEM_shared_sm_reception_latch, m_regular_rf, m_config, 1, "MEM_SUBCORE_UNIT", shared_sm, MEM__OP, true, true, m_config->memory_subcore_queue_size,
      num_intermediate_cycles_until_fu_execution, nullptr, 0, true, m_config->memory_intermidiate_stages_subcore_unit, 
      m_config->num_cycles_to_wait_to_dispatch_another_inst_from_subcore_to_sm_shared_pipeline_when_is_mem_inst, TraceEnhancedOperandType::NONE);

  if (m_config->is_dp_pipeline_shared_for_subcores) {
    m_dp_pipeline = new functional_unit_with_queue(m_EX_DP_shared_sm_reception_latch, m_regular_rf, m_config, m_config->dp_subcore_max_latency, "DP_SUBCORE_UNIT", shared_sm, DP__OP, true, true, 
        m_config->dp_subcore_queue_size, num_intermediate_cycles_until_fu_execution, nullptr, 0, true, m_config->dp_shared_intermidiate_stages, 
        m_config->num_cycles_to_wait_to_dispatch_another_inst_from_subcore_to_sm_shared_pipeline_when_is_dp_inst, TraceEnhancedOperandType::NONE);
  } else {
    m_dp_pipeline = new functional_unit(nullptr, m_regular_rf, m_config, m_config->max_dp_latency, "DP", shared_sm, DP__OP, true, false, 1, num_intermediate_cycles_until_fu_execution,
        &m_regular_fixed_latency_rf_write_queue, m_config->max_size_register_file_write_queue_for_fixed_latency_instructions, false, TraceEnhancedOperandType::REG);
  }
  
  m_all_subcore_ex_pipelines.push_back(m_sp_pipeline);
  m_all_subcore_ex_pipelines.push_back(m_uniform_pipeline);
  m_all_subcore_ex_pipelines.push_back(m_tensor_pipeline);
  m_all_subcore_ex_pipelines.push_back(m_branch_pipeline);
  m_all_subcore_ex_pipelines.push_back(m_miscellaneous_no_queue_pipeline);
  m_all_subcore_ex_pipelines.push_back(m_memory_unit_subcore);
  m_all_subcore_ex_pipelines.push_back(m_dp_pipeline);
  m_all_subcore_ex_pipelines.push_back(m_sfu_pipeline);
  m_all_subcore_ex_pipelines.push_back(m_miscellaneous_with_queue_pipeline);
}

void Subcore::create_register_file(SM *shared_sm) {
  m_num_regular_rf_banks = m_config->gpgpu_num_reg_banks / shared_sm->get_num_subcores();
  assert( (m_config->gpgpu_num_reg_banks % shared_sm->get_num_subcores()) == 0);
  m_regular_rf = new Register_file(m_num_regular_rf_banks, m_config->num_regular_register_file_read_ports_per_bank, m_config->num_regular_register_file_write_ports_per_bank, m_config->max_latency_regular_register_file_latency, false, false, m_config->max_operands_regular_register_file, m_stats, getptr(), TraceEnhancedOperandType::REG, m_config->is_rf_cache_enabled);
  m_uniform_rf = new Register_file(m_num_regular_rf_banks, MAX_SRC, MAX_DST, m_config->max_latency_regular_register_file_latency, true, true, m_config->max_operands_regular_register_file, m_stats, getptr(), TraceEnhancedOperandType::UREG, false);
  m_regular_rf->init();
  m_uniform_rf->init();
}

void Subcore::create_L0s(mem_fetch_interface *icnt_icache) {
#define STRSIZE 1024
  SM *shared_sm = get_sm();
  char nameL0I[STRSIZE];
  char nameL0C[STRSIZE];
  snprintf(nameL0I, STRSIZE, "L0I_%03d_%d", shared_sm->get_sid(), m_subcore_id);
  m_L0I = new first_level_instruction_cache(
      "L0I", m_config->m_L0I_config, shared_sm->get_sid(),
      get_shader_instruction_cache_id(), icnt_icache, IN_L0_MISS_QUEUE, m_config->is_instruction_prefetching_enabled, m_subcore_id, get_sm(), m_config->num_instruction_prefetches_per_cycle, m_config->ibuffer_coalescing, m_config->prefetch_per_stream_buffer_size, m_config->prefetch_num_stream_buffers);
  m_L0I->initiate_stream_buffers();

  snprintf(nameL0C, STRSIZE, "L0C_%03d_%d", shared_sm->get_sid(), m_subcore_id);
  m_L0C_cache = new read_only_cache(nameL0C, m_config->m_L0C_config, shared_sm->get_sid(),
                              get_shader_constant_cache_id(), icnt_icache,
                              IN_L1C_MISS_QUEUE);
}

first_level_instruction_cache* Subcore::get_L0I() { return m_L0I; }
read_only_cache* Subcore::get_L0C() { return m_L0C_cache; }

void Subcore::get_L0I_sub_stats(struct cache_sub_stats &css) const {
  m_L0I->get_sub_stats(css);
}

register_set_uniptr *Subcore::get_EX_WB_sm_shared_units_latch() {
  return &m_EX_WB_sm_shared_units_latch;
}

unsigned int Subcore::get_subcore_id() { return m_subcore_id; }

SM *Subcore::get_sm() { 
  return m_sm;
}

void Subcore::manage_instruction_operand_stats(SM *shared_sm, warp_inst_t *pI) {
  unsigned int first_read_operand = pI->get_extra_trace_instruction_info().get_num_destination_registers();
  if(pI->get_extra_trace_instruction_info().has_destination_registers()) {
    unsigned int num_of_accesses = get_number_of_uses_per_operand(pI->get_extra_trace_instruction_info(), pI->get_extra_trace_instruction_info().get_operand(0).get_operand_reg_number(), 0, pI->get_extra_trace_instruction_info().get_operand(0).get_operand_type());
    switch(pI->get_extra_trace_instruction_info().get_operand(0).get_operand_type()) {
      case TraceEnhancedOperandType::MREF: // Only these two are used for power compute
      case TraceEnhancedOperandType::REG:
        manage_operand_stat(shared_sm, pI, num_of_accesses, &Subcore::inc_regular_regfile_writes);
        break;
      case TraceEnhancedOperandType::UREG:
        manage_operand_stat(shared_sm, pI, num_of_accesses, &Subcore::inc_uniform_regfile_writes);
        break;
      case TraceEnhancedOperandType::PRED:
        manage_operand_stat(shared_sm, pI, num_of_accesses, &Subcore::inc_predicate_regfile_writes);
        break;
      case TraceEnhancedOperandType::UPRED:
        manage_operand_stat(shared_sm, pI, num_of_accesses, &Subcore::inc_uniform_predicate_regfile_writes);
        break;
      default:
        manage_operand_stat(shared_sm, pI, num_of_accesses, &Subcore::incnon_rf_operands);
        break;
    }
  }

  for(unsigned int i = first_read_operand; i < pI->get_extra_trace_instruction_info().get_num_operands(); i++) {
    unsigned int reg_id = pI->get_extra_trace_instruction_info().get_operand(i).get_operand_reg_number();
    TraceEnhancedOperandType reg_type = pI->get_extra_trace_instruction_info().get_operand(i).get_operand_type();
    if(pI->get_extra_trace_instruction_info().get_operand(i).get_has_reg() && !is_reserved_reg(reg_id, reg_type)) {
      unsigned int num_of_accesses = get_number_of_uses_per_operand(pI->get_extra_trace_instruction_info(), reg_id , i, reg_type);
      if(pI->is_tensor_core_op() && num_of_accesses == 4) {
        pI->m_is_tensor_core_op_with_4_registers_per_op = true;
      }
      switch(reg_type) {
        case TraceEnhancedOperandType::MREF: // Only these two are used for power compute
        case TraceEnhancedOperandType::REG:
          manage_operand_stat(shared_sm, pI, num_of_accesses, &Subcore::inc_regular_regfile_reads);
          break;
        case TraceEnhancedOperandType::UREG:
          manage_operand_stat(shared_sm, pI, num_of_accesses, &Subcore::inc_uniform_regfile_reads);
          break;
        case TraceEnhancedOperandType::PRED:
          manage_operand_stat(shared_sm, pI, num_of_accesses, &Subcore::inc_predicate_regfile_reads);
          break;
        case TraceEnhancedOperandType::UPRED:
          manage_operand_stat(shared_sm, pI, num_of_accesses, &Subcore::inc_uniform_predicate_regfile_reads);
          break;
        case TraceEnhancedOperandType::CBANK:
          manage_operand_stat(shared_sm, pI, num_of_accesses, &Subcore::inc_constant_cache_reads);
          break;
        default: // Operands that might seems like registers but we are not considering like that. (e.g SR, SB, B)
          manage_operand_stat(shared_sm, pI, num_of_accesses, &Subcore::incnon_rf_operands);
          break;
      }
    }else {
      // Operands that does not have registers as source (e.g. immediate values)
      // For the moment, we count reserved registers (RZ, URZ, PT, UPT) as non RF accesses.
      manage_operand_stat(shared_sm, pI, 1, &Subcore::incnon_rf_operands);
      break;
    }
  }
}


void Subcore::manage_operand_stat(SM *shared_sm, const warp_inst_t *pI, unsigned int num_accesses_per_operand,
                                  void (Subcore::*increase_stat)(unsigned int, SM *shared_sm)) {
  if (shared_sm->get_config()->gpgpu_clock_gated_reg_file) {
    unsigned active_count = 0;
    for (unsigned i = 0; i < shared_sm->get_config()->warp_size;
         i = i + shared_sm->get_config()->n_regfile_gating_group) {
      for (unsigned j = 0; j < shared_sm->get_config()->n_regfile_gating_group;
           j++) {
        if (pI->get_active_mask().test(i + j)) {
          active_count += shared_sm->get_config()->n_regfile_gating_group;
          break;
        }
      }
    }
    (this->*increase_stat)(num_accesses_per_operand * active_count, shared_sm);
  } else {
    (this->*increase_stat)(num_accesses_per_operand * (shared_sm->get_config()->warp_size), shared_sm);
  }
}

void Subcore::inc_regular_regfile_reads(unsigned int active_count, SM *shared_sm) {
  shared_sm->m_sm_stats.m_stats_map["total_num_regular_regfile_reads"]->increment_with_integer(active_count);
  get_sm()->incregfile_reads(active_count);
}

void Subcore::inc_regular_regfile_writes(unsigned int active_count, SM *shared_sm) {
  shared_sm->m_sm_stats.m_stats_map["total_num_regular_regfile_writes"]->increment_with_integer(active_count);
  get_sm()->incregfile_writes(active_count); 
}

void Subcore::inc_uniform_regfile_reads(unsigned int active_count, SM *shared_sm) {
  shared_sm->m_sm_stats.m_stats_map["total_num_uniform_regfile_reads"]->increment_with_integer(active_count);
}

void Subcore::inc_uniform_regfile_writes(unsigned int active_count, SM *shared_sm) {
  shared_sm->m_sm_stats.m_stats_map["total_num_uniform_regfile_writes"]->increment_with_integer(active_count);
}

void Subcore::inc_predicate_regfile_reads(unsigned int active_count, SM *shared_sm) {
  shared_sm->m_sm_stats.m_stats_map["total_num_predicate_regfile_reads"]->increment_with_integer(active_count);
}

void Subcore::inc_predicate_regfile_writes(unsigned int active_count, SM *shared_sm) {
  shared_sm->m_sm_stats.m_stats_map["total_num_predicate_regfile_writes"]->increment_with_integer(active_count);
}

void Subcore::inc_uniform_predicate_regfile_reads(unsigned int active_count, SM *shared_sm) {
  shared_sm->m_sm_stats.m_stats_map["total_num_uniform_predicate_regfile_reads"]->increment_with_integer(active_count);
}

void Subcore::inc_uniform_predicate_regfile_writes(unsigned int active_count, SM *shared_sm) {
  shared_sm->m_sm_stats.m_stats_map["total_num_uniform_predicate_regfile_writes"]->increment_with_integer(active_count);
}

void Subcore::inc_constant_cache_reads(unsigned int active_count, SM *shared_sm) { 
  shared_sm->m_sm_stats.m_stats_map["total_num_constant_cache_reads"]->increment_with_integer(active_count);
}

void Subcore::incnon_rf_operands(unsigned int active_count, SM *shared_sm) { 
  get_sm()->incnon_rf_operands(active_count); 
}