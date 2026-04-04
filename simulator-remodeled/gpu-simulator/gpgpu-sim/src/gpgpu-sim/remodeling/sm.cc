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


#include "sm.h"

#include "../../../../trace-driven/trace_driven.h"
#include "../../abstract_hardware_model.h"
#include "../../cuda-sim/cuda-sim.h"
#include "../../cuda-sim/ptx_ir.h"

#include "../gpu-sim.h"
#include "l0_icnt.h"
#include "../shader.h"
#include "../shader_trace.h"
#include "../stat-tool.h"

#include "first_level_instruction_cache.h"
#include "functional_unit.h"
#include "warp_dependency_state.h"

#include "../../../../../util/traces_enhanced/src/traced_operand.h"


#define STRSIZE 1024

unsigned int translate_warp_id_of_sm_to_subcore(unsigned int warp_id,
                                                unsigned int num_subcores) {
  return warp_id / num_subcores;
}

TraceEnhancedOperandType get_reg_type_eval(traced_operand& op) {
  TraceEnhancedOperandType reg_type = op.get_operand_type();
  if( (reg_type == TraceEnhancedOperandType::MREF) || (reg_type == TraceEnhancedOperandType::CBANK) || (reg_type == TraceEnhancedOperandType::DESC)) {
    if(op.get_operand_string().find("UR") != std::string::npos) {
      reg_type = TraceEnhancedOperandType::UREG;
    }else if(op.get_operand_string().find("R") != std::string::npos) {
      reg_type = TraceEnhancedOperandType::REG;
    }
  }
  return reg_type;
}

bool check_is_reserved_regs_remodeling(int reg, TraceEnhancedOperandType reg_type, bool is_trace_mode) {
  bool res = false;
  if(is_trace_mode) {
    if(reg_type == TraceEnhancedOperandType::REG) {
      res = (reg == 255);
    }else if(reg_type == TraceEnhancedOperandType::UREG) {
      res = (reg == 63);
    }else if(reg_type == TraceEnhancedOperandType::PRED) {
      res = (reg == 7);
    }else if(reg_type == TraceEnhancedOperandType::UPRED) {
      res = (reg == 7);
    }
  }
  return res;
}

unsigned int translate_reg_to_global_id(int reg, TraceEnhancedOperandType reg_type) {
  unsigned int global_id = 0;
  if(reg_type == TraceEnhancedOperandType::REG) {
    global_id = reg;
  }else if(reg_type == TraceEnhancedOperandType::UREG) {
    global_id = 256 + reg;
  }else if(reg_type == TraceEnhancedOperandType::PRED) {
    global_id = 512;
    if(reg != PR) {
      global_id += reg;
    }
  }else if(reg_type == TraceEnhancedOperandType::UPRED) {
    global_id = 520;
    if(reg != UPR) {
      global_id += reg;
    }
  }
  return global_id;
}

SM::SM(unsigned int num_subcores, gpgpu_sim *gpu, simt_core_cluster *cluster,
       unsigned shader_id, unsigned tpc_id, const shader_core_config *config,
       const memory_config *mem_config, shader_core_stats *stats)
    : core_t(gpu, NULL, config->warp_size, config->n_thread_per_shader),
      m_sm_stats("SM_" + std::to_string(shader_id)),
      m_barriers(this, config->max_warps_per_shader, config->max_cta_per_core,
                 config->max_barriers_per_cta, config->warp_size) {
  m_sm_id = shader_id;
  m_tpc_id = tpc_id;
  m_num_subcores = num_subcores;
  m_last_inst_gpu_sim_cycle = 0;
  m_last_inst_gpu_tot_sim_cycle = 0;
  // Jin: for concurrent kernels on a SM
  m_occupied_n_threads = 0;
  m_occupied_shmem = 0;
  m_occupied_regs = 0;
  m_occupied_ctas = 0;
  m_occupied_hwtid.reset();
  m_occupied_cta_to_hwtid.clear();
  m_active_warps = 0;
  m_dynamic_warp_id = 0;
  m_gpu = gpu;
  m_cluster = cluster;
  m_stats = stats;
  m_memory_config = mem_config;
  m_config = config;
  m_num_cycles_to_wait_to_dispatch_another_inst_from_subcore_to_sm_shared_pipeline = 0;
  if(config->is_interwarp_coalescing_enabled && is_using_interwarp_coal_warps_waiting_dep_counter()) {
    m_interwarp_coal_warps_waiting_dep_counter = new InterWarp_Coalescing_Waiting_Dep_Counters(config->max_warps_per_shader);
  }
}

SM::~SM() {
  for (auto warp : m_physical_warp) {
    delete warp;
  }
  for(unsigned int i = 0; i < m_EX_MEM_reception_latches_per_subcore.size(); i++) {
    delete m_EX_MEM_reception_latches_per_subcore[i];
  }
  m_EX_WB_sm_shared_units_subcore_latches.clear();
  delete m_ldst_unit_shared_of_sm;
  if(m_config->is_dp_pipeline_shared_for_subcores) {
    delete m_shared_dp_unit;
  }
  delete m_icnt_L0s;
  delete m_icnt;
  delete m_scaling_coeffs;
  free(m_threadState);
  for(unsigned int i = 0; i < m_subcores.size(); i++) {
    delete m_subcores[i];
  }
  if(m_config->is_interwarp_coalescing_enabled && is_using_interwarp_coal_warps_waiting_dep_counter()) {
    delete m_interwarp_coal_warps_waiting_dep_counter;
  }
}

void SM::init() {
  create_logical_structures();
  create_memory_interfaces();

  if (get_gpu()->get_config().g_power_simulation_enabled) {
    m_scaling_coeffs = get_gpu()->get_scaling_coeffs();
  }
}

bool SM::is_using_interwarp_coal_warps_waiting_dep_counter() {
  bool res = (m_config->interwarp_coalescing_selection_policy == DEP_COUNT_WAIT_OLDEST_INST_IBUFFER_GENERIC) ||
              (m_config->interwarp_coalescing_selection_policy == DEP_COUNT_WAIT_OLDEST_INST_IBUFFER_CHECKING_WARP_ID) ||
              (m_config->interwarp_coalescing_selection_policy == DEP_COUNT_WAIT_DETECTED_AT_DECODE_GENERIC) ||
              (m_config->interwarp_coalescing_selection_policy == DEP_COUNT_WAIT_DETECTED_AT_DECODE_CHECKING_WARP_ID);
  return res;
}

void SM::num_cycles_to_stall_SM(unsigned int num_cycles) {
  for(unsigned int i = 0; i < m_subcores.size(); i++) {
    m_subcores[i]->num_cycles_to_stall(num_cycles);
  }
}

void SM::create_gpu_per_sm_stats(Element_stats &all_stats) {
  for(auto stat_name : all_stats.m_stats_name) {
    auto stat = all_stats.m_stats_map[stat_name];
    if(stat->get_allowed_type() == AllowedTypesStats::UNSIGNED_LONG_LONG) {
      m_sm_stats.add_unsigned_long_long_stat(stat->get_name(), stat->get_allowed_type(), stat->get_value(), stat->get_between_name_and_value(), stat->get_suffix(),
                                            stat->get_is_reset_allowed() ,stat->get_is_erase_after_gather_in_sm(), true);
    }else if(stat->get_allowed_type() == AllowedTypesStats::DOUBLE) {
      m_sm_stats.add_double_stat(stat->get_name(), stat->get_allowed_type(), stat->get_value(), stat->get_between_name_and_value(), stat->get_suffix(),
                                            stat->get_is_reset_allowed() ,stat->get_is_erase_after_gather_in_sm(), true);
    }else {
      std::cout << "Error. Stat type not allowed" << std::endl;
      stat->print(stdout);
      fflush(stdout);
      abort();
    }
    
  }
}

void SM::reset_cycless_access_history() {
  m_ldst_unit_shared_of_sm->reset_coalescingHistory();
}

void SM::gather_gpu_per_sm_stats(Element_stats &all_stats, coalescingStatsAcrossSms& coal_stats_l1d, coalescingStatsAcrossSms& coal_stats_const, coalescingStatsAcrossSms& coal_stats_sharedmem) {
  for(auto stat_name : all_stats.m_stats_name) {
    auto stat_value = m_sm_stats.m_stats_map[stat_name]->get_value();
    all_stats.m_stats_map[stat_name]->increment_with_integer(stat_value);
    if(all_stats.m_stats_map[stat_name]->get_is_erase_after_gather_in_sm()) {
      m_sm_stats.m_stats_map[stat_name]->reset();
    }
  }
  coal_stats_l1d.addStats(m_ldst_unit_shared_of_sm->get_coalescingStatPerSm_l1d());
  coal_stats_const.addStats(m_ldst_unit_shared_of_sm->get_coalescingStatPerSm_const());
  coal_stats_sharedmem.addStats(m_ldst_unit_shared_of_sm->get_coalescingStatPerSm_sharedmem());
}

void SM::gather_gpu_per_sm_single_stat(Element_stats &all_stats, std::string stat_name) {
  auto stat_value = m_sm_stats.m_stats_map[stat_name]->get_value();
  all_stats.m_stats_map[stat_name]->increment_with_integer(stat_value);
  if(all_stats.m_stats_map[stat_name]->get_is_erase_after_gather_in_sm()) {
    m_sm_stats.m_stats_map[stat_name]->reset();
  }
}

void SM::increment_sm_stat_by_integer(std::string stat_name, int val_to_increment) {
  m_sm_stats.m_stats_map[stat_name]->increment_with_integer(val_to_increment);
}

void SM::cycle() {
  if (!isactive() && get_not_completed() == 0) return;

  m_stats->shader_cycles[m_sm_id]++;

  // MOD. Begin. Custom Stats
  m_stats->shader_cycles_per_kernel[m_stats->m_current_kernel_pos][m_sm_id]++;
  m_stats->shader_active_warps_per_kernel[m_stats->m_current_kernel_pos]
                                         [m_sm_id] += m_active_warps;
  m_stats->shader_maximum_theoretical_warps_per_kernel
      [m_stats->m_current_kernel_pos][m_sm_id] +=
      m_config->max_warps_per_shader;
  // MOD. End. Custom Stats

  L0_icnt *l0_icnt = static_cast<L0_icnt *>(m_icnt_L0s);
  l0_icnt->cycle();
  m_L1I_L1_half_C_cache->cycle();

  if(m_config->is_dp_pipeline_shared_for_subcores) {
    m_shared_dp_unit->cycle();
  }

  m_ldst_unit_shared_of_sm->cycle();

  if(m_config->is_interwarp_coalescing_enabled && 
        ((m_config->interwarp_coalescing_selection_policy == DEP_COUNT_WAIT_OLDEST_INST_IBUFFER_GENERIC) ||
        (m_config->interwarp_coalescing_selection_policy == DEP_COUNT_WAIT_OLDEST_INST_IBUFFER_CHECKING_WARP_ID))) {
    m_interwarp_coal_warps_waiting_dep_counter->clear();
  }
  
  for (auto subcore : m_subcores) {
    subcore->cycle();
  }

  consume_pending_wait_barrier_actions(m_pending_wait_barrier_increments);
  consume_pending_wait_barrier_actions(m_pending_wait_barrier_decrements);
  if(m_num_cycles_to_wait_to_dispatch_another_inst_from_subcore_to_sm_shared_pipeline > 0) {
    m_num_cycles_to_wait_to_dispatch_another_inst_from_subcore_to_sm_shared_pipeline--;
  }
}

void SM::set_num_cycles_to_wait_to_dispatch_another_inst_from_subcore_to_sm_shared_pipeline(unsigned int num_cycles) {
  m_num_cycles_to_wait_to_dispatch_another_inst_from_subcore_to_sm_shared_pipeline = num_cycles;
}

bool SM::can_send_inst_from_subcore_to_sm_shared_pipeline() const {
  return (m_num_cycles_to_wait_to_dispatch_another_inst_from_subcore_to_sm_shared_pipeline == 0);
}

void SM::consume_pending_wait_barrier_actions(std::stack<Wait_Barrier_Entry_Modifier> &actions) {
  while (!actions.empty()) {
    Wait_Barrier_Entry_Modifier current_action = actions.top();
    m_physical_warp[current_action.sm_warp_id]->get_dependency_state()->action_over_wait_barrier(&current_action);
    actions.pop();
  }
}

void SM::add_pending_wait_barrier_decrement(warp_inst_t *inst,
                                            Wait_Barrier_Type barrier_type, unsigned int barrier_id) {                                    
  m_pending_wait_barrier_decrements.push(Wait_Barrier_Entry_Modifier(
    inst->warp_id(), barrier_id, barrier_type, Wait_Barrier_Action::DECREASE_COUNTER, inst->pc));
}

void SM::add_pending_wait_barrier_increment(warp_inst_t *inst,
                                            Wait_Barrier_Type barrier_type, unsigned int barrier_id) {                                    
  m_pending_wait_barrier_increments.push(Wait_Barrier_Entry_Modifier(
    inst->warp_id(), barrier_id, barrier_type, Wait_Barrier_Action::INCREASE_COUNTER, inst->pc));
}


void SM::instruction_retirement(warp_inst_t *instruction) {
  unsigned int warp_id = instruction->warp_id();
  // if(m_sm_id == 0 && warp_id == 0) { {
  //   std::cout << "WB. SM: " << m_sm_id << ". Subcore: " << instruction->get_subcore_id() << ". Warp_ID: " << warp_id << ". PC: " << std::hex << instruction->pc << std::dec << ". Cycle: " << get_current_gpu_cycle() << std::endl;
  //   fflush(stdout);
  // }
  bool use_traditional_scoreboarding = !m_physical_warp[warp_id]->get_kernel_info()->is_captured_from_binary;
  if (use_traditional_scoreboarding ||
    m_config->is_remodeling_scoreboarding_enabled ||
    !m_config->is_trace_mode) {
    if ( (use_traditional_scoreboarding && m_config->is_trace_mode) || (m_config->is_trace_mode && m_config->is_remodeling_scoreboarding_enabled) ) {
      if ((m_scoreboard_WAR->getMode() ==
           scoreboard_reads_mode::RELEASE_AT_WB)) {
        m_scoreboard_WAR->releaseRegisters_remodeling(instruction);
      }
      m_scoreboard->releaseRegisters_remodeling(instruction);
    } else {
      if ((m_scoreboard_WAR->getMode() ==
           scoreboard_reads_mode::RELEASE_AT_WB)) {
        m_scoreboard_WAR->releaseRegisters(instruction);
      }
      m_scoreboard->releaseRegisters(instruction);
    }

  } else {
    if (instruction->get_extra_trace_instruction_info()
            .get_control_bits()
            .get_is_new_write_barrier()) {
      add_pending_wait_barrier_decrement(instruction, Wait_Barrier_Type::WRITE_WAIT_BARRIER,
      instruction->get_extra_trace_instruction_info().get_control_bits().get_id_new_write_barrier());       
    }
    if(instruction->m_is_ldgsts) {
      m_physical_warp[warp_id]->get_dependency_state()->decrease_num_pending_ldgsts();
    }
  }
  m_physical_warp[warp_id]->dec_inst_in_pipeline();
  warp_inst_complete(*instruction);
  get_gpu()->gpu_sim_insn_last_update_sid = get_sid();
  get_gpu()->gpu_sim_insn_last_update = get_gpu()->gpu_sim_cycle;
  set_last_inst_gpu_sim_cycle(get_gpu()->gpu_sim_cycle);
  set_last_inst_gpu_tot_sim_cycle(get_gpu()->gpu_tot_sim_cycle);
  instruction->clear();
}

void SM::issue_warp(register_set_uniptr &pipe_reg_set, warp_inst_t *next_inst,
                    const active_mask_t &active_mask, unsigned warp_id,
                    unsigned subcore_id, bool use_traditional_scoreboarding) {
  assert(next_inst->valid());
  std::unique_ptr<warp_inst_t> &pipe_reg = pipe_reg_set.get_free_smartptr(); 
  std::unique_ptr<warp_inst_t> inst_smart(next_inst);
  move_warp_uniptr(pipe_reg, inst_smart);

  m_physical_warp[warp_id]->get_IBuffer_remodeled()->issued();

  pipe_reg->issue(active_mask, warp_id,
                     m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle,
                     m_physical_warp[warp_id]->get_dynamic_warp_id(),
                     subcore_id);  // dynamic instruction information
  int num_active_threads = pipe_reg->active_count();

  if (num_active_threads > 0) {
    m_sm_stats.m_stats_map["warp_occ_dist" + std::to_string(num_active_threads)]->increment_with_integer(1);
  }
  
  func_exec_inst(*pipe_reg);

  // if(m_sm_id == 0 && warp_id == 0) { //&& get_stats()->m_last_kernel_id == 11) { // && warp_id == 0) {
  //   std::cout << "Issue. SM: " << m_sm_id << ". Subcore: " << subcore_id << ". Warp_ID: " << warp_id << ". PC: " << std::hex << pipe_reg->pc << ". Next traced PC:" << pipe_reg->next_traced_pc << std::dec << ". Cycle: " << get_current_gpu_cycle() << std::endl;
  //   fflush(stdout);
  // }

  // if(m_sm_id == 0 && subcore_id == 0 && ((*pipe_reg)->pc==0x260 || (*pipe_reg)->pc==0x2c0)) { // && warp_id == 0) {
  //   std::cout << "Measure. Issue. SM: " << m_sm_id << ". Subcore: " << subcore_id << ". Warp_ID: " << warp_id << ". PC: " << std::hex << (*pipe_reg)->pc << std::dec << ". Cycle: " << get_current_gpu_cycle() << std::endl;
  //   fflush(stdout);
  // }

  m_stats->warp_issues_from_last_power_sample[m_sm_id]
                                             [warp_id]++;  // MOD. Custom
                                                           // powermodel stats

  if ((pipe_reg->op == BARRIER_OP) || (pipe_reg->op == MEMORY_BARRIER_OP)) {
    if(pipe_reg->op == MEMORY_BARRIER_OP) {
      pipe_reg->m_num_cycles_to_stall_SM = m_config->num_cycles_to_stall_SM_at_gpu_memory_barrier;
      if(m_config->is_trace_mode && pipe_reg->get_extra_trace_instruction_info().get_is_system_memory_barrier()) {
        pipe_reg->m_num_cycles_to_stall_SM = m_config->num_cycles_to_stall_SM_at_system_memory_barrier;
      }else if(m_config->is_trace_mode && pipe_reg->get_extra_trace_instruction_info().get_is_cta_memory_barrier()) {
        pipe_reg->m_num_cycles_to_stall_SM = m_config->num_cycles_to_stall_SM_at_cta_memory_barrier;
      }
      m_physical_warp[warp_id]->set_membar();
    }
    m_physical_warp[warp_id]->store_info_of_last_inst_at_barrier(pipe_reg.get());
    m_barriers.warp_reaches_barrier(m_physical_warp[warp_id]->get_cta_id(),
                                    warp_id, pipe_reg.get());    
  }else if(pipe_reg->op == GRID_BARRIER_OP) {
    m_physical_warp[warp_id]->set_gridbar();
    m_physical_warp[warp_id]->store_info_of_last_inst_at_barrier(pipe_reg.get());
  }

  ib_ooo_simt_info ib_ooo_simt_status;
  ib_ooo_simt_status.m_is_inst_reissued =
      pipe_reg->get_vpreg_need_to_reissue();
  if (!m_config->is_trace_mode) {
    updateSIMTStack(warp_id, pipe_reg.get());
  }
  if (use_traditional_scoreboarding ||
      m_config->is_remodeling_scoreboarding_enabled ||
      !m_config->is_trace_mode) {

    if ( (use_traditional_scoreboarding && m_config->is_trace_mode) || (m_config->is_trace_mode && m_config->is_remodeling_scoreboarding_enabled) ) {
      m_scoreboard->reserveRegisters_remodeling(pipe_reg.get());
      if (m_scoreboard_WAR->isEnabled()) {
        m_scoreboard_WAR->reserveRegisters_remodeling(pipe_reg.get());
      }
    } else {
      m_scoreboard->reserveRegisters(pipe_reg.get());
      if (m_scoreboard_WAR->isEnabled()) {
        m_scoreboard_WAR->reserveRegisters(pipe_reg.get());
      }
    }
  } else {
    bool is_yield = pipe_reg
                        ->get_extra_trace_instruction_info()
                        .get_control_bits()
                        .get_is_yield();
    unsigned int stall_count = pipe_reg
                                   ->get_extra_trace_instruction_info()
                                   .get_control_bits()
                                   .get_stall_count();
    if (is_yield) {
      m_physical_warp[warp_id]->get_dependency_state()->set_yield();
      stall_count = (stall_count == 0) ? m_config->num_stall_cycles_wait_after_bits_stall_0_and_yield : stall_count;
    }
    m_physical_warp[warp_id]->get_dependency_state()->set_stall_counter(
        stall_count);
    if( pipe_reg->m_is_ldgsts ) {
      m_physical_warp[warp_id]->get_dependency_state()->increase_num_pending_ldgsts();
    }
  }

  m_physical_warp[warp_id]->set_next_pc(pipe_reg->pc + pipe_reg->isize);

  if(m_config->is_trace_mode) {
    assert(pipe_reg->unique_function_id == m_physical_warp[warp_id]->get_current_unique_function_id_call());
    bool is_any_thread_active = pipe_reg->get_active_mask().any();
    if(pipe_reg->op == CALL_OPS && is_any_thread_active) {
      new_addr_type call_address = pipe_reg->get_first_addr_valid();
      search_func_addr_result search_result = get_gpu()->get_extra_trace_info().search_function_addr(call_address);
      if(search_result.m_has_been_traced) {
        m_physical_warp[warp_id]->push_function_call(search_result.m_unique_function_id, pipe_reg->get_active_mask());
      }else if(pipe_reg->get_extra_trace_instruction_info().get_is_call_or_ret_with_relative()) {
        m_physical_warp[warp_id]->push_function_call(pipe_reg->unique_function_id, pipe_reg->get_active_mask());
      }else{
        assert(pipe_reg->next_traced_pc == (pipe_reg->pc + pipe_reg->isize) );
      }
    } else if(pipe_reg->op == RET_OPS && is_any_thread_active) {
      m_physical_warp[warp_id]->pop_function_call(pipe_reg->get_active_mask());
    }
  }
}

void SM::func_exec_inst(warp_inst_t &inst) {
  if (m_config->is_trace_mode) {
    for (unsigned t = 0; t < m_warp_size; t++) {
      if (inst.active(t)) {
        unsigned warpId = inst.warp_id();
        unsigned tid = m_warp_size * warpId + t;

        // virtual function
        checkExecutionStatusAndUpdate(inst, t, tid);
      }
    }

    // here, we generate memory acessess and set the status if thread (done?)
    if (inst.is_load() || inst.is_store()) {
      inst.generate_mem_accesses();
    }

    trace_shd_warp_t *m_trace_warp =
        static_cast<trace_shd_warp_t *>(m_physical_warp[inst.warp_id()]);
    if (m_trace_warp->trace_done() && m_trace_warp->functional_done()) {
      m_trace_warp->get_IBuffer_remodeled()->flush(true);
      m_barriers.warp_exit(inst.warp_id());
    }
  } else {
    execute_warp_inst_t(inst);
    if (inst.is_load() || inst.is_store()) {
      inst.generate_mem_accesses();
    }
  }
}

void SM::check_if_warp_has_finished_executing_and_can_be_reclaim(
    shd_warp_t *warp) {
  unsigned int warp_id = warp->get_warp_id();
  if (warp->hardware_done() && !warp->done_exit() &&
      !m_scoreboard->pendingWrites(warp_id) &&
      !m_scoreboard_WAR->pendingReads(warp_id) && !warp->get_dependency_state()->are_pending_dependencies() &&
      !warp->is_atomic_pending()) {
    bool did_exit = false;
    for (unsigned t = 0; t < m_config->warp_size; t++) {
      unsigned tid = warp_id * m_config->warp_size + t;
      if (m_threadState[tid].m_active == true) {
        m_threadState[tid].m_active = false;
        unsigned cta_id = warp->get_cta_id();
        if (m_thread[tid] == NULL) {
          register_cta_thread_exit(cta_id, warp->get_kernel_info());
        } else {
          register_cta_thread_exit(cta_id, &(m_thread[tid]->get_kernel()));
        }
        m_not_completed -= 1;
        m_active_threads.reset(tid);
        did_exit = true;
      }
    }
    if (did_exit) {
      warp->set_done_exit();
      warp->get_dependency_state()->reset();
    }
    unsigned int m_subcore_id = warp_id %  m_num_subcores;
    m_subcores[m_subcore_id]->decrease_active_warp();
    --m_active_warps;
    assert(m_active_warps >= 0);
  }
}

unsigned int SM::get_kernel_id(unsigned warp_id) {
  return m_physical_warp[warp_id]->m_kernel_id;
}

void SM::init_warps(unsigned cta_id, unsigned start_thread, unsigned end_thread,
                    unsigned ctaid, int cta_size, kernel_info_t &kernel) {
  address_type start_pc = next_pc(start_thread);
  unsigned kernel_id = kernel.get_uid();
  if (m_config->model == POST_DOMINATOR) {
    unsigned start_warp = start_thread / m_config->warp_size;
    unsigned warp_per_cta = cta_size / m_config->warp_size;
    unsigned end_warp = end_thread / m_config->warp_size +
                        ((end_thread % m_config->warp_size) ? 1 : 0);

    m_stats->number_of_warps_per_kernel[m_stats->m_current_kernel_pos] +=
        end_warp - start_warp;  // MOD. Custom Stats
    for (unsigned i = start_warp; i < end_warp; ++i) {
      unsigned n_active = 0;
      simt_mask_t active_threads;
      for (unsigned t = 0; t < m_config->warp_size; t++) {
        unsigned hwtid = i * m_config->warp_size + t;
        if (hwtid < end_thread) {
          n_active++;
          assert(!m_active_threads.test(hwtid));
          m_active_threads.set(hwtid);
          active_threads.set(t);
        }
      }
      m_simt_stack[i]->launch(start_pc, active_threads);

      if (m_gpu->resume_option == 1 && kernel_id == m_gpu->resume_kernel &&
          ctaid >= m_gpu->resume_CTA && ctaid < m_gpu->checkpoint_CTA_t) {
        char fname[2048];
        snprintf(fname, 2048, "checkpoint_files/warp_%d_%d_simt.txt",
                 i % warp_per_cta, ctaid);
        unsigned pc, rpc;
        m_simt_stack[i]->resume(fname);
        m_simt_stack[i]->get_pdom_stack_top_info(&pc, &rpc);
        for (unsigned t = 0; t < m_config->warp_size; t++) {
          if (m_thread != NULL) {
            m_thread[i * m_config->warp_size + t]->set_npc(pc);
            m_thread[i * m_config->warp_size + t]->update_pc();
          }
        }
        start_pc = pc;
      }
      unsigned int subcore_id = i % m_num_subcores;
      m_subcores[subcore_id]->increase_active_warp();
      m_physical_warp[i]->init(start_pc, cta_id, i, active_threads,
                               m_dynamic_warp_id, m_sm_id);
      m_physical_warp[i]->m_kernel_id = kernel_id;
      ++m_dynamic_warp_id;
      m_not_completed += n_active;
      ++m_active_warps;
    }
  }
  if (m_config->is_trace_mode) {
    std::string kernel_name = kernel.name();
    unsigned int kernel_unique_function_id = get_gpu()->get_extra_trace_info().get_unique_function_id(kernel_name);
    // Init traces
    unsigned start_warp = start_thread / m_config->warp_size;
    unsigned end_warp = end_thread / m_config->warp_size +
                        ((end_thread % m_config->warp_size) ? 1 : 0);

    // init_traces(start_warp, end_warp, kernel);
    std::vector<std::map<address_type, traced_instructions_by_pc> *> threadblock_traces;
    std::vector<std::vector<address_type> *> threadblock_traced_pcs;
    for (unsigned i = start_warp; i < end_warp; ++i) {
      trace_shd_warp_t *m_trace_warp =
          static_cast<trace_shd_warp_t *>(m_physical_warp[i]);
      m_trace_warp->clear();
      threadblock_traces.push_back(&(m_trace_warp->map_warp_traces));
      threadblock_traced_pcs.push_back(&(m_trace_warp->traced_pcs));
    }
    trace_kernel_info_t &trace_kernel =
        static_cast<trace_kernel_info_t &>(kernel);
    trace_kernel.get_next_threadblock_traces(threadblock_traces, threadblock_traced_pcs, m_gpu, m_gpu->get_extra_trace_info());

    // set the pc from the traces and ignore the functional model
    for (unsigned i = start_warp; i < end_warp; ++i) {
      m_physical_warp[i]->push_function_call(kernel_unique_function_id, m_physical_warp[i]->get_active_mask());
      trace_shd_warp_t *m_trace_warp =
          static_cast<trace_shd_warp_t *>(m_physical_warp[i]);
      m_trace_warp->set_next_pc(m_trace_warp->get_start_trace_pc());
      m_trace_warp->set_kernel(&trace_kernel);
    }
  }
}

void SM::checkExecutionStatusAndUpdate(warp_inst_t &inst, unsigned t,
                                       unsigned tid) {
  if (m_config->is_trace_mode) {
    if (inst.isatomic()) {
      m_physical_warp[inst.warp_id()]->inc_n_atomic();
    }

    if (inst.space.is_local() && (inst.is_load() || inst.is_store())) {
      new_addr_type localaddrs[MAX_ACCESSES_PER_INSN_PER_THREAD];
      unsigned num_addrs;
      num_addrs = translate_local_memaddr(
          inst.get_addr(t), tid,
          m_config->n_simt_clusters * m_config->n_simt_cores_per_cluster,
          inst.data_size, (new_addr_type *)localaddrs);
      inst.set_addr(t, (new_addr_type *)localaddrs, num_addrs);
    }

    if (inst.op == EXIT_OPS) {
      m_physical_warp[inst.warp_id()]->set_completed(t);
    }
  } else {
    if (inst.isatomic()) m_physical_warp[inst.warp_id()]->inc_n_atomic();
    if (inst.space.is_local() && (inst.is_load() || inst.is_store())) {
      new_addr_type localaddrs[MAX_ACCESSES_PER_INSN_PER_THREAD];
      unsigned num_addrs;
      num_addrs = translate_local_memaddr(
          inst.get_addr(t), tid,
          m_config->n_simt_clusters * m_config->n_simt_cores_per_cluster,
          inst.data_size, (new_addr_type *)localaddrs);
      inst.set_addr(t, (new_addr_type *)localaddrs, num_addrs);
    }
    if (ptx_thread_done(tid)) {
      m_physical_warp[inst.warp_id()]->set_completed(t);
      m_physical_warp[inst.warp_id()]->get_IBuffer_remodeled()->flush(false);
    }

    // PC-Histogram Update
    unsigned warp_id = inst.warp_id();
    unsigned pc = inst.pc;
    for (unsigned t = 0; t < m_config->warp_size; t++) {
      if (inst.active(t)) {
        int tid = warp_id * m_config->warp_size + t;
        cflog_update_thread_pc(m_sm_id, tid, pc);
      }
    }
  }
}

// Returns numbers of addresses in translated_addrs, each addr points to a 4B
// (32-bit) word
unsigned int SM::translate_local_memaddr(address_type localaddr, unsigned tid,
                                         unsigned num_shader, unsigned datasize,
                                         new_addr_type *translated_addrs) {
  // During functional execution, each thread sees its own memory space for
  // local memory, but these need to be mapped to a shared address space for
  // timing simulation.  We do that mapping here.

  address_type thread_base = 0;
  unsigned max_concurrent_threads = 0;
  if (m_config->gpgpu_local_mem_map) {
    // Dnew = D*N + T%nTpC + nTpC*C
    // N = nTpC*nCpS*nS (max concurent threads)
    // C = nS*K + S (hw cta number per gpu)
    // K = T/nTpC   (hw cta number per core)
    // D = data index
    // T = thread
    // nTpC = number of threads per CTA
    // nCpS = number of CTA per shader
    //
    // for a given local memory address threads in a CTA map to contiguous
    // addresses, then distribute across memory space by CTAs from successive
    // shader cores first, then by successive CTA in same shader core
    thread_base =
        4 *
        (kernel_padded_threads_per_cta *
             (m_sm_id + num_shader * (tid / kernel_padded_threads_per_cta)) +
         tid % kernel_padded_threads_per_cta);
    max_concurrent_threads =
        kernel_padded_threads_per_cta * kernel_max_cta_per_shader * num_shader;
  } else {
    // legacy mapping that maps the same address in the local memory space of
    // all threads to a single contiguous address region
    thread_base = 4 * (m_config->n_thread_per_shader * m_sm_id + tid);
    max_concurrent_threads = num_shader * m_config->n_thread_per_shader;
  }
  assert(thread_base < 4 /*word size*/ * max_concurrent_threads);

  // If requested datasize > 4B, split into multiple 4B accesses
  // otherwise do one sub-4 byte memory access
  unsigned num_accesses = 0;

  if (datasize >= 4) {
    // >4B access, split into 4B chunks
    assert(datasize % 4 == 0);  // Must be a multiple of 4B
    num_accesses = datasize / 4;
    assert(num_accesses <= MAX_ACCESSES_PER_INSN_PER_THREAD);  // max 32B
    assert(
        localaddr % 4 ==
        0);  // Address must be 4B aligned - required if accessing 4B per
             // request, otherwise access will overflow into next thread's space
    for (unsigned i = 0; i < num_accesses; i++) {
      address_type local_word = localaddr / 4 + i;
      address_type linear_address = local_word * max_concurrent_threads * 4 +
                                    thread_base + LOCAL_GENERIC_START;
      translated_addrs[i] = linear_address;
    }
  } else {
    // Sub-4B access, do only one access
    assert(datasize > 0);
    num_accesses = 1;
    address_type local_word = localaddr / 4;
    address_type local_word_offset = localaddr % 4;
    assert((localaddr + datasize - 1) / 4 ==
           local_word);  // Make sure access doesn't overflow into next 4B chunk
    address_type linear_address = local_word * max_concurrent_threads * 4 +
                                  local_word_offset + thread_base +
                                  LOCAL_GENERIC_START;
    translated_addrs[0] = linear_address;
  }
  return num_accesses;
}

void SM::create_logical_structures() {
  m_threadState = (thread_ctx_t *)calloc(sizeof(thread_ctx_t),
                                         m_config->n_thread_per_shader);
  m_not_completed = 0;
  m_active_threads.reset();
  m_n_active_cta = 0;
  for (unsigned i = 0; i < MAX_CTA_PER_SHADER; i++) m_cta_status[i] = 0;
  for (unsigned i = 0; i < m_config->n_thread_per_shader; i++) {
    m_thread[i] = NULL;
    m_threadState[i].m_cta_id = -1;
    m_threadState[i].m_active = false;
  }
  create_shd_warp();
  std::string latch_name;
  for (unsigned int i = 0; i < m_num_subcores; i++) {
    latch_name =
        "EX_MEM_shared_reception_latch_for_subcore_" + std::to_string(i);
    register_set_uniptr *mem_sm_reception_latch_for_subcore = new register_set_uniptr(1, latch_name.c_str());
    m_EX_MEM_reception_latches_per_subcore.push_back(mem_sm_reception_latch_for_subcore);
    
    Subcore *subcore = new Subcore(i, m_config, m_stats, this,
                                   &m_EX_DP_shared_sm_reception_latch,
                                   mem_sm_reception_latch_for_subcore);
    subcore->create_pipeline();
    m_subcores.push_back(subcore);
    m_EX_WB_sm_shared_units_subcore_latches.push_back(
        subcore->get_EX_WB_sm_shared_units_latch());
  }

  latch_name = "MEM_icnt_latch"; // Used for LDGSTS communication inside ldst_unit_sm
  register_set_uniptr *mem_sm_icnt_latch = new register_set_uniptr(1, latch_name.c_str());
  m_EX_MEM_reception_latches_per_subcore.push_back(mem_sm_icnt_latch);

  for (unsigned int i = 0; i < m_physical_warp.size(); i++) {
    m_subcores[i % m_num_subcores]->assign_warp_to_subcore(m_physical_warp[i]);
  }

  for (auto subcore : m_subcores) {
    subcore->finilized_warps_assignation();
  }

  m_scoreboard = std::make_shared<Scoreboard>(m_sm_id, m_config->max_warps_per_shader, m_gpu, m_config->is_trace_mode);
  m_scoreboard_WAR =
      std::make_shared<Scoreboard_reads>(m_sm_id, m_config->max_warps_per_shader, m_gpu,
                           m_config->scoreboard_war_reads_mode,
                           m_config->scoreboard_war_max_uses_per_reg,
                           m_config->is_trace_mode, m_stats);
  if (m_config->is_dp_pipeline_shared_for_subcores) {
    std::vector<register_set_uniptr*> m_EX_DP_shared_sm_reception_latches;
    m_EX_DP_shared_sm_reception_latches.push_back(&m_EX_DP_shared_sm_reception_latch);
    m_shared_dp_unit = new functional_unit_shared_sm_part(
        m_EX_WB_sm_shared_units_subcore_latches, m_config,
        m_config->max_dp_latency, "DP_SM_shared", this, DP__OP, false, false, 1,
        m_EX_DP_shared_sm_reception_latches, NUM_INTERMEDIATE_CYCLES_UN_BETWEEN_ISSUE_AND_FU_EXECUTION_FOR_FIXED_LATENCY_INST, nullptr, 0, false, TraceEnhancedOperandType::NONE);
  }
}

void SM::create_memory_interfaces() {
  if (m_config->gpgpu_perfect_mem) {
    m_icnt = new perfect_memory_interface(this, m_cluster);
  } else {
    m_icnt = new shader_memory_interface(this, m_cluster);
  }
  m_mem_fetch_allocator = std::make_shared<shader_core_mem_fetch_allocator>(m_sm_id, m_tpc_id, m_memory_config);

  char name[STRSIZE];
  snprintf(name, STRSIZE, "L1I_%03d", m_sm_id);
  m_L1I_L1_half_C_cache = std::make_shared<read_only_cache>(name, m_config->m_L1I_L1_half_C_cache_config, m_sm_id,
                              get_shader_instruction_cache_id(), m_icnt,
                              IN_L1I_MISS_QUEUE);

  m_icnt_L0s =
      new L0_icnt(m_L1I_L1_half_C_cache.get(), m_gpu, this, m_config->max_reply_allowed_from_L1I,
                   m_config->max_request_allowed_to_L1I,
                   m_config->latency_L0_to_L1,
                   m_config->latency_L1_to_L0);

  for (auto subcore : m_subcores) {
    subcore->create_L0s(m_icnt_L0s);
    static_cast<L0_icnt *>(m_icnt_L0s)
        ->add_L0(static_cast<read_only_cache *>(subcore->get_L0I()));
  }
  for (auto subcore : m_subcores) {
    static_cast<L0_icnt *>(m_icnt_L0s)
        ->add_L0(static_cast<read_only_cache *>(subcore->get_L0C()));
  }
  m_subcore_req_fetch_L1I_priority = 0;
  m_ldst_unit_shared_of_sm = new ldst_unit_sm(
      m_EX_WB_sm_shared_units_subcore_latches,
      m_EX_MEM_reception_latches_per_subcore, m_icnt, m_icnt_L0s, m_mem_fetch_allocator,
      this, m_scoreboard, m_scoreboard_WAR, m_config, m_memory_config, m_stats,
      m_sm_id, m_tpc_id, m_config->memory_sm_prt_size);
  static_cast<L0_icnt *>(m_icnt_L0s)
        ->add_L0(static_cast<read_only_cache *>(m_ldst_unit_shared_of_sm->get_L1C()));
}

void SM::create_shd_warp() {
  if (m_config->is_trace_mode) {
    m_physical_warp.resize(m_config->max_warps_per_shader);
    for (unsigned k = 0; k < m_config->max_warps_per_shader; ++k) {
      m_physical_warp[k] = new trace_shd_warp_t(
          this, m_config->warp_size, m_stats);
    }
  } else {
    m_physical_warp.resize(m_config->max_warps_per_shader);
    for (unsigned k = 0; k < m_config->max_warps_per_shader; ++k) {
      m_physical_warp[k] = new shd_warp_t(
          this, m_config->warp_size, m_stats);
    }
  }
}

unsigned int SM::sim_init_thread(kernel_info_t &kernel,
                                 ptx_thread_info **thread_info, int sid,
                                 unsigned tid, unsigned threads_left,
                                 unsigned num_threads, core_t *core,
                                 unsigned hw_cta_id, unsigned hw_warp_id,
                                 gpgpu_t *gpu) {
  if (m_config->is_trace_mode) {
    if (kernel.no_more_ctas_to_run()) {
      return 0;  // finished!
    }

    if (kernel.more_threads_in_cta()) {
      kernel.increment_thread_id();
    }

    if (!kernel.more_threads_in_cta()) kernel.increment_cta_id();

    return 1;
  } else {
    return ptx_sim_init_thread(kernel, thread_info, sid, tid, threads_left,
                               num_threads, core, hw_cta_id, hw_warp_id, gpu);
  }
}

void SM::reinit(unsigned start_thread, unsigned end_thread,
                bool reset_not_completed) {
  if (reset_not_completed) {
    m_not_completed = 0;
    m_active_threads.reset();

    // Jin: for concurrent kernels on a SM
    m_occupied_n_threads = 0;
    m_occupied_shmem = 0;
    m_occupied_regs = 0;
    m_occupied_ctas = 0;
    m_occupied_hwtid.reset();
    m_occupied_cta_to_hwtid.clear();
    m_active_warps = 0;
  }
  for (unsigned i = start_thread; i < end_thread; i++) {
    m_threadState[i].n_insn = 0;
    m_threadState[i].m_cta_id = -1;
  }
  for (unsigned i = start_thread / m_config->warp_size;
       i < end_thread / m_config->warp_size; ++i) {
    m_physical_warp[i]->reset();
    m_simt_stack[i]->reset();
  }
}

void SM::register_cta_thread_exit(unsigned cta_num, kernel_info_t *kernel) {
  assert(m_cta_status[cta_num] > 0);
  m_cta_status[cta_num]--;
  if (!m_cta_status[cta_num]) {
    // Increment the completed CTAs
    m_sm_stats.m_stats_map["ctas_completed"]->increment_with_integer(1);
    m_n_active_cta--;
    m_barriers.deallocate_barrier(cta_num);
    shader_CTA_count_unlog(m_sm_id, 1);

    SHADER_DPRINTF(
        LIVENESS,
        "GPGPU-Sim uArch: Finished CTA #%u (%lld,%lld), %u CTAs running\n",
        cta_num, m_gpu->gpu_sim_cycle, m_gpu->gpu_tot_sim_cycle,
        m_n_active_cta);

    if (m_n_active_cta == 0) {
      SHADER_DPRINTF(
          LIVENESS,
          "GPGPU-Sim uArch: Empty (last released kernel %u \'%s\').\n",
          kernel->get_uid(), kernel->name().c_str());
      fflush(stdout);

      // Shader can only be empty when no more cta are dispatched
      if (kernel != m_kernel) {
        assert(m_kernel == NULL || !m_gpu->kernel_more_cta_left(m_kernel));
      }
      m_kernel = NULL;
    }

    // Jin: for concurrent kernels on sm
    release_shader_resource_1block(cta_num, *kernel);
    #pragma omp critical
    {
      m_gpu->inc_completed_cta();
      m_gpu->decrease_num_threads_kernel(kernel->get_uid(), kernel->threads_per_cta());
      kernel->dec_running();
      if (!m_gpu->kernel_more_cta_left(kernel)) {
        if (!kernel->running()) {
          SHADER_DPRINTF(LIVENESS,
                        "GPGPU-Sim uArch: GPU detected kernel %u \'%s\' "
                        "finished on shader %u.\n",
                        kernel->get_uid(), kernel->name().c_str(), m_sm_id);

          if (m_kernel == kernel) m_kernel = NULL;
          m_gpu->set_kernel_done(kernel);
        }
      }
    }
  }
}

address_type SM::next_pc(int tid) const {
  if (tid == -1) return -1;
  ptx_thread_info *the_thread = m_thread[tid];
  if (the_thread == NULL) return -1;
  return the_thread
      ->get_pc();  // PC should already be updatd to next PC at this point (was
                   // set in shader_decode() last time thread ran)
}

void SM::set_kernel(kernel_info_t *k) {
  assert(k);
  m_kernel = k;
  //        k->inc_running();
  printf("GPGPU-Sim uArch: Shader %d bind to kernel %u \'%s\'\n", m_sm_id,
         m_kernel->get_uid(), m_kernel->name().c_str());
}
kernel_info_t *SM::get_kernel() { return this->core_t::get_kernel_info(); }
kernel_info_t *SM::get_kernel_info() { return this->core_t::get_kernel_info(); }

unsigned long long SM::get_current_gpu_cycle() {
  return m_gpu->get_current_gpu_cycle();
}

unsigned int SM::get_num_subcores() { return m_num_subcores; }
unsigned int SM::get_sid() const { return m_sm_id; }
unsigned int SM::get_tpc_id() const { return m_tpc_id; }
gpgpu_sim *SM::get_gpu() { return this->core_t::get_gpu(); }
shader_core_mem_fetch_allocator &SM::get_memf_fetch_allocator() { return *m_mem_fetch_allocator; }
read_only_cache* SM::get_L1C() { return m_ldst_unit_shared_of_sm->get_L1C(); }
std::shared_ptr<Scoreboard_reads> SM::get_scoreboard_WAR() { return m_scoreboard_WAR; }
std::shared_ptr<Scoreboard> SM::get_scoreboard() { return m_scoreboard; }
const memory_config *SM::get_memory_config() const { return m_memory_config; }
const shader_core_config *SM::get_config() const { return m_config; }
shader_core_stats *SM::get_stats() { return m_stats; }

std::list<unsigned> SM::get_regs_written(const inst_t &fvt) const {
  std::list<unsigned> result;
  for (unsigned op = 0; op < MAX_REG_OPERANDS; op++) {
    int reg_num = fvt.arch_reg.dst[op];  // this math needs to match that used
                                         // in function_info::ptx_decode_inst
    if (reg_num >= 0)                    // valid register
      result.push_back(reg_num);
  }
  return result;
}

shd_warp_t *SM::get_shd_warp(int id) { return m_physical_warp[id]; }
int SM::get_subcore_req_fetch_L1I_priority() {
  return m_subcore_req_fetch_L1I_priority;
}
void SM::set_subcore_req_fetch_L1I_priority(
    int new_subcore_req_fetch_L1I_priority) {
  m_subcore_req_fetch_L1I_priority = new_subcore_req_fetch_L1I_priority;
}

void SM::set_last_inst_gpu_sim_cycle(
    unsigned long long last_inst_gpu_sim_cycle) {
  m_last_inst_gpu_sim_cycle = last_inst_gpu_sim_cycle;
}
void SM::set_last_inst_gpu_tot_sim_cycle(
    unsigned long long last_inst_gpu_tot_sim_cycle) {
  m_last_inst_gpu_tot_sim_cycle = last_inst_gpu_tot_sim_cycle;
}

bool SM::is_any_subcore_problems_of_fordward_progress() const {
  bool res = false;
  for(auto subcore : m_subcores) {
    res = res || subcore->is_subcore_with_problems_of_fordward_progress();
  }
  return res;
}

bool SM::get_is_loog_enabled() { return m_config->is_loog_enabled; }
RRS *SM::get_loog_rrs() {
  throw std::logic_error(
      "LOOG is not compatible with this new accurate remodeling");
}

void SM::get_pdom_stack_top_info(unsigned tid, unsigned *pc,
                                 unsigned *rpc) const {
  unsigned warp_id = tid / m_config->warp_size;
  m_simt_stack[warp_id]->get_pdom_stack_top_info(pc, rpc);
}

void SM::get_pdom_stack_top_info(unsigned warp_id, const warp_inst_t *pI,
                                 unsigned *pc, unsigned *rpc) {
  if (m_config->is_trace_mode) {
    // In trace-driven mode, we assume no control hazard
    *pc = pI->pc;
    *rpc = pI->pc;
  } else {
    m_simt_stack[warp_id]->get_pdom_stack_top_info(pc, rpc);
  }
}

const active_mask_t &SM::get_active_mask(unsigned warp_id,
                                         const warp_inst_t *pI) {
  if (m_config->is_trace_mode) {
    return pI->get_active_mask();
  } else {
    return m_simt_stack[warp_id]->get_active_mask();
  }
}

void SM::warp_exit(unsigned warp_id) {
  bool done = true;
  for (unsigned i = warp_id * get_config()->warp_size;
       i < (warp_id + 1) * get_config()->warp_size; i++) {
    //		if(this->m_thread[i]->m_functional_model_thread_state &&
    // this->m_thread[i].m_functional_model_thread_state->donecycle()==0) {
    // done = false;
    //		}

    if (m_thread[i] && !m_thread[i]->is_done()) done = false;
  }
  // if (m_warp[warp_id].get_n_completed() == get_config()->warp_size)
  // if (this->m_simt_stack[warp_id]->get_num_entries() == 0)
  if (done) m_barriers.warp_exit(warp_id);
}

void SM::cache_flush() { m_ldst_unit_shared_of_sm->flush(); }

void SM::cache_invalidate() { 
  m_ldst_unit_shared_of_sm->invalidate(); 
  if(m_config->invalidate_instruction_caches_at_kernel_end) {
    m_icnt_L0s->flush();
    m_L1I_L1_half_C_cache->invalidate();
    for(auto subcore : m_subcores) {
      subcore->get_L0I()->invalidate();
    }
  }  
}
void SM::data_cache_invalidate() { 
  m_ldst_unit_shared_of_sm->invalidate(); 
}

void SM::accept_fetch_response(mem_fetch *mf) {
  mf->set_status(IN_SHADER_FETCHED,
                 m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
  m_L1I_L1_half_C_cache->fill(mf, m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
}

void SM::accept_ldst_unit_response(mem_fetch *mf) {
  m_ldst_unit_shared_of_sm->fill(mf);
}

bool SM::fetch_unit_response_buffer_full() const { return false; }

bool SM::ldst_unit_response_buffer_full() const {
  return m_ldst_unit_shared_of_sm->response_buffer_full();
}

// Cuda barrier management

bool SM::check_if_non_released_reduction_barrier(warp_inst_t &inst) {
  unsigned warp_id = inst.warp_id();
  bool bar_red_op = (inst.op == BARRIER_OP) && (inst.bar_type == RED);
  bool non_released_barrier_reduction = false;
  bool warp_stucked_at_barrier = warp_waiting_at_barrier(warp_id);
  bool single_inst_in_pipeline =
      (m_physical_warp[warp_id]->num_issued_inst_in_pipeline() == 1);
  non_released_barrier_reduction =
      single_inst_in_pipeline and warp_stucked_at_barrier and bar_red_op;
  printf("non_released_barrier_reduction=%u\n", non_released_barrier_reduction);
  return non_released_barrier_reduction;
}

bool SM::warp_waiting_at_barrier(unsigned warp_id) const {
  return m_barriers.warp_waiting_at_barrier(warp_id);
}

bool SM::are_all_wait_barrier_ready(unsigned int warp_id) {
  std::vector<Wait_Barrier_Checking> wait_barriers_checking;
  for (unsigned int i = 0; i < m_config->num_wait_barriers_per_warp; i++) {
    wait_barriers_checking.push_back(Wait_Barrier_Checking(i, 0));
  }
  bool are_wait_barriers_ready =
      m_physical_warp[warp_id]
          ->get_dependency_state()
          ->are_wait_barriers_ready(wait_barriers_checking);
  return are_wait_barriers_ready;
}

bool SM::warp_waiting_at_mem_barrier(unsigned warp_id) {
  if (!m_physical_warp[warp_id]->get_membar()) return false;
  bool use_traditional_scoreboarding = !m_physical_warp[warp_id]->get_kernel_info()->is_captured_from_binary || m_config->is_remodeling_scoreboarding_enabled || !m_config->is_trace_mode;
  bool clear_membar = false;
  if (use_traditional_scoreboarding) {
    clear_membar = (!m_scoreboard->pendingWrites(warp_id) && !m_scoreboard_WAR->pendingReads(warp_id)) ;
  }else {
     clear_membar = are_all_wait_barrier_ready(warp_id);
  }
  if(clear_membar) {
    m_physical_warp[warp_id]->clear_membar();
    if (m_gpu->get_config().flush_l1()) {
      // Mahmoud fixed this on Nov 2019
      // Invalidate L1 cache
      // Based on Nvidia Doc, at MEM barrier, we have to
      //(1) wait for all pending writes till they are acked
      //(2) invalidate L1 cache to ensure coherence and avoid reading stall data
      data_cache_invalidate();
    }
  }
  return !clear_membar;
}


bool SM::warp_waiting_grid_barrier(unsigned warp_id) {
  return m_physical_warp[warp_id]->get_gridbar();
}

void SM::clear_gridbar(unsigned int kernel_id) {
  for (unsigned int i = 0; i < m_config->max_warps_per_shader; i++) {
    if (m_physical_warp[i]->m_kernel_id == kernel_id) {
      m_physical_warp[i]->clear_gridbar();
    }
  }
}

void SM::decrement_atomic_count(unsigned wid, unsigned n) {
  assert(m_physical_warp[wid]->get_n_atomic() >= n);
  m_physical_warp[wid]->dec_n_atomic(n);
}

unsigned int SM::get_atomic_count(unsigned wid) {
  return m_physical_warp[wid]->get_n_atomic();
}

void SM::broadcast_barrier_reduction(unsigned cta_id, unsigned bar_id,
                                     warp_set_t warps) {
  for (unsigned i = 0; i < m_config->max_warps_per_shader; i++) {
    if (warps.test(i)) {
      const warp_inst_t *inst =
          m_physical_warp[i]->restore_info_of_last_inst_at_barrier();
      const_cast<warp_inst_t *>(inst)->broadcast_barrier_reduction(
          inst->get_active_mask());
    }
  }
}

// Block Occupancy Functions

void SM::set_max_cta(const kernel_info_t &kernel) {
  // calculate the max cta count and cta size for local memory address mapping
  kernel_max_cta_per_shader = m_config->max_cta(kernel);
  unsigned int gpu_cta_size = kernel.threads_per_cta();
  kernel_padded_threads_per_cta =
      (gpu_cta_size % m_config->warp_size)
          ? m_config->warp_size * ((gpu_cta_size / m_config->warp_size) + 1)
          : gpu_cta_size;
}

unsigned int SM::get_n_active_cta() const { return m_n_active_cta; }
unsigned int SM::get_not_completed() const { return m_not_completed; }

unsigned int SM::isactive() const {
  if (m_n_active_cta > 0)
    return 1;
  else
    return 0;
}

float SM::get_current_occupancy(unsigned long long &active,
                                unsigned long long &total) const {
  // To match the achieved_occupancy in nvprof, only SMs that are active are
  // counted toward the occupancy.
  if (m_active_warps > 0) {
    total += m_physical_warp.size();
    active += m_active_warps;
    return float(active) / float(total);
  } else {
    return 0;
  }
}

void SM::issue_block2core(kernel_info_t &kernel) {
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
  std::unique_ptr<checkpoint> g_checkpoint = std::make_unique<checkpoint>();
  for (unsigned i = start_thread; i < end_thread; i++) {
    m_threadState[i].m_cta_id = free_cta_hw_id;
    unsigned warp_id = i / m_config->warp_size;
    nthreads_in_block += sim_init_thread(
        kernel, &m_thread[i], m_sm_id, i, cta_size - (i - start_thread),
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

  shader_CTA_count_log(m_sm_id, 1);
  SHADER_DPRINTF(LIVENESS,
                 "GPGPU-Sim uArch: cta:%2u, start_tid:%4u, end_tid:%4u, "
                 "initialized @(%lld,%lld), kernel_uid:%u, kernel_name:%s\n",
                 free_cta_hw_id, start_thread, end_thread, m_gpu->gpu_sim_cycle,
                 m_gpu->gpu_tot_sim_cycle, kernel.get_uid(),
                 kernel.get_name().c_str());
}

bool SM::can_issue_1block(kernel_info_t &kernel) {
  // Jin: concurrent kernels on one SM
  if (m_config->gpgpu_concurrent_kernel_sm) {
    if (m_config->max_cta(kernel) < 1) return false;

    return occupy_shader_resource_1block(kernel, false);
  } else {
    return (get_n_active_cta() < m_config->max_cta(kernel));
  }
}

int SM::find_available_hwtid(unsigned int cta_size, bool occupy) {
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

bool SM::occupy_shader_resource_1block(kernel_info_t &k, bool occupy) {
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
                   m_occupied_ctas, m_sm_id);
  }

  return true;
}

void SM::release_shader_resource_1block(unsigned hw_ctaid, kernel_info_t &k) {
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

void SM::warp_inst_complete(const warp_inst_t &inst) {
#if 0
      printf("[warp_inst_complete] uid=%u core=%u warp=%u pc=%#x @ time=%llu \n",
             inst.get_uid(), m_sm_id, inst.warp_id(), inst.pc,  m_gpu->gpu_tot_sim_cycle +  m_gpu->gpu_sim_cycle);
#endif

  if (inst.op_pipe == SP__OP)
    m_stats->m_num_sp_committed[m_sm_id]++;
  else if (inst.op_pipe == SFU__OP)
    m_stats->m_num_sfu_committed[m_sm_id]++;
  else if (inst.op_pipe == MEM__OP)
    m_stats->m_num_mem_committed[m_sm_id]++;

  if (m_config->gpgpu_clock_gated_lanes == false)
    m_stats->m_num_sim_insn[m_sm_id] += m_config->warp_size;
  else
    m_stats->m_num_sim_insn[m_sm_id] += inst.active_count();

  m_stats->m_num_sim_winsn[m_sm_id]++;

  m_stats
      ->m_num_sim_winsn_per_shader_per_kernel[m_stats->m_current_kernel_pos]
                                             [m_sm_id]++;  // MOD. Custom Stats
  m_stats->m_num_sim_winsn_per_shader[m_sm_id]++;          // MOD. Custom Stats

  m_sm_stats.m_stats_map["gpu_sim_insn"]->increment_with_integer(inst.active_count());
  m_sm_stats.m_stats_map["total_num_warp_instructions"]->increment_with_integer(1);

  if(inst.is_load() || inst.is_store()) {
    m_sm_stats.m_stats_map["total_num_ldst_unit_instructions"]->increment_with_integer(1);
  }else if(inst.is_dp_op()) {
    m_sm_stats.m_stats_map["total_num_dp_instructions"]->increment_with_integer(1);
  }

  inst.completed(m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle);

  if (inst.active_count() < 32 && inst.active_count() != 0) {
    m_sm_stats.m_stats_map["Total_effective_incomplete_warps"]->increment_with_integer(1);
  }
}

bool SM::ptx_thread_done(unsigned hw_thread_id) const {
  return this->core_t::ptx_thread_done(hw_thread_id);
}

void SM::dec_inst_in_pipeline(unsigned warp_id) {
  m_physical_warp[warp_id]->dec_inst_in_pipeline();
}

void SM::store_ack(class mem_fetch *mf) {
  assert(mf->get_type() == WRITE_ACK ||
         (m_config->gpgpu_perfect_mem && mf->get_is_write()));
  unsigned warp_id = mf->get_wid();
  m_physical_warp[warp_id]->dec_store_req();
}

void SM::inc_store_req(unsigned warp_id) {
  m_physical_warp[warp_id]->inc_store_req();
}

void SM::display_simt_state(FILE *fout, int mask) const {
  if ((mask & 4) && m_config->model == POST_DOMINATOR) {
    fprintf(fout, "per warp SIMT control-flow state:\n");
    unsigned n = m_config->n_thread_per_shader / m_config->warp_size;
    for (unsigned i = 0; i < n; i++) {
      unsigned nactive = 0;
      for (unsigned j = 0; j < m_config->warp_size; j++) {
        unsigned tid = i * m_config->warp_size + j;
        int done = ptx_thread_done(tid);
        nactive += (ptx_thread_done(tid) ? 0 : 1);
        if (done && (mask & 8)) {
          unsigned done_cycle = m_thread[tid]->donecycle();
          if (done_cycle) {
            printf("\n w%02u:t%03u: done @ cycle %u", i, tid, done_cycle);
          }
        }
      }
      if (nactive == 0) {
        continue;
      }
      m_simt_stack[i]->print(fout);
    }
    fprintf(fout, "\n");
  }
}

void SM::display_pipeline(FILE *fout, int print_mem, int mask3bit) const {
  display_SM(fout, print_mem, mask3bit);
}

void SM::display_SM(FILE *fout, int print_mem, int mask3bit) const {
  throw std::logic_error("Function not yet implemented");
}

void SM::dump_warp_state(FILE *fout) const {
  fprintf(fout, "\n");
  fprintf(fout, "per warp functional simulation status:\n");
  for (unsigned w = 0; w < m_config->max_warps_per_shader; w++)
    m_physical_warp[w]->print(fout);
}

// Stats Functions

void SM::print_cache_stats(FILE *fp, unsigned &dl1_accesses,
                           unsigned &dl1_misses) {
  m_ldst_unit_shared_of_sm->print_cache_stats(fp, dl1_accesses, dl1_misses);
}

void SM::get_cache_stats(cache_stats &cs) {
  // Adds stats from each cache to 'cs'
  cs += m_L1I_L1_half_C_cache->get_stats();                       // Get L1I stats
  m_ldst_unit_shared_of_sm->get_cache_stats(cs);  // Get L1D, L1C, L1T stats
}

void SM::get_L0I_sub_stats(struct cache_sub_stats &css) const {
  struct cache_sub_stats temp_css;
  struct cache_sub_stats total_css;
  temp_css.clear();
  total_css.clear();
  for (auto subcore : m_subcores) {
    subcore->get_L0I_sub_stats(temp_css);
    total_css += temp_css;
  }
  css = total_css;
}

void SM::get_L1I_sub_stats(struct cache_sub_stats &css) const {
  if (m_L1I_L1_half_C_cache) m_L1I_L1_half_C_cache->get_sub_stats(css);
}

void SM::get_L1D_sub_stats(struct cache_sub_stats &css) const {
  m_ldst_unit_shared_of_sm->get_L1D_sub_stats(css);
}
void SM::get_L1C_sub_stats(struct cache_sub_stats &css) const {
  m_ldst_unit_shared_of_sm->get_L1C_sub_stats(css);
}
void SM::get_L1T_sub_stats(struct cache_sub_stats &css) const {
  m_ldst_unit_shared_of_sm->get_L1T_sub_stats(css);
}

void SM::mem_instruction_stats(const warp_inst_t &inst) {
  unsigned active_count = inst.active_count();
  // this breaks some encapsulation: the is_[space] functions, if you change
  // those, change this.
  switch (inst.space.get_type()) {
    case undefined_space:
    case reg_space:
      break;
    case shared_space:
      m_sm_stats.m_stats_map["gpgpu_n_shmem_insn"]->increment_with_integer(
            active_count);
      break;
    case sstarr_space:
      m_sm_stats.m_stats_map["gpgpu_n_sstarr_insn"]->increment_with_integer(
            active_count);
      break;
    case const_space:
      m_sm_stats.m_stats_map["gpgpu_n_const_mem_insn"]->increment_with_integer(
            active_count);
      break;
    case param_space_kernel:
    case param_space_local:
      m_sm_stats.m_stats_map["gpgpu_n_param_mem_insn"]->increment_with_integer(
            active_count);
      break;
    case tex_space:
      m_sm_stats.m_stats_map["gpgpu_n_tex_insn"]->increment_with_integer(
            active_count);
      break;
    case global_space:
    case local_space:
      if (inst.is_store())
        m_sm_stats.m_stats_map["gpgpu_n_store_insn"]->increment_with_integer(
            active_count);
      else
        m_sm_stats.m_stats_map["gpgpu_n_load_insn"]->increment_with_integer(
            active_count);
      break;
    default:
      abort();
  }
}

unsigned SM::inactive_lanes_accesses_sfu(unsigned active_count,
                                         double latency) {
  return (((32 - active_count) >> 1) * latency) +
         (((32 - active_count) >> 3) * latency) +
         (((32 - active_count) >> 3) * latency);
}
unsigned SM::inactive_lanes_accesses_nonsfu(unsigned active_count,
                                            double latency) {
  return (((32 - active_count) >> 1) * latency);
}
void SM::incload_stat() { m_stats->m_num_loadqueued_insn[m_sm_id]++; }
void SM::incstore_stat() { m_stats->m_num_storequeued_insn[m_sm_id]++; }
void SM::incialu_stat(unsigned active_count, double latency) {
  if (m_config->gpgpu_clock_gated_lanes == false) {
    m_stats->m_num_ialu_acesses[m_sm_id] =
        m_stats->m_num_ialu_acesses[m_sm_id] + (double)active_count * latency +
        inactive_lanes_accesses_nonsfu(active_count, latency);
  } else {
    m_stats->m_num_ialu_acesses[m_sm_id] =
        m_stats->m_num_ialu_acesses[m_sm_id] + (double)active_count * latency;
  }
  m_stats->m_active_exu_threads[m_sm_id] += active_count;
  m_stats->m_active_exu_warps[m_sm_id]++;
}
void SM::incimul_stat(unsigned active_count, double latency) {
  if (m_config->gpgpu_clock_gated_lanes == false) {
    m_stats->m_num_imul_acesses[m_sm_id] =
        m_stats->m_num_imul_acesses[m_sm_id] + (double)active_count * latency +
        inactive_lanes_accesses_nonsfu(active_count, latency);
  } else {
    m_stats->m_num_imul_acesses[m_sm_id] =
        m_stats->m_num_imul_acesses[m_sm_id] + (double)active_count * latency;
  }
  m_stats->m_active_exu_threads[m_sm_id] += active_count;
  m_stats->m_active_exu_warps[m_sm_id]++;
}
void SM::incimul24_stat(unsigned active_count, double latency) {
  if (m_config->gpgpu_clock_gated_lanes == false) {
    m_stats->m_num_imul24_acesses[m_sm_id] =
        m_stats->m_num_imul24_acesses[m_sm_id] +
        (double)active_count * latency +
        inactive_lanes_accesses_nonsfu(active_count, latency);
  } else {
    m_stats->m_num_imul24_acesses[m_sm_id] =
        m_stats->m_num_imul24_acesses[m_sm_id] + (double)active_count * latency;
  }
  m_stats->m_active_exu_threads[m_sm_id] += active_count;
  m_stats->m_active_exu_warps[m_sm_id]++;
}
void SM::incimul32_stat(unsigned active_count, double latency) {
  if (m_config->gpgpu_clock_gated_lanes == false) {
    m_stats->m_num_imul32_acesses[m_sm_id] =
        m_stats->m_num_imul32_acesses[m_sm_id] +
        (double)active_count * latency +
        inactive_lanes_accesses_sfu(active_count, latency);
  } else {
    m_stats->m_num_imul32_acesses[m_sm_id] =
        m_stats->m_num_imul32_acesses[m_sm_id] + (double)active_count * latency;
  }
  m_stats->m_active_exu_threads[m_sm_id] += active_count;
  m_stats->m_active_exu_warps[m_sm_id]++;
}
void SM::incidiv_stat(unsigned active_count, double latency) {
  if (m_config->gpgpu_clock_gated_lanes == false) {
    m_stats->m_num_idiv_acesses[m_sm_id] =
        m_stats->m_num_idiv_acesses[m_sm_id] + (double)active_count * latency +
        inactive_lanes_accesses_sfu(active_count, latency);
  } else {
    m_stats->m_num_idiv_acesses[m_sm_id] =
        m_stats->m_num_idiv_acesses[m_sm_id] + (double)active_count * latency;
  }
  m_stats->m_active_exu_threads[m_sm_id] += active_count;
  m_stats->m_active_exu_warps[m_sm_id]++;
}
void SM::incfpalu_stat(unsigned active_count, double latency) {
  if (m_config->gpgpu_clock_gated_lanes == false) {
    m_stats->m_num_fp_acesses[m_sm_id] =
        m_stats->m_num_fp_acesses[m_sm_id] + (double)active_count * latency +
        inactive_lanes_accesses_nonsfu(active_count, latency);
  } else {
    m_stats->m_num_fp_acesses[m_sm_id] =
        m_stats->m_num_fp_acesses[m_sm_id] + (double)active_count * latency;
  }
  m_stats->m_active_exu_threads[m_sm_id] += active_count;
  m_stats->m_active_exu_warps[m_sm_id]++;
}
void SM::incfpmul_stat(unsigned active_count, double latency) {
  // printf("FP MUL stat increament\n");
  if (m_config->gpgpu_clock_gated_lanes == false) {
    m_stats->m_num_fpmul_acesses[m_sm_id] =
        m_stats->m_num_fpmul_acesses[m_sm_id] + (double)active_count * latency +
        inactive_lanes_accesses_nonsfu(active_count, latency);
  } else {
    m_stats->m_num_fpmul_acesses[m_sm_id] =
        m_stats->m_num_fpmul_acesses[m_sm_id] + (double)active_count * latency;
  }
  m_stats->m_active_exu_threads[m_sm_id] += active_count;
  m_stats->m_active_exu_warps[m_sm_id]++;
}
void SM::incfpdiv_stat(unsigned active_count, double latency) {
  if (m_config->gpgpu_clock_gated_lanes == false) {
    m_stats->m_num_fpdiv_acesses[m_sm_id] =
        m_stats->m_num_fpdiv_acesses[m_sm_id] + (double)active_count * latency +
        inactive_lanes_accesses_sfu(active_count, latency);
  } else {
    m_stats->m_num_fpdiv_acesses[m_sm_id] =
        m_stats->m_num_fpdiv_acesses[m_sm_id] + (double)active_count * latency;
  }
  m_stats->m_active_exu_threads[m_sm_id] += active_count;
  m_stats->m_active_exu_warps[m_sm_id]++;
}
void SM::incdpalu_stat(unsigned active_count, double latency) {
  if (m_config->gpgpu_clock_gated_lanes == false) {
    m_stats->m_num_dp_acesses[m_sm_id] =
        m_stats->m_num_dp_acesses[m_sm_id] + (double)active_count * latency +
        inactive_lanes_accesses_nonsfu(active_count, latency);
  } else {
    m_stats->m_num_dp_acesses[m_sm_id] =
        m_stats->m_num_dp_acesses[m_sm_id] + (double)active_count * latency;
  }
  m_stats->m_active_exu_threads[m_sm_id] += active_count;
  m_stats->m_active_exu_warps[m_sm_id]++;
}
void SM::incdpmul_stat(unsigned active_count, double latency) {
  // printf("FP MUL stat increament\n");
  if (m_config->gpgpu_clock_gated_lanes == false) {
    m_stats->m_num_dpmul_acesses[m_sm_id] =
        m_stats->m_num_dpmul_acesses[m_sm_id] + (double)active_count * latency +
        inactive_lanes_accesses_nonsfu(active_count, latency);
  } else {
    m_stats->m_num_dpmul_acesses[m_sm_id] =
        m_stats->m_num_dpmul_acesses[m_sm_id] + (double)active_count * latency;
  }
  m_stats->m_active_exu_threads[m_sm_id] += active_count;
  m_stats->m_active_exu_warps[m_sm_id]++;
}
void SM::incdpdiv_stat(unsigned active_count, double latency) {
  if (m_config->gpgpu_clock_gated_lanes == false) {
    m_stats->m_num_dpdiv_acesses[m_sm_id] =
        m_stats->m_num_dpdiv_acesses[m_sm_id] + (double)active_count * latency +
        inactive_lanes_accesses_sfu(active_count, latency);
  } else {
    m_stats->m_num_dpdiv_acesses[m_sm_id] =
        m_stats->m_num_dpdiv_acesses[m_sm_id] + (double)active_count * latency;
  }
  m_stats->m_active_exu_threads[m_sm_id] += active_count;
  m_stats->m_active_exu_warps[m_sm_id]++;
}

void SM::incsqrt_stat(unsigned active_count, double latency) {
  if (m_config->gpgpu_clock_gated_lanes == false) {
    m_stats->m_num_sqrt_acesses[m_sm_id] =
        m_stats->m_num_sqrt_acesses[m_sm_id] + (double)active_count * latency +
        inactive_lanes_accesses_sfu(active_count, latency);
  } else {
    m_stats->m_num_sqrt_acesses[m_sm_id] =
        m_stats->m_num_sqrt_acesses[m_sm_id] + (double)active_count * latency;
  }
  m_stats->m_active_exu_threads[m_sm_id] += active_count;
  m_stats->m_active_exu_warps[m_sm_id]++;
}

void SM::inclog_stat(unsigned active_count, double latency) {
  if (m_config->gpgpu_clock_gated_lanes == false) {
    m_stats->m_num_log_acesses[m_sm_id] =
        m_stats->m_num_log_acesses[m_sm_id] + (double)active_count * latency +
        inactive_lanes_accesses_sfu(active_count, latency);
  } else {
    m_stats->m_num_log_acesses[m_sm_id] =
        m_stats->m_num_log_acesses[m_sm_id] + (double)active_count * latency;
  }
  m_stats->m_active_exu_threads[m_sm_id] += active_count;
  m_stats->m_active_exu_warps[m_sm_id]++;
}

void SM::incexp_stat(unsigned active_count, double latency) {
  if (m_config->gpgpu_clock_gated_lanes == false) {
    m_stats->m_num_exp_acesses[m_sm_id] =
        m_stats->m_num_exp_acesses[m_sm_id] + (double)active_count * latency +
        inactive_lanes_accesses_sfu(active_count, latency);
  } else {
    m_stats->m_num_exp_acesses[m_sm_id] =
        m_stats->m_num_exp_acesses[m_sm_id] + (double)active_count * latency;
  }
  m_stats->m_active_exu_threads[m_sm_id] += active_count;
  m_stats->m_active_exu_warps[m_sm_id]++;
}

void SM::incsin_stat(unsigned active_count, double latency) {
  if (m_config->gpgpu_clock_gated_lanes == false) {
    m_stats->m_num_sin_acesses[m_sm_id] =
        m_stats->m_num_sin_acesses[m_sm_id] + (double)active_count * latency +
        inactive_lanes_accesses_sfu(active_count, latency);
  } else {
    m_stats->m_num_sin_acesses[m_sm_id] =
        m_stats->m_num_sin_acesses[m_sm_id] + (double)active_count * latency;
  }
  m_stats->m_active_exu_threads[m_sm_id] += active_count;
  m_stats->m_active_exu_warps[m_sm_id]++;
}

void SM::inctensor_stat(unsigned active_count, double latency) {
  if (m_config->gpgpu_clock_gated_lanes == false) {
    m_stats->m_num_tensor_core_acesses[m_sm_id] =
        m_stats->m_num_tensor_core_acesses[m_sm_id] +
        (double)active_count * latency +
        inactive_lanes_accesses_sfu(active_count, latency);
  } else {
    m_stats->m_num_tensor_core_acesses[m_sm_id] =
        m_stats->m_num_tensor_core_acesses[m_sm_id] +
        (double)active_count * latency;
  }
  m_stats->m_active_exu_threads[m_sm_id] += active_count;
  m_stats->m_active_exu_warps[m_sm_id]++;
}

void SM::inctex_stat(unsigned active_count, double latency) {
  if (m_config->gpgpu_clock_gated_lanes == false) {
    m_stats->m_num_tex_acesses[m_sm_id] =
        m_stats->m_num_tex_acesses[m_sm_id] + (double)active_count * latency +
        inactive_lanes_accesses_sfu(active_count, latency);
  } else {
    m_stats->m_num_tex_acesses[m_sm_id] =
        m_stats->m_num_tex_acesses[m_sm_id] + (double)active_count * latency;
  }
  m_stats->m_active_exu_threads[m_sm_id] += active_count;
  m_stats->m_active_exu_warps[m_sm_id]++;
}

void SM::inc_const_accesses(unsigned active_count) {
  m_stats->m_num_const_acesses[m_sm_id] =
      m_stats->m_num_const_acesses[m_sm_id] + active_count;
}

void SM::incsfu_stat(unsigned active_count, double latency) {
  m_stats->m_num_sfu_acesses[m_sm_id] =
      m_stats->m_num_sfu_acesses[m_sm_id] + (double)active_count * latency;
}
void SM::incsp_stat(unsigned active_count, double latency) {
  m_stats->m_num_sp_acesses[m_sm_id] =
      m_stats->m_num_sp_acesses[m_sm_id] + (double)active_count * latency;
}
void SM::incmem_stat(unsigned active_count, double latency) {
  if (m_config->gpgpu_clock_gated_lanes == false) {
    m_stats->m_num_mem_acesses[m_sm_id] =
        m_stats->m_num_mem_acesses[m_sm_id] + (double)active_count * latency +
        inactive_lanes_accesses_nonsfu(active_count, latency);
  } else {
    m_stats->m_num_mem_acesses[m_sm_id] =
        m_stats->m_num_mem_acesses[m_sm_id] + (double)active_count * latency;
  }
}

void SM::incregfile_reads(unsigned active_count) {
  m_stats->m_read_regfile_acesses[m_sm_id] =
      m_stats->m_read_regfile_acesses[m_sm_id] + active_count;
}
void SM::incregfile_writes(unsigned active_count) {
  m_stats->m_write_regfile_acesses[m_sm_id] =
      m_stats->m_write_regfile_acesses[m_sm_id] + active_count;
}
void SM::incnon_rf_operands(unsigned active_count) {
  m_stats->m_non_rf_operands[m_sm_id] =
      m_stats->m_non_rf_operands[m_sm_id] + active_count;
}

void SM::incspactivelanes_stat(unsigned active_count) {
  m_stats->m_active_sp_lanes[m_sm_id] =
      m_stats->m_active_sp_lanes[m_sm_id] + active_count;
}
void SM::incsfuactivelanes_stat(unsigned active_count) {
  m_stats->m_active_sfu_lanes[m_sm_id] =
      m_stats->m_active_sfu_lanes[m_sm_id] + active_count;
}
void SM::incfuactivelanes_stat(unsigned active_count) {
  m_stats->m_active_fu_lanes[m_sm_id] =
      m_stats->m_active_fu_lanes[m_sm_id] + active_count;
}
void SM::incfumemactivelanes_stat(unsigned active_count) {
  m_stats->m_active_fu_mem_lanes[m_sm_id] =
      m_stats->m_active_fu_mem_lanes[m_sm_id] + active_count;
}

void SM::inc_simt_to_mem(unsigned n_flits) {
  m_stats->n_simt_to_mem[m_sm_id] += n_flits;
}

void SM::incexecstat(warp_inst_t *&inst) {
  // Latency numbers for next operations are used to scale the power values
  // for special operations, according observations from microbenchmarking
  // TODO: put these numbers in the xml configuration
  if (get_gpu()->get_config().g_power_simulation_enabled) {
    switch (inst->sp_op) {
      case INT__OP:
        incialu_stat(inst->active_count(), m_scaling_coeffs->int_coeff);
        break;
      case INT_MUL_OP:
        incimul_stat(inst->active_count(), m_scaling_coeffs->int_mul_coeff);
        break;
      case INT_MUL24_OP:
        incimul24_stat(inst->active_count(), m_scaling_coeffs->int_mul24_coeff);
        break;
      case INT_MUL32_OP:
        incimul32_stat(inst->active_count(), m_scaling_coeffs->int_mul32_coeff);
        break;
      case INT_DIV_OP:
        incidiv_stat(inst->active_count(), m_scaling_coeffs->int_div_coeff);
        break;
      case FP__OP:
        incfpalu_stat(inst->active_count(), m_scaling_coeffs->fp_coeff);
        break;
      case FP_MUL_OP:
        incfpmul_stat(inst->active_count(), m_scaling_coeffs->fp_mul_coeff);
        break;
      case FP_DIV_OP:
        incfpdiv_stat(inst->active_count(), m_scaling_coeffs->fp_div_coeff);
        break;
      case DP___OP:
        incdpalu_stat(inst->active_count(), m_scaling_coeffs->dp_coeff);
        break;
      case DP_MUL_OP:
        incdpmul_stat(inst->active_count(), m_scaling_coeffs->dp_mul_coeff);
        break;
      case DP_DIV_OP:
        incdpdiv_stat(inst->active_count(), m_scaling_coeffs->dp_div_coeff);
        break;
      case FP_SQRT_OP:
        incsqrt_stat(inst->active_count(), m_scaling_coeffs->sqrt_coeff);
        break;
      case FP_LG_OP:
        inclog_stat(inst->active_count(), m_scaling_coeffs->log_coeff);
        break;
      case FP_SIN_OP:
        incsin_stat(inst->active_count(), m_scaling_coeffs->sin_coeff);
        break;
      case FP_EXP_OP:
        incexp_stat(inst->active_count(), m_scaling_coeffs->exp_coeff);
        break;
      case TENSOR__OP:
        inctensor_stat(inst->active_count(), m_scaling_coeffs->tensor_coeff);
        break;
      case TEX__OP:
        inctex_stat(inst->active_count(), m_scaling_coeffs->tex_coeff);
        break;
      default:
        break;
    }
    if (inst->const_cache_operand)  // warp has const address space load as one
                                    // operand
      inc_const_accesses(1);
  }
}

void SM::get_icnt_power_stats(long &n_simt_to_mem, long &n_mem_to_simt) const {
  n_simt_to_mem += m_stats->n_simt_to_mem[m_sm_id];
  n_mem_to_simt += m_stats->n_mem_to_simt[m_sm_id];
}

address_type SM::from_local_pc_to_global_pc_address(address_type local_pc, unsigned int unique_function_id) {
  address_type first_pc_of_kernel = get_gpu()->get_extra_trace_info().get_kernel_by_unique_function_id(unique_function_id).get_function_addr();
  address_type res = local_pc + first_pc_of_kernel;
  return res;
}

address_type SM::from_global_pc_address_to_local_pc(address_type global_pc, unsigned int unique_function_id) {
  address_type first_pc_of_kernel = get_gpu()->get_extra_trace_info().get_kernel_by_unique_function_id(unique_function_id).get_function_addr();
  address_type res = global_pc - first_pc_of_kernel;
  return res;
}