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


#include "functional_unit.h"

#include "../gpu-sim.h"
#include "../shader.h"
#include "sm.h"
#include "warp_dependency_state.h"
#include "register_file.h"


int find_next_stage_index(const std::vector<unsigned int> &cycles_per_stage, int current_idx, int max_stage_idx) {
  int candidate_idx = current_idx + 1;
  while((cycles_per_stage[candidate_idx] == 0) && (candidate_idx < max_stage_idx)) {
    candidate_idx++;
  }
  return candidate_idx;
}

functional_unit::functional_unit(register_set_uniptr *result_port, Register_file* regular_rf,
                                 const shader_core_config *config,
                                 unsigned max_latency, std::string name, SM *sm,
                                 operation_pipeline_t type_of_pipeline,
                                 bool can_set_wait_barriers, bool has_queue,
                                 unsigned int max_queue_size, unsigned int num_intermediate_cycles_until_fu_execution,
                                 register_set_uniptr *fixed_latency_rf_write_queue, unsigned int max_size_rf_write_queue,
                                 bool has_to_go_to_shared_sm_structure, TraceEnhancedOperandType result_queue_type) {
  m_regular_rf = regular_rf;
  m_result_port = result_port;
  m_config = config;
  m_pipeline_depth = max_latency;
  m_pipeline_extra_predicate_stages_depth = m_config->predicate_latency;
  m_pipeline_reg.resize(m_pipeline_depth);
  for (unsigned i = 0; i < m_pipeline_depth; i++) {
    m_pipeline_reg[i] = std::make_unique<warp_inst_t>(config);
  }
  m_pipeline_extra_predicate_stages_reg.resize(m_pipeline_extra_predicate_stages_depth);
  for(unsigned int i = 0; i < m_pipeline_extra_predicate_stages_depth; i++) {
    m_pipeline_extra_predicate_stages_reg[i] = std::make_unique<warp_inst_t>(config);
  }
  m_active_insts_in_pipeline = 0;
  m_name = name;
  m_sm = sm;
  m_type_of_pipeline = type_of_pipeline;
  m_can_set_wait_barriers = can_set_wait_barriers;
  m_has_queue = has_queue;
  m_max_queue_size = max_queue_size;
  m_current_queue_size = 0;
  m_dispatch_pending_reserved_cycles = 0;
  m_num_intermediate_cycles_until_fu_execution = num_intermediate_cycles_until_fu_execution;
  assert(m_max_queue_size > 0);
  m_dispatch_reg = std::make_unique<warp_inst_t>(config);


  if(type_of_pipeline == MEM__OP) {
    m_rf_read_width_per_operand = m_config->num_threads_granularity_read_regular_register_file_mem_inst;
  }else if(type_of_pipeline == DP__OP) {
    m_rf_read_width_per_operand = m_config->num_threads_granularity_read_regular_register_file_dp_inst;
  }else if(type_of_pipeline == SFU__OP) {
    m_rf_read_width_per_operand = m_config->num_threads_granularity_read_regular_register_file_sfu_inst;
  }else {
    m_rf_read_width_per_operand = m_config->num_threads_granularity_read_regular_register_file_other_inst;
  }
  m_rf_num_read_cycles = config->warp_size / m_rf_read_width_per_operand;
  m_rf_write_queue = fixed_latency_rf_write_queue;
  m_max_size_rf_write_queue = max_size_rf_write_queue;
  m_has_to_go_to_shared_sm_structure = has_to_go_to_shared_sm_structure;
  m_is_sfu = false;
  m_result_queue_type = result_queue_type;
}

functional_unit::~functional_unit() {}

const char *functional_unit::get_name() { return m_name.c_str(); }

bool functional_unit::get_has_queue() { return m_has_queue; }

TraceEnhancedOperandType functional_unit::get_result_queue_type() {
  return m_result_queue_type;
}

bool functional_unit::is_fixed_latency_unit() { 
  return !m_has_queue && !m_is_sfu;
}

bool functional_unit::get_has_to_go_to_shared_sm_structure() {
  return m_has_to_go_to_shared_sm_structure;
}

bool functional_unit::is_latency_available(unsigned int target_latency) {
  return !occupied.test(target_latency);
}

void functional_unit::reserve_latency(unsigned int target_latency) {
  occupied.set(target_latency);
}

void functional_unit::reserve_unit(register_set_uniptr &source_reg) {
  assert(!m_has_queue);
  warp_inst_t *ready_reg = source_reg.get_ready();
  if (ready_reg) {
    m_dispatch_pending_reserved_cycles = ready_reg->initiation_interval;
  }
}

void functional_unit::add_extra_cycle_initiation_interval() {
  m_dispatch_pending_reserved_cycles++;
}

bool functional_unit::can_issue(const warp_inst_t *inst) const {
  assert(!m_has_queue);
  return m_dispatch_pending_reserved_cycles == 0;
}

unsigned int functional_unit::get_rf_read_width_per_operand() const {
  return m_rf_read_width_per_operand;
}

unsigned int functional_unit::get_rf_num_read_cycles() const {
  return m_rf_num_read_cycles;
}

void functional_unit::issue(register_set_uniptr &source_reg) {
  warp_inst_t *ready_reg = source_reg.get_ready();
  if (ready_reg) {
    ready_reg->op_pipe = m_type_of_pipeline;
    switch (m_type_of_pipeline) {
      case SP__OP:
      case DP__OP:
      case INTP__OP:
      case SPECIALIZED__OP:
        m_sm->incsp_stat(m_sm->get_config()->warp_size, ready_reg->latency);
        break;
      case MEM__OP:
        m_sm->incmem_stat(m_sm->get_config()->warp_size, 1);
        break;
      case SFU__OP:
      case TENSOR_CORE__OP:
        m_sm->incsfu_stat(m_sm->get_config()->warp_size, ready_reg->latency);
        break;
      default:
        assert(0);
        break;
    }

    m_sm->incexecstat(ready_reg);
    source_reg.move_out_to(m_dispatch_reg);
  }
}

void functional_unit::modify_wait_barrier_states(
    Wait_Barrier_Entry_Modifier **wait_barrier_entries) {
  if (wait_barrier_entries[0] != nullptr) {
    m_sm->get_shd_warp(wait_barrier_entries[0]->sm_warp_id)
        ->get_dependency_state()
        ->action_over_wait_barrier(wait_barrier_entries[0]);
    delete wait_barrier_entries[0];
    wait_barrier_entries[0] = nullptr;
  }
  if (wait_barrier_entries[1] != nullptr) {
    wait_barrier_entries[0] = wait_barrier_entries[1];
    wait_barrier_entries[1] = nullptr;
  }
}

void functional_unit::release_read_barrier(std::unique_ptr<warp_inst_t> &pipe_reg_target) {
  if ((!m_sm->get_config()->is_trace_mode ||
       (m_sm->get_config()->is_trace_mode &&
        ( !m_sm->get_shd_warp(pipe_reg_target->warp_id())->get_kernel_info()->is_captured_from_binary ||
         m_sm->get_config()->is_remodeling_scoreboarding_enabled) ) ) &&
      (m_sm->get_scoreboard_WAR()->getMode() == RELEASE_AT_OPC)) {

    if ( ( !m_sm->get_shd_warp(pipe_reg_target->warp_id())->get_kernel_info()->is_captured_from_binary && m_config->is_trace_mode) || (m_config->is_trace_mode && m_config->is_remodeling_scoreboarding_enabled) ) {
      m_sm->get_scoreboard_WAR()->releaseRegisters_remodeling(pipe_reg_target.get());
    }else {
      m_sm->get_scoreboard_WAR()->releaseRegisters(pipe_reg_target.get());
    }
  } else if (!m_sm->get_config()->is_remodeling_scoreboarding_enabled && m_can_set_wait_barriers && pipe_reg_target
                                            ->get_extra_trace_instruction_info()
                                            .get_control_bits()
                                            .get_is_new_read_barrier()) {
    m_sm->add_pending_wait_barrier_decrement(
        pipe_reg_target.get(), Wait_Barrier_Type::READ_WAIT_BARRIER,
        pipe_reg_target
            ->get_extra_trace_instruction_info()
            .get_control_bits()
            .get_id_new_read_barrier());
  }
}

bool functional_unit::instruction_finishing_execution(std::unique_ptr<warp_inst_t> &pipe_reg_target) {
  bool retired = false;
  if(m_has_to_go_to_shared_sm_structure) {
    if(m_result_port->has_free() && m_sm->can_send_inst_from_subcore_to_sm_shared_pipeline()) {
      m_result_port->move_in(pipe_reg_target);
      retired = true;
    }
  }else {
    if (!pipe_reg_target
            ->get_extra_trace_instruction_info()
            .has_destination_registers()) {
      m_sm->instruction_retirement(pipe_reg_target.get());
      retired = true;
    } else {
      if(is_fixed_latency_unit()) {
        assert(m_rf_write_queue->has_free());  
        m_rf_write_queue->move_in(pipe_reg_target);   
        retired = true;
      }else {
        if(m_rf_write_queue->has_free()) {
          m_rf_write_queue->move_in(pipe_reg_target);
          retired = true;
        }
      }
    }
  }

  if(retired) {
    assert(m_active_insts_in_pipeline > 0);
    m_active_insts_in_pipeline--;
    if(m_has_queue) {
      assert(m_current_queue_size > 0);
      m_current_queue_size--;
      pipe_reg_target->clear();
    }
  }
  return retired;
}

void functional_unit::cycle() {
  assert(!m_has_queue);
  if(m_dispatch_pending_reserved_cycles > 0) {
    m_dispatch_pending_reserved_cycles--;
  }
  if(!m_pipeline_extra_predicate_stages_reg[0]->empty()) {
    instruction_finishing_execution(m_pipeline_extra_predicate_stages_reg[0]);
  }
  if(m_active_insts_in_pipeline) {
    for (unsigned int stage = 0; (stage + 1) < m_pipeline_extra_predicate_stages_depth; stage++) {
      if (m_pipeline_extra_predicate_stages_reg[stage]->empty()) {
        move_warp_uniptr(m_pipeline_extra_predicate_stages_reg[stage], m_pipeline_extra_predicate_stages_reg[stage + 1]);
      }
    }
  }
  if (!m_pipeline_reg[0]->empty()) {
    if(m_pipeline_reg[0]->latency_extra_predicate_op > 0) {
      unsigned int target_stage = m_pipeline_reg[0]->latency_extra_predicate_op - 1;
      if(m_pipeline_extra_predicate_stages_reg[target_stage]->empty()) {
        move_warp_uniptr( m_pipeline_extra_predicate_stages_reg[target_stage], m_pipeline_reg[0]);
      }
    }else {
      instruction_finishing_execution(m_pipeline_reg[0]);
    }
  }


  if (m_active_insts_in_pipeline) {
    for (unsigned int stage = 0; (stage + 1) < m_pipeline_depth; stage++) {
      if (m_pipeline_reg[stage]->empty()) {
        move_warp_uniptr(m_pipeline_reg[stage], m_pipeline_reg[stage + 1]);
      }
    }
  }

  if (!m_dispatch_reg->empty()) {
    if (!m_dispatch_reg->dispatch_delay()) {
      int start_stage = m_dispatch_reg->latency -1;
      assert(start_stage >= 0);
      if (m_pipeline_reg[start_stage]->empty()) {
        if(!is_fixed_latency_unit()){
          release_read_barrier(m_dispatch_reg);
        }
        move_warp_uniptr(m_pipeline_reg[start_stage], m_dispatch_reg);
        m_active_insts_in_pipeline++;
      }
    }
  }

  occupied >>= 1;
}

unsigned int functional_unit::get_active_lanes_in_pipeline() {
  active_mask_t active_lanes;
  active_lanes.reset();
  if (m_sm->get_gpu()->get_config().g_power_simulation_enabled) {
    for (unsigned stage = 0; (stage + 1) < m_pipeline_depth; stage++) {
      if (!m_pipeline_reg[stage]->empty())
        active_lanes |= m_pipeline_reg[stage]->get_active_mask();
    }
  }
  return active_lanes.count();
}

void functional_unit::active_lanes_in_pipeline() {
  unsigned active_count = get_active_lanes_in_pipeline();
  assert(active_count <= m_sm->get_config()->warp_size);
  switch (m_type_of_pipeline) {
    case SP__OP:
    case INTP__OP:
    case SPECIALIZED__OP:
      m_sm->incspactivelanes_stat(active_count);
      m_sm->incfuactivelanes_stat(active_count);
      break;
    case DP__OP:
      m_sm->incfuactivelanes_stat(active_count);
      break;
    case TENSOR_CORE__OP:
    case SFU__OP:
      m_sm->incsfuactivelanes_stat(active_count);
      m_sm->incfuactivelanes_stat(active_count);
      break;
    default:
      assert(0);
      break;
  }
  m_sm->incfumemactivelanes_stat(active_count);
}

void functional_unit::print(FILE *fp) const {
  fprintf(fp, "Dispatch reg: ");
  if (!m_dispatch_reg->empty()) {
    m_dispatch_reg->print(fp);
  } else {
    fprintf(fp, "Empty.");
  }
  for (int s = m_pipeline_depth - 1; s >= 0; s--) {
    if (m_pipeline_reg[s]) {
      fprintf(fp, "  %s ", m_name.c_str());
      m_pipeline_reg[s]->print(fp);
    }
  }
}

functional_unit_with_queue::functional_unit_with_queue(
    register_set_uniptr *result_port, Register_file* regular_rf, const shader_core_config *config,
    unsigned int max_latency, std::string name, SM *sm,
    operation_pipeline_t type_of_pipeline, bool can_set_wait_barriers,
    bool has_queue, unsigned int max_queue_size, unsigned int num_intermediate_cycles_until_fu_execution,
    register_set_uniptr *fixed_latency_rf_write_queue, unsigned int max_size_rf_write_queue, 
    bool has_to_go_to_shared_sm_structure, unsigned int num_intermediate_stages,
    unsigned int num_cycles_to_wait_to_dispatch_another_inst_from_this_unit_to_sm_shared_pipeline, TraceEnhancedOperandType result_queue_type)
    : functional_unit(result_port, regular_rf, config, max_latency, name, sm, type_of_pipeline,
                      can_set_wait_barriers, has_queue, max_queue_size, num_intermediate_cycles_until_fu_execution,
                      fixed_latency_rf_write_queue, max_size_rf_write_queue, has_to_go_to_shared_sm_structure, result_queue_type) {
  
  m_num_intermediate_stages = num_intermediate_stages;  
  m_intermediate_stages.resize(m_num_intermediate_stages);
  m_num_cycles_to_wait_to_dispatch_another_inst_from_this_unit_to_sm_shared_pipeline = num_cycles_to_wait_to_dispatch_another_inst_from_this_unit_to_sm_shared_pipeline;
}

functional_unit_with_queue::~functional_unit_with_queue() {
  while(!m_queue.empty()) {
    m_queue.pop();
  }
  m_intermediate_stages.clear();
}

bool functional_unit_with_queue::can_issue(const warp_inst_t *inst) const {
  return m_dispatch_reg->empty();
}

void functional_unit_with_queue::cycle() {
  assert(m_has_queue);
  if(!m_pipeline_extra_predicate_stages_reg[0]->empty()) {
    instruction_finishing_execution(m_pipeline_extra_predicate_stages_reg[0]);
  }
  if(m_active_insts_in_pipeline) {
    for (unsigned int stage = 0; (stage + 1) < m_pipeline_extra_predicate_stages_depth; stage++) {
      if (m_pipeline_extra_predicate_stages_reg[stage]->empty()) {
        move_warp_uniptr(m_pipeline_extra_predicate_stages_reg[stage], m_pipeline_extra_predicate_stages_reg[stage + 1]);
      }
    }
  }

  for(int i = (m_num_intermediate_stages - 1); i >= 0; i--) {
    if(m_intermediate_stages[i].valid) {
      if(m_intermediate_stages[i].inst->m_num_cycles_to_wait_to_free_WAR > 0) {
        m_intermediate_stages[i].inst->m_num_cycles_to_wait_to_free_WAR--;
        if(m_intermediate_stages[i].inst->m_num_cycles_to_wait_to_free_WAR == 0) {
          release_read_barrier(m_intermediate_stages[i].inst);
        }
      }
      if(m_intermediate_stages[i].remaining_cycles > 0) {
        m_intermediate_stages[i].remaining_cycles--;
      }
      m_intermediate_stages[i].completed = (m_intermediate_stages[i].remaining_cycles == 0) && m_intermediate_stages[i].valid;
      if(m_intermediate_stages[i].completed) {
        bool advanced = false;
        if(i == (m_num_intermediate_stages - 1)) {
          assert(m_intermediate_stages[i].inst->m_num_cycles_to_wait_to_free_WAR == 0);
          advanced = instruction_finishing_execution(m_intermediate_stages[i].inst);
          if(advanced && (m_num_cycles_to_wait_to_dispatch_another_inst_from_this_unit_to_sm_shared_pipeline > 0)) {
            m_sm->set_num_cycles_to_wait_to_dispatch_another_inst_from_subcore_to_sm_shared_pipeline(m_sm->get_config()->num_cycles_to_wait_to_dispatch_another_inst_from_subcore_to_sm_shared_pipeline_when_is_mem_inst);
          }
        }else {
          unsigned int target_stage = find_next_stage_index(m_intermediate_stages[i].inst->m_num_cycles_per_intermediate_stage, i, m_num_intermediate_stages - 1);
          unsigned int target_cycle = m_intermediate_stages[i].inst->m_num_cycles_per_intermediate_stage[target_stage];
          if(!m_intermediate_stages[target_stage].valid) {
            m_intermediate_stages[target_stage].valid = true;
            m_intermediate_stages[target_stage].completed = false;
            m_intermediate_stages[target_stage].remaining_cycles = target_cycle; 
            move_warp_uniptr(m_intermediate_stages[target_stage].inst, m_intermediate_stages[i].inst);
            advanced = true;
          }
        }
        if(advanced) {
          m_intermediate_stages[i].reset();
        }
      }
    }
  }

  if(!m_queue.empty()) {
    bool is_rf_ready = true;
    RF_instruction_read_request rf_var_lat_analysis;
    assert(m_queue.front());
    rf_var_lat_analysis = m_regular_rf->is_possible_to_read_non_cacheable(m_queue.front().get(), m_rf_num_read_cycles);
    is_rf_ready = rf_var_lat_analysis.m_is_possible_to_read;
    if(is_rf_ready) {
      unsigned int next_stage = find_next_stage_index(m_queue.front()->m_num_cycles_per_intermediate_stage, -1, m_num_intermediate_stages - 1);
      if(!m_intermediate_stages[next_stage].valid) {
        std::unique_ptr<warp_inst_t> inst_to_pop = std::move(m_queue.front());
        m_sm->m_sm_stats.m_stats_map["total_num_register_file_cache_allocations"]->increment_with_integer(rf_var_lat_analysis.m_rf_cache_allocate_requests.size());
        m_regular_rf->allocate_reads_non_cacheable(rf_var_lat_analysis, inst_to_pop.get(), m_rf_num_read_cycles);
        m_intermediate_stages[next_stage].valid = true;
        m_intermediate_stages[next_stage].completed = false;
        m_intermediate_stages[next_stage].remaining_cycles = inst_to_pop->m_num_cycles_per_intermediate_stage[next_stage];
        move_warp_uniptr(m_intermediate_stages[next_stage].inst, inst_to_pop);
        m_queue.pop();
      }
      
    }
    
  }

  if (!m_dispatch_reg->empty()) {
    if(m_current_queue_size < m_max_queue_size) {
      std::unique_ptr<warp_inst_t> aux_int = std::make_unique<warp_inst_t>(m_config);
      move_warp_uniptr(aux_int, m_dispatch_reg);
      m_active_insts_in_pipeline++;
      m_queue.push(std::move(aux_int));
      m_current_queue_size++;
    }
  }

  occupied >>= 1;
}

functional_unit_sfu::functional_unit_sfu(
    register_set_uniptr *result_port, Register_file* regular_rf, const shader_core_config *config,
    unsigned int max_latency, std::string name, SM *sm,
    operation_pipeline_t type_of_pipeline, bool can_set_wait_barriers,
    bool has_queue, unsigned int max_queue_size, unsigned int num_intermediate_cycles_until_fu_execution,
    register_set_uniptr *fixed_latency_rf_write_queue, unsigned int max_size_rf_write_queue, 
    bool has_to_go_to_shared_sm_structure, TraceEnhancedOperandType result_queue_type)
    : functional_unit(result_port, regular_rf, config, max_latency, name, sm, type_of_pipeline,
                      can_set_wait_barriers, has_queue, max_queue_size, num_intermediate_cycles_until_fu_execution,
                      fixed_latency_rf_write_queue, max_size_rf_write_queue, has_to_go_to_shared_sm_structure, result_queue_type) {
  m_is_sfu = true;
}

bool functional_unit_sfu::can_issue(const warp_inst_t *inst) const {
  return m_dispatch_reg->empty();
}


void functional_unit_shared_sm_part::issue(register_set_uniptr &source_reg) {
  functional_unit::issue(source_reg);
}

functional_unit_shared_sm_part::functional_unit_shared_sm_part(
    std::vector<register_set_uniptr*> result_ports, const shader_core_config *config,
    unsigned int max_latency, std::string name, SM *sm,
    operation_pipeline_t type_of_pipeline, bool can_set_wait_barriers,
    bool has_queue, unsigned int max_queue_size,
    std::vector<register_set_uniptr*> reception_ports, unsigned int num_intermediate_cycles_until_fu_execution,
    register_set_uniptr *fixed_latency_rf_write_queue, unsigned int max_size_rf_write_queue,
    bool has_to_go_to_shared_sm_structure, TraceEnhancedOperandType result_queue_type)
    : functional_unit(nullptr, nullptr, config, max_latency, name, sm, type_of_pipeline,
                     can_set_wait_barriers, has_queue, max_queue_size, num_intermediate_cycles_until_fu_execution,
                     fixed_latency_rf_write_queue, max_size_rf_write_queue, has_to_go_to_shared_sm_structure, result_queue_type) {
  m_result_ports = result_ports;
  m_reception_ports = reception_ports;
}

void functional_unit_shared_sm_part::cycle() {
  if (!m_pipeline_reg[0]->empty()) {
    unsigned int subcore_id = m_pipeline_reg[0]->get_subcore_id();
    if(!m_pipeline_reg[0]
            ->get_extra_trace_instruction_info()
            .has_destination_registers()) {
      m_sm->instruction_retirement(m_pipeline_reg[0].get());
      assert(m_active_insts_in_pipeline > 0);
      m_active_insts_in_pipeline--;
    }
    else if (m_result_ports[subcore_id]->has_free()) {
      assert(m_active_insts_in_pipeline > 0);
      m_active_insts_in_pipeline--;
      m_result_ports[subcore_id]->move_in(m_pipeline_reg[0]);
    }
  }
  if (m_active_insts_in_pipeline) {
    for (unsigned stage = 0; (stage + 1) < m_pipeline_depth; stage++) {
      if (m_pipeline_reg[stage]->empty()) {
        move_warp_uniptr(m_pipeline_reg[stage], m_pipeline_reg[stage + 1]);
      }
    }
  }

  if (!m_dispatch_reg->empty()) {
    if (!m_dispatch_reg->dispatch_delay()) {
      int start_stage = m_dispatch_reg->latency -
                        m_dispatch_reg->initiation_interval;
      if (m_pipeline_reg[start_stage]->empty()) {
        m_current_queue_size--;
        move_warp_uniptr(m_pipeline_reg[start_stage], m_dispatch_reg);
        m_active_insts_in_pipeline++;
      }
    }
  }

  occupied >>= 1;

  for (unsigned int i = 0; i < m_reception_ports.size(); i++) {
    if (m_reception_ports[i]->has_ready() && m_sm->can_send_inst_from_subcore_to_sm_shared_pipeline() &&
        can_issue((m_reception_ports[i]->get_ready()))) {
      issue(*m_reception_ports[i]);
    }
  }
}

bool functional_unit_shared_sm_part::can_issue(const warp_inst_t *inst) const {
  return m_dispatch_reg->empty();
}