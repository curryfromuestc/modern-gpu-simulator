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

#include <bitset>
#include <queue>
#include <string>
#include <memory>
#include <vector>

#include "../../abstract_hardware_model.h"
#include "../../../../../util/traces_enhanced/src/traced_instruction.h"


class shader_core_config;
class warp_inst_t;
class Subcore;
class SM;
class Wait_Barrier_Entry_Modifier;
class Register_file;
class register_set_uniptr;

class functional_unit {
 public:
  functional_unit(register_set_uniptr *result_port, Register_file* regular_rf, const shader_core_config *config,
                  unsigned int max_latency, std::string name, SM *sm,
                  operation_pipeline_t type_of_pipeline,
                  bool can_set_wait_barriers, bool has_queue,
                  unsigned int max_queue_size, unsigned int num_intermediate_cycles_until_fu_execution,
                  register_set_uniptr *fixed_latency_rf_write_queue, unsigned int max_size_rf_write_queue, bool has_to_go_to_shared_sm_structure, TraceEnhancedOperandType result_queue_type);
  
  virtual ~functional_unit();
  
  const char *get_name();
  bool get_has_queue();

  bool is_fixed_latency_unit();
  bool get_has_to_go_to_shared_sm_structure();
  TraceEnhancedOperandType get_result_queue_type();
  bool is_latency_available(unsigned int target_latency);
  void reserve_latency(unsigned int target_latency);
  void reserve_unit(register_set_uniptr &source_reg);
  void add_extra_cycle_initiation_interval();
  virtual bool can_issue(const warp_inst_t *inst) const;
  virtual void issue(register_set_uniptr &source_reg);
  virtual void cycle();

  unsigned int get_active_lanes_in_pipeline();
  void active_lanes_in_pipeline();

  virtual void print(FILE *fp) const;

  unsigned int get_rf_read_width_per_operand() const;
  unsigned int get_rf_num_read_cycles() const;

  void release_read_barrier(std::unique_ptr<warp_inst_t> &pipe_reg_target);

 protected:
  unsigned int m_rf_read_width_per_operand;
  unsigned int m_rf_num_read_cycles;
  Register_file* m_regular_rf;
  unsigned int m_pipeline_depth;
  unsigned int m_pipeline_extra_predicate_stages_depth;
  std::vector<std::unique_ptr<warp_inst_t>> m_pipeline_reg;
  std::vector<std::unique_ptr<warp_inst_t>> m_pipeline_extra_predicate_stages_reg;
  register_set_uniptr *m_result_port;

  unsigned int m_active_insts_in_pipeline;
  std::string m_name;
  const shader_core_config *m_config;
  std::unique_ptr<warp_inst_t> m_dispatch_reg;
  static const unsigned MAX_ALU_LATENCY = 512;
  std::bitset<MAX_ALU_LATENCY> occupied;
  bool m_has_queue;
  bool m_is_sfu;
  bool m_has_to_go_to_shared_sm_structure;
  bool m_can_set_wait_barriers;
  unsigned int m_max_queue_size;
  unsigned int m_current_queue_size;
  unsigned int m_dispatch_pending_reserved_cycles;
  unsigned int m_num_intermediate_cycles_until_fu_execution;
  register_set_uniptr *m_rf_write_queue;
  unsigned int m_max_size_rf_write_queue;
  operation_pipeline_t m_type_of_pipeline;
  SM *m_sm;
  TraceEnhancedOperandType m_result_queue_type;
  void modify_wait_barrier_states(Wait_Barrier_Entry_Modifier ** wait_barrier_entries);
  bool instruction_finishing_execution(std::unique_ptr<warp_inst_t> &pipe_reg_target);
};

struct functional_unit_with_queue_stage {

  functional_unit_with_queue_stage() {
    inst = std::make_unique<warp_inst_t>();
    remaining_cycles = 0;
    valid = false;
    completed = false;
  }

  ~functional_unit_with_queue_stage() {}

  // Add move constructor
  functional_unit_with_queue_stage(functional_unit_with_queue_stage&& other) noexcept 
      : inst(std::move(other.inst)),
        remaining_cycles(other.remaining_cycles),
        valid(other.valid),
        completed(other.completed) {}

  // Add move assignment
  functional_unit_with_queue_stage& operator=(functional_unit_with_queue_stage&& other) noexcept {
      inst = std::move(other.inst);
      remaining_cycles = other.remaining_cycles;
      valid = other.valid;
      completed = other.completed;
      return *this;
  }

  // Delete copy operations
  functional_unit_with_queue_stage(const functional_unit_with_queue_stage&) = delete;
  functional_unit_with_queue_stage& operator=(const functional_unit_with_queue_stage&) = delete;


  void reset() {
    remaining_cycles = 0;
    valid = false;
    completed = false;
  }

  std::unique_ptr<warp_inst_t> inst;
  unsigned int remaining_cycles;
  bool valid;
  bool completed;
};

class functional_unit_with_queue : public functional_unit {
  public:
    functional_unit_with_queue(register_set_uniptr *result_port, Register_file* regular_rf, const shader_core_config *config,
                  unsigned int max_latency, std::string name, SM *sm,
                  operation_pipeline_t type_of_pipeline,
                  bool can_set_wait_barriers, bool has_queue,
                  unsigned int max_queue_size, unsigned int num_intermediate_cycles_until_fu_execution,
                  register_set_uniptr *fixed_latency_rf_write_queue, unsigned int max_size_rf_write_queue,
                  bool has_to_go_to_shared_sm_structure, unsigned int num_intermediate_stages,
                  unsigned int num_cycles_to_wait_to_dispatch_another_inst_from_this_unit_to_sm_shared_pipeline, TraceEnhancedOperandType result_queue_type);

    ~functional_unit_with_queue();

    virtual void cycle() override;

    virtual bool can_issue(const warp_inst_t *inst) const override;

    unsigned int calculate_num_cycles_need_for_transfer_to_sm(warp_inst_t *inst);

  protected:
    int m_num_intermediate_stages;
    std::vector<functional_unit_with_queue_stage>  m_intermediate_stages;
    std::queue<std::unique_ptr<warp_inst_t>> m_queue;
    // std::vector<> m_pending_release_d
    unsigned int m_num_cycles_to_wait_to_dispatch_another_inst_from_this_unit_to_sm_shared_pipeline;
};


class functional_unit_sfu : public functional_unit {
  public:
    functional_unit_sfu(register_set_uniptr *result_port, Register_file* regular_rf, const shader_core_config *config,
                  unsigned int max_latency, std::string name, SM *sm,
                  operation_pipeline_t type_of_pipeline,
                  bool can_set_wait_barriers, bool has_queue,
                  unsigned int max_queue_size, unsigned int num_intermediate_cycles_until_fu_execution,
                  register_set_uniptr *fixed_latency_rf_write_queue, unsigned int max_size_rf_write_queue,
                  bool has_to_go_to_shared_sm_structure, TraceEnhancedOperandType result_queue_type);

    ~functional_unit_sfu() override = default;

    virtual bool can_issue(const warp_inst_t *inst) const override;
};

class functional_unit_shared_sm_part : public functional_unit {
  public:
    functional_unit_shared_sm_part(
    std::vector<register_set_uniptr*> result_ports,
    const shader_core_config *config,
    unsigned int max_latency, std::string name, SM *sm,
    operation_pipeline_t type_of_pipeline,
    bool can_set_wait_barriers, bool has_queue,
    unsigned int max_queue_size,
    std::vector<register_set_uniptr*> reception_ports,
    unsigned int num_intermediate_cycles_until_fu_execution,
    register_set_uniptr *fixed_latency_rf_write_queue,
    unsigned int max_size_rf_write_queue,
    bool has_to_go_to_shared_sm_structure,
    TraceEnhancedOperandType result_queue_type);

    virtual bool can_issue(const warp_inst_t *inst) const override;
    virtual void issue(register_set_uniptr &source_reg) override;
    virtual void cycle() override;

  protected:
    std::vector<register_set_uniptr*> m_result_ports;
    std::vector<register_set_uniptr*> m_reception_ports;
};