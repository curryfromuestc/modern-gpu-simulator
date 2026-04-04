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

#include "register_file.h"

#include "subcore.h"
#include "sm.h"

#include "../../../../../util/traces_enhanced/src/traced_operand.h"
#include "../../../../../util/traces_enhanced/src/traced_instruction.h"
#include "../../abstract_hardware_model.h"
#include "../shader.h"


Register_file_cache_entry::Register_file_cache_entry(shader_core_stats *stats, Register_file_cache* rf_cache) {
    m_stats = stats;
    m_rf_cache = rf_cache;
    flush();
}

Register_file_cache::~Register_file_cache() {}

bool Register_file_cache_entry::is_valid() {
  return m_is_valid;
}

void Register_file_cache_entry::set_valid(bool valid) {
  m_is_valid = valid;
}

unsigned int Register_file_cache_entry::get_warp_id() {
  return m_warp_id;
}

void Register_file_cache_entry::set_warp_id(unsigned int warp_id) {
  m_warp_id = warp_id;
}

unsigned int Register_file_cache_entry::get_reg_id() {
  return m_reg_id;
}

void Register_file_cache_entry::set_reg_id(unsigned int reg_id) {
  m_reg_id = reg_id;
}

void Register_file_cache_entry::allocate_new_register(unsigned int warp_id, unsigned int reg_id) {
  m_is_valid = true;
  m_warp_id = warp_id;
  m_reg_id = reg_id;
}

bool Register_file_cache_entry::is_hit(unsigned int warp_id, unsigned int reg_id) {
  bool is_hit = m_is_valid && (m_warp_id == warp_id) && (m_reg_id == reg_id);
  return is_hit;
}

void Register_file_cache_entry::flush() {
  m_is_valid = false;
  m_warp_id = 0;
  m_reg_id = 0;
}

void Register_file_cache_entry::print(FILE *fp) {
  fprintf(fp, "Register_file_cache_entry: valid=%d, warp_id=%d, reg_id=%d\n", m_is_valid, m_warp_id, m_reg_id);
}


Register_file_cache::Register_file_cache(unsigned int num_banks, unsigned int max_num_operands, shader_core_stats *stats,
                      Register_file* rf) {
  m_num_banks = num_banks;
  m_max_num_operands = max_num_operands;
  m_stats = stats;
  m_rf = rf;
}

Register_file_cache* Register_file_cache::getptr() {
  return this;
}

void Register_file_cache::init() {
  for (unsigned int i = 0; i < m_max_num_operands; i++) {
    m_entries.push_back(std::vector<Register_file_cache_entry>());
    for (unsigned int j = 0; j < m_num_banks; j++) {
      m_entries[i].push_back(
          Register_file_cache_entry(m_stats, getptr()));
    }
  }
}

void Register_file_cache::allocate_new_register(unsigned int operand_pos, unsigned int warp_id, unsigned int reg_id){
  assert(operand_pos < m_max_num_operands);
  m_entries[operand_pos][m_rf->calculate_target_bank(reg_id)].allocate_new_register(warp_id, reg_id);
}

bool Register_file_cache::is_hit(unsigned int operand_pos, unsigned int warp_id, unsigned int reg_id){
  assert(operand_pos < m_max_num_operands);
  return m_entries[operand_pos][m_rf->calculate_target_bank(reg_id)].is_hit(warp_id, reg_id);
}

void Register_file_cache::flush(){
  for(auto &operand : m_entries){
    for(auto &entry : operand){
      entry.flush();
    }
  }
}

void Register_file_cache::flush_entry(unsigned int operand_pos, unsigned int reg_id){
  assert(operand_pos < m_max_num_operands);
  m_entries[operand_pos][m_rf->calculate_target_bank(reg_id)].flush();
}

Register_file* Register_file_cache::get_register_file(){
  return m_rf;
}

void Register_file_cache::print(FILE *fp){
  fprintf(fp, "Register_file_cache:\n");
  int operand_pos = 0;
  for(auto &operand : m_entries){
    fprintf(fp, "Operand %d:\n", operand_pos);
    for(auto &entry : operand){
      entry.flush();
    }
    operand_pos++;
  }
}

Register_file_bank::Register_file_bank(unsigned int num_read_ports,  unsigned int num_write_ports, unsigned int max_supported_latency, Register_file *rf) {
  m_num_read_ports = num_read_ports;
  m_num_write_ports = num_write_ports;
  m_max_supported_latency = max_supported_latency;
  m_rf = rf;
  m_num_used_read_ports.resize(m_max_supported_latency, 0);
  m_num_used_write_ports.resize(m_max_supported_latency, 0);
}

void Register_file_bank::cycle() {
  for (unsigned stage = 0; stage < m_max_supported_latency-1; ++stage) {
    if(stage < (m_max_supported_latency - 1)) {
      m_num_used_read_ports[stage] = m_num_used_read_ports[stage+1];
      m_num_used_write_ports[stage] = m_num_used_write_ports[stage+1];
    }
  }
  m_num_used_read_ports[m_max_supported_latency-1] = 0;
}

unsigned int Register_file_bank::get_num_ports() {
  return m_num_read_ports;
}

unsigned int Register_file_bank::get_max_supported_latency() {
  return m_max_supported_latency;
}

bool Register_file_bank::is_read_available(unsigned int init_latency, unsigned int max_latency, unsigned int num_reads) {
  bool res = true;
  int num_pending_reads_to_assign = num_reads;
  if(!m_rf->is_unlimited_reads_per_cycles()) {
    assert(init_latency < m_max_supported_latency);
    assert(max_latency < m_max_supported_latency);
    for(unsigned int i = init_latency; (i < max_latency) && (num_pending_reads_to_assign > 0); i++){
      int current_available_ports = m_num_read_ports - m_num_used_read_ports[i];
      num_pending_reads_to_assign = std::max(0, num_pending_reads_to_assign - current_available_ports);
    }
    res = num_pending_reads_to_assign == 0;
  }
  return res;
}

void Register_file_bank::allocate_read_ports(unsigned int init_latency, unsigned int max_latency, unsigned int num_reads) {
  assert(init_latency < m_max_supported_latency);
  int num_pending_reads_to_assign = num_reads;
  for(unsigned int i = init_latency; i < max_latency; i++){
    int current_available_ports = m_num_read_ports - m_num_used_read_ports[i];
    int ports_to_use_cycle = std::min(num_pending_reads_to_assign, current_available_ports);
    num_pending_reads_to_assign = std::max(0, num_pending_reads_to_assign - ports_to_use_cycle);
    m_num_used_read_ports[i] += ports_to_use_cycle;
    assert(m_num_used_read_ports[i] <= m_num_read_ports);
  }
  assert(num_pending_reads_to_assign == 0);
}

bool Register_file_bank::is_write_available_this_cycle() {
  bool res = true;
  if(!m_rf->is_unlimited_writes_per_cycles()) {
    res = m_num_used_write_ports[0] < m_num_write_ports;
  }
  return res;
}

bool Register_file_bank::is_write_available_at_given_cycle(unsigned int given_cycle) {
  bool res = true;
  if(!m_rf->is_unlimited_writes_per_cycles()) {
    res = m_num_used_write_ports[given_cycle] < m_num_write_ports;
  }
  return res;
}

void Register_file_bank::allocate_write_port_this_cycle() {
  if(!m_rf->is_unlimited_writes_per_cycles()) {
    m_num_used_write_ports[0]++;
    assert(m_num_used_write_ports[0] <= m_num_write_ports);
  }
}

void Register_file_bank::allocate_write_port_at_given_cycle(unsigned int given_cycle) {
  if(!m_rf->is_unlimited_writes_per_cycles()) {
    m_num_used_write_ports[given_cycle]++;
    assert(m_num_used_write_ports[given_cycle] <= m_num_write_ports);
  }
}

void Register_file_bank::print(FILE *fp) {
  fprintf(fp, "Register_file_bank: num_read_ports_fixed_latency=%d, max_supported_latency=%d\n", m_num_read_ports, m_max_supported_latency);
  for(unsigned int i = 0; i < m_max_supported_latency; i++){
    if(m_num_used_read_ports[i] > 0) {
      fprintf(fp, "Stage %d: %d\n", i, m_num_used_read_ports[i]);
    }
  }
}

Register_file::Register_file(unsigned int num_banks, unsigned int num_read_ports_per_bank,  unsigned int num_write_ports_per_bank, unsigned int max_supported_latency, bool is_unlimited_reads_per_cycles, bool is_unlimited_writes_per_cycles, unsigned int max_num_operands,
                shader_core_stats *stats, Subcore* subcore, TraceEnhancedOperandType type, bool is_rf_cache_enabled) {
  m_num_banks = num_banks;
  m_max_num_operands = max_num_operands;
  m_num_read_ports_per_bank = num_read_ports_per_bank;
  m_num_write_ports_per_bank = num_write_ports_per_bank;
  m_max_supported_latency = max_supported_latency;
  m_stats = stats;
  m_is_unlimited_reads_per_cycles = is_unlimited_reads_per_cycles;
  m_is_unlimited_writes_per_cycles = is_unlimited_writes_per_cycles;
  m_max_num_operands = max_num_operands;
  m_subcore = subcore;
  m_is_rf_cache_enabled = is_rf_cache_enabled;
  m_type = type;
}

Register_file::~Register_file() {
  delete m_rf_cache;
}

Register_file* Register_file::getptr() {
  return this;
}

void Register_file::init() {
  for(unsigned int i = 0; i < m_num_banks; i++){
    m_banks.push_back(Register_file_bank(m_num_read_ports_per_bank, m_num_write_ports_per_bank, m_max_supported_latency, getptr()));
  }
  m_rf_cache = new Register_file_cache(m_num_banks, m_max_num_operands, m_stats, getptr());
  m_rf_cache->init();
}

Subcore *Register_file::get_subcore() {
  return m_subcore;
}

TraceEnhancedOperandType Register_file::get_type() {
  return m_type;
}

unsigned int Register_file::get_num_banks() {
  return m_num_banks;
}

unsigned int Register_file::get_num_read_ports_per_bank() {
  return m_num_read_ports_per_bank;
}

unsigned int Register_file::get_num_write_ports_per_bank() {
  return m_num_write_ports_per_bank;
}

unsigned int Register_file::get_max_supported_latency() {
  return m_max_supported_latency;
}

unsigned int Register_file::get_max_num_operands() {
  return m_max_num_operands;
}

bool Register_file::is_unlimited_reads_per_cycles() {
  return m_is_unlimited_reads_per_cycles;
}

bool Register_file::is_unlimited_writes_per_cycles() {
  return m_is_unlimited_writes_per_cycles;
}

bool Register_file::is_rf_cache_enabled() {
  return m_is_rf_cache_enabled;
}

shader_core_stats *Register_file::get_stats() {
  return m_stats;
}

void Register_file::cycle() {
  for(auto &bank : m_banks){
    bank.cycle();
  }
}

bool Register_file::is_operand_to_be_considered(traced_operand &operand) {
  unsigned reg_id = operand.get_operand_reg_number();
  TraceEnhancedOperandType op_type = get_reg_type_eval(operand);
  return !is_reserved_reg(reg_id, op_type) && (m_type == op_type);
}

int Register_file::compute_read_slack(int max_uses) {
  int denominator = 0;
  if(m_num_banks > m_num_read_ports_per_bank) {
    denominator = m_num_banks / m_num_read_ports_per_bank;
  }else {
    denominator = m_num_read_ports_per_bank / m_num_banks;
  }
  assert(denominator > 0);
  int result = std::max(0, static_cast<int>(ceil(max_uses / (denominator))) - 1);
  return result;
}

RF_instruction_read_request Register_file::is_possible_to_read_cacheable(const warp_inst_t *inst, unsigned int warp_id, unsigned int read_cycles) {
  RF_instruction_read_request res(m_num_banks);
  int max_num_uses = 0;
  bool can_inst_read_from_rfc = can_read_from_rf_cache(inst);
  if(!m_is_unlimited_reads_per_cycles) {
    // First dimension is the number of sets, second is the ID of registers that are going to request a read to RF

    unsigned int first_read_operand = inst->get_extra_trace_instruction_info().get_num_destination_registers();
    unsigned int operand_position = first_read_operand;
    unsigned int operand_rf_cache_position = 0;

    for(unsigned int i = first_read_operand; i < inst->get_extra_trace_instruction_info().get_num_operands(); i++){
      bool is_operand_cached = false;
      if(is_operand_to_be_considered(inst->get_extra_trace_instruction_info().get_operand(i))) {
        traced_instruction& trace_inst = inst->get_extra_trace_instruction_info();
        int num_uses = get_number_of_uses_per_operand(trace_inst, trace_inst.get_operand(i).get_operand_reg_number(), operand_position, trace_inst.get_operand(i).get_operand_type());
        max_num_uses = std::max(max_num_uses, num_uses);
        for(int j = 0; j < num_uses; j++) {
          unsigned int current_reg_id = inst->get_extra_trace_instruction_info().get_operand(i).get_operand_reg_number() + j;
          if(m_is_rf_cache_enabled && can_inst_read_from_rfc && (inst->get_extra_trace_instruction_info().get_operand(i).get_operand_type() == TraceEnhancedOperandType::REG))  {
            is_operand_cached = m_rf_cache->is_hit(operand_rf_cache_position, warp_id, current_reg_id);
            if(inst->get_extra_trace_instruction_info().get_operand(i).is_reuse_bit_set()){
              res.m_rf_cache_allocate_requests.push_back(RF_cache_action(operand_rf_cache_position, warp_id, current_reg_id));
            }      
          }

          if(is_operand_cached) {
            res.m_rf_cache_read_requests.push_back(RF_cache_action(operand_rf_cache_position, warp_id, current_reg_id));
          } else{
            res.m_requested_reads[calculate_target_bank(current_reg_id)].insert(current_reg_id);
          }
        }
      }
      if(inst->get_extra_trace_instruction_info().get_operand(i).get_operand_type() == TraceEnhancedOperandType::REG) {
        operand_rf_cache_position++;
      }
      operand_position++;
    }
    unsigned int latency_read_fixed_latency_inst = inst->is_tensor_core_op_with_4_registers_per_op() ? MAXIMUM_LATENCY_READ_FIXED_LATENCY_INST : NO_TENSOR_OP_4REG_PER_OP_LATENCY_READ_FIXED_LATENCY_INST;
    for(unsigned int i = 0; (i < m_num_banks) && res.m_is_possible_to_read; i++){
      res.m_is_possible_to_read = m_banks[i].is_read_available(1, 1 + latency_read_fixed_latency_inst ,res.m_requested_reads[i].size() * read_cycles);
    }
  }
  res.max_slack_due_to_double_use_of_banks = compute_read_slack(max_num_uses);
  return res;
}


RF_instruction_read_request Register_file::is_possible_to_read_non_cacheable(const warp_inst_t *inst, unsigned int read_cycles) {
  RF_instruction_read_request res(m_num_banks);
  int max_num_uses = 0;
  if(!m_is_unlimited_reads_per_cycles) {
    // First dimension is the number of sets, second is the ID of registers that are going to request a read to RF

    unsigned int first_read_operand = inst->get_extra_trace_instruction_info().get_num_destination_registers();
    unsigned int operand_position = first_read_operand;

    for(unsigned int i = first_read_operand;( i < inst->get_extra_trace_instruction_info().get_num_operands()) && res.m_is_possible_to_read; i++){
      if(is_operand_to_be_considered(inst->get_extra_trace_instruction_info().get_operand(i))) {
        traced_instruction& trace_inst = inst->get_extra_trace_instruction_info();
        int num_uses = get_number_of_uses_per_operand(trace_inst, trace_inst.get_operand(i).get_operand_reg_number(), operand_position, trace_inst.get_operand(i).get_operand_type());
        max_num_uses = std::max(max_num_uses, num_uses);
        for(int j = 0; (j < num_uses) && res.m_is_possible_to_read; j++) {
          unsigned int current_reg_id = inst->get_extra_trace_instruction_info().get_operand(i).get_operand_reg_number() + j;
          unsigned int bank_id = calculate_target_bank(current_reg_id);
          res.m_requested_reads[bank_id].insert(current_reg_id);
        }
      }
      operand_position++;
    }
    for(unsigned int i = 0; (i < m_num_banks) && res.m_is_possible_to_read; i++){
      res.m_is_possible_to_read = m_banks[i].is_read_available(1, 2, 1); 
    }
  }
  res.max_slack_due_to_double_use_of_banks = compute_read_slack(max_num_uses);
  return res;
}

void Register_file::allocate_reads_cacheable(RF_instruction_read_request rf_requests,
                                   const warp_inst_t *pI, unsigned int warp_id, unsigned int read_cycles) {
  assert(rf_requests.m_is_possible_to_read);
  unsigned int first_read_operand = pI->get_extra_trace_instruction_info().get_num_destination_registers();
  unsigned int operand_position = first_read_operand;
  unsigned int operand_rf_cache_position = 0;
  bool can_inst_read_from_rfc = can_read_from_rf_cache(pI);
  for(unsigned int i = first_read_operand; (i < pI->get_extra_trace_instruction_info().get_num_operands() ) && m_is_rf_cache_enabled && can_inst_read_from_rfc; i++) {
    if(is_operand_to_be_considered(pI->get_extra_trace_instruction_info().get_operand(i))) {
      traced_instruction& trace_inst = pI->get_extra_trace_instruction_info();
      unsigned int num_uses = get_number_of_uses_per_operand(trace_inst, trace_inst.get_operand(i).get_operand_reg_number(), operand_position, trace_inst.get_operand(i).get_operand_type());
      for(unsigned j = 0; j < num_uses; j++) {
        unsigned int current_reg_id = pI->get_extra_trace_instruction_info().get_operand(i).get_operand_reg_number() + j;
        if(pI->get_extra_trace_instruction_info().get_operand(i).get_operand_type() == TraceEnhancedOperandType::REG) {
          m_rf_cache->flush_entry(operand_rf_cache_position, current_reg_id);
        }
        if(pI->get_extra_trace_instruction_info().get_operand(i).is_reuse_bit_set()) {
          m_rf_cache->allocate_new_register(operand_rf_cache_position, warp_id, current_reg_id);
        }
      }  
    }

    if(pI->get_extra_trace_instruction_info().get_operand(i).get_operand_type() == TraceEnhancedOperandType::REG) {
      operand_rf_cache_position++;
    }
    operand_position++;
  }
  unsigned int latency_read_fixed_latency_inst = pI->is_tensor_core_op_with_4_registers_per_op() ? MAXIMUM_LATENCY_READ_FIXED_LATENCY_INST : NO_TENSOR_OP_4REG_PER_OP_LATENCY_READ_FIXED_LATENCY_INST;
  for(unsigned int i = 0; i < m_num_banks; i++){
    m_banks[i].allocate_read_ports(1, 1 + latency_read_fixed_latency_inst + rf_requests.max_slack_due_to_double_use_of_banks, rf_requests.m_requested_reads[i].size() * read_cycles);
  }
}


void Register_file::allocate_reads_non_cacheable(RF_instruction_read_request rf_requests, const warp_inst_t *pI, unsigned int read_cycles) {
  assert(rf_requests.m_is_possible_to_read);
  for(unsigned int i = 0; i < m_num_banks; i++){
    m_banks[i].allocate_read_ports(1, 2 + rf_requests.max_slack_due_to_double_use_of_banks, 1);
  }
}

unsigned int Register_file::calculate_target_bank(unsigned int reg_id){
  return reg_id % m_num_banks;
}

bool Register_file::is_rf_bank_write_port_available_this_cycle(unsigned int bank_id) {
  assert(bank_id < m_num_banks);
  return m_banks[bank_id].is_write_available_this_cycle();
}

bool Register_file::is_rf_bank_write_port_available_at_given_cycle(unsigned int bank_id, unsigned int given_cycle) {
  assert(bank_id < m_num_banks);
  return m_banks[bank_id].is_write_available_at_given_cycle(given_cycle);
}

void Register_file::allocate_rf_bank_write_port_this_cycle(unsigned int bank_id) {
  assert(bank_id < m_num_banks);
  m_banks[bank_id].allocate_write_port_this_cycle();
}

void Register_file::allocate_rf_bank_write_port_at_given_cycle(unsigned int bank_id, unsigned int given_cycle) {
  assert(bank_id < m_num_banks);
  m_banks[bank_id].allocate_write_port_at_given_cycle(given_cycle);
}

void Register_file::flush() {
  for(auto &bank : m_banks){
    bank.cycle();
  }
  m_rf_cache->flush();
}

bool Register_file::can_read_from_rf_cache(const warp_inst_t *inst) {
  bool res = false;
  if(m_type == TraceEnhancedOperandType::REG) {
    switch (inst->op) {
    case INTP_OP:
    case SP_OP:
    case DP_OP:
    case HALF_OP:
    case TENSOR_CORE_OP:
    case PREDICATE_OP:
      res = true;
      break;
    default:
      res = false;
      break;
    }
  }
  return res;
}