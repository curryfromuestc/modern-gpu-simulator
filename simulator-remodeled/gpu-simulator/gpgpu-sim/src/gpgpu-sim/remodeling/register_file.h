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
#include <cstdio>
#include <memory>
#include <set>

#include "../../../../../util/traces_enhanced/src/traced_constants.h"


class shader_core_stats;
class Subcore;
class Register_file_cache;
class Register_file;
class warp_inst_t;
class traced_operand;

struct RF_cache_action {
  RF_cache_action(unsigned int operand_position, unsigned int warp_id, unsigned int reg_id) {
    m_operand_position = operand_position;
    m_warp_id = warp_id;
    m_reg_id = reg_id;
  }
  unsigned int m_operand_position;
  unsigned int m_warp_id;
  unsigned int m_reg_id;
};

struct RF_instruction_read_request {
  
  RF_instruction_read_request() {
    m_is_possible_to_read = false;
    max_slack_due_to_double_use_of_banks = 0;
  }

  RF_instruction_read_request(unsigned int num_banks) {
    m_is_possible_to_read = true;
    m_requested_reads.resize(num_banks);
    max_slack_due_to_double_use_of_banks = 0;
  }

  bool m_is_possible_to_read;
  std::vector<std::set<unsigned int>> m_requested_reads;
  std::vector<RF_cache_action> m_rf_cache_read_requests;
  std::vector<RF_cache_action> m_rf_cache_allocate_requests;

  unsigned int max_slack_due_to_double_use_of_banks;

};

struct RF_requests {
  RF_requests(){};
  bool is_possible_to_read() {
    return m_regular.m_is_possible_to_read && m_uniform.m_is_possible_to_read;// && m_predicate.m_is_possible_to_read && m_predicate_uniform.m_is_possible_to_read && m_B.m_is_possible_to_read;
  }
  RF_instruction_read_request m_regular;
  RF_instruction_read_request m_uniform;
  RF_instruction_read_request m_predicate;
  RF_instruction_read_request m_predicate_uniform;
  RF_instruction_read_request m_B;
};

class Register_file_cache_entry {
 public:
  Register_file_cache_entry(shader_core_stats *stats, Register_file_cache* rf_cache);
  bool is_valid();
  void set_valid(bool valid);
  unsigned int get_warp_id();
  void set_warp_id(unsigned int warp_id);
  unsigned int get_reg_id();
  void set_reg_id(unsigned int reg_id);
  void allocate_new_register(unsigned int warp_id, unsigned int reg_id);
  bool is_hit(unsigned int warp_id, unsigned int reg_id);
  void flush();
  void print(FILE *fp);
 private:
  bool m_is_valid;
  unsigned int m_warp_id;
  unsigned int m_reg_id;
  Register_file_cache *m_rf_cache;
  shader_core_stats *m_stats;
};

class Register_file_cache {
 public:
  Register_file_cache(unsigned int num_banks, unsigned int max_num_operands, shader_core_stats *stats,
                      Register_file* rf);
  ~Register_file_cache();
  void init();
  Register_file_cache* getptr();
  void allocate_new_register(unsigned int operand_pos, unsigned int warp_id, unsigned int reg_id);
  bool is_hit(unsigned int operand_pos, unsigned int warp_id, unsigned int reg_id);
  void flush();
  void flush_entry(unsigned int operand_pos, unsigned int reg_id);
  Register_file *get_register_file();
  void print(FILE *fp);
 private:
  // First dimension is the number of operands, second dimension is the number of banks
  std::vector<std::vector<Register_file_cache_entry>> m_entries;
  unsigned int m_num_banks;
  unsigned int m_max_num_operands;
  shader_core_stats *m_stats;
  Register_file *m_rf;
};

class Register_file_bank {
  public:
    Register_file_bank(unsigned int num_read_ports, unsigned int num_write_ports, unsigned int max_supported_latency, Register_file* rf);
    void cycle();
    unsigned int get_num_ports();
    unsigned int get_max_supported_latency();
    bool is_read_available(unsigned int init_latency, unsigned int max_latency, unsigned int num_reads);
    void allocate_read_ports(unsigned int init_latency, unsigned int max_latency, unsigned int num_reads);
    bool is_write_available_this_cycle();
    bool is_write_available_at_given_cycle(unsigned int given_cycle);
    void allocate_write_port_this_cycle();
    void allocate_write_port_at_given_cycle(unsigned int given_cycle);
    void print(FILE *fp);
  private:
    unsigned int m_num_read_ports;
    unsigned int m_num_write_ports;
    unsigned int m_max_supported_latency;
    Register_file* m_rf;
    // It has a list of cycles 
    std::vector<unsigned int> m_num_used_read_ports;
    std::vector<unsigned int> m_num_used_write_ports;
};

class Register_file {
 public:
  Register_file(unsigned int num_banks, unsigned int num_read_ports_per_bank, unsigned int num_write_ports_per_bank, unsigned int max_supported_latency, bool is_unlimited_reads_per_cycles, bool is_unlimited_writes_per_cycles, unsigned int max_num_operands,
                shader_core_stats *stats, Subcore* subcore, TraceEnhancedOperandType type, bool is_rf_cache_enabled);
  ~Register_file();
  Register_file* getptr();
  void init();
  Subcore* get_subcore();
  TraceEnhancedOperandType get_type();
  unsigned int get_num_banks();
  unsigned int get_num_read_ports_per_bank();
  unsigned int get_num_write_ports_per_bank();
  unsigned int get_max_supported_latency();
  unsigned int get_max_num_operands();
  bool is_unlimited_reads_per_cycles();
  bool is_unlimited_writes_per_cycles();
  bool is_rf_cache_enabled();
  bool is_rf_bank_write_port_available_this_cycle(unsigned int bank_id);
  bool is_rf_bank_write_port_available_at_given_cycle(unsigned int bank_id, unsigned int given_cycle);
  void allocate_rf_bank_write_port_this_cycle(unsigned int bank_id);
  void allocate_rf_bank_write_port_at_given_cycle(unsigned int bank_id, unsigned int given_cycle);
  shader_core_stats *get_stats();
  RF_instruction_read_request is_possible_to_read_cacheable(const warp_inst_t *inst, unsigned int warp_id, unsigned int read_cycles);
  RF_instruction_read_request is_possible_to_read_non_cacheable(const warp_inst_t *inst, unsigned int read_cycles);
  void allocate_reads_cacheable(RF_instruction_read_request rf_requests, const warp_inst_t *pI, unsigned int warp_id, unsigned int read_cycles);
  void allocate_reads_non_cacheable(RF_instruction_read_request rf_requests, const warp_inst_t *pI, unsigned int read_cycles);
  void cycle();
  unsigned int calculate_target_bank(unsigned int reg_id);
  bool is_operand_to_be_considered(traced_operand &operand);
  void flush();
  void print(FILE *fp);
 private:
  shader_core_stats *m_stats;
  Subcore* m_subcore;
  bool m_is_rf_cache_enabled;
  bool m_is_unlimited_reads_per_cycles;
  bool m_is_unlimited_writes_per_cycles;
  Register_file_cache *m_rf_cache;
  unsigned int m_num_banks;
  unsigned int m_num_read_ports_per_bank;
  unsigned int m_num_write_ports_per_bank;
  unsigned int m_max_num_operands;
  unsigned int m_max_supported_latency;
  TraceEnhancedOperandType m_type;
  std::vector<Register_file_bank> m_banks;
  bool can_read_from_rf_cache(const warp_inst_t *inst);
  int compute_read_slack(int max_uses);
};