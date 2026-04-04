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

#include <string>
#include <vector>
#include <memory>
#include "traced_constants.h"
#include "traced_operand.h"
#include "register_usage.h"
#include "control_bits.h"

#include "JSONBase.h"
#include "JSONIncludes.h"

#include "../../../gpu-simulator/gpgpu-sim/src/operation_type.h"
#include "../../../gpu-simulator/ISA_Def/trace_opcode.h"


enum membar_traced_type {
    MEMBAR_TRACED_NONE = 0,
    MEMBAR_TRACED_CTA,
    MEMBAR_TRACED_VC,
    MEMBAR_TRACED_GPU,
    MEMBAR_TRACED_SYS
};

enum s2r_operand_type {
    NONE_S2R_OPERAND_TYPE = 0,
    CTA_ID_X_S2R_OPERAND_TYPE,
    CTA_ID_Y_S2R_OPERAND_TYPE,
    CTA_ID_Z_S2R_OPERAND_TYPE,
    THREAD_ID_X_S2R_OPERAND_TYPE,
    THREAD_ID_Y_S2R_OPERAND_TYPE,
    THREAD_ID_Z_S2R_OPERAND_TYPE,
    LANE_ID_S2R_OPERAND_TYPE,
    LTMASK_S2R_OPERAND_TYPE
};

class traced_instruction;

bool is_reserved_reg(unsigned int reg_id, TraceEnhancedOperandType reg_type);

bool has_the_operand_type_been_captured(TraceEnhancedOperandType reg_type);

void advance_operand_idx(unsigned int &operand_idx, traced_instruction &static_inst_info);

void get_opcode_map(const std::unordered_map<std::string, OpcodeChar> *&OpcodeMap, int binary_verion);

s2r_operand_type get_s2r_operand_type(std::string operand_string);

unsigned int get_number_of_uses_per_operand(traced_instruction &trace_inst, unsigned int reg_id,  unsigned int operand_pos, TraceEnhancedOperandType reg_type);

struct Tensor_core_instruction_information {
  Tensor_core_instruction_information(){
    this->is_set = false;
    this->is_sparse = false;
    this->size_m = 0;
    this->size_n = 0;
    this->size_k = 0;
    this->operand_bit_size = 0;
    this->is_16816_fp32_1688_fp32 = false;
  }
  Tensor_core_instruction_information(unsigned int size_m, unsigned int size_n, unsigned int size_k, unsigned int operand_bit_size, bool is_sparse, bool is_16816_fp32_1688_fp32) {
    this->size_m = size_m;
    this->size_n = size_n;
    this->size_k = size_k;
    this->operand_bit_size = operand_bit_size;
    this->is_sparse = is_sparse;
    this->is_set = true;
    this->is_16816_fp32_1688_fp32 = is_16816_fp32_1688_fp32;
  }
  unsigned int size_m;
  unsigned int size_n;
  unsigned int size_k;
  unsigned int operand_bit_size;
  bool is_sparse;
  bool is_set;
  bool is_16816_fp32_1688_fp32;
};

class traced_instruction : public JSONBase{

    public:
        traced_instruction(std::string pc_string, unsigned pc_num, std::string op_code, std::vector<std::string> encoded_instruction, bool is_predicated, bool is_uniform_predicate, bool is_predicate_negate, unsigned predicate_register);
        traced_instruction(std::string pc_string, unsigned pc_num, std::string op_code, bool is_predicated, bool is_uniform_predicate, bool is_predicate_negate, unsigned predicate_register);
        traced_instruction(); // Used for Deserializing

        ~traced_instruction();

        virtual bool Deserialize(const rapidjson::Value& obj);
        virtual bool Serialize(rapidjson::Writer<rapidjson::StringBuffer>* writer) const;

        std::string get_pc_string() const;
        void set_pc_string(std::string pc_string);

        void add_operand(std::unique_ptr<traced_operand> operand);

        void add_register_usage(std::string reg_file_name, unsigned num_regs);

        void add_call_target(std::string target_name, int target_id);

        void calculate_num_destination_registers();

        unsigned int get_num_destination_registers();

        bool has_destination_registers();

        control_bits& get_control_bits();

        std::vector<std::unique_ptr<traced_operand>> &get_operands();
        
        std::size_t get_num_operands();

        traced_operand& get_operand(unsigned int idx);

        bool get_is_imad();

        bool get_is_system_memory_barrier();

        bool get_is_cta_memory_barrier();

        bool get_is_call_or_ret_with_relative();

        bool get_is_imad_wide();
        
        bool get_contains_setp();

        std::string get_op_code();

        void set_op_code(std::string op_code);

        bool is_first_operand_of_mref_cbank_desc_using_regular_reg(unsigned int idx_op);

        Tensor_core_instruction_information& get_tensor_core_instruction_info();

        void set_tensor_core_instruction_info();

        void set_simulation_opcode(const std::unordered_map<std::string, OpcodeChar> *OpcodeMap, std::string str_op);
        op_type get_simulation_opcode();
        bool is_dp_op();
        bool is_load_op();
        bool is_store_op();
        bool is_tensor_core_op();

    private:
        std::string m_pc_string;
        unsigned int m_pc_num;
        std::string m_op_code;
        bool m_is_predicated;
        bool m_is_uniform_predicate;
        bool m_is_predicate_negate;
        unsigned int m_predicate_register;
        std::vector<std::string> m_encoded_instruction;
        std::vector<std::unique_ptr<traced_operand>> m_operands;
        unsigned int m_num_destination_registers;
        std::unique_ptr<register_usage> m_register_usage;
        std::unique_ptr<control_bits> m_control_bits;
        bool m_is_imad;
        bool m_is_imad_wide;
        bool m_contains_setp;
        membar_traced_type m_membar_type;
        Tensor_core_instruction_information m_tensor_core_instruction_info;
        op_type m_simulation_opcode;
};