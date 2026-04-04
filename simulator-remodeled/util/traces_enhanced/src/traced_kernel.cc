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

#include "traced_kernel.h"

#include <assert.h>
#include <algorithm>
#include <sstream>
#include "traced_operand.h"
#include "string_utilities.h"

bool track_this_instruction(int icount, int threshold_unique_kernel_checking, std::string opcode_str) {
  bool res = (icount < threshold_unique_kernel_checking) && (opcode_str.find("BRA") == std::string::npos) && (opcode_str.find("BSSY") == std::string::npos) && (opcode_str.find("CALL") == std::string::npos) && (opcode_str.find("RET") == std::string::npos) && (opcode_str.find("MOV") == std::string::npos);
  return res;
}

std::shared_ptr<traced_instruction> create_no_binay_instruction(unsigned int pc, std::string instruction_str) {
    unsigned int  next_idx_to_check = 0;
    bool is_predicated = false;
    bool is_predicate_negate = false;
    bool is_uniform_predicate = false;
    unsigned int predicate_register = 0;
    std::vector<std::string> stripped_inst = split_string(instruction_str, ' ');
    if(stripped_inst[next_idx_to_check][0] == '@') {
        is_predicated = true;
        unsigned int predicate_idx_to_check = 1;
        if(stripped_inst[next_idx_to_check][predicate_idx_to_check] == '!') {
            is_predicate_negate = true;
            predicate_idx_to_check++;
        }
        if(stripped_inst[next_idx_to_check][predicate_idx_to_check] == 'U') {
            is_uniform_predicate = true;
            predicate_idx_to_check++;
        }
        predicate_idx_to_check++;

        if(stripped_inst[next_idx_to_check][predicate_idx_to_check] == 'T') {
            predicate_register = PT;
        } else {
            predicate_register = std::stoul(std::string(1, stripped_inst[next_idx_to_check][predicate_idx_to_check]), nullptr, 10);
        }
        next_idx_to_check++;
    }
    std::string op_code = stripped_inst[next_idx_to_check];
    std::stringstream streamStrPC;
    streamStrPC << std::hex << pc;
    std::string pc_string( streamStrPC.str() );
    std::shared_ptr<traced_instruction> instruction = std::make_shared<traced_instruction>(pc_string, pc, op_code, is_predicated, is_uniform_predicate, is_predicate_negate, predicate_register);
    next_idx_to_check++;

    // Create operands
    for(unsigned i = next_idx_to_check; i < stripped_inst.size(); i++) {
        if(!stripped_inst[i].empty() && (stripped_inst[i][0]!='&') && (stripped_inst[i][0]!='?') && (stripped_inst[i][0]!=';')) {
            instruction->add_operand(std::make_unique<traced_operand>(stripped_inst[i]));
        }
    }
    instruction->calculate_num_destination_registers();
    return instruction;
}

traced_kernel::traced_kernel(std::string kernel_name, unsigned int architecture_version) {
    m_kernel_name = kernel_name;
    m_architecture_version = architecture_version;
    m_unique_function_id = 0;
    m_func_addr = 0;
    m_is_captured_from_binary = false;
    m_last_parsed_pc = 0;
    m_OpcodeMap = nullptr;
}

traced_kernel::traced_kernel() {
    // Used for Deserializing
}

traced_kernel::~traced_kernel() {
    m_instructions.clear();
    m_key_instructions_pcs.clear();
}

void traced_kernel::set_unique_function_id(unsigned int unique_function_id) {
    m_unique_function_id = unique_function_id;
}

unsigned int traced_kernel::get_unique_function_id() {
    return m_unique_function_id;
}

uint64_t traced_kernel::get_function_addr() {
    return m_func_addr;
}

void traced_kernel::set_function_addr(uint64_t function_addr) {
    m_func_addr = function_addr;
}

bool traced_kernel::is_captured_from_binary() const {
    return m_is_captured_from_binary;
}

void traced_kernel::set_is_captured_from_binary(bool is_captured_from_binary) {
    m_is_captured_from_binary = is_captured_from_binary;
}

void traced_kernel::add_instruction(std::vector<std::string> instruction_part1, std::vector<std::string> instruction_part2, int threshold_unique_kernel_checking, std::string full_instruction_str) {
    assert(instruction_part1.size() > 0);
    if(instruction_part1.size() > 1) {
        std::string pc_string = instruction_part1[0];
        unsigned int pc_num = std::stoul(pc_string, nullptr, 16);
        bool already_exists = m_instructions.find(pc_num) != m_instructions.end();
        assert(!already_exists);
        unsigned int  next_idx_to_check = 1;
        bool is_predicated = false;
        bool is_predicate_negate = false;
        bool is_uniform_predicate = false;
        unsigned int predicate_register = 0;
        if(instruction_part1[next_idx_to_check][0] == '@') {
            is_predicated = true;
            unsigned int predicate_idx_to_check = 1;
            if(instruction_part1[next_idx_to_check][predicate_idx_to_check] == '!') {
                is_predicate_negate = true;
                predicate_idx_to_check++;
            }
            if(instruction_part1[next_idx_to_check][predicate_idx_to_check] == 'U') {
                is_uniform_predicate = true;
                predicate_idx_to_check++;
            }
            predicate_idx_to_check++;

            if(instruction_part1[next_idx_to_check][predicate_idx_to_check] == 'T') {
                predicate_register = PT;
            } else {
                predicate_register = std::stoul(std::string(1,instruction_part1[next_idx_to_check][predicate_idx_to_check]), nullptr, 10);
            }
            next_idx_to_check++;
        }

        std::string op_code = instruction_part1[next_idx_to_check];
        if(track_this_instruction(m_instructions.size(), threshold_unique_kernel_checking, op_code)) {
            std::string sass_instr = create_sass_instr(full_instruction_str, false, "");
            m_key_instructions_pcs[pc_num] = sass_instr;
        }
        next_idx_to_check++;

        std::vector<std::string> encoded_instruction;
        encoded_instruction.push_back(instruction_part1[instruction_part1.size() - 1]);
        if(m_architecture_version >= 70) {
            assert(instruction_part2.size() > 0);
            encoded_instruction.push_back(instruction_part2[0]);
        }
        m_instructions[pc_num] = std::make_shared<traced_instruction>(pc_string, pc_num, op_code, encoded_instruction, is_predicated, is_uniform_predicate, is_predicate_negate, predicate_register);
        
        // Create operands
        for(unsigned i = next_idx_to_check; (i < instruction_part1.size() -1); i++) {
            if(!instruction_part1[i].empty() && (instruction_part1[i][0]!='&') && (instruction_part1[i][0]!='?') && (instruction_part1[i][0]!=';')) {
                m_instructions[pc_num]->add_operand(std::make_unique<traced_operand>(instruction_part1[i]));
            }
        }

        m_instructions[pc_num]->calculate_num_destination_registers();
        m_last_parsed_pc = pc_num;
    }else {
        unsigned int candidate_pc = m_last_parsed_pc + 16;
        std::string pc_str = decimalToHexString(candidate_pc, 4);
        m_instructions[candidate_pc] = std::make_shared<traced_instruction>(pc_str, candidate_pc, "NOP", false, false, false, 0);
        m_last_parsed_pc = candidate_pc;
    }
}

void traced_kernel::add_no_binary_instruction(unsigned int pc, std::string instruction_str) {
    bool already_exists = m_instructions.find(pc) != m_instructions.end();
    assert(!already_exists);
    m_instructions[pc] = create_no_binay_instruction(pc, instruction_str);
}

void traced_kernel::add_register_usage(unsigned pc, std::string reg_file_name, unsigned num_regs) {
    assert(m_instructions.find(pc) != m_instructions.end());
    m_instructions[pc]->add_register_usage(reg_file_name, num_regs);
}

void traced_kernel::add_call_target(unsigned pc, std::string target_name, int target_id) {
    assert(m_instructions.find(pc) != m_instructions.end());
    m_instructions[pc]->add_call_target(target_name, target_id);
}

std::string traced_kernel::get_kernel_name() const {
    return m_kernel_name;
}

traced_instruction& traced_kernel::get_instruction(unsigned pc) {
    assert(m_instructions.find(pc) != m_instructions.end());
    return *m_instructions[pc];
}

std::shared_ptr<traced_instruction> traced_kernel::get_instruction_ptr(unsigned pc) {
    if (m_instructions.find(pc) != m_instructions.end())
    {
        return m_instructions[pc];
    }
    else
    {
        return std::shared_ptr<traced_instruction>(nullptr);
    }
}

bool traced_kernel::is_kernel_in_use() const {
    return m_unique_function_id != 0;
}

std::map<int, std::string> traced_kernel::get_key_instructions_pcs() const {
    return m_key_instructions_pcs;
}

void traced_kernel::set_kernel_name(std::string kernel_name) {
    m_kernel_name = kernel_name;
}

void traced_kernel::set_opcode_map() {
    get_opcode_map(m_OpcodeMap, m_architecture_version);
}

bool traced_kernel::Serialize(rapidjson::Writer<rapidjson::StringBuffer> * writer) const {
    writer->StartObject();

    writer->String("kernel_name"); 
    writer->String(m_kernel_name.c_str());

    writer->String("architecture_version"); 
    writer->Int64(m_architecture_version);

    writer->String("is_captured_from_binary");
    writer->Bool(m_is_captured_from_binary);

    writer->String("unique_function_id");
    writer->Int64(m_unique_function_id);

    writer->String("function_address");
    writer->Uint64(m_func_addr);

    std::vector<unsigned> keys_pc;
    for(auto it = m_instructions.begin(); it != m_instructions.end(); it++) {
        keys_pc.push_back(it->first);
    }
    std::sort(keys_pc.begin(), keys_pc.end());
    writer->String("instructions"); 
    writer->StartArray();
    for(auto it = keys_pc.begin(); it != keys_pc.end(); it++) {
        m_instructions.at(*it)->Serialize(writer);
    }
    writer->EndArray();

    writer->EndObject();

    return true;
}


bool traced_kernel::Deserialize(const rapidjson::Value & obj) {
    m_kernel_name = obj["kernel_name"].GetString();

    m_architecture_version = obj["architecture_version"].GetInt64();

    set_opcode_map();

    m_is_captured_from_binary = obj["is_captured_from_binary"].GetBool();

    m_unique_function_id = obj["unique_function_id"].GetInt64();    

    m_func_addr = obj["function_address"].GetUint64();

    const rapidjson::Value &instructions = obj["instructions"];
    for (rapidjson::SizeType i = 0; i < instructions.Size(); i++)
    {
        std::shared_ptr<traced_instruction> instruction = std::make_shared<traced_instruction>();
        instruction->Deserialize(instructions[i]);
        instruction->set_simulation_opcode(m_OpcodeMap, instruction->get_op_code());
        unsigned pc_num = std::stoul(instruction->get_pc_string(), nullptr, 16);
        m_instructions[pc_num] = std::move(instruction);
    }
  return true;
}