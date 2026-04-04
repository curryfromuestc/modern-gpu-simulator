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
#include <map>
#include <set>

#include "JSONBase.h"
#include "JSONIncludes.h"

#include "traced_instruction.h"

bool track_this_instruction(int icount, int threshold_unique_kernel_checking, std::string opcode_str);

std::shared_ptr<traced_instruction> create_no_binay_instruction(unsigned int pc, std::string full_instruction_str);

class traced_kernel : public JSONBase {

    public:
        traced_kernel(std::string kernel_name, unsigned int architecture_version);
        traced_kernel(); // Used for Deserializing

        virtual ~traced_kernel();

        virtual bool Deserialize(const rapidjson::Value& obj);
        virtual bool Serialize(rapidjson::Writer<rapidjson::StringBuffer>* writer) const;

        std::string get_kernel_name() const;

        traced_instruction& get_instruction(unsigned int pc);
        std::shared_ptr<traced_instruction> get_instruction_ptr(unsigned int pc);

        void set_kernel_name(std::string kernel_name);

        bool is_kernel_in_use() const;

        std::map<int, std::string> get_key_instructions_pcs() const;

        void set_unique_function_id(unsigned int unique_function_id);

        unsigned int get_unique_function_id();

        uint64_t get_function_addr();

        void set_function_addr(uint64_t func_addr);

        bool is_captured_from_binary() const;

        void set_is_captured_from_binary(bool is_captured_from_binary);

        void add_instruction(std::vector<std::string> instruction_part1, std::vector<std::string> instruction_part2, int threshold_unique_kernel_checking, std::string full_instruction_str);
        
        void add_no_binary_instruction(unsigned int pc, std::string full_instruction_str);

        void add_register_usage(unsigned int pc, std::string reg_file_name, unsigned int num_regs);

        void add_call_target(unsigned int pc, std::string target_name, int target_id);

        traced_instruction& get_instruction(std::string pc);

        void set_opcode_map();

    private:
        std::string m_kernel_name;
        unsigned int m_architecture_version;
        unsigned int m_unique_function_id;
        uint64_t m_func_addr;
        std::map<unsigned int, std::shared_ptr<traced_instruction>> m_instructions;
        std::map<int, std::string> m_key_instructions_pcs; // Used to uniquely identify kernels that can have the same name
        bool m_is_captured_from_binary;
        unsigned int m_last_parsed_pc;
        const std::unordered_map<std::string, OpcodeChar> *m_OpcodeMap;
};