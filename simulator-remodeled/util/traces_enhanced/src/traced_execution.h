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

#include "traced_kernel.h"

#include "JSONBase.h"
#include "JSONIncludes.h"

#include <string>

struct search_func_addr_result {
    search_func_addr_result() {
        m_new_func_base_addr = 0;
        m_unique_function_id = 0;
        m_kernel_name = "";
        m_has_been_traced = false;
    }

    search_func_addr_result(uint64_t new_func_base_addr, unsigned int unique_function_id, std::string kernel_name, bool has_been_traced) {
        m_new_func_base_addr = new_func_base_addr;
        m_unique_function_id = unique_function_id;
        m_kernel_name = kernel_name;
        m_has_been_traced = has_been_traced;
    }

    uint64_t m_new_func_base_addr;
    unsigned int m_unique_function_id;
    std::string m_kernel_name;
    bool m_has_been_traced;
};

class traced_execution : public JSONBase
{

public:
    traced_execution(std::string benchmark_name);
    traced_execution(); // Used for Deserializing
    ~traced_execution();

    virtual bool Deserialize(const rapidjson::Value &obj);
    virtual bool Serialize(rapidjson::Writer<rapidjson::StringBuffer> *writer) const;

    std::string get_benchmark_name() const;

    bool has_kernel(std::string kernel_name);

    bool has_kernel_with_unique_function_id(unsigned int unique_function_id);

    traced_kernel &get_kernel(std::string kernel_name);

    traced_kernel &get_kernel_by_unique_function_id(unsigned int unique_function_id);

    unsigned int get_unique_function_id(std::string kernel_name);

    search_func_addr_result search_function_addr(uint64_t func_addr);

    void set_benchmark_name(std::string benchmark_name);

    void add_traced_kernel(std::string kernel_name, unsigned int unique_function_id, traced_kernel *kernel, uint64_t func_addr, bool is_captured_from_binary);

    void add_no_binary_kernel(std::string kernel_name, unsigned int unique_function_id, uint64_t func_addr, unsigned int architecture_version, std::map<int,std::string> &captured_instrs , bool is_captured_from_binary);

    void add_instruction_to_a_kernel(std::string kernel_name, std::vector<std::string> instruction_part1, std::vector<std::string> instruction_part2, int threshold_unique_kernel_checking, std::string full_instruction_str);

    void add_register_usage_to_a_kernel_instruction(std::string kernel_name, unsigned pc, std::string reg_file_name, unsigned num_regs);

    void add_register_usage_to_a_kernel(std::string kernel_name, std::map<unsigned int, std::map<std::string, unsigned int>> &reg_usage, std::map<unsigned int, std::tuple<std::string, int>> &call_target_by_pc);

    void remove_useless_kernels();
    
    void remove_specific_kernel(std::string kernel_name);

    void print(FILE *fp);

private:
    std::map<std::string, traced_kernel *> m_kernels;
    std::map<unsigned int, std::string> m_function_id_to_kernel_name;
    std::map<uint64_t, unsigned int> m_func_addr_to_unique_function_id;
    std::string m_benchmark_name;

    void generate_kernel_maps();
};
