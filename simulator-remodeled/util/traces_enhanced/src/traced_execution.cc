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

#include "traced_execution.h"

#include <assert.h>
#include "string_utilities.h"


traced_execution::traced_execution(std::string benchmark_name) {
    m_benchmark_name = benchmark_name;
}

traced_execution::traced_execution() {
    // Used for Deserializing
}

traced_execution::~traced_execution() {
    for(auto it = m_kernels.begin(); it != m_kernels.end(); it++) {
        delete it->second;
    }
    m_kernels.clear();
}

bool traced_execution::has_kernel(std::string kernel_name) {
    return m_kernels.find(kernel_name) != m_kernels.end();
}

bool traced_execution::has_kernel_with_unique_function_id(unsigned int unique_function_id) {
    return m_function_id_to_kernel_name.find(unique_function_id) != m_function_id_to_kernel_name.end();
}

traced_kernel& traced_execution::get_kernel(std::string kernel_name) {
    assert(m_kernels.find(kernel_name) != m_kernels.end());
    return *m_kernels[kernel_name];
}

traced_kernel &traced_execution::get_kernel_by_unique_function_id(unsigned int unique_function_id) {
    auto it_funct = m_function_id_to_kernel_name.find(unique_function_id);
    assert(it_funct != m_function_id_to_kernel_name.end());
    return get_kernel(it_funct->second);
}

unsigned int traced_execution::get_unique_function_id(std::string kernel_name) {
    assert(m_kernels.find(kernel_name) != m_kernels.end());
    return m_kernels[kernel_name]->get_unique_function_id();
}

search_func_addr_result traced_execution::search_function_addr(uint64_t func_addr) {
    auto it_func = m_func_addr_to_unique_function_id.find(func_addr);
    if(it_func != m_func_addr_to_unique_function_id.end()) {
        auto it_kernel = m_function_id_to_kernel_name.find(it_func->second);
        assert(it_kernel != m_function_id_to_kernel_name.end());
        std::string kernel_name = it_kernel->second;
        return search_func_addr_result(func_addr, it_func->second, kernel_name, true);
    } else {
        return search_func_addr_result();
    }
}

void traced_execution::add_traced_kernel(std::string kernel_name, unsigned int unique_function_id, traced_kernel* kernel, uint64_t func_addr, bool is_captured_from_binary) {
    assert(m_kernels.find(kernel_name) == m_kernels.end());
    m_kernels[kernel_name] = kernel;
    m_kernels[kernel_name]->set_unique_function_id(unique_function_id);
    m_kernels[kernel_name]->set_function_addr(func_addr);
    m_kernels[kernel_name]->set_is_captured_from_binary(is_captured_from_binary);
}


void traced_execution::add_no_binary_kernel(std::string kernel_name, unsigned int unique_function_id, uint64_t func_addr, unsigned int architecture_version, std::map<int,std::string> &captured_instrs , bool is_captured_from_binary) {
    assert(m_kernels.find(kernel_name) == m_kernels.end());
    traced_kernel *kernel = new traced_kernel(kernel_name, architecture_version);
    for(auto inst : captured_instrs) {
        kernel->add_no_binary_instruction(inst.first, ReplaceAll(inst.second, ";", " "));
    }
    m_kernels[kernel_name] = kernel;
    m_kernels[kernel_name]->set_unique_function_id(unique_function_id);
    m_kernels[kernel_name]->set_function_addr(func_addr);
    m_kernels[kernel_name]->set_is_captured_from_binary(is_captured_from_binary);
}

void traced_execution::add_instruction_to_a_kernel(std::string kernel_name, std::vector<std::string> instruction_part1, std::vector<std::string> instruction_part2, int threshold_unique_kernel_checking, std::string full_instruction_str) {
    assert(m_kernels.find(kernel_name) != m_kernels.end());
    m_kernels[kernel_name]->add_instruction(instruction_part1, instruction_part2, threshold_unique_kernel_checking, full_instruction_str);
}

void traced_execution::add_register_usage_to_a_kernel_instruction(std::string kernel_name, unsigned pc, std::string reg_file_name, unsigned num_regs) {
    assert(m_kernels.find(kernel_name) != m_kernels.end());
    m_kernels[kernel_name]->add_register_usage(pc, reg_file_name, num_regs);
}

void traced_execution::add_register_usage_to_a_kernel(std::string kernel_name, std::map<unsigned int, std::map<std::string, unsigned int>> &reg_usage, std::map<unsigned int, std::tuple<std::string, int>> &call_target_by_pc) {
    for(auto inst_entry : reg_usage) {
        for(auto reg_file : inst_entry.second) {
            add_register_usage_to_a_kernel_instruction(kernel_name, inst_entry.first, reg_file.first, reg_file.second);
        }
    }
    for(auto call_target : call_target_by_pc) {
        m_kernels[kernel_name]->add_call_target(call_target.first, std::get<0>(call_target.second), std::get<1>(call_target.second));
    }
}

std::string traced_execution::get_benchmark_name() const {
    return m_benchmark_name;
}

void traced_execution::set_benchmark_name(std::string benchmark_name) {
    m_benchmark_name = benchmark_name;
}

void traced_execution::remove_useless_kernels() {
    std::vector<std::string> kernels_to_remove;
    for(auto it = m_kernels.begin(); it != m_kernels.end(); it++) {
        if(!it->second->is_kernel_in_use()) {
            kernels_to_remove.push_back(it->first);
        }
    }
    for(auto it = kernels_to_remove.begin(); it != kernels_to_remove.end(); it++) {
        m_kernels.erase(*it);
    }
}

void traced_execution::remove_specific_kernel(std::string kernel_name) {
    auto it = m_kernels.find(kernel_name);
    assert(it != m_kernels.end());
    m_kernels.erase(it);
}

bool traced_execution::Serialize(rapidjson::Writer<rapidjson::StringBuffer> * writer) const {
    writer->StartObject();

    writer->String("benchmark_name"); 
    writer->String(m_benchmark_name.c_str());
    
    writer->String("kernels");
    writer->StartArray();
    for(auto it = m_kernels.begin(); it != m_kernels.end(); it++) {
        it->second->Serialize(writer);
    }
    writer->EndArray();

    writer->EndObject();
    
    return true;
}

void traced_execution::generate_kernel_maps() {
    for(auto kernel : m_kernels) {
        unsigned int function_id = kernel.second->get_unique_function_id();
        uint64_t kernel_func_addr = kernel.second->get_function_addr();
        m_function_id_to_kernel_name[function_id] = kernel.first;
        m_func_addr_to_unique_function_id[kernel_func_addr] = function_id;
    }
}

bool traced_execution::Deserialize(const rapidjson::Value &obj)
{

    set_benchmark_name(obj["benchmark_name"].GetString());

    const rapidjson::Value &kernels = obj["kernels"];
    for (rapidjson::SizeType i = 0; i < kernels.Size(); i++)
    {
        traced_kernel *kernel = new traced_kernel();
        if (kernel->Deserialize(kernels[i]))
        {
            m_kernels[kernel->get_kernel_name()] = kernel;
        }
        else
        {
            delete kernel;
        }
    }
    generate_kernel_maps();
    return true;
}

void traced_execution::print(FILE *fp) {
    fprintf(fp, "Benchmark: %s\n", m_benchmark_name.c_str());
    for(auto it = m_kernels.begin(); it != m_kernels.end(); it++) {
        fprintf(fp, "Kernel: %s\n", it->first.c_str());
    }
}