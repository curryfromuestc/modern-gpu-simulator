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


#include <bits/stdc++.h>
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "trace_parser.h"

#include "../../util/traces_enhanced/src/string_utilities.h" // MOD. Improved tracer
#include "../../util/traces_enhanced/src/traced_execution.h" // MOD. Improved tracer
#include "../../util/traces_enhanced/pb_trace/include/threadblock.pb.h" // MOD. Improved tracer
#include "../../util/traces_enhanced/pb_trace/include/gpu_device.pb.h" // MOD. Improved tracer

#include "../gpgpu-sim/src/gpgpu-sim/gpu-sim.h" 

int extractNumberAfterPattern(const std::string& str, const std::string& pattern) {
  size_t pos = str.find(pattern);
  if (pos == std::string::npos) return -1;
  
  // Move to position after the pattern
  pos += pattern.length();
  
  // Find the end of the consecutive digits
  size_t end_pos = pos;
  while (end_pos < str.length() && std::isdigit(str[end_pos])) {
    end_pos++;
  }
  
  // If we found at least one digit
  if (end_pos > pos) {
    // Extract the substring of digits and convert to integer
    std::string number_str = str.substr(pos, end_pos - pos);
    return std::stoi(number_str);
  }
  
  return -1;
}

void parseKernelAndStreamID(const std::string& commandString, int& kernelID, int& streamID, int& gpuDeviceID) {
  kernelID = extractNumberAfterPattern(commandString, "kernel-");
  streamID = extractNumberAfterPattern(commandString, "streamid-");
  gpuDeviceID = extractNumberAfterPattern(commandString, "deviceid-");
  assert(kernelID >= 0 && streamID >= 0 && gpuDeviceID >= 0);
}

bool is_number(const std::string &s) {
  std::string::const_iterator it = s.begin();
  while (it != s.end() && std::isdigit(*it)) ++it;
  return !s.empty() && it == s.end();
}

void split(const std::string &str, std::vector<std::string> &cont,
           char delimi = ' ') {
  std::stringstream ss(str);
  std::string token;
  while (std::getline(ss, token, delimi)) {
    cont.push_back(token);
  }
}

inst_trace_t::inst_trace_t() { 
  is_instruction_traced = true;
  is_constant_addr_already_calculated = false;
}

inst_trace_t::~inst_trace_t() {}

inst_trace_t::inst_trace_t(const inst_trace_t &b) {
  m_pc = b.m_pc;
  m_next_traced_pc = b.m_next_traced_pc;
  m_unique_function_id = b.m_unique_function_id;
  mask = b.mask;
  opcode = b.opcode;
  memadd_info.resize(b.memadd_info.size());
  for(unsigned int i = 0; i < b.memadd_info.size(); ++i) {
    memadd_info[i] = std::make_unique<inst_memadd_info_t>();
    memadd_info[i]->width = b.memadd_info[i]->width;
    memadd_info[i]->u_desc_value = b.memadd_info[i]->u_desc_value;
    std::copy(b.memadd_info[i]->addrs, b.memadd_info[i]->addrs + WARP_SIZE,
              memadd_info[i]->addrs);
  }
  is_constant_addr_already_calculated = b.is_constant_addr_already_calculated;
  is_instruction_traced = b.is_instruction_traced;
  block_idx_x = b.block_idx_x;
  block_idx_y = b.block_idx_y;
  block_idx_z = b.block_idx_z;
}

inst_trace_t::inst_trace_t(address_type pc, unsigned int unique_function_id, bool is_instruction_traced) {
  m_pc = pc;
  m_next_traced_pc = pc;
  m_unique_function_id = unique_function_id;
  this->is_instruction_traced = is_instruction_traced;
  mask = 0;
  opcode = "NOP";
  is_constant_addr_already_calculated = false;
}

bool inst_trace_t::check_opcode_contain(const std::vector<std::string> &opcode,
                                        std::string param) const {
  for (unsigned i = 0; i < opcode.size(); ++i)
    if (opcode[i] == param) return true;

  return false;
}

unsigned inst_trace_t::get_datawidth_from_opcode(
    const std::vector<std::string> &opcode) const {
  if(opcode[0].find("LDSM") != std::string::npos) {
    if (is_number(opcode[opcode.size() - 1])) {
        int num_matrix = std::stoi(opcode[opcode.size() - 1], NULL);
        unsigned int total_size = num_matrix * 32;
        unsigned int num_bytes = total_size / 8;
        return num_bytes;
    }
    return 4;  // default is 4 bytes
  }
  for (unsigned i = 0; i < opcode.size(); ++i) {
    if (is_number(opcode[i])) {
      return (std::stoi(opcode[i], NULL) / 8);
    } else if (opcode[i][0] == 'U' && is_number(opcode[i].substr(1))) {
      // handle the U* case
      unsigned bits;
      sscanf(opcode[i].c_str(), "U%u", &bits);
      return bits / 8;
    }
  }

  return 4;  // default is 4 bytes
}

kernel_trace_t::kernel_trace_t() {
  kernel_name = "Empty";
  shmem_base_addr = 0;
  local_base_addr = 0;
  binary_verion = 0;
  trace_verion = 0;
}

void inst_memadd_info_t::base_stride_decompress(
    unsigned long long base_address, int stride,
    const std::bitset<WARP_SIZE> &mask) {
  bool first_bit1_found = false;
  bool last_bit1_found = false;
  unsigned long long addra = base_address;
  for (int s = 0; s < WARP_SIZE; s++) {
    if (mask.test(s) && !first_bit1_found) {
      first_bit1_found = true;
      addrs[s] = base_address;
    } else if (first_bit1_found && !last_bit1_found) {
      if (mask.test(s)) {
        addra += stride;
        addrs[s] = addra;
      } else
        last_bit1_found = true;
    } else
      addrs[s] = 0;
  }
}

void inst_memadd_info_t::base_delta_decompress(
    unsigned long long base_address, const std::vector<long long> &deltas,
    const std::bitset<WARP_SIZE> &mask) {
  bool first_bit1_found = false;
  long long last_address = 0;
  unsigned delta_index = 0;
  for (int s = 0; s < 32; s++) {
    if (mask.test(s) && !first_bit1_found) {
      addrs[s] = base_address;
      first_bit1_found = true;
      last_address = base_address;
    } else if (mask.test(s) && first_bit1_found) {
      assert(delta_index < deltas.size());
      addrs[s] = last_address + deltas[delta_index++];
      last_address = addrs[s];
    } else
      addrs[s] = 0;
  }
}

void inst_trace_t::parse_memref(unsigned int idx, unsigned int mem_width, const std::bitset<WARP_SIZE> &mask_bits, dynamic_trace::address addr_info) {
  assert(mem_width > 0);
  unsigned int address_mode = 0;
  memadd_info[idx] = std::make_unique<inst_memadd_info_t>();

  // read the memory width from the opcode, as nvbit can report it incorrectly
  std::vector<std::string> opcode_tokens = get_opcode_tokens(opcode);
  memadd_info[idx]->width = get_datawidth_from_opcode(opcode_tokens);

  address_mode = addr_info.compression_format();
  if (address_mode == address_format::list_all) {
    // read addresses one by one from the file
    unsigned int last_addr_parsed = 0;
    for (int s = 0; s < WARP_SIZE; s++) {
      if (mask_bits.test(s)) {
        memadd_info[idx]->addrs[s] = addr_info.addrs(last_addr_parsed);
        last_addr_parsed++;
      }else {
        memadd_info[idx]->addrs[s] = 0;
      }
    }
  } else if (address_mode == address_format::base_stride) {
    // read addresses as base address and stride
    unsigned long long base_address = 0;
    int stride = 0;
    base_address = addr_info.base_address();
    stride = addr_info.stride();
    memadd_info[idx]->base_stride_decompress(base_address, stride, mask_bits);
  } else if (address_mode == address_format::base_delta) {
    unsigned long long base_address = 0;
    std::vector<long long> deltas;
    // read addresses as base address and deltas
    base_address = addr_info.base_address();
    bool skip_first= true;
    unsigned int last_addr_parsed = 0;
    for (int s = 0; s < WARP_SIZE; s++) {
      if(skip_first && mask_bits.test(s)) {
        skip_first = false;
      }else if (mask_bits.test(s)) {
        long long delta = 0;
        delta = addr_info.addrs(last_addr_parsed);
        last_addr_parsed++;
        deltas.push_back(delta);
      }
    }
    memadd_info[idx]->base_delta_decompress(base_address, deltas, mask_bits);
  }
}

bool inst_trace_t::parse_from_pb(dynamic_trace::instruction pb_inst,
                                     unsigned trace_version, gpgpu_sim *gpu, std::string kernel_name, traced_execution &static_trace_info) {
  m_pc = pb_inst.pc();
  mask = pb_inst.active_mask() & pb_inst.predicate_mask();
  m_unique_function_id = pb_inst.function_unique_id();
  unsigned int num_memrefs = pb_inst.addresses_size();
  std::bitset<WARP_SIZE> mask_bits(mask);
  opcode = static_trace_info.get_kernel_by_unique_function_id(m_unique_function_id).get_instruction(m_pc).get_op_code();
  m_next_traced_pc = 0;

  // parse mem info
  unsigned int mem_width = 0;

  memadd_info.resize(num_memrefs);
  for(unsigned int i = 0; i < num_memrefs; ++i) {
    mem_width = pb_inst.addresses(i).data_width();
    parse_memref(i, mem_width, mask_bits, pb_inst.addresses(i));
    memadd_info[i]->u_desc_value = pb_inst.addresses(i).udesc_value();
  }

  return true;
}

trace_parser::trace_parser(const char *kernellist_filepath, bool is_extra_trace_enabled, unsigned int kernel_id_filter_start, unsigned int kernel_id_filter_end) { // MOD. Improved tracer.
  kernellist_filename = kernellist_filepath;
  // MOD. Begin. Improved tracer
  m_is_extra_trace_enabled = is_extra_trace_enabled; 
  if(m_is_extra_trace_enabled) {
    std::size_t found_pos = kernellist_filename.find_last_of("/");
    if(found_pos == std::string::npos) {
      std::cout << "Error. Unable to find the directory of the kernel list file." << std::endl;
      fflush(stdout);
      abort();
    }
    m_extra_trace_info_filename = kernellist_filename.substr(0, found_pos);
    m_threadblocks_main_path = kernellist_filename.substr(0, found_pos);
    m_extra_trace_info_filename += "/extra_info/enhanced_execution_info.json";
    m_threadblocks_main_path += "/threadblocks/";
    m_threadblocks_register_values_main_path = m_threadblocks_main_path + "register_values/";
    m_kernel_id_filter_start = kernel_id_filter_start;
    m_kernel_id_filter_end = kernel_id_filter_end;
    if(m_kernel_id_filter_end == 0) {
      m_kernel_id_filter_end = std::numeric_limits<unsigned int>::max();
    }
  }
  // MOD. End. Improved tracer
}

std::vector<trace_command> trace_parser::parse_commandlist_file() {
  // Open the file named dynamic_trace.pb in binary mode
  std::string directory(kernellist_filename);
  std::ifstream input(directory, std::ios::in | std::ios::binary);
  if (!input) {
      std::cout << "Error: file not found in path: " << directory << "\n";
      fflush(stdout);
      abort();
  }
  if (!dyn_trace.ParseFromIstream(&input)) {
      std::cerr << "Error: Failed to parse dynamic_trace.pb\n";
      fflush(stdout);
      abort();
  }
  const size_t last_slash_idx = directory.rfind('/');
  if (std::string::npos != last_slash_idx) {
    directory = directory.substr(0, last_slash_idx);
  }

  std::string line, filepath;
  std::vector<trace_command> commandlist;
  dynamic_trace::gpu_device &gpu_dev = (*dyn_trace.mutable_gpu_device())[0];
  unsigned int gpu_device_id = gpu_dev.id();
  unsigned int kernel_id = 1;
  for(auto it_com : gpu_dev.streams()) {
    const dynamic_trace::cuda_stream& stream = it_com.second;
    for(int i = 0; i < stream.ordered_cuda_events_size(); ++i) {
      trace_command command;
      bool is_pushed = false;
      command.command_string = stream.ordered_cuda_events(i);
      if(command.command_string.substr(0, 10) == "MemcpyHtoD") {
        command.m_type = command_type::cpu_gpu_mem_copy;
        is_pushed = true;
      } else if(command.command_string.substr(0, 6) == "kernel") {
        command.m_type = command_type::kernel_launch;
        if(kernel_id >= m_kernel_id_filter_start && kernel_id <= m_kernel_id_filter_end) {
          is_pushed = true;
        }
        kernel_id++;
      }
      if(is_pushed) {
        command.command_string += ",deviceid-" + std::to_string(gpu_device_id) + ",streamid-" + std::to_string(stream.id());
        commandlist.push_back(command);
      }
    }
  }
  return commandlist;
}

void trace_parser::parse_memcpy_info(const std::string &memcpy_command,
                                     size_t &address, size_t &count) {
  std::vector<std::string> params;
  split(memcpy_command, params, ',');
  assert(params.size() == 5);
  std::stringstream ss;
  ss.str(params[1]);
  ss >> std::hex >> address;
  ss.clear();
  ss.str(params[2]);
  ss >> std::dec >> count;
}

kernel_trace_t *trace_parser::parse_kernel_info(
    const std::string &kerneltraces_filepath, traced_execution &enahnced_trace_info) {
  kernel_trace_t *kernel_info = new kernel_trace_t;
  int kernelid = -1;
  int streamid = -1;
  int gpuDeviceID = -1;
  parseKernelAndStreamID(kerneltraces_filepath, kernelid, streamid, gpuDeviceID);
  const dynamic_trace::gpu_device &gpu_dev = (*dyn_trace.mutable_gpu_device())[0];
  const dynamic_trace::kernel &ker = gpu_dev.streams().at(streamid).kernels(kernelid-1);

  std::cout << "Processing kernel " << kerneltraces_filepath << std::endl;
  
  kernel_info->kernel_name = ker.name();
  kernel_info->kernel_id = ker.id();
  kernel_info->grid_dim_x = ker.grid_dim().x();
  kernel_info->grid_dim_y = ker.grid_dim().y();
  kernel_info->grid_dim_z = ker.grid_dim().z();
  kernel_info->tb_dim_x = ker.block_dim().x();
  kernel_info->tb_dim_y = ker.block_dim().y();
  kernel_info->tb_dim_z = ker.block_dim().z();
  kernel_info->shmem = ker.size_shared_memory();
  kernel_info->nregs = ker.number_of_registers();
  kernel_info->gpu_device_id = gpuDeviceID;
  kernel_info->cuda_stream_id = streamid;
  kernel_info->binary_verion = dyn_trace.binary_version();
  kernel_info->trace_verion = dyn_trace.accelsim_version();
  kernel_info->nvbit_verion = dyn_trace.nvbit_version();
  kernel_info->shmem_base_addr = ker.shared_memory_base_address();
  kernel_info->local_base_addr = ker.local_memory_base_address();
  kernel_info->func_unique_id = ker.function_unique_id();
  kernel_info->is_cap_from_binary = enahnced_trace_info.get_kernel_by_unique_function_id(kernel_info->func_unique_id).is_captured_from_binary();
  kernel_info->next_tb_to_parse_x = 0;
  kernel_info->next_tb_to_parse_y = 0;
  kernel_info->next_tb_to_parse_z = 0;

  return kernel_info;
}

void trace_parser::kernel_finalizer(kernel_trace_t *trace_info) {
  assert(trace_info);
  delete trace_info;
}

void trace_parser::get_next_threadblock_traces(
    std::vector<std::map<address_type, traced_instructions_by_pc> *> threadblock_traces,
    std::vector<std::vector<address_type> *> threadblock_traced_pcs, unsigned int gpu_device_id, unsigned int streamid, unsigned int kernelid,
    unsigned trace_version, unsigned int block_id_x, unsigned int block_id_y, unsigned int block_id_z, gpgpu_sim *gpu, std::string kernel_name, traced_execution &static_trace_info) {
  for (unsigned i = 0; i < threadblock_traces.size(); ++i) {
    threadblock_traces[i]->clear();
  }

  unsigned warp_id = 0;
  unsigned insts_num = 0;
  unsigned inst_count = 0;
  address_type previous_traced_pc = 0;
  dynamic_trace::threadblock tb_cur;
  
  std::string device_folder = m_threadblocks_main_path + "device_" + std::to_string(gpu_device_id);
  std::string stream_folder = device_folder + "/stream_" + std::to_string(streamid);
  std::string kernel_folder = stream_folder + "/kernel_" + std::to_string(kernelid);
  
  std::string tb_string_id = "d_" + std::to_string(gpu_device_id) + 
                            "_s_" + std::to_string(streamid) + 
                            "_k_" + std::to_string(kernelid) + "_" + 
                            std::to_string(block_id_x) + "," + std::to_string(block_id_y) + "," + std::to_string(block_id_z);
  
  std::string tb_path = kernel_folder + "/" + tb_string_id + ".pb";
  
  std::ifstream input(tb_path, std::ios::in | std::ios::binary);
  if (!input) {
      std::cout << "Error: file not found in path: " << tb_path << "\n";
      fflush(stdout);
      abort();
  }
  if (!tb_cur.ParseFromIstream(&input)) {
      std::cout << "Error: Failed to parse threadblock file\n";
      fflush(stdout);
      abort();
  }
  input.close();
  unsigned int size_traced_instructions_num_used_vector = gpu == nullptr ? 32 : 1;
  std::cout << "thread block = " << tb_cur.block_id().x() << "," << tb_cur.block_id().y() << "," << tb_cur.block_id().z() << std::endl;
  for(auto warp : tb_cur.warps()) {
    previous_traced_pc = 0;
    inst_count = 0;
    insts_num = warp.second.instructions_size();
    warp_id = warp.first;
    threadblock_traced_pcs[warp_id]->resize(insts_num);
    for(auto pb_inst : warp.second.instructions()) {
      inst_trace_t current_inst;
      current_inst.parse_from_pb(pb_inst, trace_version, gpu, kernel_name, static_trace_info);
      current_inst.block_idx_x = block_id_x;
      current_inst.block_idx_y = block_id_y;
      current_inst.block_idx_z = block_id_z;
      std::map<address_type, traced_instructions_by_pc> *map_inst_of_warp =
          threadblock_traces[warp_id];
      auto it_find_pc = map_inst_of_warp->find(current_inst.m_pc);
      if (it_find_pc == map_inst_of_warp->end()) {
        map_inst_of_warp->insert(std::pair<address_type, traced_instructions_by_pc>(
            current_inst.m_pc, traced_instructions_by_pc(current_inst.m_pc, size_traced_instructions_num_used_vector)));
        it_find_pc = map_inst_of_warp->find(current_inst.m_pc);
      }
      it_find_pc->second.instructions.push_back(current_inst);
      threadblock_traced_pcs[warp_id]->at(inst_count) = current_inst.m_pc;
    
      auto it_find_prev_pc = map_inst_of_warp->find(previous_traced_pc);
      if(inst_count > 0) {
        assert(it_find_prev_pc != map_inst_of_warp->end());
        it_find_prev_pc->second.instructions[it_find_prev_pc->second.num_traced_instructions - 1]
            .m_next_traced_pc = current_inst.m_pc;
      }
      previous_traced_pc = current_inst.m_pc;

      it_find_pc->second.num_traced_instructions++;
      inst_count++;
    }
  }
  tb_cur.Clear();
}
