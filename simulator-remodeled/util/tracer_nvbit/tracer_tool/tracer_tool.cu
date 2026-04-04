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


#include <algorithm>
#include <assert.h>
#include <bitset>
#include <inttypes.h>
#include <iostream>
#include <iterator>
#include <map>
#include <sstream>
#include <stdint.h>
#include <stdio.h>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_set>
#include <vector>
#include <regex>

// MOD. Begin. Enhanced Tracer
#include <execinfo.h> /* backtrace, backtrace_symbols_fd */
#include <unistd.h> /* STDOUT_FILENO */
#include <filesystem>
#include <fstream>

#include "../../traces_enhanced/src/traced_execution.h"
#include "../../traces_enhanced/src/string_utilities.h"
#include "../../../gpu-simulator/ISA_Def/trace_opcode.h"
// MOD. End. Enhanced Tracer

#include "../../traces_enhanced/pb_trace/include/trace.pb.h"
#include "../../traces_enhanced/pb_trace/include/gpu_device.pb.h"
#include "../../traces_enhanced/pb_trace/include/cuda_stream.pb.h"
#include "../../traces_enhanced/pb_trace/include/kernel.pb.h"
#include "../../traces_enhanced/pb_trace/include/threadblock.pb.h"
#include "../../traces_enhanced/pb_trace/include/warp.pb.h"
#include "../../traces_enhanced/pb_trace/include/instruction.pb.h"
#include "../../traces_enhanced/pb_trace/include/address.pb.h"
#include "../../traces_enhanced/pb_trace/include/dim3d.pb.h"

/* every tool needs to include this once */
#include "nvbit_tool.h"

/* nvbit interface file */
#include "nvbit.h"

/* for channel */
#include "utils/channel.hpp"

/* contains definition of the inst_trace_t structure */
#include "common.h"

#define TRACER_VERSION 4

/* Channel used to communicate from GPU to CPU receiving thread */
#define CHANNEL_SIZE (1l << 20)

int num_devices = 0;

enum class RecvThreadState {
  WORKING,
  STOP,
  FINISHED,
};

struct CTXstate {
  /* context id */
  int id;

  /* Channel used to communicate from GPU to CPU receiving thread */
  ChannelDev* channel_dev;
  ChannelHost channel_host;

  // After initialization, set it to WORKING to make recv thread get data,
  // parent thread sets it to STOP to make recv thread stop working.
  // recv thread sets it to FINISHED when it cleans up.
  // parent thread should wait until the state becomes FINISHED to clean up.
  volatile RecvThreadState recv_thread_done = RecvThreadState::STOP;
};

/* lock */
pthread_mutex_t mutex;

/* map to store context state */
std::unordered_map<CUcontext, CTXstate*> ctx_state_map;

/* skip flag used to avoid re-entry on the nvbit_callback when issuing
 * flush_channel kernel call */
bool skip_callback_flag = false;
bool is_first_init_context_call = true;
std::map<CUcontext, bool> recv_thread_receiving;
bool *stop_report;

/* global control variables for this tool */
uint32_t instr_begin_interval = 0;
uint32_t instr_end_interval = UINT32_MAX;
int verbose = 0;
int intermediate_extra_files_persistance = 0;
int gather_registers = 0;
int enable_compress = 1;
int print_core_id = 0;
int exclude_pred_off = 1;
int active_from_start = 1;
/* used to select region of interest when active from start is 0 */
bool active_region = true;

/* Should we terminate the program once we are done tracing? */
int terminate_after_limit_number_of_kernels_reached = 0;
int user_defined_folders = 0;

/* opcode to id map and reverse map  */
std::map<std::string, int> opcode_to_id_map;
std::map<int, std::string> id_to_opcode_map;

std::string cwd = getcwd(NULL,0);
std::string traces_location = cwd + "/traces/";
std::string stats_location = cwd + "/traces/stats.csv";

/* kernel instruction counter, updated by the GPU */
int dynamic_kernel_limit_start =
    0;                                 // 0 means start from the begging kernel
int dynamic_kernel_limit_end = 0; // 0 means no limit

enum address_format { list_all = 0, base_stride = 1, base_delta = 2 };

int binary_version;
std::vector<int> kernel_id;
std::vector<int> current_stream_id;

// MOD. Begin. Enhanced Tracer
int next_candidate_unique_function_id = 0;
std::set<int> opcodes_id_ldgsts;
int threshold_unique_kernel_checking;
std::string variant_delimiter_str = "___";

struct traced_operand_instrument {
  traced_operand_instrument() = default; // Add this default constructor
  traced_operand_instrument(TRACED_REG_TYPE reg_type, int num_regs, int first_reg_id) {
    this->reg_type = reg_type;
    this->num_regs = num_regs;
    this->first_reg_id = first_reg_id;
  }
  TRACED_REG_TYPE reg_type;
  int num_regs;
  int first_reg_id;
};

struct traced_kernel_id {
  traced_kernel_id(std::string kernel_name, unsigned int variant_id, unsigned int candidate_unique_function_id,std::map<int, std::string> key_instructions_by_pc, std::map<int, std::string> call_or_ret_by_pc, uint64_t func_addr) {
    this->original_kernel_name = kernel_name;
    this->variant_id = variant_id;
    this->unique_function_id = candidate_unique_function_id;
    this->key_instructions_by_pc = key_instructions_by_pc;
    this->call_or_ret_by_pc = call_or_ret_by_pc;
    this->func_addr = func_addr;
    sass_has_been_parsed = false;
    rfu_has_been_parsed = false;
  }
  std::string original_kernel_name;
  unsigned int unique_function_id; // ID to identify statically the code of the kernel. It should be repetead over all the kernels with the same name and variant
  unsigned int variant_id;
  bool sass_has_been_parsed;
  bool rfu_has_been_parsed;
  std::map<int, std::string> key_instructions_by_pc;
  std::map<int, std::string> call_or_ret_by_pc;
  uint64_t func_addr;
};

// This structures are used to uniquely identify kernels. Some of them have the same name, even that they have different code
std::map<std::string,std::vector<traced_kernel_id>> all_kernels_key_instructions_by_pc; // Stores all the kernels informations ofthe execution
std::map<int, std::string> current_kernel_key_instructions_by_pc; // Only stores the information of the current analyzed kernel.
std::map<int, std::string> current_kernel_call_or_ret_by_pc; // Only stores the information of the current analyzed kernel.
std::map<CUfunction, std::tuple<std::string,int>> map_function_to_kernel_name_and_variant_id; // This is used to store the kernel name and variant id of the kernels that are being instrumented
std::map<std::string, int> map_function_name_to_unique_function_id_with_variant; // This is used to store the unique function identifier of the kernels that are being instrumented
std::map<std::string, int> map_function_name_to_unique_function_id_without_variant;
std::map<uint64_t, std::string> map_func_addr_to_kernel_name;
std::map<uint64_t, std::map<int,std::string>> map_func_addr_to_pc_to_sass_instr; // Used for kernels that does not appear in the binary
std::map<std::string, uint64_t> map_kernel_name_to_func_addr;
// First element of tuple is kernel name and second the variant id

traced_execution *m_enhanced_traced_execution;
std::string traces_path = "traces";
std::string extrainfo_path = traces_path + "/extra_info";
std::string cubin_path = extrainfo_path + "/cubin";
std::string sass_path = extrainfo_path + "/sass";
std::string register_usage_path = extrainfo_path + "/register_usage";
std::string threadblock_trace_path = traces_path + "/threadblocks";
std::string threadblock_register_values_path = traces_path + "/threadblocks/register_values";

std::string get_program_path() {
    size_t size;
    enum Constexpr { MAX_SIZE = 1024 };
    void *array[MAX_SIZE];
    size = backtrace(array, MAX_SIZE);
    char** calls;
    calls = backtrace_symbols(array, size);
    std::string program_caller = calls[size-1];
    std::string program_path = program_caller.substr(0, program_caller.find_last_of("("));
    return program_path;
}

dynamic_trace::Trace dyn_trace;
// Key: d_{device}_s_{stream}_k_{kernel}_{cta_id_x},{cta_id_y},{cta_id_z}
std::unordered_map<std::string, dynamic_trace::threadblock> threadblocks;
//TB key, warp id, int remaining injects
std::unordered_map<std::string, std::map<unsigned int, unsigned int>> remaining_injects_to_current_instruction;
const std::unordered_map<std::string, OpcodeChar> *OpcodeMap = nullptr;

dynamic_trace::threadblock& get_threadblock(std::string tb_key) {
  if(threadblocks.find(tb_key) == threadblocks.end()) {
    threadblocks[tb_key] = dynamic_trace::threadblock();
  }
  return threadblocks[tb_key];
}

// Get by reference the bool of is_last_warp_inst_ldgsts_half
unsigned int& get_remaining_injects_to_current_instruction(std::string tb_key, unsigned int warp_id) {
  if(remaining_injects_to_current_instruction.find(tb_key) == remaining_injects_to_current_instruction.end()) {
    remaining_injects_to_current_instruction[tb_key] = std::map<unsigned int, unsigned int>();
  }
  if(remaining_injects_to_current_instruction[tb_key].find(warp_id) == remaining_injects_to_current_instruction[tb_key].end()) {
    remaining_injects_to_current_instruction[tb_key][warp_id] = 0;
  }
  return remaining_injects_to_current_instruction[tb_key][warp_id];
}

void create_folder(const char * folder_path) {
  if (mkdir(folder_path, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == -1) {
    if (errno == EEXIST) {
      // alredy exists
    } else {
      // something else
      std::cout << "cannot create folder error:" << strerror(errno)
                << std::endl;
      return;
    }
  }
}

void remove_folder(const char * folder_path) {
  std::filesystem::remove_all(folder_path);
}

std::string read_stripped_line(std::ifstream &ifs) {
  std::string line;
  std::getline(ifs, line);
  std::string clear_line = ReplaceAll(line, "/*", " ");
  clear_line = ReplaceAll(clear_line, "*/", " ");
  clear_line = ReplaceAll(clear_line, ",", " ");
  clear_line = ReplaceAll(clear_line, ";", " ");
  return strip_string(clear_line);
}

std::string getEnclosedSubstring(std::string str) {
  size_t start_pos = str.find('(');
  if(start_pos == std::string::npos)
      return "";
  size_t end_pos = str.find(')', start_pos);
  if(end_pos == std::string::npos)
      return "";
  return str.substr(start_pos + 1, end_pos - start_pos - 1);
}

// Regex to remove dependency requirements, e.g. " &req={1}"
std::regex reqRegex("\\s*&req=\\{[^}]+\\}");
// Regex to remove write-slot annotations, e.g. " &wr=0x4"
std::regex wrRegex("\\s*&wr=0x[0-9A-Fa-f]+");
// Regex to remove write-slot annotations, e.g. " &wr=0x4"
std::regex rdRegex("\\s*&rd=0x[0-9A-Fa-f]+");
// Regex to remove transaction/synchronization annotations, e.g. " ?trans1;" or " ?WAIT4_END_GROUP;"
std::regex transRegex("\\s*\\?[A-Za-z0-9_]+");

std::string replaceInstructionNewExtraInformation(std::string original_sass_string) {
  std::string transformed = original_sass_string;
  transformed = std::regex_replace(transformed, reqRegex, "");
  transformed = std::regex_replace(transformed, wrRegex, "");
  transformed = std::regex_replace(transformed, rdRegex, "");
  transformed = std::regex_replace(transformed, transRegex, "");
  return transformed;
}

void print_map(const std::map<int, std::string> &map_to_print) {
  std::cout << "Number of elements in the map: " << map_to_print.size() << std::endl;
  for(auto it = map_to_print.begin(); it != map_to_print.end(); ++it) {
    std::cout << std::hex << it->first << std::dec << " " << it->second << std::endl;
  }
  std::cout << std::endl;
}

void print_all_traced_kernels() {
  for(auto it = all_kernels_key_instructions_by_pc.begin(); it != all_kernels_key_instructions_by_pc.end(); ++it) {
    for(unsigned int i = 0; i < it->second.size(); i++) {
      std::cout << "Kernel " << it->second[i].original_kernel_name << " variant " << i << " has been traced. SASS parsed? " <<  it->second[i].sass_has_been_parsed
        << " RFU parsed? " << it->second[i].rfu_has_been_parsed << std::endl;
    }
  }
}

void erase_not_tracked_call_or_rets(std::map<int, std::string> &kernel_call_or_ret_by_pc, std::set<unsigned int> &call_or_ret_pcs_not_to_consider) {
  for(auto it = call_or_ret_pcs_not_to_consider.begin(); it != call_or_ret_pcs_not_to_consider.end(); ++it) {
    auto it_candidate = kernel_call_or_ret_by_pc.find(*it);
    if(it_candidate != kernel_call_or_ret_by_pc.end()) {
      kernel_call_or_ret_by_pc.erase(it_candidate);
    }
  }
}

bool are_two_kernels_equal(traced_kernel_id &kernel_to_be_compared, std::map<int, std::string> &candidate_key_instructions_by_pc, std::map<int, std::string> &candidate_call_or_ret_by_pc, bool compare_call_or_ret) {
  bool are_same_key_instructions = (kernel_to_be_compared.key_instructions_by_pc == candidate_key_instructions_by_pc);
  bool are_same_call_or_ret = true;
  if(compare_call_or_ret) {
    are_same_call_or_ret = (kernel_to_be_compared.call_or_ret_by_pc == candidate_call_or_ret_by_pc);
  }
  return are_same_key_instructions && are_same_call_or_ret;
}

bool has_the_kernel_been_traced(std::string kernel_name, std::map<int, std::string> candidate_parsed_kernel, std::map<int, std::string> &candidate_call_or_ret_by_pc, unsigned int &variant_id, bool is_rfu_parsing, bool compare_call_or_ret, std::set<unsigned int> &call_or_ret_pcs_not_to_consider) {
  bool has_been_traced = false;
  auto it_already_traced = all_kernels_key_instructions_by_pc.find(kernel_name);
  if(it_already_traced != all_kernels_key_instructions_by_pc.end()) {
    for(unsigned int i = 0; !has_been_traced && (i < all_kernels_key_instructions_by_pc[kernel_name].size()); i++) {
      bool is_alread_parsed = is_rfu_parsing ? it_already_traced->second[i].rfu_has_been_parsed : it_already_traced->second[i].sass_has_been_parsed;
      if(compare_call_or_ret) {
        erase_not_tracked_call_or_rets(it_already_traced->second[i].call_or_ret_by_pc, call_or_ret_pcs_not_to_consider);
      }
      if( are_two_kernels_equal(it_already_traced->second[i], candidate_parsed_kernel, candidate_call_or_ret_by_pc, compare_call_or_ret ) && !is_alread_parsed) {
        has_been_traced = true;
        variant_id = it_already_traced->second[i].variant_id;
      }
    }
  }
  return has_been_traced;
}

void parse_sass(int binary_version, const std::filesystem::directory_entry &entry) {
  std::string absolute_sass_path = cwd + "/" + sass_path;
  std::string sass_file = absolute_sass_path + "/" + entry.path().stem().string() + ".sass";
  std::ifstream ifs_sass; // input file stream
  ifs_sass.open(sass_file, std::ios::in);
  if (!ifs_sass)
  {
    std::cerr << "Error opening file " << sass_file << std::endl;
    fflush(stderr);
    abort();
  }
  
  bool is_starting_reading_kernel = false;
  std::string kernel_name = "";
  traced_kernel *current_traced_kernel = nullptr;
  std::map<int, std::string> full_inst_call_or_ret_by_pc;
  std::set<unsigned int> call_or_ret_pcs_not_to_consider;

  while (!ifs_sass.eof())
  {
    std::string stripped_line = read_stripped_line(ifs_sass);

    if (!is_starting_reading_kernel)
    {
      if ((stripped_line.find("Function") != std::string::npos))
      {
        std::vector<std::string> aux_splitted = split_string(stripped_line, ' ');
        assert(aux_splitted.size() == 3);
        kernel_name = aux_splitted[2];
        current_traced_kernel = new traced_kernel(kernel_name, static_cast<unsigned>(binary_version));
        is_starting_reading_kernel = true;
      }
    }
    else
    {
      if (stripped_line.find("..........") != std::string::npos)
      {
        is_starting_reading_kernel = false;

        // Check if the kernel was captured during the execution
        unsigned int variant_id = 0;
        bool has_been_traced = has_the_kernel_been_traced(kernel_name, current_traced_kernel->get_key_instructions_pcs(), full_inst_call_or_ret_by_pc, variant_id, false, false, call_or_ret_pcs_not_to_consider);

        if(has_been_traced) {
          std::string final_kernel_name = kernel_name + variant_delimiter_str + std::to_string(variant_id);
          auto it_search_funct = map_kernel_name_to_func_addr.find(kernel_name);
          assert(it_search_funct != map_kernel_name_to_func_addr.end());
          all_kernels_key_instructions_by_pc[kernel_name][variant_id].sass_has_been_parsed = true;
          current_traced_kernel->set_kernel_name(final_kernel_name);
          m_enhanced_traced_execution->add_traced_kernel(final_kernel_name, all_kernels_key_instructions_by_pc[kernel_name][variant_id].unique_function_id, current_traced_kernel, it_search_funct->second, true);
          current_traced_kernel = nullptr;
        }else {
          delete current_traced_kernel;
        }
      }
      else if (stripped_line.find("headerflags") == std::string::npos)
      {
        stripped_line = replaceInstructionNewExtraInformation(stripped_line);
        std::vector<std::string> aux_list = split_string(stripped_line, ' ');
        std::vector<std::string> aux_list2;
        if (binary_version >= 70)
        {
          assert(!ifs_sass.eof());
          aux_list2 = split_string(read_stripped_line(ifs_sass), ' ');
        }
        current_traced_kernel->add_instruction(aux_list, aux_list2, threshold_unique_kernel_checking, stripped_line);
      }
    }
  }
  ifs_sass.close();
}

std::string replaceEnclosedSubstring(std::string str, const std::string &replacement)
{
  size_t start_pos = str.find('`');
  if (start_pos == std::string::npos)
    return str;
  size_t end_pos = str.find(')', start_pos);
  if (end_pos == std::string::npos)
    return str;
  str.replace(start_pos, end_pos - start_pos + 1, replacement);
  return str;
}

void parse_rfu_instruction_info(std::vector<std::string> splitted_text, std::vector<std::string> reg_order, std::string full_instruction_str, std::map<int, std::string> &key_instructions_by_pc,
  std::map<unsigned int, std::map<std::string, unsigned int>> &reg_usage_by_pc, std::map<unsigned int, std::tuple<std::string, int>> &call_target_by_pc, std::map<int, std::string> &full_inst_call_or_ret_by_pc, std::set<unsigned int> &call_or_ret_pcs_not_to_consider)
{
  unsigned pc = std::stoul(splitted_text[0], nullptr, 16);
  bool is_all_reg_use_gathered = false;
  int num_already_visited_reg_files = 0;
  std::string instruction_str = full_instruction_str;
  instruction_str = replaceInstructionNewExtraInformation(instruction_str);
  std::string sass_string = create_sass_instr(instruction_str, true, "//");
  if(track_this_instruction(reg_usage_by_pc.size(), threshold_unique_kernel_checking, instruction_str)){
    key_instructions_by_pc[pc] = sass_string;
  }

  if((sass_string.find("CALL")!= std::string::npos) || (sass_string.find("RET")!= std::string::npos)) {
    std::string funct_name = getEnclosedSubstring(sass_string);
    auto it_search_funct = map_kernel_name_to_func_addr.find(funct_name);
    bool is_rel_type = (sass_string.find("REL") != std::string::npos);
    if(!is_rel_type && (it_search_funct != map_kernel_name_to_func_addr.end())) {
      uint64_t func_addr = it_search_funct->second;
      std::stringstream ss;
      ss << "0x" << std::hex << func_addr;
      std::string new_sass_string = replaceEnclosedSubstring(sass_string, ss.str());
      full_inst_call_or_ret_by_pc[pc] = new_sass_string;
    }else if(!is_rel_type && ((sass_string.find("printf") != std::string::npos) || (sass_string.find("assert") != std::string::npos) ) ) {
      call_or_ret_pcs_not_to_consider.insert(pc);
    }else if(!is_rel_type){
      full_inst_call_or_ret_by_pc[pc] = sass_string;
    }
  }

  if(sass_string.find("CALL") != std::string::npos) {
    std::size_t first_char =  sass_string.find("(");
    std::size_t last_char =  sass_string.find(")");
    if((first_char != std::string::npos) && (last_char != std::string::npos)) {
      std::string call_target = sass_string.substr(first_char+1, last_char - first_char - 1);
      auto search_unique_function_id = map_function_name_to_unique_function_id_without_variant.find(call_target);
      if(search_unique_function_id != map_function_name_to_unique_function_id_without_variant.end()) {
        int candidate_unique_function_id = search_unique_function_id->second;
        call_target_by_pc[pc]= std::make_tuple(call_target, candidate_unique_function_id);
      }
    }
  }

  for (unsigned int i = (splitted_text.size() - 1); (i > 0) && !is_all_reg_use_gathered; --i)
  {
    if (splitted_text[i] == "//")
    {
      is_all_reg_use_gathered = true;
    }
    else if (splitted_text[i] == "|")
    {
      num_already_visited_reg_files++;
    }
    else
    {
      unsigned num_regs = std::stoul(splitted_text[i]);
      std::string reg_file_name = reg_order[reg_order.size() - num_already_visited_reg_files];
      reg_usage_by_pc[pc][reg_file_name] = num_regs;
    }
  }
}

void parse_rfu(const std::filesystem::directory_entry &entry) {
  std::string absolute_rfu_path = cwd + "/" + register_usage_path;
  std::string rfu_file = absolute_rfu_path + "/" + entry.path().stem().string() + ".rfu";
  std::ifstream ifs_rfu; // input file stream
  ifs_rfu.open(rfu_file, std::ios::in);
  if (!ifs_rfu)
  {
    std::cerr << "Error opening file " << rfu_file << std::endl;
    fflush(stderr);
    abort();
  }
  
  bool is_starting_reading_kernel = false;
  bool is_code_region_started = false;
  std::string kernel_name = "";
  std::vector<std::string> reg_order;
  std::map<int, std::string> key_instructions_by_pc;
  std::map<unsigned int, std::map<std::string, unsigned int>> reg_usage_by_pc;
  std::map<unsigned int, std::tuple<std::string, int>> call_target_by_pc;
  std::map<int, std::string> full_call_or_ret_inst_by_pc;
  std::set<unsigned int> call_or_ret_pcs_not_to_consider;
  key_instructions_by_pc.clear();
  reg_usage_by_pc.clear();
  call_target_by_pc.clear();

  while (!ifs_rfu.eof())
  {
    std::string stripped_line = read_stripped_line(ifs_rfu);
    
    if (!is_starting_reading_kernel)
    {
      if ((stripped_line.find("GPR") != std::string::npos))
      {
        stripped_line = ReplaceAll(stripped_line, "|", "");
        std::vector<std::string> aux_splitted = split_string(stripped_line, ' ');
        aux_splitted.erase(aux_splitted.begin());
        reg_order = aux_splitted;
        is_starting_reading_kernel = true;
      }
    }
    else
    {
      if (stripped_line.find("Legend:") != std::string::npos)
      {
        is_starting_reading_kernel = false;
        is_code_region_started = false;
        unsigned int variant_id = 0;
        bool has_been_traced = has_the_kernel_been_traced(kernel_name, key_instructions_by_pc, full_call_or_ret_inst_by_pc, variant_id, true, true, call_or_ret_pcs_not_to_consider);
        if(has_been_traced) {
          std::string final_kernel_name = kernel_name + variant_delimiter_str + std::to_string(variant_id);
          all_kernels_key_instructions_by_pc[kernel_name][variant_id].rfu_has_been_parsed = true;
          m_enhanced_traced_execution->add_register_usage_to_a_kernel(final_kernel_name, reg_usage_by_pc, call_target_by_pc);
        }
        key_instructions_by_pc.clear();
        reg_usage_by_pc.clear();
        call_target_by_pc.clear();
        full_call_or_ret_inst_by_pc.clear();
        call_or_ret_pcs_not_to_consider.clear();
      }else {
        std::vector<std::string> splitted_text = split_string(stripped_line, ' ');
        if(!splitted_text.empty()) {
          if(is_code_region_started && (stripped_line.find(":") == std::string::npos) &&
            (stripped_line.find(".weak") == std::string::npos) && (stripped_line.find(".type") == std::string::npos)
            && (stripped_line.find(".size") == std::string::npos)) {
            parse_rfu_instruction_info(splitted_text, reg_order, stripped_line, key_instructions_by_pc, reg_usage_by_pc, call_target_by_pc, full_call_or_ret_inst_by_pc, call_or_ret_pcs_not_to_consider);
          }else if( (splitted_text[0].find("0000") != std::string::npos) && (stripped_line.find("//") != std::string::npos) && (stripped_line.find(":") == std::string::npos) &&
            (stripped_line.find(".weak") == std::string::npos) && (stripped_line.find(".type") == std::string::npos)
            && (stripped_line.find(".size") == std::string::npos)) {
            is_code_region_started = true;
            parse_rfu_instruction_info(splitted_text, reg_order, stripped_line, key_instructions_by_pc, reg_usage_by_pc, call_target_by_pc, full_call_or_ret_inst_by_pc, call_or_ret_pcs_not_to_consider);
          }else if(splitted_text[0].find(".text") != std::string::npos) {
            kernel_name = splitted_text[0].substr(6);
            kernel_name.pop_back(); // Remove the last  : char
          }
        }
      }
    }
  }
  ifs_rfu.close();
}
// MOD. End. Enhanced Tracer

void nvbit_at_init() {
  setenv("CUDA_MANAGED_FORCE_DEVICE_ALLOC", "1", 1);
  GET_VAR_INT(
      instr_begin_interval, "INSTR_BEGIN", 0,
      "Beginning of the instruction interval where to apply instrumentation");
  GET_VAR_INT(instr_end_interval, "INSTR_END", UINT32_MAX,
              "End of the instruction interval where to apply instrumentation");
  GET_VAR_INT(exclude_pred_off, "EXCLUDE_PRED_OFF", 1,
              "Exclude predicated off instruction from count");
  GET_VAR_INT(dynamic_kernel_limit_end, "DYNAMIC_KERNEL_LIMIT_END", 0,
              "Limit of the number kernel to be printed, 0 means no limit");
  GET_VAR_INT(dynamic_kernel_limit_start, "DYNAMIC_KERNEL_LIMIT_START", 0,
              "start to report kernel from this kernel id, 0 means starts from "
              "the beginning, i.e. first kernel");
  GET_VAR_INT(
         active_from_start, "ACTIVE_FROM_START", 1,
         "Start instruction tracing from start or wait for cuProfilerStart "
         "and cuProfilerStop. If set to 0, DYNAMIC_KERNEL_LIMIT options have no effect");
  GET_VAR_INT(verbose, "TOOL_VERBOSE", 0, "Enable verbosity inside the tool");
  GET_VAR_INT(intermediate_extra_files_persistance, "INTERMEDIATE_EXTRA_FILES_PERSISTANCE", 0, "Enable to not delete intermediate files from the extra trace generation");
  GET_VAR_INT(print_core_id, "TOOL_TRACE_CORE", 0,
              "write the core id in the traces");
  GET_VAR_INT(terminate_after_limit_number_of_kernels_reached, "TERMINATE_UPON_LIMIT", 0, 
              "Stop the process once the current kernel > DYNAMIC_KERNEL_LIMIT_END");
  GET_VAR_INT(user_defined_folders, "USER_DEFINED_FOLDERS", 0, "Uses the user defined "
              "folder TRACES_FOLDER path environment");
  GET_VAR_INT(threshold_unique_kernel_checking, "THRESHOLD_UNIQUE_KERNEL_CHECKING", 10,
              "Number of instructions used to check if kernels with the same name are different");
  GET_VAR_INT(gather_registers, "GATHER_REGISTERS", 0, "Enable gathering of GPU register values. Not available in this version.");
  std::string pad(100, '-');
  printf("%s\n", pad.c_str());

  if (active_from_start == 0) {
    active_region = false;
  }

  /* set mutex as recursive */
  pthread_mutexattr_t attr;
  pthread_mutexattr_init(&attr);
  pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);
  pthread_mutex_init(&mutex, &attr);
}

// Key is unique function ID. The next key is PC and finally the Value is the datawidh of the opcode ID of the instruction.
std::map<int,std::map<int, int>>  pc_to_opcode;

/* Set used to avoid re-instrumenting the same functions multiple times */
std::unordered_set<CUfunction> already_instrumented;

/* instrument each memory instruction adding a call to the above instrumentation
 * function */
void instrument_function_if_needed(CUcontext ctx, CUfunction func, int device_id) {
  assert(ctx_state_map.find(ctx) != ctx_state_map.end());
  CTXstate* ctx_state = ctx_state_map[ctx];

  std::vector<CUfunction> related_functions =
      nvbit_get_related_functions(ctx, func);

  /* add kernel itself to the related function vector */
  related_functions.push_back(func);

  /* iterate on function */
  for (auto f : related_functions) {
    current_kernel_key_instructions_by_pc.clear();
    current_kernel_call_or_ret_by_pc.clear();
    /* "recording" function was instrumented, if set insertion failed
     * we have already encountered this function */
    if (!already_instrumented.insert(f).second) {
      continue;
    }

    std::string current_kernel_name(nvbit_get_func_name(ctx, f, true));
    uint64_t addr_funct = (uint64_t)nvbit_get_func_addr(ctx, f);
    
    next_candidate_unique_function_id++;

    const std::vector<Instr *> &instrs = nvbit_get_instrs(ctx, f);
    if (verbose) {
      printf("Inspecting function %s at address 0x%lx\n", nvbit_get_func_name(ctx, f), addr_funct);
    }

    uint32_t cnt = 0;
    /* iterate on all the static instructions in the function */
    for (auto instr : instrs) {

      if (cnt < instr_begin_interval || cnt >= instr_end_interval) {
        cnt++;
        continue;
      }
      if (verbose) {
        instr->printDecoded();
      }

      if (opcode_to_id_map.find(instr->getOpcode()) == opcode_to_id_map.end()) {
        int opcode_id = opcode_to_id_map.size();
        opcode_to_id_map[instr->getOpcode()] = opcode_id;
        id_to_opcode_map[opcode_id] = instr->getOpcode();
      }

      int opcode_id = opcode_to_id_map[instr->getOpcode()];
      int vpc = (int)instr->getOffset();
      pc_to_opcode[next_candidate_unique_function_id][vpc] = opcode_id;

      std::string inst_str = ReplaceAll(instr->getSass(), ",", " ");
      inst_str = replaceInstructionNewExtraInformation(inst_str);
      inst_str = ReplaceAll(inst_str, ";", "");
      inst_str = strip_string(inst_str);

      if(std::string(instr->getOpcode()).find("LDGSTS") != std::string::npos) {
        opcodes_id_ldgsts.insert(opcode_id);
      }

      bool is_call_or_ret = (std::string(instr->getOpcode()).find("CALL") != std::string::npos) || (std::string(instr->getOpcode()).find("RET") != std::string::npos);
      bool is_rel_type = (std::string(instr->getOpcode()).find("REL") != std::string::npos);
      bool is_call_or_ret_with_reg = false;

      if(track_this_instruction(cnt, threshold_unique_kernel_checking ,instr->getOpcode())) {
        current_kernel_key_instructions_by_pc[vpc] = inst_str;
      }

      std::shared_ptr<traced_instruction> inst_parsed = create_no_binay_instruction(vpc, inst_str);
      inst_parsed->set_simulation_opcode(OpcodeMap, inst_parsed->get_op_code());

      if(is_call_or_ret && !is_rel_type) {
        current_kernel_call_or_ret_by_pc[vpc] = inst_str;
      }

      map_func_addr_to_pc_to_sass_instr[addr_funct][vpc] = inst_str;
      
      /* We only report memory addresses */
      int mem_oper_idx = -1;
      int num_mref = 0;

      bool has_ldc_with_reg = false;
      bool has_ldc_with_ureg = false;
      std::vector<uint32_t> desc_ureg_ids;
      std::map<uint32_t, uint32_t> memref_idx_with_desc;
      std::map<uint32_t, traced_operand_instrument> per_operand_type;
      int ldc_reg_id = -1;

      std::vector<int> aux_reg_ids;
      uint64_t call_ret_imm = 0;
      uint32_t num_of_injects = 0;

      if(inst_parsed->is_tensor_core_op()) {
        inst_parsed->set_tensor_core_instruction_info();
      }
      for(int i = 0; i < instr->getNumOperands(); ++i){
        const InstrType::operand_t *op = instr->getOperand(i);
        if (op->type == InstrType::OperandType::MREF) {
          mem_oper_idx++;
          num_mref++;
          per_operand_type[num_of_injects] = traced_operand_instrument(TRACED_REG_TYPE::MEMORY_REF, 0, 0);
          if(op->u.mref.has_desc) {
            memref_idx_with_desc[mem_oper_idx] = op->u.mref.desc_ureg_num;
          }
          num_of_injects++;
        }else if( (op->type == InstrType::OperandType::CBANK) && (instr->getOperand(i)->u.cbank.has_reg_offset || (std::string(op->str).find("UR") != std::string::npos) ) ) {
          if(std::string(op->str).find("UR") != std::string::npos) {
            has_ldc_with_ureg = true;
            ldc_reg_id = get_ur_register(std::string(op->str));
          }else {
            has_ldc_with_reg = true;
            ldc_reg_id = instr->getOperand(i)->u.cbank.reg_offset;
          }
          per_operand_type[num_of_injects] = traced_operand_instrument(TRACED_REG_TYPE::MEMORY_REF, 0, 0);
          num_of_injects++;
        }else if(op->type == InstrType::OperandType::IMM_UINT64 && is_call_or_ret) {
          call_ret_imm = op->u.imm_uint64.value;
          per_operand_type[num_of_injects] = traced_operand_instrument(TRACED_REG_TYPE::MEMORY_REF, 0, 0);
          num_of_injects++;
        }else if(op->type == InstrType::OperandType::REG && is_call_or_ret) {
          aux_reg_ids.push_back(op->u.reg.num);
          aux_reg_ids.push_back(op->u.reg.num + 1);
          is_call_or_ret_with_reg = true;
          per_operand_type[num_of_injects] = traced_operand_instrument(TRACED_REG_TYPE::MEMORY_REF, 0, 0);
          num_of_injects++;
        }else if(op->type == InstrType::OperandType::REG) {
          TraceEnhancedOperandType reg_type = TraceEnhancedOperandType::REG;
          int num_uses = get_number_of_uses_per_operand(*inst_parsed, op->u.reg.num, i, reg_type);
          per_operand_type[num_of_injects] = traced_operand_instrument(TRACED_REG_TYPE::REGULAR, num_uses, op->u.reg.num);
          if(num_uses == 2) {
            per_operand_type[num_of_injects].reg_type = TRACED_REG_TYPE::REGULAR_2_REGS;
          }else if(num_uses == 4) {
            per_operand_type[num_of_injects].reg_type = TRACED_REG_TYPE::REGULAR_4_REGS;
          }
          num_of_injects++;
        }else if(op->type == InstrType::OperandType::UREG) {
          TraceEnhancedOperandType reg_type = TraceEnhancedOperandType::UREG;
          int num_uses = get_number_of_uses_per_operand(*inst_parsed, op->u.reg.num, i, reg_type);
          per_operand_type[num_of_injects] = traced_operand_instrument(TRACED_REG_TYPE::UNIFORM, num_uses, op->u.reg.num);
          if(num_uses == 2) {
            per_operand_type[num_of_injects].reg_type = TRACED_REG_TYPE::UNIFORM_2_REGS;
          }
          num_of_injects++;
        }else if(op->type == InstrType::OperandType::PRED) {
          int num_uses = get_number_of_uses_per_operand(*inst_parsed, op->u.reg.num, i, TraceEnhancedOperandType::PRED);
          per_operand_type[num_of_injects] = traced_operand_instrument(TRACED_REG_TYPE::PREDICATE, num_uses, op->u.reg.num);
          num_of_injects++;
        }else if(op->type == InstrType::OperandType::UPRED) {
          int num_uses = get_number_of_uses_per_operand(*inst_parsed, op->u.reg.num, i, TraceEnhancedOperandType::UPRED);
          per_operand_type[num_of_injects] = traced_operand_instrument(TRACED_REG_TYPE::UNIFORM_PREDICATE, num_uses, op->u.reg.num);
          num_of_injects++;
        }
      }

      if(num_of_injects == 0) {
        per_operand_type[0] = traced_operand_instrument(TRACED_REG_TYPE::NO_REGS, 0, 0);
        num_of_injects++;
      }

      for(unsigned int i = 0; i < num_of_injects; i++) {
          /* insert call to the instrumentation function with its arguments */
          nvbit_insert_call(instr, "instrument_inst", IPOINT_BEFORE);
          /* pass predicate value */
          nvbit_add_call_arg_guard_pred_val(instr);
          /* send Unique Function Identifier and PC */
          nvbit_add_call_arg_const_val32(instr, next_candidate_unique_function_id);
          nvbit_add_call_arg_const_val32(instr, (int)instr->getOffset());
          /* send number of injects and type of operand */
          nvbit_add_call_arg_const_val32(instr, num_of_injects);
          nvbit_add_call_arg_const_val32(instr, per_operand_type[i].reg_type);

          if(per_operand_type[i].reg_type == TRACED_REG_TYPE::MEMORY_REF) {
            if (mem_oper_idx >= 0) {
              nvbit_add_call_arg_const_val32(instr, MEM_TYPE::STANDARD_MEM);
              assert(num_mref <= 2);
              nvbit_add_call_arg_mref_addr64(instr, mem_oper_idx);
              nvbit_add_call_arg_const_val32(instr, (int)instr->getSize());
              if(memref_idx_with_desc.find(mem_oper_idx) != memref_idx_with_desc.end()) {
                nvbit_add_call_arg_ureg_val(instr, memref_idx_with_desc[mem_oper_idx]);
              }else {
                nvbit_add_call_arg_const_val32(instr, SECRET_UREG_DESC_NOT_USED);
              }
              nvbit_add_call_arg_const_val32(instr, SECRET_UREG_DESC_NOT_USED);
              nvbit_add_call_arg_const_val32(instr, SECRET_UREG_DESC_NOT_USED);
              mem_oper_idx--;
            }else if(has_ldc_with_reg || has_ldc_with_ureg) {
              nvbit_add_call_arg_const_val32(instr, MEM_TYPE::CONSTANT_MEM);
              nvbit_add_call_arg_const_val64(instr, 0);
              if(has_ldc_with_ureg) {
                nvbit_add_call_arg_ureg_val(instr, ldc_reg_id);
              }else {
                nvbit_add_call_arg_reg_val(instr, ldc_reg_id);
              }
              nvbit_add_call_arg_const_val32(instr, SECRET_UREG_DESC_NOT_USED);
              nvbit_add_call_arg_const_val32(instr, SECRET_UREG_DESC_NOT_USED);
              nvbit_add_call_arg_const_val32(instr, SECRET_UREG_DESC_NOT_USED);
            }else if(is_call_or_ret) {
              nvbit_add_call_arg_const_val32(instr, MEM_TYPE::CALL_OR_RET);
              nvbit_add_call_arg_const_val64(instr, call_ret_imm);
              if(is_call_or_ret_with_reg) {
                nvbit_add_call_arg_reg_val(instr, aux_reg_ids[0]);
                nvbit_add_call_arg_reg_val(instr, aux_reg_ids[1]);
              }else {
                nvbit_add_call_arg_const_val32(instr, 0);
                nvbit_add_call_arg_const_val32(instr, SECRET_UREG_DESC_NOT_USED);
              }
              nvbit_add_call_arg_const_val32(instr, SECRET_UREG_DESC_NOT_USED);
              nvbit_add_call_arg_const_val32(instr, SECRET_UREG_DESC_NOT_USED);
            }
          }else if(per_operand_type[i].reg_type == TRACED_REG_TYPE::REGULAR) {
            nvbit_add_call_arg_const_val32(instr, MEM_TYPE::NONE);
            nvbit_add_call_arg_const_val64(instr, per_operand_type[i].first_reg_id); 
            nvbit_add_call_arg_reg_val(instr, per_operand_type[i].first_reg_id);
            nvbit_add_call_arg_const_val32(instr, SECRET_UREG_DESC_NOT_USED);
            nvbit_add_call_arg_const_val32(instr, SECRET_UREG_DESC_NOT_USED);
            nvbit_add_call_arg_const_val32(instr, SECRET_UREG_DESC_NOT_USED);
          }else if(per_operand_type[i].reg_type == TRACED_REG_TYPE::REGULAR_2_REGS) {
            nvbit_add_call_arg_const_val32(instr, MEM_TYPE::NONE);
            nvbit_add_call_arg_const_val64(instr, per_operand_type[i].first_reg_id);
            nvbit_add_call_arg_reg_val(instr, per_operand_type[i].first_reg_id);
            nvbit_add_call_arg_reg_val(instr, per_operand_type[i].first_reg_id + 1);
            nvbit_add_call_arg_const_val32(instr, SECRET_UREG_DESC_NOT_USED);
            nvbit_add_call_arg_const_val32(instr, SECRET_UREG_DESC_NOT_USED);
          }else if(per_operand_type[i].reg_type == TRACED_REG_TYPE::REGULAR_4_REGS) {
            nvbit_add_call_arg_const_val32(instr, MEM_TYPE::NONE);
            nvbit_add_call_arg_const_val64(instr, per_operand_type[i].first_reg_id);
            nvbit_add_call_arg_reg_val(instr, per_operand_type[i].first_reg_id);
            nvbit_add_call_arg_reg_val(instr, per_operand_type[i].first_reg_id + 1);
            nvbit_add_call_arg_reg_val(instr, per_operand_type[i].first_reg_id + 2);
            nvbit_add_call_arg_reg_val(instr, per_operand_type[i].first_reg_id + 3);
          }else if(per_operand_type[i].reg_type == TRACED_REG_TYPE::UNIFORM) {
            nvbit_add_call_arg_const_val32(instr, MEM_TYPE::NONE);
            nvbit_add_call_arg_const_val64(instr, per_operand_type[i].first_reg_id);
            nvbit_add_call_arg_ureg_val(instr, per_operand_type[i].first_reg_id);
            nvbit_add_call_arg_const_val32(instr, SECRET_UREG_DESC_NOT_USED);
            nvbit_add_call_arg_const_val32(instr, SECRET_UREG_DESC_NOT_USED);
            nvbit_add_call_arg_const_val32(instr, SECRET_UREG_DESC_NOT_USED);
          }else if(per_operand_type[i].reg_type == TRACED_REG_TYPE::UNIFORM_2_REGS) {
            nvbit_add_call_arg_const_val32(instr, MEM_TYPE::NONE);
            nvbit_add_call_arg_const_val64(instr, per_operand_type[i].first_reg_id);
            nvbit_add_call_arg_ureg_val(instr, per_operand_type[i].first_reg_id);
            nvbit_add_call_arg_ureg_val(instr, per_operand_type[i].first_reg_id + 1);
            nvbit_add_call_arg_const_val32(instr, SECRET_UREG_DESC_NOT_USED);
            nvbit_add_call_arg_const_val32(instr, SECRET_UREG_DESC_NOT_USED);
          }else if(per_operand_type[i].reg_type == TRACED_REG_TYPE::PREDICATE) {
            nvbit_add_call_arg_const_val32(instr, MEM_TYPE::NONE);
            nvbit_add_call_arg_const_val64(instr, per_operand_type[i].first_reg_id);
            nvbit_add_call_arg_pred_reg(instr);
            nvbit_add_call_arg_const_val32(instr, SECRET_UREG_DESC_NOT_USED);
            nvbit_add_call_arg_const_val32(instr, SECRET_UREG_DESC_NOT_USED);
            nvbit_add_call_arg_const_val32(instr, SECRET_UREG_DESC_NOT_USED);
          }else if(per_operand_type[i].reg_type == TRACED_REG_TYPE::UNIFORM_PREDICATE) {
            nvbit_add_call_arg_const_val32(instr, MEM_TYPE::NONE);
            nvbit_add_call_arg_const_val64(instr, per_operand_type[i].first_reg_id);
            nvbit_add_call_arg_upred_reg(instr);
            nvbit_add_call_arg_const_val32(instr, SECRET_UREG_DESC_NOT_USED);
            nvbit_add_call_arg_const_val32(instr, SECRET_UREG_DESC_NOT_USED);
            nvbit_add_call_arg_const_val32(instr, SECRET_UREG_DESC_NOT_USED);
          }else {
            assert(per_operand_type[i].reg_type == TRACED_REG_TYPE::NO_REGS);
            assert(num_of_injects == 1);
            nvbit_add_call_arg_const_val32(instr, MEM_TYPE::NONE);
            nvbit_add_call_arg_const_val64(instr, SECRET_UREG_DESC_NOT_USED);
            nvbit_add_call_arg_const_val32(instr, SECRET_UREG_DESC_NOT_USED);
            nvbit_add_call_arg_const_val32(instr, SECRET_UREG_DESC_NOT_USED);
            nvbit_add_call_arg_const_val32(instr, SECRET_UREG_DESC_NOT_USED);
            nvbit_add_call_arg_const_val32(instr, SECRET_UREG_DESC_NOT_USED);
          }

          /* add pointer to channel_dev and other counters*/
          nvbit_add_call_arg_const_val64(instr, (uint64_t)ctx_state->channel_dev);
          nvbit_add_call_arg_const_val64(instr,
                                        (uint64_t)&total_dynamic_instr_counter);
          nvbit_add_call_arg_const_val64(instr,
                                        (uint64_t)&reported_dynamic_instr_counter);
          nvbit_add_call_arg_const_val64(instr, (uint64_t)&stop_report[device_id]);
      }
      cnt++;
    }

    pthread_mutex_lock(&mutex);
    int variant_id = 0;
    auto it_map_already_instrumented = map_function_to_kernel_name_and_variant_id.find(f);
    assert(it_map_already_instrumented == map_function_to_kernel_name_and_variant_id.end());

    auto it_already_traced = all_kernels_key_instructions_by_pc.find(current_kernel_name);
    if (it_already_traced == all_kernels_key_instructions_by_pc.end())
    {
      all_kernels_key_instructions_by_pc[current_kernel_name].push_back(traced_kernel_id(current_kernel_name, 0, next_candidate_unique_function_id, current_kernel_key_instructions_by_pc, current_kernel_call_or_ret_by_pc, addr_funct));
    }
    else
    {
      bool is_already_traced = false;
      for (unsigned int i = 0; !is_already_traced && (i < it_already_traced->second.size()); i++)
      {
        if (are_two_kernels_equal(it_already_traced->second[i], current_kernel_key_instructions_by_pc, current_kernel_call_or_ret_by_pc, true)) // CAMBIAR A FUNCION
        {
          is_already_traced = true;
          variant_id = it_already_traced->second[i].variant_id;
        }
      }

      if (!is_already_traced)
      {
        variant_id = it_already_traced->second.size();
        all_kernels_key_instructions_by_pc[current_kernel_name].push_back(traced_kernel_id(current_kernel_name, it_already_traced->second.size(), next_candidate_unique_function_id, current_kernel_key_instructions_by_pc, current_kernel_call_or_ret_by_pc, addr_funct));
      }
    }
    
    std::string candidate_final_kernel_name = current_kernel_name + variant_delimiter_str + std::to_string(variant_id);
    map_function_to_kernel_name_and_variant_id[f] = std::make_tuple(current_kernel_name, variant_id);
    map_function_name_to_unique_function_id_with_variant[candidate_final_kernel_name] = next_candidate_unique_function_id;
    map_function_name_to_unique_function_id_without_variant[current_kernel_name] = next_candidate_unique_function_id;
    map_func_addr_to_kernel_name[addr_funct] = current_kernel_name;
    map_kernel_name_to_func_addr[current_kernel_name] = addr_funct;
    pthread_mutex_unlock(&mutex);
  }
}

__global__ void flush_channel(ChannelDev* ch_dev) {
  /* push memory access with negative cta id to communicate the kernel is
   * completed */
  inst_trace_t ma;
  ma.cta_id_x = -1;
  ch_dev->push(&ma, sizeof(inst_trace_t));

  /* flush channel */
  ch_dev->flush();
}

static FILE *statsFile = NULL;
static bool first_call = true;
static unsigned int pending_devices_to_finish = 0;

unsigned old_total_insts = 0;
unsigned old_total_reported_insts = 0;

void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
                         const char *name, void *params, CUresult *pStatus) {
  // std::cout << "nvbit_at_cuda_event: "  << cbid << std::endl; fflush(stdout);
  pthread_mutex_lock(&mutex);
  /* we prevent re-entry on this callback when issuing CUDA functions inside
    * this function */
  if (skip_callback_flag) {
      pthread_mutex_unlock(&mutex);
      return;
  }

  assert(ctx_state_map.find(ctx) != ctx_state_map.end());
  CTXstate* ctx_state = ctx_state_map[ctx];

  if (first_call == true) {

    first_call = false;
    
    create_folder(traces_path.c_str());

    if (active_from_start && !dynamic_kernel_limit_start || dynamic_kernel_limit_start == 1)
      active_region = true;
    else {
      if (active_from_start)
        active_region = false;
    }
    
    if(user_defined_folders == 1)
    {
      std::string usr_folder = std::getenv("TRACES_FOLDER");
      std::string temp_traces_location = usr_folder;
      std::string temp_stats_location = usr_folder + "/stats.csv";
      traces_location.resize(temp_traces_location.size());
      stats_location.resize(temp_stats_location.size());
      traces_location.replace(traces_location.begin(), traces_location.end(),temp_traces_location);
      stats_location.replace(stats_location.begin(), stats_location.end(),temp_stats_location);
      printf("\n Traces location is %s \n", traces_location.c_str());
      printf("Stats location is %s \n", stats_location.c_str());
    }

    statsFile = fopen(stats_location.c_str(), "w");
    fprintf(statsFile,
            "device_id, stream_id, kernel id, kernel mangled name, grid_dimX, grid_dimY, grid_dimZ, "
            "#blocks, block_dimX, block_dimY, block_dimZ, #threads, "
            "total_insts, total_reported_insts\n");
    fclose(statsFile);
  }

  if (cbid == API_CUDA_cuMemcpyHtoD_v2) {
    if (!is_exit) {
      cuMemcpyHtoD_v2_params *p = (cuMemcpyHtoD_v2_params *)params;
      uint64_t stream_key = 0;// Memcpy unless they are asynchronous, they are always in stream 0.
      char buffer[1024];
      int device_id;
      cuCtxGetDevice(&device_id);
      sprintf(buffer, "MemcpyHtoD,0x%016llx,%lu", p->dstDevice, p->ByteCount);
      dynamic_trace::gpu_device &gpu_dev = (*dyn_trace.mutable_gpu_device())[device_id];
      dynamic_trace::cuda_stream &stream = (*gpu_dev.mutable_streams())[stream_key];
      stream.add_ordered_cuda_events(buffer);
    }

  } else if (cbid == API_CUDA_cuLaunchKernel_ptsz ||
             cbid == API_CUDA_cuLaunchKernel) {
    cuLaunchKernel_params *p = (cuLaunchKernel_params *)params;

    if (!is_exit) {
      int device_id;
      cuCtxGetDevice(&device_id);
      if (active_from_start && dynamic_kernel_limit_start && kernel_id[device_id] == dynamic_kernel_limit_start)
        active_region = true;

      if (terminate_after_limit_number_of_kernels_reached && dynamic_kernel_limit_end != 0 && kernel_id[device_id] > dynamic_kernel_limit_end)
      {
        pthread_mutex_unlock(&mutex);
        exit(0);
      }
      
      int nregs = 0;
      CUDA_SAFECALL(
          cuFuncGetAttribute(&nregs, CU_FUNC_ATTRIBUTE_NUM_REGS, p->f));

      int shmem_static_nbytes = 0;
      CUDA_SAFECALL(cuFuncGetAttribute(
          &shmem_static_nbytes, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, p->f));

      CUDA_SAFECALL(cuFuncGetAttribute(&binary_version,
                                       CU_FUNC_ATTRIBUTE_BINARY_VERSION, p->f));

      get_opcode_map(OpcodeMap, binary_version);
      instrument_function_if_needed(ctx, p->f, device_id);

      if (active_region) {
        nvbit_enable_instrumented(ctx, p->f, true);
        stop_report[device_id] = false;
      } else {
        nvbit_enable_instrumented(ctx, p->f, false);
        stop_report[device_id] = true;
      }

      char buffer[1024];
      sprintf(buffer, std::string(traces_location+"/kernel-%d.trace").c_str(), kernel_id[device_id]);
      dynamic_trace::gpu_device &gpu_dev = (*dyn_trace.mutable_gpu_device())[device_id];

      if (!stop_report[device_id]) {
        int variant_id = 0;
        auto it_map_already_instrumented = map_function_to_kernel_name_and_variant_id.find(p->f);
        assert(it_map_already_instrumented != map_function_to_kernel_name_and_variant_id.end());
        variant_id = std::get<1>(it_map_already_instrumented->second);
        std::string kernel_name(nvbit_get_func_name(ctx, p->f, true));

        // std::vector<int> kernel_argument_sizes = nvbit_get_kernel_argument_sizes(p->f);
        
        std::string final_kernel_name = kernel_name + variant_delimiter_str + std::to_string(variant_id);
        
        auto it_map_unique_function_id = map_function_name_to_unique_function_id_with_variant.find(final_kernel_name);
        assert(it_map_unique_function_id != map_function_name_to_unique_function_id_with_variant.end());
        dyn_trace.set_binary_version(binary_version);
        auto stream_map = gpu_dev.streams();
        uint64_t stream_key = (uint64_t)p->hStream;
        stream_key = 0; // We use 0 as stream key for now, as we do not support multiple streams yet. WIPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP
        current_stream_id[device_id] = stream_key;
        dynamic_trace::cuda_stream &stream = (*gpu_dev.mutable_streams())[stream_key];
        stream.set_id(stream_key);
        dynamic_trace::kernel *ker = stream.add_kernels();
        ker->set_id(kernel_id[device_id]);
        ker->set_name(final_kernel_name);
        ker->set_function_unique_id(it_map_unique_function_id->second);
        ker->set_size_shared_memory(shmem_static_nbytes + p->sharedMemBytes);
        ker->set_number_of_registers(nregs);
        ker->set_shared_memory_base_address((uint64_t)nvbit_get_shmem_base_addr(ctx));
        ker->set_local_memory_base_address((uint64_t)nvbit_get_local_mem_base_addr(ctx));
        dynamic_trace::dim3d *grid_dim = ker->mutable_grid_dim();
        grid_dim->set_x(p->gridDimX);
        grid_dim->set_y(p->gridDimY);
        grid_dim->set_z(p->gridDimZ);
        dynamic_trace::dim3d *block_dim = ker->mutable_block_dim();
        block_dim->set_x(p->blockDimX);
        block_dim->set_y(p->blockDimY);
        block_dim->set_z(p->blockDimZ);
      }

      // This will be a relative path to the traces file
      sprintf(buffer,"kernel-%d.trace", kernel_id[device_id]);
      if (!stop_report[device_id]) {
        uint64_t stream_key = (uint64_t)p->hStream;
        stream_key = 0; // We use 0 as stream key for now, as we do not support multiple streams yet. WIPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP
        dynamic_trace::cuda_stream &stream = (*gpu_dev.mutable_streams())[stream_key];
        stream.add_ordered_cuda_events(buffer);
      }

      statsFile = fopen(stats_location.c_str(), "a");
      unsigned blocks = p->gridDimX * p->gridDimY * p->gridDimZ;
      unsigned threads = p->blockDimX * p->blockDimY * p->blockDimZ;

      fprintf(statsFile, "%d, %d, %s, %s, %d, %d, %d, %d, %d, %d, %d, %d, ", device_id , current_stream_id[device_id], buffer,
              nvbit_get_func_name(ctx, p->f, true), p->gridDimX, p->gridDimY,
              p->gridDimZ, blocks, p->blockDimX, p->blockDimY, p->blockDimZ,
              threads);

      fclose(statsFile);

      kernel_id[device_id]++;
      recv_thread_receiving[ctx] = true;

    } else {
      int device_id;
      cuCtxGetDevice(&device_id);
      /* make sure current kernel is completed */
      cudaDeviceSynchronize();
      // GET CUDA ERROR:
      cudaError_t cuErr = cudaGetLastError();
      if (cuErr != cudaSuccess) {
        fprintf(stdout, "CUDA error in file '%s' in line %i : %s.\n", __FILE__,
                __LINE__, cudaGetErrorString(cuErr));
        fflush(stdout);
        abort();
      }
      assert(cuErr == cudaSuccess);

      /* make sure we prevent re-entry on the nvbit_callback when issuing
       * the flush_channel kernel */
      skip_callback_flag = true;

      /* issue flush of channel so we are sure all the memory accesses
       * have been pushed */
      flush_channel<<<1, 1>>>(ctx_state->channel_dev);
      cudaDeviceSynchronize();
      assert(cudaGetLastError() == cudaSuccess);

      /* unset the skip flag */
      skip_callback_flag = false;

      /* wait here until the receiving thread has not finished with the
       * current kernel */
      while (recv_thread_receiving[ctx]) {
        pthread_yield();
      }

      unsigned total_insts_per_kernel =
          total_dynamic_instr_counter - old_total_insts;
      old_total_insts = total_dynamic_instr_counter;

      unsigned reported_insts_per_kernel =
          reported_dynamic_instr_counter - old_total_reported_insts;
      old_total_reported_insts = reported_dynamic_instr_counter;

      statsFile = fopen(stats_location.c_str(), "a");
      fprintf(statsFile, "%d,%d", total_insts_per_kernel,
              reported_insts_per_kernel);
      fprintf(statsFile, "\n");
      fclose(statsFile);
      
      if (!stop_report[device_id]) {
        create_folder(threadblock_trace_path.c_str());
        create_folder(threadblock_register_values_path.c_str());
        auto it_tb = threadblocks.begin();
        while(it_tb != threadblocks.end()) {
          std::string tb_string_id = it_tb->first;
          ThreadblockStringParseInfo tb_info = parse_tb_string_id(tb_string_id);
          if((tb_info.device_id == device_id) && (tb_info.stream_id == current_stream_id[device_id]) && (tb_info.kernel_id == (kernel_id[device_id]-1))) {
            // Create hierarchical folder structure
            std::string device_folder = threadblock_trace_path + "/device_" + std::to_string(tb_info.device_id);
            std::string stream_folder = device_folder + "/stream_" + std::to_string(tb_info.stream_id);
            std::string kernel_folder = stream_folder + "/kernel_" + std::to_string(tb_info.kernel_id);
            
            create_folder(device_folder.c_str());
            create_folder(stream_folder.c_str());
            create_folder(kernel_folder.c_str());
            
            dynamic_trace::threadblock &tb = it_tb->second;
            std::string tb_file = kernel_folder + "/" + tb_string_id + ".pb";
            std::ofstream ofs_tb(tb_file, std::ios::out | std::ios::binary);
            if (!tb.SerializeToOstream(&ofs_tb)) {
              std::cerr << "Failed to write threadblock content." << std::endl;
              fflush(stdout);
              abort();
            }else {
              tb.Clear();
            }
            ofs_tb.close();
            it_tb = threadblocks.erase(it_tb);
          }else {
            it_tb++;
          }
        }
      }
      if (active_from_start && dynamic_kernel_limit_end && kernel_id[device_id] > dynamic_kernel_limit_end)
        active_region = false;
    }
  } else if (cbid == API_CUDA_cuProfilerStart && is_exit) {
      if (!active_from_start) {
        active_region = true;
      }
  } else if (cbid == API_CUDA_cuProfilerStop && is_exit) {
      if (!active_from_start) {
        active_region = false;
      }
  }

  pthread_mutex_unlock(&mutex);
}

bool is_number(const std::string &s) {
  std::string::const_iterator it = s.begin();
  while (it != s.end() && std::isdigit(*it))
    ++it;
  return !s.empty() && it == s.end();
}

unsigned get_datawidth_from_opcode(const std::vector<std::string> &opcode) {
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

  return 4; // default is 4 bytes
}

bool check_opcode_contain(const std::vector<std::string> &opcode,
                          std::string param) {
  for (unsigned i = 0; i < opcode.size(); ++i)
    if (opcode[i] == param)
      return true;

  return false;
}

bool base_stride_compress(const uint64_t *addrs, const std::bitset<32> &mask,
                          uint64_t &base_addr, int &stride) {

  // calulcate the difference between addresses
  // write cosnsctive addresses with constant stride in a more
  // compressed way (i.e. start adress and stride)
  bool const_stride = true;
  bool first_bit1_found = false;
  bool last_bit1_found = false;

  for (int s = 0; s < 32; s++) {
    if (mask.test(s) && !first_bit1_found) {
      first_bit1_found = true;
      base_addr = addrs[s];
      if (s < 31 && mask.test(s + 1))
        stride = addrs[s + 1] - addrs[s];
      else {
        const_stride = false;
        break;
      }
    } else if (first_bit1_found && !last_bit1_found) {
      if (mask.test(s)) {
        int diff_addr = addrs[s] - addrs[s - 1];
        if (stride != diff_addr) {
          const_stride = false;
          break;
        }
      } else
        last_bit1_found = true;
    } else if (last_bit1_found) {
      if (mask.test(s)) {
        const_stride = false;
        break;
      }
    }
  }

  return const_stride;
}

void base_delta_compress(const uint64_t *addrs, const std::bitset<32> &mask,
                         uint64_t &base_addr, std::vector<long long> &deltas) {

  // save the delta from the previous address
  bool first_bit1_found = false;
  uint64_t last_address = 0;
  for (int s = 0; s < 32; s++) {
    if (mask.test(s) && !first_bit1_found) {
      base_addr = addrs[s];
      first_bit1_found = true;
      last_address = addrs[s];
    } else if (mask.test(s) && first_bit1_found) {
      deltas.push_back(addrs[s] - last_address);
      last_address = addrs[s];
    }
  }
}

void *recv_thread_fun(void *args) {
  CUcontext ctx = (CUcontext)args;
  pthread_mutex_lock(&mutex);
  /* get context state from map */
  assert(ctx_state_map.find(ctx) != ctx_state_map.end());
  CTXstate* ctx_state = ctx_state_map[ctx];
  CUcontext current_ctx;
  int device_id;
  cuCtxGetCurrent(&current_ctx);
  cuCtxSetCurrent(ctx);
  cuCtxGetDevice(&device_id);
  cuCtxSetCurrent(current_ctx);
  ChannelHost* ch_host = &ctx_state->channel_host;
  dynamic_trace::gpu_device &gpu_dev = (*dyn_trace.mutable_gpu_device())[device_id];
  gpu_dev.set_id(device_id);
  pthread_mutex_unlock(&mutex);

  char *recv_buffer = (char *)malloc(CHANNEL_SIZE);

  while (ctx_state->recv_thread_done == RecvThreadState::WORKING) {
    uint32_t num_recv_bytes = ch_host->recv(recv_buffer, CHANNEL_SIZE);
    if (num_recv_bytes > 0) {
      uint32_t num_processed_bytes = 0;
      dynamic_trace::cuda_stream &stream = (*gpu_dev.mutable_streams())[current_stream_id[device_id]];
      dynamic_trace::kernel &ker = (*stream.mutable_kernels())[kernel_id[device_id]-2]; 
      while (num_processed_bytes < num_recv_bytes) {
        inst_trace_t *ma = (inst_trace_t *)&recv_buffer[num_processed_bytes];

        /* when we receive a CTA_id_x it means all the kernels
        * completed, this is the special token we receive from the
        * flush channel kernel that is issues at the end of the
        * context */
        if (ma->cta_id_x == -1) {
          recv_thread_receiving[ctx] = false;
          break;
        }
        while(!recv_thread_receiving[ctx]) {
          pthread_yield();
        }
        // Key: d_{device}_s_{stream}_k_{kernel}_{cta_id_x},{cta_id_y},{cta_id_z}
        std::string tb_string_id = "d_" + std::to_string(device_id) + "_s_" + std::to_string(current_stream_id[device_id]) + "_k_" + std::to_string(kernel_id[device_id]-1) + "_" + std::to_string(ma->cta_id_x) + "," +
                                   std::to_string(ma->cta_id_y) + "," +
                                   std::to_string(ma->cta_id_z);
        dynamic_trace::threadblock &tb = get_threadblock(tb_string_id);
        dynamic_trace::dim3d *cta_id = tb.mutable_block_id();
        cta_id->set_x(ma->cta_id_x);
        cta_id->set_y(ma->cta_id_y);
        cta_id->set_z(ma->cta_id_z);
        int warp_id_tb = ma->warpid_tb;
        dynamic_trace::warp &wp = (*tb.mutable_warps())[warp_id_tb];
        wp.set_id(warp_id_tb);
        // if (print_core_id) {
        //   tb.set_sm_id(ma->sm_id);
        //   wp.set_warp_id_in_sm(ma->warpid_sm);
        // }
        int opcode_id = pc_to_opcode[ma->unique_function_id][ma->vpc];
        unsigned int &remaining_injects = get_remaining_injects_to_current_instruction(tb_string_id, warp_id_tb);
        dynamic_trace::instruction *inst;
        if(remaining_injects == 0) {
          inst = wp.add_instructions();
          inst->set_pc(ma->vpc);
          inst->set_active_mask(ma->active_mask);
          inst->set_predicate_mask(ma->predicate_mask);
          inst->set_function_unique_id(ma->unique_function_id);
          remaining_injects = ma->num_of_injects;
        }else {
          inst = wp.mutable_instructions(wp.instructions_size()-1);
        }
        remaining_injects--;
        std::bitset<32> mask(ma->active_mask & ma->predicate_mask);
        if (ma->mem_type != MEM_TYPE::NONE) {
          dynamic_trace::address *addr = inst->add_addresses();
          std::istringstream iss(id_to_opcode_map[opcode_id]);
          std::vector<std::string> tokens;
          std::string token;
          addr->set_udesc_value(ma->ureg_desc_value);
          if(ma->mem_type == MEM_TYPE::CALL_OR_RET) {
            addr->set_data_width(1);
          }else {
            while (std::getline(iss, token, '.'))
            {
              if (!token.empty())
                tokens.push_back(token);
            }
            addr->set_data_width(get_datawidth_from_opcode(tokens));
          }

          bool base_stride_success = false;
          uint64_t base_addr = 0;
          int stride = 0;
          std::vector<long long> deltas;

          if (enable_compress) {
            // try base+stride format
            base_stride_success =
                base_stride_compress(ma->addrs_or_reg_val_0, mask, base_addr, stride);
            if (!base_stride_success) {
              // if base+stride fails, try base+delta format
              base_delta_compress(ma->addrs_or_reg_val_0, mask, base_addr, deltas);
            }
          }

          if (base_stride_success && enable_compress) {
            // base + stride format
            addr->set_compression_format(address_format::base_stride);
            addr->set_base_address(base_addr);
            addr->set_stride(stride);
          } else if (!base_stride_success && enable_compress) {
            // base + delta format
            addr->set_compression_format(address_format::base_delta);
            addr->set_base_address(base_addr);
            for (unsigned int s = 0; s < deltas.size(); s++) {
              addr->add_addrs(deltas[s]);
            }
          } else {
            // list all the addresses
            addr->set_compression_format(address_format::list_all);
            for (int s = 0; s < 32; s++) {
              if (mask.test(s))
                addr->add_addrs(ma->addrs_or_reg_val_0[s]);
            }
          }
        }
        num_processed_bytes += sizeof(inst_trace_t);
      }
    }
  }
  free(recv_buffer);
  ctx_state->recv_thread_done = RecvThreadState::FINISHED;
  return NULL;
}

int check_system_call(int system_res, const char* syscall) {
  if(system_res != 0) {
    std::cout << "Error. System call failed. System call: " << syscall << std::endl;
    fflush(stdout);
    std::string str_error(syscall);
    if(str_error.find("nvdisasm -lrm count") == std::string::npos) {
      abort();
    }
  }
  return system_res;
}

void enhanced_tracer() {
  create_folder(extrainfo_path.c_str());
  create_folder(cubin_path.c_str());
  create_folder(sass_path.c_str());
  create_folder(register_usage_path.c_str());
  std::string program_path = get_program_path();
  std::size_t found = program_path.find_last_of("/");
  std::string program_name = program_path.substr(found + 1);
  std::string command_get_cubin = "cd " + cubin_path + " && cuobjdump " + program_path + " -xelf all -arch=sm_" + std::to_string(binary_version);
  std::cout << "Generating extra information for the enhanced traces of benchmark: " << program_name << std::endl;
  check_system_call(system(command_get_cubin.c_str()), command_get_cubin.c_str());
  m_enhanced_traced_execution = new traced_execution(program_name);
  std::string absolute_cubin_path = cwd + "/" + cubin_path;
  for (const auto &entry : std::filesystem::directory_iterator(absolute_cubin_path))
  {
    std::string aux_cubin = entry.path().filename().string();
    std::string base_name = entry.path().stem().string();
    std::string command_get_register_usage = "cd " + register_usage_path + " && nvdisasm -lrm count ../cubin/" + aux_cubin + " > " + base_name + ".rfu";
    std::string command_get_sass = "cd " + sass_path + " && cuobjdump -sass ../cubin/" + aux_cubin + " > " + base_name + ".sass";
    std::cout << "Parsing cubin: " << aux_cubin << std::endl;
    int call_code_rfu = check_system_call(system(command_get_register_usage.c_str()), command_get_register_usage.c_str());
    int call_code_sass = check_system_call(system(command_get_sass.c_str()), command_get_sass.c_str());
    if(call_code_sass == 0) {
      parse_sass(binary_version, entry);
    }
    if(call_code_rfu == 0) {
      parse_rfu(entry);
    }
  }
  for(auto kernel_name : all_kernels_key_instructions_by_pc) {
    for(auto variant : kernel_name.second) {
      if(!variant.sass_has_been_parsed || !variant.rfu_has_been_parsed) {
        std::cout << "Error. Kernel " << kernel_name.first << " variant " << variant.variant_id << " has not been parsed." << std::endl;
        std::cout << "SASS parsed: " << variant.sass_has_been_parsed << std::endl;
        std::cout << "RFU parsed: " << variant.rfu_has_been_parsed << std::endl;
        std::cout << "Traced instructions: " << std::endl;
        print_map(variant.key_instructions_by_pc);
        auto it_already_captured_instr = map_func_addr_to_pc_to_sass_instr.find(variant.func_addr);
        assert(it_already_captured_instr != map_func_addr_to_pc_to_sass_instr.end());
        std::string kernel_name_to_add = variant.original_kernel_name + variant_delimiter_str + std::to_string(variant.variant_id);
        m_enhanced_traced_execution->add_no_binary_kernel(kernel_name_to_add, variant.unique_function_id, variant.func_addr, binary_version, it_already_captured_instr->second, false);
      }
    }
  }
  std::cout << "Enhanced tracer has parsed " << all_kernels_key_instructions_by_pc.size() << "/" << already_instrumented.size() << " kernels" << std::endl;
  if(!intermediate_extra_files_persistance) {
    remove_folder(cubin_path.c_str());
    remove_folder(sass_path.c_str());
    remove_folder(register_usage_path.c_str());
  }
  m_enhanced_traced_execution->remove_useless_kernels();
  m_enhanced_traced_execution->SerializeToFile(extrainfo_path +"/enhanced_execution_info.json");
}

void nvbit_at_ctx_init(CUcontext ctx) {
  pthread_mutex_lock(&mutex);
  pending_devices_to_finish++;
  if (verbose) {
      printf("Tracer: Starting context %p\n", ctx);
  }
  assert(ctx_state_map.find(ctx) == ctx_state_map.end());
  CTXstate* ctx_state = new CTXstate;
  ctx_state_map[ctx] = ctx_state;
  pthread_mutex_unlock(&mutex);
}

void init_context_state(CUcontext ctx) {
  if(is_first_init_context_call) {
    is_first_init_context_call = false;
    cudaGetDeviceCount(&num_devices);
    cudaMallocManaged(&stop_report, num_devices * sizeof(bool));
    for(int i = 0; i < num_devices; ++i) {
      stop_report[i] = false;
      kernel_id.push_back(1);
      current_stream_id.push_back(0);
    }
  }
  CTXstate* ctx_state = ctx_state_map[ctx];
  ctx_state->recv_thread_done = RecvThreadState::WORKING;
  cudaMallocManaged(&ctx_state->channel_dev, sizeof(ChannelDev));
  ctx_state->channel_host.init((int)ctx_state_map.size() - 1, CHANNEL_SIZE,
                               ctx_state->channel_dev, recv_thread_fun, ctx);
  nvbit_set_tool_pthread(ctx_state->channel_host.get_thread());
}

void nvbit_tool_init(CUcontext ctx) {
  pthread_mutex_lock(&mutex);
  assert(ctx_state_map.find(ctx) != ctx_state_map.end());
  init_context_state(ctx);
  pthread_mutex_unlock(&mutex);
}

void nvbit_at_ctx_term(CUcontext ctx) {
  pthread_mutex_lock(&mutex);
  skip_callback_flag = true;
  if (verbose) {
      printf("Tracer: Terminating context %p\n", ctx);
  }
  /* get context state from map */
  assert(ctx_state_map.find(ctx) != ctx_state_map.end());
  CTXstate* ctx_state = ctx_state_map[ctx];
  pending_devices_to_finish--;
  if(pending_devices_to_finish == 0) {
    dyn_trace.set_nvbit_version(NVBIT_VERSION);
    dyn_trace.set_accelsim_version(TRACER_VERSION);
    dyn_trace.set_is_gathered_registers_values(static_cast<bool>(gather_registers));
    std::string program_name = get_program_path();
    std::size_t found = program_name.find_last_of("/");
    if(found != std::string::npos) {
      program_name = program_name.substr(found + 1);
    }
    dyn_trace.set_name(program_name);
    std::string filename = traces_path + "/dynamic_trace.pb";
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs) {
      std::cerr << "Failed to open file " << filename << " for writing." << std::endl;
  } else {
      if (!dyn_trace.SerializeToOstream(&ofs)) {
          std::cerr << "Failed to serialize dyn_trace." << std::endl;
      } else {
          std::cout << "Serialized protocol buffer written to " << filename << std::endl;
          dyn_trace.Clear();
      }
      ofs.close();
  }
    std::cout << "Starting the enhanced tracer" << std::endl;
    enhanced_tracer();
    std::cout << "Terminated the enhanced tracer" << std::endl;
  }
  ctx_state->recv_thread_done = RecvThreadState::STOP;
  while (ctx_state->recv_thread_done != RecvThreadState::FINISHED);
  ctx_state->channel_host.destroy(false);
  cudaFree(ctx_state->channel_dev);
  skip_callback_flag = false;
  delete ctx_state;
  pthread_mutex_unlock(&mutex);
}
