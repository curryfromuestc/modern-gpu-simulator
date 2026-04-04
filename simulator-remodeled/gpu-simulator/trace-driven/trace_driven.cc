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

// Copyright (c) 2018-2021, Mahmoud Khairy, Vijay Kandiah, Timothy Rogers, Tor M. Aamodt, Nikos Hardavellas
// Northwestern University, Purdue University, The University of British Columbia
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer;
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution;
// 3. Neither the names of Northwestern University, Purdue University,
//    The University of British Columbia nor the names of their contributors
//    may be used to endorse or promote products derived from this software
//    without specific prior written permission.
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
#include <memory>

#include "../ISA_Def/blackwell_opcode.h"
#include "../ISA_Def/ampere_opcode.h"
#include "../ISA_Def/kepler_opcode.h"
#include "../ISA_Def/pascal_opcode.h"
#include "../ISA_Def/trace_opcode.h"
#include "../ISA_Def/turing_opcode.h"
#include "../ISA_Def/volta_opcode.h"
#include "../ISA_Def/accelwattch_component_mapping.h"
#include "abstract_hardware_model.h"
#include "cuda-sim/cuda-sim.h"
#include "cuda-sim/ptx_ir.h"
#include "cuda-sim/ptx_parser.h"
#include "gpgpu_context.h"
#include "gpgpusim_entrypoint.h"
#include "option_parser.h"
#include "trace_driven.h"

#include "../gpgpu-sim/src/gpgpu-sim/remodeling/sm.h"
#include "../gpgpu-sim/src/gpgpu-sim/remodeling/ldst_unit_sm.h"

#include "../../util/traces_enhanced/src/traced_instruction.h"
#include "../../util/traces_enhanced/src/string_utilities.h"


void advance_trace_cta_id(kernel_trace_t *kernel_trace_info) {
  if(kernel_trace_info->next_tb_to_parse_x < (kernel_trace_info->grid_dim_x - 1)){
    kernel_trace_info->next_tb_to_parse_x++; 
  }else if(kernel_trace_info->next_tb_to_parse_y < (kernel_trace_info->grid_dim_y - 1)){
    kernel_trace_info->next_tb_to_parse_x = 0;
    kernel_trace_info->next_tb_to_parse_y++;
  }else if(kernel_trace_info->next_tb_to_parse_z < (kernel_trace_info->grid_dim_z - 1)) {
    kernel_trace_info->next_tb_to_parse_x = 0;
    kernel_trace_info->next_tb_to_parse_y = 0;
    kernel_trace_info->next_tb_to_parse_z++;
  }
}

trace_warp_inst_t *trace_shd_warp_t::get_next_trace_inst(address_type pc) {
  if (used_insts < traced_pcs.size()) {
    trace_warp_inst_t *new_inst =
        new trace_warp_inst_t(get_shader()->get_config());
    auto it_inst_trace = map_warp_traces.find(pc);
    inst_trace_t *trace_ptr;
    bool is_pc_found = true;
    if ((it_inst_trace != map_warp_traces.end()) && (it_inst_trace->second.num_used_instructions[0] < it_inst_trace->second.num_traced_instructions)) {
      trace_ptr = &(it_inst_trace->second.instructions[it_inst_trace->second.num_used_instructions[0]]);
    } else {
      is_pc_found = false;
      trace_ptr = new inst_trace_t(pc, get_current_unique_function_id_call(), is_pc_found);
    }
    inst_trace_t &trace = *trace_ptr;
    traced_execution& trc_exec = get_shader()->get_gpu()->get_extra_trace_info();
    new_inst->parse_from_trace_struct(
        trace, m_kernel_info->OpcodeMap,
        m_kernel_info->m_tconfig, m_kernel_info->m_kernel_trace_info, trc_exec);
    new_inst->set_extra_trace_instruction_info(trc_exec.get_kernel_by_unique_function_id(new_inst->unique_function_id).get_instruction_ptr(pc));
    if(!is_pc_found) {
      delete trace_ptr;
    }else {
      it_inst_trace->second.num_used_instructions[0]++;
      used_insts++;
    }
    return new_inst;
  } else
    return NULL;
}

void trace_shd_warp_t::decrease_num_used_inst(address_type pc){
  auto it_inst_trace = map_warp_traces.find(pc);
  assert(it_inst_trace != map_warp_traces.end());
  assert(it_inst_trace->second.num_used_instructions[0] > 0);
  it_inst_trace->second.num_used_instructions[0]--;
  assert(used_insts > 0);
  used_insts--;
}

// MOD. Begin. VPREG. Not totally compatible now.
void trace_shd_warp_t::decrement_trace_pc() {
  if(used_insts > 0)
  {
    used_insts--;
  }
}
// MOD. End. VPREG


void trace_shd_warp_t::clear() {
  used_insts = 0;
  map_warp_traces.clear();
  traced_pcs.clear();
}

// functional_done
bool trace_shd_warp_t::trace_done() { return used_insts == (traced_pcs.size()); }

address_type trace_shd_warp_t::get_start_trace_pc() {
  assert(traced_pcs.size() > 0);
  return traced_pcs.at(0);
}

address_type trace_shd_warp_t::get_pc() {
  assert(traced_pcs.size() > 0);
  assert(used_insts < traced_pcs.size());
  return traced_pcs[used_insts];
}

trace_kernel_info_t::trace_kernel_info_t(dim3 gridDim, dim3 blockDim,
                                         trace_function_info *m_function_info,
                                         trace_parser *parser,
                                         class trace_config *config,
                                         kernel_trace_t *kernel_trace_info)
    : kernel_info_t(gridDim, blockDim, m_function_info) {
  m_parser = parser;
  m_tconfig = config;
  m_kernel_trace_info = kernel_trace_info;
  m_was_launched = false;

  // resolve the binary version
  get_opcode_map(OpcodeMap, kernel_trace_info->binary_verion);
}

void trace_kernel_info_t::get_next_threadblock_traces(
    std::vector<std::map<address_type, traced_instructions_by_pc> *> threadblock_traces,
    std::vector<std::vector<address_type> *> threadblock_traced_pcs, gpgpu_sim *gpu, traced_execution &static_trace_info) {
  m_parser->get_next_threadblock_traces(threadblock_traces, threadblock_traced_pcs,
              m_kernel_trace_info->gpu_device_id, m_kernel_trace_info->cuda_stream_id, m_kernel_trace_info->kernel_id,
              m_kernel_trace_info->trace_verion, m_kernel_trace_info->next_tb_to_parse_x,
              m_kernel_trace_info->next_tb_to_parse_y, m_kernel_trace_info->next_tb_to_parse_z, gpu, get_name(), static_trace_info);
  advance_trace_cta_id(m_kernel_trace_info);
}

types_of_operands get_oprnd_type(op_type op, special_ops sp_op){
  switch (op) {
    case SP_OP:
    case SFU_OP:
    case SPECIALIZED_UNIT_2_OP:
    case SPECIALIZED_UNIT_3_OP:
    case MISCELLANEOUS_NO_QUEUE_OP:
    case MISCELLANEOUS_QUEUE_OP:
    case TEXTURE_OP:
    case SURFACE_OP:
    case PREDICATE_OP:
    case TENSOR_CORE_OP:
    case HALF_OP:
    case DP_OP:
    case LOAD_OP:
    case STORE_OP:
      return FP_OP;
    case BRANCH_OP:
    case UNIFORM_OP:
    case INTP_OP:
    case SPECIALIZED_UNIT_4_OP:
      return INT_OP;
    case ALU_OP:
      if ((sp_op == FP__OP) || (sp_op == TEX__OP) || (sp_op == OTHER_OP))
        return FP_OP;
      else if (sp_op == INT__OP)
        return INT_OP;
    default: 
      return UN_OP;
  }
}

bool trace_warp_inst_t::parse_from_trace_struct(
    inst_trace_t &trace,
    const std::unordered_map<std::string, OpcodeChar> *OpcodeMap,
    const class trace_config *tconfig,
    const class kernel_trace_t *kernel_trace_info,
    traced_execution &static_trace_info) {
  // fill the inst_t and warp_inst_t params

  // fill active mask
  active_mask_t active_mask = trace.mask;
  unsigned int config_warp_size = 32;
  if(m_config != nullptr) {
    config_warp_size = m_config->warp_size;
  }

  set_active(active_mask, config_warp_size);

  // fill and initialize common params
  m_has_the_instruction_been_traced = trace.is_instruction_traced;
  m_decoded = true;
  pc = (address_type)trace.m_pc;
  next_traced_pc = (address_type)trace.m_next_traced_pc;
  unique_function_id = trace.m_unique_function_id;

  isize =
      16;  // starting from MAXWELL isize=16 bytes (including the control bytes)
  for (unsigned i = 0; i < MAX_OUTPUT_VALUES; i++) {
    out[i] = 0;
    vpreg_virtual_out[i] = 0; // MOD. VPREG
    vpreg_physical_out[i] = 0; // MOD. VPREG
  }
  for (unsigned i = 0; i < MAX_INPUT_VALUES; i++) {
    in[i] = 0;
    vpreg_virtual_in[i] = 0; // MOD. VPREG
    vpreg_physical_in[i] = 0; // MOD. VPREG
  }

  is_vectorin = 0;
  is_vectorout = 0;
  ar1 = 0;
  ar2 = 0;
  memory_op = no_memory_op;
  data_size = 0;
  op = ALU_OP;
  sp_op = OTHER_OP;
  mem_op = NOT_TEX;
  const_cache_operand = 0;
  oprnd_type = UN_OP;

  // MOD. Begin. VPREG
  vpreg_virtual_ar1 = 0;
  vpreg_virtual_ar2 = 0;
  vpreg_physical_ar1 = 0;
  vpreg_physical_ar2 = 0;
  // MOD. End. VPREG


  // get the opcode
  std::vector<std::string> opcode_tokens = get_opcode_tokens(trace.opcode);
  std::string opcode1 = opcode_tokens[0];

  std::unordered_map<std::string, OpcodeChar>::const_iterator it =
      OpcodeMap->find(opcode1);
  if (it != OpcodeMap->end()) {
    m_opcode = it->second.opcode;
    op = (op_type)(it->second.opcode_category);
    const std::unordered_map<unsigned, unsigned> *OpcPowerMap = &OpcodePowerMap;
    std::unordered_map<unsigned, unsigned>::const_iterator it2 =
      OpcPowerMap->find(m_opcode);
    if(it2 != OpcPowerMap->end()) {
      sp_op = (special_ops) (it2->second);
    }
    oprnd_type = get_oprnd_type(op, sp_op);
  } else {
    std::cout << "ERROR:  undefined instruction : " << trace.opcode
              << " Opcode: " << opcode1 << std::endl;
    assert(0 && "undefined instruction");
  }
  std::string opcode = trace.opcode;
  if(opcode1 == "MUFU"){ // Differentiate between different MUFU operations for power model
    if ((opcode == "MUFU.SIN") || (opcode == "MUFU.COS"))
      sp_op = FP_SIN_OP;
    if ((opcode == "MUFU.EX2") || (opcode == "MUFU.RCP"))
      sp_op = FP_EXP_OP;
    if (opcode == "MUFU.RSQ") 
      sp_op = FP_SQRT_OP;
    if (opcode == "MUFU.LG2") 
      sp_op = FP_LG_OP;
  }

  if(opcode1 == "IMAD"){ // Differentiate between different IMAD operations for power model
    if ((opcode == "IMAD.MOV") || (opcode == "IMAD.IADD"))
      sp_op = INT__OP;
  }
  
  // fill regs information
  num_regs = 0;
  num_operands = num_regs;
  outcount = 0;

  incount = 0;
  pred = 0; // Fix not determinism

  // fill latency and initl
  if(tconfig != nullptr) {
    tconfig->set_latency(op, latency, initiation_interval);
  }

  if( ((m_opcode == OP_LDC) || (m_opcode == OP_ULDC)) && !trace.is_constant_addr_already_calculated) {
    traced_operand &op_c = static_trace_info.get_kernel_by_unique_function_id(unique_function_id).get_instruction(pc).get_operand(1);
    if(trace.memadd_info.empty()) {
      trace.memadd_info.resize(1);
      trace.memadd_info[0] = std::make_unique<inst_memadd_info_t>();
      trace.memadd_info[0]->width = trace.get_datawidth_from_opcode(opcode_tokens);
      uint64_t constant_address = calculate_constant_address(0, op_c);
      for(unsigned int i = 0; i < config_warp_size; i++) {
        trace.memadd_info[0]->addrs[i] = constant_address;
      }
    }else {
      for(unsigned int i = 0; i < config_warp_size; i++) {
        if(active_mask.test(i)) {
          trace.memadd_info[0]->addrs[i] = calculate_constant_address(trace.memadd_info[0]->addrs[i], op_c);
        }else {
          trace.memadd_info[0]->addrs[i] = 0;
        }
      }
    }
    trace.is_constant_addr_already_calculated = true;
    m_has_the_constant_addr_already_calculated = true;
  }
  // fill addresses
  if(!trace.memadd_info.empty()) {
    data_size = trace.memadd_info[0]->width;
    for (unsigned i = 0; i < config_warp_size; ++i) {
      set_addr(i, trace.memadd_info[0]->addrs[i]);
      if(trace.memadd_info.size() == 2){
        set_addr_memref2(i, trace.memadd_info[1]->addrs[i]);
      }
    }
  }


  // handle special cases and fill memory space
  switch (m_opcode) {
    case OP_LDC: //handle Load from Constant
      memory_op = memory_load;
      const_cache_operand = 1;
      space.set_type(const_space);
      cache_op = CACHE_ALL;
      break;
    case OP_LDG:
    // LDGSTS is loading the values needed directly from the global memory to shared memory.
    // Before this feature, the values need to be loaded to registers first, then store to 
    // the shared memory.
    case OP_LDGSTS: // Add for memcpy_async
    case OP_LDL:
      assert(data_size > 0);
      memory_op = memory_load;
      cache_op = CACHE_ALL;
      if (m_opcode == OP_LDL)
        space.set_type(local_space);
      else
        space.set_type(global_space);
      // check the cache scope, if its strong GPU, then bypass L1
      // Add for LDGSTS instruction
      if (m_opcode == OP_LDGSTS) {
        m_is_ldgsts = true;
        if (trace.check_opcode_contain(opcode_tokens, "BYPASS")) {
          cache_op = CACHE_GLOBAL;
        }
      }
      if ( (trace.check_opcode_contain(opcode_tokens, "STRONG") &&
          trace.check_opcode_contain(opcode_tokens, "GPU")) ||
          (trace.check_opcode_contain(opcode_tokens, "STRONG") &&
          trace.check_opcode_contain(opcode_tokens, "SYS")) ){
        cache_op = CACHE_GLOBAL;
      }
      break;
    case OP_RED:
    case OP_REDG:
    case OP_STG:
    case OP_STL:
      assert(data_size > 0);
      memory_op = memory_store;
      cache_op = CACHE_ALL;
      if ( (trace.check_opcode_contain(opcode_tokens, "STRONG") &&
          trace.check_opcode_contain(opcode_tokens, "GPU")) ||
          (trace.check_opcode_contain(opcode_tokens, "STRONG") &&
          trace.check_opcode_contain(opcode_tokens, "SYS")) ){
        cache_op = CACHE_GLOBAL;
      }
      if (m_opcode == OP_STL) {
        space.set_type(local_space);
      }else {
        space.set_type(global_space);
      }
      if((m_opcode == OP_RED) || (m_opcode == OP_REDG)) {
        m_isatomic = true;
      }
      break;
    case OP_ATOMG:
    case OP_ATOM:
      assert(data_size > 0);
      memory_op = memory_load;
      op = LOAD_OP;
      space.set_type(global_space);
      m_isatomic = true;
      cache_op = CACHE_GLOBAL;  // all the atomics should be done at L2
      break;
    case OP_LDS:
      assert(data_size > 0);
      memory_op = memory_load;
      space.set_type(shared_space);
      break;
    case OP_STS:
      assert(data_size > 0);
      memory_op = memory_store;
      space.set_type(shared_space);
      break;
    case OP_ATOMS:
      assert(data_size > 0);
      m_isatomic = true;
      memory_op = memory_load;
      space.set_type(shared_space);
      break;
    case OP_LDSM:
      assert(data_size > 0);
      space.set_type(shared_space);
      break;
    case OP_ST:
    case OP_LD:
      assert(data_size > 0);
      if (m_opcode == OP_LD)
        memory_op = memory_load;
      else
        memory_op = memory_store;
      // resolve generic loads
      if (kernel_trace_info->shmem_base_addr == 0 ||
          kernel_trace_info->local_base_addr == 0) {
        // shmem and local addresses are not set
        // assume all the mem reqs are shared by default
        space.set_type(shared_space);
      } else {
        // check the first active address
        for (unsigned i = 0; i < config_warp_size; ++i)
          if (active_mask.test(i)) {
            if (trace.memadd_info[0]->addrs[i] >=
                    kernel_trace_info->shmem_base_addr &&
                trace.memadd_info[0]->addrs[i] <
                    kernel_trace_info->local_base_addr)
              space.set_type(shared_space);
            else if (trace.memadd_info[0]->addrs[i] >=
                         kernel_trace_info->local_base_addr &&
                     trace.memadd_info[0]->addrs[i] <
                         kernel_trace_info->local_base_addr +
                             LOCAL_MEM_SIZE_MAX) {
              space.set_type(local_space);
              cache_op = CACHE_ALL;
            } else {
              space.set_type(global_space);
              cache_op = CACHE_ALL;
            }
            break;
          }
      }

      break;
    case OP_CGAERRBAR:
    case OP_MEMBAR:
    case OP_BAR:
      // TO DO: fill this correctly
      bar_id = 0;
      bar_count = (unsigned)-1;
      bar_type = SYNC;
      // TO DO
      // if bar_type = RED;
      // set bar_type
      // barrier_type bar_type;
      // reduction_type red_type;
      break;
    case OP_HADD2:
    case OP_HADD2_32I:
    case OP_HFMA2:
    case OP_HFMA2_32I:
    case OP_HMUL2_32I:
    case OP_HSET2:
    case OP_HSETP2:
      initiation_interval =
          initiation_interval / 2;  // FP16 has 2X throughput than FP32
      break;
    default:
      break;
  }

  // MOD. Begin. MOD. VPREG. MOD. Improving branch behavior in traces
  if (op == BRANCH_OP) {
    switch (m_opcode) {
      case OP_WARPSYNC:
        control_flow_type = IS_WARPSYNC;
        break;
      case OP_RPCMOV:
        control_flow_type = IS_RPCMOV;
        break;
      case OP_BSYNC:
        control_flow_type = IS_BSYNC;
        break; 
      case OP_YIELD:
        control_flow_type = IS_YIELD;
        break;
      case OP_BRX:
      case OP_BRXU:
      case OP_BRA:
        control_flow_type = IS_BRANCH;
        break;
      case OP_JMX:
      case OP_JMXU:
      case OP_JMP:
        control_flow_type = IS_JUMP;
        break; 
      case OP_EXIT:
        control_flow_type = IS_ENDCALL;
        break;
      default:
        control_flow_type = NOT_DEFINED;
        break;
    }
  }
  // MOD. End. MOD. VPREG. MOD. Improving branch behavior in traces

  return true;
}

trace_config::trace_config() {}

void trace_config::reg_options(option_parser_t opp) {
  option_parser_register(opp, "-trace", OPT_CSTR, &g_traces_filename,
                         "traces kernel file"
                         "traces kernel file directory",
                         "./traces/kernelslist.g");

  option_parser_register(opp, "-trace_opcode_latency_initiation_int", OPT_CSTR,
                         &trace_opcode_latency_initiation_int,
                         "Opcode latencies and initiation for integers in "
                         "trace driven mode <latency,initiation>",
                         "4,1");
  option_parser_register(opp, "-trace_opcode_latency_initiation_sp", OPT_CSTR,
                         &trace_opcode_latency_initiation_sp,
                         "Opcode latencies and initiation for sp in trace "
                         "driven mode <latency,initiation>",
                         "4,1");
  option_parser_register(opp, "-trace_opcode_latency_initiation_dp", OPT_CSTR,
                         &trace_opcode_latency_initiation_dp,
                         "Opcode latencies and initiation for dp in trace "
                         "driven mode <latency,initiation>",
                         "4,1");
  option_parser_register(opp, "-trace_opcode_latency_initiation_sfu", OPT_CSTR,
                         &trace_opcode_latency_initiation_sfu,
                         "Opcode latencies and initiation for sfu in trace "
                         "driven mode <latency,initiation>",
                         "4,1");
  option_parser_register(opp, "-trace_opcode_latency_initiation_tensor",
                         OPT_CSTR, &trace_opcode_latency_initiation_tensor,
                         "Opcode latencies and initiation for tensor in trace "
                         "driven mode <latency,initiation>",
                         "4,1");

  option_parser_register(opp, "-trace_opcode_latency_initiation_branch",
                         OPT_CSTR, &trace_opcode_latency_initiation_branch,
                         "Opcode latencies and initiation for branch in trace "
                         "driven mode <latency,initiation>",
                         "1,1");
  option_parser_register(opp, "-trace_opcode_latency_initiation_half",
                         OPT_CSTR, &trace_opcode_latency_initiation_half,
                         "Opcode latencies and initiation for half in trace "
                         "driven mode <latency,initiation>",
                         "6,2");
  option_parser_register(opp, "-trace_opcode_latency_initiation_uniform",
                         OPT_CSTR, &trace_opcode_latency_initiation_uniform,
                         "Opcode latencies and initiation for uniform in trace "
                         "driven mode <latency,initiation>",
                         "2,2");
  option_parser_register(opp, "-trace_opcode_latency_initiation_predicate",
                         OPT_CSTR, &trace_opcode_latency_initiation_predicate,
                         "Opcode latencies and initiation for predicate in trace "
                         "driven mode <latency,initiation>",
                         "2,2");
  option_parser_register(opp, "-trace_opcode_latency_initiation_miscellaneous_queue",
                         OPT_CSTR, &trace_opcode_latency_initiation_miscellaneous_queue,
                         "Opcode latencies and initiation for miscellaneous queue in trace "
                         "driven mode <latency,initiation>",
                         "2,2");
  option_parser_register(opp, "-trace_opcode_latency_initiation_miscellaneous_no_queue",
                         OPT_CSTR, &trace_opcode_latency_initiation_miscellaneous_no_queue,
                         "Opcode latencies and initiation for miscellaneous no queue in trace "
                         "driven mode <latency,initiation>",
                         "1,1");

  for (unsigned j = 0; j < SPECIALIZED_UNIT_NUM; ++j) {
    std::stringstream ss;
    ss << "-trace_opcode_latency_initiation_spec_op_" << j + 1;
    option_parser_register(opp, ss.str().c_str(), OPT_CSTR,
                           &trace_opcode_latency_initiation_specialized_op[j],
                           "specialized unit config"
                           " <latency,initiation>",
                           "4,4");
  }

  // MOD. Begin. Improved tracer
  option_parser_register(opp, "-is_extra_traces_enabled", OPT_BOOL,
                         &is_extra_traces_enabled,
                         "If enabled, the simulator will use an extra file (json) which has useful information for the simulation (.gz)."
                         "is_extra_traces_enabled (default = disabled)",
                         "0");
  // MOD. End. Improved tracer.
}

void trace_config::parse_config() {
  sscanf(trace_opcode_latency_initiation_int, "%u,%u", &int_latency, &int_init);
  sscanf(trace_opcode_latency_initiation_sp, "%u,%u", &fp_latency, &fp_init);
  sscanf(trace_opcode_latency_initiation_dp, "%u,%u", &dp_latency, &dp_init);
  sscanf(trace_opcode_latency_initiation_sfu, "%u,%u", &sfu_latency, &sfu_init);
  sscanf(trace_opcode_latency_initiation_tensor, "%u,%u", &tensor_latency,
         &tensor_init);

  sscanf(trace_opcode_latency_initiation_branch, "%u,%u", &branch_latency, &branch_init);
  sscanf(trace_opcode_latency_initiation_half, "%u,%u", &half_latency, &half_init);
  sscanf(trace_opcode_latency_initiation_uniform, "%u,%u", &uniform_latency, &uniform_init);
  sscanf(trace_opcode_latency_initiation_predicate, "%u,%u", &predicate_latency, &predicate_init);
  sscanf(trace_opcode_latency_initiation_miscellaneous_queue, "%u,%u", &miscellaneous_queue_latency, &miscellaneous_queue_init);
  sscanf(trace_opcode_latency_initiation_miscellaneous_no_queue, "%u,%u", &miscellaneous_no_queue_latency, &miscellaneous_no_queue_init);

  for (unsigned j = 0; j < SPECIALIZED_UNIT_NUM; ++j) {
    sscanf(trace_opcode_latency_initiation_specialized_op[j], "%u,%u",
           &specialized_unit_latency[j], &specialized_unit_initiation[j]);
  }
}
void trace_config::set_latency(unsigned category, unsigned &latency,
                               unsigned &initiation_interval) const {
  initiation_interval = latency = 1;

  switch (category) {
    case ALU_OP:
    case INTP_OP:
    case CALL_OPS:
    case RET_OPS:
      latency = int_latency;
      initiation_interval = int_init;
      break;
    case SP_OP:
      latency = fp_latency;
      initiation_interval = fp_init;
      break;
    case DP_OP:
      latency = dp_latency;
      initiation_interval = dp_init;
      break;
    case SFU_OP:
      latency = sfu_latency;
      initiation_interval = sfu_init;
      break;
    case TENSOR_CORE_OP:
      latency = tensor_latency;
      initiation_interval = tensor_init;
      break;
    case MISCELLANEOUS_QUEUE_OP:
      latency = miscellaneous_queue_latency;
      initiation_interval = miscellaneous_queue_init;
      break;
    case MISCELLANEOUS_NO_QUEUE_OP:
      latency = miscellaneous_no_queue_latency;
      initiation_interval = miscellaneous_no_queue_init;
      break;
    case BRANCH_OP:
      latency = branch_latency;
      initiation_interval = branch_init;
      break;
    case HALF_OP:
      latency = half_latency;
      initiation_interval = half_init;
      break;
    case UNIFORM_OP:
      latency = uniform_latency;
      initiation_interval = uniform_init;
      break;
    case PREDICATE_OP:
      latency = predicate_latency;
      initiation_interval = predicate_init;
      break;
    default:
      break;
  }
  // for specialized units
  if (category >= SPEC_UNIT_START_ID) {
    unsigned spec_id = category - SPEC_UNIT_START_ID;
    assert(spec_id >= 0 && spec_id < SPECIALIZED_UNIT_NUM);
    latency = specialized_unit_latency[spec_id];
    initiation_interval = specialized_unit_initiation[spec_id];
  }
}

void trace_gpgpu_sim::createSIMTCluster() {
  m_cluster = new simt_core_cluster *[m_shader_config->n_simt_clusters];
  for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++)
    m_cluster[i] =
        new trace_simt_core_cluster(this, i, m_shader_config, m_memory_config,
                                    m_shader_stats, m_memory_stats);
}

void trace_simt_core_cluster::create_shader_core_ctx() {
  m_core.resize(m_config->n_simt_cores_per_cluster);
  for (unsigned i = 0; i < m_config->n_simt_cores_per_cluster; i++) {
    unsigned sid = m_config->cid_to_sid(i, m_cluster_id);
    if(m_config->is_SM_remodeling_enabled) {
      m_core[i] = new SM(m_config->num_subcores_in_SM, m_gpu, this, sid, m_cluster_id,
                                          m_config, m_mem_config, m_stats);
      m_core[i]->init();
    }else {
      m_core[i] = new trace_shader_core_ctx(m_gpu, this, sid, m_cluster_id,
                                          m_config, m_mem_config, m_stats);
    }
    m_core[i]->create_gpu_per_sm_stats(m_gpu->m_gpu_per_sm_stats);
    m_core_sim_order.push_back(i);
  }
}

void trace_shader_core_ctx::create_shd_warp() {
  m_warp.resize(m_config->max_warps_per_shader);
  for (unsigned k = 0; k < m_config->max_warps_per_shader; ++k) {
    m_warp[k] = new trace_shd_warp_t(this, m_config->warp_size, m_stats);
  }
}

void trace_shader_core_ctx::get_pdom_stack_top_info(unsigned warp_id,
                                                    const warp_inst_t *pI,
                                                    unsigned *pc,
                                                    unsigned *rpc) {
  // In trace-driven mode, we assume no control hazard
  *pc = pI->pc;
  *rpc = pI->pc;
}

const active_mask_t &trace_shader_core_ctx::get_active_mask(
    unsigned warp_id, const warp_inst_t *pI) {
  // For Trace-driven, the active mask already set in traces, so
  // just read it from the inst
  return pI->get_active_mask();
}

unsigned trace_shader_core_ctx::sim_init_thread(
    kernel_info_t &kernel, ptx_thread_info **thread_info, int sid, unsigned tid,
    unsigned threads_left, unsigned num_threads, core_t *core,
    unsigned hw_cta_id, unsigned hw_warp_id, gpgpu_t *gpu) {
  if (kernel.no_more_ctas_to_run()) {
    return 0;  // finished!
  }

  if (kernel.more_threads_in_cta()) {
    kernel.increment_thread_id();
  }

  if (!kernel.more_threads_in_cta()) kernel.increment_cta_id();

  return 1;
}

void trace_shader_core_ctx::init_warps(unsigned cta_id, unsigned start_thread,
                                       unsigned end_thread, unsigned ctaid,
                                       int cta_size, kernel_info_t &kernel) {
  // call base class
  shader_core_ctx::init_warps(cta_id, start_thread, end_thread, ctaid, cta_size,
                              kernel);

  // then init traces
  unsigned start_warp = start_thread / m_config->warp_size;
  unsigned end_warp = end_thread / m_config->warp_size +
                      ((end_thread % m_config->warp_size) ? 1 : 0);

  init_traces(start_warp, end_warp, kernel);
}

warp_inst_t *trace_shader_core_ctx::get_next_inst(unsigned warp_id, // MOD. VPREG
                                                        address_type pc) {
  // read the inst from the traces
  trace_shd_warp_t *m_trace_warp =
      static_cast<trace_shd_warp_t *>(m_warp[warp_id]);
  return m_trace_warp->get_next_trace_inst(pc);
}

// MOD. Begin. VPREG
void trace_shader_core_ctx::decrement_trace_pc(unsigned warp_id) { 
  // read the inst from the traces
  trace_shd_warp_t *m_trace_warp =
      static_cast<trace_shd_warp_t *>(m_warp[warp_id]);
  m_trace_warp->decrement_trace_pc();
}
// MOD. End. VPREG

void trace_shader_core_ctx::updateSIMTStack(unsigned warpId,
                                            warp_inst_t *inst, ib_ooo_simt_info *ib_ooo_simt_status) { // MOD. IBuffer_ooo
  // No SIMT-stack in trace-driven  mode
}

void trace_shader_core_ctx::init_traces(unsigned start_warp, unsigned end_warp,
                                        kernel_info_t &kernel) {
  std::vector<std::map<address_type, traced_instructions_by_pc> *> threadblock_traces;
  std::vector<std::vector<address_type> *> threadblock_traced_pcs;
  for (unsigned i = start_warp; i < end_warp; ++i) {
    trace_shd_warp_t *m_trace_warp = static_cast<trace_shd_warp_t *>(m_warp[i]);
    m_trace_warp->clear();
    threadblock_traces.push_back(&(m_trace_warp->map_warp_traces));
    threadblock_traced_pcs.push_back(&(m_trace_warp->traced_pcs));
  }
  trace_kernel_info_t &trace_kernel =
      static_cast<trace_kernel_info_t &>(kernel);
  trace_kernel.get_next_threadblock_traces(threadblock_traces, threadblock_traced_pcs, m_gpu, m_gpu->get_extra_trace_info());

  // set the pc from the traces and ignore the functional model
  for (unsigned i = start_warp; i < end_warp; ++i) {
    trace_shd_warp_t *m_trace_warp = static_cast<trace_shd_warp_t *>(m_warp[i]);
    m_trace_warp->set_next_pc(m_trace_warp->get_start_trace_pc());
    m_trace_warp->set_kernel(&trace_kernel);
  }
}

void trace_shader_core_ctx::checkExecutionStatusAndUpdate(warp_inst_t &inst,
                                                          unsigned t,
                                                          unsigned tid) {
  if (inst.isatomic()) m_warp[inst.warp_id()]->inc_n_atomic();

  if (inst.space.is_local() && (inst.is_load() || inst.is_store())) {
    new_addr_type localaddrs[MAX_ACCESSES_PER_INSN_PER_THREAD];
    unsigned num_addrs;
    num_addrs = translate_local_memaddr(
        inst.get_addr(t), tid,
        m_config->n_simt_clusters * m_config->n_simt_cores_per_cluster,
        inst.data_size, (new_addr_type *)localaddrs);
    inst.set_addr(t, (new_addr_type *)localaddrs, num_addrs);
  }

  if (inst.op == EXIT_OPS) {
    m_warp[inst.warp_id()]->set_completed(t);
  }
}

void trace_shader_core_ctx::func_exec_inst(warp_inst_t &inst) {

}

void trace_shader_core_ctx::issue_warp(register_set &warp,
                                       const warp_inst_t *pI,
                                       const active_mask_t &active_mask,
                                       unsigned warp_id, unsigned sch_id) {
  shader_core_ctx::issue_warp(warp, pI, active_mask, warp_id, sch_id);

  // delete warp_inst_t class here, it is not required anymore by gpgpu-sim
  // after issue
    delete pI;
}
