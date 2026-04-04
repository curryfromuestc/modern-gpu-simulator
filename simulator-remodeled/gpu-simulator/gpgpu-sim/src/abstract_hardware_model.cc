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

// Copyright (c) 2009-2021, Tor M. Aamodt, Inderpreet Singh, Timothy Rogers, Vijay Kandiah, Nikos Hardavellas, 
// Mahmoud Khairy, Junrui Pan, Timothy G. Rogers
// The University of British Columbia, Northwestern University, Purdue University
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
// 3. Neither the names of The University of British Columbia, Northwestern 
//    University nor the names of their contributors may be used to
//    endorse or promote products derived from this software without specific
//    prior written permission.
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


#include "abstract_hardware_model.h"
#include <sys/stat.h>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <regex>

#include "../libcuda/gpgpu_context.h"
#include "cuda-sim/cuda-sim.h"
#include "cuda-sim/memory.h"
#include "cuda-sim/ptx-stats.h"
#include "cuda-sim/ptx_ir.h"
#include "gpgpu-sim/gpu-sim.h"
#include "gpgpusim_entrypoint.h"
#include "option_parser.h"
#include "gpgpu-sim/gpu-cache.h" // MOD. Fixed LDST_Unit model
#include "../../trace-driven/trace_driven.h"
#include "../../../util/traces_enhanced/src/string_utilities.h"
#include "gpgpu-sim/remodeling/register_file.h"

void mem_access_t::init(gpgpu_context *ctx) {
  gpgpu_ctx = ctx;
  m_uid = ++(gpgpu_ctx->sm_next_access_uid);
  m_addr = 0;
  m_req_size = 0;
}
void warp_inst_t::issue(const active_mask_t &mask, unsigned warp_id,
                        unsigned long long cycle, int dynamic_warp_id,
                        int sch_id) {
  m_warp_active_mask = mask;
  m_warp_issued_mask = mask;
  m_uid = ++(m_config->gpgpu_ctx->warp_inst_sm_next_uid);
  m_warp_id = warp_id;
  m_dynamic_warp_id = dynamic_warp_id;
  issue_cycle = cycle;
  cycles = initiation_interval;
  m_cache_hit = false;
  m_empty = false;
  m_scheduler_id = sch_id;
}

void warp_inst_t::set_some_warp_attributes(unsigned int warp_id, unsigned int dynamic_warp_id) {
  m_warp_id = warp_id;
  m_dynamic_warp_id = dynamic_warp_id;
  m_empty = false;
}

checkpoint::checkpoint() {
  struct stat st = {0};

  if (stat("checkpoint_files", &st) == -1) {
    mkdir("checkpoint_files", 0777);
  }
}
void checkpoint::load_global_mem(class memory_space *temp_mem, char *f1name) {
  FILE *fp2 = fopen(f1name, "r");
  assert(fp2 != NULL);
  char line[128]; /* or other suitable maximum line size */
  unsigned int offset = 0;
  while (fgets(line, sizeof line, fp2) != NULL) /* read a line */
  {
    unsigned int index;
    char *pch;
    pch = strtok(line, " ");
    if (pch[0] == 'g' || pch[0] == 's' || pch[0] == 'l') {
      pch = strtok(NULL, " ");

      std::stringstream ss;
      ss << std::hex << pch;
      ss >> index;

      offset = 0;
    } else {
      unsigned int data;
      std::stringstream ss;
      ss << std::hex << pch;
      ss >> data;
      temp_mem->write_only(offset, index, 4, &data);
      offset = offset + 4;
    }
    // fputs ( line, stdout ); /* write the line */
  }
  fclose(fp2);
}

void checkpoint::store_global_mem(class memory_space *mem, char *fname,
                                  char *format) {
  FILE *fp3 = fopen(fname, "w");
  assert(fp3 != NULL);
  mem->print(format, fp3);
  fclose(fp3);
}

void move_warp(warp_inst_t *&dst, warp_inst_t *&src) {
  assert(dst->empty());
  warp_inst_t *temp = dst;
  dst = src;
  src = temp;
  src->clear();
}

void gpgpu_functional_sim_config::reg_options(class OptionParser *opp) {
  option_parser_register(opp, "-gpgpu_ptx_use_cuobjdump", OPT_BOOL,
                         &m_ptx_use_cuobjdump,
                         "Use cuobjdump to extract ptx and sass from binaries",
#if (CUDART_VERSION >= 4000)
                         "1"
#else
                         "0"
#endif
  );
  option_parser_register(opp, "-gpgpu_experimental_lib_support", OPT_BOOL,
                         &m_experimental_lib_support,
                         "Try to extract code from cuda libraries [Broken "
                         "because of unknown cudaGetExportTable]",
                         "0");
  option_parser_register(opp, "-checkpoint_option", OPT_INT32,
                         &checkpoint_option,
                         " checkpointing flag (0 = no checkpoint)", "0");
  option_parser_register(
      opp, "-checkpoint_kernel", OPT_INT32, &checkpoint_kernel,
      " checkpointing during execution of which kernel (1- 1st kernel)", "1");
  option_parser_register(
      opp, "-checkpoint_CTA", OPT_INT32, &checkpoint_CTA,
      " checkpointing after # of CTA (< less than total CTA)", "0");
  option_parser_register(opp, "-resume_option", OPT_INT32, &resume_option,
                         " resume flag (0 = no resume)", "0");
  option_parser_register(opp, "-resume_kernel", OPT_INT32, &resume_kernel,
                         " Resume from which kernel (1= 1st kernel)", "0");
  option_parser_register(opp, "-resume_CTA", OPT_INT32, &resume_CTA,
                         " resume from which CTA ", "0");
  option_parser_register(opp, "-checkpoint_CTA_t", OPT_INT32, &checkpoint_CTA_t,
                         " resume from which CTA ", "0");
  option_parser_register(opp, "-checkpoint_insn_Y", OPT_INT32,
                         &checkpoint_insn_Y, " resume from which CTA ", "0");

  option_parser_register(
      opp, "-gpgpu_ptx_convert_to_ptxplus", OPT_BOOL, &m_ptx_convert_to_ptxplus,
      "Convert SASS (native ISA) to ptxplus and run ptxplus", "0");
  option_parser_register(opp, "-gpgpu_ptx_force_max_capability", OPT_UINT32,
                         &m_ptx_force_max_capability,
                         "Force maximum compute capability", "0");
  option_parser_register(
      opp, "-gpgpu_ptx_inst_debug_to_file", OPT_BOOL, &g_ptx_inst_debug_to_file,
      "Dump executed instructions' debug information to file", "0");
  option_parser_register(
      opp, "-gpgpu_ptx_inst_debug_file", OPT_CSTR, &g_ptx_inst_debug_file,
      "Executed instructions' debug output file", "inst_debug.txt");
  option_parser_register(opp, "-gpgpu_ptx_inst_debug_thread_uid", OPT_INT32,
                         &g_ptx_inst_debug_thread_uid,
                         "Thread UID for executed instructions' debug output",
                         "1");
}

void gpgpu_functional_sim_config::ptx_set_tex_cache_linesize(
    unsigned linesize) {
  m_texcache_linesize = linesize;
}

gpgpu_t::gpgpu_t(const gpgpu_functional_sim_config &config, gpgpu_context *ctx)
    : m_function_model_config(config) {
  gpgpu_ctx = ctx;
  m_global_mem = new memory_space_impl<8192>("global", 64 * 1024);
  m_tex_mem = new memory_space_impl<8192>("tex", 64 * 1024);
  m_surf_mem = new memory_space_impl<8192>("surf", 64 * 1024);

  m_dev_malloc = GLOBAL_HEAP_START;
  checkpoint_option = m_function_model_config.get_checkpoint_option();
  checkpoint_kernel = m_function_model_config.get_checkpoint_kernel();
  checkpoint_CTA = m_function_model_config.get_checkpoint_CTA();
  resume_option = m_function_model_config.get_resume_option();
  resume_kernel = m_function_model_config.get_resume_kernel();
  resume_CTA = m_function_model_config.get_resume_CTA();
  checkpoint_CTA_t = m_function_model_config.get_checkpoint_CTA_t();
  checkpoint_insn_Y = m_function_model_config.get_checkpoint_insn_Y();

  // initialize texture mappings to empty
  m_NameToTextureInfo.clear();
  m_NameToCudaArray.clear();
  m_TextureRefToName.clear();
  m_NameToAttribute.clear();

  if (m_function_model_config.get_ptx_inst_debug_to_file() != 0)
    ptx_inst_debug_file =
        fopen(m_function_model_config.get_ptx_inst_debug_file(), "w");

  gpu_sim_cycle = 0;
  gpu_tot_sim_cycle = 0;

  dram_sim_cycle = 0;
  dram_tot_sim_cycle = 0;
}

gpgpu_t::~gpgpu_t() {
  delete m_global_mem;
  delete m_tex_mem;
  delete m_surf_mem;
}

new_addr_type line_size_based_tag_func(new_addr_type address,
                                       new_addr_type line_size) {
  // gives the tag for an address based on a given line size
  return address & ~(line_size - 1);
}

const char *mem_access_type_str(enum mem_access_type access_type) {
#define MA_TUP_BEGIN(X) static const char *access_type_str[] = {
#define MA_TUP(X) #X
#define MA_TUP_END(X) \
  }                   \
  ;
  MEM_ACCESS_TYPE_TUP_DEF
#undef MA_TUP_BEGIN
#undef MA_TUP
#undef MA_TUP_END

  assert(access_type < NUM_MEM_ACCESS_TYPE);

  return access_type_str[access_type];
}

void warp_inst_t::clear_active(const active_mask_t &inactive) {
  active_mask_t test = m_warp_active_mask;
  test &= inactive;
  assert(test == inactive);  // verify threads being disabled were active
  m_warp_active_mask &= ~inactive;
}

void warp_inst_t::set_not_active(unsigned lane_id) {
  m_warp_active_mask.reset(lane_id);
}

void warp_inst_t::set_active(const active_mask_t &active, unsigned int warp_size) {
  m_warp_active_mask = active;
  if (m_isatomic) {
    for (unsigned i = 0; i < warp_size; i++) {
      if (!m_warp_active_mask.test(i)) {
        m_per_scalar_thread[i].callback.function = NULL;
        m_per_scalar_thread[i].callback.instruction = NULL;
        m_per_scalar_thread[i].callback.thread = NULL;
      }
    }
  }
}

void warp_inst_t::do_atomic(bool forceDo) {
  do_atomic(m_warp_active_mask, forceDo);
}

void warp_inst_t::do_atomic(const active_mask_t &access_mask, bool forceDo) {
  assert(m_isatomic && (!m_empty || forceDo));
  if (!should_do_atomic) return;
  for (unsigned i = 0; i < m_config->warp_size; i++) {
    if (access_mask.test(i)) {
      dram_callback_t &cb = m_per_scalar_thread[i].callback;
      if (cb.thread) cb.function(cb.instruction, cb.thread);
    }
  }
}

void warp_inst_t::broadcast_barrier_reduction(
    const active_mask_t &access_mask) {
  for (unsigned i = 0; i < m_config->warp_size; i++) {
    if (access_mask.test(i)) {
      dram_callback_t &cb = m_per_scalar_thread[i].callback;
      if (cb.thread) {
        cb.function(cb.instruction, cb.thread);
      }
    }
  }
}

// MOD. Begin. Fixed LDST_Unit model
std::vector<mem_access_t> warp_inst_t::granted_accesses(std::vector<bool> &used_banks, std::vector<std::deque<mem_fetch *>> &l1_latency_queue, unsigned int inst_latency, l1d_cache_config &L1D_config, unsigned int max_allowed_searches, bool &is_a_bank_conflict) {
  std::vector<mem_access_t> res;
  auto it = m_accessq.begin();
  unsigned int n_searches = 0;
  
  while((it != m_accessq.end()) && (n_searches <= max_allowed_searches)) {
    unsigned int acc_bank = L1D_config.set_bank(it->get_addr());
    if(!used_banks[acc_bank] && (l1_latency_queue[acc_bank][inst_latency - 1] == NULL)) { // && == NULL. PASAR L!D
      used_banks[acc_bank] = true;
      res.push_back(*it);
      it = m_accessq.erase(it);
    } else {
      is_a_bank_conflict = true;
      it++;
    }
    n_searches++;
  }
  return res;
}

void warp_inst_t::generate_fixed_latency_constant_accesses(new_addr_type c_addr) {
  mem_access_type access_type = CONST_ACC_R;
  bool is_write = is_store();
  new_addr_type cache_block_size = m_config->gpgpu_cache_constl1_linesize;
  if (cache_block_size) {
    mem_access_byte_mask_t byte_mask;
    std::map<new_addr_type, active_mask_t>
        accesses;  // block address -> set of thread offsets in warp
    std::map<new_addr_type, active_mask_t>::iterator a;
    for (unsigned thread = 0; thread < m_config->warp_size; thread++) {
      if (!active(thread)) continue;
      new_addr_type block_address =
          line_size_based_tag_func(c_addr, cache_block_size);
      accesses[block_address].set(thread);
      unsigned idx = c_addr - block_address;
      for (unsigned i = 0; i < data_size; i++) byte_mask.set(idx + i);
    }
    for (a = accesses.begin(); a != accesses.end(); ++a)
      m_accessq.push_back(mem_access_t(
          access_type, a->first, cache_block_size, is_write, a->second,
          byte_mask, mem_access_sector_mask_t(), m_config->gpgpu_ctx));
  }
}
// MOD. End. Fixed LDST_Unit model

void warp_inst_t::ldgsts_change_to_sts_mode(gpgpu_sim *gpu) {
    assert(m_is_ldgsts);
    assert(m_ldgsts_state == Ldgsts_State::LOAD_STAGE);
    space.set_type(shared_space);
    memory_op = memory_store;
    op = STORE_OP;
    assert(data_size> 0);
    m_per_scalar_thread_valid = m_per_scalar_thread_valid_memref2;
    assert(m_per_scalar_thread_memref2.size() == m_per_scalar_thread.size());
    for(unsigned int i = 0; i< m_per_scalar_thread_memref2.size(); i++) {
      m_per_scalar_thread[i] = m_per_scalar_thread_memref2[i];
    }
    m_mem_accesses_created = false;
    generate_mem_accesses();
    generate_mem_latencies(gpu);
  }

bool warp_inst_t::has_sm_shared_wb_finished() {
  assert(is_load() || is_dp_op());
  bool res = m_num_pending_cycles_to_finish_wb_from_sm_struct_to_subcore == 0;
  return res;
}

bool warp_inst_t::sm_shared_wb_consumed(bool can_do_wb_this_cycle, unsigned int num_cycles_to_transfer_reg, bool &conflict_detected) {
  bool wb_performed_this_cycle = false;
  assert(is_load() || is_dp_op());
  if(m_num_pending_cycles_to_finish_wb_from_sm_struct_to_subcore > 0) {
    bool decremented = false;
    if(can_do_wb_this_cycle) {
      m_num_pending_cycles_to_finish_wb_from_sm_struct_to_subcore--;
      decremented = true;
    }else if( (m_num_pending_cycles_to_finish_wb_from_sm_struct_to_subcore%num_cycles_to_transfer_reg) != 1) {
      m_num_pending_cycles_to_finish_wb_from_sm_struct_to_subcore--;
      decremented = true;
    }else {
      conflict_detected = true;
    }
    if(decremented && ( (m_num_pending_cycles_to_finish_wb_from_sm_struct_to_subcore % num_cycles_to_transfer_reg) == 0) ) {
      wb_performed_this_cycle = true;
      m_reg_offset = (m_num_pending_cycles_to_finish_wb_from_sm_struct_to_subcore == 0)  ? m_reg_offset : (m_reg_offset + 1);
    }
  }
  return wb_performed_this_cycle;
}

void warp_inst_t::get_tensor_core_instruction_info() {
  this->get_extra_trace_instruction_info().set_tensor_core_instruction_info();
}

void warp_inst_t::generate_tensor_core_latencies(gpgpu_sim *gpu) {
  assert(get_extra_trace_instruction_info().get_tensor_core_instruction_info().is_set);
  unsigned int number_of_elements = get_extra_trace_instruction_info().get_tensor_core_instruction_info().size_m * get_extra_trace_instruction_info().get_tensor_core_instruction_info().size_n * get_extra_trace_instruction_info().get_tensor_core_instruction_info().size_k;
  assert(number_of_elements > 0);
  assert(get_extra_trace_instruction_info().get_tensor_core_instruction_info().operand_bit_size > 0);
  const shader_core_config &shader_config = gpu->get_config().get_gpgpu_sim_config();
  unsigned int number_of_cycles = number_of_elements * get_extra_trace_instruction_info().get_tensor_core_instruction_info().operand_bit_size / shader_config.tensor_rate_per_cycle;
  if(get_extra_trace_instruction_info().get_tensor_core_instruction_info().is_sparse) {
    number_of_cycles = number_of_cycles / 2;
  } 
  initiation_interval = number_of_cycles / 2;
  latency = number_of_cycles - initiation_interval;
  if(get_extra_trace_instruction_info().get_tensor_core_instruction_info().is_16816_fp32_1688_fp32) {
    initiation_interval += gpu->get_config().get_gpgpu_sim_config().tensor_extra_latency_16816_fp32_1688_fp32;
    latency += gpu->get_config().get_gpgpu_sim_config().tensor_extra_latency_16816_fp32_1688_fp32;
  }
}

void warp_inst_t::assign_predicate_latencies_if_needed(gpgpu_sim *gpu) {
  const shader_core_config &shader_config = gpu->get_config().get_gpgpu_sim_config();
  // const trace_config &trace_config = gpu->ker
  const trace_config *trace_conf = gpu->gpgpu_ctx->the_gpgpusim->g_trace_config;
  if(op == op_type::PREDICATE_OP) {
    latency = trace_conf->get_int_latency();
    initiation_interval = trace_conf->get_int_init();
    latency_extra_predicate_op = shader_config.predicate_latency - latency - initiation_interval;
  }else if(m_extra_trace_instruction_info->get_contains_setp()) {
    latency_extra_predicate_op = shader_config.predicate_latency - latency - initiation_interval;
  }
}

void warp_inst_t::generate_miscellaneous_queue_latencies(gpgpu_sim *gpu) {
  m_num_cycles_per_intermediate_stage.resize(1);
  m_num_cycles_to_wait_to_free_WAR = 1;
}

void warp_inst_t::generate_texture_latencies(gpgpu_sim *gpu) {
  const shader_core_config &shader_config = gpu->get_config().get_gpgpu_sim_config();
  m_num_cycles_per_intermediate_stage.resize(shader_config.dp_shared_intermidiate_stages, 1);
  m_num_cycles_to_wait_to_free_WAR = shader_config.memory_intermidiate_stages_subcore_unit;
}

void warp_inst_t::generate_other_mem_ops_latencies(gpgpu_sim *gpu) {
  const shader_core_config &shader_config = gpu->get_config().get_gpgpu_sim_config();
  m_num_cycles_per_intermediate_stage.resize(shader_config.memory_intermidiate_stages_subcore_unit, 1);
  m_num_cycles_to_wait_to_free_WAR = shader_config.memory_intermidiate_stages_subcore_unit;
}

void warp_inst_t::generate_dp_latencies(gpgpu_sim *gpu) {
  const shader_core_config &shader_config = gpu->get_config().get_gpgpu_sim_config();
  m_num_cycles_per_intermediate_stage.resize(shader_config.dp_shared_intermidiate_stages, 1);
  unsigned int num_cycles_transfer_operands = 0;
  unsigned int first_read_operand = get_extra_trace_instruction_info().get_num_destination_registers();
  for(unsigned int i = first_read_operand; i < get_extra_trace_instruction_info().get_num_operands(); i++){
    if(get_extra_trace_instruction_info().get_operand(i).get_operand_type() == TraceEnhancedOperandType::REG) {
      num_cycles_transfer_operands += (warp_size()* 8) / (shader_config.memory_subcore_link_to_sm_byte_size / 2);
    }else {
      num_cycles_transfer_operands += 1;
    }
  }
  m_num_cycles_per_intermediate_stage[m_num_cycles_per_intermediate_stage.size() - 1] = 1 + num_cycles_transfer_operands;
  m_num_cycles_to_wait_to_free_WAR = shader_config.dp_shared_intermidiate_stages + num_cycles_transfer_operands - 2;
  if (shader_config.is_dp_pipeline_shared_for_subcores) {
    m_has_wb_from_sm_struct_to_subcore = true;
    unsigned int num_dsts = get_number_of_uses_per_operand(get_extra_trace_instruction_info(), get_extra_trace_instruction_info().get_operand(0).get_operand_reg_number(), 0, get_extra_trace_instruction_info().get_operand(0).get_operand_type());
    m_num_pending_cycles_to_finish_wb_from_sm_struct_to_subcore = shader_config.num_cycles_needed_to_write_a_reg_from_sm_struct_to_subcore * num_dsts;
  }
}

void warp_inst_t::generate_mem_latencies(gpgpu_sim *gpu) {
  assert(is_load() || is_store());
  const shader_core_config &shader_config = gpu->get_config().get_gpgpu_sim_config();
  assert(shader_config.is_trace_mode);
  m_num_cycles_per_intermediate_stage.resize(shader_config.memory_intermidiate_stages_subcore_unit, 0);
  bool is_shared = space.is_shared();
  bool is_consider_global = space.is_global() || space.is_local();
  unsigned int total_byte_size_for_warp = data_size * warp_size();
  unsigned int idx_of_memref = is_load() ? 1 : 0;
  bool is_regular_reg_in_mref =
        m_extra_trace_instruction_info
            ->is_first_operand_of_mref_cbank_desc_using_regular_reg(idx_of_memref);
  unsigned int standard_num_cycles_per_stage_in_subcore = shader_config.cycles_needed_for_address_calculation;
  unsigned int cycles_at_last_stage_in_subcore = 1;
  if(is_shared || !is_regular_reg_in_mref) {
    standard_num_cycles_per_stage_in_subcore = standard_num_cycles_per_stage_in_subcore / 2;
  }
  unsigned int cycles_at_first_stage = standard_num_cycles_per_stage_in_subcore;
  m_num_cycles_to_wait_to_free_WAR = 1;
  int extra_offset_store = 0;
  if(is_store()) {
    if(is_shared) {
      m_num_cycles_per_intermediate_stage[1] = 1;
    }else {
      
      if (!is_regular_reg_in_mref) {
        m_num_cycles_per_intermediate_stage[1] = 2;
      }else {
        cycles_at_first_stage = cycles_at_first_stage / 2;
      }
    }
    unsigned int link_transfer_size =
        shader_config
                .is_store_half_bandwidth_in_the_subcore_link_to_sm_enabled
            ? (shader_config.memory_subcore_link_to_sm_byte_size / 2)
            : shader_config.memory_subcore_link_to_sm_byte_size;
    cycles_at_last_stage_in_subcore = total_byte_size_for_warp / link_transfer_size;

    if(is_regular_reg_in_mref) {
      if(is_consider_global) {
        cycles_at_last_stage_in_subcore += 4;
      }else if(is_shared) {
        cycles_at_last_stage_in_subcore += 2;
      }
    }

    unsigned int num_to_substract_WAR = 0;
    unsigned int num_to_add_WAR = 0;
    if(is_shared) {
      num_to_substract_WAR = 2;
    }else if(is_consider_global) {
      num_to_substract_WAR = 3;
      num_to_add_WAR = standard_num_cycles_per_stage_in_subcore - 2;
    }
    m_num_cycles_to_wait_to_free_WAR += 1 + cycles_at_last_stage_in_subcore + num_to_add_WAR - num_to_substract_WAR;

  }else { // is_load
    if(!space.is_const() || (space.is_const() && is_regular_reg_in_mref)) {
      m_num_cycles_to_wait_to_free_WAR++;
    }
    if(is_consider_global) {
      cycles_at_last_stage_in_subcore += !is_regular_reg_in_mref;
    }
    bool is_half_bandwidth =
        shader_config
            .is_load_half_bandwidth_in_the_subcore_link_to_sm_enabled &&
        !space.is_shared();
    unsigned int link_transfer_size =
        is_half_bandwidth
            ? (shader_config.memory_subcore_link_to_sm_byte_size / 2)
            : shader_config.memory_subcore_link_to_sm_byte_size;
    m_num_pending_cycles_to_finish_wb_from_sm_struct_to_subcore =
        total_byte_size_for_warp / link_transfer_size;

    if (space.is_shared()) {
      if (data_size == 4) {
        m_num_pending_cycles_to_finish_wb_from_sm_struct_to_subcore += m_num_pending_cycles_to_finish_wb_from_sm_struct_to_subcore;
      }
    }else if(space.is_const()) {
      m_num_pending_cycles_to_finish_wb_from_sm_struct_to_subcore = 1;
      if(!is_regular_reg_in_mref) { 
        cycles_at_first_stage = 1;
        standard_num_cycles_per_stage_in_subcore = 1;
        cycles_at_last_stage_in_subcore = 1;
        for (unsigned int i = 1; i < (shader_config.memory_intermidiate_stages_subcore_unit - 1); i++) {
          m_num_cycles_per_intermediate_stage[i] = 1;
        }
      }
    }

    if(m_is_ldgsts) {
      m_num_pending_cycles_to_finish_wb_from_sm_struct_to_subcore = 1;
      extra_offset_store = shader_config.memory_subcore_extra_latency_load_shared_mem;
      m_num_cycles_to_wait_to_free_WAR += shader_config.memory_subcore_extra_latency_load_shared_mem;
    }
  }

  assert(static_cast<int>(standard_num_cycles_per_stage_in_subcore) >= shader_config.offset_latency_firts_stage_memory_subcore);
  m_num_cycles_per_intermediate_stage[0] = cycles_at_first_stage + shader_config.offset_latency_firts_stage_memory_subcore;
  m_num_cycles_per_intermediate_stage[shader_config.memory_intermidiate_stages_subcore_unit - 2] = standard_num_cycles_per_stage_in_subcore + extra_offset_store;
  m_num_cycles_per_intermediate_stage[shader_config.memory_intermidiate_stages_subcore_unit - 1] = cycles_at_last_stage_in_subcore;  
  
  // When the instruction Frees WAR dependence Counters
  for(unsigned int i = 0; i < (m_num_cycles_per_intermediate_stage.size() - 2); i++) {
    m_num_cycles_to_wait_to_free_WAR += m_num_cycles_per_intermediate_stage[i];
  }

  // GENERAL latencies of instructions at the SM shared unit
  if(is_shared) {
    m_latency_of_mem_operation_at_sm_structure = shader_config.memory_shared_memory_minimum_latency + is_regular_reg_in_mref;
    std::string opcode = m_extra_trace_instruction_info->get_op_code();
    if(opcode.find("LDSM") != std::string::npos) {
      if(endsWith(opcode, ".4") || endsWith(opcode, ".2")) {
        m_latency_of_mem_operation_at_sm_structure += shader_config.memory_shared_memory_extra_latency_ldsm_multiple_matrix;
      }
    }
  }else if(is_consider_global) {
    m_latency_of_mem_operation_at_sm_structure = shader_config.memory_l1d_minimum_latency;
  }else if(space.is_const()) {
    m_latency_of_mem_operation_at_sm_structure = shader_config.constant_cache_latency_at_sm_structure;
  }

  m_has_wb_from_sm_struct_to_subcore = true;
}

void warp_inst_t::generate_mem_accesses() {
  if (empty() || (op == MEMORY_BARRIER_OP) || (op == GRID_BARRIER_OP) || m_mem_accesses_created) return;
  if (!((op == LOAD_OP) || (op == TENSOR_CORE_LOAD_OP) || (op == STORE_OP) ||
        (op == TENSOR_CORE_STORE_OP) ))
    return;
  if (m_warp_active_mask.count() == 0) return;  // predicated off

  const size_t starting_queue_size = m_accessq.size();

  assert(is_load() || is_store());

  //if((space.get_type() != tex_space) && (space.get_type() != const_space))
    assert(m_per_scalar_thread_valid);  // need address information per thread

  bool is_write = is_store();

  mem_access_type access_type;
  switch (space.get_type()) {
    case const_space:
    case param_space_kernel:
      access_type = CONST_ACC_R;
      break;
    case tex_space:
      access_type = TEXTURE_ACC_R;
      break;
    case global_space:
      access_type = is_write ? GLOBAL_ACC_W : GLOBAL_ACC_R;
      break;
    case local_space:
    case param_space_local:
      access_type = is_write ? LOCAL_ACC_W : LOCAL_ACC_R;
      break;
    case shared_space:
      access_type = is_write ? LOCAL_ACC_W : LOCAL_ACC_R;
      break;
    case sstarr_space:
      access_type = is_write ? LOCAL_ACC_W : LOCAL_ACC_R;
      break;
    default:
      access_type = GLOBAL_ACC_R;
      assert(0); // CREATE CRASH
      break;
  }

  // Calculate memory accesses generated by this warp
  new_addr_type cache_block_size = 0;  // in bytes

  switch (space.get_type()) {
    case shared_space:
    case sstarr_space: {
      unsigned subwarp_size = m_config->warp_size / m_config->mem_warp_parts;
      unsigned total_accesses = 0;
      for (unsigned subwarp = 0; subwarp < m_config->mem_warp_parts;
           subwarp++) {
        // data structures used per part warp
        std::map<unsigned, std::map<new_addr_type, unsigned> >
            bank_accs;  // bank -> word address -> access count

        // step 1: compute accesses to words in banks
        std::set<new_addr_type> addr_set_to_track_coalescing;
        for (unsigned thread = subwarp * subwarp_size;
             thread < (subwarp + 1) * subwarp_size; thread++) {
          if (!active(thread)) continue;
          new_addr_type addr = m_per_scalar_thread[thread].memreqaddr[0];
          // FIXME: deferred allocation of shared memory should not accumulate
          // across kernel launches assert( addr < m_config->gpgpu_shmem_size );
          unsigned bank = m_config->shmem_bank_func(addr);
          new_addr_type word =
              line_size_based_tag_func(addr, m_config->WORD_SIZE);
          bank_accs[bank][word]++;
          addr_set_to_track_coalescing.insert(word);
        }
        for(auto it_s = addr_set_to_track_coalescing.begin(); it_s != addr_set_to_track_coalescing.end(); it_s++) {
          m_accessq.push_back(mem_access_t(
              access_type, *it_s, m_config->WORD_SIZE, is_write, active_mask_t(),
              mem_access_byte_mask_t(), mem_access_sector_mask_t(), m_config->gpgpu_ctx));
        }

        if (m_config->shmem_limited_broadcast) {
          // step 2: look for and select a broadcast bank/word if one occurs
          bool broadcast_detected = false;
          new_addr_type broadcast_word = (new_addr_type)-1;
          unsigned broadcast_bank = (unsigned)-1;
          std::map<unsigned, std::map<new_addr_type, unsigned> >::iterator b;
          for (b = bank_accs.begin(); b != bank_accs.end(); b++) {
            unsigned bank = b->first;
            std::map<new_addr_type, unsigned> &access_set = b->second;
            std::map<new_addr_type, unsigned>::iterator w;
            for (w = access_set.begin(); w != access_set.end(); ++w) {
              if (w->second > 1) {
                // found a broadcast
                broadcast_detected = true;
                broadcast_bank = bank;
                broadcast_word = w->first;
                break;
              }
            }
            if (broadcast_detected) break;
          }

          // step 3: figure out max bank accesses performed, taking account of
          // broadcast case
          unsigned max_bank_accesses = 0;
          for (b = bank_accs.begin(); b != bank_accs.end(); b++) {
            unsigned bank_accesses = 0;
            std::map<new_addr_type, unsigned> &access_set = b->second;
            std::map<new_addr_type, unsigned>::iterator w;
            for (w = access_set.begin(); w != access_set.end(); ++w)
              bank_accesses += w->second;
            if (broadcast_detected && broadcast_bank == b->first) {
              for (w = access_set.begin(); w != access_set.end(); ++w) {
                if (w->first == broadcast_word) {
                  unsigned n = w->second;
                  assert(n > 1);  // or this wasn't a broadcast
                  assert(bank_accesses >= (n - 1));
                  bank_accesses -= (n - 1);
                  break;
                }
              }
            }
            if (bank_accesses > max_bank_accesses)
              max_bank_accesses = bank_accesses;
          }

          // step 4: accumulate
          total_accesses += max_bank_accesses;
        } else {
          // step 2: look for the bank with the maximum number of access to
          // different words
          unsigned max_bank_accesses = 0;
          std::map<unsigned, std::map<new_addr_type, unsigned> >::iterator b;
          for (b = bank_accs.begin(); b != bank_accs.end(); b++) {
            max_bank_accesses =
                std::max(max_bank_accesses, (unsigned)b->second.size());
          }

          // step 3: accumulate
          total_accesses += max_bank_accesses;
        }
      }
      assert(total_accesses > 0 && total_accesses <= m_config->warp_size);
      cycles = total_accesses;  // shared memory conflicts modeled as larger
                                // initiation interval
      m_config->gpgpu_ctx->stats->ptx_file_line_stats_add_smem_bank_conflict(
          pc, total_accesses);
      break;
    }

    case tex_space:
      cache_block_size = m_config->gpgpu_cache_texl1_linesize;
      break;
    case const_space:
    case param_space_kernel:
      cache_block_size = m_config->gpgpu_cache_constl1_linesize;
      break;

    case global_space:
    case local_space:
    case param_space_local:
      if (m_config->gpgpu_coalesce_arch >= 13) {
        if (isatomic())
          memory_coalescing_arch_atomic(is_write, access_type);
        else
          memory_coalescing_arch(is_write, access_type);
      } else
        abort();

      break;

    default:
      abort();
  }

  if (cache_block_size) {
    assert(m_accessq.empty());
    mem_access_byte_mask_t byte_mask;
    std::map<new_addr_type, active_mask_t>
        accesses;  // block address -> set of thread offsets in warp
    std::map<new_addr_type, active_mask_t>::iterator a;
    for (unsigned thread = 0; thread < m_config->warp_size; thread++) {
      if (!active(thread)) continue;
      new_addr_type addr = m_per_scalar_thread[thread].memreqaddr[0];
      new_addr_type block_address =
          line_size_based_tag_func(addr, cache_block_size);
      accesses[block_address].set(thread);
      unsigned idx = addr - block_address;
      for (unsigned i = 0; i < data_size; i++) {
        byte_mask.set(idx + i);
      }
    }
    for (a = accesses.begin(); a != accesses.end(); ++a)
      m_accessq.push_back(mem_access_t(
          access_type, a->first, cache_block_size, is_write, a->second,
          byte_mask, mem_access_sector_mask_t(), m_config->gpgpu_ctx));
  }

  if (space.get_type() == global_space) {
    m_config->gpgpu_ctx->stats->ptx_file_line_stats_add_uncoalesced_gmem(
        pc, m_accessq.size() - starting_queue_size);
  }
  m_mem_accesses_created = true;
}

void warp_inst_t::memory_coalescing_arch(bool is_write,
                                         mem_access_type access_type) {
  // see the CUDA manual where it discusses coalescing rules before reading this
  unsigned segment_size = 0;
  unsigned warp_parts = m_config->mem_warp_parts;
  bool sector_segment_size = false;

  if (m_config->gpgpu_coalesce_arch >= 20 &&
      m_config->gpgpu_coalesce_arch < 39) {
    // Fermi and Kepler, L1 is normal and L2 is sector
    if (m_config->gmem_skip_L1D || cache_op == CACHE_GLOBAL)
      sector_segment_size = true;
    else
      sector_segment_size = false;
  } else if (m_config->gpgpu_coalesce_arch >= 40) {
    // Maxwell, Pascal and Volta, L1 and L2 are sectors
    // all requests should be 32 bytes
    sector_segment_size = true;
  }

  switch (data_size) {
    case 1:
      segment_size = 32;
      break;
    case 2:
      segment_size = sector_segment_size ? 32 : 64;
      break;
    case 4:
    case 8:
    case 16:
      segment_size = sector_segment_size ? 32 : 128;
      break;
  }
  unsigned subwarp_size = m_config->warp_size / warp_parts;

  for (unsigned subwarp = 0; subwarp < warp_parts; subwarp++) {
    std::map<new_addr_type, transaction_info> subwarp_transactions;

    // step 1: find all transactions generated by this subwarp
    for (unsigned thread = subwarp * subwarp_size;
         thread < subwarp_size * (subwarp + 1); thread++) {
      if (!active(thread)) continue;

      unsigned data_size_coales = data_size;
      unsigned num_accesses = 1;

      if (space.get_type() == local_space ||
          space.get_type() == param_space_local) {
        // Local memory accesses >4B were split into 4B chunks
        if (data_size >= 4) {
          data_size_coales = 4;
          num_accesses = data_size / 4;
        }
        // Otherwise keep the same data_size for sub-4B access to local memory
      }

      assert(num_accesses <= MAX_ACCESSES_PER_INSN_PER_THREAD);

      //            for(unsigned access=0; access<num_accesses; access++) {
      for (unsigned access = 0;
           (access < MAX_ACCESSES_PER_INSN_PER_THREAD) &&
           (m_per_scalar_thread[thread].memreqaddr[access] != 0);
           access++) {
        new_addr_type addr = m_per_scalar_thread[thread].memreqaddr[access];
        new_addr_type block_address =
            line_size_based_tag_func(addr, segment_size);
        unsigned chunk =
            (addr & 127) / 32;  // which 32-byte chunk within in a 128-byte
                                // chunk does this thread access?
        transaction_info &info = subwarp_transactions[block_address];

        // can only write to one segment
        // it seems like in trace driven, a thread can write to more than one
        // segment assert(block_address ==
        // line_size_based_tag_func(addr+data_size_coales-1,segment_size));

        info.chunks.set(chunk);
        info.active.set(thread);
        unsigned idx = (addr & 127);
        for (unsigned i = 0; i < data_size_coales; i++)
          if ((idx + i) < MAX_MEMORY_ACCESS_SIZE) info.bytes.set(idx + i);

        // it seems like in trace driven, a thread can write to more than one
        // segment handle this special case
        if (block_address != line_size_based_tag_func(
                                 addr + data_size_coales - 1, segment_size)) {
          addr = addr + data_size_coales - 1;
          new_addr_type block_address =
              line_size_based_tag_func(addr, segment_size);
          unsigned chunk = (addr & 127) / 32;
          transaction_info &info = subwarp_transactions[block_address];
          info.chunks.set(chunk);
          info.active.set(thread);
          unsigned idx = (addr & 127);
          for (unsigned i = 0; i < data_size_coales; i++)
            if ((idx + i) < MAX_MEMORY_ACCESS_SIZE) info.bytes.set(idx + i);
        }
      }
    }

    // step 2: reduce each transaction size, if possible
    std::map<new_addr_type, transaction_info>::iterator t;
    for (t = subwarp_transactions.begin(); t != subwarp_transactions.end();
         t++) {
      new_addr_type addr = t->first;
      const transaction_info &info = t->second;

      memory_coalescing_arch_reduce_and_send(is_write, access_type, info, addr,
                                             segment_size);
    }
  }
}

void warp_inst_t::memory_coalescing_arch_atomic(bool is_write,
                                                mem_access_type access_type) {
  assert(space.get_type() ==
         global_space);  // Atomics allowed only for global memory

  // see the CUDA manual where it discusses coalescing rules before reading this
  unsigned segment_size = 0;
  unsigned warp_parts = m_config->mem_warp_parts;
  bool sector_segment_size = false;

  if (m_config->gpgpu_coalesce_arch >= 20 &&
      m_config->gpgpu_coalesce_arch < 39) {
    // Fermi and Kepler, L1 is normal and L2 is sector
    if (m_config->gmem_skip_L1D || cache_op == CACHE_GLOBAL)
      sector_segment_size = true;
    else
      sector_segment_size = false;
  } else if (m_config->gpgpu_coalesce_arch >= 40) {
    // Maxwell, Pascal and Volta, L1 and L2 are sectors
    // all requests should be 32 bytes
    sector_segment_size = true;
  }

  switch (data_size) {
    case 1:
      segment_size = 32;
      break;
    case 2:
      segment_size = sector_segment_size ? 32 : 64;
      break;
    case 4:
    case 8:
    case 16:
      segment_size = sector_segment_size ? 32 : 128;
      break;
  }
  unsigned subwarp_size = m_config->warp_size / warp_parts;

  for (unsigned subwarp = 0; subwarp < warp_parts; subwarp++) {
    std::map<new_addr_type, std::list<transaction_info> >
        subwarp_transactions;  // each block addr maps to a list of transactions

    // step 1: find all transactions generated by this subwarp
    for (unsigned thread = subwarp * subwarp_size;
         thread < subwarp_size * (subwarp + 1); thread++) {
      if (!active(thread)) continue;

      new_addr_type addr = m_per_scalar_thread[thread].memreqaddr[0];
      new_addr_type block_address =
          line_size_based_tag_func(addr, segment_size);
      unsigned chunk =
          (addr & 127) / 32;  // which 32-byte chunk within in a 128-byte chunk
                              // does this thread access?

      // can only write to one segment
      assert(block_address ==
             line_size_based_tag_func(addr + data_size - 1, segment_size));

      // Find a transaction that does not conflict with this thread's accesses
      bool new_transaction = true;
      std::list<transaction_info>::iterator it;
      transaction_info *info;
      for (it = subwarp_transactions[block_address].begin();
           it != subwarp_transactions[block_address].end(); it++) {
        unsigned idx = (addr & 127);
        if (not it->test_bytes(idx, idx + data_size - 1)) {
          new_transaction = false;
          info = &(*it);
          break;
        }
      }
      if (new_transaction) {
        // Need a new transaction
        subwarp_transactions[block_address].push_back(transaction_info());
        info = &subwarp_transactions[block_address].back();
      }
      assert(info);

      info->chunks.set(chunk);
      info->active.set(thread);
      unsigned idx = (addr & 127);
      for (unsigned i = 0; i < data_size; i++) {
        assert(!info->bytes.test(idx + i));
        info->bytes.set(idx + i);
      }
    }

    // step 2: reduce each transaction size, if possible
    std::map<new_addr_type, std::list<transaction_info> >::iterator t_list;
    for (t_list = subwarp_transactions.begin();
         t_list != subwarp_transactions.end(); t_list++) {
      // For each block addr
      new_addr_type addr = t_list->first;
      const std::list<transaction_info> &transaction_list = t_list->second;

      std::list<transaction_info>::const_iterator t;
      for (t = transaction_list.begin(); t != transaction_list.end(); t++) {
        // For each transaction
        const transaction_info &info = *t;
        memory_coalescing_arch_reduce_and_send(is_write, access_type, info,
                                               addr, segment_size);
      }
    }
  }
}

void warp_inst_t::memory_coalescing_arch_reduce_and_send(
    bool is_write, mem_access_type access_type, const transaction_info &info,
    new_addr_type addr, unsigned segment_size) {
  assert((addr & (segment_size - 1)) == 0);

  const std::bitset<SECTOR_CHUNCK_SIZE> &q = info.chunks;
  assert(q.count() >= 1);
  std::bitset<2> h;  // halves (used to check if 64 byte segment can be
                     // compressed into a single 32 byte segment)

  unsigned size = segment_size;
  if (segment_size == 128) {
    bool lower_half_used = q[0] || q[1];
    bool upper_half_used = q[2] || q[3];
    if (lower_half_used && !upper_half_used) {
      // only lower 64 bytes used
      size = 64;
      if (q[0]) h.set(0);
      if (q[1]) h.set(1);
    } else if ((!lower_half_used) && upper_half_used) {
      // only upper 64 bytes used
      addr = addr + 64;
      size = 64;
      if (q[2]) h.set(0);
      if (q[3]) h.set(1);
    } else {
      assert(lower_half_used && upper_half_used);
    }
  } else if (segment_size == 64) {
    // need to set halves
    if ((addr % 128) == 0) {
      if (q[0]) h.set(0);
      if (q[1]) h.set(1);
    } else {
      assert((addr % 128) == 64);
      if (q[2]) h.set(0);
      if (q[3]) h.set(1);
    }
  }
  if (size == 64) {
    bool lower_half_used = h[0];
    bool upper_half_used = h[1];
    if (lower_half_used && !upper_half_used) {
      size = 32;
    } else if ((!lower_half_used) && upper_half_used) {
      addr = addr + 32;
      size = 32;
    } else {
      assert(lower_half_used && upper_half_used);
    }
  }
  m_accessq.push_back(mem_access_t(access_type, addr, size, is_write,
                                   info.active, info.bytes, info.chunks,
                                   m_config->gpgpu_ctx));
}

void warp_inst_t::completed(unsigned long long cycle) const {
  unsigned long long latency = cycle - issue_cycle;
  assert(latency <= cycle);  // underflow detection
  m_config->gpgpu_ctx->stats->ptx_file_line_stats_add_latency(
      pc, latency * active_count());
}

kernel_info_t::kernel_info_t(dim3 gridDim, dim3 blockDim,
                             class function_info *entry) {
  m_kernel_entry = entry;
  m_grid_dim = gridDim;
  m_block_dim = blockDim;
  m_next_cta.x = 0;
  m_next_cta.y = 0;
  m_next_cta.z = 0;
  m_next_tid = m_next_cta;
  m_num_cores_running = 0;
  m_parent_kernel = NULL;
  m_param_mem = new memory_space_impl<8192>("param", 64 * 1024);
  if(entry->gpgpu_ctx == nullptr){
    m_kernel_TB_latency = 0;
    m_launch_latency = 0;
    m_uid = entry->get_uid();
  }else {
    // Jin: parent and child kernel management for CDP
    m_uid = (entry->gpgpu_ctx->kernel_info_m_next_uid)++;
    // Jin: launch latency management
    m_launch_latency = entry->gpgpu_ctx->device_runtime->g_kernel_launch_latency;

    m_kernel_TB_latency =
        entry->gpgpu_ctx->device_runtime->g_kernel_launch_latency +
        num_blocks() * entry->gpgpu_ctx->device_runtime->g_TB_launch_latency;
  }

  cache_config_set = false;

  function_unique_id = 0;
  is_captured_from_binary = false;
}

/*A snapshot of the texture mappings needs to be stored in the kernel's info as
kernels should use the texture bindings seen at the time of launch and textures
 can be bound/unbound asynchronously with respect to streams. */
kernel_info_t::kernel_info_t(
    dim3 gridDim, dim3 blockDim, class function_info *entry,
    std::map<std::string, const struct cudaArray *> nameToCudaArray,
    std::map<std::string, const struct textureInfo *> nameToTextureInfo) {
  m_kernel_entry = entry;
  m_grid_dim = gridDim;
  m_block_dim = blockDim;
  m_next_cta.x = 0;
  m_next_cta.y = 0;
  m_next_cta.z = 0;
  m_next_tid = m_next_cta;
  m_num_cores_running = 0;
  m_uid = (entry->gpgpu_ctx->kernel_info_m_next_uid)++;
  m_param_mem = new memory_space_impl<8192>("param", 64 * 1024);

  // Jin: parent and child kernel management for CDP
  m_parent_kernel = NULL;

  // Jin: launch latency management
  m_launch_latency = entry->gpgpu_ctx->device_runtime->g_kernel_launch_latency;

  m_kernel_TB_latency =
      entry->gpgpu_ctx->device_runtime->g_kernel_launch_latency +
      num_blocks() * entry->gpgpu_ctx->device_runtime->g_TB_launch_latency;

  cache_config_set = false;
  m_NameToCudaArray = nameToCudaArray;
  m_NameToTextureInfo = nameToTextureInfo;

  function_unique_id = 0;
  is_captured_from_binary = false;
}

kernel_info_t::~kernel_info_t() {
  assert(m_active_threads.empty());
  destroy_cta_streams();
  delete m_param_mem;
}

std::string kernel_info_t::name() const { return m_kernel_entry->get_name(); }

// Jin: parent and child kernel management for CDP
void kernel_info_t::set_parent(kernel_info_t *parent, dim3 parent_ctaid,
                               dim3 parent_tid) {
  m_parent_kernel = parent;
  m_parent_ctaid = parent_ctaid;
  m_parent_tid = parent_tid;
  parent->set_child(this);
}

void kernel_info_t::set_child(kernel_info_t *child) {
  m_child_kernels.push_back(child);
}

void kernel_info_t::remove_child(kernel_info_t *child) {
  assert(std::find(m_child_kernels.begin(), m_child_kernels.end(), child) !=
         m_child_kernels.end());
  m_child_kernels.remove(child);
}

bool kernel_info_t::is_finished() {
  if (done() && children_all_finished())
    return true;
  else
    return false;
}

bool kernel_info_t::children_all_finished() {
  if (!m_child_kernels.empty()) return false;

  return true;
}

void kernel_info_t::notify_parent_finished() {
  if (m_parent_kernel) {
    m_kernel_entry->gpgpu_ctx->device_runtime->g_total_param_size -=
        ((m_kernel_entry->get_args_aligned_size() + 255) / 256 * 256);
    m_parent_kernel->remove_child(this);
    m_kernel_entry->gpgpu_ctx->the_gpgpusim->g_stream_manager
        ->register_finished_kernel(m_parent_kernel->get_uid());
  }
}

CUstream_st *kernel_info_t::create_stream_cta(dim3 ctaid) {
  assert(get_default_stream_cta(ctaid));
  CUstream_st *stream = new CUstream_st();
  m_kernel_entry->gpgpu_ctx->the_gpgpusim->g_stream_manager->add_stream(stream);
  assert(m_cta_streams.find(ctaid) != m_cta_streams.end());
  assert(m_cta_streams[ctaid].size() >= 1);  // must have default stream
  m_cta_streams[ctaid].push_back(stream);

  return stream;
}

CUstream_st *kernel_info_t::get_default_stream_cta(dim3 ctaid) {
  if (m_cta_streams.find(ctaid) != m_cta_streams.end()) {
    assert(m_cta_streams[ctaid].size() >=
           1);  // already created, must have default stream
    return *(m_cta_streams[ctaid].begin());
  } else {
    m_cta_streams[ctaid] = std::list<CUstream_st *>();
    CUstream_st *stream = new CUstream_st();
    m_kernel_entry->gpgpu_ctx->the_gpgpusim->g_stream_manager->add_stream(
        stream);
    m_cta_streams[ctaid].push_back(stream);
    return stream;
  }
}

bool kernel_info_t::cta_has_stream(dim3 ctaid, CUstream_st *stream) {
  if (m_cta_streams.find(ctaid) == m_cta_streams.end()) return false;

  std::list<CUstream_st *> &stream_list = m_cta_streams[ctaid];
  if (std::find(stream_list.begin(), stream_list.end(), stream) ==
      stream_list.end())
    return false;
  else
    return true;
}

void kernel_info_t::print_parent_info() {
  if (m_parent_kernel) {
    printf("Parent %d: \'%s\', Block (%d, %d, %d), Thread (%d, %d, %d)\n",
           m_parent_kernel->get_uid(), m_parent_kernel->name().c_str(),
           m_parent_ctaid.x, m_parent_ctaid.y, m_parent_ctaid.z, m_parent_tid.x,
           m_parent_tid.y, m_parent_tid.z);
  }
}

void kernel_info_t::destroy_cta_streams() {
  printf("Destroy streams for kernel %d: ", get_uid());
  size_t stream_size = 0;
  for (auto s = m_cta_streams.begin(); s != m_cta_streams.end(); s++) {
    stream_size += s->second.size();
    for (auto ss = s->second.begin(); ss != s->second.end(); ss++)
      m_kernel_entry->gpgpu_ctx->the_gpgpusim->g_stream_manager->destroy_stream(
          *ss);
    s->second.clear();
  }
  printf("size %lu\n", stream_size);
  m_cta_streams.clear();
}

simt_stack::simt_stack(unsigned wid, unsigned warpSize, class gpgpu_sim *gpu) {
  m_warp_id = wid;
  m_warp_size = warpSize;
  m_gpu = gpu;
  m_last_num_entries = 0;
  reset();
}

void simt_stack::reset() { m_stack.clear(); }

void simt_stack::launch(address_type start_pc, const simt_mask_t &active_mask) {
  reset();
  simt_stack_entry new_stack_entry;
  new_stack_entry.m_pc = start_pc;
  new_stack_entry.m_calldepth = 1;
  new_stack_entry.m_active_mask = active_mask;
  new_stack_entry.m_type = STACK_ENTRY_TYPE_NORMAL;
  m_stack.push_back(new_stack_entry);
}

void simt_stack::resume(char *fname) {
  reset();

  FILE *fp2 = fopen(fname, "r");
  assert(fp2 != NULL);

  char line[200]; /* or other suitable maximum line size */

  while (fgets(line, sizeof line, fp2) != NULL) /* read a line */
  {
    simt_stack_entry new_stack_entry;
    char *pch;
    pch = strtok(line, " ");
    for (unsigned j = 0; j < m_warp_size; j++) {
      if (pch[0] == '1')
        new_stack_entry.m_active_mask.set(j);
      else
        new_stack_entry.m_active_mask.reset(j);
      pch = strtok(NULL, " ");
    }

    new_stack_entry.m_pc = atoi(pch);
    pch = strtok(NULL, " ");
    new_stack_entry.m_calldepth = atoi(pch);
    pch = strtok(NULL, " ");
    new_stack_entry.m_recvg_pc = atoi(pch);
    pch = strtok(NULL, " ");
    new_stack_entry.m_branch_div_cycle = atoi(pch);
    pch = strtok(NULL, " ");
    if (pch[0] == '0')
      new_stack_entry.m_type = STACK_ENTRY_TYPE_NORMAL;
    else
      new_stack_entry.m_type = STACK_ENTRY_TYPE_CALL;
    m_stack.push_back(new_stack_entry);
  }
  fclose(fp2);
}

const simt_mask_t &simt_stack::get_active_mask() const {
  assert(m_stack.size() > 0);
  return m_stack.back().m_active_mask;
}

void simt_stack::get_pdom_stack_top_info(unsigned *pc, unsigned *rpc) const {
  assert(m_stack.size() > 0);
  *pc = m_stack.back().m_pc;
  *rpc = m_stack.back().m_recvg_pc;
}

unsigned simt_stack::get_rp() const {
  assert(m_stack.size() > 0);
  return m_stack.back().m_recvg_pc;
}

void simt_stack::print(FILE *fout) const {
  for (unsigned k = 0; k < m_stack.size(); k++) {
    simt_stack_entry stack_entry = m_stack[k];
    if (k == 0) {
      fprintf(fout, "w%02d %1u ", m_warp_id, k);
    } else {
      fprintf(fout, "    %1u ", k);
    }
    for (unsigned j = 0; j < m_warp_size; j++)
      fprintf(fout, "%c", (stack_entry.m_active_mask.test(j) ? '1' : '0'));
    fprintf(fout, " pc: 0x%03llx", stack_entry.m_pc);
    if (stack_entry.m_recvg_pc == (unsigned)-1) {
      fprintf(fout, " rp: ---- tp: %s cd: %2u ",
              (stack_entry.m_type == STACK_ENTRY_TYPE_CALL ? "C" : "N"),
              stack_entry.m_calldepth);
    } else {
      fprintf(fout, " rp: %4llu tp: %s cd: %2u ", stack_entry.m_recvg_pc,
              (stack_entry.m_type == STACK_ENTRY_TYPE_CALL ? "C" : "N"),
              stack_entry.m_calldepth);
    }
    if (stack_entry.m_branch_div_cycle != 0) {
      fprintf(fout, " bd@%6u ", (unsigned)stack_entry.m_branch_div_cycle);
    } else {
      fprintf(fout, " ");
    }
    m_gpu->gpgpu_ctx->func_sim->ptx_print_insn(stack_entry.m_pc, fout);
    fprintf(fout, "\n");
  }
}

void simt_stack::print_checkpoint(FILE *fout) const {
  for (unsigned k = 0; k < m_stack.size(); k++) {
    simt_stack_entry stack_entry = m_stack[k];

    for (unsigned j = 0; j < m_warp_size; j++)
      fprintf(fout, "%c ", (stack_entry.m_active_mask.test(j) ? '1' : '0'));
    fprintf(fout, "%llu %d %llu %llu %d ", stack_entry.m_pc,
            stack_entry.m_calldepth, stack_entry.m_recvg_pc,
            stack_entry.m_branch_div_cycle, stack_entry.m_type);
    fprintf(fout, "%d %d\n", m_warp_id, m_warp_size);
  }
}

void simt_stack::update(simt_mask_t &thread_done, addr_vector_t &next_pc, 
                        address_type recvg_pc, op_type next_inst_op,
                        unsigned next_inst_size, address_type next_inst_pc) {
}

void core_t::execute_warp_inst_t(warp_inst_t &inst, unsigned warpId) {
  for (unsigned t = 0; t < m_warp_size; t++) {
    if (inst.active(t)) {
      if (warpId == (unsigned(-1))) warpId = inst.warp_id();
      unsigned tid = m_warp_size * warpId + t;
      m_thread[tid]->ptx_exec_inst(inst, t);

      // virtual function
      checkExecutionStatusAndUpdate(inst, t, tid);
    }
  }
}

bool core_t::ptx_thread_done(unsigned hw_thread_id) const {
  return ((m_thread[hw_thread_id] == NULL) ||
          m_thread[hw_thread_id]->is_done());
}

void core_t::updateSIMTStack(unsigned warpId, warp_inst_t *inst) {
  simt_mask_t thread_done;
  addr_vector_t next_pc;
  unsigned wtid = warpId * m_warp_size;
  for (unsigned i = 0; i < m_warp_size; i++) {
    if (ptx_thread_done(wtid + i)) {
      thread_done.set(i);
      next_pc.push_back((address_type)-1);
    } else {
      if (inst->reconvergence_pc == RECONVERGE_RETURN_PC)
        inst->reconvergence_pc = get_return_pc(m_thread[wtid + i]);
      next_pc.push_back(m_thread[wtid + i]->get_pc());
    }
  }
  m_simt_stack[warpId]->update(thread_done, next_pc, inst->reconvergence_pc,
                               inst->op, inst->isize, inst->pc); // MOD. IBuffer_ooo
}

//! Get the warp to be executed using the data taken form the SIMT stack
warp_inst_t core_t::getExecuteWarp(unsigned warpId) {
  unsigned pc, rpc;
  m_simt_stack[warpId]->get_pdom_stack_top_info(&pc, &rpc);
  warp_inst_t wi = *(m_gpu->gpgpu_ctx->ptx_fetch_inst(pc));
  wi.set_active(m_simt_stack[warpId]->get_active_mask(), m_gpu->getShaderCoreConfig()->warp_size);
  return wi;
}

void core_t::deleteSIMTStack() {
  if (m_simt_stack) {
    for (unsigned i = 0; i < m_warp_count; ++i) delete m_simt_stack[i];
    delete[] m_simt_stack;
    m_simt_stack = NULL;
  }
}

void core_t::initilizeSIMTStack(unsigned warp_count, unsigned warp_size) {
  m_simt_stack = new simt_stack *[warp_count];
  for (unsigned i = 0; i < warp_count; ++i)
    m_simt_stack[i] = new simt_stack(i, warp_size, m_gpu);
  m_warp_size = warp_size;
  m_warp_count = warp_count;
}

void core_t::get_pdom_stack_top_info(unsigned warpId, unsigned *pc,
                                     unsigned *rpc) const {
  m_simt_stack[warpId]->get_pdom_stack_top_info(pc, rpc);
}
