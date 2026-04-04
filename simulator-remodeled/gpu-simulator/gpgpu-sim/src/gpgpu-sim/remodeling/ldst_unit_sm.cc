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

#include "ldst_unit_sm.h"
#include <functional>
#include <limits>
#include "../gpu-sim.h"
#include "../shader.h"
#include "../stat-tool.h"
#include "sm.h"
#include "fusedMemory/coalescingStats.h"

#define STRSIZE 1024

uint64_t calculate_constant_address(uint64_t reg_offset_value, traced_operand& op_c) {
  assert(op_c.get_operand_type() == TraceEnhancedOperandType::CBANK);
  std::vector<double> c_imms = op_c.get_operands_inmediates();
  unsigned int c_region =c_imms[0];
  double c_region_offset = 0;
  if(c_imms.size() > 1)   {
    c_region_offset = c_imms[1];
  }

  uint64_t total_offset_addr = reg_offset_value + c_region_offset;
  assert(total_offset_addr <= MAX_CONSTANT_REGION_SIZE);
  uint64_t region_first_addr = c_region * MAX_CONSTANT_REGION_SIZE;
  uint64_t final_constant_addr = FIRST_CONSTANT_CACHE_ADDR + region_first_addr + total_offset_addr;
  assert(final_constant_addr <= LAST_CONSTANT_CACHE_ADDR);
  return final_constant_addr;
}

ldst_unit_sm::ldst_unit_sm(
    std::vector<register_set_uniptr*> result_ports,
    std::vector<register_set_uniptr*> reception_ports, mem_fetch_interface *icnt,
    mem_fetch_interface *icnt_L1C_L1_half_C,
    std::shared_ptr<shader_core_mem_fetch_allocator> mf_allocator, SM *core,
    std::shared_ptr<Scoreboard> scoreboard, std::shared_ptr<Scoreboard_reads> scoreboard_reads,
    const shader_core_config *config, const memory_config *mem_config,
    shader_core_stats *stats, unsigned sid, unsigned tpc,
    unsigned int max_size_arbiter_to_subpipeline_reg_for_icnt_and_subcores)
    : functional_unit_shared_sm_part(result_ports, config, 0,
                             "MEM_SM_shared", core, MEM__OP, false, false, 1,
                             reception_ports, NUM_INTERMEDIATE_CYCLES_UN_BETWEEN_ISSUE_AND_FU_EXECUTION_FOR_FIXED_LATENCY_INST, nullptr, 0, false, TraceEnhancedOperandType::NONE),
                             m_access_queue_to_l1c(config->sm_memory_unit_l1c_access_queue_size), m_access_queue_to_l1t(config->sm_memory_unit_l1t_access_queue_size), m_access_queue_to_l1d_preTLB(config->m_L1D_config.l1_banks), m_access_queue_to_l1d_postTLB(config->m_L1D_config.l1_banks), m_access_queue_to_shmem(config->sm_memory_unit_shmem_access_queue_size), m_access_queue_to_bypass_to_l2(config->sm_memory_unit_bypass_l1d_directly_go_to_l2_access_queue_size), m_access_queue_to_miscellaneous(config->sm_memory_unit_miscellaneous_access_queue_size) {
  assert(config->maximum_shared_memory_latency_at_sm_structure > 1);
  m_max_size_arbiter_to_subpipeline_reg_for_icnt_and_subcores =
      max_size_arbiter_to_subpipeline_reg_for_icnt_and_subcores;
  init(icnt, icnt_L1C_L1_half_C, mf_allocator, core, scoreboard, scoreboard_reads,
       config,  // MOD. Fix WAR at baseline.
       mem_config, stats, sid, tpc);
  if (!m_config->m_L1D_config.disabled()) {
    char L1D_name[STRSIZE];
    snprintf(L1D_name, STRSIZE, "L1D_%03d", m_sid);
    m_L1D = new l1_cache(L1D_name, m_config->m_L1D_config, m_sid,
                         get_shader_normal_cache_id(), m_icnt, m_mf_allocator.get(),
                         IN_L1D_MISS_QUEUE, core->get_gpu());

    constant_cache_l1_latency_queue.resize(m_config->constant_cache_latency_at_sm_structure, (mem_fetch *)NULL);
    assert(m_config->maximum_l1d_latency_at_sm_structure > 0);

    l1d_latency_queue.resize(m_config->m_L1D_config.l1_banks);
    for (unsigned int j = 0; j < m_config->m_L1D_config.l1_banks; j++) {
      l1d_latency_queue[j].resize(m_config->maximum_l1d_latency_at_sm_structure, nullptr);
    }
  }
  m_name = "MEM ";
  m_current_num_shared_mem_inst = 0;
  m_current_num_normal_mem_inst = 0;
  m_num_cycles_to_wait_to_issue_another_mem_inst_from_the_subcores = 0;
  m_num_reserved_associativity_currently_processing = 0;
  if(m_config->is_interwarp_coalescing_enabled) {
    m_intercoalescing_unit = new InterWarpCoalescingUnit(this, m_config->num_interwarp_coalescing_tables, m_config->max_size_interwarp_coalescing_per_table);
  }
}

ldst_unit_sm::ldst_unit_sm(
    std::vector<register_set_uniptr*> result_ports,
    std::vector<register_set_uniptr*> reception_ports, mem_fetch_interface *icnt,
    mem_fetch_interface *icnt_L1C_L1_half_C,
    std::shared_ptr<shader_core_mem_fetch_allocator> mf_allocator, SM *core,
    std::shared_ptr<Scoreboard> scoreboard, std::shared_ptr<Scoreboard_reads> Scoreboard_reads,
    const shader_core_config *config,  // MOD. Fix WAR at baseline.
    const memory_config *mem_config, shader_core_stats *stats, unsigned sid,
    unsigned tpc, l1_cache *new_l1d_cache,
    unsigned int max_size_arbiter_to_subpipeline_reg_for_icnt_and_subcores)
    : functional_unit_shared_sm_part(result_ports, config, 0,
                             "MEM_SM_shared", core, MEM__OP, false, false, 1,
                             reception_ports, NUM_INTERMEDIATE_CYCLES_UN_BETWEEN_ISSUE_AND_FU_EXECUTION_FOR_FIXED_LATENCY_INST, nullptr, 0, false, TraceEnhancedOperandType::NONE),
      m_L1D(new_l1d_cache), m_access_queue_to_l1c(config->sm_memory_unit_l1c_access_queue_size), m_access_queue_to_l1t(config->sm_memory_unit_l1t_access_queue_size), m_access_queue_to_l1d_preTLB(config->m_L1D_config.l1_banks), m_access_queue_to_l1d_postTLB(config->m_L1D_config.l1_banks), m_access_queue_to_shmem(config->sm_memory_unit_shmem_access_queue_size), m_access_queue_to_bypass_to_l2(config->sm_memory_unit_bypass_l1d_directly_go_to_l2_access_queue_size), m_access_queue_to_miscellaneous(config->sm_memory_unit_miscellaneous_access_queue_size)  {
  m_max_size_arbiter_to_subpipeline_reg_for_icnt_and_subcores =
      max_size_arbiter_to_subpipeline_reg_for_icnt_and_subcores;
  init(icnt, icnt_L1C_L1_half_C, mf_allocator, core, scoreboard, Scoreboard_reads,
       config,  // MOD. Fix WAR at baseline.
       mem_config, stats, sid, tpc);
  m_current_num_shared_mem_inst = 0;
  m_current_num_normal_mem_inst = 0;
  m_num_cycles_to_wait_to_issue_another_mem_inst_from_the_subcores = 0;
  m_num_reserved_associativity_currently_processing = 0;
}

ldst_unit_sm::~ldst_unit_sm() {
  delete m_L1T;
  delete m_L1C;
  if (m_L1D) {
    delete m_L1D;
  }
  delete m_prt;
  for (unsigned int i = 0; i < m_config->m_L1D_config.l1_banks; i++) {
    delete m_access_queue_to_l1d_preTLB[i];
    delete m_access_queue_to_l1d_postTLB[i];
  }
  for(unsigned int i = 0; i < m_config->maximum_shared_memory_latency_at_sm_structure; i++) {
    if(m_shmem_pipeline[i] != nullptr) {
      delete m_shmem_pipeline[i];
    }
  }
  delete[] m_shmem_pipeline;
  delete m_coalescing_stats_l1d;
  delete m_coalescing_stats_const;
  delete m_coalescing_stats_sharedmem;
  if(m_config->is_interwarp_coalescing_enabled) {
    delete m_intercoalescing_unit;
  }
}

void ldst_unit_sm::init(
    mem_fetch_interface *icnt, mem_fetch_interface *icnt_L1C_L1_half_C, std::shared_ptr<shader_core_mem_fetch_allocator> mf_allocator,
    SM *core, std::shared_ptr<Scoreboard> scoreboard, std::shared_ptr<Scoreboard_reads> scoreboard_reads,
    const shader_core_config *config,  // MOD. Fix WAR at baseline.
    const memory_config *mem_config, shader_core_stats *stats, unsigned sid,
    unsigned tpc) {
  m_memory_config = mem_config;
  m_icnt = icnt;
  m_icnt_L1C_L1_half_C = icnt_L1C_L1_half_C;
  m_mf_allocator = mf_allocator;
  m_core = core;
  m_scoreboard = scoreboard;
  m_scoreboard_reads = scoreboard_reads;  // MOD. Fix WAR at baseline.
  m_stats = stats;
  m_sid = sid;
  m_tpc = tpc;
  char L1T_name[STRSIZE];
  char L1C_name[STRSIZE];
  snprintf(L1T_name, STRSIZE, "L1T_%03d", m_sid);
  snprintf(L1C_name, STRSIZE, "L1C_%03d", m_sid);
  m_prt = new PendingRequestTable(m_config->memory_sm_prt_size, this);
  m_L1T = new tex_cache(L1T_name, m_config->m_L1T_config, m_sid,
                        get_shader_texture_cache_id(), icnt, IN_L1T_MISS_QUEUE,
                        IN_SHADER_L1T_ROB);
  m_L1C = new read_only_cache(L1C_name, m_config->m_L1C_config, m_sid,
                              get_shader_constant_cache_id(), m_icnt_L1C_L1_half_C,
                              IN_L1C_MISS_QUEUE);
  m_shmem_pipeline = new mem_access_t *[m_config->maximum_shared_memory_latency_at_sm_structure];
  for (unsigned i = 0; i < m_config->maximum_shared_memory_latency_at_sm_structure; i++) {
    m_shmem_pipeline[i] = nullptr;
  }
  m_L1D = NULL;
  m_access_queue_to_l1d_preTLB.resize(m_config->m_L1D_config.l1_banks);
  m_access_queue_to_l1d_postTLB.resize(m_config->m_L1D_config.l1_banks);
  for(unsigned int i = 0; i < m_config->m_L1D_config.l1_banks; i++) {
    m_access_queue_to_l1d_preTLB[i] = new AccessQueue(m_config->sm_memory_unit_l1d_access_queue_size);
    m_access_queue_to_l1d_postTLB[i] = new AccessQueue(m_config->sm_memory_unit_l1d_access_queue_size);
  }
  m_num_writeback_clients =
      5;  // = shared memory, global/local (uncached), L1D, L1T, L1C
  m_next_global = NULL;
  m_last_inst_gpu_sim_cycle = 0;
  m_last_inst_gpu_tot_sim_cycle = 0;

  m_dispatch_subpipeline_arb_between_icnt_and_subcores = 0;
  m_writeback_arb_between_icnt_and_subcores = 0;
  m_num_icnt_and_subcores_clients = m_core->get_num_subcores() + 1;
  m_reserved_idx_icnt_to_shmem = m_core->get_num_subcores();

  m_pending_wbs_per_subcore.resize(m_core->get_num_subcores());
  m_writeback_arb_icnt_and_subcores.resize(m_num_icnt_and_subcores_clients);
  m_mem_rc_icnt_and_subcores.resize(m_num_icnt_and_subcores_clients);

  for (unsigned int i = 0; i < m_num_icnt_and_subcores_clients; i++) {
    m_evaluating_wb_icnt_and_subcores.push_back(warp_inst_t(config));
    m_writeback_arb_icnt_and_subcores[i] = 0;
    m_mem_rc_icnt_and_subcores[i] = NO_RC_FAIL;
  }

  m_global_shared_latency_queue_for_ldgsts.resize(m_config->memory_global_shared_latency_for_ldgsts);
  for(unsigned int i = 0; i < m_config->memory_global_shared_latency_for_ldgsts; i++) {
    m_global_shared_latency_queue_for_ldgsts[i] = nullptr;
  }

  m_coalescing_stats_l1d = new coalescingAddressStats(m_sm, "l1d", _memory_space_t::global_space);
  m_coalescing_stats_const = new coalescingAddressStats(m_sm, "const", _memory_space_t::const_space);
  m_coalescing_stats_sharedmem = new coalescingAddressStats(m_sm, "sharedmem", _memory_space_t::shared_space);
}

bool ldst_unit_sm::is_dispatch_reg_empty(unsigned int icnt_id) const {
  return m_reception_ports[icnt_id]->has_free();
}

bool ldst_unit_sm::can_issue(const warp_inst_t *inst) const {
  switch (inst->op) {
    case LOAD_OP:
      break;
    case TENSOR_CORE_LOAD_OP:
      break;
    case STORE_OP:
      break;
    case TENSOR_CORE_STORE_OP:
      break;
    case MEMORY_BARRIER_OP:
      break;
    case GRID_BARRIER_OP:
      break;
    default:
      return false;
  }
  return m_reception_ports[inst->get_subcore_id()]->has_free();
}

void ldst_unit_sm::active_lanes_in_pipeline() {
  unsigned active_count = functional_unit::get_active_lanes_in_pipeline();
  assert(active_count <= m_core->get_config()->warp_size);
  m_core->incfumemactivelanes_stat(active_count);
}

void ldst_unit_sm::invalidate() {
  // Flush L1D cache
  m_L1D->invalidate();
}

void ldst_unit_sm::shared_dispatch() {
  if(!m_access_queue_to_shmem.empty()) {
    mem_access_t *acc = m_access_queue_to_shmem.front();
    assert(acc->get_inst()->m_prt_assigned);
    mem_stage_stall_type rc_fail = NO_RC_FAIL;
    bool is_shd_mem_available = m_shmem_pipeline[acc->get_inst()->m_latency_of_mem_operation_at_sm_structure - 1] == nullptr;

    if(!is_shd_mem_available) {
      m_stats->gpgpu_n_shmem_bank_access[m_sid]++;
      rc_fail = DATA_PORT_STALL;
      m_sm->m_sm_stats.m_stats_map["gpgpu_n_shmem_port_conflict"]->increment_with_integer(1);
    }else {
      m_shmem_pipeline[acc->get_inst()->m_latency_of_mem_operation_at_sm_structure - 1] = acc;
    }
    if (rc_fail == NO_RC_FAIL) {
      m_access_queue_to_shmem.pop();
    }
  }
}

mem_stage_stall_type ldst_unit_sm::process_cache_access(
    cache_t &cache, new_addr_type address, warp_inst_t &inst,
    std::list<cache_event> &events, mem_fetch *mf,
    enum cache_request_status status) {
  mem_stage_stall_type result = NO_RC_FAIL;
  bool write_sent = was_write_sent(events);
  bool read_sent = was_read_sent(events);
  if (write_sent) {
    unsigned inc_ack = (m_config->m_L1D_config.get_mshr_type() == SECTOR_ASSOC)
                           ? (mf->get_data_size() / SECTOR_SIZE)
                           : 1;

    for (unsigned i = 0; i < inc_ack; ++i)
      m_core->inc_store_req(inst.warp_id());
  }
  if (status == HIT) {
    assert(!read_sent);
    pending_access_logic(mf->get_access().get_access_coal_info().m_prts_requesting);
    if (!write_sent) delete mf;
  } else if (status == RESERVATION_FAIL) {
    result = BK_CONF;
    assert(!read_sent);
    assert(!write_sent);
    delete mf;
  } else {
    assert(status == MISS || status == HIT_RESERVED);
  }
  return result;
}

mem_stage_stall_type ldst_unit_sm::process_memory_access_queue(cache_t &cache, mem_access_t *acc, bool is_const_cache) {
  if (!cache.data_port_free()) return DATA_PORT_STALL;

  mem_fetch *mf = m_mf_allocator->alloc(
      *(acc->get_inst()), *acc,
      m_core->get_gpu()->gpu_sim_cycle + m_core->get_gpu()->gpu_tot_sim_cycle);
  std::list<cache_event> events;
  enum cache_request_status status = cache.access(
      mf->get_addr(), mf,
      m_core->get_gpu()->gpu_sim_cycle + m_core->get_gpu()->gpu_tot_sim_cycle,
      events);
  return process_cache_access(cache, mf->get_addr(), *(acc->get_inst()), events, mf, status);
}

mem_stage_stall_type ldst_unit_sm::dispatch_to_memory_access_queue_l1Dcache(cache_t &cache, mem_access_t *acc) { 
  mem_stage_stall_type result = NO_RC_FAIL;
  if (m_config->maximum_l1d_latency_at_sm_structure > 0) {
    unsigned int inst_latency = acc->get_inst()->m_latency_of_mem_operation_at_sm_structure;
    // unsigned int max_num_accesses_per_cycle = m_config->memory_l1d_max_lookups_per_cycle_per_bank * m_config->m_L1D_config.l1_banks;// VER QUE HACER
    bool is_a_bank_conflict = false;
    unsigned int acc_bank = acc->get_l1d_bank();
    assert(acc_bank < m_config->m_L1D_config.l1_banks);
    bool inserted = false;
    if(!l1d_latency_queue[acc_bank][inst_latency - 1]) {
      l1d_latency_queue[acc_bank][inst_latency - 1] = std::make_shared<l1d_queue_element>();
      inserted = true;
    }else if(l1d_latency_queue[acc_bank][inst_latency - 1]->mfs.size() < m_config->memory_l1d_max_lookups_per_cycle_per_bank) {
      inserted = true;
    }else {
      is_a_bank_conflict = true;
    }
    
    if(inserted) {
      mem_fetch *mf =
          m_mf_allocator->alloc(*(acc->get_inst()), *acc,
                                m_core->get_gpu()->gpu_sim_cycle +
                                    m_core->get_gpu()->gpu_tot_sim_cycle);
      if (mf->get_inst().is_store()) {
        unsigned inc_ack =
            (m_config->m_L1D_config.get_mshr_type() == SECTOR_ASSOC)
                ? (mf->get_data_size() / SECTOR_SIZE)
                : 1;

        for (unsigned i = 0; i < inc_ack; ++i) {
          m_core->inc_store_req(acc->get_inst()->warp_id());
        }
        pending_access_logic(acc->get_access_coal_info().m_prts_requesting);
      }
      l1d_latency_queue[acc_bank][inst_latency - 1]->mfs.push_back(mf);
    }

    if(is_a_bank_conflict) {
      result = BK_CONF;
      m_sm->m_sm_stats.m_stats_map["gpgpu_n_l1cache_bkconflict"]->increment_with_integer(1);
    }
    assert(inserted || (result == BK_CONF));

    return result;
  } else {
    mem_fetch *mf =
        m_mf_allocator->alloc(*(acc->get_inst()), *acc,
                              m_core->get_gpu()->gpu_sim_cycle +
                                  m_core->get_gpu()->gpu_tot_sim_cycle);
    std::list<cache_event> events;
    enum cache_request_status status = cache.access(
        mf->get_addr(), mf,
        m_core->get_gpu()->gpu_sim_cycle + m_core->get_gpu()->gpu_tot_sim_cycle,
        events);
    return process_cache_access(cache, mf->get_addr(), *(acc->get_inst()), events, mf, status);
  }
}

mem_stage_stall_type ldst_unit_sm::dispatch_to_memory_access_queue_l1Ccache(cache_t &cache, mem_access_t *acc) { 
  mem_stage_stall_type result = NO_RC_FAIL;
  if (m_config->constant_cache_latency_at_sm_structure > 0) {
    bool access_granted = false;
    unsigned int inst_latency = acc->get_inst()->m_latency_of_mem_operation_at_sm_structure;
    assert(inst_latency <= m_config->constant_cache_latency_at_sm_structure);
    if(constant_cache_l1_latency_queue[inst_latency - 1] == NULL) {
      mem_fetch *mf = m_mf_allocator->alloc(*(acc->get_inst()), *acc,
                                            m_core->get_current_gpu_cycle());
      mf->set_subcore(m_core->get_num_subcores()*2); // The L1C is in the last position. Because we add the L0Is and then the L0Cs and we have each per subcore of this caches. 
      constant_cache_l1_latency_queue[inst_latency - 1] = mf;
      access_granted = true;
    }
    if (!access_granted) {
      result = BK_CONF;
      m_sm->m_sm_stats.m_stats_map["gpgpu_n_cmem_portconflict"]->increment_with_integer(1);
    }
    return result;
  } else {
    mem_fetch *mf =
        m_mf_allocator->alloc(*(acc->get_inst()), *acc,
                              m_core->get_gpu()->gpu_sim_cycle +
                                  m_core->get_gpu()->gpu_tot_sim_cycle);
    std::list<cache_event> events;
    enum cache_request_status status = cache.access(
        mf->get_addr(), mf,
        m_core->get_gpu()->gpu_sim_cycle + m_core->get_gpu()->gpu_tot_sim_cycle,
        events);
    return process_cache_access(cache, mf->get_addr(), *(acc->get_inst()), events, mf,
                                status);
  }
}

bool ldst_unit_sm::can_entry_be_selected_for_processing(unsigned int value) {
  bool res = (value + m_num_reserved_associativity_currently_processing) <= m_L1D->get_config().get_assoc();
  return res;
}

void ldst_unit_sm::increment_num_reserved_associativity_currently_processing(unsigned int value) {
  assert((m_num_reserved_associativity_currently_processing + value) <= m_L1D->get_config().get_assoc());
  m_num_reserved_associativity_currently_processing += value;
}

void ldst_unit_sm::decrement_num_reserved_associativity_currently_processing(unsigned int value) {
  assert((m_num_reserved_associativity_currently_processing - value) >= 0);
  m_num_reserved_associativity_currently_processing -= value;
  assert(m_num_reserved_associativity_currently_processing >= 0);
}

unsigned ldst_unit_sm::get_first_key_pending_writes(warp_inst_t *inst) {
  if (m_core->get_is_loog_enabled()) {
    return inst->m_cu_rrs_id;
  } else {
    return inst->warp_id();
  }
}

long double ldst_unit_sm::get_second_key_pending_writes(warp_inst_t *inst,
                                                     int idx) {
  if (m_core->get_config()->is_vpreg_enabled) {
    return inst->vpreg_virtual_out[idx];
  } else {
    return get_instruction_id(inst, idx);
  }
}

void ldst_unit_sm::print_L1_constant_latency_queue(FILE *f) {
  fprintf(f, "L1 constant latency queue: \n");
  for (unsigned int stage = 0; stage < m_config->constant_cache_latency_at_sm_structure; ++stage) {
    fprintf(f, "l1_constant_latency_queue[%d] =", stage);
    if (constant_cache_l1_latency_queue[stage] != NULL) {
      constant_cache_l1_latency_queue[stage]->print(f, true);
    } else {
      fprintf(f, " empty\n");
    }
  }
}

void ldst_unit_sm::pending_access_logic(std::vector<unsigned int> &prt_list) {
  assert(!prt_list.empty());
  for(auto prt_id : prt_list) {
    m_prt->solve_access(prt_id);
  }
}


void ldst_unit_sm::L1_constant_cache_latency_queue_cycle() {
  if ((constant_cache_l1_latency_queue[0]) != NULL) {
    mem_fetch *mf_next = constant_cache_l1_latency_queue[0];
    std::list<cache_event> events;
    bool useless = false;
    enum cache_request_status status =
          m_L1C->access(mf_next->get_addr(), mf_next, m_core->get_current_gpu_cycle(), events, useless);
    bool write_sent = was_write_sent(events);
    bool safe_to_delete = true;
    if(status != RESERVATION_FAIL) {
      constant_cache_l1_latency_queue[0] = NULL;
      if (status == HIT) {
        pending_access_logic(mf_next->get_access().get_access_coal_info().m_prts_requesting);
        if (!write_sent && safe_to_delete) {
          delete mf_next;
        }
      } else {
        assert(status == MISS || status == HIT_RESERVED);
      }
    }
  }

  for (unsigned stage = 0;
       stage < m_config->constant_cache_latency_at_sm_structure - 1; ++stage) {
    if (constant_cache_l1_latency_queue[stage] == NULL) {
      constant_cache_l1_latency_queue[stage] =
          constant_cache_l1_latency_queue[stage + 1];
      constant_cache_l1_latency_queue[stage + 1] = NULL;
    }
  }
}


void ldst_unit_sm::print_L1_latency_queue(FILE *f) {
  fprintf(f, "L1D latency queue: \n");
  for (unsigned int j = 0; j < m_config->m_L1D_config.l1_banks; j++) {
    for (unsigned int stage = 0; stage < m_config->maximum_l1d_latency_at_sm_structure; ++stage) {
      fprintf(f, "l1_latency_queue[%d][%d] =", j, stage);
      if (l1d_latency_queue[j][stage]) {
        fprintf(f," Number of mem_fetches: %ld\n", l1d_latency_queue[j][stage]->mfs.size());
        for(auto mf : l1d_latency_queue[j][stage]->mfs) {
          mf->print(f, true);
        }
      } else {
        fprintf(f, " empty\n");
      }
    }
  }
}

void ldst_unit_sm::L1_latency_queue_cycle() {
  for (unsigned int j = 0; j < m_config->m_L1D_config.l1_banks; j++) {
    m_stats->l1d_evals_per_sid_per_bank[m_sid][j]++;  // MOD. Memory stats
    if (l1d_latency_queue[j][0]) {
      auto it_mfs = l1d_latency_queue[j][0]->mfs.begin();
      while(l1d_latency_queue[j][0] && (it_mfs != l1d_latency_queue[j][0]->mfs.end()) ) {
        bool erased_it = false;
        mem_fetch *mf_next = *it_mfs;
        std::list<cache_event> events;
        enum cache_request_status status;
        if(mf_next->get_access_sector_mask().count() == 0){
          // Fix some bugged instructions that were not tracking properly mem addresses
          status = HIT;
        }else {
          status =
              m_L1D->access(mf_next->get_addr(), mf_next,
                            m_core->get_gpu()->gpu_sim_cycle +
                                m_core->get_gpu()->gpu_tot_sim_cycle,
                            events);
        }
        bool write_sent = was_write_sent(events);
        bool read_sent = was_read_sent(events);

        if ((status == HIT) || (status == MISS) ||
            (status == HIT_RESERVED)) {  // MOD. Memory stats
          m_stats->l1d_accesses_per_sid_per_bank[m_sid][j]++;
        }

        if (status == HIT) {
          bool safe_to_delete = true;
          assert(!read_sent);
          if (mf_next->get_inst().is_load()) {
            pending_access_logic(mf_next->get_access().get_access_coal_info().m_prts_requesting);
          }

          // For write hit in WB policy
          if (mf_next->get_inst().is_store() && !write_sent) {
            unsigned dec_ack =
                (m_config->m_L1D_config.get_mshr_type() == SECTOR_ASSOC)
                    ? (mf_next->get_data_size() / SECTOR_SIZE)
                    : 1;

            mf_next->set_reply();

            for (unsigned i = 0; i < dec_ack; ++i) m_core->store_ack(mf_next);
          }

          if (!write_sent && safe_to_delete) {
            delete mf_next;
          }

          if(safe_to_delete) {
            erased_it = true;
            it_mfs = l1d_latency_queue[j][0]->mfs.erase(it_mfs);
            if(l1d_latency_queue[j][0]->mfs.empty()) {
              l1d_latency_queue[j][0] = nullptr;
            }
          }

        } else if (status == RESERVATION_FAIL) {
          assert(!read_sent);
          assert(!write_sent);
        } else {
          assert(status == MISS || status == HIT_RESERVED);
          erased_it = true;
          it_mfs = l1d_latency_queue[j][0]->mfs.erase(it_mfs);
          if(l1d_latency_queue[j][0]->mfs.empty()) {
            l1d_latency_queue[j][0] = nullptr;
          }
          if (m_config->m_L1D_config.get_write_policy() != WRITE_THROUGH &&
              mf_next->get_inst().is_store() &&
              (m_config->m_L1D_config.get_write_allocate_policy() ==
                   FETCH_ON_WRITE ||
               m_config->m_L1D_config.get_write_allocate_policy() ==
                   LAZY_FETCH_ON_READ) &&
              !was_writeallocate_sent(events)) {
            unsigned dec_ack =
                (m_config->m_L1D_config.get_mshr_type() == SECTOR_ASSOC)
                    ? (mf_next->get_data_size() / SECTOR_SIZE)
                    : 1;
            mf_next->set_reply();
            for (unsigned i = 0; i < dec_ack; ++i) m_core->store_ack(mf_next);
            if (!write_sent && !read_sent) delete mf_next;
          }
        }
        if(!erased_it) {
          it_mfs++;
        }
      }
    }

    for (unsigned stage = 0; stage < m_config->maximum_l1d_latency_at_sm_structure - 1;
         ++stage)
      if (!l1d_latency_queue[j][stage]) {
        l1d_latency_queue[j][stage] = std::move(l1d_latency_queue[j][stage + 1]);
        l1d_latency_queue[j][stage + 1] = nullptr;
      }
  }
}

mem_stage_stall_type ldst_unit_sm::dispatch_to_memory_access_queue_l1Tcache(cache_t &cache, mem_access_t *acc) {
  mem_stage_stall_type fail = process_memory_access_queue(*m_L1T, acc, false);
  return fail;
}

bool ldst_unit_sm::response_buffer_full() const {
  return m_response_fifo.size() >= m_config->ldst_unit_response_queue_size;
}

void ldst_unit_sm::fill(mem_fetch *mf) {
  mf->set_status(
      IN_SHADER_LDST_RESPONSE_FIFO,
      m_core->get_gpu()->gpu_sim_cycle + m_core->get_gpu()->gpu_tot_sim_cycle);
  m_response_fifo.push_back(mf);
}

void ldst_unit_sm::flush() {
  // Flush L1D cache
  m_L1D->flush();
}

void ldst_unit_sm::issue(register_set_uniptr &reg_set, unsigned int icnt_id) {
  std::unique_ptr<warp_inst_t> new_mem_inst = std::make_unique<warp_inst_t>(m_config);
  reg_set.move_out_to(new_mem_inst);
  std::shared_ptr<warp_inst_t> inst = std::move(new_mem_inst);
  // record how many pending register writes/memory accesses there are for this
  // instruction
  assert(inst->empty() == false);

  if ((inst->space.get_type() == global_space) ||
      (inst->space.get_type() == local_space) ||
      (inst->space.get_type() == param_space_local)) {
    m_sm->m_sm_stats.m_stats_map["total_accesses_l1d_instructions"]->increment_with_integer(
        inst->accessq_count());
    m_sm->m_sm_stats.m_stats_map["total_l1d_instructions"]->increment_with_integer(1);
  } else if (inst->space.get_type() == shared_space) {
    m_sm->m_sm_stats.m_stats_map["total_conflicts_shared_instructions"]->increment_with_integer(
        inst->get_num_cycles());
    m_sm->m_sm_stats.m_stats_map["total_shared_instructions"]->increment_with_integer(1);
  }

  inst->op_pipe = MEM__OP;
  inst->m_icnt_mem_pipe_id = m_reserved_idx_icnt_to_shmem;
  // stat collection
  m_core->mem_instruction_stats(*inst);
  m_core->incmem_stat(m_core->get_config()->warp_size, 1);

  warp_inst_t* inst_ptr = inst.get();
  m_core->incexecstat(inst_ptr);
  
  assert(icnt_id == inst->get_mem_pipe_icnt_id());
  inst->change_ldgsts_state();

  if(inst->is_load()) {
    unsigned long long gpu_cycle = m_sm->get_current_gpu_cycle();
    if(m_config->measure_coalescing_potential_stats) {
      if ((inst->space.get_type() == global_space) ||
        (inst->space.get_type() == local_space) ||
        (inst->space.get_type() == param_space_local)) {
        m_coalescing_stats_l1d->registerInst(gpu_cycle,  inst.get());
      }else if(inst->space.get_type() == shared_space) {
        m_coalescing_stats_sharedmem->registerInst(gpu_cycle, inst.get());
      }else if(inst->space.get_type() == const_space) {
        m_coalescing_stats_const->registerInst(gpu_cycle, inst.get());
      }
    }
  }
  if(inst->m_is_ldgsts && (inst->m_ldgsts_state == STORE_STAGE)) {
    m_prt->reactivate_entry(inst);
  }else {
    m_prt->assign_entry(inst);
  }
}

void ldst_unit_sm::push_to_wb_icnt(warp_inst_t inst, unsigned int icnt_id) {
  if(inst.m_is_ldgsts) {
    assert(inst.m_ldgsts_state == LOAD_STAGE);
    assert(!m_ldgsts_icnt_between_ldg_and_sts_part1);
    inst.ldgsts_change_to_sts_mode(m_sm->get_gpu());
    m_ldgsts_icnt_between_ldg_and_sts_part1 = std::make_unique<warp_inst_t>(inst);
  }else {
    assert(!m_pending_wbs_per_subcore[icnt_id]);
    m_pending_wbs_per_subcore[icnt_id] = std::make_unique<warp_inst_t>(inst);
  }
}

bool ldst_unit_sm::is_possible_to_push_to_wb_icnt(unsigned int icnt_id, bool is_ldgsts) {
  bool res = false;
  if(is_ldgsts) {
    res = !m_ldgsts_icnt_between_ldg_and_sts_part1;
  }else {
    res = !m_pending_wbs_per_subcore[icnt_id];
  }
  return res;
}

void ldst_unit_sm::writeback(unsigned int icnt_id) {
  bool is_icnt_to_shmem = icnt_id == m_reserved_idx_icnt_to_shmem;
  if(is_possible_to_push_to_wb_icnt(icnt_id, is_icnt_to_shmem)) {
    std::shared_ptr<warp_inst_t> inst = m_prt->pop_entries(icnt_id);
    if(inst != nullptr) {
      push_to_wb_icnt(*inst, icnt_id);
    }
  }
  if(icnt_id == m_reserved_idx_icnt_to_shmem && m_ldgsts_icnt_between_ldg_and_sts_part1 && m_reception_ports[m_reserved_idx_icnt_to_shmem]->has_free()) {
    assert(m_current_num_normal_mem_inst > 0);
    move_warp_uniptr(m_reception_ports[m_reserved_idx_icnt_to_shmem]->get_free_smartptr(), m_ldgsts_icnt_between_ldg_and_sts_part1);
    m_ldgsts_icnt_between_ldg_and_sts_part1.reset();
    m_current_num_normal_mem_inst--;
  }else if ((icnt_id != m_reserved_idx_icnt_to_shmem) && m_pending_wbs_per_subcore[icnt_id] &&
      m_result_ports[icnt_id]->has_free()) {
    // process next instruction that is going to writeback
    move_warp_uniptr(m_result_ports[icnt_id]->get_free_smartptr(), m_pending_wbs_per_subcore[icnt_id]);
    m_pending_wbs_per_subcore[icnt_id].reset();
    std::unique_ptr<warp_inst_t> &aux_wb = m_result_ports[icnt_id]->get_ready_smartptr();
    if(aux_wb->space.get_type() == shared_space) {
      assert(m_current_num_shared_mem_inst > 0);
      m_current_num_shared_mem_inst--;
    }else {
      assert(m_current_num_normal_mem_inst > 0);
      m_current_num_normal_mem_inst--;
    }
    
  }
}

unsigned ldst_unit_sm::clock_multiplier() const {
  // to model multiple read port, we give multiple cycles for the memory units
  if (m_config->mem_unit_ports)
    return m_config->mem_unit_ports;
  else
    return m_config->mem_warp_parts;
}

void ldst_unit_sm::reset_is_this_l1d_bank_allocated_this_cycle() {
  is_already_dispatched_to_shared_mem_this_cycle = false;
  is_already_dispatched_to_texture_mem_this_cycle = false;
  is_already_dispatched_to_constant_mem_this_cycle = false;
}

void ldst_unit_sm::solve_next_missed_access(cache_t *cache, bool is_constant) {
  if(cache->access_ready()) {
    mem_fetch *mf = cache->next_access();
    warp_inst_t &inst = mf->get_inst();
    if(is_constant) {
      assert(inst.is_load());
    }
    pending_access_logic(mf->get_access().get_access_coal_info().m_prts_requesting);
    delete mf;
  }
}

void ldst_unit_sm::cycle() {
  global_shared_latency_queue_cycle();
  for (unsigned int i = 0; i < m_num_icnt_and_subcores_clients; i++) {
    unsigned int icnt_id =
        (i + m_writeback_arb_between_icnt_and_subcores) % m_num_icnt_and_subcores_clients;
    writeback(icnt_id);
  }
  m_writeback_arb_between_icnt_and_subcores =
      (m_writeback_arb_between_icnt_and_subcores + 1) % m_num_icnt_and_subcores_clients;

  solve_next_missed_access(m_L1T, false);
  solve_next_missed_access(m_L1C, true);
  solve_next_missed_access(m_L1D, false);

  if (!m_response_fifo.empty()) {
    mem_fetch *mf = m_response_fifo.front();
    if (mf->get_access_type() == TEXTURE_ACC_R) {
      if (m_L1T->fill_port_free()) {
        m_L1T->fill(mf, m_core->get_gpu()->gpu_sim_cycle +
                            m_core->get_gpu()->gpu_tot_sim_cycle);
        m_response_fifo.pop_front();
      }
    } else if (mf->get_access_type() == CONST_ACC_R) {
      if (m_L1C->fill_port_free()) {
        mf->set_status(IN_SHADER_FETCHED,
                       m_core->get_gpu()->gpu_sim_cycle +
                           m_core->get_gpu()->gpu_tot_sim_cycle);
        m_L1C->fill(mf, m_core->get_gpu()->gpu_sim_cycle +
                            m_core->get_gpu()->gpu_tot_sim_cycle);
        m_response_fifo.pop_front();
      }
    } else if(mf->get_access_type() == GRID_BARRIER_ACC) {
      m_sm->clear_gridbar(mf->get_kernel_id());
      m_response_fifo.pop_front();
      delete mf;
    }else if(mf->get_access_type() == TLB_MISS_ACC_DATA) {
      m_response_fifo.pop_front();
    }else {
      if (mf->get_type() == WRITE_ACK ||
          (m_config->gpgpu_perfect_mem && mf->get_is_write())) {
        m_core->store_ack(mf);
        m_response_fifo.pop_front();
        delete mf;
      } else {
        assert(!mf->get_is_write());  // L1 cache is write evict, allocate line
                                      // on load miss only

        bool bypassL1D = false;
        if (CACHE_GLOBAL == mf->get_inst().cache_op || (m_L1D == NULL)) {
          bypassL1D = true;
        } else if (mf->get_access_type() == GLOBAL_ACC_R ||
                   mf->get_access_type() ==
                       GLOBAL_ACC_W) {  // global memory access
          if (m_core->get_config()->gmem_skip_L1D) bypassL1D = true;
        }
        if (bypassL1D) {
          mf->set_status(IN_SHADER_FETCHED,
                          m_core->get_gpu()->gpu_sim_cycle +
                              m_core->get_gpu()->gpu_tot_sim_cycle);
          m_response_fifo.pop_front();
          pending_access_logic(mf->get_access().get_access_coal_info().m_prts_requesting);
          delete mf;
        } else {
          if (m_L1D->fill_port_free()) {
            m_L1D->fill(mf, m_core->get_gpu()->gpu_sim_cycle +
                                m_core->get_gpu()->gpu_tot_sim_cycle);
            m_response_fifo.pop_front();
          }
        }
      }
    }
  }

  cache_cycles();
  reset_is_this_l1d_bank_allocated_this_cycle();

  for(unsigned int i = 0;  i < m_config->m_L1D_config.l1_banks; i++) {
    for(unsigned int j = 0; (j < m_config->memory_l1d_max_lookups_per_cycle_per_bank) && !m_access_queue_to_l1d_postTLB[i]->empty(); j++) {
      execute_cache_dispatch(m_access_queue_to_l1d_postTLB[i], m_L1D, [this](cache_t &cache, mem_access_t *acc) {
        return this->dispatch_to_memory_access_queue_l1Dcache(cache, acc);
      });
    }
  }
  execute_cache_dispatch(&m_access_queue_to_l1c, m_L1C, [this](cache_t &cache, mem_access_t *acc) {
      return this->dispatch_to_memory_access_queue_l1Ccache(cache, acc);
  });
  execute_cache_dispatch(&m_access_queue_to_l1t, m_L1T, [this](cache_t &cache, mem_access_t *acc) {
      return this->dispatch_to_memory_access_queue_l1Tcache(cache, acc);
  });
  
  dispatch_access_directly_to_l2();
  execute_miscellaneous_dispatch();
  shared_dispatch();

  bool can_continue_this_bank = true;
  
  for(unsigned int i = 0; i < m_config->m_L1D_config.l1_banks; i++) {
    can_continue_this_bank = true;
    for(unsigned int j = 0; (j < m_config->memory_l1d_max_lookups_per_cycle_per_bank) && !m_access_queue_to_l1d_preTLB[i]->empty() && can_continue_this_bank; j++) {
      mem_access_t *acc = m_access_queue_to_l1d_preTLB[i]->front();
      cache_request_status tlb_acc = HIT;
      if((tlb_acc == HIT)) {
        if(!m_access_queue_to_l1d_postTLB[i]->full()) {
          m_access_queue_to_l1d_postTLB[i]->push(acc);
          m_access_queue_to_l1d_preTLB[i]->pop();
        }else {
          can_continue_this_bank = false;
        }
      }else if((tlb_acc == MISS) || (tlb_acc == MSHR_HIT)) {
        m_access_queue_to_l1d_preTLB[i]->pop(); 
      }else {
        assert(tlb_acc == RESERVATION_FAIL);
        can_continue_this_bank = false;
      }
    }
  }

  unsigned int num_trials = 0;
  while(!m_next_access_to_queue.empty() && (num_trials < m_config->memory_l1d_max_lookups_per_cycle_per_bank)) {
    bool inserted_acc = false;
    mem_access_t *acc_candidate = m_next_access_to_queue.front();
    if((acc_candidate->get_space() == global_space) || (acc_candidate->get_space() == local_space) || (acc_candidate->get_space() == param_space_local)) {
      if(acc_candidate->is_l1d_bypass()) {
        if(!m_access_queue_to_bypass_to_l2.full()) {
          inserted_acc = true;
          m_access_queue_to_bypass_to_l2.push(acc_candidate);
        }
      }else {
        unsigned int id_bank = acc_candidate->get_l1d_bank(); 
        assert(id_bank < m_config->m_L1D_config.l1_banks);
        if(!m_access_queue_to_l1d_preTLB[id_bank]->full()) {
          inserted_acc = true;
          m_access_queue_to_l1d_preTLB[id_bank]->push(acc_candidate);
        }
      }
    }else if(acc_candidate->get_space() == shared_space) {
      if(!m_access_queue_to_shmem.full()) {
        inserted_acc = true;
        m_access_queue_to_shmem.push(acc_candidate);
      }
    }else if(acc_candidate->get_space() == const_space) {
       if(!m_access_queue_to_l1c.full()) {
        inserted_acc = true;
        m_access_queue_to_l1c.push(acc_candidate);
       }
    }else if((acc_candidate->get_space() == tex_space) || (acc_candidate->get_space() == surf_space)) {
      if(!m_access_queue_to_l1t.full()) {
        inserted_acc = true;
        m_access_queue_to_l1t.push(acc_candidate);
      }
    }else if(acc_candidate->get_space() == miscellaneous_space) {
      if(!m_access_queue_to_miscellaneous.full()) {
        inserted_acc = true;
        m_access_queue_to_miscellaneous.push(acc_candidate);
      }
    }else {
      std::cout << "Error: Invalid access type" << std::endl;
      fflush(stdout);
      abort();
    }
    num_trials++;
    if(inserted_acc) {
      m_next_access_to_queue.pop();
    }else {
      break;
    }
  }
  m_prt->management_entries_to_process();
  if(m_config->is_interwarp_coalescing_enabled) {
    bool need_to_drain_intercoalescing_unit = false;
    // Priority for accesses that cannot be coalesced
    for(unsigned int i = 0; (i < m_next_access_to_intercoalescing.size()) && (m_next_access_to_queue.size() < m_config->memory_l1d_max_lookups_per_cycle_per_bank) && !need_to_drain_intercoalescing_unit; i++) {
      if(m_next_access_to_intercoalescing[i]->get_inst() != nullptr && m_next_access_to_intercoalescing[i]->get_inst()->is_any_kind_of_barrier()) {
        need_to_drain_intercoalescing_unit = !m_intercoalescing_unit->is_empty();
      }
      if(!need_to_drain_intercoalescing_unit && !m_intercoalescing_unit->access_is_candidate_to_be_inserted(m_next_access_to_intercoalescing[i])) {
        m_next_access_to_queue.push(m_next_access_to_intercoalescing[i]);
        m_next_access_to_intercoalescing.erase(m_next_access_to_intercoalescing.begin() + i);
        get_SM()->m_sm_stats.m_stats_map["total_accesses"]->increment_with_integer(1);
      }
    }

    if( (m_next_access_to_queue.size() < m_config->memory_l1d_max_lookups_per_cycle_per_bank) && m_intercoalescing_unit->can_pop_access() ) {
      m_next_access_to_queue.push(m_intercoalescing_unit->pop_access(need_to_drain_intercoalescing_unit));
    }

    auto it_acc_to_coal = m_next_access_to_intercoalescing.begin();
    while(it_acc_to_coal != m_next_access_to_intercoalescing.end()) {
      if(m_intercoalescing_unit->access_is_candidate_to_be_inserted(*it_acc_to_coal)) {
        bool is_inserted = m_intercoalescing_unit->insert_access(*it_acc_to_coal);
        if(is_inserted) {
          m_next_access_to_intercoalescing.erase(it_acc_to_coal);
          get_SM()->m_sm_stats.m_stats_map["total_accesses"]->increment_with_integer(1);
        }else {
          it_acc_to_coal++;
        }
      }else {
        it_acc_to_coal++;
      }
    }
    m_prt->get_accesses_to_coalescing(m_next_access_to_intercoalescing);
  }else {
    m_prt->get_access_to_next_stage(m_next_access_to_queue);
  }

  bool has_been_issued = false;

  if(m_reception_ports[m_reserved_idx_icnt_to_shmem]->has_ready() && !m_global_shared_latency_queue_for_ldgsts[m_config->memory_global_shared_latency_for_ldgsts - 1]){
    m_current_num_shared_mem_inst++;
    std::unique_ptr<warp_inst_t> aux = std::make_unique<warp_inst_t>(m_config);
    move_warp_uniptr(aux, m_reception_ports[m_reserved_idx_icnt_to_shmem]->get_ready_smartptr());
    m_global_shared_latency_queue_for_ldgsts[m_config->memory_global_shared_latency_for_ldgsts - 1] = std::move(aux);
  }
  
  if(m_ldgsts_icnt_between_ldg_and_sts_part2) {
    m_ldgsts_aux.move_in(m_ldgsts_icnt_between_ldg_and_sts_part2);
    m_ldgsts_icnt_between_ldg_and_sts_part2.reset();
    issue(m_ldgsts_aux, m_reserved_idx_icnt_to_shmem);
    has_been_issued = true;
  }

  unsigned int icnt_id = 0;
  for (unsigned int i = 0; (i < m_core->get_num_subcores()) &&  !has_been_issued && !m_prt->is_full() &&
        m_core->can_send_inst_from_subcore_to_sm_shared_pipeline(); i++) {
    icnt_id = (m_dispatch_subpipeline_arb_between_icnt_and_subcores + i) % m_core->get_num_subcores();
    if (m_reception_ports[icnt_id]->has_ready()) {
      if (m_reception_ports[icnt_id]->get_ready()->space.get_type() ==
          shared_space) {
        if (m_current_num_shared_mem_inst <
            m_config->memmory_max_concurrent_requests_shmem_per_sm) {
          m_current_num_shared_mem_inst++;
          has_been_issued = true;
        } else {
          has_been_issued = false;
        }
      } else {
        if (m_current_num_normal_mem_inst <
            m_config->memmory_max_concurrent_requests_standard_per_sm) {
          m_current_num_normal_mem_inst++;
          has_been_issued = true;
        } else {
          has_been_issued = false;
        }
      }
      if (has_been_issued) {
        issue(*m_reception_ports[icnt_id], icnt_id);
        break;
      }
    }
  }

  if(has_been_issued) {
    m_dispatch_subpipeline_arb_between_icnt_and_subcores = (icnt_id + 1) % m_core->get_num_subcores();
  }else {
    m_dispatch_subpipeline_arb_between_icnt_and_subcores = (m_dispatch_subpipeline_arb_between_icnt_and_subcores + 1) % m_core->get_num_subcores();
  }

  if(m_config->is_interwarp_coalescing_enabled && (m_config->interwarp_coalescing_selection_policy == InterWarpCoalescingSelectionPolicies::WARPPOOL_HYBRID) &&
      ((m_sm->get_current_gpu_cycle() % m_config->interwarp_coalescing_quanta) == 0 ) ) {
    float quanta_miss_ratio = m_L1D->get_tag_array()->quanta_miss_ratio();
    if(quanta_miss_ratio > m_config->interwarp_coalescing_quanta_warppool_policy_miss_ratio_threshold) {
      if(m_intercoalescing_unit->get_warppool_selection_policy() == InterWarpCoalescingSelectionPolicies::IWCOAL_OLDEST) {
        m_intercoalescing_unit->change_warppool_current_policy(InterWarpCoalescingSelectionPolicies::GTL_WARPID);
      }else {
        m_intercoalescing_unit->change_warppool_current_policy(InterWarpCoalescingSelectionPolicies::IWCOAL_OLDEST);
      }
    }
    m_L1D->get_tag_array()->clear_quanta_stats();
  }

}

void ldst_unit_sm::dispatch_access_directly_to_l2() {
  if(!m_access_queue_to_bypass_to_l2.empty()) {
    mem_access_t* acc = m_access_queue_to_bypass_to_l2.front();
    unsigned control_size =
      acc->get_inst()->is_store() ? WRITE_PACKET_SIZE : READ_PACKET_SIZE;
    unsigned size = acc->get_size() + control_size;
    if (m_icnt->full(size, acc->get_inst()->is_store() || acc->get_inst()->isatomic())) {
      bool iswrite = acc->get_inst()->is_store();
      mem_stage_access_type access_type;
      if (acc->get_inst()->space.is_local()) {
        access_type = (iswrite) ? L_MEM_ST : L_MEM_LD;
      }else {
        access_type = (iswrite) ? G_MEM_ST : G_MEM_LD;
      }
      std::string stat_name = "gpgpu_stall_shd_mem[" + mem_stage_access_type_to_string(static_cast<mem_stage_access_type>(access_type)) + "][ICNT_RC_FAIL]";
      m_sm->m_sm_stats.m_stats_map[stat_name]->increment_with_integer(1);
    } else {
      if (acc->get_inst()->isatomic() && acc->is_last_access()) {
        acc->get_inst()->do_atomic();
        m_core->decrement_atomic_count(acc->get_inst()->warp_id(), acc->get_inst()->active_count());
      }
      mem_fetch *mf =
          m_mf_allocator->alloc(*(acc->get_inst()), *acc,m_sm->get_current_gpu_cycle());
      m_icnt->push(mf);
      if (acc->get_inst()->is_store()) {
        m_core->inc_store_req(acc->get_inst()->warp_id());
        pending_access_logic(acc->get_access_coal_info().m_prts_requesting);
      }
      m_access_queue_to_bypass_to_l2.pop();
      delete acc;
    }
  }
}

void ldst_unit_sm::execute_cache_dispatch(AccessQueue *qu, cache_t *cache, std::function<mem_stage_stall_type(cache_t&, mem_access_t*)> func_process) {
  if(!qu->empty()) {
    mem_access_t *acc = qu->front();
    assert(acc->get_inst()->m_prt_assigned);
    mem_stage_stall_type fail = func_process(*cache, acc);
    if (fail == NO_RC_FAIL) {
      qu->pop();
      delete acc;
    }
  }
}

void ldst_unit_sm::execute_miscellaneous_dispatch() {
  if(!m_access_queue_to_miscellaneous.empty()) {
    mem_access_t *acc = m_access_queue_to_miscellaneous.front();
    assert(acc->get_inst()->m_prt_assigned);
    bool completed = true;
    if(acc->get_inst()->op == GRID_BARRIER_OP) {
      unsigned control_size = READ_PACKET_SIZE;
      unsigned size = acc->get_size() + control_size;
      if (m_icnt->full(size, acc->get_inst()->is_store() || acc->get_inst()->isatomic())) {
        completed = false;
      }else {
        mem_fetch *mf =
            m_mf_allocator->alloc(*(acc->get_inst()), *acc,m_sm->get_current_gpu_cycle());
        mf->set_kernel_id(m_sm->get_kernel_id(acc->get_inst()->warp_id()));
        m_icnt->push(mf);
      }
    }
    if(completed) {
      pending_access_logic(acc->get_access_coal_info().m_prts_requesting);
      m_access_queue_to_miscellaneous.pop();
      delete acc;
    }
  }
}

void ldst_unit_sm::cache_cycles() {
  //SHARED MEMORY Begin
  if (m_shmem_pipeline[0] != nullptr) {
    if (m_shmem_pipeline[0]->get_inst()->isatomic() && m_shmem_pipeline[0]->is_last_access()) {
      m_shmem_pipeline[0]->get_inst()->do_atomic();
      m_core->decrement_atomic_count(
          m_shmem_pipeline[0]->get_inst()->warp_id(), m_shmem_pipeline[0]->get_inst()->active_count());
          // m_shmem_pipeline[0]->get_inst()->warp_id(), 1);
    }
    pending_access_logic(m_shmem_pipeline[0]->get_access_coal_info().m_prts_requesting);
    delete m_shmem_pipeline[0];
    m_shmem_pipeline[0] = nullptr;
  }

  for (unsigned stage = 0; (stage + 1) < m_config->maximum_shared_memory_latency_at_sm_structure; stage++) {
    if ((m_shmem_pipeline[stage] == nullptr) && (m_shmem_pipeline[stage + 1] != nullptr)) {
      m_shmem_pipeline[stage] = m_shmem_pipeline[stage + 1];
      m_shmem_pipeline[stage + 1] = nullptr;
    }
  }
  //SHARED MEMORY END
  m_L1T->cycle();
  m_L1C->cycle();
  if (m_L1D) {
    m_L1D->cycle();
    if (m_config->maximum_l1d_latency_at_sm_structure > 0) L1_latency_queue_cycle();
    if (m_config->constant_cache_latency_at_sm_structure > 0) L1_constant_cache_latency_queue_cycle();
  }
}

void ldst_unit_sm::global_shared_latency_queue_cycle() {
  if(m_global_shared_latency_queue_for_ldgsts[0] && !m_ldgsts_icnt_between_ldg_and_sts_part2) {
    m_ldgsts_icnt_between_ldg_and_sts_part2 = std::move(m_global_shared_latency_queue_for_ldgsts[0]);
    m_global_shared_latency_queue_for_ldgsts[0] = nullptr;
  }

  for (unsigned stage = 0; stage < m_config->memory_global_shared_latency_for_ldgsts - 1;
       stage++) {
    if (!m_global_shared_latency_queue_for_ldgsts[stage]) {
      m_global_shared_latency_queue_for_ldgsts[stage] = std::move(m_global_shared_latency_queue_for_ldgsts[stage + 1]);
      m_global_shared_latency_queue_for_ldgsts[stage + 1] = nullptr;
    }
  }
}

read_only_cache *ldst_unit_sm::get_L1C() { return m_L1C; }
l1_cache*ldst_unit_sm::get_L1D() { return m_L1D; }
SM* ldst_unit_sm::get_SM() { return m_sm; }

unsigned long long ldst_unit_sm::get_instruction_id(warp_inst_t* inst, unsigned int idx) {
  unsigned long long res = inst->get_unique_inst_id();
  assert(res != 0);
  if(idx >= inst->get_extra_trace_instruction_info().get_num_destination_registers()) {
    res = 0;
  }
  return res;
}

void ldst_unit_sm::print_cache_stats(FILE *fp, unsigned &dl1_accesses,
                                     unsigned &dl1_misses) {
  if (m_L1D) {
    m_L1D->print(fp, dl1_accesses, dl1_misses);
  }
}

void ldst_unit_sm::get_cache_stats(cache_stats &cs) {
  // Adds stats to 'cs' from each cache
  if (m_L1D) cs += m_L1D->get_stats();
  if (m_L1C) cs += m_L1C->get_stats();
  if (m_L1T) cs += m_L1T->get_stats();
}

void ldst_unit_sm::get_L1D_sub_stats(struct cache_sub_stats &css) const {
  if (m_L1D) m_L1D->get_sub_stats(css);
}
void ldst_unit_sm::get_L1C_sub_stats(struct cache_sub_stats &css) const {
  if (m_L1C) m_L1C->get_sub_stats(css);
}
void ldst_unit_sm::get_L1T_sub_stats(struct cache_sub_stats &css) const {
  if (m_L1T) m_L1T->get_sub_stats(css);
}

void ldst_unit_sm::print(FILE *fout) const {
  fprintf(fout, "LD/ST unit: \n");
  fprintf(fout, "ID of icnt to shared memory = %u\n",
          m_reserved_idx_icnt_to_shmem);
  fprintf(fout, "Dispatch reception to address calculation latch\n");
  for (unsigned int i = 0; i < m_num_icnt_and_subcores_clients; i++) {
    fprintf(fout, "Subcore %u: ", i);
    m_reception_ports[i]->print(fout);
  }
  fprintf(fout, "Wait table to desired execution memory unit latch\n");
  fprintf(fout, "LD/ST stall condition per subcore: ");
  for (unsigned int i = 0; i < m_num_icnt_and_subcores_clients; i++) {
    fprintf(fout, "MEM_RC_FAIL of subcore %u = ", i);
    if (m_mem_rc_icnt_and_subcores[i] != NO_RC_FAIL) {
      switch (m_mem_rc_icnt_and_subcores[i]) {
        case BK_CONF:
          fprintf(fout, "BK_CONF");
          break;
        case MSHR_RC_FAIL:
          fprintf(fout, "MSHR_RC_FAIL");
          break;
        case ICNT_RC_FAIL:
          fprintf(fout, "ICNT_RC_FAIL");
          break;
        case COAL_STALL:
          fprintf(fout, "COAL_STALL");
          break;
        case WB_ICNT_RC_FAIL:
          fprintf(fout, "WB_ICNT_RC_FAIL");
          break;
        case WB_CACHE_RSRV_FAIL:
          fprintf(fout, "WB_CACHE_RSRV_FAIL");
          break;
        case N_MEM_STAGE_STALL_TYPE:
          fprintf(fout, "N_MEM_STAGE_STALL_TYPE");
          break;
        default:
          abort();
      }
      fprintf(fout, "\n");
    } else {
      fprintf(fout, "NO_RC_FAIL\n");
    }
  }

  fprintf(fout, "LD/ST wb    = ");
  for (unsigned int i = 0; i < m_num_icnt_and_subcores_clients; i++) {
    m_evaluating_wb_icnt_and_subcores[i].print(fout);
  }
  fprintf(fout, "LD/ST WB back to subcore    = ");
  for (unsigned int i = 0; i < m_core->get_num_subcores(); i++) {
    fprintf(fout, "Subcore %u: ", i);
    m_result_ports[i]->print(fout);
  }
  fprintf(fout, "LD/ST wb arbiter   = ");
  for (unsigned int i = 0; i < m_num_icnt_and_subcores_clients; i++) {
    fprintf(fout, "Subcore %u has priority for: %u ", i,
            m_writeback_arb_icnt_and_subcores[i]);
  }
  fprintf(fout, "\n");
  fprintf(
      fout,
      "Last LD/ST writeback @ %llu + %llu (gpu_sim_cycle+gpu_tot_sim_cycle)\n",
      m_last_inst_gpu_sim_cycle, m_last_inst_gpu_tot_sim_cycle);
  fprintf(fout, "Pending register writes:\n");
  m_L1C->display_state(fout);
  m_L1T->display_state(fout);
  if (!m_config->m_L1D_config.disabled()) m_L1D->display_state(fout);
  fprintf(fout, "LD/ST response FIFO (occupancy = %zu):\n",
          m_response_fifo.size());
  for (std::list<mem_fetch *>::const_iterator i = m_response_fifo.begin();
       i != m_response_fifo.end(); i++) {
    const mem_fetch *mf = *i;
    mf->print(fout);
  }
  fflush(stdout);
}

coalescingStatsPerSm *ldst_unit_sm::get_coalescingStatPerSm_l1d() {
  return m_coalescing_stats_l1d->getStats();
}

coalescingStatsPerSm *ldst_unit_sm::get_coalescingStatPerSm_sharedmem() {
  return m_coalescing_stats_sharedmem->getStats();
}

coalescingStatsPerSm *ldst_unit_sm::get_coalescingStatPerSm_const() {
  return m_coalescing_stats_const->getStats();
}

void ldst_unit_sm::reset_coalescingHistory() {
  m_coalescing_stats_l1d->resetHistory();
  m_coalescing_stats_sharedmem->resetHistory();
  m_coalescing_stats_const->resetHistory();
}

PendingRequestTable& ldst_unit_sm::get_prt() { return *m_prt; }

unsigned int ldst_unit_sm::get_reserved_idx_icnt_to_shmem() {
  return m_reserved_idx_icnt_to_shmem;
}

PendingRequestTableEntry::PendingRequestTableEntry() : m_inst(nullptr), m_id(0), m_num_pending_accesses_to_solve(0), m_is_free(true), m_assignation_cycle(0) {

}

void PendingRequestTableEntry::set_id(unsigned id) {
  m_id = id;
}

void PendingRequestTableEntry::assign_entry(std::shared_ptr<warp_inst_t> &inst) {
  m_inst = inst;
  m_is_free = false;
  m_num_pending_accesses_to_solve = 0;
  m_total_num_accesses_to_do = inst->accessq_count(); // Only useful for global memory accesses with active threads
}

void PendingRequestTableEntry::release() {
  m_inst = nullptr;
  m_is_free = true;
  m_num_pending_accesses_to_solve = 0;
  m_assignation_cycle = 0;
  m_total_num_accesses_to_do = 0;
}

void PendingRequestTableEntry::decrement_num_pending_accesses_to_solve() {
  assert(m_num_pending_accesses_to_solve > 0);
  m_num_pending_accesses_to_solve--;
}

void PendingRequestTableEntry::increment_num_pending_accesses_to_solve() {
  m_num_pending_accesses_to_solve++;
}

unsigned int PendingRequestTableEntry::get_total_num_accesses_to_do() const { 
  return m_total_num_accesses_to_do; 
}


std::shared_ptr<warp_inst_t> &PendingRequestTableEntry::get_inst() { return m_inst; }
unsigned int PendingRequestTableEntry::get_id() const { return m_id; }
unsigned int PendingRequestTableEntry::get_num_pending_accesses_to_solve() const { return m_num_pending_accesses_to_solve; }
unsigned long long PendingRequestTableEntry::get_assignation_cycle() const { return m_assignation_cycle; }
void PendingRequestTableEntry::set_assignation_cycle(unsigned long long cycle) { m_assignation_cycle = cycle; }
bool PendingRequestTableEntry::is_free() const { return m_is_free; }
bool PendingRequestTableEntry::is_pending_to_receive_requests() const { return m_num_pending_accesses_to_solve > 0; }

void PendingRequestTableEntry::print(FILE *fout) const {
  fprintf(fout, "PRT Entry[%u]: ", m_id);
  
  if (m_is_free) {
      fprintf(fout, "FREE\n");
      return;
  }else {

    if (m_inst == nullptr) {
        fprintf(fout, "ERROR - Not free but null instruction. Assignation cycle: %llu\n", m_assignation_cycle);
        return;
    }

    // Print instruction details
    fprintf(fout, "warp %u PC=0x%llx ", 
            m_inst->warp_id(), 
            (unsigned long long)m_inst->pc);

    // Print operation type
    fprintf(fout, "op=");
    switch (m_inst->op) {
        case LOAD_OP:           fprintf(fout, "LOAD"); break;
        case STORE_OP:          fprintf(fout, "STORE"); break;
        case MEMORY_BARRIER_OP: fprintf(fout, "BARRIER"); break;
        case GRID_BARRIER_OP: fprintf(fout, "GRID_BARRIER"); break;
        case TENSOR_CORE_LOAD_OP:  fprintf(fout, "TENSOR_LD"); break;
        case TENSOR_CORE_STORE_OP: fprintf(fout, "TENSOR_ST"); break;
        default:               fprintf(fout, "OTHER(%d)", m_inst->op);
    }

    // Print memory space
    fprintf(fout, " space=");
    switch (m_inst->space.get_type()) {
        case global_space:      fprintf(fout, "global"); break;
        case shared_space:      fprintf(fout, "shared"); break;
        case const_space:       fprintf(fout, "const"); break;
        case local_space:       fprintf(fout, "local"); break;
        case tex_space:         fprintf(fout, "texture"); break;
        case param_space_local: fprintf(fout, "param"); break;
        case undefined_space:   fprintf(fout, "undefined"); break;
        default:               fprintf(fout, "other(%d)", m_inst->space.get_type());
    }
    fprintf(fout, ". Assignation cycle: %llu\n", m_assignation_cycle);
  }
}

PendingRequestTable::PendingRequestTable(unsigned int num_entries, ldst_unit_sm *ldst_unit_sm) : m_max_num_entries(num_entries), m_ldst_unit_sm(ldst_unit_sm) {
  m_entries.resize(num_entries);
  for(unsigned int i = 0; i < num_entries; i++) {
    m_entries[i].set_id(i);
    m_entries_id_free_list.push(i);
  }
  m_entries_id_pending_list_to_free.resize(m_ldst_unit_sm->get_SM()->get_num_subcores() + 1);
  m_selection_policy = ldst_unit_sm->get_SM()->get_config()->prt_selection_policy;
  m_max_num_entries_to_process_concurrently = ldst_unit_sm->get_SM()->get_config()->number_of_coalescers;
  m_last_warp_id = std::numeric_limits<unsigned int>::max();
  m_last_pc = std::numeric_limits<address_type>::max();
}

bool PendingRequestTable::is_full() {
  return m_entries_id_free_list.empty();
}

bool PendingRequestTable::is_empty() {
  return m_entries_id_free_list.size() == m_max_num_entries;
}

bool PendingRequestTable::are_entries_to_pop_icnt_id(unsigned int icnt_id) {
  return !m_entries_id_pending_list_to_free[icnt_id].empty();
}

void PendingRequestTable::reactivate_entry(std::shared_ptr<warp_inst_t> &inst) {
  unsigned int id = inst->m_prt_id;
  m_entries_id_pending_list_to_process.push_back(id);
  m_entries[id].assign_entry(inst);
  m_entries[id].set_assignation_cycle(m_ldst_unit_sm->get_SM()->get_current_gpu_cycle());
}

void PendingRequestTable::assign_entry(std::shared_ptr<warp_inst_t> &inst) {
  assert(!is_full());
  unsigned int id = m_entries_id_free_list.front();
  m_entries_id_free_list.pop();
  inst->m_prt_assigned = true;
  inst->m_prt_id = id;
  m_entries[id].assign_entry(inst);
  m_entries_id_pending_list_to_process.push_back(id);
  m_entries[id].set_assignation_cycle(m_ldst_unit_sm->get_SM()->get_current_gpu_cycle());
}

void PendingRequestTable::solve_access(unsigned int id) {
  assert(id < m_max_num_entries);
  assert(!m_entries[id].is_free());
  assert(m_entries[id].is_pending_to_receive_requests());

  m_entries[id].decrement_num_pending_accesses_to_solve();
  if(!m_entries[id].is_pending_to_receive_requests() && m_entries[id].get_inst()->accessq_empty()) {
    unsigned int subid = m_entries[id].get_inst()->get_subcore_id();
    if(m_entries[id].get_inst()->m_is_ldgsts && (m_entries[id].get_inst()->m_ldgsts_state == LOAD_STAGE)) {
      subid = m_ldst_unit_sm->get_reserved_idx_icnt_to_shmem();
    }
    m_entries_id_pending_list_to_free[subid].push(id);
  }
}

std::shared_ptr<warp_inst_t> PendingRequestTable::pop_entry(unsigned int icnt_id) {
  assert(!m_entries_id_pending_list_to_free[icnt_id].empty());
  unsigned int id = m_entries_id_pending_list_to_free[icnt_id].front();
  std::shared_ptr<warp_inst_t> res = std::move(m_entries[id].get_inst());
  bool is_ldgsts = res->m_is_ldgsts;
  bool is_ldgsts_store = is_ldgsts && (res->m_ldgsts_state == STORE_STAGE);
  m_entries_id_pending_list_to_free[icnt_id].pop();
  bool skip_wb = res->skip_wb;
  if(is_ldgsts && !is_ldgsts_store) {
    skip_wb = false;
  }
  if(skip_wb || res->is_store()) {
    if(res->space.is_shared()) {
      m_ldst_unit_sm->m_current_num_shared_mem_inst--;
    }else {
      m_ldst_unit_sm->m_current_num_normal_mem_inst--;
    }
    m_ldst_unit_sm->get_SM()->instruction_retirement(res.get());
    res = nullptr;
  }
  if(!is_ldgsts || is_ldgsts_store) {
    m_entries[id].release();
    m_entries_id_free_list.push(id);
  }
  return res;
}

std::shared_ptr<warp_inst_t> PendingRequestTable::pop_entries(unsigned int icnt_id) {
  std::shared_ptr<warp_inst_t> res = nullptr;
  bool something_pop = false;
  while(are_entries_to_pop_icnt_id(icnt_id) && res == nullptr) {
    res = pop_entry(icnt_id);
    something_pop = true;
  }
  if (something_pop) {
    m_ldst_unit_sm->m_last_inst_gpu_sim_cycle = m_ldst_unit_sm->get_SM()->get_gpu()->gpu_sim_cycle;
    m_ldst_unit_sm->m_last_inst_gpu_tot_sim_cycle = m_ldst_unit_sm->get_SM()->get_gpu()->gpu_tot_sim_cycle;
  }
  return res;
}

bool PendingRequestTable::are_entries_to_process_coalescing() {
  return !m_current_entries_id_being_processed.empty();
}

unsigned int PendingRequestTable::oldest_selection_policy() {
  unsigned int id = m_entries_id_pending_list_to_process.front();
  bool is_safe_to_select = true;
  unsigned int value_of_increment = 0;
  if(is_entry_going_to_global_memory(id) && is_entry_going_to_l1d(id)) { 
    value_of_increment = m_entries[id].get_inst()->accessq_count();
    is_safe_to_select = m_ldst_unit_sm->can_entry_be_selected_for_processing(value_of_increment);
  }

  if(is_safe_to_select) {
    m_ldst_unit_sm->increment_num_reserved_associativity_currently_processing(value_of_increment);
    m_entries_id_pending_list_to_process.erase(m_entries_id_pending_list_to_process.begin());
  }else {
    id = std::numeric_limits<unsigned int>::max();
  }
  return id;
}

bool PendingRequestTable::is_entry_going_to_global_memory(unsigned int id) {
  return (m_entries[id].get_inst()->space == global_space )|| (m_entries[id].get_inst()->space == local_space ) || (m_entries[id].get_inst()->space == param_space_local );
}

bool PendingRequestTable::is_entry_going_to_l1d(unsigned int id) {
  bool res = true;
  if((m_entries[id].get_inst()->cache_op == CACHE_GLOBAL) || (m_ldst_unit_sm->get_L1D() == NULL)) {
    res = false;
  }
  return res;
}

unsigned int PendingRequestTable::same_last_warp_id() {
  unsigned int id = std::numeric_limits<unsigned int>::max();
  for(auto it = m_entries_id_pending_list_to_process.begin(); it != m_entries_id_pending_list_to_process.end(); it++) {
    if(m_entries[*it].get_inst()->warp_id() == m_last_warp_id) {
      id = *it;
      m_entries_id_pending_list_to_process.erase(it);
      break;
    }
  }
  return id;
}

unsigned int PendingRequestTable::same_last_pc() {
  unsigned int id = std::numeric_limits<unsigned int>::max();
  for(auto it = m_entries_id_pending_list_to_process.begin(); it != m_entries_id_pending_list_to_process.end(); it++) {
    if(m_entries[*it].get_inst()->pc == m_last_pc) {
      id = *it;
      m_entries_id_pending_list_to_process.erase(it);
      break;
    }
  }
  return id;
}

unsigned int PendingRequestTable::warp_id_N_cluster_priority_and_oldest_inside_each_cluster() {
  std::vector<cluster_prt_candidate> clusters;
  unsigned int candidate_cluster_id = std::numeric_limits<unsigned int>::max();
  unsigned int num_warpid_N_clusters = m_ldst_unit_sm->get_SM()->get_config()->number_of_clusters_for_prt_selection;
  clusters.resize(num_warpid_N_clusters);
  auto it_to_select = m_entries_id_pending_list_to_process.begin();
  for(auto it = m_entries_id_pending_list_to_process.begin(); it != m_entries_id_pending_list_to_process.end(); it++) {
    auto &entry = m_entries[*it];
    unsigned int wid = entry.get_inst()->warp_id();
    unsigned int cluster_id = wid / num_warpid_N_clusters;
    bool changed = false;
    if(entry.get_assignation_cycle() < clusters[cluster_id].m_cycle) {
      clusters[cluster_id].m_cycle = entry.get_assignation_cycle();
      changed = true;
      clusters[cluster_id].m_id = entry.get_id();
      if(cluster_id == candidate_cluster_id) {
        it_to_select = it;
      }
    }
    if(changed && (candidate_cluster_id == std::numeric_limits<unsigned int>::max())) {
      candidate_cluster_id = cluster_id;
      it_to_select = it;
    }else if(changed && (cluster_id < candidate_cluster_id)) {
      candidate_cluster_id = cluster_id;
      it_to_select = it;
    }
  }
  assert(candidate_cluster_id != std::numeric_limits<unsigned int>::max());
  m_entries_id_pending_list_to_process.erase(it_to_select);
  return clusters[candidate_cluster_id].m_id;
}

unsigned int PendingRequestTable::dep_counters_waiting(bool checking_warp_id) {
  unsigned int id = std::numeric_limits<unsigned int>::max();
  bool found = false;
  for(auto it = m_entries_id_pending_list_to_process.begin(); !found && (it != m_entries_id_pending_list_to_process.end()); it++) {
    for(unsigned int wid = 0; (wid < m_ldst_unit_sm->get_SM()->get_config()->max_warps_per_shader) && !found; wid++) {
      auto &waiting_deps_of_warp = m_ldst_unit_sm->get_SM()->m_interwarp_coal_warps_waiting_dep_counter->m_waiting_dep_counters_per_warp[wid].m_waiting_dep_counters;
      for(auto it_deps = waiting_deps_of_warp.begin(); !found && (it_deps != waiting_deps_of_warp.end()); it_deps++) {
        if(m_entries[*it].get_inst()->get_extra_trace_instruction_info().get_control_bits().get_is_new_write_barrier()) {
          bool has_dep_id = it_deps->first == m_entries[*it].get_inst()->get_extra_trace_instruction_info().get_control_bits().get_id_new_write_barrier();
          has_dep_id = checking_warp_id ? (wid == m_entries[*it].get_inst()->warp_id()) : has_dep_id;
          if(has_dep_id) {
            found = true;
            id = *it;
            m_entries_id_pending_list_to_process.erase(it);
          }
        }      
      }
    }
  }
  return id;
}

void PendingRequestTable::management_entries_to_process() {
  while(!m_entries_id_finishing_processed.empty()) {
    unsigned int id = m_entries_id_finishing_processed.front();
    m_entries_id_finishing_processed.erase(m_entries_id_finishing_processed.begin());
    bool erased = false;
    for(auto it_proc = m_current_entries_id_being_processed.begin(); it_proc != m_current_entries_id_being_processed.end(); it_proc++) {
      if(*it_proc == id) {
        m_current_entries_id_being_processed.erase(it_proc);
        erased = true;
        break;
      }
    }
    assert(erased);
  }
  bool checking_warp_id = (m_selection_policy == PRTSelectionPolicies::DEP_COUNT_WAIT_CHECKING_WARP_ID_THEN_OLDEST);
  bool can_continue = true; // this logic may be need to implemented in other policies if interwarp coalescing is used. As I have discarded that line of research, I have not invested time in doing it.
  while(can_continue && (m_current_entries_id_being_processed.size() < m_max_num_entries_to_process_concurrently)
      && !m_entries_id_pending_list_to_process.empty()) {
    unsigned int id = std::numeric_limits<unsigned int>::max();
    switch(m_selection_policy) {
      case PRTSelectionPolicies::OLDEST:
        id = oldest_selection_policy();
        if(id != std::numeric_limits<unsigned int>::max()) {
          m_current_entries_id_being_processed.push_back(id);
        }else {
          can_continue = false;
        }
        break;
      case PRTSelectionPolicies::SAME_LAST_WARP_ID_THEN_OLDEST:
        id = same_last_warp_id();
        if(id == std::numeric_limits<unsigned int>::max()) {
          id = oldest_selection_policy();
          if(id != std::numeric_limits<unsigned int>::max()) {
            m_current_entries_id_being_processed.push_back(id);
          }else {
            can_continue = false;
          }
        }
        break;
      case PRTSelectionPolicies::SAME_LAST_INST_PC_THEN_OLDEST:
        id = same_last_pc();
        if(id == std::numeric_limits<unsigned int>::max()) {
          id = oldest_selection_policy();
          if(id != std::numeric_limits<unsigned int>::max()) {
            m_current_entries_id_being_processed.push_back(id);
          }else {
            can_continue = false;
          }
        }
        break;
      case PRTSelectionPolicies::WARPID_N_CLUSTERS_WITH_OLDEST:
        id = warp_id_N_cluster_priority_and_oldest_inside_each_cluster();
        assert(id != std::numeric_limits<unsigned int>::max());
        m_current_entries_id_being_processed.push_back(id);
        break;
      case PRTSelectionPolicies::DEP_COUNT_WAIT_GENERIC_THEN_OLDEST:
      case PRTSelectionPolicies::DEP_COUNT_WAIT_CHECKING_WARP_ID_THEN_OLDEST:
        id = dep_counters_waiting(checking_warp_id);
        if(id == std::numeric_limits<unsigned int>::max()) {
          id = oldest_selection_policy();
          if(id != std::numeric_limits<unsigned int>::max()) {
            m_current_entries_id_being_processed.push_back(id);
          }else {
            can_continue = false;
          }
        }
        break;
      default:
        std::cout << "Error: Invalid selection policy" << std::endl;
        fflush(stdout);
        abort();
    }
  }
}

void PendingRequestTable::get_accesses_to_coalescing(std::vector<mem_access_t*> &current_accs) {
  auto it_proc = m_current_entries_id_being_processed.begin();
  bool inserted = false;
  while( (current_accs.size() < m_max_num_entries_to_process_concurrently ) && (it_proc != m_current_entries_id_being_processed.end())) {
    unsigned int id = *it_proc;
    mem_access_t *acc = get_next_processed_access(id);
    assert(acc != nullptr);
    current_accs.push_back(acc);
    it_proc++;
    inserted = true;
  }
  if(!inserted) {
    m_ldst_unit_sm->get_SM()->m_sm_stats.m_stats_map["gpgpu_n_stall_dispatch_to_subpipeline_mem"]->increment_with_integer(1);
  }
}

void PendingRequestTable::get_access_to_next_stage(std::queue<mem_access_t*> &current_accs) {
  auto it_proc = m_current_entries_id_being_processed.begin();
  bool inserted = false;
  while( (current_accs.size() < m_max_num_entries_to_process_concurrently ) && (it_proc != m_current_entries_id_being_processed.end())) {
    unsigned int id = *it_proc;
    mem_access_t *acc = get_next_processed_access(id);
    assert(acc != nullptr);
    current_accs.push(acc);
    it_proc++;
    inserted = true;
  }
  if(!inserted && (current_accs.size() == m_ldst_unit_sm->get_SM()->get_config()->number_of_coalescers ) ) {
    m_ldst_unit_sm->get_SM()->m_sm_stats.m_stats_map["gpgpu_n_stall_dispatch_to_subpipeline_mem"]->increment_with_integer(1);
  }
}

mem_access_t* PendingRequestTable::get_next_processed_access(unsigned int id) {
  assert(id < m_max_num_entries);
  mem_access_t *res_acc = nullptr;
  assert(!m_entries[id].is_free());
  if((m_entries[id].get_inst()->active_count() == 0) || (m_entries[id].get_inst()->op == MEMORY_MISCELLANEOUS_OP) || 
      (m_entries[id].get_inst()->is_any_kind_of_barrier()) ) {
    m_entries[id].get_inst()->skip_wb = true;
    m_entries_id_finishing_processed.push_back(id);
    res_acc = new mem_access_t(m_ldst_unit_sm->get_SM()->get_config()->gpgpu_ctx);
    res_acc->set_space(miscellaneous_space);
    res_acc->set_write(false);
    res_acc->set_last_access(true);
    res_acc->set_size(32);
    if(m_entries[id].get_inst()->op == GRID_BARRIER_OP) {
      res_acc->set_type(GRID_BARRIER_ACC);
    }
  }else if((m_entries[id].get_inst()->op == TEXTURE_OP) && m_entries[id].get_inst()->accessq_empty()) {
    // It seems that texture accesses are not being processed because traces have not captured them properly
    m_entries_id_finishing_processed.push_back(id);
    bool is_write = m_entries[id].get_inst()->is_store();
    mem_access_type access_type = TEXTURE_ACC_R;
    res_acc = new mem_access_t(access_type, 0, 32, is_write, m_ldst_unit_sm->get_SM()->get_config()->gpgpu_ctx);
    res_acc->set_space(tex_space);
    res_acc->set_last_access(true);
  }else if((m_entries[id].get_inst()->space == sstarr_space) || (m_entries[id].get_inst()->space == shared_space)) {
    assert(m_entries[id].get_inst()->has_dispatch_delay());
    bool is_write = m_entries[id].get_inst()->is_store();
    mem_access_type access_type = is_write ? LOCAL_ACC_W : LOCAL_ACC_R;
    res_acc = new mem_access_t(access_type, 0, 32, is_write, m_ldst_unit_sm->get_SM()->get_config()->gpgpu_ctx);
    res_acc->set_space(shared_space);
    if(!m_entries[id].get_inst()->dispatch_delay()) {
      m_entries[id].get_inst()->accessq_clear();
      res_acc->set_last_access(true);
      m_entries_id_finishing_processed.push_back(id);
    }else {
      m_ldst_unit_sm->get_SM()->m_sm_stats.m_stats_map["gpgpu_n_shmem_bkconflict"]->increment_with_integer(1);
      m_ldst_unit_sm->get_SM()->get_stats()->gpgpu_n_shmem_bank_access[m_ldst_unit_sm->get_SM()->get_sid()]++;
    }
  }else {
    if(!m_entries[id].get_inst()->accessq_empty()) {
      res_acc = new mem_access_t(m_entries[id].get_inst()->accessq_back());
      m_entries[id].get_inst()->accessq_pop_back();
    }else {
      bool is_write = m_entries[id].get_inst()->is_store();
      mem_access_type access_type = is_write ? LOCAL_ACC_W : LOCAL_ACC_R;
      res_acc = new mem_access_t(access_type, 0, 32, is_write, m_ldst_unit_sm->get_SM()->get_config()->gpgpu_ctx);
    }
    
    res_acc->set_space(m_entries[id].get_inst()->space);
    if(is_entry_going_to_global_memory(id)) {
      res_acc->set_l1d_bank(m_ldst_unit_sm->get_SM()->get_config()->m_L1D_config.set_bank(res_acc->get_addr()));
    }
    if((m_entries[id].get_inst()->cache_op == CACHE_GLOBAL) || (m_ldst_unit_sm->get_L1D() == NULL) ||
       (m_entries[id].get_inst()->space.is_global() && (m_ldst_unit_sm->get_SM()->get_config()->gmem_skip_L1D && (CACHE_L1 != m_entries[id].get_inst()->cache_op)) )) {
      res_acc->set_l1d_bypass(true);
    }
    
    if(m_entries[id].get_inst()->accessq_empty()) {
      res_acc->set_last_access(true);
      m_entries_id_finishing_processed.push_back(id);
      if(is_entry_going_to_global_memory(id) && is_entry_going_to_l1d(id)) {
        m_ldst_unit_sm->decrement_num_reserved_associativity_currently_processing(m_entries[id].get_total_num_accesses_to_do());
      }
    }else {
      //STATS//
      if(res_acc->is_l1d_bypass()) {
        // it goes directly to l2
        m_ldst_unit_sm->get_SM()->m_sm_stats.m_stats_map["gpgpu_n_directly_to_l2_coalescing_conflicts"]->increment_with_integer(1);
      }else if(res_acc->get_space() == const_space) {
        // It goes to constant cache
        m_ldst_unit_sm->get_SM()->m_sm_stats.m_stats_map["gpgpu_n_cmem_coalescing_conflicts"]->increment_with_integer(1);
      }else if((res_acc->get_space() == tex_space ) || (res_acc->get_space() == surf_space )) {
        // It goes to texture cache 
      }else {
        // It goes to l1d
        m_ldst_unit_sm->get_SM()->m_sm_stats.m_stats_map["gpgpu_n_l1cache_coalescing_conflicts"]->increment_with_integer(1);
      }
      
    }
  }
  m_entries[id].increment_num_pending_accesses_to_solve();
  res_acc->set_inst(m_entries[id].get_inst().get());
  res_acc->get_access_coal_info().m_pcs_requesting.insert(res_acc->get_inst()->pc);
  res_acc->get_access_coal_info().m_warp_id_requesting.insert(res_acc->get_inst()->warp_id());
  res_acc->get_access_coal_info().m_prts_requesting.push_back(res_acc->get_inst()->m_prt_id);
  if(res_acc->get_inst()->get_extra_trace_instruction_info().get_control_bits().get_is_new_write_barrier()) {
    res_acc->get_access_coal_info().m_dep_counters_id_requesting.insert(res_acc->get_inst()->get_extra_trace_instruction_info().get_control_bits().get_id_new_write_barrier());
  }
  m_last_warp_id = res_acc->get_inst()->warp_id();
  m_last_pc = res_acc->get_inst()->pc;
  return res_acc;
}

void PendingRequestTable::print(FILE *fout) const {
  // Print summary information
  fprintf(fout, "\nPending Request Table Summary:\n");
  fprintf(fout, "Total entries: %u\n", m_max_num_entries);
  fprintf(fout, "Number of free entries : %zu\n", m_entries_id_free_list.size());
  fprintf(fout, "Number of entries pending to free: %zu\n", m_entries_id_pending_list_to_free.size());
  fprintf(fout, "Number of entries being processed: %zu\n", m_current_entries_id_being_processed.size());
  fprintf(fout, "Number of entries pending to process: %zu\n", m_entries_id_pending_list_to_process.size());
  unsigned int num_pending_list_to_free = 0;
  for(unsigned int i = 0; i < m_entries_id_pending_list_to_free.size(); i++) {
    fprintf(fout, "Pending list to free in icnt: %u: %zu\n", i, m_entries_id_pending_list_to_free[i].size());
    std::queue<unsigned int> pending_list_to_free_copy = m_entries_id_pending_list_to_free[i];
    while(!pending_list_to_free_copy.empty()) {
      fprintf(fout, "Entry ID: %u\n", pending_list_to_free_copy.front());
      num_pending_list_to_free++;
      pending_list_to_free_copy.pop();
    }
  }
  fprintf(fout, "Total number of entries pending to free: %u\n", num_pending_list_to_free);
    
  // Print all entries
  fprintf(fout, "\nDetailed Entry Status:\n");
  for (unsigned int i = 0; i < m_max_num_entries; i++) {
      m_entries[i].print(fout);
  }
}

AccessQueue::AccessQueue(unsigned int max_size) : m_max_size(max_size) {}

AccessQueue::~AccessQueue() {
  while(!m_accesses.empty()) {
    mem_access_t* inst = m_accesses.front();
    m_accesses.pop();
    delete inst;
  }
}

void AccessQueue::push(mem_access_t* access) {
  assert(m_accesses.size() < m_max_size);
  m_accesses.push(access);
}

void AccessQueue::pop() {
  assert(!m_accesses.empty());
  m_accesses.pop();
}

mem_access_t* AccessQueue::front() {
  assert(!m_accesses.empty());
  return m_accesses.front();
}

bool AccessQueue::full() {
  return m_accesses.size() == m_max_size;
}

bool AccessQueue::empty() {
  return m_accesses.empty();
}

unsigned int AccessQueue::size() {
  return m_accesses.size();
}

InterWarpCoalescingUnit::InterWarpCoalescingUnit(ldst_unit_sm* mem_unit, 
  unsigned int num_tables, 
  unsigned int max_size_per_table) 
  : m_ldst_unit_sm(mem_unit),
    m_num_tables(num_tables),
    m_max_size_per_table(max_size_per_table) {
  m_selection_policy = mem_unit->get_SM()->get_config()->interwarp_coalescing_selection_policy;
  m_warppool_current_policy = InterWarpCoalescingSelectionPolicies::IWCOAL_OLDEST;
  m_intercoalescing_tables.resize(num_tables);
  m_last_greedy_warp_id = 0;
}

InterWarpCoalescingUnit::~InterWarpCoalescingUnit() {
  // Clean up any remaining entries in the tables
  for (auto& table : m_intercoalescing_tables) {
    for (auto& entry : table) {
      if (entry.second) {
        delete entry.second;
      }
    }
    table.clear();
  }
}

new_addr_type InterWarpCoalescingUnit::get_addr_signature(new_addr_type addr, memory_space_t space) {
  constexpr unsigned int SPACE_BITS = 4; // If we add more _memory_space_t we must increased it
  return (static_cast<new_addr_type>(space.get_type()) << (sizeof(new_addr_type)*8 - SPACE_BITS)) | 
         (addr & ((static_cast<new_addr_type>(1) << (sizeof(new_addr_type)*8 - SPACE_BITS)) - 1));
}

bool InterWarpCoalescingUnit::insert_access(mem_access_t* acc) {
  assert(acc);
  assert(!acc->is_write());
  bool inserted = false;
  new_addr_type signature = get_addr_signature(acc->get_addr(), acc->get_space());
  
  unsigned int table_idx = 0;
  if(m_num_tables > 1) {
    // Figure out the table in case that there are several
    std::cout << "Error: Multiple tables not supported yet" << std::endl;
    fflush(stdout);
    abort();
  }
  
  // Check if we already have this address in the table
  auto& target_table = m_intercoalescing_tables[table_idx];
  auto it = target_table.find(signature);
  if (it != target_table.end()) {
    // QUE PASA SI VIENE L1 BYPASS y HAY L1D ya ahi o viceversa? De momento que haga lo que decida el primer acceso.
    
    // Append information to the existing entry
    it->second->get_access_coal_info().m_pcs_requesting.insert(acc->get_inst()->pc);
    it->second->get_access_coal_info().m_warp_id_requesting.insert(acc->get_inst()->warp_id());
    it->second->get_access_coal_info().m_prts_requesting.push_back(acc->get_inst()->m_prt_id);
    unsigned int size_acc = std::max(acc->get_size(), it->second->get_size());
    it->second->set_size(size_acc);
    it->second->set_sector_mask(it->second->get_sector_mask() | acc->get_sector_mask());
    if(acc->get_inst()->get_extra_trace_instruction_info().get_control_bits().get_is_new_write_barrier()) {
      it->second->get_access_coal_info().m_dep_counters_id_requesting.insert(acc->get_inst()->get_extra_trace_instruction_info().get_control_bits().get_id_new_write_barrier());
    }
    inserted = true;
    delete acc;
    m_ldst_unit_sm->get_SM()->m_sm_stats.m_stats_map["total_accesses_coalesced"]->increment_with_integer(1);
  }else if(target_table.size() < m_max_size_per_table) {
    acc->set_cycle_inserted_inter_coal(m_ldst_unit_sm->get_SM()->get_current_gpu_cycle());
    target_table[signature] = acc;
    inserted = true;
    m_ldst_unit_sm->get_SM()->m_sm_stats.m_stats_map["total_accesses_not_coalesced"]->increment_with_integer(1);
  }
  return inserted;
}

bool InterWarpCoalescingUnit::access_is_candidate_to_be_inserted(mem_access_t *acc) {
  bool res = true;
  if( acc->is_write() || (acc->get_space() == miscellaneous_space) ||
      (acc->get_space() == tex_space) || (acc->get_space() == surf_space)) { // As texture and surface addresses are not generated at all by traces, we are not confident to coalesce them.
    res = false;
  }else if(acc->get_access_coal_info().m_dep_counters_id_requesting.empty()) {
    res = false;
  }// Falta el de STRONG y el de Atomics
  return res;
}

bool InterWarpCoalescingUnit::can_pop_access() {
  bool can_pop = false;
  for (auto& table : m_intercoalescing_tables) {
    if(!table.empty()) {
      can_pop = true;
      break;
    }
  }
  return can_pop;
}

InterWarpCoalescingSelectionPolicies InterWarpCoalescingUnit::get_warppool_selection_policy() {
  return m_warppool_current_policy;
}

void InterWarpCoalescingUnit::change_warppool_current_policy(InterWarpCoalescingSelectionPolicies new_policy) {
  m_warppool_current_policy = new_policy;
}

pop_interwarp_result InterWarpCoalescingUnit::pop_policy_oldest() {
  pop_interwarp_result res;
  res.m_it_to_pop = m_intercoalescing_tables[0].begin();
  unsigned long long cycle_to_pop = std::numeric_limits<unsigned long long>::max();
  for(unsigned int idx_table = 0; idx_table < m_num_tables; idx_table++) {
    for(auto it = m_intercoalescing_tables[idx_table].begin(); it != m_intercoalescing_tables[idx_table].end(); it++) {
      if(it->second->get_cycle_inserted_inter_coal() < cycle_to_pop) {
        res.m_it_to_pop = it;
        cycle_to_pop = it->second->get_cycle_inserted_inter_coal();
        res.m_found = true;
        res.m_table_idx = idx_table;
      }
    }
  }
  return res;
}

pop_interwarp_result InterWarpCoalescingUnit::pop_policy_gtl_warpid() {
  pop_interwarp_result res_greedy;
  pop_interwarp_result res_lowest;
  pop_interwarp_result res;
  unsigned int lowest_warp_id_candidate = std::numeric_limits<unsigned int>::max();
  for(unsigned int idx_table = 0; (idx_table < m_num_tables) && !res_greedy.m_found; idx_table++) {
    for(auto it = m_intercoalescing_tables[idx_table].begin(); (it != m_intercoalescing_tables[idx_table].end()) && !res_greedy.m_found; it++) {
      if(it->second->get_access_coal_info().m_warp_id_requesting.find(m_last_greedy_warp_id) != it->second->get_access_coal_info().m_warp_id_requesting.end()) {
        res_greedy.m_it_to_pop = it;
        res_greedy.m_found = true;
        res_greedy.m_table_idx = idx_table;
        lowest_warp_id_candidate = m_last_greedy_warp_id;
      }else {
        for(auto wid : it->second->get_access_coal_info().m_warp_id_requesting) {
          if(wid < lowest_warp_id_candidate) {
            lowest_warp_id_candidate = wid;
            res_lowest.m_it_to_pop = it;
            res_lowest.m_found = true;
            res_lowest.m_table_idx = idx_table;
          }
        }
      }
    }
  }
  if(res_greedy.m_found) {
    res = res_greedy;
  }else {
    res = res_lowest;
  }
  if(res.m_found) {
    m_last_greedy_warp_id = lowest_warp_id_candidate;
  }
  return res;
}

pop_interwarp_result InterWarpCoalescingUnit::pop_policy_dep_counters(bool checking_warp_id) {
  pop_interwarp_result res;
  assert(m_num_tables == 1);// DE MOMENTO
  unsigned int idx_table = 0;
  unsigned long long cycle_to_pop = std::numeric_limits<unsigned long long>::max();
  for(auto it_acc = m_intercoalescing_tables[idx_table].begin(); (it_acc != m_intercoalescing_tables[idx_table].end()); it_acc++) {
    for(unsigned int wid = 0; (wid < m_ldst_unit_sm->get_SM()->get_config()->max_warps_per_shader); wid++) { 
      auto &waiting_deps_of_warp = m_ldst_unit_sm->get_SM()->m_interwarp_coal_warps_waiting_dep_counter->m_waiting_dep_counters_per_warp[wid].m_waiting_dep_counters;
      for(auto it_deps = waiting_deps_of_warp.begin(); (it_deps != waiting_deps_of_warp.end()); it_deps++) {
        bool has_dep_id = it_acc->second->get_access_coal_info().m_dep_counters_id_requesting.find(it_deps->first) != it_acc->second->get_access_coal_info().m_dep_counters_id_requesting.end();
        if(has_dep_id)  {
          bool found = checking_warp_id ? (it_acc->second->get_access_coal_info().m_warp_id_requesting.find(wid) != it_acc->second->get_access_coal_info().m_warp_id_requesting.end()) : true;
          if(found && (it_acc->second->get_cycle_inserted_inter_coal() < cycle_to_pop)) {
            res.m_it_to_pop = it_acc;
            res.m_found = true;
            res.m_table_idx = 0; // DE MOMENTO
            cycle_to_pop = it_acc->second->get_cycle_inserted_inter_coal();
          }
        }
      }
    }
  }
  return res;
}

mem_access_t* InterWarpCoalescingUnit::pop_access(bool need_to_drain_intercoalescing_unit) {
  mem_access_t* res = nullptr;
  pop_interwarp_result pop_info;
  bool checking_warp_id = (m_selection_policy == DEP_COUNT_WAIT_OLDEST_INST_IBUFFER_CHECKING_WARP_ID) || (m_selection_policy == DEP_COUNT_WAIT_DETECTED_AT_DECODE_CHECKING_WARP_ID);
  switch(m_selection_policy) {
    case InterWarpCoalescingSelectionPolicies::IWCOAL_OLDEST:
      pop_info = pop_policy_oldest();
      break;
    case GTL_WARPID:
      pop_info = pop_policy_gtl_warpid();
      if(!pop_info.m_found) {
        pop_info = pop_policy_oldest();
      }
      break;
    case WARPPOOL_HYBRID:
      if(m_warppool_current_policy == IWCOAL_OLDEST) {
        pop_info = pop_policy_oldest();
      }else {
        pop_info = pop_policy_gtl_warpid();
      }
      break;
    case InterWarpCoalescingSelectionPolicies::DEP_COUNT_WAIT_OLDEST_INST_IBUFFER_GENERIC:
    case InterWarpCoalescingSelectionPolicies::DEP_COUNT_WAIT_OLDEST_INST_IBUFFER_CHECKING_WARP_ID:
    case InterWarpCoalescingSelectionPolicies::DEP_COUNT_WAIT_DETECTED_AT_DECODE_GENERIC:
    case InterWarpCoalescingSelectionPolicies::DEP_COUNT_WAIT_DETECTED_AT_DECODE_CHECKING_WARP_ID:
      pop_info = pop_policy_dep_counters(checking_warp_id);
      if(!pop_info.m_found && m_ldst_unit_sm->get_SM()->is_any_subcore_problems_of_fordward_progress() && m_ldst_unit_sm->get_prt().is_full()) {
        pop_info = pop_policy_oldest();
      }
      break;
    default:
      std::cout << "Error: Invalid selection policy" << std::endl;
      fflush(stdout);
      abort();
  }
  if(need_to_drain_intercoalescing_unit && !pop_info.m_found) {
    pop_info = pop_policy_oldest();
  }
  if(pop_info.m_found) {
    res = pop_info.m_it_to_pop->second;
    m_intercoalescing_tables[pop_info.m_table_idx].erase(pop_info.m_it_to_pop);
  }
  return res;
}

bool InterWarpCoalescingUnit::is_empty() {
  bool res = true;
  for(unsigned int idx_table = 0; idx_table < m_num_tables; idx_table++) {
    if(!m_intercoalescing_tables[idx_table].empty()) {
      res = false;
      break;
    }
  }
  return res;
}