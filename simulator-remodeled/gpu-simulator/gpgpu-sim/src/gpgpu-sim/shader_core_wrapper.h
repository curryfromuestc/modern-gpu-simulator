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

#include <bitset>
#include "../constants.h"
#include "remodeling/new_stats.h"

typedef std::bitset<WARP_PER_CTA_MAX> warp_set_t;

class gpgpu_sim;
class warp_inst_t;
class shd_warp_t;
class mem_fetch;
class shader_core_config;
class shader_core_stats;
class cache_stats;
class cache_sub_stats;
class kernel_info_t;
class RRS;
class coalescingStatsAcrossSms;

class shader_core_ctx_wrapper {
 public:
  virtual ~shader_core_ctx_wrapper() {}
  virtual gpgpu_sim *get_gpu() = 0;
  virtual const shader_core_config *get_config() const = 0;
  virtual shader_core_stats* get_stats() = 0;
  virtual RRS* get_loog_rrs() = 0;

  virtual void num_cycles_to_stall_SM(unsigned int num_cycles) = 0;

  virtual void cycle() = 0;
  virtual void init() = 0;
  virtual void reinit(unsigned start_thread, unsigned end_thread,
              bool reset_not_completed) = 0;

  virtual void cache_flush() = 0;
  virtual void cache_invalidate() = 0;

  virtual void accept_fetch_response(mem_fetch *mf) = 0;
  virtual void accept_ldst_unit_response(class mem_fetch *mf) = 0;
  virtual bool fetch_unit_response_buffer_full() const = 0;
  virtual bool ldst_unit_response_buffer_full() const = 0;

  virtual void set_kernel(kernel_info_t *k) = 0;;
  virtual kernel_info_t *get_kernel() = 0;
  virtual unsigned int get_sid() const = 0;
  virtual kernel_info_t *get_kernel_info() = 0;
  virtual shd_warp_t *get_shd_warp(int id) = 0;
  virtual void warp_inst_complete(const warp_inst_t &inst) = 0;
  virtual bool ptx_thread_done(unsigned hw_thread_id) const = 0;
  virtual void get_pdom_stack_top_info(unsigned tid, unsigned *pc, unsigned *rpc) const = 0;
  virtual void get_pdom_stack_top_info(unsigned warp_id, const warp_inst_t *pI, unsigned *pc, unsigned *rpc) = 0;

  virtual void set_subcore_req_fetch_L1I_priority(
      int new_subcore_req_fetch_L1I_priority) = 0;
  virtual unsigned int get_num_subcores() = 0;

  virtual bool warp_waiting_at_mem_barrier(unsigned warp_id) = 0;
  virtual bool warp_waiting_at_barrier(unsigned warp_id) const = 0;
  virtual bool warp_waiting_grid_barrier(unsigned warp_id) = 0;
  virtual void broadcast_barrier_reduction(unsigned cta_id, unsigned bar_id,
                                           warp_set_t warps) = 0;


  virtual void decrement_atomic_count(unsigned wid, unsigned n) = 0;

  virtual unsigned int get_n_active_cta() const = 0;
  virtual unsigned int get_not_completed() const = 0;
  virtual unsigned int isactive() const = 0;
  virtual float get_current_occupancy(unsigned long long &active,
                              unsigned long long &total) const = 0;

  virtual void issue_block2core(class kernel_info_t &kernel) = 0;
  virtual bool can_issue_1block(kernel_info_t &kernel) = 0;

  virtual void mem_instruction_stats(const warp_inst_t &inst) = 0;
  virtual void store_ack(class mem_fetch *mf) = 0;
  virtual void inc_store_req(unsigned warp_id) = 0;
  virtual void dec_inst_in_pipeline(unsigned warp_id) = 0;
  
  virtual void display_pipeline(FILE *fout, int print_mem, int mask3bit) const = 0;

  virtual void print_cache_stats(FILE *fp, unsigned &dl1_accesses,
                         unsigned &dl1_misses) = 0;
  virtual void get_cache_stats(cache_stats &cs) = 0;
  virtual void get_L0I_sub_stats(struct cache_sub_stats &css) const = 0;
  virtual void get_L1I_sub_stats(struct cache_sub_stats &css) const = 0;
  virtual void get_L1D_sub_stats(struct cache_sub_stats &css) const = 0;
  virtual void get_L1C_sub_stats(struct cache_sub_stats &css) const = 0;
  virtual void get_L1T_sub_stats(struct cache_sub_stats &css) const = 0;
  virtual void incload_stat() = 0;
  virtual void incstore_stat() = 0;
  virtual void incialu_stat(unsigned active_count, double latency) = 0;
  virtual void incimul_stat(unsigned active_count, double latency) = 0;
  virtual void incimul24_stat(unsigned active_count, double latency) = 0;
  virtual void incimul32_stat(unsigned active_count, double latency) = 0;
  virtual void incidiv_stat(unsigned active_count, double latency)  = 0;
  virtual void incfpalu_stat(unsigned active_count, double latency) = 0;
  virtual void incfpmul_stat(unsigned active_count, double latency) = 0;
  virtual void incfpdiv_stat(unsigned active_count, double latency) = 0;
  virtual void incdpalu_stat(unsigned active_count, double latency) = 0;
  virtual void incdpmul_stat(unsigned active_count, double latency) = 0;
  virtual void incdpdiv_stat(unsigned active_count, double latency) = 0;
  virtual void incsqrt_stat(unsigned active_count, double latency) = 0;
  virtual void inclog_stat(unsigned active_count, double latency) = 0;
  virtual void incexp_stat(unsigned active_count, double latency) = 0;
  virtual void incsin_stat(unsigned active_count, double latency) = 0;
  virtual void inctensor_stat(unsigned active_count, double latency) = 0;
  virtual void inctex_stat(unsigned active_count, double latency) = 0;
  virtual void inc_const_accesses(unsigned active_count) = 0;
  virtual void incsfu_stat(unsigned active_count, double latency) = 0;
  virtual void incsp_stat(unsigned active_count, double latency) = 0;
  virtual void incmem_stat(unsigned active_count, double latency) = 0;
  virtual void incregfile_reads(unsigned active_count) = 0;
  virtual void incregfile_writes(unsigned active_count) = 0;
  virtual void incnon_rf_operands(unsigned active_count) = 0;
  virtual void incspactivelanes_stat(unsigned active_count) = 0;
  virtual void incsfuactivelanes_stat(unsigned active_count) = 0;
  virtual void incfuactivelanes_stat(unsigned active_count) = 0;
  virtual void incfumemactivelanes_stat(unsigned active_count) = 0;
  virtual void inc_simt_to_mem(unsigned n_flits) = 0;
  virtual void incexecstat(warp_inst_t *&inst) = 0;
  virtual void get_icnt_power_stats(long &n_simt_to_mem, long &n_mem_to_simt) const = 0;

  virtual bool get_is_loog_enabled() = 0;
  virtual unsigned long long get_current_gpu_cycle() = 0;
  virtual address_type from_local_pc_to_global_pc_address(address_type local_pc, unsigned int unique_function_id) = 0;
  virtual address_type from_global_pc_address_to_local_pc(address_type global_pc, unsigned int unique_function_id) = 0;

  virtual void create_gpu_per_sm_stats(Element_stats &all_stats) = 0;
  virtual void reset_cycless_access_history() = 0;
  virtual void gather_gpu_per_sm_stats(Element_stats &all_stats, coalescingStatsAcrossSms& coal_stats_l1d, coalescingStatsAcrossSms& coal_stats_const, coalescingStatsAcrossSms& coal_stats_sharedmem) = 0;
  virtual void gather_gpu_per_sm_single_stat(Element_stats &all_stats, std::string stat_name) = 0;
  virtual void increment_sm_stat_by_integer(std::string stat_name, int val_to_increment) = 0;
};