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

// Copyright (c) 2009-2021, Tor M. Aamodt, Wilson W.L. Fung, Andrew Turner,
// Ali Bakhoda, Vijay Kandiah, Nikos Hardavellas, 
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

#ifndef SHADER_H
#define SHADER_H

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <bitset>
#include <deque>
#include <list>
#include <map>
#include <set>
#include <utility>
#include <vector>
#include <memory>

//#include "../cuda-sim/ptx.tab.h"

#include "../abstract_hardware_model.h"
#include "delayqueue.h"
#include "dram.h"
#include "gpu-cache.h"
#include "mem_fetch.h"
#include "scoreboard.h"
#include "scoreboard_reads.h" // MOD. Fix WAR at baseline.
#include "remodeling/ibuffer_remodeled.h" // MOD. Remodeling
#include "remodeling/warp_dependency_state.h" // MOD. Remodeling
#include "remodeling/l0_icnt.h" // MOD. Added L0I
#include "result_bus.h" // MOD. Improved Result bus to take into account conflicts with RF banks
#include <stack>
#include "stats.h"
#include "traffic_breakdown.h"

#include "shader_core_wrapper.h"
# include <omp.h>

#define NO_OP_FLAG 0xFF

/* READ_PACKET_SIZE:
   bytes: 6 address (flit can specify chanel so this gives up to ~2GB/channel,
   so good for now), 2 bytes   [shaderid + mshrid](14 bits) + req_size(0-2 bits
   if req_size variable) - so up to 2^14 = 16384 mshr total
 */

#define READ_PACKET_SIZE 8

// WRITE_PACKET_SIZE: bytes: 6 address, 2 miscelaneous.
#define WRITE_PACKET_SIZE 8

#define WRITE_MASK_SIZE 8

class gpgpu_context;
class ldst_unit_remake; // MOD. Fixed LDST_Unit model
class coalescingStatsAcrossSms;
class Subcore;

void check_kernel_launch_limitation(
    const kernel_info_t &k, const shader_core_config *shader_config,
    shader_core_stats *stats);

class thread_ctx_t {
 public:
  unsigned m_cta_id;  // hardware CTA this thread belongs

  // per thread stats (ac stands for accumulative).
  unsigned n_insn;
  unsigned n_insn_ac;
  unsigned n_l1_mis_ac;
  unsigned n_l1_mrghit_ac;
  unsigned n_l1_access_ac;

  bool m_active;
};

struct function_call_entry_info {
  function_call_entry_info() {
    unique_function_id = 0;
    active_mask.reset();
  }
  unsigned int unique_function_id;
  active_mask_t active_mask;
};

class shd_warp_t {
 public:
  shd_warp_t(class shader_core_ctx_wrapper *shader, unsigned warp_size, shader_core_stats *stats) 
      : m_shader(shader), m_warp_size(warp_size) {
    m_stores_outstanding = 0;
    m_inst_in_pipeline = 0;
    m_IBuffer_remodeled = new IBuffer_Remodeled(shader->get_config(), this, stats); // MOD. Remodeling
    m_dependency_state = new Dependency_State(shader->get_config(), stats); // MOD. Remodeling
    m_last_unique_inst_id = 0;
    m_kernel_id = 0;
    m_gridbar = false;
    reset();
  }

  virtual ~shd_warp_t() {
    delete m_IBuffer_remodeled; // MOD. Remodeling
    delete m_dependency_state; // MOD. Remodeling
  }

  void reset() {
    assert(m_stores_outstanding == 0);
    assert(m_inst_in_pipeline == 0);
    m_imiss_pending = false;
    m_warp_id = (unsigned)-1;
    m_dynamic_warp_id = (unsigned)-1;
    n_completed = m_warp_size;
    m_n_atomic = 0;
    m_membar = false;
    m_done_exit = true;
    m_last_fetch = 0;
    m_next = 0;
    m_last_unique_inst_id = 0;

    // Jin: cdp support
    m_cdp_latency = 0;
    m_cdp_dummy = false;
    while(!m_function_call_stack.empty()) {
      m_function_call_stack.pop();
    }
  }
  void init(address_type start_pc, unsigned cta_id, unsigned wid,
            const std::bitset<MAX_WARP_SIZE> &active,
            unsigned dynamic_warp_id, int shader_id) {
    m_cta_id = cta_id;
    m_warp_id = wid;
    m_dynamic_warp_id = dynamic_warp_id;
    m_next_pc = start_pc;
    assert(n_completed >= active.count());
    assert(n_completed <= m_warp_size);
    n_completed -= active.count();  // active threads are not yet completed
    m_active_threads = active;
    m_done_exit = false;

    // Jin: cdp support
    m_cdp_latency = 0;
    m_cdp_dummy = false;

    m_last_unique_inst_id = 1;

    m_is_pending_store = false; // MOD. Fix load after stores
    m_is_pending_load = false; // MOD. Fix load after stores
  }

  const active_mask_t& get_active_mask() {
    return m_active_threads;
  }

  void push_function_call(unsigned int unique_function_id, active_mask_t active_mask) {
    if(active_mask.any()) {
      function_call_entry_info entry_info;
      entry_info.unique_function_id = unique_function_id;
      entry_info.active_mask = active_mask;
      m_function_call_stack.push(entry_info);
    }
  }

  void pop_function_call(active_mask_t active_mask) {
    assert(!m_function_call_stack.empty());
    m_function_call_stack.top().active_mask ^= active_mask;
    if(m_function_call_stack.top().active_mask.none()) {
      m_function_call_stack.pop();
    }
  }

  unsigned int get_current_unique_function_id_call() {
    assert(!m_function_call_stack.empty());
    return m_function_call_stack.top().unique_function_id;
  }

  bool functional_done() const;
  bool waiting();  // not const due to membar
  bool hardware_done() const;

  bool done_exit() const { return m_done_exit; }

  void set_done_exit() { 
    pop_function_call(m_active_threads);
    m_done_exit = true;
  }

  void print(FILE *fout) const;
  void print_ibuffer(FILE *fout) const;

  void set_scheduler(scheduler_unit* scheduler) { m_scheduler = scheduler; } // MOD. Added L0I
  scheduler_unit* get_scheduler() { return m_scheduler; } // MOD. Added L0I
  bool get_is_pending_store() { return m_is_pending_store; } // MOD. Fix load after stores
  void set_is_pending_store(bool pending) { m_is_pending_store = pending; } // MOD. Fix load after stores
  bool get_is_pending_load() { return m_is_pending_load; } // MOD. Fix load after stores
  void set_is_pending_load(bool pending) { m_is_pending_load = pending; } // MOD. Fix load after stores

  IBuffer_Remodeled* get_IBuffer_remodeled(){ return m_IBuffer_remodeled; } // MOD. Remodeling
  Dependency_State* get_dependency_state(){ return m_dependency_state; } // MOD. Remodeling
  unsigned get_n_completed() const { return n_completed; }
  void set_completed(unsigned lane) {
    assert(m_active_threads.test(lane));
    m_active_threads.reset(lane);
    n_completed++;
  }

  void set_last_fetch(unsigned long long sim_cycle) {
    m_last_fetch = sim_cycle;
  }

  unsigned get_n_atomic() const { return m_n_atomic; }
  void inc_n_atomic() { m_n_atomic++; }
  void dec_n_atomic(unsigned n) { m_n_atomic -= n; }

  bool is_atomic_pending() const { return m_n_atomic > 0; }

  void set_membar() { m_membar = true; }
  void clear_membar() { m_membar = false; }
  bool get_membar() const { return m_membar; }
  void set_gridbar() { m_gridbar = true; }
  void clear_gridbar() { m_gridbar = false; }
  bool get_gridbar() const { return m_gridbar; }
  virtual address_type get_pc() const { return m_next_pc; }
  virtual kernel_info_t* get_kernel_info() const;
  void set_next_pc(address_type pc) { 
    m_next_pc = pc; 
  }

  void store_info_of_last_inst_at_barrier(const warp_inst_t *pI) {
    m_inst_at_barrier = *pI;
  }
  warp_inst_t *restore_info_of_last_inst_at_barrier() {
    return &m_inst_at_barrier;
  }

  void ibuffer_fill(unsigned slot, const warp_inst_t *pI) {
    assert(slot < IBUFFER_SIZE);
    m_ibuffer[slot].m_inst = pI;
    m_ibuffer[slot].m_valid = true;
    m_next = 0;
  }
  bool ibuffer_empty() const {
    for (unsigned i = 0; i < IBUFFER_SIZE; i++)
      if (m_ibuffer[i].m_valid) return false;
    return true;
  }
  void ibuffer_flush() {
    for (unsigned i = 0; i < IBUFFER_SIZE; i++) {
      if (m_ibuffer[i].m_valid) dec_inst_in_pipeline();
      m_ibuffer[i].m_inst = NULL;
      m_ibuffer[i].m_valid = false;
    }
  }
  const warp_inst_t *ibuffer_next_inst() { return m_ibuffer[m_next].m_inst; }
  bool ibuffer_next_valid() { return m_ibuffer[m_next].m_valid; }
  void ibuffer_free() {
    m_ibuffer[m_next].m_inst = NULL;
    m_ibuffer[m_next].m_valid = false;
  }
  void ibuffer_step() { m_next = (m_next + 1) % IBUFFER_SIZE; }

  bool imiss_pending() const { return m_imiss_pending; }
  void set_imiss_pending() { m_imiss_pending = true; }
  void clear_imiss_pending() { m_imiss_pending = false; }

  bool stores_done() const { return m_stores_outstanding == 0; }
  void inc_store_req() { m_stores_outstanding++; }
  void dec_store_req() {
    assert(m_stores_outstanding > 0);
    m_stores_outstanding--;
  }

  unsigned num_inst_in_buffer() const {
    unsigned count = 0;
    for (unsigned i = 0; i < IBUFFER_SIZE; i++) {
      if (m_ibuffer[i].m_valid) count++;
    }
    return count;
  }
  unsigned num_inst_in_pipeline() const { return m_inst_in_pipeline; }
  unsigned num_issued_inst_in_pipeline() const {
    return (num_inst_in_pipeline() - num_inst_in_buffer());
  }
  bool inst_in_pipeline() const { return m_inst_in_pipeline > 0; }
  void inc_inst_in_pipeline() { m_inst_in_pipeline++; }
  void dec_inst_in_pipeline() {
    assert(m_inst_in_pipeline > 0);
    m_inst_in_pipeline--;
  }

  unsigned get_cta_id() const { return m_cta_id; }

  unsigned get_dynamic_warp_id() const { return m_dynamic_warp_id; }
  unsigned get_warp_id() const { return m_warp_id; }

  // MOD. Begin IBuffer_ooo debug
  std::map<unsigned,unsigned> pc_incs;
  std::map<unsigned,unsigned> pc_decs;

  void add_inc_pc(unsigned pc)
  {
    std::map<unsigned,unsigned>::const_iterator it = pc_incs.find(pc);
    if(it == pc_incs.end())
    {
      pc_incs[pc] = 1;
    }else {
      unsigned current_incs = it->second;
      pc_incs[pc] = current_incs + 1;
    }
  }

  void add_dec_pc(unsigned pc)
  {
    std::map<unsigned,unsigned>::const_iterator it = pc_decs.find(pc);
    if(it == pc_decs.end())
    {
      pc_decs[pc] = 1;
    }else {
      unsigned current_decs = it->second;
      pc_decs[pc] = current_decs + 1;
    }
  }
  void print_inc_decs() 
  {
    std::cout << "Size pc_incs: " << pc_incs.size() << ", size pc_decs: " << pc_decs.size() << ". PC comp:" <<std::endl;
    std::map<unsigned,unsigned>::const_iterator it, it2;
    unsigned aux_pc, pc_incs_val, pc_decs_val;
    for(it = pc_incs.begin(); it != pc_incs.end(); it++)
    {
      aux_pc = it->first;
      pc_incs_val = it -> second;
      it2 = pc_decs.find(aux_pc);
      if(it2 == pc_decs.end())
      {
        std::cout << "ERROR, pc not found in decs: " << std::hex << aux_pc << std::dec << std::endl;
      }else {
        pc_decs_val = it2->second;
        std::cout << "Inc_Dec. PC: " << std::hex << aux_pc << std::dec << ", incs: " << pc_incs_val << ", decs: " << pc_decs_val << std::endl;
      }
    }
  }
  // MOD. end IBuffer_ooo debug

  class shader_core_ctx_wrapper *get_shader() {
    return m_shader;
  }

 private:
  static const unsigned IBUFFER_SIZE = 2;
  class shader_core_ctx_wrapper *m_shader;
  unsigned m_cta_id;
  unsigned m_warp_id;
  unsigned m_warp_size;
  unsigned m_dynamic_warp_id;

  address_type m_next_pc;
  unsigned n_completed;  // number of threads in warp completed
  std::bitset<MAX_WARP_SIZE> m_active_threads;

  bool m_imiss_pending;

  struct ibuffer_entry {
    ibuffer_entry() {
      m_valid = false;
      m_inst = NULL;
    }
    const warp_inst_t *m_inst;
    bool m_valid;
  };

  warp_inst_t m_inst_at_barrier;
  ibuffer_entry m_ibuffer[IBUFFER_SIZE];
  unsigned m_next;

  unsigned m_n_atomic;  // number of outstanding atomic operations
  bool m_membar;        // if true, warp is waiting at memory barrier
  bool m_gridbar;      // if true, warp is waiting at grid barrier

  bool m_done_exit;  // true once thread exit has been registered for threads in
                     // this warp

  unsigned long long m_last_fetch;

  unsigned m_stores_outstanding;  // number of store requests sent but not yet
                                  // acknowledged
  unsigned m_inst_in_pipeline;

  scheduler_unit *m_scheduler; // MOD. Added L0I
  int m_is_pending_store; // MOD. Fix loads after store
  int m_is_pending_load; // MOD. Fix loads after store
  IBuffer_Remodeled *m_IBuffer_remodeled; // MOD. Remodeling
  Dependency_State *m_dependency_state; // MOD. Remodeling
  // MOD. End. VPREG
  // Jin: cdp support
 public:
  unsigned int m_cdp_latency;
  bool m_cdp_dummy;
  std::stack<function_call_entry_info> m_function_call_stack;
  unsigned long long m_last_unique_inst_id;
  unsigned int m_kernel_id;
  Subcore *m_subcore;
};

inline unsigned hw_tid_from_wid(unsigned wid, unsigned warp_size, unsigned i) {
  return wid * warp_size + i;
};
inline unsigned wid_from_hw_tid(unsigned tid, unsigned warp_size) {
  return tid / warp_size;
};


int register_bank(int regnum, int wid, unsigned num_banks,
                  unsigned bank_warp_shift, bool sub_core_model,
                  int banks_per_sched, unsigned sched_id);

class shader_core_ctx;
class shader_core_config;
class shader_core_stats;

enum scheduler_prioritization_type {
  SCHEDULER_PRIORITIZATION_LRR = 0,   // Loose Round Robin
  SCHEDULER_PRIORITIZATION_SRR,       // Strict Round Robin
  SCHEDULER_PRIORITIZATION_GTO,       // Greedy Then Oldest
  SCHEDULER_PRIORITIZATION_GTLRR,     // Greedy Then Loose Round Robin
  SCHEDULER_PRIORITIZATION_GTY,       // Greedy Then Youngest
  SCHEDULER_PRIORITIZATION_OLDEST,    // Oldest First
  SCHEDULER_PRIORITIZATION_YOUNGEST,  // Youngest First
};

// Each of these corresponds to a string value in the gpgpsim.config file
// For example - to specify the LRR scheudler the config must contain lrr
enum concrete_scheduler {
  CONCRETE_SCHEDULER_LRR = 0,
  CONCRETE_SCHEDULER_GTO,
  CONCRETE_SCHEDULER_TWO_LEVEL_ACTIVE,
  CONCRETE_SCHEDULER_RRR,
  CONCRETE_SCHEDULER_WARP_LIMITING,
  CONCRETE_SCHEDULER_OLDEST_FIRST,
  NUM_CONCRETE_SCHEDULERS
};

class scheduler_unit {  // this can be copied freely, so can be used in std
                        // containers.
 public:
  scheduler_unit(shader_core_stats *stats, shader_core_ctx *shader,
                 Scoreboard *scoreboard, Scoreboard_reads *scoreboard_reads, // MOD. Fix WAR at baseline.
                 simt_stack **simt,
                 std::vector<shd_warp_t *> *warp, register_set *sp_out,
                 register_set *dp_out, register_set *sfu_out,
                 register_set *int_out, register_set *tensor_core_out,
                 std::vector<register_set *> &spec_cores_out,
                 register_set *mem_out, int id, const concrete_scheduler scheduler)
      : m_stats(stats),
        m_shader(shader),
        m_scoreboard(scoreboard),
        m_scoreboard_reads(scoreboard_reads), // MOD. Fix WAR at baseline.
        m_simt_stack(simt),
        m_warp(warp),
        m_sp_out(sp_out),
        m_dp_out(dp_out),
        m_sfu_out(sfu_out),
        m_int_out(int_out),
        m_tensor_core_out(tensor_core_out),
        m_mem_out(mem_out),
        m_spec_cores_out(spec_cores_out),
        m_id(id) {}

  virtual ~scheduler_unit() {
  }
  virtual void add_supervised_warp_id(int i) {
    m_supervised_warps.push_back(&warp(i));
    warp(i).set_scheduler(this); // MOD. Added L0I
  }

  virtual void done_adding_supervised_warps() {
    m_last_supervised_issued = m_supervised_warps.end();
  }

  // The core scheduler cycle method is meant to be common between
  // all the derived schedulers.  The scheduler's behaviour can be
  // modified by changing the contents of the m_next_cycle_prioritized_warps
  // list.
  void cycle();

  shader_core_ctx* get_shader(){return m_shader;}; // MOD. IBuffer_ooo

  // These are some common ordering fucntions that the
  // higher order schedulers can take advantage of
  template <typename T>
  void order_lrr(
      typename std::vector<T> &result_list,
      const typename std::vector<T> &input_list,
      const typename std::vector<T>::const_iterator &last_issued_from_input,
      unsigned num_warps_to_add);
  template <typename T>
  void order_rrr(
      typename std::vector<T> &result_list,
      const typename std::vector<T> &input_list,
      const typename std::vector<T>::const_iterator &last_issued_from_input,
      unsigned num_warps_to_add);

  enum OrderingType {
    // The item that issued last is prioritized first then the sorted result
    // of the priority_function
    ORDERING_GREEDY_THEN_PRIORITY_FUNC = 0,
    // No greedy scheduling based on last to issue. Only the priority function
    // determines priority
    ORDERED_PRIORITY_FUNC_ONLY,
    NUM_ORDERING,
  };
  template <typename U>
  void order_by_priority(
      std::vector<U> &result_list, const typename std::vector<U> &input_list,
      const typename std::vector<U>::const_iterator &last_issued_from_input,
      unsigned num_warps_to_add, OrderingType age_ordering,
      bool (*priority_func)(U lhs, U rhs));
  static bool sort_warps_by_oldest_dynamic_id(shd_warp_t *lhs, shd_warp_t *rhs);

  // Derived classes can override this function to populate
  // m_supervised_warps with their scheduling policies
  virtual void order_warps() = 0;

  int get_schd_id() const { return m_id; }
  // MOD. IBuffer_ooo. Begin. Ease access to them
  register_set *get_sp_out() { return m_sp_out; }
  register_set *get_dp_out() { return m_dp_out; }
  register_set *get_sfu_out() { return m_sfu_out; }
  register_set *get_int_out() { return m_int_out; }
  register_set *get_tensor_core_out() { return m_tensor_core_out; }
  register_set *get_mem_out() { return m_mem_out; }
  std::vector<register_set *> &get_spec_cores_out() { return m_spec_cores_out; }
  // MOD. IBuffer_ooo. End. Ease access to them

 protected:
  virtual void do_on_warp_issued(
      unsigned warp_id, unsigned num_issued,
      const std::vector<shd_warp_t *>::const_iterator &prioritized_iter);
  inline unsigned int get_sid() const;

 protected:
  shd_warp_t &warp(int i);

  // This is the prioritized warp list that is looped over each cycle to
  // determine which warp gets to issue.
  std::vector<shd_warp_t *> m_next_cycle_prioritized_warps;
  // The m_supervised_warps list is all the warps this scheduler is supposed to
  // arbitrate between.  This is useful in systems where there is more than
  // one warp scheduler. In a single scheduler system, this is simply all
  // the warps assigned to this core.
  std::vector<shd_warp_t *> m_supervised_warps;
  // This is the iterator pointer to the last supervised warp you issued
  std::vector<shd_warp_t *>::const_iterator m_last_supervised_issued;
  shader_core_stats *m_stats;
  shader_core_ctx *m_shader;
  // these things should become accessors: but would need a bigger rearchitect
  // of how shader_core_ctx interacts with its parts.
  Scoreboard *m_scoreboard;
  Scoreboard_reads *m_scoreboard_reads; // MOD. Fix WAR at baseline.
  simt_stack **m_simt_stack;
  // warp_inst_t** m_pipeline_reg;
  std::vector<shd_warp_t *> *m_warp;

  // MOD. IBuffer_ooo. Begin. Ease access to them
  register_set *m_sp_out;
  register_set *m_dp_out;
  register_set *m_sfu_out;
  register_set *m_int_out;
  register_set *m_tensor_core_out;
  register_set *m_mem_out;
  std::vector<register_set *> &m_spec_cores_out;
  // MOD. IBuffer_ooo. End. Ease access to them

  unsigned m_num_issued_last_cycle;
  unsigned m_current_turn_warp;

  int m_id;

};

class lrr_scheduler : public scheduler_unit {
 public:
  lrr_scheduler(shader_core_stats *stats, shader_core_ctx *shader,
                Scoreboard *scoreboard, Scoreboard_reads *scoreboard_reads, // MOD. Fix WAR at baseline.
                simt_stack **simt,
                std::vector<shd_warp_t *> *warp, register_set *sp_out,
                register_set *dp_out, register_set *sfu_out,
                register_set *int_out, register_set *tensor_core_out,
                std::vector<register_set *> &spec_cores_out,
                register_set *mem_out, int id, const concrete_scheduler scheduler) 
      : scheduler_unit(stats, shader, scoreboard, scoreboard_reads, simt, warp, sp_out, dp_out, // MOD. Fix WAR at baseline.
                       sfu_out, int_out, tensor_core_out, spec_cores_out,
                       mem_out, id, scheduler) {} // MOD. VPREG
  virtual ~lrr_scheduler() {}
  virtual void order_warps();
  virtual void done_adding_supervised_warps() {
    m_last_supervised_issued = m_supervised_warps.end();
  }
};

class rrr_scheduler : public scheduler_unit {
 public:
  rrr_scheduler(shader_core_stats *stats, shader_core_ctx *shader,
                Scoreboard *scoreboard, Scoreboard_reads *scoreboard_reads, // MOD. Fix WAR at baseline.
                simt_stack **simt,
                std::vector<shd_warp_t *> *warp, register_set *sp_out,
                register_set *dp_out, register_set *sfu_out,
                register_set *int_out, register_set *tensor_core_out,
                std::vector<register_set *> &spec_cores_out,
                register_set *mem_out, int id, const concrete_scheduler scheduler) 
      : scheduler_unit(stats, shader, scoreboard, scoreboard_reads, simt, warp, sp_out, dp_out,
                       sfu_out, int_out, tensor_core_out, spec_cores_out,
                       mem_out, id, scheduler) {}
  virtual ~rrr_scheduler() {}
  virtual void order_warps();
  virtual void done_adding_supervised_warps() {
    m_last_supervised_issued = m_supervised_warps.end();
  }
};

class gto_scheduler : public scheduler_unit {
 public:
  gto_scheduler(shader_core_stats *stats, shader_core_ctx *shader,
                Scoreboard *scoreboard, Scoreboard_reads *scoreboard_reads, // MOD. Fix WAR at baseline.
                simt_stack **simt,
                std::vector<shd_warp_t *> *warp, register_set *sp_out,
                register_set *dp_out, register_set *sfu_out,
                register_set *int_out, register_set *tensor_core_out,
                std::vector<register_set *> &spec_cores_out,
                register_set *mem_out, int id, const concrete_scheduler scheduler) 
      : scheduler_unit(stats, shader, scoreboard, scoreboard_reads, simt, warp, sp_out, dp_out, // MOD. Fix WAR at baseline.
                       sfu_out, int_out, tensor_core_out, spec_cores_out,
                       mem_out, id, scheduler) {}
  virtual ~gto_scheduler() {}
  virtual void order_warps();
  virtual void done_adding_supervised_warps() {
    m_last_supervised_issued = m_supervised_warps.begin();
  }
};

class oldest_scheduler : public scheduler_unit {
 public:
  oldest_scheduler(shader_core_stats *stats, shader_core_ctx *shader,
                   Scoreboard *scoreboard, Scoreboard_reads *scoreboard_reads, // MOD. Fix WAR at baseline.
                   simt_stack **simt,
                   std::vector<shd_warp_t *> *warp, register_set *sp_out,
                   register_set *dp_out, register_set *sfu_out,
                   register_set *int_out, register_set *tensor_core_out,
                   std::vector<register_set *> &spec_cores_out,
                   register_set *mem_out, int id, const concrete_scheduler scheduler)
      : scheduler_unit(stats, shader, scoreboard, scoreboard_reads, simt, warp, sp_out, dp_out, // MOD. Fix WAR at baseline.
                       sfu_out, int_out, tensor_core_out, spec_cores_out,
                       mem_out, id, scheduler) {} 
  virtual ~oldest_scheduler() {}
  virtual void order_warps();
  virtual void done_adding_supervised_warps() {
    m_last_supervised_issued = m_supervised_warps.begin();
  }
};

class two_level_active_scheduler : public scheduler_unit {
 public:
  two_level_active_scheduler(shader_core_stats *stats, shader_core_ctx *shader,
                             Scoreboard *scoreboard, Scoreboard_reads *scoreboard_reads, // MOD. Fix WAR at baseline.
                             simt_stack **simt,
                             std::vector<shd_warp_t *> *warp,
                             register_set *sp_out, register_set *dp_out,
                             register_set *sfu_out, register_set *int_out,
                             register_set *tensor_core_out,
                             std::vector<register_set *> &spec_cores_out,
                             register_set *mem_out, int id, char *config_str, const concrete_scheduler scheduler)
      : scheduler_unit(stats, shader, scoreboard, scoreboard_reads, simt, warp, sp_out, dp_out, // MOD. Fix WAR at baseline.
                       sfu_out, int_out, tensor_core_out, spec_cores_out,
                       mem_out, id, scheduler), 
        m_pending_warps() {
    unsigned inner_level_readin;
    unsigned outer_level_readin;
    int ret =
        sscanf(config_str, "two_level_active:%d:%d:%d", &m_max_active_warps,
               &inner_level_readin, &outer_level_readin);
    assert(3 == ret);
    m_inner_level_prioritization =
        (scheduler_prioritization_type)inner_level_readin;
    m_outer_level_prioritization =
        (scheduler_prioritization_type)outer_level_readin;
    
  }
  virtual ~two_level_active_scheduler() {
  }
  virtual void order_warps();
  void add_supervised_warp_id(int i) {
    if (m_next_cycle_prioritized_warps.size() < m_max_active_warps) {
      m_next_cycle_prioritized_warps.push_back(&warp(i));
    } else {
      m_pending_warps.push_back(&warp(i));
    }
  }
  virtual void done_adding_supervised_warps() {
    m_last_supervised_issued = m_supervised_warps.begin();
  }

 protected:
  virtual void do_on_warp_issued(
      unsigned warp_id, unsigned num_issued,
      const std::vector<shd_warp_t *>::const_iterator &prioritized_iter);

 private:
  std::deque<shd_warp_t *> m_pending_warps;
  scheduler_prioritization_type m_inner_level_prioritization;
  scheduler_prioritization_type m_outer_level_prioritization;
  unsigned m_max_active_warps;
};

// Static Warp Limiting Scheduler
class swl_scheduler : public scheduler_unit {
 public:
  swl_scheduler(shader_core_stats *stats, shader_core_ctx *shader,
                Scoreboard *scoreboard, Scoreboard_reads *scoreboard_reads, // MOD. Fix WAR at baseline.
                simt_stack **simt,
                std::vector<shd_warp_t *> *warp, register_set *sp_out,
                register_set *dp_out, register_set *sfu_out,
                register_set *int_out, register_set *tensor_core_out,
                std::vector<register_set *> &spec_cores_out,
                register_set *mem_out, int id, char *config_string, const concrete_scheduler scheduler);
  virtual ~swl_scheduler() {}
  virtual void order_warps();
  virtual void done_adding_supervised_warps() {
    m_last_supervised_issued = m_supervised_warps.begin();
  }

 protected:
  scheduler_prioritization_type m_prioritization;
  unsigned m_num_warps_to_limit;
};

class opndcoll_rfu_t {  // operand collector based register file unit
 public:
  // constructors
  opndcoll_rfu_t() {
    m_num_banks = 0;
    m_shader = NULL;
    m_initialized = false;
  }
  void add_cu_set(unsigned cu_set, unsigned num_cu, unsigned num_dispatch);
  typedef std::vector<register_set *> port_vector_t;
  typedef std::vector<unsigned int> uint_vector_t;
  void add_port(port_vector_t &input, port_vector_t &ouput,
                uint_vector_t cu_sets);
  void init(unsigned num_banks, shader_core_ctx *shader); 

  // modifiers
  bool writeback(warp_inst_t &warp);

  void step() {
    dispatch_ready_cu();
    allocate_reads();
    for (unsigned p = 0; p < m_in_ports.size(); p++) {
      allocate_cu(p);
    }
    process_banks();
  }

  void dump(FILE *fp) const {
    fprintf(fp, "\n");
    fprintf(fp, "Operand Collector State:\n");
    for (unsigned n = 0; n < m_cu.size(); n++) {
      fprintf(fp, "   CU-%2u: ", n);
      m_cu[n]->dump(fp, m_shader);
    }
    m_arbiter.dump(fp);
  }

  shader_core_ctx *shader_core() { return m_shader; }

  // MOD. Begin. OPC custom stats
  void reset_structures_opc_custom_stats();
  void calculate_opc_custom_stats();
  // MOD. End. OPC custom stats
  
  // MOD. Begin. Improved Result Bus
  unsigned int get_bank_warp_shift() { return m_bank_warp_shift; }
  bool get_is_sub_core_model() { return sub_core_model; }
  unsigned int get_banks_per_sched() { return m_num_banks_per_sched; }
  // MOD. End. Improved Result bus
  
 private:
  void process_banks() { m_arbiter.reset_alloction(); }

  void dispatch_ready_cu();
  void allocate_cu(unsigned port);
  void allocate_reads();

  // types

  class collector_unit_t;

  // MOD. Begin. VPREG
  bool m_is_vpreg_enabled;
  bool m_is_vpreg_balanced_banks_mode_enabled;
  int m_regs_per_bank;
  int m_banks_per_subcore;
  // MOD. End. VPREG

  // MOD. Begin. OPC custom stats
  std::vector<bool> m_has_subcore_allocated_cu;
  std::vector<bool> m_has_subcore_something_to_allocate_in_cu;
  std::vector<int> m_num_dispatched_cus_this_cycle;
  int m_num_cu_units_per_subcore;
  // MOD. End. OPC custom stats

  class op_t {
   public:
    op_t() { m_valid = false;} 
    op_t(collector_unit_t *cu, unsigned op, unsigned reg, unsigned num_banks,
         unsigned bank_warp_shift, bool sub_core_model,
         unsigned banks_per_sched, unsigned sched_id,
         bool is_renamed, rrs_id_type renamed_rrs_id) { // MOD. LOOG
      m_valid = true;
      m_warp = NULL;
      m_cu = cu;
      m_operand = op;
      m_register = reg;
      m_shced_id = sched_id;
      m_bank = register_bank(reg, cu->get_warp_id(), num_banks, bank_warp_shift,
                             sub_core_model, banks_per_sched, sched_id); 
    }
    op_t(const warp_inst_t *warp, unsigned reg, unsigned num_banks,
         unsigned bank_warp_shift, bool sub_core_model,
         unsigned banks_per_sched, unsigned sched_id,
         opndcoll_rfu_t* rfu) {
      m_valid = true;
      m_warp = warp;
      m_register = reg;
      m_cu = NULL;
      m_operand = -1;
      m_shced_id = sched_id;
      m_bank = register_bank(reg, warp->warp_id(), num_banks, bank_warp_shift,
                             sub_core_model, banks_per_sched, sched_id );
    }

    // accessors
    bool valid() const { return m_valid; }
    unsigned int get_reg() const {
      assert(m_valid);
      return m_register;
    }
    unsigned int get_wid() const {
      if (m_warp)
        return m_warp->warp_id();
      else if (m_cu)
        return m_cu->get_warp_id();
      else
        abort();
    }
    unsigned int get_sid() const { return m_shced_id; }
    unsigned int get_active_count() const {
      if (m_warp)
        return m_warp->active_count();
      else if (m_cu)
        return m_cu->get_active_count();
      else
        abort();
    }
    const active_mask_t &get_active_mask() {
      if (m_warp)
        return m_warp->get_active_mask();
      else if (m_cu)
        return m_cu->get_active_mask();
      else
        abort();
    }
    unsigned int get_sp_op() const {
      if (m_warp)
        return m_warp->sp_op;
      else if (m_cu)
        return m_cu->get_sp_op();
      else
        abort();
    }
    unsigned get_oc_id() const { return m_cu->get_id(); }
    unsigned get_bank() const { return m_bank; }
    unsigned get_operand() const { return m_operand; }
    void dump(FILE *fp) const {
      if (m_cu)
        fprintf(fp, " <R%u, CU:%u, w:%02u> ", m_register, m_cu->get_id(),
                m_cu->get_warp_id());
      else if (!m_warp->empty())
        fprintf(fp, " <R%u, wid:%02u> ", m_register, m_warp->warp_id());
    }
    std::string get_reg_string() const {
      char buffer[64];
      snprintf(buffer, 64, "R%u", m_register);
      return std::string(buffer);
    }

    // modifiers
    void reset() { m_valid = false; }

   private:
    bool m_valid;
    collector_unit_t *m_cu;
    const warp_inst_t *m_warp;
    unsigned m_operand;  // operand offset in instruction. e.g., add r1,r2,r3;
                         // r2 is oprd 0, r3 is 1 (r1 is dst)
    unsigned m_register;
    unsigned m_bank;
    unsigned m_shced_id;  // scheduler id that has issued this inst

  };

  enum alloc_t {
    NO_ALLOC,
    READ_ALLOC,
    WRITE_ALLOC,
  };

  class allocation_t {
   public:
    allocation_t() { m_allocation = NO_ALLOC; }
    bool is_read() const { return m_allocation == READ_ALLOC; }
    bool is_write() const { return m_allocation == WRITE_ALLOC; }
    bool is_free() const { return m_allocation == NO_ALLOC; }
    void dump(FILE *fp) const {
      if (m_allocation == NO_ALLOC) {
        fprintf(fp, "<free>");
      } else if (m_allocation == READ_ALLOC) {
        fprintf(fp, "rd: ");
        m_op.dump(fp);
      } else if (m_allocation == WRITE_ALLOC) {
        fprintf(fp, "wr: ");
        m_op.dump(fp);
      }
      fprintf(fp, "\n");
    }
    void alloc_read(const op_t &op) {
      assert(is_free());
      m_allocation = READ_ALLOC;
      m_op = op;
    }
    void alloc_write(const op_t &op) {
      assert(is_free());
      m_allocation = WRITE_ALLOC;
      m_op = op;
    }
    void reset() { m_allocation = NO_ALLOC; }

   private:
    enum alloc_t m_allocation;
    op_t m_op;
  };

  class arbiter_t {
   public:
    // constructors
    arbiter_t() {
      m_queue = NULL;
      m_allocated_bank = NULL;
      m_allocator_rr_head = NULL;
      _inmatch = NULL;
      _outmatch = NULL;
      _request = NULL;
      m_last_cu = 0;
    }
    void init(unsigned int num_cu, unsigned int num_banks, bool is_opc_improved, unsigned int ports_per_bank, unsigned ports_per_cu) { // MOD. Improved OPC
      assert(num_cu > 0);
      assert(num_banks > 0);
      m_num_collectors = num_cu;
      m_num_banks = num_banks;
      _inmatch = new int[m_num_banks];
      _outmatch = new int[m_num_collectors];
      _request = new int *[m_num_banks];
      // MOD. Begin. Improved OPC
      m_allocated_bank = new allocation_t *[num_banks];
      for (unsigned i = 0; i < m_num_banks; i++) {
        _request[i] = new int[m_num_collectors];
        if(is_opc_improved) {
          m_allocated_bank[i] = new allocation_t[ports_per_bank];
        } else  {
          m_allocated_bank[i] = new allocation_t[1];
        }
      }
      // MOD. End. Improved OPC
      m_queue = new std::deque<op_t>[num_banks]; // MOD. Improved OPC
      
      m_allocator_rr_head = new unsigned[num_cu];
      for (unsigned n = 0; n < num_cu; n++)
        m_allocator_rr_head[n] = n % num_banks;
      reset_alloction();
      // MOD. Begin. Improved OPC
      m_is_improved_opc = is_opc_improved;
      m_ports_per_bank = ports_per_bank;
      m_ports_per_cu = ports_per_cu;
      // MOD. End. Improved OPC
    }

    // accessors
    void dump(FILE *fp) const {
      fprintf(fp, "\n");
      fprintf(fp, "  Arbiter State:\n");
      fprintf(fp, "  requests:\n");
      for (unsigned b = 0; b < m_num_banks; b++) {
        fprintf(fp, "    bank %u : ", b);
        std::deque<op_t>::const_iterator o = m_queue[b].begin(); // MOD. Improved OPC
        for (; o != m_queue[b].end(); o++) {
          o->dump(fp);
        }
        fprintf(fp, "\n");
      }
      fprintf(fp, "  grants:\n");
      for (unsigned int b = 0; b < m_num_banks; b++) {
        // MOD. Begin. Improved OPC
        for(unsigned int p = 0; p < m_ports_per_bank; p++) {
          fprintf(fp, "    bank %u, port %d : ", b, p);
          m_allocated_bank[b][p].dump(fp);
        }
        // MOD. End. Improved OPC
      }
      fprintf(fp, "\n");
    }

    // modifiers
    std::list<op_t> allocate_reads();

    void add_read_requests(collector_unit_t *cu) {
      const op_t *src = cu->get_operands();
      for (unsigned i = 0; i < MAX_REG_OPERANDS * 2; i++) {
        const op_t &op = src[i];
        bool read_from_rf =  true; 
        if (op.valid() && read_from_rf) { 
          unsigned bank = op.get_bank();
          m_queue[bank].push_back(op);
        }
      }
    }

    // MOD. Begin. Improved OPC
    int get_num_free_ports(unsigned bank) const {
      int res = 0;
      for(unsigned int p = 0; p < m_ports_per_bank ; p++) {
        if(m_allocated_bank[bank][p].is_free()) {
          res++;
        }
      }
      return res;
    }

    unsigned int get_num_used_ports(unsigned bank) const {
      unsigned int res = 0;
      for(unsigned int p = 0; p < m_ports_per_bank ; p++) {
        if(!m_allocated_bank[bank][p].is_free()) {
          res++;
        }
      }
      return res;
    }

    bool bank_idle(unsigned bank) const {
      bool is_free = false;
      for(unsigned int p = 0; (p < m_ports_per_bank) && !is_free ; p++) {
        is_free = m_allocated_bank[bank][p].is_free();
      }
      return is_free;
    }
    void allocate_bank_for_write(unsigned bank, const op_t &op) {
      assert(bank < m_num_banks);
      bool allocated = false;
      for(unsigned int p = 0; (p < m_ports_per_bank) && !allocated ; p++) {
        if(m_allocated_bank[bank][p].is_free()) {
          m_allocated_bank[bank][p].alloc_write(op);
          allocated = true;
        }
      }
      assert(allocated);
    }
    void allocate_for_read(unsigned bank, const op_t &op) {
      assert(bank < m_num_banks);
      bool allocated = false;
      for(unsigned int p = 0; (p < m_ports_per_bank) && !allocated ; p++) {
        if(m_allocated_bank[bank][p].is_free()) {
          m_allocated_bank[bank][p].alloc_read(op);
          allocated = true;
        }
      }
      assert(allocated);
    }
    void reset_alloction() {
      for (unsigned b = 0; b < m_num_banks; b++) {
        for(unsigned int p = 0; p < m_ports_per_bank; p++) {
          m_allocated_bank[b][p].reset();
        }
      } 
    }
    // MOD. End. Improved OPC

    int get_conflicts() {return m_conflicts;} // MOD. VPREG
    int get_total_requests() {return m_total_req;} // MOD. VPREG

   private:
    unsigned m_num_banks;
    unsigned m_num_collectors;

    allocation_t **m_allocated_bank;  // bank # -> register that wins. MOD. Improved OPC
    std::deque<op_t> *m_queue; // MOD. Improved OPC

    unsigned *
        m_allocator_rr_head;  // cu # -> next bank to check for request (rr-arb)
    unsigned m_last_cu;       // first cu to check while arb-ing banks (rr)

    int *_inmatch;
    int *_outmatch;
    int **_request;

    int m_conflicts; // MOD. VPREG
    int m_total_req; // MOD. VPREG
    bool m_is_improved_opc; // MOD. Improved OPC
    unsigned int m_ports_per_bank; // MOD. Improved OPC
    unsigned int m_ports_per_cu; // MOD. Improved OPC
  };

  class input_port_t {
   public:
    input_port_t(port_vector_t &input, port_vector_t &output,
                 uint_vector_t cu_sets)
        : m_in(input), m_out(output), m_cu_sets(cu_sets) {
      assert(input.size() == output.size());
      assert(not m_cu_sets.empty());
    }
    // private:
    port_vector_t m_in, m_out;
    uint_vector_t m_cu_sets;
  };

  class collector_unit_t {
   public:
    // constructors
    collector_unit_t() {
      m_free = true;
      m_warp = NULL;
      m_output_register = NULL;
      m_src_op = new op_t[MAX_REG_OPERANDS * 2];
      m_not_ready.reset();
      m_warp_id = -1;
      m_num_banks = 0;
      m_bank_warp_shift = 0;
      m_allocation_cycle = 0; // MOD. CU stats
    }
    // accessors
    bool ready() const;
    const op_t *get_operands() const { return m_src_op; }
    void dump(FILE *fp, const shader_core_ctx *shader) const;

    unsigned get_warp_id() const { return m_warp_id; }
    unsigned get_active_count() const { return m_warp->active_count(); }
    const active_mask_t &get_active_mask() const {
      return m_warp->get_active_mask();
    }
    unsigned get_sp_op() const { return m_warp->sp_op; }
    unsigned get_id() const { return m_cuid; }  // returns CU hw id
    unsigned get_reg_id() const { return m_reg_id; }

    // modifiers
    void init(unsigned n, unsigned num_banks, unsigned log2_warp_size,
              const core_config *config, opndcoll_rfu_t *rfu,
              bool m_sub_core_model, unsigned reg_id,
              unsigned num_banks_per_sched);
    bool allocate(register_set *pipeline_reg, register_set *output_reg);

    bool are_all_operands_ready() { return m_not_ready.none(); } // MOD. OPC custom stats
    bool is_dispatch_register_free() { return (*m_output_register).has_free(m_sub_core_model, m_reg_id); } // MOD. OPC custom stats
    void collect_operand(unsigned op) { m_not_ready.reset(op); }
    unsigned get_num_operands() const { return m_warp->get_num_operands(); }
    unsigned get_num_regs() const { return m_warp->get_num_regs(); }
    void dispatch();
    bool is_free() { return m_free; }

    opndcoll_rfu_t * get_rfu() { return m_rfu;} // MOD. LOOG
    warp_inst_t* get_warp_inst() { return m_warp;} // MOD. LOOG

   private:
    unsigned long long m_allocation_cycle; // MOD. CU stats
    bool m_free;
    unsigned m_cuid;  // collector unit hw id
    unsigned m_warp_id;
    warp_inst_t *m_warp;
    register_set
        *m_output_register;  // pipeline register to issue to when ready
    op_t *m_src_op;
    std::bitset<MAX_REG_OPERANDS * 2> m_not_ready;
    unsigned m_num_banks;
    unsigned m_bank_warp_shift;
    opndcoll_rfu_t *m_rfu;

    unsigned m_num_banks_per_sched;
    bool m_sub_core_model;
    unsigned m_reg_id;  // if sub_core_model enabled, limit regs this cu can r/w
  };

  class dispatch_unit_t {
   public:
    dispatch_unit_t(std::vector<collector_unit_t> *cus) {
      m_last_cu = 0;
      m_collector_units = cus;
      m_num_collectors = (*cus).size();
      m_next_cu = 0;
    }
    void init(bool sub_core_model, unsigned num_warp_scheds) {
      m_sub_core_model = sub_core_model;
      m_num_warp_scheds = num_warp_scheds;
    }

    collector_unit_t *find_ready() {
      // With sub-core enabled round robin starts with the next cu assigned to a
      // different sub-core than the one that dispatched last
      unsigned cusPerSched = m_num_collectors / m_num_warp_scheds;
      unsigned rr_increment = m_sub_core_model ?
                              cusPerSched - (m_last_cu % cusPerSched) : 1;
      for (unsigned n = 0; n < m_num_collectors; n++) {
        unsigned c = (m_last_cu + n + rr_increment) % m_num_collectors;
        if ((*m_collector_units)[c].ready()) {
          m_last_cu = c;
          return &((*m_collector_units)[c]);
        }
      }
      return NULL;
    }

   private:
    unsigned m_num_collectors;
    std::vector<collector_unit_t> *m_collector_units;
    unsigned m_last_cu;  // dispatch ready cu's rr
    unsigned m_next_cu;  // for initialization
    bool m_sub_core_model;
    unsigned m_num_warp_scheds;
  };

  // opndcoll_rfu_t data members
  bool m_initialized;

  unsigned m_num_collector_sets;
  // unsigned m_num_collectors;
  unsigned m_num_banks;
  unsigned m_bank_warp_shift;
  unsigned m_warp_size;
  std::vector<collector_unit_t *> m_cu;
  arbiter_t m_arbiter;

  unsigned m_num_banks_per_sched;
  unsigned m_num_warp_scheds;
  bool sub_core_model;

  // unsigned m_num_ports;
  // std::vector<warp_inst_t**> m_input;
  // std::vector<warp_inst_t**> m_output;
  // std::vector<unsigned> m_num_collector_units;
  // warp_inst_t **m_alu_port;

  std::vector<input_port_t> m_in_ports;
  typedef std::map<unsigned /* collector set */,
                   std::vector<collector_unit_t> /*collector sets*/>
      cu_sets_t;
  cu_sets_t m_cus;
  std::vector<dispatch_unit_t> m_dispatch_units;

  // typedef std::map<warp_inst_t**/*port*/,dispatch_unit_t> port_to_du_t;
  // port_to_du_t                     m_dispatch_units;
  // std::map<warp_inst_t**,std::list<collector_unit_t*> > m_free_cu;
  shader_core_ctx *m_shader;
};

class barrier_set_t {
 public:
  barrier_set_t(shader_core_ctx_wrapper *shader, unsigned max_warps_per_core,
                unsigned max_cta_per_core, unsigned max_barriers_per_cta,
                unsigned warp_size);

  // during cta allocation
  void allocate_barrier(unsigned cta_id, warp_set_t warps);

  // during cta deallocation
  void deallocate_barrier(unsigned cta_id);

  typedef std::map<unsigned, warp_set_t> cta_to_warp_t;
  typedef std::map<unsigned, warp_set_t>
      bar_id_to_warp_t; /*set of warps reached a specific barrier id*/

  // individual warp hits barrier
  void warp_reaches_barrier(unsigned cta_id, unsigned warp_id,
                            warp_inst_t *inst);

  // warp reaches exit
  void warp_exit(unsigned warp_id);

  // assertions
  bool warp_waiting_at_barrier(unsigned warp_id) const;

  // debug
  void dump();

 private:
  unsigned m_max_cta_per_core;
  unsigned m_max_warps_per_core;
  unsigned m_max_barriers_per_cta;
  unsigned m_warp_size;
  cta_to_warp_t m_cta_to_warps;
  bar_id_to_warp_t m_bar_id_to_warps;
  warp_set_t m_warp_active;
  warp_set_t m_warp_at_barrier;
  shader_core_ctx_wrapper *m_shader;
};

struct insn_latency_info {
  unsigned pc;
  unsigned long latency;
};

struct ifetch_buffer_t {
  ifetch_buffer_t() { m_valid = false; }

  ifetch_buffer_t(address_type pc, unsigned nbytes, unsigned warp_id) {
    m_valid = true;
    m_pc = pc;
    m_nbytes = nbytes;
    m_warp_id = warp_id;
    only_read_I2 = false; // MOD. VPREG
  }

  bool m_valid;
  address_type m_pc;
  unsigned m_nbytes;
  unsigned m_warp_id;
  bool only_read_I2; // MOD. VPREG
};

class shader_core_config;

class simd_function_unit {
 public:
  simd_function_unit(const shader_core_config *config);
  ~simd_function_unit() { delete m_dispatch_reg; }

  // modifiers
  virtual void issue(register_set &source_reg, unsigned int subcore_id); // MOD. Fixed LDST_Unit model
  virtual void cycle() = 0;
  virtual void active_lanes_in_pipeline() = 0;

  // accessors
  virtual unsigned clock_multiplier() const { return 1; }
  virtual bool can_issue(const warp_inst_t &inst) const {
    return m_dispatch_reg->empty() && !occupied.test(inst.latency);
  }
  virtual bool is_issue_partitioned() = 0;
  virtual unsigned get_issue_reg_id() = 0;
  virtual bool stallable() const = 0;
  virtual void print(FILE *fp) const {
    fprintf(fp, "%s dispatch= ", m_name.c_str());
    m_dispatch_reg->print(fp);
  }
  const char *get_name() { return m_name.c_str(); }

 protected:
  std::string m_name;
  const shader_core_config *m_config;
  warp_inst_t *m_dispatch_reg;
  static const unsigned MAX_ALU_LATENCY = 512;
  std::bitset<MAX_ALU_LATENCY> occupied;
};

class pipelined_simd_unit : public simd_function_unit {
 public:
  pipelined_simd_unit(register_set *result_port,
                      const shader_core_config *config, unsigned max_latency,
                      shader_core_ctx_wrapper *core, unsigned issue_reg_id);

  // modifiers
  virtual void cycle();
  virtual void issue(register_set &source_reg, unsigned int subcore_id); // MOD. Fixed LDST_Unit model
  virtual unsigned get_active_lanes_in_pipeline();

  virtual void active_lanes_in_pipeline() = 0;
  /*
      virtual void issue( register_set& source_reg )
      {
          //move_warp(m_dispatch_reg,source_reg);
          //source_reg.move_out_to(m_dispatch_reg);
          simd_function_unit::issue(source_reg);
      }
  */
  // accessors
  virtual bool stallable() const { return false; }
  virtual bool can_issue(const warp_inst_t &inst) const {
    return simd_function_unit::can_issue(inst);
  }
  virtual bool is_issue_partitioned() = 0;
  unsigned get_issue_reg_id() { return m_issue_reg_id; }
  virtual void print(FILE *fp) const {
    simd_function_unit::print(fp);
    for (int s = m_pipeline_depth - 1; s >= 0; s--) {
      if (!m_pipeline_reg[s]->empty()) {
        fprintf(fp, "      %s[%2d] ", m_name.c_str(), s);
        m_pipeline_reg[s]->print(fp);
      }
    }
  }

 protected:
  unsigned m_pipeline_depth;
  warp_inst_t **m_pipeline_reg;
  register_set *m_result_port;
  class shader_core_ctx_wrapper *m_core;
  unsigned m_issue_reg_id;  // if sub_core_model is enabled we can only issue
                            // from a subset of operand collectors

  unsigned active_insts_in_pipeline;
};

class sfu : public pipelined_simd_unit {
 public:
  sfu(register_set *result_port, const shader_core_config *config,
      shader_core_ctx *core, unsigned issue_reg_id);
  virtual bool can_issue(const warp_inst_t &inst) const {
    switch (inst.op) {
      case SFU_OP:
        break;
      case ALU_SFU_OP:
        break;
      case DP_OP:
        break;  // for compute <= 29 (i..e Fermi and GT200)
      default:
        return false;
    }
    return pipelined_simd_unit::can_issue(inst);
  }
  virtual void active_lanes_in_pipeline();
  virtual void issue(register_set &source_reg, unsigned int subcore_id); // MOD. Fixed LDST_Unit model
  bool is_issue_partitioned() { return true; }
};

class dp_unit : public pipelined_simd_unit {
 public:
  dp_unit(register_set *result_port, const shader_core_config *config,
          shader_core_ctx *core, unsigned issue_reg_id);
  virtual bool can_issue(const warp_inst_t &inst) const {
    switch (inst.op) {
      case DP_OP:
        break;
      default:
        return false;
    }
    return pipelined_simd_unit::can_issue(inst);
  }
  virtual void active_lanes_in_pipeline();
  virtual void issue(register_set &source_reg, unsigned int subcore_id); // MOD. Fixed LDST_Unit model
  bool is_issue_partitioned() { return true; }
};

class tensor_core : public pipelined_simd_unit {
 public:
  tensor_core(register_set *result_port, const shader_core_config *config,
              shader_core_ctx *core, unsigned issue_reg_id);
  virtual bool can_issue(const warp_inst_t &inst) const {
    switch (inst.op) {
      case TENSOR_CORE_OP:
        break;
      default:
        return false;
    }
    return pipelined_simd_unit::can_issue(inst);
  }
  virtual void active_lanes_in_pipeline();
  virtual void issue(register_set &source_reg, unsigned int subcore_id); // MOD. Fixed LDST_Unit model
  bool is_issue_partitioned() { return true; }
};

class int_unit : public pipelined_simd_unit {
 public:
  int_unit(register_set *result_port, const shader_core_config *config,
           shader_core_ctx *core, unsigned issue_reg_id);
  virtual bool can_issue(const warp_inst_t &inst) const {
    switch (inst.op) {
      case SFU_OP:
        return false;
      case LOAD_OP:
        return false;
      case TENSOR_CORE_LOAD_OP:
        return false;
      case STORE_OP:
        return false;
      case TENSOR_CORE_STORE_OP:
        return false;
      case MEMORY_BARRIER_OP:
        return false;
      case SP_OP:
        return false;
      case DP_OP:
        return false;
      default:
        break;
    }
    return pipelined_simd_unit::can_issue(inst);
  }
  virtual void active_lanes_in_pipeline();
  virtual void issue(register_set &source_reg, unsigned int subcore_id); // MOD. Fixed LDST_Unit model
  bool is_issue_partitioned() { return true; }
};

class sp_unit : public pipelined_simd_unit {
 public:
  sp_unit(register_set *result_port, const shader_core_config *config,
          shader_core_ctx *core, unsigned issue_reg_id);
  virtual bool can_issue(const warp_inst_t &inst) const {
    switch (inst.op) {
      case SFU_OP:
        return false;
      case LOAD_OP:
        return false;
      case TENSOR_CORE_LOAD_OP:
        return false;
      case STORE_OP:
        return false;
      case TENSOR_CORE_STORE_OP:
        return false;
      case MEMORY_BARRIER_OP:
        return false;
      case DP_OP:
        return false;
      default:
        break;
    }
    return pipelined_simd_unit::can_issue(inst);
  }
  virtual void active_lanes_in_pipeline();
  virtual void issue(register_set &source_reg, unsigned int subcore_id); // MOD. Fixed LDST_Unit model
  bool is_issue_partitioned() { return true; }
};

class specialized_unit : public pipelined_simd_unit {
 public:
  specialized_unit(register_set *result_port, const shader_core_config *config,
                   shader_core_ctx *core, int supported_op,
                   char *unit_name, unsigned latency, unsigned issue_reg_id);
  virtual bool can_issue(const warp_inst_t &inst) const {
    if (inst.op != m_supported_op) {
      return false;
    }
    return pipelined_simd_unit::can_issue(inst);
  }
  virtual void active_lanes_in_pipeline();
  virtual void issue(register_set &source_reg, unsigned int subcore_id); // MOD. Fixed LDST_Unit model
  bool is_issue_partitioned() { return true; }

 private:
  int m_supported_op;
};

class simt_core_cluster;
class shader_memory_interface;
class shader_core_mem_fetch_allocator;
class cache_t;

class ldst_unit : public pipelined_simd_unit {
 public:
  ldst_unit(mem_fetch_interface *icnt,
            shader_core_mem_fetch_allocator *mf_allocator,
            shader_core_ctx *core, opndcoll_rfu_t *operand_collector,
            Scoreboard *scoreboard, Scoreboard_reads *scoreboard_reads, // MOD. Fix WAR at baseline.
            const shader_core_config *config,
            const memory_config *mem_config, class shader_core_stats *stats,
            unsigned sid, unsigned tpc);

  // modifiers
  virtual void issue(register_set &inst, unsigned int subcore_id); // MOD. Fixed LDST_Unit model
  bool is_issue_partitioned() { return false; }
  virtual void cycle();

  void fill(mem_fetch *mf);
  void flush();
  void invalidate();
  void writeback();

  // accessors
  virtual unsigned clock_multiplier() const;

  virtual bool can_issue(const warp_inst_t &inst) const {
    switch (inst.op) {
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
      default:
        return false;
    }
    return m_dispatch_reg->empty();
  }

  bool is_dispatch_reg_empty() const { return m_dispatch_reg->empty(); }

  virtual void active_lanes_in_pipeline();
  virtual bool stallable() const { return true; }
  bool response_buffer_full() const;
  void print(FILE *fout) const;
  void print_cache_stats(FILE *fp, unsigned &dl1_accesses,
                         unsigned &dl1_misses);
  void get_cache_stats(unsigned &read_accesses, unsigned &write_accesses,
                       unsigned &read_misses, unsigned &write_misses,
                       unsigned cache_type);
  void get_cache_stats(cache_stats &cs);

  void get_L1D_sub_stats(struct cache_sub_stats &css) const;
  void get_L1C_sub_stats(struct cache_sub_stats &css) const;
  void get_L1T_sub_stats(struct cache_sub_stats &css) const;

 protected:
  ldst_unit(mem_fetch_interface *icnt,
            shader_core_mem_fetch_allocator *mf_allocator,
            shader_core_ctx *core, opndcoll_rfu_t *operand_collector,
            Scoreboard *scoreboard, Scoreboard_reads *scoreboard_reads, const shader_core_config *config, // MOD. Fix WAR at baseline.
            const memory_config *mem_config, shader_core_stats *stats,
            unsigned sid, unsigned tpc, l1_cache *new_l1d_cache);
  void init(mem_fetch_interface *icnt,
            shader_core_mem_fetch_allocator *mf_allocator,
            shader_core_ctx *core, opndcoll_rfu_t *operand_collector,
            Scoreboard *scoreboard, Scoreboard_reads *scoreboard_reads, const shader_core_config *config, // MOD. Fix WAR at baseline.
            const memory_config *mem_config, shader_core_stats *stats,
            unsigned sid, unsigned tpc);

 protected:
  bool shared_cycle(warp_inst_t &inst, mem_stage_stall_type &rc_fail,
                    mem_stage_access_type &fail_type);
  bool constant_cycle(warp_inst_t &inst, mem_stage_stall_type &rc_fail,
                      mem_stage_access_type &fail_type);
  bool texture_cycle(warp_inst_t &inst, mem_stage_stall_type &rc_fail,
                     mem_stage_access_type &fail_type);
  bool memory_cycle(warp_inst_t &inst, mem_stage_stall_type &rc_fail,
                    mem_stage_access_type &fail_type);

  virtual mem_stage_stall_type process_cache_access(
      cache_t *cache, new_addr_type address, warp_inst_t &inst,
      std::list<cache_event> &events, mem_fetch *mf,
      enum cache_request_status status);
  mem_stage_stall_type process_memory_access_queue(cache_t *cache,
                                                   warp_inst_t &inst);
  mem_stage_stall_type process_memory_access_queue_l1cache(l1_cache *cache,
                                                           warp_inst_t &inst);

  unsigned get_first_key_pending_writes(warp_inst_t *inst); // MOD. LOOG
  unsigned get_second_key_pending_writes(warp_inst_t *inst, int idx); // MOD. VPREG

  const memory_config *m_memory_config;
  class mem_fetch_interface *m_icnt;
  shader_core_mem_fetch_allocator *m_mf_allocator;
  class shader_core_ctx *m_core;
  unsigned m_sid;
  unsigned m_tpc;

  tex_cache *m_L1T;        // texture cache
  read_only_cache *m_L1C;  // constant cache
  l1_cache *m_L1D;         // data cache
  std::map<unsigned /*warp_id*/,
           std::map<unsigned /*regnum*/, unsigned /*count*/>>
      m_pending_writes;
  std::list<mem_fetch *> m_response_fifo;
  opndcoll_rfu_t *m_operand_collector;
  Scoreboard *m_scoreboard;
  Scoreboard_reads *m_scoreboard_reads; // MOD. Fix WAR at baseline.
  unsigned long long m_dispatch_reg_allocation_cycle; // MOD. Memory stats

  mem_fetch *m_next_global;
  warp_inst_t m_next_wb;
  unsigned m_writeback_arb;  // round-robin arbiter for writeback contention
                             // between L1T, L1C, shared
  unsigned m_num_writeback_clients;

  enum mem_stage_stall_type m_mem_rc;

  shader_core_stats *m_stats;

  // for debugging
  unsigned long long m_last_inst_gpu_sim_cycle;
  unsigned long long m_last_inst_gpu_tot_sim_cycle;

  std::vector<std::deque<mem_fetch *>> l1_latency_queue;
  void L1_latency_queue_cycle();

  void print_L1_latency_queue(FILE *f); // MOD. VPREG
};

enum pipeline_stage_name_t {
  ID_OC_SP = 0,
  ID_OC_DP,
  ID_OC_INT,
  ID_OC_SFU,
  ID_OC_MEM,
  OC_EX_SP,
  OC_EX_DP,
  OC_EX_INT,
  OC_EX_SFU,
  OC_EX_MEM,
  EX_WB,
  ID_OC_TENSOR_CORE,
  OC_EX_TENSOR_CORE,
  N_PIPELINE_STAGES
};

const char *const pipeline_stage_name_decode[] = {
    "ID_OC_SP",          "ID_OC_DP",         "ID_OC_INT", "ID_OC_SFU",
    "ID_OC_MEM",         "OC_EX_SP",         "OC_EX_DP",  "OC_EX_INT",
    "OC_EX_SFU",         "OC_EX_MEM",        "EX_WB",     "ID_OC_TENSOR_CORE",
    "OC_EX_TENSOR_CORE", "N_PIPELINE_STAGES"};

struct specialized_unit_params {
  unsigned latency;
  unsigned num_units;
  unsigned id_oc_spec_reg_width;
  unsigned oc_ex_spec_reg_width;
  char name[20];
  unsigned ID_OC_SPEC_ID;
  unsigned OC_EX_SPEC_ID;
};

class shader_core_config : public core_config {
 public:
  shader_core_config(gpgpu_context *ctx) : core_config(ctx) {
    pipeline_widths_string = NULL;
    gpgpu_ctx = ctx;
  }

  void init() {
    int ntok = sscanf(gpgpu_shader_core_pipeline_opt, "%d:%d",
                      &n_thread_per_shader, &warp_size);
    if (ntok != 2) {
      printf(
          "GPGPU-Sim uArch: error while parsing configuration string "
          "gpgpu_shader_core_pipeline_opt\n");
      abort();
    }

    char *toks = new char[100];
    char *tokd = toks;
    strcpy(toks, pipeline_widths_string);

    toks = strtok(toks, ",");

    /*	Removing the tensorcore pipeline while reading the config files if the
       tensor core is not available. If we won't remove it, old regression will
       be broken. So to support the legacy config files it's best to handle in
       this way.
     */
    int num_config_to_read = N_PIPELINE_STAGES - 2 * (!gpgpu_tensor_core_avail);

    for (int i = 0; i < num_config_to_read; i++) {
      assert(toks);
      ntok = sscanf(toks, "%d", &pipe_widths[i]);
      assert(ntok == 1);
      toks = strtok(NULL, ",");
    }

    delete[] tokd;

    if (n_thread_per_shader > MAX_THREAD_PER_SM) {
      printf(
          "GPGPU-Sim uArch: Error ** increase MAX_THREAD_PER_SM in "
          "abstract_hardware_model.h from %u to %u\n",
          MAX_THREAD_PER_SM, n_thread_per_shader);
      abort();
    }
    max_warps_per_shader = n_thread_per_shader / warp_size;
    assert(!(n_thread_per_shader % warp_size));

    set_pipeline_latency();

    m_L0I_config.init(m_L0I_config.m_config_string, FuncCachePreferNone); // MOD. Added L0I
    m_L1I_L1_half_C_cache_config.init(m_L1I_L1_half_C_cache_config.m_config_string, FuncCachePreferNone);
    m_L1T_config.init(m_L1T_config.m_config_string, FuncCachePreferNone);
    m_L1C_config.init(m_L1C_config.m_config_string, FuncCachePreferNone);
    m_L0C_config.init(m_L0C_config.m_config_string, FuncCachePreferNone);
    m_L1D_config.init(m_L1D_config.m_config_string, FuncCachePreferNone);
    gpgpu_cache_texl1_linesize = m_L1T_config.get_line_sz();
    gpgpu_cache_constl1_linesize = m_L1C_config.get_line_sz();
    m_valid = true;

    m_specialized_unit_num = 0;
    // parse the specialized units
    for (unsigned i = 0; i < SPECIALIZED_UNIT_NUM; ++i) {
      unsigned enabled;
      specialized_unit_params sparam;
      sscanf(specialized_unit_string[i], "%u,%u,%u,%u,%u,%s", &enabled,
             &sparam.num_units, &sparam.latency, &sparam.id_oc_spec_reg_width,
             &sparam.oc_ex_spec_reg_width, sparam.name);

      if (enabled) {
        m_specialized_unit.push_back(sparam);
        strncpy(m_specialized_unit.back().name, sparam.name,
                sizeof(m_specialized_unit.back().name));
        m_specialized_unit_num += sparam.num_units;
      } else
        break;  // we only accept continuous specialized_units, i.e., 1,2,3,4
    }

    // parse gpgpu_shmem_option for adpative cache config
    if (adaptive_cache_config) {
      std::stringstream ss(gpgpu_shmem_option);
      while (ss.good()) {
        std::string option;
        std::getline(ss, option, ',');
        shmem_opt_list.push_back((unsigned)std::stoi(option) * 1024);
      }
      std::sort(shmem_opt_list.begin(), shmem_opt_list.end());
    }
  }
  void reg_options(class OptionParser *opp);
  unsigned int max_cta(const kernel_info_t &k) const;
  unsigned int num_shader() const {
    return n_simt_clusters * n_simt_cores_per_cluster;
  }
  unsigned sid_to_cluster(unsigned sid) const {
    return sid / n_simt_cores_per_cluster;
  }
  unsigned sid_to_cid(unsigned sid) const {
    return sid % n_simt_cores_per_cluster;
  }
  unsigned cid_to_sid(unsigned cid, unsigned cluster_id) const {
    return cluster_id * n_simt_cores_per_cluster + cid;
  }
  void set_pipeline_latency();

  // backward pointer
  class gpgpu_context *gpgpu_ctx;
  // data
  char *gpgpu_shader_core_pipeline_opt;
  bool gpgpu_perfect_mem;
  bool gpgpu_clock_gated_reg_file;
  bool gpgpu_clock_gated_lanes;
  enum divergence_support_t model;
  unsigned int n_thread_per_shader;
  unsigned int n_regfile_gating_group;
  unsigned int max_warps_per_shader;
  unsigned
      max_cta_per_core;  // Limit on number of concurrent CTAs in shader core
  unsigned max_barriers_per_cta;
  char *gpgpu_scheduler_string;
  unsigned gpgpu_shmem_per_block;
  unsigned gpgpu_registers_per_block;
  char *pipeline_widths_string;
  int pipe_widths[N_PIPELINE_STAGES];

  mutable cache_config m_L0I_config; // MOD. Added L0I
  mutable cache_config m_L1I_L1_half_C_cache_config;
  mutable cache_config m_L1T_config;
  mutable cache_config m_L1C_config;
  mutable cache_config m_L0C_config;
  mutable l1d_cache_config m_L1D_config;

  bool gpgpu_dwf_reg_bankconflict;

  unsigned gpgpu_num_sched_per_core;
  int gpgpu_max_insn_issue_per_warp;
  bool gpgpu_dual_issue_diff_exec_units;

  // op collector
  bool enable_specialized_operand_collector;
  int gpgpu_operand_collector_num_units_sp;
  int gpgpu_operand_collector_num_units_dp;
  int gpgpu_operand_collector_num_units_sfu;
  int gpgpu_operand_collector_num_units_tensor_core;
  int gpgpu_operand_collector_num_units_mem;
  unsigned int gpgpu_operand_collector_num_units_gen;
  int gpgpu_operand_collector_num_units_int;

  unsigned int gpgpu_operand_collector_num_in_ports_sp;
  unsigned int gpgpu_operand_collector_num_in_ports_dp;
  unsigned int gpgpu_operand_collector_num_in_ports_sfu;
  unsigned int gpgpu_operand_collector_num_in_ports_tensor_core;
  unsigned int gpgpu_operand_collector_num_in_ports_mem;
  unsigned int gpgpu_operand_collector_num_in_ports_gen;
  unsigned int gpgpu_operand_collector_num_in_ports_int;

  unsigned int gpgpu_operand_collector_num_out_ports_sp;
  unsigned int gpgpu_operand_collector_num_out_ports_dp;
  unsigned int gpgpu_operand_collector_num_out_ports_sfu;
  unsigned int gpgpu_operand_collector_num_out_ports_tensor_core;
  unsigned int gpgpu_operand_collector_num_out_ports_mem;
  unsigned int gpgpu_operand_collector_num_out_ports_gen;
  unsigned int gpgpu_operand_collector_num_out_ports_int;

  unsigned int gpgpu_num_sp_units;
  unsigned int gpgpu_tensor_core_avail;
  unsigned int gpgpu_num_dp_units;
  unsigned int gpgpu_num_sfu_units;
  unsigned int gpgpu_num_tensor_core_units;
  unsigned int gpgpu_num_mem_units;
  unsigned int gpgpu_num_int_units;

  // Shader core resources
  unsigned gpgpu_shader_registers;
  int gpgpu_warpdistro_shader;
  int gpgpu_warp_issue_shader;
  unsigned gpgpu_num_reg_banks;
  bool gpgpu_reg_bank_use_warp_id;
  bool gpgpu_local_mem_map;
  bool gpgpu_ignore_resources_limitation;
  bool sub_core_model;

  unsigned max_sp_latency;
  unsigned max_int_latency;
  unsigned max_sfu_latency;
  unsigned max_dp_latency;
  unsigned max_tensor_core_latency;

  unsigned n_simt_cores_per_cluster;
  unsigned n_simt_clusters;
  unsigned n_simt_ejection_buffer_size;
  unsigned ldst_unit_response_queue_size;

  int simt_core_sim_order;

  unsigned smem_latency;

  unsigned mem2device(unsigned memid) const { return memid + n_simt_clusters; }

  // Jin: concurrent kernel on sm
  bool gpgpu_concurrent_kernel_sm;

  bool perfect_inst_const_cache;
  unsigned inst_fetch_throughput;
  unsigned reg_file_port_throughput;

  // specialized unit config strings
  char *specialized_unit_string[SPECIALIZED_UNIT_NUM];
  mutable std::vector<specialized_unit_params> m_specialized_unit;
  unsigned m_specialized_unit_num;

  bool is_trace_predication_enabled; // MOD. Predication
  // MOD. Begin. Fix WAR at baseline.
  char *scoreboard_war_mode; // Indicates the mode of use of the scoreboard_reads in order to fix the war hazards at the baseline with a string
  scoreboard_reads_mode scoreboard_war_reads_mode; // Indicates the mode of use of the scoreboard_reads in order to fix the war hazards at the baseline with an enum
  unsigned int scoreboard_war_max_uses_per_reg; // Maximum of concurrent uses per register in the scoreboard_reads
  double scoreboard_war_static_power;
  double scoreboard_war_dynamic_power;
  // MOD. End

  bool is_fix_memory_reordering_enabled_baseline;  // MOD. Fix loads after stores in the baseline.

  bool is_L0I_enabled; // MOD. Added L0I
  bool is_fix_instruction_fetch_misalignment; // MOD. Fix misaligned fetched instructions
  bool is_fix_different_kernels_pc_addresses; // MOD. Fix instruction addresses of different kernels to have a different address request in memory
  bool is_fix_not_decoding_not_contiguos_instructions; // MOD. Not decoding instructions that have separated PC.
  bool is_improved_ldst_unit_enabled; // MOD. Fixed LDST_Unit model.
  bool is_improved_result_bus; // MOD. Improved Result bus to take into account conflicts with RF banks.
  int max_request_allowed_to_L1I; // MOD. Added L0I
  int max_reply_allowed_from_L1I; // MOD. Added L0I
  int latency_L0_to_L1; // MOD. Added L0I
  int latency_L1_to_L0; // MOD. Added L0I
  bool is_fetch_and_decode_improved; // MOD. Improving fetch and decode
  bool is_opc_improved; // MOD. Improving OPC
  int cu_num_ports; // MOD. Improving OPC
  bool is_skip_rf_limit_enabled; // MOD. Skip RF limitation.
  bool is_relax_barriers_baseline; // MOD. Relax barriers in baseline

  concrete_scheduler warp_scheduling_mode;

  bool is_trace_mode; // MOD. General Config Helper
  unsigned int filter_first_kernel_id; // If it has a value of 1 or 0 it is disabled
  unsigned int filter_last_kernel_id; // If it has a value of 1 or 0 it is disabled


  // MOD. Begin. Extended IBuffer
  bool is_extended_ibuffer_enabled;
  int extended_ibuffer_size;
  int fetch_decode_width;
  double extended_ibuffer_static_power;
  double extended_ibuffer_dynamic_power;
  // MOD. End. Extended IBuffer

  // MOD. Begin. LOOG
  bool is_loog_enabled;
  int loog_frontend_size;
  int loog_rrs_size;
  int loog_memory_queues_size;
  // MOD. End. LOOG

  // MOD. Begin VPREG
  char *vpreg_mode_string;
  bool is_vpreg_enabled;
  bool is_vpreg_predicated_war_waw_dependencies_ignored;
  bool is_vpreg_predicated_dest_reg_dependencies_ignored;
  bool is_vpreg_balanced_banks_mode_enabled;
  int vpreg_num_virtual_regs_per_sm;
  int vpreg_num_physical_regs_per_sm;
  int vpreg_reissue_informed_socgpu_threshold;
  int vpreg_max_rollback_entries_done_in_a_cycle;

  double vpreg_merge_module_static_power;
  double vpreg_merge_module_dynamic_power;
  double vpreg_collector_unit_extra_static_power;
  double vpreg_collector_unit_extra_dynamic_power;
  // MOD. Begin VPREG
  // MOD. Begin. Remodeling
  bool is_SM_remodeling_enabled; 
  bool is_remodeling_scoreboarding_enabled; 
  int num_subcores_in_SM;
  bool is_ibuffer_remodeled_enabled;
  int ibuffer_remodeled_size;
  unsigned int num_wait_barriers_per_warp;
  int sfu_latency;
  int tensor_latency;
  int tensor_extra_latency_16816_fp32_1688_fp32;
  int tensor_rate_per_cycle;
  int branch_latency;
  int half_latency;
  int uniform_latency;
  unsigned int predicate_latency;
  int miscellaneous_queue_latency;
  int miscellaneous_no_queue_latency;
  int sfu_initiation;
  int tensor_initiation;
  int branch_initiation;
  int half_initiation;
  int uniform_initiation;
  int predicate_initiation;
  int miscellaneous_queue_initiation;
  int miscellaneous_no_queue_initiation;
  unsigned int memory_intermidiate_stages_subcore_unit;
  unsigned int dp_shared_intermidiate_stages;
  unsigned int miscellaneous_queue_size;
  unsigned int memory_subcore_queue_size;
  unsigned int memory_sm_prt_size;
  unsigned int num_cycles_to_wait_to_dispatch_another_inst_from_subcore_to_sm_shared_pipeline_when_is_mem_inst;
  unsigned int num_cycles_to_wait_to_dispatch_another_inst_from_subcore_to_sm_shared_pipeline_when_is_dp_inst;
  unsigned int memory_shared_memory_minimum_latency;
  unsigned int memory_shared_memory_extra_latency_ldsm_multiple_matrix;
  unsigned int memmory_max_concurrent_requests_shmem_per_sm;
  unsigned int memmory_max_concurrent_requests_standard_per_sm;
  unsigned int sm_memory_unit_l1c_access_queue_size;
  unsigned int sm_memory_unit_l1t_access_queue_size;
  unsigned int sm_memory_unit_l1d_access_queue_size;
  unsigned int sm_memory_unit_shmem_access_queue_size;
  unsigned int sm_memory_unit_bypass_l1d_directly_go_to_l2_access_queue_size;
  unsigned int sm_memory_unit_miscellaneous_access_queue_size;
  unsigned int constant_cache_latency_at_sm_structure;
  unsigned int constant_cache_miss_latency_at_subcore_to_access_upper_level;
  unsigned int memory_l1d_minimum_latency;
  unsigned int memory_global_shared_latency_for_ldgsts;
  unsigned int memory_l1d_max_lookups_per_cycle_per_bank;
  unsigned int memory_maximum_coalescing_cycles;
  unsigned int memory_subcore_extra_latency_load_shared_mem;
  unsigned int memory_num_scalar_units_per_subcore;
  unsigned int cycles_needed_for_address_calculation;
  unsigned int memory_subcore_link_to_sm_byte_size;
  unsigned int maximum_l1d_latency_at_sm_structure;
  unsigned int maximum_shared_memory_latency_at_sm_structure;
  unsigned int dp_subcore_queue_size;
  unsigned int dp_subcore_max_latency;
  unsigned int dp_sm_shared_queue_size;
  bool is_dp_pipeline_shared_for_subcores;
  bool is_load_half_bandwidth_in_the_subcore_link_to_sm_enabled;
  bool is_store_half_bandwidth_in_the_subcore_link_to_sm_enabled;
  bool is_fp32ops_allowed_in_int_pipeline;
  bool is_fp32_and_int_unified_pipeline;
  bool is_const_cache_accessed_blocks_tracking_enabled;
  bool is_global_memory_accesses_blocks_tracking_enabled;
  bool is_num_virtual_pages_tracking_enabled;
  unsigned int virtual_page_size_in_bytes;
  unsigned int num_const_cache_cycle_misses_before_switch_to_other_warp;
  unsigned int num_cycles_issue_port_busy_after_imadwide;
  unsigned int num_stall_cycles_wait_after_bits_stall_0_and_yield;
  unsigned int num_cycles_to_stall_SM_at_gpu_memory_barrier;
  unsigned int num_cycles_to_stall_SM_at_system_memory_barrier;
  unsigned int num_cycles_to_stall_SM_at_cta_memory_barrier;
  
  int offset_latency_firts_stage_memory_subcore;

  bool invalidate_instruction_caches_at_kernel_end;
  bool ibuffer_coalescing;
  bool perfect_instruction_cache;
  bool perfect_constant_cache;
  bool is_instruction_prefetching_enabled;
  unsigned int prefetch_per_stream_buffer_size;
  unsigned int prefetch_num_stream_buffers;
  unsigned int num_instruction_prefetches_per_cycle;

  bool is_rf_cache_enabled;
  int max_operands_regular_register_file; 
  int max_latency_regular_register_file_latency; 
  int num_regular_register_file_read_ports_per_bank;
  int num_regular_register_file_write_ports_per_bank;
  int max_size_register_file_write_queue_for_fixed_latency_instructions;
  int max_pops_per_cycle_register_file_write_queue_for_fixed_latency_instructions;
  int num_threads_granularity_read_regular_register_file_dp_inst;
  int num_threads_granularity_read_regular_register_file_mem_inst;
  int num_threads_granularity_read_regular_register_file_sfu_inst;
  int num_threads_granularity_read_regular_register_file_other_inst;
  int num_cycles_needed_to_write_a_reg_from_sm_struct_to_subcore;
  // MOD. End. Remodeling

  // MOD. Begin. Parallelism
  bool is_custom_omp_scheduler_enabled;
  float custom_omp_scheduler_ratio_to_dynamic;
  // MOD. End. Parallelism

  // MOD. Begin. InterWarp coalescing
  bool measure_coalescing_potential_stats;
  bool is_interwarp_coalescing_enabled;
  unsigned int num_interwarp_coalescing_tables;
  unsigned int max_size_interwarp_coalescing_per_table;
  unsigned int interwarp_coalescing_quanta;
  double interwarp_coalescing_quanta_warppool_policy_miss_ratio_threshold;
  unsigned int number_of_coalescers;
  unsigned int number_of_clusters_for_prt_selection;
  char* interwarp_coalescing_selection_policy_string;
  char* prt_selection_policy_string;
  InterWarpCoalescingSelectionPolicies interwarp_coalescing_selection_policy;
  PRTSelectionPolicies prt_selection_policy;
  // MOD. End. InterWarp coalescing
};

struct shader_core_stats_pod {
  void *
      shader_core_stats_pod_start[0];  // DO NOT MOVE FROM THE TOP - spaceless
                                       // pointer to the start of this structure
  unsigned long long *shader_cycles;

  // MOD. Begin. Custom Stats
  //First dimension is the number of kernel. Second dimension is the number of SM of the GPU
  std::vector<std::vector<unsigned long long>> m_num_sim_winsn_per_shader_per_kernel;
  std::vector<std::vector<unsigned long long>> shader_active_warps_per_kernel;
  std::vector<std::vector<unsigned long long>> shader_maximum_theoretical_warps_per_kernel;
  std::vector<std::vector<unsigned long long>> shader_cycles_per_kernel;
  std::vector<std::vector<double>> shader_warp_ipc_per_kernel;
  std::vector<std::vector<double>> shader_occupancy_per_kernel;
  std::vector<unsigned long long> gpu_cycles_per_kernel;
  std::vector<double> weighted_average_shader_warp_ipc_per_kernel;
  std::vector<double> weighted_average_shader_occupancy_per_kernel;
  std::vector<double> average_num_shader_active_per_kernel;
  std::vector<unsigned long long> number_of_warps_per_kernel;
  std::vector<unsigned long long> m_num_sim_winsn_per_shader;
  std::vector<double> shader_warp_ipc_per_shader;

  double total_weighted_average_warp_ipc_between_shaders;
  double total_weighted_average_shader_warp_ipc_with_kernels;
  double total_weighted_average_shader_occupancy;
  double total_weighted_average_num_shader_active;

  long double total_weighted_average_warps_per_kernel;

  unsigned long long number_of_total_warps;

  unsigned m_last_kernel_id;
  unsigned m_current_kernel_pos;
  unsigned numEffectiveIncompleteWarps;
  unsigned numberOfTotalWarps;

  unsigned long long tot_scheduler_cycles;
  unsigned long long tot_scheduler_issues;

  unsigned long long tot_num_expected_wb; // MOD. Custom stats
  unsigned long long tot_num_allocated_wb; // MOD. Custom stats

  unsigned long long tot_fetch_instruction_misalignments; // MOD. Fix misaligned fetched instructions
  unsigned long long tot_fetch_requests; // MOD. Fix misaligned fetched instructions

  unsigned num_scoreboard_reads_check_collision;// MOD. Scoreboard_reads
  unsigned num_scoreboard_reads_collision_due_to_max_uses_per_reg;// MOD. Scoreboard_reads
  unsigned num_scheduler_stall_cycle_due_to_war_scoreboard; // MOD. Scoreboard_reads
  unsigned num_scheduler_stall_cycle_dependencies_other_reasons_not_war_scoreboard; // MOD. Scoreboard_reads


  // MOD. Begin. IBuffer_ooo stats
  // First dimension is kernel, second is shader id, third dimension is warp
   
  std::vector<std::vector<std::vector<unsigned long long>>> ins_issued_per_kernel_per_sid_per_warp;
  std::vector<std::vector<std::vector<unsigned long long>>> ins_released_wb_per_kernel_per_sid_per_warp;
  std::vector<std::vector<std::vector<unsigned long long>>> ins_released_opc_per_kernel_per_sid_per_warp;
  std::vector<std::vector<std::vector<unsigned long long>>> num_flushes_kernel_per_sid_per_warp;
  std::vector<std::vector<std::vector<unsigned long long>>> num_times_ibooo_empty;
  std::vector<std::vector<std::vector<unsigned long long>>> num_times_ibooo_empty_evaluated;
  std::vector<std::vector<std::vector<unsigned long long>>> num_times_ibooo_full;
  std::vector<std::vector<std::vector<unsigned long long>>> num_times_fetch_ibooo_tried;

  unsigned long long total_ins_issued_per_kernel_per_sid_per_warp;
  unsigned long long total_ins_released_wb_per_kernel_per_sid_per_warp;
  unsigned long long total_ins_released_opc_per_kernel_per_sid_per_warp;
  unsigned long long total_num_flushes_kernel_per_sid_per_warp;
  unsigned long long total_num_barriers;
  unsigned long long total_num_returns;
  unsigned long long total_num_branches;
  unsigned long long total_num_jumps;
  unsigned long long total_num_warpsyncs;
  unsigned long long total_num_bsyncs;
  unsigned long long total_num_rpcmovs;
  unsigned long long total_num_yields;
  unsigned long long total_num_barriers_and_controlflows;
  unsigned long long total_num_times_ibooo_empty;
  unsigned long long total_num_times_ibooo_empty_evaluated;
  unsigned long long total_num_times_ibooo_full;
  unsigned long long total_num_times_fetch_ibooo_tried;
  unsigned long long total_ibooo_num_entries_valid_and_not_issued;
  unsigned long long total_ibooo_num_entries_valid_not_issued_and_ready;
  unsigned long long total_ibooo_num_entries;
  unsigned long long total_ibooo_num_times_without_any_candidate;
  unsigned long long total_ibooo_num_times_without_any_ready_candidate;
  unsigned long long total_ibooo_evaluations_compute_selection_stats;
  double total_percentage_ibooo_full;
  double total_percentage_ibooo_empty;
  unsigned long long total_instructions_inserted_in_ibooo;
  unsigned long long total_war_waw_dependencies;
  unsigned long long total_raw_dependencies;
  unsigned long long total_stop_point_dependencies;
  unsigned long long total_memory_reordering_dependencies;

  unsigned long long last_ins_issued_per_kernel_per_sid_per_warp;
  unsigned long long last_ins_released_wb_per_kernel_per_sid_per_warp;
  unsigned long long last_ins_released_opc_per_kernel_per_sid_per_warp;
  unsigned long long last_num_flushes_kernel_per_sid_per_warp;
  unsigned long long last_num_times_ibooo_empty;
  unsigned long long last_num_times_ibooo_empty_evaluated;
  unsigned long long last_num_times_ibooo_full;
  unsigned long long last_num_times_fetch_ibooo_tried;
  double last_percentage_ibooo_full;
  double last_percentage_ibooo_empty;
  // MOD. End. IBuffer_ooo stats

  // MOD. Begin. VPREG
  unsigned long long total_vpreg_predication_dependencies;
  unsigned long long total_vpreg_merges;
  unsigned long long total_vpreg_extra_rf_reads;
  unsigned long long total_rf_reads;
  unsigned int total_number_of_kernels_limited_by_regs;
  unsigned int total_number_of_kernels_limited_by_ctas;
  unsigned int total_number_of_kernels_limited_by_threads;
  unsigned int total_number_of_kernels_limited_by_shared_memory;
  unsigned int total_number_of_vpreg_decode_rollbacks;
  unsigned int total_number_of_vpreg_reissues;
  unsigned int total_number_of_vpreg_not_enough_virtual_at_decode;
  int max_vpreg_virtual_regs_used_in_subcore;
  int max_vpreg_physical_regs_used_in_subcore;
  int max_vpreg_physical_freepool_usage_in_bank;
  int max_vpreg_number_of_consumers;
  // MOD. End. VPREG

  // MOD. Begin. OPC custom stats
  unsigned long long total_number_of_opc_conflicts;
  unsigned long long total_number_of_opc_requests;
  
  unsigned long long num_times_cu_subcore_custom_stats_evaluated;
  unsigned long long num_times_no_cu_dispatched;
  unsigned long long num_times_no_cu_allocated_and_nothing_to_allocate;
  unsigned long long num_times_no_cu_allocated;
  unsigned long long num_times_no_cu_allocated_due_to_cus_are_full;
  unsigned long long num_times_no_cu_dispatched_due_to_dispatch_reg_full;
  unsigned long long num_times_no_cu_dispatched_due_to_no_ready_operands;
  unsigned long long num_times_no_cu_dispatched_due_to_all_cus_empty;
  unsigned long long num_times_no_cu_dispatched_and_all_cus_full;
  unsigned long long num_times_no_cu_dispatched_and_all_cus_full_and_at_least_one_ready;
  unsigned long long num_times_no_cu_dispatched_and_all_cus_full_and_not_any_ready;
  unsigned long long num_times_no_cu_dispatched_due_to_dispatch_reg_full_is_mem_op_and_ldst_unit_stalled;
  unsigned long long num_times_no_cu_dispatched_and_all_cus_full_and_at_least_one_ready_mem_dispatch_full_and_ldst_unit_stalled;
  unsigned long long total_num_try_ldst_unit_dispatches;
  unsigned long long total_num_ldst_unit_dispatches_failed_due_to_not_empty_dispatch_reg;
  // MOD. End. OPC custom stats

  // MOD. Begin. Memory stats
  std::vector<std::vector<unsigned long long>> l1d_accesses_per_sid_per_bank;
  std::vector<std::vector<unsigned long long>> l1d_evals_per_sid_per_bank;
  long double total_avg_usage_l1d_bank;
  double max_avg_usage_l1d_bank;
  unsigned long long total_shared_mem_evals;
  unsigned long long total_shared_mem_accesses;
  unsigned long long total_num_dp_instructions;
  unsigned long long total_num_ldst_unit_instructions;
  unsigned long long total_num_warp_instructions;
  unsigned long long total_l1d_instructions;
  unsigned long long total_accesses_l1d_instructions;
  unsigned long long total_avg_cycles_to_schedule_accesses;
  unsigned long long total_shared_instructions;
  unsigned long long total_conflicts_shared_instructions;
  unsigned long long total_cycles_instructions_in_ldst_unit_dispatch_reg;
  unsigned long long total_cycles_instructions_in_ldst_unit_arbiter_latch; // MOD. Fixed LDST_Unit model.
  // MOD. End. Memory stats

  unsigned long long total_cycles_instructions_in_cu; // MOD. CU stats

  // First dimension SM, second warp ID
  std::vector<std::vector<unsigned long long>> warp_issues_from_last_power_sample; // MOD. Custom powermodel stats
  std::vector<std::vector<unsigned long long>> bank_wb_from_last_power_sample; // MOD. Custom powermodel stats 
  std::vector<std::vector<unsigned long long>> collector_unit_allocations_from_last_power_sample; // MOD. Custom powermodel stats 
  // MOD. End. custom Stats


  unsigned *m_num_sim_insn;   // number of scalar thread instructions committed
                              // by this shader core
  unsigned *m_num_sim_winsn;  // number of warp instructions committed by this
                              // shader core
  unsigned *m_last_num_sim_insn;
  unsigned *m_last_num_sim_winsn;
  unsigned *
      m_num_decoded_insn;  // number of instructions decoded by this shader core
  float *m_pipeline_duty_cycle;
  unsigned *m_num_FPdecoded_insn;
  unsigned *m_num_INTdecoded_insn;
  unsigned *m_num_storequeued_insn;
  unsigned *m_num_loadqueued_insn;
  unsigned *m_num_tex_inst;
  double *m_num_ialu_acesses;
  double *m_num_fp_acesses;
  double *m_num_imul_acesses;
  double *m_num_fpmul_acesses;
  double *m_num_idiv_acesses;
  double *m_num_fpdiv_acesses;
  double *m_num_sp_acesses;
  double *m_num_sfu_acesses;
  double *m_num_tensor_core_acesses;
  double *m_num_tex_acesses;
  double *m_num_const_acesses;
  double *m_num_dp_acesses;
  double *m_num_dpmul_acesses;
  double *m_num_dpdiv_acesses;
  double *m_num_sqrt_acesses;
  double *m_num_log_acesses;
  double *m_num_sin_acesses;
  double *m_num_exp_acesses;
  double *m_num_mem_acesses;
  unsigned *m_num_sp_committed;
  unsigned *m_num_tlb_hits;
  unsigned *m_num_tlb_accesses;
  unsigned *m_num_sfu_committed;
  unsigned *m_num_tensor_core_committed;
  unsigned *m_num_mem_committed;
  unsigned *m_read_regfile_acesses;
  unsigned *m_write_regfile_acesses;
  unsigned *m_non_rf_operands;
  double *m_num_imul24_acesses;
  double *m_num_imul32_acesses;
  unsigned *m_active_sp_lanes;
  unsigned *m_active_sfu_lanes;
  unsigned *m_active_tensor_core_lanes;
  unsigned *m_active_fu_lanes;
  unsigned *m_active_fu_mem_lanes;
  double *m_active_exu_threads; //For power model
  double *m_active_exu_warps; //For power model
  unsigned *m_n_diverge;  // number of divergence occurring in this shader
  unsigned long long gpgpu_n_load_insn;
  unsigned long long gpgpu_n_store_insn;
  unsigned long long gpgpu_n_shmem_insn;
  unsigned long long gpgpu_n_sstarr_insn;
  unsigned long long gpgpu_n_tex_insn;
  unsigned long long gpgpu_n_const_insn;
  unsigned long long gpgpu_n_param_insn;
  unsigned long long gpgpu_n_shmem_bkconflict;
  unsigned long long gpgpu_n_l1cache_bkconflict;
  unsigned long long gpgpu_n_l1cache_coalescing_conflicts;
  int gpgpu_n_intrawarp_mshr_merge;
  unsigned gpgpu_n_cmem_portconflict;
  unsigned gpgpu_n_cmem_coalescing_conflicts;
  unsigned gpu_stall_shd_mem_breakdown[N_MEM_STAGE_ACCESS_TYPE]
                                      [N_MEM_STAGE_STALL_TYPE];
  unsigned gpu_reg_bank_conflict_stalls;
  unsigned *shader_cycle_distro;
  unsigned *last_shader_cycle_distro;
  unsigned *num_warps_issuable;
  unsigned gpgpu_n_stall_dispatch_to_subpipeline_mem;
  unsigned *single_issue_nums;
  unsigned *dual_issue_nums;

  unsigned ctas_completed;
  // memory access classification
  int gpgpu_n_mem_read_local;
  int gpgpu_n_mem_write_local;
  int gpgpu_n_mem_texture;
  int gpgpu_n_mem_const;
  int gpgpu_n_mem_read_global;
  int gpgpu_n_mem_write_global;
  int gpgpu_n_mem_read_inst;

  int gpgpu_n_mem_l2_writeback;
  int gpgpu_n_mem_l1_write_allocate;
  int gpgpu_n_mem_l2_write_allocate;

  unsigned made_write_mfs;
  unsigned made_read_mfs;

  unsigned *gpgpu_n_shmem_bank_access;
  long *n_simt_to_mem;  // Interconnect power stats
  long *n_mem_to_simt;

  // MOD. Begin. Remodeling
  unsigned long long total_num_register_file_cache_hits;
  unsigned long long total_num_register_file_cache_allocations;
  unsigned long long total_num_regular_regfile_reads;
  unsigned long long total_num_regular_regfile_writes;
  unsigned long long total_num_uniform_regfile_reads;
  unsigned long long total_num_uniform_regfile_writes;
  unsigned long long total_num_predicate_regfile_reads;
  unsigned long long total_num_predicate_regfile_writes;
  unsigned long long total_num_uniform_predicate_regfile_reads;
  unsigned long long total_num_uniform_predicate_regfile_writes;
  unsigned long long total_num_constant_cache_reads;
  std::set<new_addr_type> all_const_cache_accessed_blocks;
  std::set<new_addr_type> all_global_memory_accessed_blocks;
  std::set<unsigned int> all_virtual_pages_accessed;

  unsigned long long total_num_times_wb_evaluated;
  unsigned long long total_num_times_wb_port_conflict;

  unsigned long long total_num_cycles_issue_stage_evaluated;
  unsigned long long total_num_cycles_issue_stage_issuing;
  unsigned long long total_num_cycles_issue_stage_stall_issue_port_busy;
  unsigned long long total_num_cycles_issue_stage_stall_no_valid_instruction;
  unsigned long long total_num_cycles_issue_stage_stall_no_warps_ready;

  unsigned long long total_num_cycles_issue_stage_stall_at_least_one_warp_with_fu_occupied;
  unsigned long long total_num_cycles_issue_stage_stall_at_least_one_warp_waiting_inst_barrier;
  unsigned long long total_num_cycles_issue_stage_stall_at_least_one_warp_waiting_wait_barrier;
  unsigned long long total_num_cycles_issue_stage_stall_at_least_one_warp_waiting_yield;
  unsigned long long total_num_cycles_issue_stage_stall_at_least_one_warp_waiting_stall_count;
  unsigned long long total_num_cycles_issue_stage_stall_at_least_one_warp_waiting_l1c;

  unsigned int num_kernel_not_in_binary;
  // MOD. End. Remodeling

};

class shader_core_stats : public shader_core_stats_pod {
 public:
  shader_core_stats(const shader_core_config *config, gpgpu_sim *gpu) {
    m_config = config;
    shader_core_stats_pod *pod = reinterpret_cast<shader_core_stats_pod *>(
        this->shader_core_stats_pod_start);
    memset(reinterpret_cast<void *>(pod), 0, sizeof(shader_core_stats_pod));
    shader_cycles = (unsigned long long *)calloc(config->num_shader(),
                                                 sizeof(unsigned long long));
    m_num_sim_insn = (unsigned *)calloc(config->num_shader(), sizeof(unsigned));

    // MOD.. Begin. Custom Stats
    m_last_kernel_id = 0;
    total_weighted_average_warp_ipc_between_shaders = 0;
    total_weighted_average_shader_occupancy = 0;
    total_weighted_average_shader_warp_ipc_with_kernels = 0;
    total_weighted_average_num_shader_active = 0;
    m_num_sim_winsn_per_shader.resize(m_config->num_shader());
    shader_warp_ipc_per_shader.resize(m_config->num_shader());

    tot_scheduler_cycles = 0;
    tot_scheduler_issues = 0;

    tot_num_expected_wb = 0; // MOD. Custom stats
    tot_num_allocated_wb = 0; // MOD. Custom stats

    tot_fetch_instruction_misalignments = 0; // MOD. Fix misaligned fetched instructions
    tot_fetch_requests = 0; // MOD. Fix misaligned fetched instructions

    numEffectiveIncompleteWarps = 0;

    num_scoreboard_reads_check_collision = 0;// MOD. Scoreboard_reads
    num_scoreboard_reads_collision_due_to_max_uses_per_reg = 0;// MOD. Scoreboard_reads
    num_scheduler_stall_cycle_due_to_war_scoreboard = 0;// MOD. Scoreboard_reads
    num_scheduler_stall_cycle_dependencies_other_reasons_not_war_scoreboard = 0;// MOD. Scoreboard_reads

    // MOD. Begin. IBuffer_ooo stats
    total_ins_issued_per_kernel_per_sid_per_warp = 0;
    total_ins_released_wb_per_kernel_per_sid_per_warp = 0;
    total_ins_released_opc_per_kernel_per_sid_per_warp = 0;
    total_num_flushes_kernel_per_sid_per_warp = 0;
    total_num_barriers = 0;
    total_num_returns = 0;
    total_num_branches = 0;
    total_num_barriers_and_controlflows = 0;
    total_num_times_ibooo_empty = 0;
    total_num_times_ibooo_empty_evaluated = 0;
    total_num_times_ibooo_full = 0;
    total_num_times_fetch_ibooo_tried = 0;
    total_percentage_ibooo_full = 0;
    total_percentage_ibooo_empty = 0;
    total_instructions_inserted_in_ibooo = 0;
    total_war_waw_dependencies = 0;
    total_raw_dependencies = 0;
    total_stop_point_dependencies = 0;
    total_memory_reordering_dependencies = 0;
    // MOD. End. IBuffer_ooo stats

    // MOD. Begin. VPREG stats
    total_vpreg_merges = 0;
    total_vpreg_extra_rf_reads = 0;
    total_rf_reads = 0;
    total_number_of_kernels_limited_by_regs = 0;
    total_number_of_vpreg_decode_rollbacks = 0;
    total_number_of_vpreg_reissues = 0;
    total_number_of_vpreg_not_enough_virtual_at_decode = 0;
    max_vpreg_virtual_regs_used_in_subcore = 0;
    max_vpreg_physical_regs_used_in_subcore = 0;
    max_vpreg_physical_freepool_usage_in_bank = 0;
    max_vpreg_number_of_consumers = 0;
    total_vpreg_predication_dependencies = 0;
    // MOD. End. VPREG stats

    // MOD. Begin. OPC custom stats
    total_number_of_opc_conflicts = 0;
    total_number_of_opc_requests = 0;
    num_times_cu_subcore_custom_stats_evaluated = 0;
    num_times_no_cu_dispatched = 0;
    num_times_no_cu_allocated_and_nothing_to_allocate = 0;
    num_times_no_cu_allocated = 0;
    num_times_no_cu_allocated_due_to_cus_are_full = 0;
    num_times_no_cu_dispatched_due_to_dispatch_reg_full = 0;
    num_times_no_cu_dispatched_due_to_no_ready_operands = 0;
    num_times_no_cu_dispatched_due_to_all_cus_empty = 0;
    num_times_no_cu_dispatched_and_all_cus_full = 0;
    num_times_no_cu_dispatched_and_all_cus_full_and_at_least_one_ready = 0;
    num_times_no_cu_dispatched_and_all_cus_full_and_not_any_ready = 0;
    num_times_no_cu_dispatched_due_to_dispatch_reg_full_is_mem_op_and_ldst_unit_stalled = 0;
    num_times_no_cu_dispatched_and_all_cus_full_and_at_least_one_ready_mem_dispatch_full_and_ldst_unit_stalled = 0;
    total_num_try_ldst_unit_dispatches = 0;
    total_num_ldst_unit_dispatches_failed_due_to_not_empty_dispatch_reg = 0;
    total_cycles_instructions_in_cu = 0; // MOD. CU stats
    // MOD. End. OPC custom stats

    // MOD. Begin. Memory stats
    l1d_accesses_per_sid_per_bank.resize(m_config->num_shader());
    l1d_evals_per_sid_per_bank.resize(m_config->num_shader());
    max_avg_usage_l1d_bank = 0;
    total_shared_mem_evals = 0;
    total_shared_mem_accesses = 0;
    total_num_ldst_unit_instructions = 0;
    total_num_dp_instructions = 0;
    total_num_warp_instructions = 0;
    total_l1d_instructions = 0;
    total_accesses_l1d_instructions = 0;
    total_avg_cycles_to_schedule_accesses = 0;
    total_cycles_instructions_in_ldst_unit_dispatch_reg = 0;
    total_cycles_instructions_in_ldst_unit_arbiter_latch = 0; // MOD. Fixed LDST_Unit model.
    total_shared_instructions = 0;
    total_conflicts_shared_instructions = 0;
    // MOD. End. Memory stats

    // MOD. Begin. Custom powermodel stats
    warp_issues_from_last_power_sample.resize(m_config->num_shader());
    bank_wb_from_last_power_sample.resize(m_config->num_shader());
    collector_unit_allocations_from_last_power_sample.resize(m_config->num_shader());

    unsigned int max_num_j_iters = std::max(std::max(m_config->max_warps_per_shader, m_config->gpgpu_num_reg_banks), (unsigned int)m_config->gpgpu_operand_collector_num_units_gen);
    for(unsigned int i = 0; i < m_config->num_shader(); i++) {
        warp_issues_from_last_power_sample[i].resize(m_config->max_warps_per_shader);
        bank_wb_from_last_power_sample[i].resize(m_config->gpgpu_num_reg_banks);
        collector_unit_allocations_from_last_power_sample[i].resize(m_config->gpgpu_operand_collector_num_units_gen);
        
        for(unsigned int j = 0; j < max_num_j_iters; j++) {
          if(j < m_config->max_warps_per_shader) {
            warp_issues_from_last_power_sample[i][j] = 0;
          }
          if(j < m_config->gpgpu_num_reg_banks) {
            bank_wb_from_last_power_sample[i][j] = 0;
          }
          if(j < m_config->gpgpu_operand_collector_num_units_gen) {
            collector_unit_allocations_from_last_power_sample[i][j] = 0;
          }
        }
        l1d_accesses_per_sid_per_bank[i].resize( m_config->m_L1D_config.l1_banks); // MOD. Memory stats
        l1d_evals_per_sid_per_bank[i].resize( m_config->m_L1D_config.l1_banks); // MOD. Memory stats
    }
    // MOD. End. Custom powermodel stats

    // MOD. Begin. Remodeling
    total_num_register_file_cache_hits = 0;
    total_num_register_file_cache_allocations = 0;
    total_num_regular_regfile_reads = 0;
    total_num_regular_regfile_writes = 0;
    total_num_uniform_regfile_reads = 0;
    total_num_uniform_regfile_writes = 0;
    total_num_predicate_regfile_reads = 0;
    total_num_predicate_regfile_writes = 0;
    total_num_uniform_predicate_regfile_reads = 0;
    total_num_uniform_predicate_regfile_writes = 0;
    total_num_constant_cache_reads = 0;
    all_const_cache_accessed_blocks.clear();
    all_global_memory_accessed_blocks.clear();
    all_virtual_pages_accessed.clear();

    total_num_times_wb_evaluated = 0;
    total_num_times_wb_port_conflict = 0;

    total_num_cycles_issue_stage_evaluated = 0;
    total_num_cycles_issue_stage_issuing = 0;
    total_num_cycles_issue_stage_stall_issue_port_busy = 0;
    total_num_cycles_issue_stage_stall_no_valid_instruction = 0;
    total_num_cycles_issue_stage_stall_no_warps_ready = 0;

    total_num_cycles_issue_stage_stall_at_least_one_warp_with_fu_occupied = 0;
    total_num_cycles_issue_stage_stall_at_least_one_warp_waiting_inst_barrier = 0;
    total_num_cycles_issue_stage_stall_at_least_one_warp_waiting_wait_barrier = 0;
    total_num_cycles_issue_stage_stall_at_least_one_warp_waiting_yield = 0;
    total_num_cycles_issue_stage_stall_at_least_one_warp_waiting_stall_count = 0;
    total_num_cycles_issue_stage_stall_at_least_one_warp_waiting_l1c = 0;

    num_kernel_not_in_binary = 0;
    // MOD. End. Remodeling

    // Mod. End. Custom Stats

    m_num_sim_winsn =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_last_num_sim_winsn =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_last_num_sim_insn =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_pipeline_duty_cycle =
        (float *)calloc(config->num_shader(), sizeof(float));
    m_num_decoded_insn =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_num_FPdecoded_insn =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_num_storequeued_insn =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_num_loadqueued_insn =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_num_tex_inst = 
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_num_INTdecoded_insn =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_num_ialu_acesses =
        (double *)calloc(config->num_shader(), sizeof(double));
    m_num_fp_acesses =
        (double *)calloc(config->num_shader(), sizeof(double));
    m_num_imul_acesses =
        (double *)calloc(config->num_shader(), sizeof(double));
    m_num_imul24_acesses =
        (double *)calloc(config->num_shader(), sizeof(double));
    m_num_imul32_acesses =
        (double *)calloc(config->num_shader(), sizeof(double));
    m_num_fpmul_acesses =
        (double *)calloc(config->num_shader(), sizeof(double));
    m_num_idiv_acesses =
        (double *)calloc(config->num_shader(), sizeof(double));
    m_num_fpdiv_acesses =
        (double *)calloc(config->num_shader(), sizeof(double));
    m_num_dp_acesses = 
        (double*) calloc(config->num_shader(),sizeof(double));
    m_num_dpmul_acesses = 
        (double*) calloc(config->num_shader(),sizeof(double));
    m_num_dpdiv_acesses = 
        (double*) calloc(config->num_shader(),sizeof(double));
    m_num_sp_acesses =
        (double *)calloc(config->num_shader(), sizeof(double));
    m_num_sfu_acesses =
        (double *)calloc(config->num_shader(), sizeof(double));
    m_num_tensor_core_acesses = 
        (double *)calloc(config->num_shader(), sizeof(double));
    m_num_const_acesses =
        (double *)calloc(config->num_shader(), sizeof(double));
    m_num_tex_acesses =
        (double *)calloc(config->num_shader(), sizeof(double));
    m_num_sqrt_acesses = 
        (double*) calloc(config->num_shader(),sizeof(double));
    m_num_log_acesses = 
        (double*) calloc(config->num_shader(),sizeof(double));
    m_num_sin_acesses = 
        (double*) calloc(config->num_shader(),sizeof(double));
    m_num_exp_acesses = 
        (double*) calloc(config->num_shader(),sizeof(double));
    m_num_mem_acesses =
        (double *)calloc(config->num_shader(), sizeof(double));
    m_num_sp_committed =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_num_tlb_hits = 
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_num_tlb_accesses =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_active_sp_lanes =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_active_sfu_lanes =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_active_tensor_core_lanes =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_active_fu_lanes =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_active_exu_threads =
        (double *)calloc(config->num_shader(), sizeof(double));
    m_active_exu_warps =
        (double *)calloc(config->num_shader(), sizeof(double));
    m_active_fu_mem_lanes =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_num_sfu_committed =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_num_tensor_core_committed =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_num_mem_committed =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_read_regfile_acesses =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_write_regfile_acesses =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_non_rf_operands =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_n_diverge = 
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    shader_cycle_distro =
        (unsigned *)calloc(config->warp_size + 3, sizeof(unsigned));
    last_shader_cycle_distro =
        (unsigned *)calloc(config->warp_size + 3, sizeof(unsigned));
    single_issue_nums =
        (unsigned *)calloc(config->gpgpu_num_sched_per_core, sizeof(unsigned));
    dual_issue_nums =
        (unsigned *)calloc(config->gpgpu_num_sched_per_core, sizeof(unsigned));

    ctas_completed = 0;
    n_simt_to_mem = (long *)calloc(config->num_shader(), sizeof(long));
    n_mem_to_simt = (long *)calloc(config->num_shader(), sizeof(long));

    m_outgoing_traffic_stats = new traffic_breakdown("coretomem");
    m_incoming_traffic_stats = new traffic_breakdown("memtocore");

    gpgpu_n_shmem_bank_access =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));

    m_shader_dynamic_warp_issue_distro.resize(config->num_shader());
    m_shader_warp_slot_issue_distro.resize(config->num_shader());
    m_gpu = gpu;
  }

  ~shader_core_stats() {
    delete m_outgoing_traffic_stats;
    delete m_incoming_traffic_stats;
    free(m_num_sim_insn);
    free(m_num_sim_winsn);
    free(m_pipeline_duty_cycle);
    free(m_num_decoded_insn);
    free(m_num_FPdecoded_insn);
    free(m_num_INTdecoded_insn);
    free(m_num_storequeued_insn);
    free(m_num_loadqueued_insn);
    free(m_num_ialu_acesses);
    free(m_num_fp_acesses);
    free(m_num_imul_acesses);
    free(m_num_tex_inst);
    free(m_num_fpmul_acesses);
    free(m_num_idiv_acesses);
    free(m_num_fpdiv_acesses);
    free(m_num_sp_acesses);
    free(m_num_sfu_acesses);
    free(m_num_tensor_core_acesses);
    free(m_num_tex_acesses);
    free(m_num_const_acesses);
    free(m_num_dp_acesses);
    free(m_num_dpmul_acesses);
    free(m_num_dpdiv_acesses);
    free(m_num_sqrt_acesses);
    free(m_num_log_acesses);
    free(m_num_sin_acesses);
    free(m_num_exp_acesses);
    free(m_num_mem_acesses);
    free(m_num_sp_committed);
    free(m_num_tlb_hits);
    free(m_num_tlb_accesses);
    free(m_num_sfu_committed);
    free(m_num_tensor_core_committed);
    free(m_num_mem_committed);
    free(m_read_regfile_acesses);
    free(m_write_regfile_acesses);
    free(m_non_rf_operands);
    free(m_num_imul24_acesses);
    free(m_num_imul32_acesses);
    free(m_active_sp_lanes);
    free(m_active_sfu_lanes);
    free(m_active_tensor_core_lanes);
    free(m_active_fu_lanes);
    free(m_active_exu_threads);
    free(m_active_exu_warps);
    free(m_active_fu_mem_lanes);
    free(m_n_diverge);
    free(shader_cycle_distro);
    free(last_shader_cycle_distro);
    free(n_simt_to_mem);
    free(n_mem_to_simt);
    free(shader_cycles);
    free(m_last_num_sim_insn);
    free(single_issue_nums);
    free(dual_issue_nums);

    free(gpgpu_n_shmem_bank_access);
    free(m_last_num_sim_winsn);
  }

  // MOD. Begin. Custom Stats
  void allocate_for_a_new_kernel() {
    m_last_kernel_id +=1;
    m_current_kernel_pos = m_last_kernel_id - 1;

    gpu_cycles_per_kernel.resize(m_last_kernel_id);
    weighted_average_shader_warp_ipc_per_kernel.resize(m_last_kernel_id);
    weighted_average_shader_occupancy_per_kernel.resize(m_last_kernel_id);
    average_num_shader_active_per_kernel.resize(m_last_kernel_id);
    number_of_warps_per_kernel.resize(m_last_kernel_id);

    m_num_sim_winsn_per_shader_per_kernel.resize(m_last_kernel_id);
    shader_active_warps_per_kernel.resize(m_last_kernel_id);
    shader_maximum_theoretical_warps_per_kernel.resize(m_last_kernel_id);
    shader_cycles_per_kernel.resize(m_last_kernel_id);
    shader_warp_ipc_per_kernel.resize(m_last_kernel_id);
    shader_occupancy_per_kernel.resize(m_last_kernel_id);

    m_num_sim_winsn_per_shader_per_kernel[m_current_kernel_pos].resize(m_config->num_shader());
    shader_active_warps_per_kernel[m_current_kernel_pos].resize(m_config->num_shader());
    shader_maximum_theoretical_warps_per_kernel[m_current_kernel_pos].resize(m_config->num_shader());
    shader_cycles_per_kernel[m_current_kernel_pos].resize(m_config->num_shader());
    shader_warp_ipc_per_kernel[m_current_kernel_pos].resize(m_config->num_shader());
    shader_occupancy_per_kernel[m_current_kernel_pos].resize(m_config->num_shader());
    // MOD. IBuffer_ooo. Begin stats
    ins_issued_per_kernel_per_sid_per_warp.resize(m_last_kernel_id);
    ins_released_opc_per_kernel_per_sid_per_warp.resize(m_last_kernel_id);
    ins_released_wb_per_kernel_per_sid_per_warp.resize(m_last_kernel_id);
    num_flushes_kernel_per_sid_per_warp.resize(m_last_kernel_id);
    num_times_ibooo_empty.resize(m_last_kernel_id);
    num_times_ibooo_empty_evaluated.resize(m_last_kernel_id);
    num_times_ibooo_full.resize(m_last_kernel_id);
    num_times_fetch_ibooo_tried.resize(m_last_kernel_id);
    ins_issued_per_kernel_per_sid_per_warp[m_current_kernel_pos].resize(m_config->num_shader());
    ins_released_opc_per_kernel_per_sid_per_warp[m_current_kernel_pos].resize(m_config->num_shader());
    ins_released_wb_per_kernel_per_sid_per_warp[m_current_kernel_pos].resize(m_config->num_shader());
    num_flushes_kernel_per_sid_per_warp[m_current_kernel_pos].resize(m_config->num_shader());
    num_times_ibooo_empty[m_current_kernel_pos].resize(m_config->num_shader());
    num_times_ibooo_empty_evaluated[m_current_kernel_pos].resize(m_config->num_shader());
    num_times_ibooo_full[m_current_kernel_pos].resize(m_config->num_shader());
    num_times_fetch_ibooo_tried[m_current_kernel_pos].resize(m_config->num_shader());
    for(unsigned int i = 0; i < m_config->num_shader(); i++)
    {
      ins_issued_per_kernel_per_sid_per_warp[m_current_kernel_pos][i].resize(m_config->max_warps_per_shader);
      ins_released_opc_per_kernel_per_sid_per_warp[m_current_kernel_pos][i].resize(m_config->max_warps_per_shader);
      ins_released_wb_per_kernel_per_sid_per_warp[m_current_kernel_pos][i].resize(m_config->max_warps_per_shader);
      num_flushes_kernel_per_sid_per_warp[m_current_kernel_pos][i].resize(m_config->max_warps_per_shader);
      num_times_ibooo_empty[m_current_kernel_pos][i].resize(m_config->max_warps_per_shader);
      num_times_ibooo_empty_evaluated[m_current_kernel_pos][i].resize(m_config->max_warps_per_shader);
      num_times_ibooo_full[m_current_kernel_pos][i].resize(m_config->max_warps_per_shader);
      num_times_fetch_ibooo_tried[m_current_kernel_pos][i].resize(m_config->max_warps_per_shader);
      for(unsigned int j = 0; j < m_config->max_warps_per_shader; j++)
      {
        ins_issued_per_kernel_per_sid_per_warp[m_current_kernel_pos][i][j] = 0;
        ins_released_opc_per_kernel_per_sid_per_warp[m_current_kernel_pos][i][j] = 0;
        ins_released_wb_per_kernel_per_sid_per_warp[m_current_kernel_pos][i][j] = 0;
        num_flushes_kernel_per_sid_per_warp[m_current_kernel_pos][i][j] = 0;
        num_times_ibooo_empty[m_current_kernel_pos][i][j] = 0;
        num_times_ibooo_empty_evaluated[m_current_kernel_pos][i][j] = 0;
        num_times_ibooo_full[m_current_kernel_pos][i][j] = 0;
        num_times_fetch_ibooo_tried[m_current_kernel_pos][i][j] = 0;
      }
    }
    // MOD. IBuffer_ooo. Begin stats
  }

  void print_custom_shader_stats(FILE *fout) const; // MOD.
  void print_remodeling_stats(FILE *fout); // MOD. Remodeling
  void print_coalescing_stats(FILE *fout); 
  void compute_ibuffer_ooo_stats(); // MOD. IBuffer_ooo
  void print_ibuffer_ooo_stats(FILE *fout) const; // MOD. IBuffer_ooo
  void print_vpreg_stats(FILE *fout) const; // MOD. VPREG
  void print_single_custom_shader_stat_long(FILE *fout, std::string stat_name, std::vector<std::vector<unsigned long long>> vector_stat) const;
  void print_single_custom_shader_stat_double(FILE *fout, std::string stat_name, std::vector<std::vector<double>> vector_stat) const;
  void compute_derived_custom_stats();

  // MOD. End. Custom Stats

  void new_grid() {}

  void event_warp_issued(unsigned s_id, unsigned warp_id, unsigned num_issued,
                         unsigned dynamic_warp_id);

  void visualizer_print(gzFile visualizer_file);

  void print(FILE *fout);

  const std::vector<std::vector<unsigned>> &get_dynamic_warp_issue() const {
    return m_shader_dynamic_warp_issue_distro;
  }

  const std::vector<std::vector<unsigned>> &get_warp_slot_issue() const {
    return m_shader_warp_slot_issue_distro;
  }

  traffic_breakdown *m_outgoing_traffic_stats;  // core to memory partitions
  traffic_breakdown *m_incoming_traffic_stats;  // memory partition to core

 private:
  const shader_core_config *my_custom_config; // MOD. Declared attribute to prevent crashing due to segFault because of adding to many stats doesn't like it
  const shader_core_config *m_config;


  // Counts the instructions issued for each dynamic warp.
  std::vector<std::vector<unsigned>> m_shader_dynamic_warp_issue_distro;
  std::vector<unsigned> m_last_shader_dynamic_warp_issue_distro;
  std::vector<std::vector<unsigned>> m_shader_warp_slot_issue_distro;
  std::vector<unsigned> m_last_shader_warp_slot_issue_distro;

  gpgpu_sim *m_gpu;

  friend class power_stat_t;
  friend class shader_core_ctx;
  friend class ldst_unit;
  friend class simt_core_cluster;
  friend class scheduler_unit;
  friend class TwoLevelScheduler;
  friend class LooseRoundRobbinScheduler;
};

class memory_config;
class shader_core_mem_fetch_allocator : public mem_fetch_allocator {
 public:
  shader_core_mem_fetch_allocator(unsigned core_id, unsigned cluster_id,
                                  const memory_config *config) {
    m_core_id = core_id;
    m_cluster_id = cluster_id;
    m_memory_config = config;
  }
  mem_fetch *alloc(new_addr_type addr, mem_access_type type, unsigned size,
                   bool wr, unsigned long long cycle) const;
  mem_fetch *alloc(new_addr_type addr, mem_access_type type,
                   const active_mask_t &active_mask,
                   const mem_access_byte_mask_t &byte_mask,
                   const mem_access_sector_mask_t &sector_mask, unsigned size,
                   bool wr, unsigned long long cycle, unsigned wid,
                   unsigned sid, unsigned tpc, mem_fetch *original_mf) const;
  mem_fetch *alloc(const warp_inst_t &inst, const mem_access_t &access,
                   unsigned long long cycle) const {
    warp_inst_t inst_copy = inst;
    mem_fetch *mf = new mem_fetch(
        access, &inst_copy,
        access.is_write() ? WRITE_PACKET_SIZE : READ_PACKET_SIZE,
        inst.warp_id(), m_core_id, m_cluster_id, m_memory_config, cycle);
    return mf;
  }

 private:
  unsigned m_core_id;
  unsigned m_cluster_id;
  const memory_config *m_memory_config;
};

class shader_core_ctx : public core_t, public shader_core_ctx_wrapper {
 public:
  // creator:
  shader_core_ctx(class gpgpu_sim *gpu, class simt_core_cluster *cluster,
                  unsigned shader_id, unsigned tpc_id,
                  const shader_core_config *config,
                  const memory_config *mem_config, shader_core_stats *stats);
  void init() override {};
  bool  warp_waiting_grid_barrier(unsigned warp_id) override { return false; }
  void num_cycles_to_stall_SM(unsigned int num_cycles) override {};
  Scoreboard_reads* get_Scoreboard_reads(); // MOD. Fix WAR at baseline
  shader_core_stats* get_stats() {return m_stats;} // MOD. VPREG
  ldst_unit* get_ldst_unit() {return m_ldst_unit;} // MOD. OPC stats

  bool is_subcore_active(unsigned sub_core_id); // MOD.

  void create_gpu_per_sm_stats(Element_stats &all_stats) override {}
  void reset_cycless_access_history() override {}
  void gather_gpu_per_sm_stats(Element_stats &all_stats, coalescingStatsAcrossSms& coal_stats_l1d, coalescingStatsAcrossSms& coal_stats_const, coalescingStatsAcrossSms& coal_stats_sharedmem) override {}
  void gather_gpu_per_sm_single_stat(Element_stats &all_stats, std::string stat_name) override {}
  void increment_sm_stat_by_integer(std::string stat_name, int val_to_increment) override {}

  kernel_info_t *get_kernel_info() override { return this->core_t::get_kernel_info(); }
  gpgpu_sim *get_gpu() override { return this->core_t::get_gpu(); }
  bool ptx_thread_done(unsigned hw_thread_id) const override {
    return this->core_t::ptx_thread_done(hw_thread_id);
  }

  // used by simt_core_cluster:
  // modifiers
  void cycle();
  void reinit(unsigned start_thread, unsigned end_thread,
              bool reset_not_completed);
  void issue_block2core(class kernel_info_t &kernel);

  void cache_flush();
  void cache_invalidate();
  void accept_fetch_response(mem_fetch *mf);
  void accept_ldst_unit_response(class mem_fetch *mf);
  void broadcast_barrier_reduction(unsigned cta_id, unsigned bar_id,
                                   warp_set_t warps);
  void set_kernel(kernel_info_t *k) {
    assert(k);
    m_kernel = k;
    //        k->inc_running();
    printf("GPGPU-Sim uArch: Shader %d bind to kernel %u \'%s\'\n", m_sid,
           m_kernel->get_uid(), m_kernel->name().c_str());
  }
  PowerscalingCoefficients *scaling_coeffs;
  // accessors
  bool fetch_unit_response_buffer_full() const;
  bool ldst_unit_response_buffer_full() const;
  unsigned get_not_completed() const { return m_not_completed; }
  unsigned get_n_active_cta() const { return m_n_active_cta; }
  unsigned isactive() const {
    if (m_n_active_cta > 0)
      return 1;
    else
      return 0;
  }
  kernel_info_t *get_kernel() { return m_kernel; }
  unsigned int get_sid() const override { return m_sid; }

  // used by functional simulation:
  // modifiers
  virtual void warp_exit(unsigned warp_id);

  // accessors
  virtual bool warp_waiting_at_barrier(unsigned warp_id) const;
  void get_pdom_stack_top_info(unsigned tid, unsigned *pc, unsigned *rpc) const;
  float get_current_occupancy(unsigned long long &active,
                              unsigned long long &total) const;

  // used by pipeline timing model components:
  // modifiers
  void mem_instruction_stats(const warp_inst_t &inst);
  void decrement_atomic_count(unsigned wid, unsigned n);
  void inc_store_req(unsigned warp_id) { m_warp[warp_id]->inc_store_req(); }
  void dec_inst_in_pipeline(unsigned warp_id) {
    m_warp[warp_id]->dec_inst_in_pipeline();
  }  // also used in writeback()
  void store_ack(class mem_fetch *mf);
  bool warp_waiting_at_mem_barrier(unsigned warp_id);
  void set_max_cta(const kernel_info_t &kernel);
  void warp_inst_complete(const warp_inst_t &inst);
  void customStatsWarpActiveLanes(const warp_inst_t &inst); // MOD. Custom Stats

  // accessors
  std::list<unsigned> get_regs_written(const inst_t &fvt) const;
  const shader_core_config *get_config() const override { return m_config; }
  void print_cache_stats(FILE *fp, unsigned &dl1_accesses,
                         unsigned &dl1_misses);

  void get_cache_stats(cache_stats &cs);
  void get_L1I_sub_stats(struct cache_sub_stats &css) const;
  void get_L0I_sub_stats(struct cache_sub_stats &css) const; // MOD. L0I
  void get_L1D_sub_stats(struct cache_sub_stats &css) const;
  void get_L1C_sub_stats(struct cache_sub_stats &css) const;
  void get_L1T_sub_stats(struct cache_sub_stats &css) const;

  void get_icnt_power_stats(long &n_simt_to_mem, long &n_mem_to_simt) const;

  // debug:
  void display_simt_state(FILE *fout, int mask) const;
  void display_pipeline(FILE *fout, int print_mem, int mask3bit) const;

  void incload_stat() { m_stats->m_num_loadqueued_insn[m_sid]++; }
  void incstore_stat() { m_stats->m_num_storequeued_insn[m_sid]++; }
  void incialu_stat(unsigned active_count,double latency) {
    if(m_config->gpgpu_clock_gated_lanes==false){
      m_stats->m_num_ialu_acesses[m_sid]=m_stats->m_num_ialu_acesses[m_sid]+(double)active_count*latency
        + inactive_lanes_accesses_nonsfu(active_count, latency);
    }else {
      m_stats->m_num_ialu_acesses[m_sid]=m_stats->m_num_ialu_acesses[m_sid]+(double)active_count*latency;
    }
    m_stats->m_active_exu_threads[m_sid]+=active_count;
    m_stats->m_active_exu_warps[m_sid]++;
  }
  void incimul_stat(unsigned active_count,double latency) {
    if(m_config->gpgpu_clock_gated_lanes==false){
      m_stats->m_num_imul_acesses[m_sid]=m_stats->m_num_imul_acesses[m_sid]+(double)active_count*latency
        + inactive_lanes_accesses_nonsfu(active_count, latency);
    }else {
      m_stats->m_num_imul_acesses[m_sid]=m_stats->m_num_imul_acesses[m_sid]+(double)active_count*latency;
    }
    m_stats->m_active_exu_threads[m_sid]+=active_count;
    m_stats->m_active_exu_warps[m_sid]++;
  }
  void incimul24_stat(unsigned active_count,double latency) {
  if(m_config->gpgpu_clock_gated_lanes==false){
    m_stats->m_num_imul24_acesses[m_sid]=m_stats->m_num_imul24_acesses[m_sid]+(double)active_count*latency
        + inactive_lanes_accesses_nonsfu(active_count, latency);
    }else {
      m_stats->m_num_imul24_acesses[m_sid]=m_stats->m_num_imul24_acesses[m_sid]+(double)active_count*latency;
    }
    m_stats->m_active_exu_threads[m_sid]+=active_count;
    m_stats->m_active_exu_warps[m_sid]++;    
   }
   void incimul32_stat(unsigned active_count,double latency) {
    if(m_config->gpgpu_clock_gated_lanes==false){
      m_stats->m_num_imul32_acesses[m_sid]=m_stats->m_num_imul32_acesses[m_sid]+(double)active_count*latency
         + inactive_lanes_accesses_sfu(active_count, latency);          
    }else{
      m_stats->m_num_imul32_acesses[m_sid]=m_stats->m_num_imul32_acesses[m_sid]+(double)active_count*latency;
    }
    m_stats->m_active_exu_threads[m_sid]+=active_count;
    m_stats->m_active_exu_warps[m_sid]++;
  }
   void incidiv_stat(unsigned active_count,double latency) {
    if(m_config->gpgpu_clock_gated_lanes==false){
      m_stats->m_num_idiv_acesses[m_sid]=m_stats->m_num_idiv_acesses[m_sid]+(double)active_count*latency
         + inactive_lanes_accesses_sfu(active_count, latency); 
    }else {
      m_stats->m_num_idiv_acesses[m_sid]=m_stats->m_num_idiv_acesses[m_sid]+(double)active_count*latency;
    }
    m_stats->m_active_exu_threads[m_sid]+=active_count;
    m_stats->m_active_exu_warps[m_sid]++;    
  }
   void incfpalu_stat(unsigned active_count,double latency) {
    if(m_config->gpgpu_clock_gated_lanes==false){
      m_stats->m_num_fp_acesses[m_sid]=m_stats->m_num_fp_acesses[m_sid]+(double)active_count*latency
         + inactive_lanes_accesses_nonsfu(active_count, latency);
    }else {
    m_stats->m_num_fp_acesses[m_sid]=m_stats->m_num_fp_acesses[m_sid]+(double)active_count*latency;
    }
    m_stats->m_active_exu_threads[m_sid]+=active_count;
    m_stats->m_active_exu_warps[m_sid]++;     
  }
   void incfpmul_stat(unsigned active_count,double latency) {
              // printf("FP MUL stat increament\n");
    if(m_config->gpgpu_clock_gated_lanes==false){
      m_stats->m_num_fpmul_acesses[m_sid]=m_stats->m_num_fpmul_acesses[m_sid]+(double)active_count*latency
        + inactive_lanes_accesses_nonsfu(active_count, latency);
    }else {
    m_stats->m_num_fpmul_acesses[m_sid]=m_stats->m_num_fpmul_acesses[m_sid]+(double)active_count*latency;
    }
    m_stats->m_active_exu_threads[m_sid]+=active_count;
    m_stats->m_active_exu_warps[m_sid]++;
   }
   void incfpdiv_stat(unsigned active_count,double latency) {
    if(m_config->gpgpu_clock_gated_lanes==false){
      m_stats->m_num_fpdiv_acesses[m_sid]=m_stats->m_num_fpdiv_acesses[m_sid]+(double)active_count*latency
        + inactive_lanes_accesses_sfu(active_count, latency); 
    }else {
      m_stats->m_num_fpdiv_acesses[m_sid]=m_stats->m_num_fpdiv_acesses[m_sid]+(double)active_count*latency;
    }
    m_stats->m_active_exu_threads[m_sid]+=active_count;
    m_stats->m_active_exu_warps[m_sid]++;
   }
   void incdpalu_stat(unsigned active_count,double latency) {
    if(m_config->gpgpu_clock_gated_lanes==false){
      m_stats->m_num_dp_acesses[m_sid]=m_stats->m_num_dp_acesses[m_sid]+(double)active_count*latency
         + inactive_lanes_accesses_nonsfu(active_count, latency);
    }else {
    m_stats->m_num_dp_acesses[m_sid]=m_stats->m_num_dp_acesses[m_sid]+(double)active_count*latency;
    }
    m_stats->m_active_exu_threads[m_sid]+=active_count;
    m_stats->m_active_exu_warps[m_sid]++; 
   }
   void incdpmul_stat(unsigned active_count,double latency) {
              // printf("FP MUL stat increament\n");
    if(m_config->gpgpu_clock_gated_lanes==false){
      m_stats->m_num_dpmul_acesses[m_sid]=m_stats->m_num_dpmul_acesses[m_sid]+(double)active_count*latency
        + inactive_lanes_accesses_nonsfu(active_count, latency);
    }else {
    m_stats->m_num_dpmul_acesses[m_sid]=m_stats->m_num_dpmul_acesses[m_sid]+(double)active_count*latency;
    }
    m_stats->m_active_exu_threads[m_sid]+=active_count;
    m_stats->m_active_exu_warps[m_sid]++;
   }
   void incdpdiv_stat(unsigned active_count,double latency) {
    if(m_config->gpgpu_clock_gated_lanes==false){
      m_stats->m_num_dpdiv_acesses[m_sid]=m_stats->m_num_dpdiv_acesses[m_sid]+(double)active_count*latency
        + inactive_lanes_accesses_sfu(active_count, latency); 
    }else {
      m_stats->m_num_dpdiv_acesses[m_sid]=m_stats->m_num_dpdiv_acesses[m_sid]+(double)active_count*latency;
    }
    m_stats->m_active_exu_threads[m_sid]+=active_count;
    m_stats->m_active_exu_warps[m_sid]++;
   }

   void incsqrt_stat(unsigned active_count,double latency) {
    if(m_config->gpgpu_clock_gated_lanes==false){
      m_stats->m_num_sqrt_acesses[m_sid]=m_stats->m_num_sqrt_acesses[m_sid]+(double)active_count*latency
        + inactive_lanes_accesses_sfu(active_count, latency); 
    }else{
      m_stats->m_num_sqrt_acesses[m_sid]=m_stats->m_num_sqrt_acesses[m_sid]+(double)active_count*latency;
    }
    m_stats->m_active_exu_threads[m_sid]+=active_count;
    m_stats->m_active_exu_warps[m_sid]++;
   }

   void inclog_stat(unsigned active_count,double latency) {
    if(m_config->gpgpu_clock_gated_lanes==false){
      m_stats->m_num_log_acesses[m_sid]=m_stats->m_num_log_acesses[m_sid]+(double)active_count*latency
        + inactive_lanes_accesses_sfu(active_count, latency); 
    }else{
      m_stats->m_num_log_acesses[m_sid]=m_stats->m_num_log_acesses[m_sid]+(double)active_count*latency;
    }
    m_stats->m_active_exu_threads[m_sid]+=active_count;
    m_stats->m_active_exu_warps[m_sid]++;
   }

   void incexp_stat(unsigned active_count,double latency) {
    if(m_config->gpgpu_clock_gated_lanes==false){
      m_stats->m_num_exp_acesses[m_sid]=m_stats->m_num_exp_acesses[m_sid]+(double)active_count*latency
        + inactive_lanes_accesses_sfu(active_count, latency); 
    }else{
      m_stats->m_num_exp_acesses[m_sid]=m_stats->m_num_exp_acesses[m_sid]+(double)active_count*latency;
    }
    m_stats->m_active_exu_threads[m_sid]+=active_count;
    m_stats->m_active_exu_warps[m_sid]++;
  }

   void incsin_stat(unsigned active_count,double latency) {
    if(m_config->gpgpu_clock_gated_lanes==false){
      m_stats->m_num_sin_acesses[m_sid]=m_stats->m_num_sin_acesses[m_sid]+(double)active_count*latency
        + inactive_lanes_accesses_sfu(active_count, latency); 
    }else{
      m_stats->m_num_sin_acesses[m_sid]=m_stats->m_num_sin_acesses[m_sid]+(double)active_count*latency;
    }
    m_stats->m_active_exu_threads[m_sid]+=active_count;
    m_stats->m_active_exu_warps[m_sid]++;
  }


   void inctensor_stat(unsigned active_count,double latency) {
    if(m_config->gpgpu_clock_gated_lanes==false){
      m_stats->m_num_tensor_core_acesses[m_sid]=m_stats->m_num_tensor_core_acesses[m_sid]+(double)active_count*latency
        + inactive_lanes_accesses_sfu(active_count, latency); 
    }else{
      m_stats->m_num_tensor_core_acesses[m_sid]=m_stats->m_num_tensor_core_acesses[m_sid]+(double)active_count*latency;
    }
    m_stats->m_active_exu_threads[m_sid]+=active_count;
    m_stats->m_active_exu_warps[m_sid]++;
  }

  void inctex_stat(unsigned active_count,double latency) {
    if(m_config->gpgpu_clock_gated_lanes==false){
      m_stats->m_num_tex_acesses[m_sid]=m_stats->m_num_tex_acesses[m_sid]+(double)active_count*latency
        + inactive_lanes_accesses_sfu(active_count, latency); 
    }else{
      m_stats->m_num_tex_acesses[m_sid]=m_stats->m_num_tex_acesses[m_sid]+(double)active_count*latency;
    }
    m_stats->m_active_exu_threads[m_sid]+=active_count;
    m_stats->m_active_exu_warps[m_sid]++;
  }

  void inc_const_accesses(unsigned active_count) {
    m_stats->m_num_const_acesses[m_sid]=m_stats->m_num_const_acesses[m_sid]+active_count;
  }

  void incsfu_stat(unsigned active_count, double latency) {
    m_stats->m_num_sfu_acesses[m_sid] =
        m_stats->m_num_sfu_acesses[m_sid] + (double)active_count*latency;
  }
  void incsp_stat(unsigned active_count, double latency) {
    m_stats->m_num_sp_acesses[m_sid] =
        m_stats->m_num_sp_acesses[m_sid] + (double)active_count*latency;
  }
  void incmem_stat(unsigned active_count, double latency) {
    if (m_config->gpgpu_clock_gated_lanes == false) {
      m_stats->m_num_mem_acesses[m_sid] =
          m_stats->m_num_mem_acesses[m_sid] + (double)active_count*latency +
          inactive_lanes_accesses_nonsfu(active_count, latency);
    } else {
      m_stats->m_num_mem_acesses[m_sid] =
          m_stats->m_num_mem_acesses[m_sid] + (double)active_count*latency;
    }
  }
  void incexecstat(warp_inst_t *&inst);

  void incregfile_reads(unsigned active_count) {
    m_stats->m_read_regfile_acesses[m_sid] =
        m_stats->m_read_regfile_acesses[m_sid] + active_count;
  }
  void incregfile_writes(unsigned active_count) {
    m_stats->m_write_regfile_acesses[m_sid] =
        m_stats->m_write_regfile_acesses[m_sid] + active_count;
  }
  void incnon_rf_operands(unsigned active_count) {
    m_stats->m_non_rf_operands[m_sid] =
        m_stats->m_non_rf_operands[m_sid] + active_count;
  }

  void incspactivelanes_stat(unsigned active_count) {
    m_stats->m_active_sp_lanes[m_sid] =
        m_stats->m_active_sp_lanes[m_sid] + active_count;
  }
  void incsfuactivelanes_stat(unsigned active_count) {
    m_stats->m_active_sfu_lanes[m_sid] =
        m_stats->m_active_sfu_lanes[m_sid] + active_count;
  }
  void incfuactivelanes_stat(unsigned active_count) override {
    m_stats->m_active_fu_lanes[m_sid] =
        m_stats->m_active_fu_lanes[m_sid] + active_count;
  }
  void incfumemactivelanes_stat(unsigned active_count) {
    m_stats->m_active_fu_mem_lanes[m_sid] =
        m_stats->m_active_fu_mem_lanes[m_sid] + active_count;
  }

  void inc_simt_to_mem(unsigned n_flits) {
    m_stats->n_simt_to_mem[m_sid] += n_flits;
  }
  bool check_if_non_released_reduction_barrier(warp_inst_t &inst);

 protected:
  unsigned inactive_lanes_accesses_sfu(unsigned active_count, double latency) {
    return (((32 - active_count) >> 1) * latency) +
           (((32 - active_count) >> 3) * latency) +
           (((32 - active_count) >> 3) * latency);
  }
  unsigned inactive_lanes_accesses_nonsfu(unsigned active_count,
                                          double latency) {
    return (((32 - active_count) >> 1) * latency);
  }

  int test_res_bus(int latency);
  address_type next_pc(int tid) const;
  void fetch();
  void register_cta_thread_exit(unsigned cta_num, kernel_info_t *kernel);

  void decode();


  unsigned long long get_current_gpu_cycle() override;

  // Not implemented in the old model, just to complete interface for the new model
  address_type from_local_pc_to_global_pc_address(address_type local_pc, unsigned int unique_function_id) override {
    return local_pc;
  }
  // Not implemented in the old model, just to complete interface for the new model
  address_type from_global_pc_address_to_local_pc(address_type global_pc, unsigned int unique_function_id) override {
    return global_pc;
  }

  void issue();
  friend class scheduler_unit;  // this is needed to use private issue warp.
  friend class TwoLevelScheduler;
  friend class LooseRoundRobbinScheduler;
  virtual void issue_warp(register_set &warp, const warp_inst_t *pI,
                          const active_mask_t &active_mask, unsigned warp_id,
                          unsigned sch_id);

  void create_front_pipeline();
  void create_schedulers();
  void create_exec_pipeline();

  // pure virtual methods implemented based on the current execution mode
  // (execution-driven vs trace-driven)
  virtual void init_warps(unsigned cta_id, unsigned start_thread,
                          unsigned end_thread, unsigned ctaid, int cta_size,
                          kernel_info_t &kernel);
  virtual void checkExecutionStatusAndUpdate(warp_inst_t &inst, unsigned t,
                                             unsigned tid) = 0;
  virtual void func_exec_inst(warp_inst_t &inst) = 0;

  virtual unsigned sim_init_thread(kernel_info_t &kernel,
                                   ptx_thread_info **thread_info, int sid,
                                   unsigned tid, unsigned threads_left,
                                   unsigned num_threads, core_t *core,
                                   unsigned hw_cta_id, unsigned hw_warp_id,
                                   gpgpu_t *gpu) = 0;

  virtual void create_shd_warp() = 0;

  virtual warp_inst_t *get_next_inst(unsigned warp_id, // MOD. VPREG
                                           address_type pc) = 0;
  virtual void decrement_trace_pc(unsigned warp_id) = 0; // MOD. VPREG                                       
  virtual void get_pdom_stack_top_info(unsigned warp_id, const warp_inst_t *pI,
                                       unsigned *pc, unsigned *rpc) = 0;
  virtual const active_mask_t &get_active_mask(unsigned warp_id,
                                               const warp_inst_t *pI) = 0;

  // Returns numbers of addresses in translated_addrs
  unsigned translate_local_memaddr(address_type localaddr, unsigned tid,
                                   unsigned num_shader, unsigned datasize,
                                   new_addr_type *translated_addrs);

  void read_operands();

  void execute();

  void writeback();

  // used in display_pipeline():
  void dump_warp_state(FILE *fout) const;
  void print_stage(unsigned int stage, FILE *fout) const;

  unsigned long long m_last_inst_gpu_sim_cycle;
  unsigned long long m_last_inst_gpu_tot_sim_cycle;

  // general information
  unsigned m_sid;  // shader id
  unsigned m_tpc;  // texture processor cluster id (aka, node id when using
                   // interconnect concentration)
  const shader_core_config *m_config;
  const memory_config *m_memory_config;
  class simt_core_cluster *m_cluster;

  // statistics
  shader_core_stats *m_stats;

  // CTA scheduling / hardware thread allocation
  unsigned m_n_active_cta;  // number of Cooperative Thread Arrays (blocks)
                            // currently running on this shader.
  unsigned m_cta_status[MAX_CTA_PER_SHADER];  // CTAs status
  unsigned m_not_completed;  // number of threads to be completed (==0 when all
                             // thread on this core completed)
  std::bitset<MAX_THREAD_PER_SM> m_active_threads;

  // thread contexts
  thread_ctx_t *m_threadState;

  // interconnect interface
  mem_fetch_interface *m_icnt_L0I; // MOD. Added L0I
  mem_fetch_interface *m_icnt;
  shader_core_mem_fetch_allocator *m_mem_fetch_allocator;

  // fetch
  read_only_cache *m_L1I;  // instruction cache
  std::vector<read_only_cache*> m_L0I;  // MOD. Added L0I
  int m_last_warp_fetched;

  // decode/dispatch
  std::vector<shd_warp_t *> m_warp;  // per warp information array
  barrier_set_t m_barriers;
  ifetch_buffer_t m_inst_fetch_buffer;
  std::vector<ifetch_buffer_t> m_improved_fetch_decode_inst_fetch_buffer; // MOD. Improving fetch and decode
  std::vector<int> m_improved_fetch_decode_last_warp_fetched; // MOD. Improving fetch and decode
  int m_subcore_req_fetch_L1I_priority; // MOD. Added L0I
  std::vector<register_set> m_pipeline_reg;
  Scoreboard *m_scoreboard;
  Scoreboard_reads *m_scoreboard_reads; // MOD. Fix WAR at baseline.
  opndcoll_rfu_t m_operand_collector;
  int m_active_warps;
  std::vector<register_set *> m_specilized_dispatch_reg;

  // schedule
  std::vector<scheduler_unit *> schedulers;

  // issue
  unsigned int Issue_Prio;

  // execute
  unsigned m_num_function_units;
  std::vector<unsigned> m_dispatch_port;
  std::vector<unsigned> m_issue_port;
  std::vector<simd_function_unit *>
      m_fu;  // stallable pipelines should be last in this array
  ldst_unit *m_ldst_unit;
  static const unsigned MAX_ALU_LATENCY = 512;
  unsigned num_result_bus;
  std::vector<std::bitset<MAX_ALU_LATENCY> *> m_result_bus;

  ResultBusses m_res_bus_improved;  // MOD. Improved Result bus to take into account conflicts with RF banks

  // used for local address mapping with single kernel launch
  unsigned kernel_max_cta_per_shader;
  unsigned kernel_padded_threads_per_cta;
  // Used for handing out dynamic warp_ids to new warps.
  // the differnece between a warp_id and a dynamic_warp_id
  // is that the dynamic_warp_id is a running number unique to every warp
  // run on this shader, where the warp_id is the static warp slot.
  unsigned m_dynamic_warp_id;

  // Jin: concurrent kernels on a sm
 public:
  bool can_issue_1block(kernel_info_t &kernel);
  bool occupy_shader_resource_1block(kernel_info_t &kernel, bool occupy);
  void release_shader_resource_1block(unsigned hw_ctaid, kernel_info_t &kernel);
  int find_available_hwtid(unsigned int cta_size, bool occupy);
  shd_warp_t *get_shd_warp(int id) override { return m_warp[id];} // MOD. IBuffer_ooo
  unsigned int get_num_subcores() { return (m_config->sub_core_model ? m_config->gpgpu_num_sched_per_core : 1) ; } // MOD. Added L0I.
  int get_subcore_req_fetch_L1I_priority() { return m_subcore_req_fetch_L1I_priority; } // MOD. Added L0I
  void set_subcore_req_fetch_L1I_priority(int new_subcore_req_fetch_L1I_priority) { m_subcore_req_fetch_L1I_priority = new_subcore_req_fetch_L1I_priority; } // MOD. Added L0I

 private:
  unsigned int m_occupied_n_threads;
  unsigned int m_occupied_shmem;
  unsigned int m_occupied_regs;
  unsigned int m_occupied_ctas;
  std::bitset<MAX_THREAD_PER_SM> m_occupied_hwtid;
  std::map<unsigned int, unsigned int> m_occupied_cta_to_hwtid;
};

class exec_shader_core_ctx : public shader_core_ctx {
 public:
  exec_shader_core_ctx(class gpgpu_sim *gpu, class simt_core_cluster *cluster,
                       unsigned shader_id, unsigned tpc_id,
                       const shader_core_config *config,
                       const memory_config *mem_config,
                       shader_core_stats *stats)
      : shader_core_ctx(gpu, cluster, shader_id, tpc_id, config, mem_config,
                        stats) {
    create_front_pipeline();
    create_shd_warp();
    create_schedulers();
    create_exec_pipeline();
  }

  virtual void checkExecutionStatusAndUpdate(warp_inst_t &inst, unsigned t,
                                             unsigned tid);
  virtual void func_exec_inst(warp_inst_t &inst);
  virtual unsigned sim_init_thread(kernel_info_t &kernel,
                                   ptx_thread_info **thread_info, int sid,
                                   unsigned tid, unsigned threads_left,
                                   unsigned num_threads, core_t *core,
                                   unsigned hw_cta_id, unsigned hw_warp_id,
                                   gpgpu_t *gpu);
  virtual void create_shd_warp();
  virtual warp_inst_t *get_next_inst(unsigned warp_id, address_type pc); // MOD. VPREG
  virtual void decrement_trace_pc(unsigned warp_id); // MOD. VPREG
  virtual void get_pdom_stack_top_info(unsigned warp_id, const warp_inst_t *pI,
                                       unsigned *pc, unsigned *rpc);
  virtual const active_mask_t &get_active_mask(unsigned warp_id,
                                               const warp_inst_t *pI);
  
  // Implementation of pure virtual functions from shader_core_ctx_wrapper
  virtual RRS* get_loog_rrs() override { return nullptr; }
  virtual bool get_is_loog_enabled() override { return m_config->is_loog_enabled; }
};

class simt_core_cluster {
 public:
  simt_core_cluster(class gpgpu_sim *gpu, unsigned cluster_id,
                    const shader_core_config *config,
                    const memory_config *mem_config, shader_core_stats *stats,
                    memory_stats_t *mstats);
  virtual ~simt_core_cluster() {
    for(unsigned int i = 0; i < m_core.size(); i++) {
      delete m_core[i];
    }
  }

  void reset_cycless_access_history() {
    for(unsigned i = 0; i < m_core.size(); i++) {
      m_core[i]->reset_cycless_access_history();
    }
  }

  void gather_stats(Element_stats &all_stats, coalescingStatsAcrossSms& coal_stats_l1d, coalescingStatsAcrossSms& coal_stats_const, coalescingStatsAcrossSms& coal_stats_sharedmem) {
    for(unsigned i = 0; i < m_core.size(); i++) {
      m_core[i]->gather_gpu_per_sm_stats(all_stats, coal_stats_l1d, coal_stats_const, coal_stats_sharedmem);
    }
  }

  void gather_single_stat(Element_stats &all_stats, std::string stat_name) {
    for(unsigned i = 0; i < m_core.size(); i++) {
      m_core[i]->gather_gpu_per_sm_single_stat(all_stats, stat_name);
    }
  }

  traffic_breakdown& get_incomming_traffic_stats() { return m_incoming_traffic_stats; }
  traffic_breakdown& get_outgoing_traffic_stats() { return m_outgoing_traffic_stats; }

  void core_cycle();
  void icnt_cycle();

  void reinit();
  unsigned issue_block2core();
  void cache_flush();
  void cache_invalidate();
  bool icnt_injection_buffer_full(unsigned size, bool write);
  void icnt_inject_request_packet(class mem_fetch *mf);

  // for perfect memory interface
  bool response_queue_full() {
    return (m_response_fifo.size() >= m_config->n_simt_ejection_buffer_size);
  }
  void push_response_fifo(class mem_fetch *mf) {
    m_response_fifo.push_back(mf);
  }

  void get_pdom_stack_top_info(unsigned sid, unsigned tid, unsigned *pc,
                               unsigned *rpc) const;
  unsigned max_cta(const kernel_info_t &kernel);
  unsigned get_not_completed() const;
  void print_not_completed(FILE *fp) const;
  unsigned get_n_active_cta() const;
  unsigned get_n_active_sms() const;
  gpgpu_sim *get_gpu() { return m_gpu; }

  void display_pipeline(unsigned sid, FILE *fout, int print_mem, int mask);
  void print_cache_stats(FILE *fp, unsigned &dl1_accesses,
                         unsigned &dl1_misses) const;

  void get_cache_stats(cache_stats &cs) const;
  void get_L1I_sub_stats(struct cache_sub_stats &css) const;
  void get_L0I_sub_stats(struct cache_sub_stats &css) const; // MOD. L0I
  void get_L1D_sub_stats(struct cache_sub_stats &css) const;
  void get_L1C_sub_stats(struct cache_sub_stats &css) const;
  void get_L1T_sub_stats(struct cache_sub_stats &css) const;

  void get_icnt_stats(long &n_simt_to_mem, long &n_mem_to_simt) const;
  float get_current_occupancy(unsigned long long &active,
                              unsigned long long &total) const;
  virtual void create_shader_core_ctx() = 0;

  void create_gpu_per_cluster_stats(Element_stats &all_stats);

 protected:
  unsigned m_cluster_id;
  gpgpu_sim *m_gpu;
  const shader_core_config *m_config;
  shader_core_stats *m_stats;
  memory_stats_t *m_memory_stats;
  std::vector<shader_core_ctx_wrapper *> m_core;
  const memory_config *m_mem_config;

  unsigned m_cta_issue_next_core;
  std::list<unsigned> m_core_sim_order;
  std::list<mem_fetch *> m_response_fifo;

  Element_stats m_cluster_stats;
  traffic_breakdown m_outgoing_traffic_stats;//("coretomem");  // core to memory partitions
  traffic_breakdown m_incoming_traffic_stats;//("memtocore");  // memory partition to core
};

class exec_simt_core_cluster : public simt_core_cluster {
 public:
  exec_simt_core_cluster(class gpgpu_sim *gpu, unsigned cluster_id,
                         const shader_core_config *config,
                         const memory_config *mem_config,
                         class shader_core_stats *stats,
                         class memory_stats_t *mstats)
      : simt_core_cluster(gpu, cluster_id, config, mem_config, stats, mstats) {
    create_shader_core_ctx();
  }

  virtual void create_shader_core_ctx();
};

class shader_memory_interface : public mem_fetch_interface {
 public:
  shader_memory_interface(shader_core_ctx_wrapper *core, simt_core_cluster *cluster) {
    m_core = core;
    m_cluster = cluster;
  }
  ~shader_memory_interface() override {}
  virtual bool full(unsigned size, bool write) const {
    return m_cluster->icnt_injection_buffer_full(size, write);
  }
  virtual void push(mem_fetch *mf) {
    m_core->inc_simt_to_mem(mf->get_num_flits(true));
    m_cluster->icnt_inject_request_packet(mf);
  }

  virtual void flush() {}

 private:
  shader_core_ctx_wrapper *m_core;
  simt_core_cluster *m_cluster;
};

class perfect_memory_interface : public mem_fetch_interface {
 public:
  perfect_memory_interface(shader_core_ctx_wrapper *core, simt_core_cluster *cluster) {
    m_core = core;
    m_cluster = cluster;
  }
  ~perfect_memory_interface() override {}
  virtual bool full(unsigned size, bool write) const {
    return m_cluster->response_queue_full();
  }
  virtual void push(mem_fetch *mf) {
    if (mf && mf->isatomic())
      mf->do_atomic();  // execute atomic inside the "memory subsystem"
    m_core->inc_simt_to_mem(mf->get_num_flits(true));
    m_cluster->push_response_fifo(mf);
  }

  virtual void flush() {}

 private:
  shader_core_ctx_wrapper *m_core;
  simt_core_cluster *m_cluster;
};

inline unsigned int scheduler_unit::get_sid() const { return m_shader->get_sid(); }

#endif /* SHADER_H */
