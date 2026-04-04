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

#include "../../abstract_hardware_model.h"
#include "../gpu-cache.h"
#include <queue>
#include <map>
#include <set>
#include <limits>

class SM;
class first_level_instruction_cache;


struct stream_buffer_search_result {
  stream_buffer_search_result() {
    is_hit_requested_addr = false;
    is_hit_prefetch_addr = false;
    stream_buffer_id = std::numeric_limits<unsigned int>::max();
  }
  stream_buffer_search_result(bool is_hit_requested_addr, bool is_hit_prefetch_addr, unsigned int stream_buffer_id)
      : is_hit_requested_addr(is_hit_requested_addr), is_hit_prefetch_addr(is_hit_prefetch_addr), stream_buffer_id(stream_buffer_id) {}
    
  bool is_hit_requested_addr;
  bool is_hit_prefetch_addr;
  unsigned int stream_buffer_id;
};

struct prefetch_element {
  prefetch_element() {
    sm_warp_id = std::numeric_limits<unsigned int>::max();
    unique_function_id = std::numeric_limits<unsigned int>::max();
    is_ready = false;
    is_request_to_cache = false;
  }
  prefetch_element(unsigned int sm_warp_id, unsigned int unique_function_id, bool is_ready, bool is_request_to_cache)
      : sm_warp_id(sm_warp_id),
        unique_function_id(unique_function_id),
        is_ready(is_ready),
        is_request_to_cache(is_request_to_cache) {}
  unsigned int sm_warp_id;
  unsigned int unique_function_id;
  bool is_ready;
  bool is_request_to_cache;
  std::map<unsigned int, std::set<new_addr_type>> waiting_warp_ids_and_its_addrs;
  std::set<new_addr_type> waiting_addrs_of_the_block;
};

class single_stream_buffer {
 public:
  single_stream_buffer(unsigned int sb_id, int core_id, bool is_prefetching_enabled,
                        unsigned int subcore_id, SM *sm,
                        first_level_instruction_cache *cache,
                        unsigned int max_size, unsigned int line_size,
                        mem_fetch_interface *memport);
  
  ~single_stream_buffer();

  SM* get_sm();

  first_level_instruction_cache *get_cache();
  
  bool is_active();

  bool is_full();

  bool is_a_pending_request(new_addr_type addr, unsigned long long gpu_cycle);

  bool is_hit(new_addr_type addr, unsigned long long gpu_cycle);

  unsigned long long get_gpu_cycle_hit();

  void flush();

  void set_new_stream(new_addr_type addr, unsigned int unique_function_id, unsigned long long gpu_cycle, unsigned int warp_id);

  void do_prefetch();

  bool fill(mem_fetch *mf, unsigned time);

  void set_waiting_fill_in_cache(new_addr_type base_addr, new_addr_type request_addr, unsigned int warp_id);

  bool send_to_cache();

  private:
    bool m_is_enabled;
    bool m_is_currently_prefetching;
    unsigned int m_stream_buffer_id;
    unsigned int m_max_size;
    unsigned int m_line_size;
    unsigned int m_sm_id;
    unsigned int m_subcore_id;
    unsigned int m_first_sm_warp_id_reserved;
    unsigned long long gpu_cycle_hit;
    SM *m_sm;
    first_level_instruction_cache *m_cache;
    new_addr_type m_next_addr_to_prefetch;
    unsigned int m_current_unique_function_id;
    std::queue<new_addr_type> m_queue_ordered_prefetches;
    std::map<new_addr_type, prefetch_element> m_all_prefetches;
    mem_fetch_interface *m_memport;
};

class multiple_stream_buffers {
 public:
  multiple_stream_buffers(int core_id, bool is_prefetching_enabled,
                          unsigned int subcore_id, SM *sm, 
                          first_level_instruction_cache* cache,
                          unsigned int max_size_per_stream_buffer,
                          unsigned int line_size, unsigned int num_stream_buffers,
                          unsigned int max_num_prefetches_per_cycle,
                          mem_fetch_interface *memport);
  
  ~multiple_stream_buffers();

  stream_buffer_search_result search(new_addr_type base_addr_request, new_addr_type base_addr_prefetch, unsigned long long gpu_cycle);
  
  void set_new_stream(new_addr_type addr, unsigned int unique_function_id, unsigned long long gpu_cycle, unsigned int warp_id);

  void cycle(bool can_sb_send_to_cache);

  void flush();

  bool fill(mem_fetch *mf, unsigned time);

  bool is_already_allocated(new_addr_type addr, unsigned long long gpu_cycle, unsigned int stream_buffer_id);

  void set_waiting_fill_in_cache(unsigned int stream_buffer_id, new_addr_type base_addr, new_addr_type request_addr, unsigned int warp_id);

  private:
    bool m_is_enabled;
    unsigned int m_max_size_per_stream_buffer;
    unsigned int m_max_num_prefetches_per_cycle;
    unsigned int m_line_size;
    unsigned int m_sm_id;
    unsigned int m_subcore_id;
    unsigned int m_num_stream_buffers;
    unsigned int m_next_sb_send_request;
    SM *m_sm;
    first_level_instruction_cache* m_cache;
    std::vector<single_stream_buffer*> m_stream_buffers;
    mem_fetch_interface *m_memport;
};