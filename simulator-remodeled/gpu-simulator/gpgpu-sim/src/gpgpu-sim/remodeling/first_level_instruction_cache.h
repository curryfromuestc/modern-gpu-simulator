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
#include <limits>

class SM;
class multiple_stream_buffers;
struct prefetch_element;

struct response_element {
  response_element(new_addr_type addr, mem_fetch *mf, unsigned int original_warp_id, bool mf_will_be_erased) : addr(addr), mf(mf), original_warp_id(original_warp_id), mf_will_be_erased(mf_will_be_erased) {}
  new_addr_type addr;
  mem_fetch *mf;
  unsigned int original_warp_id;
  bool mf_will_be_erased;
};

struct status_element {
  status_element() {
    cache_status = RESERVATION_FAIL;
    mf = nullptr;
    mf_erased = true;
  }
  status_element(cache_request_status cache_status, mem_fetch *mf, bool mf_erased) : cache_status(cache_status), mf(mf), mf_erased(mf_erased) {}
  cache_request_status cache_status;
  mem_fetch *mf;
  bool mf_erased;
};

class first_level_instruction_cache : public read_only_cache {
 public:
  first_level_instruction_cache(const char *name, cache_config &config,
                                int core_id, int type_id,
                                mem_fetch_interface *memport,
                                enum mem_fetch_status status,
                                tag_array *new_tag_array, bool is_prefetching_enabled,
                                unsigned int subcore_id, SM *sm, unsigned int max_num_prefetches_per_cycle,
                                bool is_IB_coalescing_enabled, unsigned int size_per_stream_buffer, unsigned int num_stream_buffers);

  first_level_instruction_cache(const char *name, cache_config &config,
                                int core_id, int type_id, mem_fetch_interface *memport,
                                enum mem_fetch_status status, bool is_prefetching_enabled,
                                unsigned int subcore_id, SM *sm, unsigned int max_num_prefetches_per_cycle,
                                bool is_IB_coalescing_enabled, unsigned int size_per_stream_buffer, unsigned int num_stream_buffers);

  ~first_level_instruction_cache() override;

  void initiate_stream_buffers();

  void clear_cache();
  
  enum cache_request_status access(new_addr_type addr, mem_fetch *mf,
                                   unsigned int time,
                                   std::list<cache_event> &events) override;
  
  new_addr_type get_base_line_of_address(new_addr_type addr);

  void cycle() override;

  bool fill(mem_fetch *mf, unsigned time) override;

  bool fill_from_stream_buffer(new_addr_type prefetch_addr, unsigned time, prefetch_element &pending_information);

  bool waiting_for_fill(mem_fetch *mf) override;

  void printMapKeysMFFields();
    
  void invalidate() override;

  bool is_first_access_ready();

  mem_fetch *next_first_access();

  SM *get_sm();

  bool regular_request_search_single_warp(unsigned int warp_id, new_addr_type addr);
  bool is_regular_request_found(mem_fetch *mf, new_addr_type addr, bool search_all_warps);

  void print_without_IB_coalescing_access_status();

  private:
    std::map<new_addr_type, status_element> m_regular_access_status_with_IB_coalescing;
    std::map<unsigned int, std::map<new_addr_type, status_element>> m_regular_access_status_without_IB_coalescing; // The key of the first map is the warp ip
    std::unique_ptr<response_element> m_next_response;
    unsigned int m_sm_id;
    unsigned int m_subcore_id;
    unsigned int m_max_response_queue_size;
    bool m_is_prefetching_enabled;
    bool m_is_IB_coalescing_enabled;
    SM *m_sm;
    multiple_stream_buffers  *m_stream_buffers;

    unsigned int m_num_stream_buffers;
    unsigned int m_size_per_stream_buffer;
    unsigned int m_max_num_prefetches_per_cycle;
};