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

#include "first_level_instruction_cache.h"
#include "sm.h"
#include "stream_buffer.h"
#include "../shader.h"
#include "../gpu-sim.h"


first_level_instruction_cache::first_level_instruction_cache(const char *name, cache_config &config, int core_id,
          int type_id, mem_fetch_interface *memport,
          enum mem_fetch_status status, tag_array *new_tag_array, bool is_prefetching_enabled,
          unsigned int subcore_id, SM *sm, unsigned int max_num_prefetches_per_cycle,
          bool is_IB_coalescing_enabled, unsigned int size_per_stream_buffer, unsigned int num_stream_buffers)
    : read_only_cache(name, config, core_id, type_id, memport, status, new_tag_array) {
        m_is_prefetching_enabled = is_prefetching_enabled;
        m_sm_id = core_id;
        m_subcore_id = subcore_id;
        m_sm = sm;
        m_is_IB_coalescing_enabled = is_IB_coalescing_enabled;
        m_size_per_stream_buffer = size_per_stream_buffer;
        m_num_stream_buffers = num_stream_buffers;
        m_max_num_prefetches_per_cycle = max_num_prefetches_per_cycle;
        m_next_response.reset();
}

first_level_instruction_cache::first_level_instruction_cache(const char *name, cache_config &config, int core_id,
          int type_id, mem_fetch_interface *memport,
          enum mem_fetch_status status, bool is_prefetching_enabled,
          unsigned int subcore_id, SM *sm, unsigned int max_num_prefetches_per_cycle,
          bool is_IB_coalescing_enabled, unsigned int size_per_stream_buffer, unsigned int num_stream_buffers)
    : read_only_cache(name, config, core_id, type_id, memport, status) {
        m_is_prefetching_enabled = is_prefetching_enabled;
        m_sm_id = core_id;
        m_subcore_id = subcore_id;
        m_sm = sm;
        m_is_IB_coalescing_enabled = is_IB_coalescing_enabled;
        m_size_per_stream_buffer = size_per_stream_buffer;
        m_num_stream_buffers = num_stream_buffers;
        m_max_num_prefetches_per_cycle = max_num_prefetches_per_cycle;
        m_next_response.reset();
}

first_level_instruction_cache::~first_level_instruction_cache() {
    for(auto it = m_regular_access_status_without_IB_coalescing.begin(); it != m_regular_access_status_without_IB_coalescing.end(); it++) {
        for(auto it2 = it->second.begin(); it2 != it->second.end(); it2++) {
            if(it2->second.mf != nullptr) {
                delete it2->second.mf;
            }
        }
        it->second.clear();
    }
    clear_cache();
    delete m_stream_buffers;
}

void first_level_instruction_cache::initiate_stream_buffers() {
  m_stream_buffers = new multiple_stream_buffers(
      m_sm_id, m_is_prefetching_enabled, m_subcore_id, m_sm, this,
      m_size_per_stream_buffer, m_config.get_line_sz(), m_num_stream_buffers,
      m_max_num_prefetches_per_cycle, m_memport);
}

void first_level_instruction_cache::invalidate() {
    read_only_cache::invalidate();
    clear_cache();
    m_extra_mf_fields.clear();
    m_mshrs.clean_entries();
}

void first_level_instruction_cache::clear_cache() {
    m_regular_access_status_without_IB_coalescing.clear();
    m_next_response.reset();
    
    if(m_is_prefetching_enabled) {
        m_stream_buffers->flush();
    }
}

new_addr_type first_level_instruction_cache::get_base_line_of_address(new_addr_type addr) {
    new_addr_type base_addr = (addr / m_config.get_line_sz()) * m_config.get_line_sz();
    return base_addr;
}

void first_level_instruction_cache::print_without_IB_coalescing_access_status() {
    unsigned i = 0;
    for(auto it = m_regular_access_status_without_IB_coalescing.begin(); it != m_regular_access_status_without_IB_coalescing.end(); it++) {
        for(auto it_addr = it->second.begin(); it_addr != it->second.end(); it_addr++) {
            printf("%d. Warp id: %u, Address: %llu, Cache status: %u\n", i,  it->first, it_addr->first, it_addr->second.cache_status);
            i++;
        }
    }
    fflush(stdout);
}

bool first_level_instruction_cache::regular_request_search_single_warp(unsigned int warp_id, new_addr_type addr) {
    bool is_reg_request_found = false;
    auto it_regular_access_no_IB_coal = m_regular_access_status_without_IB_coalescing.find(warp_id);
    if( (it_regular_access_no_IB_coal != m_regular_access_status_without_IB_coalescing.end()) && !it_regular_access_no_IB_coal->second.empty()) {
        auto it_regular_access_no_IB_coal_addr = it_regular_access_no_IB_coal->second.find(addr);
        is_reg_request_found = it_regular_access_no_IB_coal_addr != it_regular_access_no_IB_coal->second.end();
    }
    return is_reg_request_found;
}

bool first_level_instruction_cache::is_regular_request_found(mem_fetch *mf, new_addr_type addr, bool search_all_warps) {
    bool is_reg_request_found = false;
    if(m_is_IB_coalescing_enabled) {
        auto it_regular_access_IB_coal = m_regular_access_status_with_IB_coalescing.find(addr);
        is_reg_request_found = it_regular_access_IB_coal != m_regular_access_status_with_IB_coalescing.end();
    }else {
        if(search_all_warps) {
            for(auto it = m_regular_access_status_without_IB_coalescing.begin(); it != m_regular_access_status_without_IB_coalescing.end(); it++) {
                is_reg_request_found = regular_request_search_single_warp(it->first, addr);
                if(is_reg_request_found) {
                    break;
                }
            }
        }else {
            is_reg_request_found = regular_request_search_single_warp(mf->get_wid(), addr);
        }
    }
    return is_reg_request_found;
}

cache_request_status first_level_instruction_cache::access(new_addr_type addr, mem_fetch *mf, unsigned int time,
                                    std::list<cache_event> &events) {
    cache_request_status status = IN_L0I_RESPONSE_QUEUE;
    assert(!m_next_response);
    new_addr_type base_addr = get_base_line_of_address(addr);
    bool is_reg_request_found = is_regular_request_found(mf, addr, false);   
    if (!is_reg_request_found) {
        if(m_is_prefetching_enabled) {
            SM * sm = get_sm();
            unsigned int useless_idx;
            new_addr_type addr_to_prefetch = base_addr + m_config.get_line_sz();
            new_addr_type mshr_addr = get_config().mshr_addr(base_addr);
            cache_request_status forecasting_addr_request = m_tag_array->probe(mshr_addr, useless_idx, mf->is_write(), mf);
            stream_buffer_search_result sb_check = m_stream_buffers->search(base_addr, addr_to_prefetch, sm->get_current_gpu_cycle());
            if(!sb_check.is_hit_requested_addr && !sb_check.is_hit_prefetch_addr && (forecasting_addr_request == MISS)) {
                m_stream_buffers->set_new_stream(addr_to_prefetch , mf->get_unique_function_id(), sm->get_current_gpu_cycle(), mf->get_wid());
            }
            if(sb_check.is_hit_requested_addr) {
                m_stream_buffers->set_waiting_fill_in_cache(sb_check.stream_buffer_id, base_addr, addr, mf->get_wid());
            }else {
                status = read_only_cache::access(addr, mf, time, events);
            }
        }else {
            status = read_only_cache::access(addr, mf, time, events); 
        }
        
        if(status != RESERVATION_FAIL) {
            bool mf_will_be_erased = (status == RESERVATION_FAIL) || (status == IN_L0I_RESPONSE_QUEUE);
            if(m_is_IB_coalescing_enabled) {
                m_regular_access_status_with_IB_coalescing[addr] = status_element(status, mf, mf_will_be_erased);
            }else {
                m_regular_access_status_without_IB_coalescing[mf->get_wid()][addr] = status_element(status, mf, mf_will_be_erased);
            }            
            if(status == HIT) {
                m_next_response = std::make_unique<response_element>(addr, mf, mf->get_wid(), mf_will_be_erased);
            }
        }        
    }
    return status;
}

bool first_level_instruction_cache::fill(mem_fetch *mf, unsigned time) {
    if(mf->get_is_prefetch()) {
        return m_stream_buffers->fill(mf, time);
    }else {
        return read_only_cache::fill(mf, time);
    }
}

bool first_level_instruction_cache::fill_from_stream_buffer(new_addr_type prefetch_addr, unsigned time, prefetch_element &pending_information) {
    assert(m_next_response == nullptr && "Next response is not null");
    bool is_safe_to_erase_entry_in_sb = false;
    get_bw_manager().use_fill_port(nullptr);
    new_addr_type mshr_addr = m_config.mshr_addr(prefetch_addr);
    mem_fetch *mf_fill = nullptr;
    if(m_is_IB_coalescing_enabled) {
        assert(!pending_information.waiting_addrs_of_the_block.empty() && "No waiting addresses found");
        auto it_first = pending_information.waiting_addrs_of_the_block.begin();
        new_addr_type top_addr = *it_first;
        auto it_regular_access = m_regular_access_status_with_IB_coalescing.find(top_addr);
        assert(it_regular_access != m_regular_access_status_with_IB_coalescing.end() && "Access status not found");
        it_regular_access->second.cache_status = HIT;
        it_regular_access->second.mf->set_unique_function_id(pending_information.unique_function_id);
        pending_information.waiting_addrs_of_the_block.erase(it_first);
        is_safe_to_erase_entry_in_sb = pending_information.waiting_addrs_of_the_block.empty();
        m_next_response = std::make_unique<response_element>(top_addr, it_regular_access->second.mf, it_regular_access->second.mf->get_wid(), it_regular_access->second.mf_erased);
        mf_fill = it_regular_access->second.mf;
    }else {
        auto it_first_wid = pending_information.waiting_warp_ids_and_its_addrs.begin();
        unsigned int top_wid = it_first_wid->first;
        auto it_regular_access_by_wid = m_regular_access_status_without_IB_coalescing.find(top_wid);
        assert(it_regular_access_by_wid != m_regular_access_status_without_IB_coalescing.end() && "Access status not found");
        assert(!it_first_wid->second.empty() && "No waiting addresses found");
        auto it_first_addr = it_first_wid->second.begin();
        new_addr_type top_addr = *it_first_addr;
        auto it_regular_access_addr = it_regular_access_by_wid->second.find(top_addr);
        assert(it_regular_access_addr != it_regular_access_by_wid->second.end() && "Access status not found");
        it_regular_access_addr->second.cache_status = HIT;   
        if(it_regular_access_addr->second.mf_erased) {
            SM * sm = get_sm();
            mem_access_t acc(INST_ACC_R, top_addr, get_config().get_line_sz(), false,
                    sm->get_gpu()->gpgpu_ctx);
            mem_fetch *mf_new = new mem_fetch(acc, NULL, READ_PACKET_SIZE, top_wid, m_sm_id, sm->get_tpc_id(), sm->get_memory_config(), time, nullptr, nullptr, pending_information.unique_function_id);
            it_regular_access_addr->second.mf = mf_new;
        }else {
            it_regular_access_addr->second.mf->set_unique_function_id(pending_information.unique_function_id);
        }
        it_first_wid->second.erase(it_first_addr);
        if(it_first_wid->second.empty()) {
            pending_information.waiting_warp_ids_and_its_addrs.erase(it_first_wid);   
        }
        is_safe_to_erase_entry_in_sb = pending_information.waiting_warp_ids_and_its_addrs.empty();
        m_next_response = std::make_unique<response_element>(top_addr, it_regular_access_addr->second.mf, top_wid, it_regular_access_addr->second.mf_erased);
        mf_fill = it_regular_access_addr->second.mf;
    }
    assert(mf_fill != nullptr && "Mem fetch is null");
    m_tag_array->fill(mshr_addr, time, mf_fill, mf_fill->is_write());
    return is_safe_to_erase_entry_in_sb;
}

void first_level_instruction_cache::printMapKeysMFFields() {
  for (const auto &pair : m_extra_mf_fields) {
    std::cout << "Key: " << pair.first << std::endl;
  }
  fflush(stdout);
}

bool first_level_instruction_cache::waiting_for_fill(mem_fetch *mf) {
  extra_mf_fields_lookup::iterator e = m_extra_mf_fields.find(mf);
  bool found = (e != m_extra_mf_fields.end());
  if(!found) {
    if(mf->get_is_prefetch()) {
      found = m_stream_buffers->is_already_allocated(mf->get_addr(), get_sm()->get_current_gpu_cycle(), mf->get_stream_buffer_id());
    }
  }
  return found;
}

void first_level_instruction_cache::cycle() {
    read_only_cache::cycle();
    if(!m_next_response) {
        if(access_ready()) {
            bool is_mf_erased = false;
            mem_fetch *mf = next_access();
            bool is_reg_request_found = is_regular_request_found(mf, mf->get_addr(), false);
            if(is_reg_request_found) {
                if(m_is_IB_coalescing_enabled) {
                    assert(is_reg_request_found && "Access status not found");
                    auto it_regular_access_IB_coal = m_regular_access_status_with_IB_coalescing.find(mf->get_addr());
                    it_regular_access_IB_coal->second.cache_status = HIT;
                    is_mf_erased = it_regular_access_IB_coal->second.mf_erased;
                }else {
                    auto it_regular_access_by_wid = m_regular_access_status_without_IB_coalescing.find(mf->get_wid());
                    assert(it_regular_access_by_wid != m_regular_access_status_without_IB_coalescing.end() && "Access status not found");
                    auto it_regular_access_addr = it_regular_access_by_wid->second.find(mf->get_addr());
                    assert(it_regular_access_addr != it_regular_access_by_wid->second.end() && "Access status not found");
                    it_regular_access_addr->second.cache_status = HIT;
                    is_mf_erased = it_regular_access_addr->second.mf_erased;
                }
                m_next_response = std::make_unique<response_element>(mf->get_addr(), mf, mf->get_wid(), is_mf_erased);
            }
        }
    }
    if(m_is_prefetching_enabled) {
        m_stream_buffers->cycle(!m_next_response);
    }
}

bool first_level_instruction_cache::is_first_access_ready() {
    if(!m_next_response) {
        return false;
    }
    if(m_is_IB_coalescing_enabled) {
        auto it_regular_access_IB_coal = m_regular_access_status_with_IB_coalescing.find(m_next_response->addr);
        assert(it_regular_access_IB_coal != m_regular_access_status_with_IB_coalescing.end() && "Access status not found");
        assert(it_regular_access_IB_coal->second.cache_status == HIT && "First access is not a hit");
    }else {
        auto it_regular_access_by_wid = m_regular_access_status_without_IB_coalescing.find(m_next_response->original_warp_id);
        assert(it_regular_access_by_wid != m_regular_access_status_without_IB_coalescing.end() && "Access status not found");
        auto it_regular_access_addr = it_regular_access_by_wid->second.find(m_next_response->addr);
        assert(it_regular_access_addr != it_regular_access_by_wid->second.end() && "Access status not found");
        assert(it_regular_access_addr->second.cache_status == HIT && "First access is not a hit");
    }
    return true;
}

mem_fetch *first_level_instruction_cache::next_first_access() {
    assert(is_first_access_ready() && "First access is not ready");
    mem_fetch *result = nullptr;
    if(m_is_IB_coalescing_enabled) {
        auto it_regular_access = m_regular_access_status_with_IB_coalescing.find(m_next_response->addr);
        assert(it_regular_access != m_regular_access_status_with_IB_coalescing.end() && "Access status not found");
        assert(it_regular_access->second.cache_status == HIT && "First access is not a hit");
        result = it_regular_access->second.mf;
        m_regular_access_status_with_IB_coalescing.erase(it_regular_access);
    }else {
        auto it_regular_access_by_wid = m_regular_access_status_without_IB_coalescing.find(m_next_response->original_warp_id);
        assert(it_regular_access_by_wid != m_regular_access_status_without_IB_coalescing.end() && "Access status not found");
        auto it_regular_access_addr = it_regular_access_by_wid->second.find(m_next_response->addr);
        assert(it_regular_access_addr != it_regular_access_by_wid->second.end() && "Access status not found");
        assert(it_regular_access_addr->second.cache_status == HIT && "First access is not a hit");
        result = it_regular_access_addr->second.mf;
        m_regular_access_status_without_IB_coalescing[m_next_response->original_warp_id].erase(it_regular_access_addr);
    }

    m_next_response.reset();
    
    return result;
}

SM *first_level_instruction_cache::get_sm() { 
  return m_sm;
}