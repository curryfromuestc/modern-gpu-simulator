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

#include "stream_buffer.h"

#include "first_level_instruction_cache.h"
#include "sm.h"
#include "../gpu-cache.h"
#include "../gpu-sim.h"

single_stream_buffer::single_stream_buffer(unsigned int sb_id, int core_id, bool is_prefetching_enabled,
        unsigned int subcore_id, SM * sm, first_level_instruction_cache *cache,
        unsigned int max_size, unsigned int line_size, mem_fetch_interface *memport) {
    m_sm_id = core_id;
    m_is_enabled = is_prefetching_enabled;
    m_subcore_id = subcore_id;
    m_sm = sm;
    m_cache = cache;
    m_max_size = max_size;
    m_line_size = line_size;
    m_memport = memport;
    gpu_cycle_hit = std::numeric_limits<unsigned long long>::max();
    m_stream_buffer_id = sb_id;
    m_next_addr_to_prefetch = std::numeric_limits<new_addr_type>::max();
    m_current_unique_function_id = std::numeric_limits<unsigned int>::max();
    m_first_sm_warp_id_reserved = 0;
}

single_stream_buffer::~single_stream_buffer() {
    flush();
}

SM* single_stream_buffer::get_sm() { 
    return m_sm;
}

bool single_stream_buffer::is_active() {
    return m_is_currently_prefetching;
}

unsigned long long single_stream_buffer::get_gpu_cycle_hit() {
    return gpu_cycle_hit;
}

first_level_instruction_cache *single_stream_buffer::get_cache() { 
    return m_cache;
}

bool single_stream_buffer::is_full() {
    return m_queue_ordered_prefetches.size() == m_max_size;
}

bool single_stream_buffer::is_a_pending_request(new_addr_type addr, unsigned long long gpu_cycle) {
    auto it = m_all_prefetches.find(addr);
    bool hit = it != m_all_prefetches.end();
    return hit;
}

bool single_stream_buffer::is_hit(new_addr_type addr, unsigned long long gpu_cycle) {
    assert(m_queue_ordered_prefetches.size() == m_all_prefetches.size());
    bool hit = false;
    if(!m_queue_ordered_prefetches.empty()) {
        new_addr_type first_addr = m_queue_ordered_prefetches.front();
        hit = addr == first_addr;
    }
    if(hit) {
        gpu_cycle_hit = gpu_cycle;
    }
    return hit;
}

void single_stream_buffer::flush() {
    m_all_prefetches.clear();
    while (!m_queue_ordered_prefetches.empty()) {
        m_queue_ordered_prefetches.pop();
    }
    m_is_currently_prefetching = false;
    m_next_addr_to_prefetch = std::numeric_limits<new_addr_type>::max();
    m_current_unique_function_id = std::numeric_limits<unsigned int>::max();
}

void single_stream_buffer::set_new_stream(new_addr_type addr, unsigned int unique_function_id, unsigned long long gpu_cycle, unsigned int warp_id) {
    bool safe_to_set = true;
    if(!m_queue_ordered_prefetches.empty()) {
        new_addr_type top_addr = m_queue_ordered_prefetches.front();
        auto it = m_all_prefetches.find(top_addr);
        assert(it != m_all_prefetches.end());
        safe_to_set = !it->second.is_request_to_cache;
    }
    if(safe_to_set && ((addr != m_next_addr_to_prefetch) || (unique_function_id != m_current_unique_function_id)) ) {
        flush();
        m_is_currently_prefetching = true;
        m_next_addr_to_prefetch = addr;
        m_current_unique_function_id = unique_function_id;
        m_first_sm_warp_id_reserved = warp_id;
    }
}

bool single_stream_buffer::fill(mem_fetch *mf, unsigned time) {
    bool res = false;
    new_addr_type addr = mf->get_addr();
    auto it = m_all_prefetches.find(addr);
    if(it != m_all_prefetches.end()) {
        it->second.is_ready = true;
    }else {
        res = true;
    }
    return res;
}

bool single_stream_buffer::send_to_cache() {
    bool can_continue_send_to_cache = true;
    if(!m_queue_ordered_prefetches.empty()) {
        new_addr_type addr = m_queue_ordered_prefetches.front();
        auto it = m_all_prefetches.find(addr);
        assert(it != m_all_prefetches.end());
        if(it->second.is_ready && it->second.is_request_to_cache) {
            can_continue_send_to_cache = false;
            bool safe_to_pop = m_cache->fill_from_stream_buffer(addr, m_sm->get_current_gpu_cycle(), it->second);
            if(safe_to_pop) {
                m_queue_ordered_prefetches.pop();
                m_all_prefetches.erase(addr);
            }
        }
    }
    return can_continue_send_to_cache;
}

void single_stream_buffer::do_prefetch() {
    bool continue_prefetching = false;
    if(m_is_enabled && m_is_currently_prefetching ) {
        continue_prefetching = true;
        if(!m_memport->full(m_line_size, false) && !is_full()) {
            first_level_instruction_cache* sh_cache = get_cache();
            SM* sm = get_sm();
            unsigned int cache_idx;
            std::list<cache_event> events;
            unsigned long long gpu_cycle = sm->get_current_gpu_cycle();
            unsigned int nbytes = m_line_size;
            mem_access_t acc(INST_ACC_R, m_next_addr_to_prefetch, nbytes, false,
                            sm->get_gpu()->gpgpu_ctx);
            mem_fetch *mf =
            new mem_fetch(acc, NULL /*we don't have an instruction yet*/,
                            READ_PACKET_SIZE, m_first_sm_warp_id_reserved, sm->get_sid(),
                            sm->get_tpc_id(), sm->get_memory_config(),
                            gpu_cycle, NULL, NULL, m_current_unique_function_id);
            mf->set_subcore(m_subcore_id);
            mf->set_is_prefetch(true);
            mf->set_stream_buffer_id(m_stream_buffer_id);
            cache_request_status status = sh_cache->get_tag_array()->probe(m_next_addr_to_prefetch, cache_idx, mf, mf->is_write());
            new_addr_type mshr_addr = sh_cache->get_config().mshr_addr(mf->get_addr());
            bool mshr_hit = sh_cache->get_mshr().probe(mshr_addr);
            if((status != HIT) && (status != RESERVATION_FAIL) && !mshr_hit) {
                m_all_prefetches[m_next_addr_to_prefetch] = prefetch_element(m_first_sm_warp_id_reserved, m_current_unique_function_id, false, false);
                m_queue_ordered_prefetches.push(m_next_addr_to_prefetch);
                m_memport->push(mf);
                m_next_addr_to_prefetch += m_line_size;
            }else{
                continue_prefetching = false;
                delete mf;
            }
        }
    }
    m_is_currently_prefetching = continue_prefetching;
}


void single_stream_buffer::set_waiting_fill_in_cache(new_addr_type base_addr, new_addr_type request_addr, unsigned int warp_id) {
    auto it = m_all_prefetches.find(base_addr);
    assert(it != m_all_prefetches.end());
    it->second.is_request_to_cache = true;
    it->second.waiting_addrs_of_the_block.insert(request_addr);
    auto it_wid = it->second.waiting_warp_ids_and_its_addrs.find(warp_id);
    if(it_wid == it->second.waiting_warp_ids_and_its_addrs.end()) {
        it->second.waiting_warp_ids_and_its_addrs[warp_id] = std::set<new_addr_type>();
    }
    it->second.waiting_warp_ids_and_its_addrs[warp_id].insert(request_addr);
}




multiple_stream_buffers::multiple_stream_buffers(int core_id, bool is_prefetching_enabled,
    unsigned int subcore_id, SM *sm, first_level_instruction_cache *cache,
    unsigned int max_size_per_stream_buffer, unsigned int line_size, unsigned int num_stream_buffers,
    unsigned int max_num_prefetches_per_cycle, mem_fetch_interface *memport) {
    m_sm_id = core_id;
    m_is_enabled = is_prefetching_enabled;
    m_subcore_id = subcore_id;
    m_sm = sm;
    m_max_size_per_stream_buffer = max_size_per_stream_buffer;
    m_line_size = line_size;
    m_cache = cache;
    m_num_stream_buffers = num_stream_buffers;
    m_next_sb_send_request = 0;
    m_max_num_prefetches_per_cycle = max_num_prefetches_per_cycle;
    m_memport = memport;
    for(unsigned int i = 0; i < m_num_stream_buffers; i++) {
        m_stream_buffers.push_back(new single_stream_buffer(i, core_id, is_prefetching_enabled, subcore_id, sm, cache, max_size_per_stream_buffer, line_size, memport));
    }
}

multiple_stream_buffers::~multiple_stream_buffers() {
    for(unsigned int i = 0; i < m_num_stream_buffers; i++) {
        delete m_stream_buffers[i];
    }
    m_stream_buffers.clear();
}

stream_buffer_search_result multiple_stream_buffers::search(new_addr_type base_addr_request, new_addr_type base_addr_prefetch, unsigned long long gpu_cycle) {
    stream_buffer_search_result result;
    unsigned int id_sb_hit_prefetch = std::numeric_limits<unsigned int>::max();
    for(unsigned int i = 0; i < m_num_stream_buffers; i++) {
        if(m_stream_buffers[i]->is_hit(base_addr_request, gpu_cycle)) {
            result.is_hit_requested_addr = true;
            result.stream_buffer_id = i;
            break;
        }else if(m_stream_buffers[i]->is_hit(base_addr_prefetch, gpu_cycle)) {
            id_sb_hit_prefetch = i;
        }
    }
    if(!result.is_hit_requested_addr && (id_sb_hit_prefetch != std::numeric_limits<unsigned int>::max())) {
        result.is_hit_prefetch_addr = true;
        result.stream_buffer_id = id_sb_hit_prefetch;
    }
    return result;
}

void multiple_stream_buffers::set_new_stream(new_addr_type addr, unsigned int unique_function_id, unsigned long long gpu_cycle, unsigned int warp_id) {
    unsigned long long far_gpu_cycle_hit = std::numeric_limits<unsigned long long>::max();
    unsigned int idx_buffer = 0;
    for(unsigned int i = 0; i < m_num_stream_buffers; i++) {
        if(!m_stream_buffers[i]->is_active()) {
            idx_buffer = i;
            break;
        }else {
            unsigned long long gpu_cycle_hit = m_stream_buffers[i]->get_gpu_cycle_hit();
            if(gpu_cycle_hit < far_gpu_cycle_hit) {
                far_gpu_cycle_hit = gpu_cycle_hit;
                idx_buffer = i;
            }
        }
    }
    m_stream_buffers[idx_buffer]->set_new_stream(addr, unique_function_id, gpu_cycle, warp_id);
}

void multiple_stream_buffers::cycle(bool can_sb_send_to_cache) {
    bool can_send_to_cache = can_sb_send_to_cache;
    for(unsigned int i = 0; i < m_num_stream_buffers; i++) {
        for(unsigned int j = 0; j < m_max_num_prefetches_per_cycle; j++) {
            m_stream_buffers[i]->do_prefetch();
        }
        if(can_send_to_cache) {
            can_send_to_cache = m_stream_buffers[m_next_sb_send_request]->send_to_cache();
            m_next_sb_send_request = (m_next_sb_send_request + 1) % m_num_stream_buffers;
        }  
    }
}

bool multiple_stream_buffers::fill(mem_fetch *mf, unsigned time) {
    return m_stream_buffers[mf->get_stream_buffer_id()]->fill(mf, time);
}

void multiple_stream_buffers::set_waiting_fill_in_cache(unsigned int stream_buffer_id, new_addr_type base_addr, new_addr_type request_addr, unsigned int warp_id) {
    m_stream_buffers[stream_buffer_id]->set_waiting_fill_in_cache(base_addr, request_addr, warp_id);
}

bool multiple_stream_buffers::is_already_allocated(new_addr_type addr, unsigned long long gpu_cycle, unsigned int stream_buffer_id) {
    return m_stream_buffers[stream_buffer_id]->is_a_pending_request(addr, gpu_cycle);
}

void multiple_stream_buffers::flush() {
    for(unsigned int i = 0; i < m_num_stream_buffers; i++) {
        m_stream_buffers[i]->flush();
    }
}