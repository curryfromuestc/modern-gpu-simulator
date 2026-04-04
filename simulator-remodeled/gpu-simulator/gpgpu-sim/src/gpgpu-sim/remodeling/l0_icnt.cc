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

#include "l0_icnt.h"

#include "../gpu-cache.h"
#include "../gpu-sim.h"
#include "../shader_core_wrapper.h"


unsigned num_bytes_cache_req(unsigned line_size, address_type pc) {
    assert((line_size % 8) == 0);
    unsigned nbytes = line_size / 8;
    unsigned offset_in_block = pc & (line_size- 1);
    if ((offset_in_block + nbytes) > line_size) {
        nbytes = (line_size - offset_in_block);
    }
    return nbytes;
}

address_type get_pc_of_request(address_type pc) {
    return pc - PROGRAM_MEM_START;
}

L0_icnt::L0_icnt(read_only_cache *L1, gpgpu_sim *gpu, shader_core_ctx_wrapper* shader, int max_num_L1_reply_ports_allowed, int max_num_L1_request_ports_allowed, int latency_L0_to_L1, int latency_L1_to_L0) {
    m_L1 = L1;
    m_gpu = gpu;
    m_shader = shader;
    m_max_num_L1_reply_ports_allowed = max_num_L1_reply_ports_allowed;
    m_max_num_L1_request_ports_allowed = max_num_L1_request_ports_allowed;
    m_latency_of_L1_to_L0s_icnt_queue = latency_L1_to_L0;
    m_latency_of_L0s_icnt_to_L1_queue = latency_L0_to_L1;

    assert(m_max_num_L1_reply_ports_allowed>0);
    assert(m_max_num_L1_request_ports_allowed>0);
    assert(m_latency_of_L1_to_L0s_icnt_queue>0);
    assert(m_latency_of_L0s_icnt_to_L1_queue>0);
    
    m_icnt_to_L1_queue.resize(m_max_num_L1_request_ports_allowed);
    for(int i = 0; i < m_max_num_L1_request_ports_allowed; i++) {
        m_icnt_to_L1_queue[i].resize(m_latency_of_L0s_icnt_to_L1_queue, nullptr); 
    }
    m_L1_to_icnt_queue.resize(m_max_num_L1_reply_ports_allowed);
    for(int i = 0; i < m_max_num_L1_reply_ports_allowed; i++) {
        m_L1_to_icnt_queue[i].resize(m_latency_of_L1_to_L0s_icnt_queue, nullptr); 
    }

    m_max_size_icnt_L1_TLB_to_cache =1;
}

L0_icnt::~L0_icnt() {
    flush();
}

void L0_icnt::add_L0(read_only_cache *L0) {
    m_L0.push_back(L0);
}

bool L0_icnt::full(unsigned size, bool write) const {
    bool res = true;
    for(int i = 0; i < m_max_num_L1_request_ports_allowed; i++) {
        if(m_icnt_to_L1_queue[i][m_latency_of_L0s_icnt_to_L1_queue-1] == nullptr) {
            res = false;
            break;
        }
    }
    return res;
}

bool L0_icnt::is_L1_to_icnt_queue_full() {
    bool res = true;
    for(int i = 0; i < m_max_num_L1_reply_ports_allowed; i++) {
        if(m_L1_to_icnt_queue[i][m_latency_of_L1_to_L0s_icnt_queue-1] == nullptr) {
            res = false;
            break;
        }
    }
    return res;
}

int L0_icnt::get_available_L1_to_icnt_port_id() {
    int res = -1;
    for(int i = 0; i < m_max_num_L1_reply_ports_allowed; i++) {
        if(m_L1_to_icnt_queue[i][m_latency_of_L1_to_L0s_icnt_queue-1] == nullptr) {
            res = i;
            break;
        }
    }
    assert(res != -1);
    return res;
}

// L0_icnt pushes are execute before L0_icnt cycle
void L0_icnt::push(mem_fetch *mf) {
    // Maximum of requests allowed  management
    bool inserted = false;
    for(int i = 0; i < m_max_num_L1_request_ports_allowed; i++) {
        if(m_icnt_to_L1_queue[i][m_latency_of_L0s_icnt_to_L1_queue-1] == nullptr) {
            m_icnt_to_L1_queue[i][m_latency_of_L0s_icnt_to_L1_queue-1] = mf;
            inserted = true;
            break;
        }
    }
    assert(inserted);

    // L1 request priority arbitation management
    m_shader->set_subcore_req_fetch_L1I_priority( (mf->get_subcore() + 1) % m_shader->get_num_subcores() );
}


void L0_icnt::cycle() {

    for(int i = 0; i < m_max_num_L1_reply_ports_allowed; i++) {
        if(m_L1_to_icnt_queue[i][0] != nullptr) {
            mem_fetch *mf = m_L1_to_icnt_queue[i][0];
            bool safe_to_pop = true;
            bool not_used = true;
            bool is_prefetch = mf->get_original_mf()->get_is_prefetch();
            unsigned int mf_subcore_id = mf->get_original_mf()->get_subcore();
            mem_access_type type = mf->get_access().get_type();
            if(is_prefetch) {
                mf->set_is_prefetch(true);
                mf->set_stream_buffer_id(mf->get_original_mf()->get_stream_buffer_id());
            }
            unsigned int cache_id = mf_subcore_id;
            if( (type == CONST_ACC_R) && (cache_id < m_shader->get_num_subcores()) ) {
                cache_id += m_shader->get_num_subcores();
            }
            assert(cache_id < m_L0.size());
            bool is_mf_for_this_subcore = m_L0[cache_id]->waiting_for_fill(mf->get_original_mf());
            bool is_L0_port_subcore_free = m_L0[cache_id]->fill_port_free();
            if(is_mf_for_this_subcore && !is_L0_port_subcore_free) {
                safe_to_pop = false;
            }
            if (is_mf_for_this_subcore && is_L0_port_subcore_free) {
                mf->set_is_filling_L0(true);
                mf->set_status(IN_L0_FILL_QUEUE, m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
                m_L0[cache_id]->fill(mf, m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
                not_used = false;
            }
            if(safe_to_pop) {
                m_L1_to_icnt_queue[i][0] = nullptr;
            }
            if( (safe_to_pop && not_used) || (safe_to_pop && is_prefetch) ) {
                delete mf->get_original_mf();
                delete mf;
            }
        }
        for (int stage = 0; stage < m_latency_of_L1_to_L0s_icnt_queue - 1; stage++) {
            if (m_L1_to_icnt_queue[i][stage] == nullptr) {
                m_L1_to_icnt_queue[i][stage] = m_L1_to_icnt_queue[i][stage + 1];
                m_L1_to_icnt_queue[i][stage + 1] = nullptr;
            }
        }
    }

    bool inserted = true;
    // From TLB Stage to Cache
    for(int i = 0; (i < m_max_num_L1_request_ports_allowed) && !is_L1_to_icnt_queue_full() && m_L1->data_port_free() && inserted && !m_icnt_L1_TLB_to_cache.empty(); i++) {
        mem_fetch *mf = m_icnt_L1_TLB_to_cache.front();
        std::list<cache_event> events;
        enum cache_request_status status = cache_request_status::NOT_INITIALIZED;
        address_type addr_req = mf->get_access_address();
        unsigned nbytes = num_bytes_cache_req(m_gpu->getShaderCoreConfig()->m_L1I_L1_half_C_cache_config.get_line_sz(), addr_req);
        mem_access_t acc(mf->get_access().get_type(), addr_req, nbytes, false, m_gpu->gpgpu_ctx);
        mem_fetch *mf_n = new mem_fetch(acc, NULL /*we don't have an instruction yet*/, READ_PACKET_SIZE,
                mf->get_wid(), mf->get_sid(), mf->get_tpc(), mf->get_mem_config(), m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle, mf, NULL, mf->get_unique_function_id());
        bool erase_orifinal_mf = false;
        status = m_L1->access((new_addr_type) mf_n->get_access_address(), mf_n, m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle, events, erase_orifinal_mf);

        if(status == RESERVATION_FAIL) {
            inserted = false;
            delete mf_n;
        }else if(status == MISS) {
            inserted = true;
        }else if(status == HIT) {
            inserted = true;
            int port = get_available_L1_to_icnt_port_id();
            m_L1_to_icnt_queue[port][m_latency_of_L1_to_L0s_icnt_queue - 1] = mf_n;
        }else {
            inserted = false;
            delete mf_n;
            assert(0);
        }

        if(erase_orifinal_mf) {
            delete mf;
        }
        if(inserted) {
           m_icnt_L1_TLB_to_cache.pop();
        }
    }

    // ICNT to IL1 TLB
    for(int i = 0; (i < m_max_num_L1_request_ports_allowed); i++) {
        if(m_icnt_to_L1_queue[i][0] != nullptr) {
            inserted = false;
            mem_fetch *mf = m_icnt_to_L1_queue[i][0];
            cache_request_status tlb_acc = HIT;
            if((tlb_acc == HIT)) {
                if(m_icnt_L1_TLB_to_cache.size() < m_max_size_icnt_L1_TLB_to_cache) {
                    m_icnt_L1_TLB_to_cache.push(mf);
                    inserted = true;
                }
            }else if((tlb_acc == MISS) || (tlb_acc == MSHR_HIT)) {
                inserted = true;
            }else {
                assert(tlb_acc == RESERVATION_FAIL);
                inserted = false;
            }

            if(inserted) {
                m_icnt_to_L1_queue[i][0] = nullptr;
            }
        }
        for (int stage = 0; stage < m_latency_of_L0s_icnt_to_L1_queue - 1; stage++) {
            if (m_icnt_to_L1_queue[i][stage] == nullptr) {
                m_icnt_to_L1_queue[i][stage] = m_icnt_to_L1_queue[i][stage + 1];
                m_icnt_to_L1_queue[i][stage + 1] = nullptr;
            }
        }
    }
    
    // L1 to ICNT
    for(int i = 0; (i < m_max_num_L1_reply_ports_allowed) && m_L1->access_ready() &&  m_L1->data_port_free(); i++) {
        if(m_L1_to_icnt_queue[i][m_latency_of_L1_to_L0s_icnt_queue - 1] == nullptr) {
            mem_fetch *mf = m_L1->next_access();
            mf->set_reply();
            m_L1_to_icnt_queue[i][m_latency_of_L1_to_L0s_icnt_queue - 1] = mf;
        }
    }
}

void L0_icnt::flush() {
    for(int i = 0; i < m_max_num_L1_request_ports_allowed; i++) {
        for(int j = 0; j < m_latency_of_L0s_icnt_to_L1_queue; j++) {
            if(m_icnt_to_L1_queue[i][j] != nullptr) {
                delete m_icnt_to_L1_queue[i][j];
                m_icnt_to_L1_queue[i][j] = nullptr;
            }
        }
    }
    
    for(int i = 0; i < m_max_num_L1_reply_ports_allowed; i++) {
        for(int j = 0; j < m_latency_of_L1_to_L0s_icnt_queue; j++) {
            if(m_L1_to_icnt_queue[i][j] != nullptr) {
                if(m_L1_to_icnt_queue[i][j]->get_original_mf() != nullptr) {
                    delete m_L1_to_icnt_queue[i][j]->get_original_mf();
                }
                delete m_L1_to_icnt_queue[i][j];
                m_L1_to_icnt_queue[i][j] = nullptr;
            }
        }
    }
}
