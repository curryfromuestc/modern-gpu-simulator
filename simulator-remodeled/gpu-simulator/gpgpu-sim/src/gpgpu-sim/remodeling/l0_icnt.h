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

#include <vector>
#include <unordered_map>
#include <queue>
#include <stdio.h>
#include "../../constants.h"

class shader_core_ctx; // Definition to be allowed to compile. Code of this class in shader.h and shader.cc
class read_only_cache;  // Definition to be allowed to compile. Code of this class in gpu-cache.h and gpu-cache.cc
class shader_core_ctx_wrapper;  // Definition to be allowed to compile. Code of this class in gpu-cache.h and gpu-cache.cc


unsigned num_bytes_cache_req(unsigned line_size, address_type pc);

address_type get_pc_of_request(address_type pc);
/**
 * @brief 
 * 
 */
class L0_icnt : public mem_fetch_interface{
    public:

        /**
         * @brief Construct a new l0 icnt object
         * 
         * @param L1 Pointer to the L1 of the SM
         * @param gpu Pointer to the structure that holds all the structures of the simulation
         * @param max_num_L1_reply_ports_allowed Maximum number of replies (reply ports) allowed in a given cycle
         * @param max_num_L1_request_ports_allowed Maximum number of request (requests ports) allowed in a given cycle
         * @param latency_L0_to_L1 Maximum size of the queue that holds the requests from the L0s to the L1
         * @param latency_L1_to_L0 Maximum size of the queue that holds the replies from the L1 to the L0s
         */
        L0_icnt(read_only_cache *L1, gpgpu_sim *gpu, shader_core_ctx_wrapper* shader, int max_num_L1_reply_ports_allowed, int max_num_L1_request_ports_allowed, int latency_L0_to_L1, int latency_L1_to_L0);
        
        ~L0_icnt() override;
        /**
         * @brief Method to add an L0 that belongs to the L0_icnt of the SM
         * 
         * @param L0 , pointer to the L0 that belong to a sub-core of the SM
         */
        void add_L0(read_only_cache *L0);
        
        /**
         * @brief Request a mem_fetch to the L1 (Lor beyond if it is needed) due to a miss in a L0
         * 
         * @param mf , pointer of the mem_fetch that is going to be requested to lower cache levels
         */
        virtual void push(mem_fetch *mf);
        
        /**
         * @brief Method that checks if the icnt to make requests to L1 is full
         * 
         * @param size 
         * @param write 
         * @return true 
         * @return false 
         */
        virtual bool full(unsigned size, bool write) const;

        /**
         * @brief It executes a cycle of the L0_icnt where it checks if there is a response ready from the L1
         * 
         */
        void cycle();

        /**
         * @brief Method that flushes the L0_icnt
         * 
         */
        void flush();

    private:

        /**
         * @brief Pointer to the structure that holds all the structures of the simulation
         * 
         */
        gpgpu_sim *m_gpu;

        /**
         * @brief Vector with pointers to all the L0 of the SM that is the owner of the L0_icnt
         * 
         */
        std::vector<read_only_cache *> m_L0;

        /**
         * @brief Pointer to the L1 of the SM that is the owner of the L0_icnt
         * 
         */
        read_only_cache *m_L1;

        /**
         * @brief Pointer to the shader that owns the L0_icnt
         * 
         */
        shader_core_ctx_wrapper* m_shader;

        /**
         * @brief Indicates how many ports to L1 are allowed as maximum in a given cycle
         * 
         */
        int m_max_num_L1_request_ports_allowed;

        /**
         * @brief Indicates how many ports from L1 are allowed as maximum in a given cycle
         * 
         */
        int m_max_num_L1_reply_ports_allowed;

        /**
         * @brief Latency of the L1 to L0s icnt queue
         */
        int m_latency_of_L0s_icnt_to_L1_queue;

        /**
         * @brief Latency of the L0s to L1 icnt queue
         */
        int m_latency_of_L1_to_L0s_icnt_queue;
        /**
         * @brief Queue that holds the requests from the L0s to the L1. First dimension the maximum number of parallel requests in the same cycle.
         * 
         */
        std::vector<std::vector<mem_fetch*>> m_icnt_to_L1_queue;

        /**
         * @brief Queue that holds the replies from the L1 to the L0s. First dimension the maximum number of replies requests in the same cycle.
         * 
         */
        std::vector<std::deque<mem_fetch*>> m_L1_to_icnt_queue;

        /**
         * @brief Queue that holds the mem_fetchs that are going to be sent to the cache from the TLB
         */
        std::queue<mem_fetch*> m_icnt_L1_TLB_to_cache;

        /**
         * @brief Maximum size of the queue that holds the mem_fetchs that are going to be sent to the cache from the TLB
         */
        unsigned int m_max_size_icnt_L1_TLB_to_cache;

        /**
         * @brief Method that checks if the queue that holds the replies from the L1 to the L0s is full
         */
        bool is_L1_to_icnt_queue_full();

        /**
         * @brief Method that returns the id of the first available port to receive a reply from the L1
         */
        int get_available_L1_to_icnt_port_id();
};