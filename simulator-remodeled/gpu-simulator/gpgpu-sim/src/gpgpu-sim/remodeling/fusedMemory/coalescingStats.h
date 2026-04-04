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


#include <map>
#include <set>

#include "../../../abstract_hardware_model.h"
#include "../sm.h"

class coalescingStatsPerSm {
    public:
        coalescingStatsPerSm(std::string name_space, _memory_space_t space_type);

        unsigned long long m_num_interwarp_coalescing;
        unsigned long long m_num_interwarp_coalescing_less_or_equal_than_5_cyc;
        unsigned long long m_num_interwarp_coalescing_less_or_equal_than_10_cyc;
        unsigned long long m_num_interwarp_coalescing_less_or_equal_than_20_cyc;
        unsigned long long m_num_interwarp_coalescing_less_or_equal_than_30_cyc;
        unsigned long long m_num_interwarp_coalescing_less_or_equal_than_40_cyc;
        unsigned long long m_num_interwarp_coalescing_less_or_equal_than_50_cyc;
        unsigned long long m_num_interwarp_coalescing_less_or_equal_than_100_cyc;
        unsigned long long m_num_interwarp_coalescing_less_or_equal_than_200_cyc;
        unsigned long long m_num_interwarp_coalescing_bigger_than_200_cyc;
        unsigned long long m_num_intrawarp_coalescing;
        unsigned long long m_num_intrawarp_coalescing_less_or_equal_than_5_cyc;
        unsigned long long m_num_intrawarp_coalescing_less_or_equal_than_10_cyc;
        unsigned long long m_num_intrawarp_coalescing_less_or_equal_than_20_cyc;
        unsigned long long m_num_intrawarp_coalescing_less_or_equal_than_30_cyc;
        unsigned long long m_num_intrawarp_coalescing_less_or_equal_than_40_cyc;
        unsigned long long m_num_intrawarp_coalescing_less_or_equal_than_50_cyc;
        unsigned long long m_num_intrawarp_coalescing_less_or_equal_than_100_cyc;
        unsigned long long m_num_intrawarp_coalescing_less_or_equal_than_200_cyc;
        unsigned long long m_num_intrawarp_coalescing_bigger_than_200_cyc;
        unsigned long long m_num_total_eval_warp_instructions;
        unsigned long long m_num_total_eval_warp_instructions_with_empty_accesses;
        unsigned long long m_num_total_eval_accesses;
        unsigned long long m_num_not_coalesced;
        
        std::string m_name_space;
        _memory_space_t m_space_type;
        // Key distance of cycles, value number of times
        std::map<unsigned long long, unsigned long long> m_histogram_interwarp_coalescing;
        std::map<unsigned long long, unsigned long long> m_histogram_intrawarp_coalescing;
    
};

class coalescingStatsAcrossSms {
    public:
        coalescingStatsAcrossSms(std::string name_space, _memory_space_t space_type);
        void addStats(coalescingStatsPerSm* coalescing_stats_per_sm);

        unsigned long long m_num_interwarp_coalescing;
        unsigned long long m_num_interwarp_coalescing_less_or_equal_than_5_cyc;
        unsigned long long m_num_interwarp_coalescing_less_or_equal_than_10_cyc;
        unsigned long long m_num_interwarp_coalescing_less_or_equal_than_20_cyc;
        unsigned long long m_num_interwarp_coalescing_less_or_equal_than_30_cyc;
        unsigned long long m_num_interwarp_coalescing_less_or_equal_than_40_cyc;
        unsigned long long m_num_interwarp_coalescing_less_or_equal_than_50_cyc;
        unsigned long long m_num_interwarp_coalescing_less_or_equal_than_100_cyc;
        unsigned long long m_num_interwarp_coalescing_less_or_equal_than_200_cyc;
        unsigned long long m_num_interwarp_coalescing_bigger_than_200_cyc;
        unsigned long long m_num_intrawarp_coalescing;
        unsigned long long m_num_intrawarp_coalescing_less_or_equal_than_5_cyc;
        unsigned long long m_num_intrawarp_coalescing_less_or_equal_than_10_cyc;
        unsigned long long m_num_intrawarp_coalescing_less_or_equal_than_20_cyc;
        unsigned long long m_num_intrawarp_coalescing_less_or_equal_than_30_cyc;
        unsigned long long m_num_intrawarp_coalescing_less_or_equal_than_40_cyc;
        unsigned long long m_num_intrawarp_coalescing_less_or_equal_than_50_cyc;
        unsigned long long m_num_intrawarp_coalescing_less_or_equal_than_100_cyc;
        unsigned long long m_num_intrawarp_coalescing_less_or_equal_than_200_cyc;
        unsigned long long m_num_intrawarp_coalescing_bigger_than_200_cyc;
        unsigned long long m_num_total_eval_warp_instructions;
        unsigned long long m_num_total_eval_warp_instructions_with_empty_accesses;
        unsigned long long m_num_total_eval_accesses;
        unsigned long long m_num_not_coalesced;
        
        std::string m_name_space;
        _memory_space_t m_space_type;
};

class coalescingWarpStats {
    public:
        coalescingWarpStats();
        void addAccess(unsigned int warp_id, new_addr_type global_pc);
        std::map<unsigned int, std::set<new_addr_type>> &getCoalescingWarpStats();

    private:
        // Key warp id
        std::map<unsigned int, std::set<new_addr_type>> m_coalescing_warp_stats;
};

class coalescingCycleHistory {
    
    public:
        coalescingCycleHistory();
        coalescingCycleHistory(coalescingStatsPerSm *coalescing_stats_per_sm);
        // True if there was previously an stored access in the cycle
        void addAccess(unsigned long long cycle, unsigned int warp_id, new_addr_type global_pc);

    private:
        // Key cycle
        std::map<unsigned long long, coalescingWarpStats> m_cycle_access_history;
        coalescingStatsPerSm *m_coalescing_stats_per_sm;
};




class coalescingAddressStats {
    
    public:
        coalescingAddressStats(SM *shared_sm, std::string name_space, _memory_space_t space_type);
        void registerInst(unsigned long long cycle, warp_inst_t *inst);
        coalescingStatsPerSm *getStats();

        void resetHistory();

    private:
        // Key cache block addr
        std::map<new_addr_type, coalescingCycleHistory> m_coalescing_address_stats;
        coalescingStatsPerSm m_coalescing_stats_per_sm;
        SM *m_shared_sm;
};