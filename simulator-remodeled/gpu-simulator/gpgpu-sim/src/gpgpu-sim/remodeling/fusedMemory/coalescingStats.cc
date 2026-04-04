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


#include "coalescingStats.h"


coalescingStatsPerSm::coalescingStatsPerSm(std::string name_space, _memory_space_t space_type) {
    m_name_space = name_space;
    m_num_interwarp_coalescing = 0;
    m_num_intrawarp_coalescing = 0;
    m_num_total_eval_warp_instructions = 0;
    m_num_total_eval_accesses = 0;
    m_num_not_coalesced = 0;
    m_space_type = space_type;
    m_num_interwarp_coalescing_less_or_equal_than_5_cyc = 0;
    m_num_interwarp_coalescing_less_or_equal_than_10_cyc = 0;
    m_num_interwarp_coalescing_less_or_equal_than_20_cyc = 0;
    m_num_interwarp_coalescing_less_or_equal_than_30_cyc = 0;
    m_num_interwarp_coalescing_less_or_equal_than_40_cyc = 0;
    m_num_interwarp_coalescing_less_or_equal_than_50_cyc = 0;
    m_num_interwarp_coalescing_less_or_equal_than_100_cyc = 0;
    m_num_interwarp_coalescing_less_or_equal_than_200_cyc = 0;
    m_num_interwarp_coalescing_bigger_than_200_cyc = 0;
    m_num_intrawarp_coalescing_less_or_equal_than_5_cyc = 0;
    m_num_intrawarp_coalescing_less_or_equal_than_10_cyc = 0;
    m_num_intrawarp_coalescing_less_or_equal_than_20_cyc = 0;
    m_num_intrawarp_coalescing_less_or_equal_than_30_cyc = 0;
    m_num_intrawarp_coalescing_less_or_equal_than_40_cyc = 0;
    m_num_intrawarp_coalescing_less_or_equal_than_50_cyc = 0;
    m_num_intrawarp_coalescing_less_or_equal_than_100_cyc = 0;
    m_num_intrawarp_coalescing_less_or_equal_than_200_cyc = 0;
    m_num_intrawarp_coalescing_bigger_than_200_cyc = 0;
    m_num_total_eval_warp_instructions_with_empty_accesses = 0;
}

coalescingStatsAcrossSms::coalescingStatsAcrossSms(std::string name_space, _memory_space_t space_type) {
    m_name_space = name_space;
    m_num_interwarp_coalescing = 0;
    m_num_intrawarp_coalescing = 0;
    m_num_total_eval_warp_instructions = 0;
    m_num_total_eval_accesses = 0;
    m_num_not_coalesced = 0;
    m_space_type = space_type;
    m_num_interwarp_coalescing_less_or_equal_than_5_cyc = 0;
    m_num_interwarp_coalescing_less_or_equal_than_10_cyc = 0;
    m_num_interwarp_coalescing_less_or_equal_than_20_cyc = 0;
    m_num_interwarp_coalescing_less_or_equal_than_30_cyc = 0;
    m_num_interwarp_coalescing_less_or_equal_than_40_cyc = 0;
    m_num_interwarp_coalescing_less_or_equal_than_50_cyc = 0;
    m_num_interwarp_coalescing_less_or_equal_than_100_cyc = 0;
    m_num_interwarp_coalescing_less_or_equal_than_200_cyc = 0;
    m_num_interwarp_coalescing_bigger_than_200_cyc = 0;
    m_num_intrawarp_coalescing_less_or_equal_than_5_cyc = 0;
    m_num_intrawarp_coalescing_less_or_equal_than_10_cyc = 0;
    m_num_intrawarp_coalescing_less_or_equal_than_20_cyc = 0;
    m_num_intrawarp_coalescing_less_or_equal_than_30_cyc = 0;
    m_num_intrawarp_coalescing_less_or_equal_than_40_cyc = 0;
    m_num_intrawarp_coalescing_less_or_equal_than_50_cyc = 0;
    m_num_intrawarp_coalescing_less_or_equal_than_100_cyc = 0;
    m_num_intrawarp_coalescing_less_or_equal_than_200_cyc = 0;
    m_num_intrawarp_coalescing_bigger_than_200_cyc = 0;
    m_num_total_eval_warp_instructions_with_empty_accesses = 0;
}

void coalescingStatsAcrossSms::addStats(coalescingStatsPerSm *coalescing_stats_per_sm) {
    m_num_interwarp_coalescing += coalescing_stats_per_sm->m_num_interwarp_coalescing;
    m_num_intrawarp_coalescing += coalescing_stats_per_sm->m_num_intrawarp_coalescing;
    m_num_total_eval_warp_instructions += coalescing_stats_per_sm->m_num_total_eval_warp_instructions;
    m_num_total_eval_accesses += coalescing_stats_per_sm->m_num_total_eval_accesses;
    m_num_not_coalesced += coalescing_stats_per_sm->m_num_not_coalesced;
    m_num_interwarp_coalescing_less_or_equal_than_5_cyc += coalescing_stats_per_sm->m_num_interwarp_coalescing_less_or_equal_than_5_cyc;
    m_num_interwarp_coalescing_less_or_equal_than_10_cyc += coalescing_stats_per_sm->m_num_interwarp_coalescing_less_or_equal_than_10_cyc;
    m_num_interwarp_coalescing_less_or_equal_than_20_cyc += coalescing_stats_per_sm->m_num_interwarp_coalescing_less_or_equal_than_20_cyc;
    m_num_interwarp_coalescing_less_or_equal_than_30_cyc += coalescing_stats_per_sm->m_num_interwarp_coalescing_less_or_equal_than_30_cyc;
    m_num_interwarp_coalescing_less_or_equal_than_40_cyc += coalescing_stats_per_sm->m_num_interwarp_coalescing_less_or_equal_than_40_cyc;
    m_num_interwarp_coalescing_less_or_equal_than_50_cyc += coalescing_stats_per_sm->m_num_interwarp_coalescing_less_or_equal_than_50_cyc;
    m_num_interwarp_coalescing_less_or_equal_than_100_cyc += coalescing_stats_per_sm->m_num_interwarp_coalescing_less_or_equal_than_100_cyc;
    m_num_interwarp_coalescing_less_or_equal_than_200_cyc += coalescing_stats_per_sm->m_num_interwarp_coalescing_less_or_equal_than_200_cyc;
    m_num_interwarp_coalescing_bigger_than_200_cyc += coalescing_stats_per_sm->m_num_interwarp_coalescing_bigger_than_200_cyc;
    m_num_intrawarp_coalescing_less_or_equal_than_5_cyc += coalescing_stats_per_sm->m_num_intrawarp_coalescing_less_or_equal_than_5_cyc;
    m_num_intrawarp_coalescing_less_or_equal_than_10_cyc += coalescing_stats_per_sm->m_num_intrawarp_coalescing_less_or_equal_than_10_cyc;
    m_num_intrawarp_coalescing_less_or_equal_than_20_cyc += coalescing_stats_per_sm->m_num_intrawarp_coalescing_less_or_equal_than_20_cyc;
    m_num_intrawarp_coalescing_less_or_equal_than_30_cyc += coalescing_stats_per_sm->m_num_intrawarp_coalescing_less_or_equal_than_30_cyc;
    m_num_intrawarp_coalescing_less_or_equal_than_40_cyc += coalescing_stats_per_sm->m_num_intrawarp_coalescing_less_or_equal_than_40_cyc;
    m_num_intrawarp_coalescing_less_or_equal_than_50_cyc += coalescing_stats_per_sm->m_num_intrawarp_coalescing_less_or_equal_than_50_cyc;
    m_num_intrawarp_coalescing_less_or_equal_than_100_cyc += coalescing_stats_per_sm->m_num_intrawarp_coalescing_less_or_equal_than_100_cyc;
    m_num_intrawarp_coalescing_less_or_equal_than_200_cyc += coalescing_stats_per_sm->m_num_intrawarp_coalescing_less_or_equal_than_200_cyc;
    m_num_intrawarp_coalescing_bigger_than_200_cyc += coalescing_stats_per_sm->m_num_intrawarp_coalescing_bigger_than_200_cyc;
    m_num_total_eval_warp_instructions_with_empty_accesses += coalescing_stats_per_sm->m_num_total_eval_warp_instructions_with_empty_accesses;
}


coalescingWarpStats::coalescingWarpStats() {}

std::map<unsigned int, std::set<new_addr_type>>  &coalescingWarpStats::getCoalescingWarpStats() {
    return m_coalescing_warp_stats;
}

void coalescingWarpStats::addAccess(unsigned int warp_id, new_addr_type global_pc) {
    if(m_coalescing_warp_stats.find(warp_id) == m_coalescing_warp_stats.end()) {
        m_coalescing_warp_stats[warp_id] = std::set<new_addr_type>();
    }
    m_coalescing_warp_stats[warp_id].insert(global_pc);
}


coalescingCycleHistory::coalescingCycleHistory() {}

coalescingCycleHistory::coalescingCycleHistory(coalescingStatsPerSm *coalescing_stats_per_sm) {
    m_coalescing_stats_per_sm = coalescing_stats_per_sm;
}

void coalescingCycleHistory::addAccess(unsigned long long cycle, unsigned int warp_id, new_addr_type global_pc) {
    unsigned long long min_distance_cycle_intrawarp = std::numeric_limits<unsigned long long>::max();
    unsigned long long min_distance_cycle_interwarp = std::numeric_limits<unsigned long long>::max();
    bool found_intrawarp = false;
    bool found_interwarp = false;
    for(auto it = m_cycle_access_history.begin(); it != m_cycle_access_history.end(); it++) {
        coalescingWarpStats *warp_stats = &it->second;
        unsigned long long distance = cycle - it->first;

        for(auto it2 = warp_stats->getCoalescingWarpStats().begin(); it2 != warp_stats->getCoalescingWarpStats().end(); it2++) {
            if(it2->first == warp_id) {
                if(distance < min_distance_cycle_intrawarp) {
                    min_distance_cycle_intrawarp = distance;
                    found_intrawarp = true;
                }
            } else {
                if(distance < min_distance_cycle_interwarp) {
                    min_distance_cycle_interwarp = distance;
                    found_interwarp = true;
                }
            }
        }
    }

    if(found_interwarp) {
        m_coalescing_stats_per_sm->m_num_interwarp_coalescing++;
        m_coalescing_stats_per_sm->m_histogram_interwarp_coalescing[min_distance_cycle_interwarp]++;
        if(min_distance_cycle_interwarp <=5 ) {
            m_coalescing_stats_per_sm->m_num_interwarp_coalescing_less_or_equal_than_5_cyc++;
        } else if(min_distance_cycle_interwarp <= 10) {
            m_coalescing_stats_per_sm->m_num_interwarp_coalescing_less_or_equal_than_10_cyc++;
        } else if(min_distance_cycle_interwarp <= 20) {
            m_coalescing_stats_per_sm->m_num_interwarp_coalescing_less_or_equal_than_20_cyc++;
        } else if(min_distance_cycle_interwarp <= 30) {
            m_coalescing_stats_per_sm->m_num_interwarp_coalescing_less_or_equal_than_30_cyc++;
        } else if(min_distance_cycle_interwarp <= 40) {
            m_coalescing_stats_per_sm->m_num_interwarp_coalescing_less_or_equal_than_40_cyc++;
        } else if(min_distance_cycle_interwarp <= 50) {
            m_coalescing_stats_per_sm->m_num_interwarp_coalescing_less_or_equal_than_50_cyc++;
        } else if(min_distance_cycle_interwarp <= 100) {
            m_coalescing_stats_per_sm->m_num_interwarp_coalescing_less_or_equal_than_100_cyc++;
        } else if(min_distance_cycle_interwarp <= 200) {
            m_coalescing_stats_per_sm->m_num_interwarp_coalescing_less_or_equal_than_200_cyc++;
        } else {
            m_coalescing_stats_per_sm->m_num_interwarp_coalescing_bigger_than_200_cyc++;
        }
    }

    if(found_intrawarp) {
        m_coalescing_stats_per_sm->m_num_intrawarp_coalescing++;
        m_coalescing_stats_per_sm->m_histogram_intrawarp_coalescing[min_distance_cycle_intrawarp]++;
        if(min_distance_cycle_intrawarp <= 5) {
            m_coalescing_stats_per_sm->m_num_intrawarp_coalescing_less_or_equal_than_5_cyc++;
        }else if(min_distance_cycle_intrawarp <= 10) {
            m_coalescing_stats_per_sm->m_num_intrawarp_coalescing_less_or_equal_than_10_cyc++;
        }else if(min_distance_cycle_intrawarp <= 20) {
            m_coalescing_stats_per_sm->m_num_intrawarp_coalescing_less_or_equal_than_20_cyc++;
        }else if(min_distance_cycle_intrawarp <= 30) {
            m_coalescing_stats_per_sm->m_num_intrawarp_coalescing_less_or_equal_than_30_cyc++;
        }else if(min_distance_cycle_intrawarp <= 40) {
            m_coalescing_stats_per_sm->m_num_intrawarp_coalescing_less_or_equal_than_40_cyc++;
        } else if(min_distance_cycle_intrawarp <= 50) {
            m_coalescing_stats_per_sm->m_num_intrawarp_coalescing_less_or_equal_than_50_cyc++;
        } else if(min_distance_cycle_intrawarp <= 100) {
            m_coalescing_stats_per_sm->m_num_intrawarp_coalescing_less_or_equal_than_100_cyc++;
        } else if(min_distance_cycle_intrawarp <= 200) {
            m_coalescing_stats_per_sm->m_num_intrawarp_coalescing_less_or_equal_than_200_cyc++;
        } else {
            m_coalescing_stats_per_sm->m_num_intrawarp_coalescing_bigger_than_200_cyc++;
        }
    }

    if(!found_interwarp && !found_intrawarp) {
        m_coalescing_stats_per_sm->m_num_not_coalesced++;
    }


    m_coalescing_stats_per_sm->m_num_total_eval_accesses++;


    if(m_cycle_access_history.find(cycle) == m_cycle_access_history.end()) {
        m_cycle_access_history[cycle] = coalescingWarpStats();
    }
    m_cycle_access_history[cycle].addAccess(warp_id, global_pc);
}

coalescingAddressStats::coalescingAddressStats(SM *shared_sm, std::string name_space, _memory_space_t space_type) : m_coalescing_stats_per_sm(name_space, space_type) {
    m_shared_sm = shared_sm;
}

void coalescingAddressStats::registerInst(unsigned long long cycle, warp_inst_t *inst) {
    if(!inst->accessq_empty()) {
        std::vector<mem_access_t> acc_inst = inst->get_mem_accesses();
        m_coalescing_stats_per_sm.m_num_total_eval_warp_instructions++;
        for(auto it = acc_inst.begin(); it != acc_inst.end(); it++) {
            new_addr_type block_addr = it->get_addr();
            if(m_coalescing_address_stats.find(block_addr) == m_coalescing_address_stats.end()) {
                m_coalescing_address_stats[block_addr] = coalescingCycleHistory(&m_coalescing_stats_per_sm);
            }
            new_addr_type global_pc = m_shared_sm->from_local_pc_to_global_pc_address(inst->pc, inst->unique_function_id);
            m_coalescing_address_stats[block_addr].addAccess(cycle, inst->warp_id(), global_pc);
        }
    }else {
        m_coalescing_stats_per_sm.m_num_total_eval_warp_instructions_with_empty_accesses++;
    }
}


coalescingStatsPerSm *coalescingAddressStats::getStats() {
    return &m_coalescing_stats_per_sm;
}

void coalescingAddressStats::resetHistory() {
    m_coalescing_address_stats.clear();
}