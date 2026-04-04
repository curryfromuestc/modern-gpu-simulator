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

#include <vector>
#include <cstdio>

#include "../../abstract_hardware_model.h"

class shader_core_stats;
class shader_core_config;

enum Wait_Barrier_Type {
    READ_WAIT_BARRIER,
    WRITE_WAIT_BARRIER
};

enum Wait_Barrier_Action {
    INCREASE_COUNTER,
    DECREASE_COUNTER
};

struct Wait_Barrier_Entry_Modifier {
    Wait_Barrier_Entry_Modifier(unsigned int sm_warp_id, unsigned int barrier_id, Wait_Barrier_Type barrier_type, Wait_Barrier_Action barrier_action,
                                new_addr_type pc_inst) {
        this->sm_warp_id = sm_warp_id;
        this->barrier_id = barrier_id;
        this->barrier_type = barrier_type;
        this->barrier_action = barrier_action;
        this->pc = pc_inst;
    }

    new_addr_type pc;
    unsigned int sm_warp_id;
    unsigned int barrier_id;
    Wait_Barrier_Type barrier_type;
    Wait_Barrier_Action barrier_action;
};

struct Wait_Barrier_Checking {
    unsigned int barrier_id;
    unsigned int min_val;

    Wait_Barrier_Checking(unsigned int barrier_id, unsigned int min_val) {
        this->barrier_id = barrier_id;
        this->min_val = min_val;
    }
};

class Wait_Barrier {
    public:
        Wait_Barrier(unsigned int barrier_id);
        bool is_ready(unsigned int min_val);
        void reset();
        void decrease_counter();
        void increase_counter();
        unsigned int get_counter();
        unsigned int get_barrier_id();

        void print_state(FILE *out);

    private:
        unsigned int m_counter;
        unsigned int m_barrier_id;
};

class Dependency_State {
    public:
        Dependency_State(const shader_core_config* config, shader_core_stats *stats);

        void cycle();
        
        void reset();

        void set_stall_counter(unsigned int stall_counter);
        void set_yield();

        void action_over_wait_barrier(Wait_Barrier_Entry_Modifier *wait_barrier_entry_modifier);

        bool is_yield_ready();
        bool is_stall_counter_0();
        bool are_wait_barriers_ready(std::vector<Wait_Barrier_Checking> wait_barriers_checking);
        
        void increase_num_pending_ldgsts();
        void decrease_num_pending_ldgsts();
        bool are_ldgsts_pending();

        bool are_pending_dependencies();

        void print_state(FILE *out);

    private:
        unsigned int m_yield;
        unsigned int m_stall_counter;
        unsigned int m_num_pending_ldgsts;
        std::vector<Wait_Barrier> m_wait_barriers;//{0,1,2,3,4,5};
        shader_core_stats* m_stats;
};


