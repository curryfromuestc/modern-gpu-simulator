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

#include "gmmu.h"

GMMU::GMMU(unsigned int num_ptw_solvers, unsigned int num_cycles_to_solve_walk, unsigned int max_size_fifo_in, unsigned int max_size_fifo_out) {
    m_num_ptw_solvers = num_ptw_solvers;
    m_num_cycles_to_solve_walk = num_cycles_to_solve_walk;
    m_max_size_fifo_in = max_size_fifo_in;
    m_max_size_fifo_out = max_size_fifo_out;
    m_ptw_solvers.resize(num_ptw_solvers);
    for(unsigned int i = 0; i < num_ptw_solvers; i++) {
        m_ptw_solvers[i].set_id(i);
    }
    m_fifo_in = std::make_shared<std::queue<mem_fetch *>>();
}

GMMU::~GMMU() {
    m_fifo_in = nullptr;
}

bool GMMU::can_accept_mf() {
    return m_fifo_in->size() < m_max_size_fifo_in;
}

bool GMMU::is_fifo_out_with_space() {
    return m_fifo_out.size() < m_max_size_fifo_out;
}

void GMMU::accept_mf(mem_fetch *mf) {
    m_fifo_in->push(mf);
}

int GMMU::ptw_free_slot() {
    int ptw_id = -1;
    for(unsigned int i = 0; (i < m_num_ptw_solvers) && (ptw_id == -1); i++) {
        if(!m_ptw_solvers[i].is_active()) {
            ptw_id = i;
        }
    }
    return ptw_id;
}

void GMMU::assign_mf_to_ptw(mem_fetch *mf, int ptw_id) {
    m_ptw_solvers[ptw_id].set_mf(mf, m_num_cycles_to_solve_walk);
}

bool GMMU::can_pop_out_mf() {
    return !m_fifo_out.empty();
}

mem_fetch *GMMU::pop_out_mf() {
    mem_fetch *mf = m_fifo_out.front();
    m_fifo_out.pop();
    return mf;
}

void GMMU::cycle() {
    for(unsigned int i = 0; i < m_num_ptw_solvers; i++) {
        if(is_fifo_out_with_space() && m_ptw_solvers[i].is_ptw_solved()) {
            mem_fetch *mf = m_ptw_solvers[i].get_mf_solved();
            m_fifo_out.push(mf);
        }
        m_ptw_solvers[i].cycle();
        if(!m_ptw_solvers[i].is_active() && !m_fifo_in->empty()) {
            mem_fetch *mf = m_fifo_in->front();
            m_ptw_solvers[i].set_mf(mf, m_num_cycles_to_solve_walk);
            m_fifo_in->pop();
        }
    }
}

std::shared_ptr<std::queue<mem_fetch *>> GMMU::get_fifo_in() {
    return m_fifo_in;
}

unsigned int GMMU::get_max_size_fifo_in() {
    return m_max_size_fifo_in;
}

unsigned int GMMU::get_max_size_fifo_out() {
    return m_max_size_fifo_out;
}