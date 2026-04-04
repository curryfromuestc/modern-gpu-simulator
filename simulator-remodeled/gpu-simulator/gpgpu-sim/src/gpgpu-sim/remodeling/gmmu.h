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

#include <queue>

#include "page_table_walker.h"

class GMMU {
public:
    GMMU(unsigned int num_ptw_solvers, unsigned int num_cycles_to_solve_walk, unsigned int max_size_fifo_in, unsigned int max_size_fifo_out);
    ~GMMU();
    void cycle();
    unsigned int get_num_ptw_solvers();
    unsigned int get_max_size_fifo_in();
    unsigned int get_max_size_fifo_out();
    int ptw_free_slot();
    void assign_mf_to_ptw(mem_fetch *mf, int ptw_id);
    bool can_accept_mf();
    bool can_pop_out_mf();
    mem_fetch *pop_out_mf();
    void accept_mf(mem_fetch *mf);
    std::shared_ptr<std::queue<mem_fetch *>> get_fifo_in();

private:
    bool is_fifo_out_with_space();
    unsigned int m_num_ptw_solvers;
    unsigned int m_num_cycles_to_solve_walk;
    unsigned int m_max_size_fifo_in;
    unsigned int m_max_size_fifo_out;
    std::vector<PageTableWalker> m_ptw_solvers;
    std::shared_ptr<std::queue<mem_fetch *>> m_fifo_in;
    std::queue<mem_fetch *> m_fifo_out;
};