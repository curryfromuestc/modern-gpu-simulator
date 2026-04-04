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

#include "page_table_walker.h"

PageTableWalker::PageTableWalker() {
    m_id = 0;
    m_is_active = false;
    m_num_pending_cycles = 0;
    m_mf = nullptr;
}

PageTableWalker::PageTableWalker(unsigned int id) {
    m_id = id;
    m_is_active = false;
    m_num_pending_cycles = 0;
    m_mf = nullptr;
}

PageTableWalker::~PageTableWalker() {
    if(m_mf != nullptr) {
        delete m_mf;
    }
}

void PageTableWalker::cycle() {
    if(m_is_active && (m_num_pending_cycles > 0)) {
        m_num_pending_cycles--;
    }
}

void PageTableWalker::set_mf(mem_fetch *mf, unsigned int num_pending_cycles) {
    m_mf = mf;
    m_num_pending_cycles = num_pending_cycles;
    m_is_active = true;
}

mem_fetch *PageTableWalker::get_mf_solved() {
    mem_fetch *mf = m_mf;
    m_mf = nullptr;
    m_is_active = false;
    m_num_pending_cycles = 0;
    return mf;
}

void PageTableWalker::set_id(unsigned int id) {
    m_id = id;
}

bool PageTableWalker::is_ptw_solved() {
    return m_is_active && (m_num_pending_cycles == 0);
}

bool PageTableWalker::is_active() {
    return m_is_active;
}

unsigned int PageTableWalker::get_id() {
    return m_id;
}