// Copyright (c) 2021-2023, Mojtaba Abaie and Rodrigo Huerta
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
// The University of British Columbia nor the names of its contributors may be
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

#include <bitset>
#include <vector>

#include "../abstract_hardware_model.h"

class opndcoll_rfu_t;

static const unsigned MAX_ALU_LATENCY = 512;


class ResultBus {
    public:
        ResultBus(unsigned int max_allowed_wb_ports_rf);
        void cycle();
        bool test(unsigned latency) const;
        void set(unsigned latency);

    private:
        unsigned int m_max_allowed_wb_ports_rf;
        std::vector<unsigned int> m_pipelined_latency;
};

class ResultBusses {
public:
    ~ResultBusses();
    void init(unsigned width, unsigned num_banks, opndcoll_rfu_t *rf);
    void cycle();
    int test(const warp_inst_t *inst);
private:
    void find_reg_banks(const warp_inst_t *inst, int &regbank1, int &regbank2) const;
    unsigned num_free_slots(unsigned latency) const;

    unsigned m_width; //max writebacks = m_width
    unsigned m_num_banks; // #RF_banks
    opndcoll_rfu_t *m_rf; //reference to register file

    std::vector<ResultBus*> m_res_busses;
};