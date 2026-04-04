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

#include <string>

#include "JSONBase.h"
#include "JSONIncludes.h"

class control_bits : public JSONBase {
    public:

        control_bits(std::string encoded_instruction_part, int shift_to_get_control_codes);
        control_bits(); // Used for Deserializing and for functions not found in the binary
        
        void set_is_yield(bool is_yield);
        void set_stall_count(int stall_count);
        void set_is_new_read_barrier(bool is_new_read_barrier);
        void set_is_new_write_barrier(bool is_new_write_barrier);
        void set_id_new_read_barrier(int id_new_read_barrier);
        void set_id_new_write_barrier(int id_new_write_barrier);
        void set_wait_barrier_bits(int wait_barrier_bits);

        bool get_is_yield();
        int get_stall_count();
        bool get_is_new_read_barrier();
        bool get_is_new_write_barrier();
        int get_id_new_read_barrier();
        int get_id_new_write_barrier();
        int get_wait_barrier_bits();

        virtual bool Deserialize(const rapidjson::Value& obj);
        virtual bool Serialize(rapidjson::Writer<rapidjson::StringBuffer>* writer) const;

    private:
        int m_stall_count;
        bool m_is_yield;
        bool m_is_new_read_barrier;
        bool m_is_new_write_barrier;
        int m_id_new_read_barrier;
        int m_id_new_write_barrier;
        int m_wait_barrier_bits;
};