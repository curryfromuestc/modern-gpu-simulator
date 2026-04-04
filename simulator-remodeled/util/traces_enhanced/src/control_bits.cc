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

#include "control_bits.h"

control_bits::control_bits(std::string encoded_instruction_part, int shift_to_get_control_codes)
{
    unsigned long long encoded_instruction_part_num = std::stoul(encoded_instruction_part, nullptr, 16);
    unsigned long long control_bits_num = encoded_instruction_part_num >> shift_to_get_control_codes;
    m_stall_count = (control_bits_num & 0x0000f) >> 0;
    m_is_yield = !((control_bits_num & 0x00010) >> 4);
    m_id_new_write_barrier = (control_bits_num & 0x000e0) >> 5;
    m_id_new_read_barrier = (control_bits_num & 0x00700) >> 8;
    m_wait_barrier_bits = (control_bits_num & 0x1f800) >> 11;
    m_is_new_write_barrier = m_id_new_write_barrier != 7;
    m_is_new_read_barrier = m_id_new_read_barrier != 7;
}

control_bits::control_bits()
{
    // Used for Deserializing and for functions not found in the binary

    m_stall_count = 0;
    m_is_yield = false;
    m_is_new_read_barrier = false;
    m_is_new_write_barrier = false;
    m_id_new_read_barrier = 0;
    m_id_new_write_barrier = 0;
    m_wait_barrier_bits = 0;
}

void control_bits::set_is_yield(bool is_yield)
{
    m_is_yield = is_yield;
}

void control_bits::set_stall_count(int stall_count)
{
    m_stall_count = stall_count;
}

void control_bits::set_is_new_read_barrier(bool is_new_read_barrier)
{
    m_is_new_read_barrier = is_new_read_barrier;
}

void control_bits::set_is_new_write_barrier(bool is_new_write_barrier)
{
    m_is_new_write_barrier = is_new_write_barrier;
}

void control_bits::set_id_new_read_barrier(int id_new_read_barrier)
{
    m_id_new_read_barrier = id_new_read_barrier;
}

void control_bits::set_id_new_write_barrier(int id_new_write_barrier)
{
    m_id_new_write_barrier = id_new_write_barrier;
}

void control_bits::set_wait_barrier_bits(int wait_barrier_bits)
{
    m_wait_barrier_bits = wait_barrier_bits;
}

bool control_bits::get_is_yield()
{
    return m_is_yield;
}

int control_bits::get_stall_count()
{
    return m_stall_count;
}

bool control_bits::get_is_new_read_barrier()
{
    return m_is_new_read_barrier;
}

bool control_bits::get_is_new_write_barrier()
{
    return m_is_new_write_barrier;
}

int control_bits::get_id_new_read_barrier()
{
    return m_id_new_read_barrier;
}

int control_bits::get_id_new_write_barrier()
{
    return m_id_new_write_barrier;
}

int control_bits::get_wait_barrier_bits()
{
    return m_wait_barrier_bits;
}

bool control_bits::Serialize(rapidjson::Writer<rapidjson::StringBuffer> *writer) const
{
    writer->StartObject();

    writer->String("stall_count");
    writer->Int(m_stall_count);

    writer->String("is_yield");
    writer->Bool(m_is_yield);

    writer->String("is_new_read_barrier");
    writer->Bool(m_is_new_read_barrier);

    writer->String("is_new_write_barrier");
    writer->Bool(m_is_new_write_barrier);

    writer->String("id_new_read_barrier");
    writer->Int(m_id_new_read_barrier);

    writer->String("id_new_write_barrier");
    writer->Int(m_id_new_write_barrier);

    writer->String("wait_barrier_bits");
    writer->Int(m_wait_barrier_bits);

    writer->EndObject();

    return true;
}

bool control_bits::Deserialize(const rapidjson::Value &obj)
{
    m_stall_count = obj["stall_count"].GetInt();
    set_is_yield(obj["is_yield"].GetBool());
    m_is_new_read_barrier = obj["is_new_read_barrier"].GetBool();
    m_is_new_write_barrier = obj["is_new_write_barrier"].GetBool();
    m_id_new_read_barrier = obj["id_new_read_barrier"].GetInt();
    m_id_new_write_barrier = obj["id_new_write_barrier"].GetInt();
    m_wait_barrier_bits = obj["wait_barrier_bits"].GetInt();
    return true;
}