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

#include "register_usage.h"

register_usage::register_usage(unsigned num_regular, unsigned num_uniform, unsigned num_regular_predicate, unsigned num_uniform_predicate)
{
    m_num_regular = num_regular;
    m_num_uniform = num_uniform;
    m_num_regular_predicate = num_regular_predicate;
    m_num_uniform_predicate = num_uniform_predicate;
    m_future_max_num_regular = 0;
    m_future_max_num_uniform = 0;
    m_future_max_num_regular_predicate = 0;
    m_future_max_num_uniform_predicate = 0;
}

register_usage::register_usage()
{
    // Used for Deserializing
}

unsigned register_usage::get_num_regular_registers() const
{
    return m_num_regular;
}

unsigned register_usage::get_num_uniform_registers() const
{
    return m_num_uniform;
}

unsigned register_usage::get_num_regular_predicate_registers() const
{
    return m_num_regular_predicate;
}

unsigned register_usage::get_num_uniform_predicate_registers() const
{
    return m_num_uniform_predicate;
}

unsigned register_usage::get_future_max_num_regular_registers() const
{
    return m_future_max_num_regular;
}

unsigned register_usage::get_future_max_num_uniform_registers() const
{
    return m_future_max_num_uniform;
}

unsigned register_usage::get_future_max_num_regular_predicate_registers() const
{
    return m_future_max_num_regular_predicate;
}

unsigned register_usage::get_future_max_num_uniform_predicate_registers() const
{
    return m_future_max_num_uniform_predicate;
}

void register_usage::set_num_regular_registers(unsigned num_regular)
{
    m_num_regular = num_regular;
}

void register_usage::set_num_uniform_registers(unsigned num_uniform)
{
    m_num_uniform = num_uniform;
}

void register_usage::set_num_regular_predicate_registers(unsigned num_regular_predicate)
{
    m_num_regular_predicate = num_regular_predicate;
}

void register_usage::set_num_uniform_predicate_registers(unsigned num_uniform_predicate)
{
    m_num_uniform_predicate = num_uniform_predicate;
}

void register_usage::set_future_max_num_regular_registers(unsigned future_max_num_regular)
{
    m_future_max_num_regular = future_max_num_regular;
}

void register_usage::set_future_max_num_uniform_registers(unsigned future_max_num_uniform)
{
    m_future_max_num_uniform = future_max_num_uniform;
}

void register_usage::set_future_max_num_regular_predicate_registers(unsigned future_max_num_regular_predicate)
{
    m_future_max_num_regular_predicate = future_max_num_regular_predicate;
}

void register_usage::set_future_max_num_uniform_predicate_registers(unsigned future_max_num_uniform_predicate)
{
    m_future_max_num_uniform_predicate = future_max_num_uniform_predicate;
}

unsigned register_usage::get_total_num_registers() const
{
    return m_num_regular + m_num_uniform + m_num_regular_predicate + m_num_uniform_predicate;
}

unsigned register_usage::get_total_future_max_num_registers() const
{
    return m_future_max_num_regular + m_future_max_num_uniform + m_future_max_num_regular_predicate + m_future_max_num_uniform_predicate;
}

bool register_usage::Serialize(rapidjson::Writer<rapidjson::StringBuffer> *writer) const
{
    writer->StartObject();

    writer->String("num_regular");
    writer->Int64(m_num_regular);

    writer->String("num_uniform");
    writer->Int64(m_num_uniform);

    writer->String("num_regular_predicate");
    writer->Int64(m_num_regular_predicate);

    writer->String("num_uniform_predicate");
    writer->Int64(m_num_uniform_predicate);

    writer->String("future_max_num_regular");
    writer->Int64(m_future_max_num_regular);

    writer->String("future_max_num_uniform");
    writer->Int64(m_future_max_num_uniform);

    writer->String("future_max_num_regular_predicate");
    writer->Int64(m_future_max_num_regular_predicate);

    writer->String("future_max_num_uniform_predicate");
    writer->Int64(m_future_max_num_uniform_predicate);

    writer->EndObject();

    return true;
}

bool register_usage::Deserialize(const rapidjson::Value &obj)
{
    set_num_regular_registers(obj["num_regular"].GetInt64());
    set_num_uniform_registers(obj["num_uniform"].GetInt64());
    set_num_regular_predicate_registers(obj["num_regular_predicate"].GetInt64());
    set_num_uniform_predicate_registers(obj["num_uniform_predicate"].GetInt64());
    set_future_max_num_regular_registers(obj["future_max_num_regular"].GetInt64());
    set_future_max_num_uniform_registers(obj["future_max_num_uniform"].GetInt64());
    set_future_max_num_regular_predicate_registers(obj["future_max_num_regular_predicate"].GetInt64());
    set_future_max_num_uniform_predicate_registers(obj["future_max_num_uniform_predicate"].GetInt64());

    return true;
}