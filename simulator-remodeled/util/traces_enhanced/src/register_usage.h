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


#include "JSONBase.h"
#include "JSONIncludes.h"

class register_usage {
    public:
        register_usage(unsigned num_regular, unsigned num_uniform, unsigned num_regular_predicate, unsigned num_uniform_predicate);
        register_usage(); // Used for Deserializing
        
        unsigned get_num_regular_registers() const;
        unsigned get_num_uniform_registers() const;
        unsigned get_num_regular_predicate_registers() const;
        unsigned get_num_uniform_predicate_registers() const;

        unsigned get_future_max_num_regular_registers() const;
        unsigned get_future_max_num_uniform_registers() const;
        unsigned get_future_max_num_regular_predicate_registers() const;
        unsigned get_future_max_num_uniform_predicate_registers() const;

        void set_num_regular_registers(unsigned num_regular);
        void set_num_uniform_registers(unsigned num_uniform);
        void set_num_regular_predicate_registers(unsigned num_regular_predicate);
        void set_num_uniform_predicate_registers(unsigned num_uniform_predicate);

        void set_future_max_num_regular_registers(unsigned future_max_num_regular);
        void set_future_max_num_uniform_registers(unsigned future_max_num_uniform);
        void set_future_max_num_regular_predicate_registers(unsigned future_max_num_regular_predicate);
        void set_future_max_num_uniform_predicate_registers(unsigned future_max_num_uniform_predicate);
        
        unsigned get_total_num_registers() const;
        unsigned get_total_future_max_num_registers() const;

        virtual bool Deserialize(const rapidjson::Value& obj);
        virtual bool Serialize(rapidjson::Writer<rapidjson::StringBuffer>* writer) const;

    private:
        unsigned m_num_regular;
        unsigned m_num_uniform;
        unsigned m_num_regular_predicate;
        unsigned m_num_uniform_predicate;

        unsigned m_future_max_num_regular;
        unsigned m_future_max_num_uniform;
        unsigned m_future_max_num_regular_predicate;
        unsigned m_future_max_num_uniform_predicate;
};