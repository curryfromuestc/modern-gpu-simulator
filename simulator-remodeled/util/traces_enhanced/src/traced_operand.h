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
#include <vector>

#include "traced_constants.h"

#include "JSONBase.h"
#include "JSONIncludes.h"

class traced_operand : public JSONBase {

    public:
        traced_operand(std::string operand_string);

        traced_operand(std::string operand_string, int target_unique_function_id);
        
        traced_operand(); // Used for Deserializing
        void set_operand_string(std::string operand_string);
        std::string get_operand_string() const;
        
        TraceEnhancedOperandType get_operand_type() const;
        TraceEnhancedOperandType get_operand_type(std::string operand_string) const;
        long get_operand_reg_number() const;
        bool get_has_reg() const;
        bool get_is_type_using_only_gathered_registers() const;
        bool get_has_inmediate() const;
        std::vector<double> get_operands_inmediates() const;
        bool is_reuse_bit_set() const;
        bool is_destination() const;

        virtual bool Deserialize(const rapidjson::Value& obj);
        virtual bool Serialize(rapidjson::Writer<rapidjson::StringBuffer>* writer) const;


    private:
        std::string m_operand_string;
        std::vector <std::string> m_modifiers;
        TraceEnhancedOperandType m_operand_type;
        bool m_has_reg;
        bool m_is_negative_reg;
        bool m_is_absolute_reg;
        int m_operand_reg_number;
        
        bool m_has_inmediate;
        std::vector<double> m_operands_inmediates;
        bool m_is_reuse_bit_set;
        

        void set_reg_num(std::string operand_string, int reserved_reg_num, char reserved_reg_char);
        void add_inmediate(std::string operand_string, TraceEnhancedOperandType operand_type);
};