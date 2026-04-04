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

#include "traced_operand.h"

#include <iostream>
#include <cmath>

#include "string_utilities.h"

traced_operand::traced_operand(std::string operand_string)
{
    m_is_reuse_bit_set = false;
    m_has_inmediate = false;
    m_has_reg = false;

    bool is_other_wait_barriers_identifiers = false;

    std::vector<std::string> splitted_operand_string = split_string(operand_string, '.');

    for (unsigned i = 1; i < splitted_operand_string.size(); i++)
    {
        std::string current_modifier = splitted_operand_string[i];
        std::size_t plus_pos = current_modifier.find("+");
        if (std::string::npos != plus_pos)
        {
            current_modifier = current_modifier.substr(0, plus_pos);
        }
        if (current_modifier.find("reuse") != std::string::npos)
        {
            m_is_reuse_bit_set = true;
        }
        current_modifier = ReplaceAll(current_modifier, "]", "");
        m_modifiers.push_back(current_modifier);
    }

    m_operand_string = operand_string;

    unsigned idx_first_reg_num = 0;

    m_is_negative_reg = false;
    m_is_absolute_reg = false;
    if ((operand_string[0] == '-') && ((operand_string[1] == 'R') || (operand_string[1] == 'U') || (operand_string[1] == 'P') || (operand_string[1] == 'c')))
    {
        idx_first_reg_num++;
        m_is_negative_reg = true;
    }
    else if ((operand_string[0] == '|') && ((operand_string[1] == 'R') || (operand_string[1] == 'U') || (operand_string[1] == 'P') || (operand_string[1] == 'c')))
    {
        idx_first_reg_num++;
        m_is_absolute_reg = true;
    }else if ((operand_string[0] == '~') && ((operand_string[1] == 'R') || (operand_string[1] == 'U') || (operand_string[1] == 'P') || (operand_string[1] == 'c')))
    {
        idx_first_reg_num++;
    }else if( (operand_string[0] == '-') && (operand_string[1] == '|') && ((operand_string[2] == 'R') || (operand_string[2] == 'U') || (operand_string[2] == 'P') || (operand_string[1] == 'c'))){
        idx_first_reg_num += 2;
        m_is_negative_reg = true;
        m_is_absolute_reg = true;
    }

    bool skip_idx_inc = false;

    if (operand_string[idx_first_reg_num] == 'R')
    {
        m_operand_type = TraceEnhancedOperandType::REG;
    }
    else if ((operand_string[idx_first_reg_num] == 'P') || (operand_string[idx_first_reg_num] == '!') || (operand_string.find("UP") != std::string::npos))
    {
        if (operand_string[idx_first_reg_num] == '!')
        {
            idx_first_reg_num++;
            m_is_negative_reg = true;
        }
        if (operand_string.find("UP") != std::string::npos)
        {
            m_operand_type = TraceEnhancedOperandType::UPRED;
            idx_first_reg_num++;
        }
        else
        {
            m_operand_type = TraceEnhancedOperandType::PRED;
        }
    }
    else if (operand_string[idx_first_reg_num] == 'B')
    {
        m_operand_type = TraceEnhancedOperandType::BREG;
    }
    else if (operand_string[idx_first_reg_num] == 'c')
    {
        m_operand_type = TraceEnhancedOperandType::CBANK;
    }
    else if (operand_string[idx_first_reg_num] == '[')
    {
        m_operand_type = TraceEnhancedOperandType::MREF;
    }
    else if (operand_string.find("SR") != std::string::npos)
    {
        m_operand_type = TraceEnhancedOperandType::SR;
        idx_first_reg_num++;
    }
    else if (operand_string.find("SB") != std::string::npos)
    {
        m_operand_type = TraceEnhancedOperandType::SB;
        idx_first_reg_num++;
    }else if(operand_string.find("{") != std::string::npos) {
        m_operand_type = TraceEnhancedOperandType::SB;
        is_other_wait_barriers_identifiers = true;
    }
    else if (operand_string.find("desc") != std::string::npos)
    {
        m_operand_type = TraceEnhancedOperandType::DESC;
        idx_first_reg_num += 3;
    }
    else if (operand_string.find("UR") != std::string::npos)
    {
        m_operand_type = TraceEnhancedOperandType::UREG;
        idx_first_reg_num++;
    }
    else
    {
        skip_idx_inc = true;
        if (operand_string.find("0x") == std::string::npos)
        {
            try
            {
                if (operand_string.find("QNAN") != std::string::npos)
                {
                    operand_string = ReplaceAll(operand_string, "QNAN", "NAN");
                }
                std::stod(operand_string);
                m_operand_type = TraceEnhancedOperandType::IMM_DOUBLE;
            }
            catch (const std::invalid_argument &ia)
            {
                std::cerr << "ERROR: Operand type not recognized: " << operand_string << std::endl;
                fflush(stderr);
                abort();
            }
        }
        else
        {
            m_operand_type = TraceEnhancedOperandType::IMM_UINT64;
        }
    }
    if (!skip_idx_inc)
    {
        idx_first_reg_num++;
    }

    // std::string number_string = operand_string.substr(idx_first_reg_num, operand_string.size() - idx_first_reg_num);
    std::string number_string = operand_string.substr(idx_first_reg_num);
    number_string = ReplaceAll(number_string, "|", "");
    number_string = ReplaceAll(number_string, "~", "");

    m_operand_reg_number = 0;
    switch (m_operand_type)
    {
    case TraceEnhancedOperandType::SB:
        if(is_other_wait_barriers_identifiers) {
            number_string = ReplaceAll(number_string, "{", "");
            number_string = ReplaceAll(number_string, "}", "");
            std::vector<std::string> splitted_operands = split_string(number_string, ',');
            for(unsigned i = 0; i < splitted_operands.size(); i++) {
                add_inmediate(splitted_operands[i], TraceEnhancedOperandType::IMM_DOUBLE);
            }
        }else{
            set_reg_num(number_string, 99999, '-');
        }
        break;
    case TraceEnhancedOperandType::BREG:
        set_reg_num(number_string, 99999, '-');
        break;
    case TraceEnhancedOperandType::UREG:
        set_reg_num(number_string, URZ, 'Z');
        break;
    case TraceEnhancedOperandType::REG:
        set_reg_num(number_string, RZ, 'Z');
        break;
    case TraceEnhancedOperandType::UPRED:
        set_reg_num(number_string, UPT, 'T');
        break;
    case TraceEnhancedOperandType::PRED:
        set_reg_num(number_string, PT, 'T');
        break;
    case TraceEnhancedOperandType::IMM_UINT64:
        add_inmediate(number_string, TraceEnhancedOperandType::IMM_UINT64);
        break;
    case TraceEnhancedOperandType::IMM_DOUBLE:
        add_inmediate(number_string, TraceEnhancedOperandType::IMM_DOUBLE);
        break;
    case TraceEnhancedOperandType::MREF:
    {
        std::string cleaned_operands = ReplaceAll(number_string, "]", "");
        std::vector<std::string> splitted_operands = split_string(cleaned_operands, '+');
        for (unsigned i = 0; i < splitted_operands.size(); i++)
        {
            if ((splitted_operands[i][0] == 'R') || (splitted_operands[i][0] == 'U'))
            {
                int reserved_reg = RZ;
                int first_substring_pos = 1;
                int search_dot = splitted_operands[i].find_first_of('.');
                int las_substring_pos = search_dot == -1 ? (splitted_operands[i].size() - 1) : (search_dot - 1);
                if (splitted_operands[i][0] == 'U')
                {
                    reserved_reg = URZ;
                    first_substring_pos++;
                }
                if (search_dot == -1)
                {
                    fflush(stdout);
                }
                std::string reg_string = splitted_operands[i].substr(first_substring_pos, las_substring_pos);
                set_reg_num(reg_string, reserved_reg, 'Z');
            }
            else
            {
                add_inmediate(splitted_operands[i], TraceEnhancedOperandType::IMM_DOUBLE);
            }
        }
    }
    break;
    case TraceEnhancedOperandType::CBANK:
    {
        std::string cleaned_operands = ReplaceAll(number_string, "]", "");
        std::vector<std::string> splitted_operands = split_string(cleaned_operands, '[');
        for (unsigned i = 0; i < splitted_operands.size(); i++)
        {
            std::vector<std::string> splitted_operands2 = split_string(splitted_operands[i], '+');
            for (unsigned j = 0; j < splitted_operands2.size(); j++)
            {
                if ((splitted_operands2[j][0] == 'R') || (splitted_operands2[j][0] == 'U'))
                {
                    int reserved_reg = RZ;
                    int first_substring_pos = 1;
                    int search_dot = splitted_operands2[j].find_first_of('.');
                    int las_substring_pos = search_dot == -1 ? (splitted_operands2[j].size() - 1) : (search_dot - 1);
                    if (splitted_operands2[j][0] == 'U')
                    {
                        reserved_reg = URZ;
                        first_substring_pos++;
                    }
                    if (search_dot == -1)
                    {
                        fflush(stdout);
                    }
                    std::string reg_string = splitted_operands2[j].substr(first_substring_pos, las_substring_pos);
                    set_reg_num(reg_string, reserved_reg, 'Z');
                }
                else
                {
                    add_inmediate(splitted_operands2[j], TraceEnhancedOperandType::IMM_DOUBLE);
                }
            }
        }
    }
    break;
    default:
        break;
    }
}

traced_operand::traced_operand(std::string operand_string, int target_unique_function_id) {
    m_operand_string = operand_string;
    m_operand_reg_number = target_unique_function_id;
    m_operand_type = TraceEnhancedOperandType::CALL_TARGET;
    m_has_reg = false; // We treated as false because it is not a true Register what we store here
    m_has_inmediate = false;
    m_is_reuse_bit_set = false;
    m_is_negative_reg = false;
    m_is_absolute_reg = false;
    m_modifiers.clear();
    m_operands_inmediates.clear();
}

traced_operand::traced_operand() {
    // Used for Deserializing
}

void traced_operand::set_reg_num(std::string operand_string, int reserved_reg_num, char reserved_reg_char)
{
    if (operand_string[0] == reserved_reg_char)
    {
        m_operand_reg_number = reserved_reg_num;
    }
    else if (reserved_reg_char == 'T' && operand_string[0] == 'R')
    {
        m_operand_reg_number = PR;
    }
    else
    {
        m_operand_reg_number = std::stoul(operand_string, nullptr, 10);
    }
    m_has_reg = true;
}

void traced_operand::add_inmediate(std::string operand_string, TraceEnhancedOperandType operand_type)
{
    double inmediate;
    if (operand_type == TraceEnhancedOperandType::IMM_DOUBLE)
    {
        inmediate = std::stod(operand_string);
    }
    else
    {
        inmediate = static_cast<double>(std::stoul(operand_string, nullptr, 16));
    }
    m_operands_inmediates.push_back(inmediate);
    m_has_inmediate = true;
}

TraceEnhancedOperandType traced_operand::get_operand_type() const
{
    return m_operand_type;
}

TraceEnhancedOperandType traced_operand::get_operand_type(std::string operand_string) const
{
    if (operand_string == "IMM_UINT64")
    {
        return TraceEnhancedOperandType::IMM_UINT64;
    }
    else if (operand_string == "IMM_DOUBLE")
    {
        return TraceEnhancedOperandType::IMM_DOUBLE;
    }
    else if (operand_string == "REG")
    {
        return TraceEnhancedOperandType::REG;
    }
    else if (operand_string == "PRED")
    {
        return TraceEnhancedOperandType::PRED;
    }
    else if (operand_string == "UREG")
    {
        return TraceEnhancedOperandType::UREG;
    }
    else if (operand_string == "UPRED")
    {
        return TraceEnhancedOperandType::UPRED;
    }
    else if (operand_string == "CBANK")
    {
        return TraceEnhancedOperandType::CBANK;
    }
    else if (operand_string == "MREF")
    {
        return TraceEnhancedOperandType::MREF;
    }
    else if (operand_string == "GENERIC")
    {
        return TraceEnhancedOperandType::GENERIC;
    }
    else if (operand_string == "BREG")
    {
        return TraceEnhancedOperandType::BREG;
    }
    else if (operand_string == "SR")
    {
        return TraceEnhancedOperandType::SR;
    }
    else if (operand_string == "SB")
    {
        return TraceEnhancedOperandType::SB;
    }
    else if (operand_string == "DESC")
    {
        return TraceEnhancedOperandType::DESC;
    }else if(operand_string == "CALL_TARGET") {
        return TraceEnhancedOperandType::CALL_TARGET;
    }else
    {
        // Handle unknown operand types
        throw std::invalid_argument("Unknown operand type: " + operand_string);
    }
}

std::string traced_operand::get_operand_string() const
{
    return m_operand_string;
}

void traced_operand::set_operand_string(std::string operand_string)
{
    m_operand_string = operand_string;
}

long traced_operand::get_operand_reg_number() const
{
    return m_operand_reg_number;
}

bool traced_operand::get_has_reg() const
{
    return m_has_reg;
}

bool traced_operand::get_is_type_using_only_gathered_registers() const {
    bool res = false;
    if((m_operand_type == TraceEnhancedOperandType::REG) || (m_operand_type == TraceEnhancedOperandType::UREG) ||
        (m_operand_type == TraceEnhancedOperandType::UPRED) || (m_operand_type == TraceEnhancedOperandType::PRED)) {
        res = true;
    }
    return res;
}

bool traced_operand::get_has_inmediate() const
{
    return m_has_inmediate;
}

std::vector<double> traced_operand::get_operands_inmediates() const
{
    return m_operands_inmediates;
}

bool traced_operand::is_reuse_bit_set() const
{
    return m_is_reuse_bit_set;
}

bool traced_operand::Serialize(rapidjson::Writer<rapidjson::StringBuffer> *writer) const
{
    writer->StartObject();

    writer->String("operand_string");
    writer->String(m_operand_string.c_str());

    writer->String("operand_type");
    writer->String(TraceEnhancedOperandTypeStr[static_cast<int>(m_operand_type)]);

    writer->String("has_reg");
    writer->Bool(m_has_reg);

    writer->String("is_negative_reg");
    writer->Bool(m_is_negative_reg);

    writer->String("is_absolute_reg");
    writer->Bool(m_is_absolute_reg);

    writer->String("operand_reg_number");
    writer->Int(m_operand_reg_number);

    writer->String("has_inmediate");
    writer->Bool(m_has_inmediate);

    writer->String("operand_inmmediates");
    writer->StartArray();
    for (unsigned i = 0; i < m_operands_inmediates.size(); i++)
    {
        writer->Double(m_operands_inmediates[i]);
    }
    writer->EndArray();

    writer->String("is_reuse_bit_set");
    writer->Bool(m_is_reuse_bit_set);

    writer->String("operand_modifiers");
    writer->StartArray();
    for (unsigned i = 0; i < m_modifiers.size(); i++)
    {
        writer->String(m_modifiers[i].c_str());
    }
    writer->EndArray();

    writer->EndObject();

    return true;
}

bool traced_operand::Deserialize(const rapidjson::Value &obj)
{
    m_operand_string = obj["operand_string"].GetString();

    const rapidjson::Value &modifiers = obj["operand_modifiers"];
    for (rapidjson::SizeType i = 0; i < modifiers.Size(); i++)
    {
        m_modifiers.push_back(modifiers[i].GetString());
    }

    m_operand_type = get_operand_type(obj["operand_type"].GetString());
    m_has_reg = obj["has_reg"].GetBool();
    m_is_negative_reg = obj["is_negative_reg"].GetBool();
    m_is_absolute_reg = obj["is_absolute_reg"].GetBool();
    m_operand_reg_number = obj["operand_reg_number"].GetInt();

    m_has_inmediate = obj["has_inmediate"].GetBool();

    const rapidjson::Value &operands_inmediates = obj["operand_inmmediates"];
    for (rapidjson::SizeType i = 0; i < operands_inmediates.Size(); i++)
    {
        m_operands_inmediates.push_back(operands_inmediates[i].GetDouble());
    }

    m_is_reuse_bit_set = obj["is_reuse_bit_set"].GetBool();

    return true;
}