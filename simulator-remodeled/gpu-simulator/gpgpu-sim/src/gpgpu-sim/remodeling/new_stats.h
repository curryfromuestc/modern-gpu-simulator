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

#include <cstdio>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <typeindex>
#include <type_traits>
#include <variant>

enum class AllowedTypesStats {
    UNSIGNED_LONG_LONG,
    DOUBLE
};

class Single_stat_abstract {
public:
    virtual ~Single_stat_abstract() = default;
    virtual std::string get_name() = 0;
    virtual std::string get_between_name_and_value() = 0;
    virtual std::string get_suffix() = 0;
    virtual bool get_is_erase_after_gather_in_sm() = 0;
    virtual bool get_is_reset_allowed() = 0;
    // Add a pure virtual function to get the value
    virtual unsigned long long get_value() const = 0; // VER COMO HACER
    // virtual double get_value() const = 0;             // VER COMO HACER
    virtual void increment_with_integer(int increment_val) = 0;
    virtual void increment_with_double(double increment_val) = 0;
    virtual AllowedTypesStats get_allowed_type() = 0;
    virtual std::string AllowedTypesStats_to_string(AllowedTypesStats type) = 0;
    virtual void reset_if_allowed() = 0;
    virtual void reset() = 0;
    virtual void print(FILE *fout) = 0;
};

class Single_stat_base : public Single_stat_abstract {
public:
    Single_stat_base(AllowedTypesStats allowed_type, std::string name, std::string between_name_and_value, std::string suffix, bool is_reset_allowed, bool is_erase_after_gather_in_sm, bool is_sm_stat)
        : m_allowed_type(allowed_type), m_name(name), m_between_name_and_value(between_name_and_value), m_suffix(suffix),
        m_is_erase_after_gather_in_sm(is_erase_after_gather_in_sm), m_is_reset_allowed(is_reset_allowed),  m_is_sm_stat(is_sm_stat) {}
    
    std::string get_name() override { return m_name; }
    std::string get_between_name_and_value() override { return m_between_name_and_value; }
    std::string get_suffix() override { return m_suffix; }
    bool get_is_erase_after_gather_in_sm() override { return m_is_erase_after_gather_in_sm; }
    bool get_is_reset_allowed() override { return m_is_reset_allowed; }

    AllowedTypesStats get_allowed_type() override { return m_allowed_type; }

    std::string AllowedTypesStats_to_string(AllowedTypesStats type) override {
        switch(type) {
            case AllowedTypesStats::UNSIGNED_LONG_LONG:
                return "UNSIGNED_LONG_LONG";
            case AllowedTypesStats::DOUBLE:
                return "DOUBLE";
            default: 
                return "UNKNOWN";
        }
    }

protected:
    AllowedTypesStats m_allowed_type;
    std::string m_name;
    std::string m_between_name_and_value;
    std::string m_suffix;
    bool m_is_erase_after_gather_in_sm;
    bool m_is_reset_allowed;
    bool m_is_sm_stat;
};

class Single_stat_unsigned_long_long : public Single_stat_base {
public:
    Single_stat_unsigned_long_long(AllowedTypesStats allowed_type, unsigned long long start_value, std::string name, std::string between_name_and_value, std::string suffix, bool is_reset_allowed, bool is_erase_after_gather_in_sm, bool is_sm_stat)
        : Single_stat_base(allowed_type, name, between_name_and_value, suffix, is_reset_allowed, is_erase_after_gather_in_sm, is_sm_stat), m_value(start_value) {}
    
    unsigned long long get_value() const override { return m_value; }
    
    void increment_with_integer(int increment_val) override {
        m_value += increment_val;
    }

    void increment_with_double(double increment_val) override {
        m_value += increment_val;
    }

    void reset_if_allowed() override {
        if(m_is_reset_allowed) {
            m_value = 0;
        }
    }

    void reset() override {
        m_value = 0;
    }

    void print(FILE *fout) {
        std::stringstream ss;
        ss << m_name << " " << m_between_name_and_value << m_value << " " << m_suffix;
        std::string str = ss.str();
        fprintf(fout, "%s\n", str.c_str());
    }
private:
    unsigned long long m_value;
};


class Single_stat_double : public Single_stat_base {
public:
    Single_stat_double(AllowedTypesStats allowed_type, double start_value, std::string name, std::string between_name_and_value, std::string suffix, bool is_reset_allowed, bool is_erase_after_gather_in_sm, bool is_sm_stat)
        : Single_stat_base(allowed_type, name, between_name_and_value, suffix, is_reset_allowed, is_erase_after_gather_in_sm, is_sm_stat), m_value(start_value) {}
    
    unsigned long long get_value() const override { return m_value; }
    
    void increment_with_integer(int increment_val) override {
        m_value += increment_val;
    }

    void increment_with_double(double increment_val) override {
        m_value += increment_val;
    }

    void reset_if_allowed() override {
        if(m_is_reset_allowed) {
            m_value = 0;
        }
    }

    void reset() override {
        m_value = 0;
    }

    void print(FILE *fout) {
        std::stringstream ss;
        ss << m_name << " " << m_between_name_and_value << m_value << " " << m_suffix;
        std::string str = ss.str();
        fprintf(fout, "%s\n", str.c_str());
    }
private:
    double m_value;
};

class Element_stats {
public:
    Element_stats(std::string name) : m_name(name) {}

    void add_unsigned_long_long_stat(std::string name, AllowedTypesStats allowed_type, unsigned long long start_value, std::string between_name_and_value, std::string suffix, bool is_reset_allowed, bool m_is_erase_after_gather_in_sm, bool m_is_sm_stat) {
        assert(allowed_type == AllowedTypesStats::UNSIGNED_LONG_LONG);
        auto stat = std::make_shared<Single_stat_unsigned_long_long>(allowed_type, start_value, name, between_name_and_value, suffix, is_reset_allowed, m_is_erase_after_gather_in_sm, m_is_sm_stat);
        m_stats_map[name] = stat;
        m_stats_name.push_back(name);
    }

    void add_double_stat(std::string name, AllowedTypesStats allowed_type, double start_value, std::string between_name_and_value, std::string suffix, bool is_reset_allowed, bool m_is_erase_after_gather_in_sm, bool m_is_sm_stat) {
        assert(allowed_type == AllowedTypesStats::DOUBLE);
        auto stat = std::make_shared<Single_stat_double>(allowed_type, start_value, name, between_name_and_value, suffix, is_reset_allowed, m_is_erase_after_gather_in_sm, m_is_sm_stat);
        m_stats_map[name] = stat;
        m_stats_name.push_back(name);
    }

    void reset_stat(std::string name) {
        m_stats_map[name]->reset_if_allowed();
    }

    std::vector<std::string> m_stats_name;
    std::map<std::string, std::shared_ptr<Single_stat_abstract>> m_stats_map;
    std::map<std::string, std::type_index> m_stats_type_map;
    std::string m_name;
};