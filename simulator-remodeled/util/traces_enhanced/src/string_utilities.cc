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


#include <cassert>
#include <regex>
#include <sstream>
#include <iomanip>

#include "string_utilities.h"


std::string ReplaceAll(std::string str, const std::string& from, const std::string& to) {
    size_t start_pos = 0;
    while((start_pos = str.find(from, start_pos)) != std::string::npos) {
        str.replace(start_pos, from.length(), to);
        start_pos += to.length(); // Handles case where 'to' is a substring of 'from'
    }
    return str;
}

std::string strip_string(const std::string &inpt)
{
    auto start_it = inpt.begin();
    auto end_it = inpt.rbegin();
    while (std::isspace(*start_it))
        ++start_it;
    while (std::isspace(*end_it))
        ++end_it;
    return std::string(start_it, end_it.base());
}

std::vector<std::string> split_string(const std::string& str, char delim) {
    std::vector<std::string> strings;
    size_t start;
    size_t end = 0;
    while ((start = str.find_first_not_of(delim, end)) != std::string::npos) {
        end = str.find(delim, start);
        strings.push_back(str.substr(start, end - start));
    }
    return strings;
}

std::string create_sass_instr(std::string full_instruction_str, bool is_substring_delimeter, std::string substring_delimeter) {
    const char *const delim = " ";
    std::size_t first_idx_sass = full_instruction_str.find_first_of(' ');
    std::size_t last_idx_sass;
    if(is_substring_delimeter) {
        last_idx_sass = full_instruction_str.find(substring_delimeter);
        assert(last_idx_sass != std::string::npos);
    } else {
        last_idx_sass = full_instruction_str.find_last_of(delim);
    }
    std::string sass_instr = strip_string(full_instruction_str.substr(first_idx_sass, last_idx_sass - 4));
    return sass_instr;
}

unsigned int get_ur_register(std::string operand_str) {
    std::regex e ("UR(\\d+)");
    std::smatch match;
    unsigned int res = 0;
    if (std::regex_search(operand_str, match, e) && match.size() > 1) {
        res = std::stoul(match.str(1));
    }
    return res;
}



std::string decimalToHexString(unsigned int num, int num_min_digits) {
    std::stringstream stream;
    stream << std::setfill('0') << std::setw(num_min_digits) << std::hex << num;
    return stream.str();
}

bool endsWith(const std::string& str, const std::string& suffix) {
  if (str.length() >= suffix.length()) {
      return (0 == str.compare(str.length() - suffix.length(), suffix.length(), suffix));
  } else {
      return false;
  }
}

ThreadblockStringParseInfo parse_tb_string_id(const std::string &tb_string_id) {
  ThreadblockStringParseInfo result;

  // Find positions of key markers
  size_t d_pos = tb_string_id.find("d_");
  size_t s_pos = tb_string_id.find("s_");
  size_t k_pos = tb_string_id.find("k_");
  size_t underscore_after_k = tb_string_id.find("_", k_pos + 2);

  if (d_pos != std::string::npos && s_pos != std::string::npos &&
      k_pos != std::string::npos && underscore_after_k != std::string::npos)
  {

    // Extract substrings containing just the numbers
    std::string device_str = tb_string_id.substr(d_pos + 2, s_pos - d_pos - 3);
    std::string stream_str = tb_string_id.substr(s_pos + 2, k_pos - s_pos - 3);
    std::string kernel_str = tb_string_id.substr(k_pos + 2, underscore_after_k - k_pos - 2);

    // Convert to integers
    result.device_id = std::stoi(device_str);
    result.stream_id = std::stoi(stream_str);
    result.kernel_id = std::stoi(kernel_str);
  }

  return result;
}

std::vector<std::string> get_opcode_tokens(std::string opcode) {
  std::istringstream iss(opcode);
  std::vector<std::string> opcode_tokens;
  std::string token;
  while (std::getline(iss, token, '.')) {
    if (!token.empty()) opcode_tokens.push_back(token);
  }
  return opcode_tokens;
}