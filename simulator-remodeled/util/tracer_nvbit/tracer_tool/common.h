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

#include <stdint.h>

static __managed__ uint64_t total_dynamic_instr_counter = 0;
static __managed__ uint64_t reported_dynamic_instr_counter = 0;
/* information collected in the instrumentation function and passed
 * on the channel from the GPU to the CPU */
#define MAX_SRC 5

enum MEM_TYPE { NONE = 0, STANDARD_MEM = 1, CONSTANT_MEM = 2, CALL_OR_RET = 3 };
enum TRACED_REG_TYPE {NO_REGS = 0, MEMORY_REF = 1, REGULAR = 2, REGULAR_2_REGS = 3, REGULAR_4_REGS = 4, UNIFORM = 5, UNIFORM_2_REGS = 6, PREDICATE = 7, UNIFORM_PREDICATE = 8};

typedef struct {
  int cta_id_x;
  int cta_id_y;
  int cta_id_z;
  int warpid_tb;
  int warpid_sm;
  int sm_id;
  uint64_t addrs_or_reg_val_0[32];
  uint32_t reg_val_1[32];
  uint32_t reg_val_2[32];
  uint32_t reg_val_3[32];
  uint32_t reg_id;
  uint32_t vpc;
  uint32_t unique_function_id;
  MEM_TYPE mem_type;
  int32_t width;
  uint32_t ureg_desc_value;
  uint32_t active_mask;
  uint32_t predicate_mask;
  uint32_t num_of_injects;
  uint32_t per_operand_type;
} inst_trace_t;
