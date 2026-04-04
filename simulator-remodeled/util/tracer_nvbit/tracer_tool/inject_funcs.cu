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

#include <cstdarg>
#include <stdint.h>
#include <stdio.h>

#include "utils/utils.h"

/* for channel */
#include "utils/channel.hpp"

/* contains definition of the inst_trace_t structure */
#include "common.h"

#include "../../traces_enhanced/src/string_utilities.h"

/* Instrumentation function that we want to inject, please note the use of
 *  extern "C" __device__ __noinline__
 *    To prevent "dead"-code elimination by the compiler.
 */
extern "C" __device__ __noinline__ void instrument_inst(
    int32_t pred, uint32_t unique_function_id, uint32_t vpc, uint32_t num_of_injects,
    uint32_t per_operand_type, uint32_t memory_type, uint64_t addr_or_offset_or_reg_id,
    uint32_t mem_width_or_op_reg_val_0, uint32_t op_reg_val_1, uint32_t op_reg_val_2,
    uint32_t op_reg_val_3, uint64_t pchannel_dev, uint64_t ptotal_dynamic_instr_counter,
    uint64_t preported_dynamic_instr_counter, uint64_t pstop_report) {

  const int active_mask = __ballot_sync(__activemask(), 1);
  const int predicate_mask = __ballot_sync(__activemask(), pred);
  const int laneid = get_laneid();
  const int first_laneid = __ffs(active_mask) - 1;

  if ((*((bool *)pstop_report))) {
    if (first_laneid == laneid) {
      atomicAdd((unsigned long long *)ptotal_dynamic_instr_counter, 1);
      return;
    }
  }

  inst_trace_t ma;
  ma.ureg_desc_value = SECRET_UREG_DESC_NOT_USED;
  ma.num_of_injects = num_of_injects;
  ma.per_operand_type = per_operand_type;
  if (memory_type == MEM_TYPE::STANDARD_MEM) {
    /* collect memory address information */
    for (int i = 0; i < 32; i++) {
      ma.addrs_or_reg_val_0[i] = __shfl_sync(active_mask, addr_or_offset_or_reg_id, i);
    }
    ma.width = mem_width_or_op_reg_val_0;
    ma.mem_type = MEM_TYPE::STANDARD_MEM;
    if(op_reg_val_1 != SECRET_UREG_DESC_NOT_USED) {
      ma.ureg_desc_value = op_reg_val_1;
    }
  }else if(memory_type == MEM_TYPE::CONSTANT_MEM) {
    ma.mem_type = MEM_TYPE::CONSTANT_MEM;
    uint32_t reg_value = mem_width_or_op_reg_val_0;
    ma.width = 4;
    for (int tid = 0; tid < 32; tid++) {
        ma.addrs_or_reg_val_0[tid] = __shfl_sync(active_mask, reg_value, tid);
    }
  }else if(memory_type == MEM_TYPE::CALL_OR_RET) {
    ma.mem_type = MEM_TYPE::CALL_OR_RET;
    ma.width = 1;
    uint32_t reg_value_1 = mem_width_or_op_reg_val_0;
    uint32_t reg_value_2 = op_reg_val_1;
    for (int tid = 0; tid < 32; tid++) {
        uint32_t th_reg_val1 = __shfl_sync(active_mask, reg_value_1, tid);
        uint32_t th_reg_val2 = __shfl_sync(active_mask, reg_value_2, tid);
        uint64_t final_addr = static_cast<uint64_t>(th_reg_val2) << 32;
        final_addr += th_reg_val1 + addr_or_offset_or_reg_id;
        ma.addrs_or_reg_val_0[tid] = final_addr;
    }
  }else {
    ma.mem_type = MEM_TYPE::NONE;
    ma.width = 0;
    ma.reg_id = addr_or_offset_or_reg_id;
    if((per_operand_type == TRACED_REG_TYPE::REGULAR) || (per_operand_type == TRACED_REG_TYPE::REGULAR_2_REGS) || 
        (per_operand_type == TRACED_REG_TYPE::REGULAR_4_REGS) || (per_operand_type == TRACED_REG_TYPE::PREDICATE)) {
      for (int tid = 0; tid < 32; tid++) {
        ma.addrs_or_reg_val_0[tid] = __shfl_sync(active_mask, mem_width_or_op_reg_val_0, tid);
      }
      if(per_operand_type == TRACED_REG_TYPE::REGULAR_2_REGS) {
        for (int tid = 0; tid < 32; tid++) {
          ma.reg_val_1[tid] = __shfl_sync(active_mask, op_reg_val_1, tid);
        }
      }
      if(per_operand_type == TRACED_REG_TYPE::REGULAR_4_REGS) {
        for (int tid = 0; tid < 32; tid++) {
          ma.reg_val_2[tid] = __shfl_sync(active_mask, op_reg_val_2, tid);
        }
        for (int tid = 0; tid < 32; tid++) {
          ma.reg_val_3[tid] = __shfl_sync(active_mask, op_reg_val_3, tid);
        }
      }
    }else if((per_operand_type == TRACED_REG_TYPE::UNIFORM) || (per_operand_type == TRACED_REG_TYPE::UNIFORM_2_REGS) ||
        (per_operand_type == TRACED_REG_TYPE::UNIFORM_PREDICATE)) {
      for (int tid = 0; tid < 32; tid++) {
        ma.addrs_or_reg_val_0[tid] = mem_width_or_op_reg_val_0;
      }
      if(per_operand_type == TRACED_REG_TYPE::UNIFORM_2_REGS) {
        for (int tid = 0; tid < 32; tid++) {
          ma.reg_val_1[tid] = op_reg_val_1;
        }
      }
    }
  }

  int4 cta = get_ctaid();
  int uniqe_threadId = threadIdx.z * blockDim.y * blockDim.x +
                       threadIdx.y * blockDim.x + threadIdx.x;
  ma.warpid_tb = uniqe_threadId / 32;

  ma.cta_id_x = cta.x;
  ma.cta_id_y = cta.y;
  ma.cta_id_z = cta.z;
  ma.warpid_sm = get_warpid();
  ma.vpc = vpc;
  ma.unique_function_id = unique_function_id;
  ma.active_mask = active_mask;
  ma.predicate_mask = predicate_mask;
  ma.sm_id = get_smid();

  /* first active lane pushes information on the channel */
  if (first_laneid == laneid) {
    ChannelDev *channel_dev = (ChannelDev *)pchannel_dev;
    channel_dev->push(&ma, sizeof(inst_trace_t));
    atomicAdd((unsigned long long *)ptotal_dynamic_instr_counter, 1);
    atomicAdd((unsigned long long *)preported_dynamic_instr_counter, 1);
  }
}
