// Copyright (c) 2021-2022, Rodrigo Huerta
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
// The University of British Columbia nor the names of its contributors may be
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

// File used to do not duplicate somo constant declarations in many files.

#ifndef CONSTANTS_H
#define CONSTANTS_H

#define MAX(a, b) (((a) > (b)) ? (a) : (b))

const unsigned WARP_PER_CTA_MAX = 128; // MOD. Allowing more warps per SM

#define MAX_DST 1
#define MAX_SRC 5 // MOD. Fix tensor Turing. Previously was 4
#define FIRST_PRED_REG 10000 // MOD. Predication. P0
#define LAST_PRED_REG 10008 // MOD. Predication. PT is P7 which is 10007. PR is P8 which is 10008
#define RESERVED_PT 10007 // MOD. VPREG
#define RESERVED_RZ 256 // MOD. VPREG
#define BANK_ID_PREDICATE_REGS_TO_DETECT_SKIP 100000 // MOD. Predication
#define MAX_TRACE_REG_ID 10009 // MOD. Predication. 1 More than the last pred reg in order to allow LOOG structures to do not go out of bounds

#define PROGRAM_MEM_START                                      \
  0xF0000000 /* should be distinct from other memory spaces... \
                check ptx_ir.h to verify this does not overlap \
                other memory spaces */

#define PROGRAM_OFFSET_START 0xFFFFF // MOD. Instruction addresses of different kernels have a different address request in memory

#define MAX_CONSTANT_REGION_SIZE 0xFFFF
#define FIRST_CONSTANT_CACHE_ADDR 0xEEEEEEEE
#define LAST_CONSTANT_CACHE_ADDR 0xEFFFFFFF

#endif  // CONSTANTS_H