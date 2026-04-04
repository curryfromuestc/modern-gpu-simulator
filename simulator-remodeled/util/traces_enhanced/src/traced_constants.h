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

#define CCPos_arch_7x_8x 41 // Shifting of the architectures from volta to ADA for the second part of the encoded instruction

// COPIED FROM TRACER_TOOL AND NVBIT

/* all supported arch have at most 255 general purpose registers */
constexpr const int RZ = 255;
/* the always true predicate is indicated as "7" on all the archs */
constexpr const int PT = 7;
/* the entire predicate register is encoded as "8" */
constexpr const int PR = 8;
constexpr const int URZ = 63;
constexpr const int UPT = 7;  // uniform predicate true
constexpr const int UPR = 8;  // entire uniform predicate register
constexpr const int MAX_CHARS = 256;

// loads and stores have 1, LDGSTS has 2
constexpr const int MAX_NUM_MREF_PER_INSTR = 2;

enum class TraceEnhancedMemorySpace {
    NONE,
    LOCAL,             // local memory operation
    GENERIC,           // generic memory operation
    GLOBAL,            // global memory operation
    SHARED,            // shared memory operation
    CONSTANT,          // constant memory operation
    GLOBAL_TO_SHARED,  // read from global memory then write to shared memory
    SURFACE,   // surface memory operation
    TEXTURE,   // texture memory operation
};
constexpr const char* TraceEnhancedMemorySpaceStr[] = {
    "NONE", "LOCAL", "GENERIC", "GLOBAL", "SHARED", "CONSTANT",
    "GLOBAL_TO_SHARED", "SURFACE", "TEXTURE",
};

enum class TraceEnhancedOperandType {
    IMM_UINT64,
    IMM_DOUBLE,
    REG,
    PRED,
    UREG,
    UPRED,
    CBANK,
    MREF,
    GENERIC,
    BREG,
    SR,
    SB,
    DESC,
    CALL_TARGET,
    NONE
};

constexpr const char* TraceEnhancedOperandTypeStr[] = {
    "IMM_UINT64", "IMM_DOUBLE", "REG",  "PRED",   "UREG",
    "UPRED",      "CBANK",      "MREF", "GENERIC", "BREG", "SR", "SB", "DESC", "CALL_TARGET", "NONE"};

enum class TraceEnhancedRegModifierType {
    NO_MOD = 0,
    /* stride modifiers */
    X1,
    X4,
    X8,
    X16,
    /* size modifiers */
    U32,
    U64,
};
constexpr const char* TraceEnhancedRegModifierTypeStr[] = {
    "NO_MOD", "X1", "X4", "X8", "X16", "U32", /* no U */ "64"};