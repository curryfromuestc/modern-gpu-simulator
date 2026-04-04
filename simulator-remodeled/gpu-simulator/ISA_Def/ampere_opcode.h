// developed by Mahmoud Khairy, Purdue Univ
// abdallm@purdue.edu

#ifndef AMPERE_OPCODE_H
#define AMPERE_OPCODE_H

#include <string>
#include <unordered_map>
#include "../gpgpu-sim/src/operation_type.h"
#include "trace_opcode.h"

#define AMPERE_RTX_BINART_VERSION 86
#define AMPERE_A100_BINART_VERSION 80

// TO DO: moving this to a yml or def files

/// Ampere ISA
// see:https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#ampere
static const std::unordered_map<std::string, OpcodeChar> Ampere_OpcodeMap = {
    // Floating Point 32 Instructions
    {"FADD", OpcodeChar(OP_FADD, SP_OP)},
    {"FADD32I", OpcodeChar(OP_FADD32I, SP_OP)},
    {"FCHK", OpcodeChar(OP_FCHK, SP_OP)},
    {"FFMA32I", OpcodeChar(OP_FFMA32I, SP_OP)},
    {"FFMA", OpcodeChar(OP_FFMA, SP_OP)},
    {"FMNMX", OpcodeChar(OP_FMNMX, SP_OP)},
    {"FMUL", OpcodeChar(OP_FMUL, SP_OP)},
    {"FMUL32I", OpcodeChar(OP_FMUL32I, SP_OP)},
    {"FSEL", OpcodeChar(OP_FSEL, SP_OP)},
    {"FSET", OpcodeChar(OP_FSET, SP_OP)},
    {"FSETP", OpcodeChar(OP_FSETP, SP_OP)},
    {"FSWZADD", OpcodeChar(OP_FSWZADD, SP_OP)},
    // SFU
    {"MUFU", OpcodeChar(OP_MUFU, SFU_OP)},

    // Floating Point 16 Instructions
    {"HADD2", OpcodeChar(OP_HADD2, HALF_OP)},
    {"HADD2_32I", OpcodeChar(OP_HADD2_32I, HALF_OP)},
    {"HFMA2", OpcodeChar(OP_HFMA2, HALF_OP)},
    {"HFMA2_32I", OpcodeChar(OP_HFMA2_32I, HALF_OP)},
    {"HMUL2", OpcodeChar(OP_HMUL2, HALF_OP)},
    {"HMUL2_32I", OpcodeChar(OP_HMUL2_32I, HALF_OP)},
    {"HSET2", OpcodeChar(OP_HSET2, HALF_OP)},
    {"HSETP2", OpcodeChar(OP_HSETP2, HALF_OP)},
    {"HMNMX2", OpcodeChar(OP_HMNMX2, HALF_OP)},

    // Tensor Core Instructions
    // Execute Tensor Core Instructions on SPECIALIZED_UNIT_3
    {"HMMA", OpcodeChar(OP_HMMA, TENSOR_CORE_OP)},
    {"DMMA", OpcodeChar(OP_DMMA, DP_OP)},
    {"BMMA", OpcodeChar(OP_BMMA, TENSOR_CORE_OP)},
    {"IMMA", OpcodeChar(OP_IMMA, TENSOR_CORE_OP)},

    // Double Point Instructions
    {"DADD", OpcodeChar(OP_DADD, DP_OP)},
    {"DFMA", OpcodeChar(OP_DFMA, DP_OP)},
    {"DMUL", OpcodeChar(OP_DMUL, DP_OP)},
    {"DSETP", OpcodeChar(OP_DSETP, DP_OP)},

    // Integer Instructions
    {"BMSK", OpcodeChar(OP_BMSK, INTP_OP)},
    {"BREV", OpcodeChar(OP_BREV, INTP_OP)},
    {"FLO", OpcodeChar(OP_FLO, INTP_OP)},
    {"IABS", OpcodeChar(OP_IABS, INTP_OP)},
    {"IADD", OpcodeChar(OP_IADD, INTP_OP)},
    {"IADD3", OpcodeChar(OP_IADD3, INTP_OP)},
    {"IADD32I", OpcodeChar(OP_IADD32I, INTP_OP)},
    {"IDP", OpcodeChar(OP_IDP, INTP_OP)},
    {"IDP4A", OpcodeChar(OP_IDP4A, INTP_OP)},
    {"IMAD", OpcodeChar(OP_IMAD, SP_OP)},
    {"IMNMX", OpcodeChar(OP_IMNMX, INTP_OP)},
    {"IMUL", OpcodeChar(OP_IMUL, INTP_OP)},
    {"IMUL32I", OpcodeChar(OP_IMUL32I, INTP_OP)},
    {"ISCADD", OpcodeChar(OP_ISCADD, INTP_OP)},
    {"ISCADD32I", OpcodeChar(OP_ISCADD32I, INTP_OP)},
    {"ISETP", OpcodeChar(OP_ISETP, INTP_OP)},
    {"LEA", OpcodeChar(OP_LEA, INTP_OP)},
    {"LOP", OpcodeChar(OP_LOP, INTP_OP)},
    {"LOP3", OpcodeChar(OP_LOP3, INTP_OP)},
    {"LOP32I", OpcodeChar(OP_LOP32I, INTP_OP)},
    {"POPC", OpcodeChar(OP_POPC, INTP_OP)},
    {"SHF", OpcodeChar(OP_SHF, INTP_OP)},
    {"SHL", OpcodeChar(OP_SHL, INTP_OP)},  //////////
    {"SHR", OpcodeChar(OP_SHR, INTP_OP)},
    {"VABSDIFF", OpcodeChar(OP_VABSDIFF, INTP_OP)},
    {"VABSDIFF4", OpcodeChar(OP_VABSDIFF4, INTP_OP)},

    // Conversion Instructions
    {"F2FP", OpcodeChar(OP_F2FP, SFU_OP)},
    {"F2F", OpcodeChar(OP_F2F, DP_OP)},
    {"F2I", OpcodeChar(OP_F2I, SFU_OP)},
    {"I2F", OpcodeChar(OP_I2F, SFU_OP)},
    {"I2I", OpcodeChar(OP_I2I, SFU_OP)},
    {"I2IP", OpcodeChar(OP_I2IP, SFU_OP)},
    {"I2FP", OpcodeChar(OP_I2FP, SFU_OP)},
    {"F2IP", OpcodeChar(OP_F2IP, SFU_OP)},
    {"FRND", OpcodeChar(OP_FRND, SFU_OP)},

    // Movement Instructions
    {"MOV", OpcodeChar(OP_MOV, INTP_OP)},
    {"MOV32I", OpcodeChar(OP_MOV32I, INTP_OP)},
    {"MOVM", OpcodeChar(OP_MOVM, TENSOR_CORE_OP)},  // move matrix
    {"PRMT", OpcodeChar(OP_PRMT, INTP_OP)},
    {"SEL", OpcodeChar(OP_SEL, INTP_OP)},
    {"SGXT", OpcodeChar(OP_SGXT, INTP_OP)},
    {"SHFL", OpcodeChar(OP_SHFL, MEMORY_MISCELLANEOUS_OP)},

    // Predicate Instructions
    {"PLOP3", OpcodeChar(OP_PLOP3, PREDICATE_OP)},
    {"PSETP", OpcodeChar(OP_PSETP, PREDICATE_OP)},
    {"P2R", OpcodeChar(OP_P2R, PREDICATE_OP)},
    {"R2P", OpcodeChar(OP_R2P, PREDICATE_OP)},

    // Load/Store Instructions
    {"LD", OpcodeChar(OP_LD, LOAD_OP)},
    // For now, we ignore constant loads, consider it as ALU_OP, TO DO
    {"LDC", OpcodeChar(OP_LDC, LOAD_OP)},
    {"LDG", OpcodeChar(OP_LDG, LOAD_OP)},
    {"LDL", OpcodeChar(OP_LDL, LOAD_OP)},
    {"LDS", OpcodeChar(OP_LDS, LOAD_OP)},
    {"LDSM", OpcodeChar(OP_LDSM, LOAD_OP)},  //
    {"ST", OpcodeChar(OP_ST, STORE_OP)},
    {"STG", OpcodeChar(OP_STG, STORE_OP)},
    {"STL", OpcodeChar(OP_STL, STORE_OP)},
    {"STS", OpcodeChar(OP_STS, STORE_OP)},
    {"MATCH", OpcodeChar(OP_MATCH, MEMORY_MISCELLANEOUS_OP)},
    {"QSPC", OpcodeChar(OP_QSPC, MEMORY_MISCELLANEOUS_OP)},
    {"ATOM", OpcodeChar(OP_ATOM, STORE_OP)},
    {"ATOMS", OpcodeChar(OP_ATOMS, STORE_OP)},
    {"ATOMG", OpcodeChar(OP_ATOMG, STORE_OP)},
    {"RED", OpcodeChar(OP_RED, STORE_OP)},
    {"CCTL", OpcodeChar(OP_CCTL, MEMORY_MISCELLANEOUS_OP)},
    {"CCTLL", OpcodeChar(OP_CCTLL, MEMORY_MISCELLANEOUS_OP)},
    {"ERRBAR", OpcodeChar(OP_ERRBAR, MISCELLANEOUS_QUEUE_OP)},
    {"MEMBAR", OpcodeChar(OP_MEMBAR, MEMORY_BARRIER_OP)},
    {"CCTLT", OpcodeChar(OP_CCTLT, MEMORY_MISCELLANEOUS_OP)},

    {"LDGDEPBAR", OpcodeChar(OP_LDGDEPBAR, LDGDEPBAR_OP)},
    {"LDGSTS", OpcodeChar(OP_LDGSTS, LOAD_OP)},

    // Uniform Datapath Instruction
    // UDP unit
    // for more info about UDP, see
    // https://www.hotchips.org/hc31/HC31_2.12_NVIDIA_final.pdf
    {"R2UR", OpcodeChar(OP_R2UR, INTP_OP)},
    {"S2UR", OpcodeChar(OP_S2UR, MISCELLANEOUS_QUEUE_OP)},
    {"UBMSK", OpcodeChar(OP_UBMSK, UNIFORM_OP)},
    {"UBREV", OpcodeChar(OP_UBREV, UNIFORM_OP)},
    {"UCLEA", OpcodeChar(OP_UCLEA, UNIFORM_OP)},
    {"UF2FP", OpcodeChar(OP_UF2FP, SFU_OP)},
    {"UFLO", OpcodeChar(OP_UFLO, UNIFORM_OP)},
    {"UIADD3", OpcodeChar(OP_UIADD3, UNIFORM_OP)},
    {"UIMAD", OpcodeChar(OP_UIMAD, UNIFORM_OP)},
    {"UISETP", OpcodeChar(OP_UISETP, UNIFORM_OP)},
    {"ULDC", OpcodeChar(OP_ULDC, UNIFORM_OP)},
    {"ULEA", OpcodeChar(OP_ULEA, UNIFORM_OP)},
    {"ULOP", OpcodeChar(OP_ULOP, UNIFORM_OP)},
    {"ULOP3", OpcodeChar(OP_ULOP3, UNIFORM_OP)},
    {"ULOP32I", OpcodeChar(OP_ULOP32I, UNIFORM_OP)},
    {"UMOV", OpcodeChar(OP_UMOV, UNIFORM_OP)},
    {"UP2UR", OpcodeChar(OP_UP2UR, UNIFORM_OP)},
    {"UPLOP3", OpcodeChar(OP_UPLOP3, UNIFORM_OP)},
    {"UPOPC", OpcodeChar(OP_UPOPC, UNIFORM_OP)},
    {"UPRMT", OpcodeChar(OP_UPRMT, UNIFORM_OP)},
    {"UPSETP", OpcodeChar(OP_UPSETP, UNIFORM_OP)},
    {"UR2UP", OpcodeChar(OP_UR2UP, UNIFORM_OP)},
    {"USEL", OpcodeChar(OP_USEL, UNIFORM_OP)},
    {"USGXT", OpcodeChar(OP_USGXT, UNIFORM_OP)},
    {"USHF", OpcodeChar(OP_USHF, UNIFORM_OP)},
    {"USHL", OpcodeChar(OP_USHL, UNIFORM_OP)},
    {"USHR", OpcodeChar(OP_USHR, UNIFORM_OP)},
    {"VOTEU", OpcodeChar(OP_VOTEU, UNIFORM_OP)},

    // Texture Instructions
    // For now, we ignore texture loads, consider it as ALU_OP
    {"TEX", OpcodeChar(OP_TEX, TEXTURE_OP)},
    {"TLD", OpcodeChar(OP_TLD, TEXTURE_OP)},
    {"TLD4", OpcodeChar(OP_TLD4, TEXTURE_OP)},
    {"TMML", OpcodeChar(OP_TMML, TEXTURE_OP)},
    {"TXD", OpcodeChar(OP_TXD, TEXTURE_OP)},
    {"TXQ", OpcodeChar(OP_TXQ, TEXTURE_OP)},

    // Surface Instructions //
    {"SUATOM", OpcodeChar(OP_SUATOM, SURFACE_OP)},
    {"SULD", OpcodeChar(OP_SULD, SURFACE_OP)},
    {"SUQUERY", OpcodeChar(OP_SUQUERY, SURFACE_OP)},
    {"SURED", OpcodeChar(OP_SURED, SURFACE_OP)},
    {"SUST", OpcodeChar(OP_SUST, SURFACE_OP)},

    // Control Instructions
    // execute branch insts on a dedicated branch unit (SPECIALIZED_UNIT_1)
    {"BMOV", OpcodeChar(OP_BMOV, BRANCH_OP)},
    {"BPT", OpcodeChar(OP_BPT, BRANCH_OP)},
    {"BRA", OpcodeChar(OP_BRA, BRANCH_OP)},
    {"BREAK", OpcodeChar(OP_BREAK, BRANCH_OP)},
    {"BRX", OpcodeChar(OP_BRX, BRANCH_OP)},
    {"BRXU", OpcodeChar(OP_BRXU, BRANCH_OP)},  //
    {"BSSY", OpcodeChar(OP_BSSY, BRANCH_OP)},
    {"BSYNC", OpcodeChar(OP_BSYNC, BRANCH_OP)},
    {"CALL", OpcodeChar(OP_CALL, CALL_OPS)},
    {"EXIT", OpcodeChar(OP_EXIT, EXIT_OPS)},
    {"JMP", OpcodeChar(OP_JMP, BRANCH_OP)},
    {"JMX", OpcodeChar(OP_JMX, BRANCH_OP)},
    {"JMXU", OpcodeChar(OP_JMXU, BRANCH_OP)},  ///
    {"KILL", OpcodeChar(OP_KILL, MISCELLANEOUS_NO_QUEUE_OP)},
    {"NANOSLEEP", OpcodeChar(OP_NANOSLEEP, MISCELLANEOUS_NO_QUEUE_OP)},
    {"RET", OpcodeChar(OP_RET, RET_OPS)},
    {"RPCMOV", OpcodeChar(OP_RPCMOV, BRANCH_OP)},
    {"RTT", OpcodeChar(OP_RTT, BRANCH_OP)},
    {"WARPSYNC", OpcodeChar(OP_WARPSYNC, BRANCH_OP)},
    {"YIELD", OpcodeChar(OP_YIELD, BRANCH_OP)},

    // Miscellaneous Instructions
    {"REDUX", OpcodeChar(OP_REDUX, MISCELLANEOUS_NO_QUEUE_OP)},
    {"B2R", OpcodeChar(OP_B2R, MISCELLANEOUS_NO_QUEUE_OP)},
    {"BAR", OpcodeChar(OP_BAR, BARRIER_OP)},
    {"CS2R", OpcodeChar(OP_CS2R, INTP_OP)},
    {"CSMTEST", OpcodeChar(OP_CSMTEST, MISCELLANEOUS_NO_QUEUE_OP)},
    {"DEPBAR", OpcodeChar(OP_DEPBAR, DEPBAR_OP)},
    {"GETLMEMBASE", OpcodeChar(OP_GETLMEMBASE, MISCELLANEOUS_NO_QUEUE_OP)},
    {"LEPC", OpcodeChar(OP_LEPC, MISCELLANEOUS_NO_QUEUE_OP)},
    {"NOP", OpcodeChar(OP_NOP, MISCELLANEOUS_NO_QUEUE_OP)},
    {"PMTRIG", OpcodeChar(OP_PMTRIG, MISCELLANEOUS_NO_QUEUE_OP)},
    {"R2B", OpcodeChar(OP_R2B, MISCELLANEOUS_NO_QUEUE_OP)},
    {"S2R", OpcodeChar(OP_S2R, MEMORY_MISCELLANEOUS_OP)},
    {"SETCTAID", OpcodeChar(OP_SETCTAID, MISCELLANEOUS_NO_QUEUE_OP)},
    {"SETLMEMBASE", OpcodeChar(OP_SETLMEMBASE, MISCELLANEOUS_NO_QUEUE_OP)},
    {"VOTE", OpcodeChar(OP_VOTE, MISCELLANEOUS_NO_QUEUE_OP)},
    {"VOTE_VTG", OpcodeChar(OP_VOTE_VTG, MISCELLANEOUS_NO_QUEUE_OP)},

};

#endif
