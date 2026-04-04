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

#include <stdio.h>
#include <stdlib.h>
#include <map>
#include <vector>
#include "assert.h"



#include "../abstract_hardware_model.h"



enum scoreboard_reads_mode
{
  DISABLED = 0,
  RELEASE_AT_WB = 1, // Releases the source registers when the instruction is in the WB stage
  RELEASE_AT_OPC = 2 // Releases the source registers when the instruction leaves the operand collector stage
};

class shader_core_stats;

class Scoreboard_reads {
 public:
  Scoreboard_reads(unsigned sid, unsigned n_warps, class gpgpu_t *gpu, scoreboard_reads_mode mode, unsigned scoreboard_war_max_uses_per_reg, bool is_trace_mode, shader_core_stats *stats);

  void reserveRegisters(const warp_inst_t *inst);
  void reserveRegisters_remodeling(const warp_inst_t *inst);
  void releaseRegisters(const warp_inst_t *inst);
  void releaseRegisters_remodeling(const warp_inst_t *inst);
  void releaseRegister(unsigned wid, unsigned regnum);

  bool checkCollision(unsigned wid, const inst_t *inst) const;
  bool checkCollision_remodeling(unsigned wid, const warp_inst_t *inst) const;
  bool pendingReads(unsigned wid) const;
  void printContents() const;

  bool isEnabled();

  scoreboard_reads_mode getMode();

 private:
  void reserveRegister(unsigned wid, unsigned regnum, const class warp_inst_t* inst);
  unsigned int get_sid() const { return m_sid; }

  unsigned m_sid;

  unsigned int m_scoreboard_war_max_uses_per_reg;

  bool m_enabled;

  bool m_is_trace_mode;

  scoreboard_reads_mode m_mode;

  // keeps track of pending writes to registers
  // indexed by warp id, reg_id => pending write count. They key is the register that is used, the value the number of times that it is in use
  std::vector<std::map<unsigned,unsigned> > reg_table;
  // Register that depend on a long operation (global, local or tex memory)

  class gpgpu_t *m_gpu;

  class shader_core_stats *m_stats;
};