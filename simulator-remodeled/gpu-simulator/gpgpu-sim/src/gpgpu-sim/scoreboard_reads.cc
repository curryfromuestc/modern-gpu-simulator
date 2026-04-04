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


#include "scoreboard_reads.h"
#include "../cuda-sim/ptx_sim.h"
#include "shader.h"
#include "shader_trace.h"
#include "remodeling/sm.h"


// Constructor
Scoreboard_reads::Scoreboard_reads(unsigned sid, unsigned n_warps, class gpgpu_t* gpu, scoreboard_reads_mode mode, unsigned scoreboard_war_max_uses_per_reg, bool is_trace_mode, shader_core_stats *stats)
{
  m_sid = sid;
  // Initialize size of table
  reg_table.resize(n_warps);
  m_gpu = gpu;
  m_scoreboard_war_max_uses_per_reg = scoreboard_war_max_uses_per_reg;
  m_mode = mode;
  m_enabled = mode != scoreboard_reads_mode::DISABLED;
  m_is_trace_mode = is_trace_mode;
  m_stats = stats;
}

// Print scoreboard contents
void Scoreboard_reads::printContents() const {
  printf("Scoreboard_reads:  contents (sid=%d): \n", m_sid);
  for (unsigned i = 0; i < reg_table.size(); i++) {
    if (reg_table[i].size() == 0) continue;
    printf("  wid = %2d: ", i);
    std::map<unsigned int ,unsigned int>::const_iterator it;
    for (it = reg_table[i].begin(); it != reg_table[i].end(); it++)
      printf("Reg: %u. Pending uses: %u ", it->first, it->second);
    printf("\n");
  }
}

void Scoreboard_reads::reserveRegister(unsigned wid, unsigned regnum, const class warp_inst_t* inst) {
  std::map<unsigned,unsigned>::const_iterator it = reg_table[wid].find(regnum);
  if(it == reg_table[wid].end())
  {
    reg_table[wid][regnum] = 1;
  }else {
    unsigned current_uses = it->second;
    if(current_uses == m_scoreboard_war_max_uses_per_reg)
    {
      printContents();
      inst->print(stdout);
      printf(
        "Scoreboard_reads: Error: trying to reserve a register more times than the allowed (sid=%d, "
        "wid=%d, regnum=%d).",
        m_sid, wid, regnum);
        fflush(stdout);
        abort();
    }
    reg_table[wid][regnum] = current_uses+1;
  }
  SHADER_DPRINTF(SCOREBOARD, "Scoreboard_reads: Reserved Register - warp:%d, reg: %d\n", wid,
                 regnum);
}

// Unmark one use of the register
void Scoreboard_reads::releaseRegister(unsigned wid, unsigned regnum) {
  if (!(reg_table[wid].find(regnum) != reg_table[wid].end())) return;
  SHADER_DPRINTF(SCOREBOARD, "Scoreboard_reads: Release one register use - warp:%d, reg: %d\n", wid,
                 regnum);
  unsigned current_uses = reg_table[wid][regnum];
  assert(current_uses != 0);
  reg_table[wid].erase(regnum);
  if(current_uses>1)
  {
    reg_table[wid][regnum] = current_uses-1;
  }
}


void Scoreboard_reads::reserveRegisters(const class warp_inst_t* inst) {

}


void Scoreboard_reads::reserveRegisters_remodeling(const class warp_inst_t* inst) {
  if(m_enabled) {
    std::set<int> regs_ins_reserve;
    unsigned int num_dsts = inst->get_extra_trace_instruction_info().get_num_destination_registers();
    for(unsigned int i = num_dsts; i < inst->get_extra_trace_instruction_info().get_num_operands(); i++) {
      traced_operand& op = inst->get_extra_trace_instruction_info().get_operand(i);
      TraceEnhancedOperandType op_type = get_reg_type_eval(op);
      if(op.get_has_reg() && !check_is_reserved_regs_remodeling(op.get_operand_reg_number(), op_type, m_is_trace_mode)) {
        for(unsigned int j = 0; j < get_number_of_uses_per_operand(inst->get_extra_trace_instruction_info(), op.get_operand_reg_number(), i, op_type); j++) {
          // Add offsets to the reserved registers in case of not being regular regs
          unsigned int final_reg_id = translate_reg_to_global_id(op.get_operand_reg_number(), op_type) +j;
          regs_ins_reserve.insert(final_reg_id);
        }
      }
    }

    std::set<int>::const_iterator it;
    for(it = regs_ins_reserve.begin(); it != regs_ins_reserve.end();it++)
    {
      reserveRegister(inst->warp_id(), *it ,inst);
      SHADER_DPRINTF(SCOREBOARD, "Scoreboard_reads: Reserved register - warp:%d, reg: %d\n",
                      inst->warp_id(), *it); 
    }
  }
}

void Scoreboard_reads::releaseRegisters_remodeling(const class warp_inst_t* inst) {
  if(m_enabled) {
    std::set<int> regs_ins_reserve;
    unsigned int num_dsts = inst->get_extra_trace_instruction_info().get_num_destination_registers();
    for(unsigned int i = num_dsts; i < inst->get_extra_trace_instruction_info().get_num_operands(); i++) {
      traced_operand& op = inst->get_extra_trace_instruction_info().get_operand(i);
      TraceEnhancedOperandType op_type = get_reg_type_eval(op);
      if(op.get_has_reg() && !check_is_reserved_regs_remodeling(op.get_operand_reg_number(), op_type, m_is_trace_mode)) {
        for(unsigned int j = 0; j < get_number_of_uses_per_operand(inst->get_extra_trace_instruction_info(), op.get_operand_reg_number(), i, op_type); j++) {
          // Add offsets to the reserved registers in case of not being regular regs
          unsigned int final_reg_id = translate_reg_to_global_id(op.get_operand_reg_number(), op_type) +j;
          regs_ins_reserve.insert(final_reg_id);
        }
      }
    }

    std::set<int>::const_iterator it;
    for(it = regs_ins_reserve.begin(); it != regs_ins_reserve.end();it++)
    {
      SHADER_DPRINTF(SCOREBOARD, "Scoreboard_reads: Register Released - warp:%d, reg: %d\n",
                      inst->warp_id(),*it);
      releaseRegister(inst->warp_id(), *it);
    }
  }
}

// Release registers for an instruction
void Scoreboard_reads::releaseRegisters(const class warp_inst_t* inst) {
}


bool Scoreboard_reads::pendingReads(unsigned wid) const {
  return !reg_table[wid].empty();
}


/**
 * Checks to see if destination register has conflicts with previous read registers .
 * Also checks if the maximum number of input registers is less than the allowed
 *
 * @return
 * true if WAR hazard or reached maximum number of uses for a read register
 **/
bool Scoreboard_reads::checkCollision(unsigned wid, const class inst_t* inst) const {

  return false;
}


bool Scoreboard_reads::checkCollision_remodeling(unsigned wid, const class warp_inst_t* inst) const {
  if(m_enabled) {
    m_stats->num_scoreboard_reads_check_collision++;
    // Check that the destination register of the new instruction is not pending to be read 
    std::set<int> inst_regs_out;

    unsigned int num_dsts = inst->get_extra_trace_instruction_info().get_num_destination_registers();

    for (unsigned int iii = 0; iii < num_dsts; iii++)
    {
      traced_operand& op = inst->get_extra_trace_instruction_info().get_operand(iii);
      TraceEnhancedOperandType op_type = get_reg_type_eval(op);
      if(op.get_has_reg() && !check_is_reserved_regs_remodeling(op.get_operand_reg_number(), op_type, m_is_trace_mode)) {
        for(unsigned int j = 0; j < get_number_of_uses_per_operand(inst->get_extra_trace_instruction_info(), op.get_operand_reg_number(), iii, op_type); j++) {
          // Add offsets to the reserved registers in case of not being regular regs
          unsigned int final_reg_id = translate_reg_to_global_id(op.get_operand_reg_number(), op_type) +j;
          inst_regs_out.insert(final_reg_id);
        }
      }
    }

    std::set<int>::const_iterator it;
    for (it = inst_regs_out.begin(); it != inst_regs_out.end(); it++)
    {
        if (reg_table[wid].find(*it) != reg_table[wid].end()) {
          return true;
        }
    }

    // Check that input registers, predication regs and ar regs don't reach a maximum number of uses
    std::set<int> inst_regs_in;

    for (unsigned int iii = num_dsts; iii < inst->get_extra_trace_instruction_info().get_num_operands(); iii++)
    {
      traced_operand& op = inst->get_extra_trace_instruction_info().get_operand(iii);
      TraceEnhancedOperandType op_type = get_reg_type_eval(op);
      if(op.get_has_reg() && !check_is_reserved_regs_remodeling(op.get_operand_reg_number(), op_type, m_is_trace_mode)) {
        for(unsigned int j = 0; j < get_number_of_uses_per_operand(inst->get_extra_trace_instruction_info(), op.get_operand_reg_number(), iii, op_type); j++) {
          // Add offsets to the reserved registers in case of not being regular regs
          unsigned int final_reg_id = translate_reg_to_global_id(op.get_operand_reg_number(), op_type) + j;
          inst_regs_in.insert(final_reg_id);
        }
      }
    }

    std::set<int>::const_iterator it_in_check;
    std::map<unsigned,unsigned>::const_iterator it_aux;
    for (it_in_check = inst_regs_in.begin(); it_in_check != inst_regs_in.end(); it_in_check++)
    {
        it_aux = reg_table[wid].find(*it_in_check);

        if ( (it_aux != reg_table[wid].end()) && ( it_aux->second == m_scoreboard_war_max_uses_per_reg) ) {
          m_stats->num_scoreboard_reads_collision_due_to_max_uses_per_reg++;
          return true;
        }
    }
  }

  return false;
}


bool Scoreboard_reads::isEnabled()
{
  return m_enabled;
}

scoreboard_reads_mode Scoreboard_reads::getMode()
{
  return m_mode;
}