#include <cassert>
#include <iostream>

#include "result_bus.h"
#include "shader.h"
#include "../constants.h"


ResultBus::ResultBus(unsigned int max_allowed_wb_ports_rf) {
  m_max_allowed_wb_ports_rf = max_allowed_wb_ports_rf;
  m_pipelined_latency.resize(MAX_ALU_LATENCY, 0);
}

void ResultBus::cycle() {
  for (unsigned stage = 0; stage < MAX_ALU_LATENCY-1; ++stage) {
    m_pipelined_latency[stage] = m_pipelined_latency[stage+1];
  }
  m_pipelined_latency[MAX_ALU_LATENCY-1] = 0;
}

bool ResultBus::test(unsigned latency) const {
  assert(latency < MAX_ALU_LATENCY);
  assert(m_pipelined_latency[latency] <= m_max_allowed_wb_ports_rf);
  return m_pipelined_latency[latency] == m_max_allowed_wb_ports_rf;
}

void ResultBus::set(unsigned latency) {
  assert(latency < MAX_ALU_LATENCY);
  assert(m_pipelined_latency[latency] < m_max_allowed_wb_ports_rf);
  m_pipelined_latency[latency]++;
}

ResultBusses::~ResultBusses() {
  for (auto bus : m_res_busses) delete bus;
}

void ResultBusses::cycle() {
  for (auto bus : m_res_busses) {
    bus->cycle();
  }
}

unsigned ResultBusses::num_free_slots(unsigned latency) const {
  unsigned free_slots_remained = m_width;
  assert(free_slots_remained > 0);
  for (auto bus : m_res_busses)
    if (bus->test(latency))
      if (!--free_slots_remained) return 0;
  return free_slots_remained;
}

int ResultBusses::test(const warp_inst_t *inst) {
  unsigned latency = inst->latency;
  assert(latency < MAX_ALU_LATENCY);
  unsigned fs_count = num_free_slots(latency);
  assert(fs_count <= m_width);
  if (!fs_count) return -1;  // there is no latch available

  int regbank1, regbank2;
  find_reg_banks(inst, regbank1, regbank2);
  if (regbank1 == -1) {
    return 1;
  }

  if (m_res_busses[regbank1]->test(latency)) {
    return -1;                   // conflict with first access
  }
  if (regbank2 != -1) {          // there is a second access. Only used for instructions that have two destination registers. MOD. Not used for the moment
    if (regbank1 == regbank2) {  // conflict within the instruction accesses
      assert(num_free_slots(latency + 1) < m_width);
      if (m_res_busses[regbank2]->test(latency + 1) ||
          !num_free_slots(latency + 1))
        return -1;  // conflict or no room for second access
      m_res_busses[regbank2]->set(latency + 1);
    } else {
      if (m_res_busses[regbank2]->test(latency) || fs_count <= 1)
        return -1;  // conflict with second access or no room for two accesses

      m_res_busses[regbank2]->set(latency);
    }
  }

  m_res_busses[regbank1]->set(latency);

  return 1;
}

void ResultBusses::find_reg_banks(const warp_inst_t *inst, int &regbank1,
                                  int &regbank2) const {
  regbank1 = regbank2 = -1;
  for (unsigned op = 0; op < MAX_REG_OPERANDS; ++op) {
    int reg_num = inst->arch_reg.dst[op];
    if (reg_num >= 0) {
      unsigned bank = register_bank(
          reg_num, inst->warp_id(), m_num_banks, m_rf->get_bank_warp_shift(),
          m_rf->get_is_sub_core_model(), m_rf->get_banks_per_sched(),
          inst->get_schd_id());
      assert(regbank2 == -1);
      if ((regbank1 == -1) && (bank != BANK_ID_PREDICATE_REGS_TO_DETECT_SKIP)) {
        regbank1 = bank;
        continue;
      }
      regbank2 = bank;
    }
  }
}
void ResultBusses::init(unsigned width, unsigned num_banks,
                        opndcoll_rfu_t *rf) {
  assert(width > 0);
  assert(num_banks > 0);
  m_width = width;
  m_num_banks = num_banks;
  m_rf = rf;

  for (size_t i = 0; i < m_width; i++) {
    m_res_busses.push_back(new ResultBus(m_rf->shader_core()->get_config()->reg_file_port_throughput));
  }
}
