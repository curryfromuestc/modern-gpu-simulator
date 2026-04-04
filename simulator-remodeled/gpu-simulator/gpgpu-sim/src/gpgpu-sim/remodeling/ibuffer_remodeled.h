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

#include "../../abstract_hardware_model.h"

#include <stdio.h>
#include <stack>
#include <unordered_map>

class shader_core_stats;   // Definition to be allowed to compile. Code of this
                           // class in shader.h and shader.cc
class shader_core_config;  // Definition to be allowed to compile. Code of this
                           // class in shader.h and shader.cc
class scheduler_unit;      // Definition to be allowed to compile. Code of this
                           // class in shader.h and shader.cc
class shd_warp_t;  // Definition to be allowed to compile. Code of this class in
                   // shader.h and shader.cc
class trace_shader_core_ctx;  // Definition to be allowed to compile. Code of
                              // this class in shader.h and shader.cc


struct IBuffer_Entry {
  IBuffer_Entry(bool is_valid, address_type pc, warp_inst_t *inst) {
    m_valid = is_valid;
    m_pc = pc;
    m_inst = inst;
  }
  bool m_valid;
  address_type m_pc;
  warp_inst_t *m_inst;
};

/**
 * @brief Class that has the behavior of the Remodeled IBuffer
 *
 */
class IBuffer_Remodeled {
 public:
  /**
   * @brief Construct a new IBuffer_Remodeled object
   *
   * @param is_remodeled_enabled bool that says if the remodeled IBuffer is
   * enabled
   * @param ibuffer_size int remodeled IBuffer number of maximum entries
   * @param fetch_decode_width int number of instructions that can be fetched
   * and decoded in one cycle
   * @param shd_warp pointer to the warp that is the owner
   * @param stats pointer to the object that tracks the stats of the simulation
   */
  IBuffer_Remodeled(const shader_core_config* config, shd_warp_t *shd_warp,
                    shader_core_stats *stats);

  /**
   * @brief Destroy the IBuffer_Remodeled object deleting the queue that holds
   * the entries
   *
   */
  ~IBuffer_Remodeled();

  /**
   * @brief Says if the remodeled IBuffer feature is enabled
   *
   * @return true if remodeled IBuffer is enabled
   * @return false if remodeled IBuffer is disabled
   */
  bool get_is_enabled();

  /**
   * @brief Says if the remodeled IBuffer is empty
   *
   * @return true  if the remodeled IBuffer is empty
   * @return false if the remodeled IBuffer has at least one entry valid
   */
  bool get_is_empty();

  /**
   * @brief Get the number of entries of the remodeled IBuffer
   *
   * @return int , number of entries of the remodeled IBuffer
   */
  unsigned int get_num_entries();

  /**
   * @brief Says if the remodeled IBuffer has enough space for the fetch-decode
   * width
   *
   * @return true if the remodeled IBuffer has enough space for the fetch-decode
   * width
   * @return false if the remodeled IBuffer has not enough space for the
   * fetch-decode width
   */
  bool can_fetch();

  /**
   * @brief Says if the remodeled IBuffer is full
   *
   * @return true if the remodeled IBuffer is full
   * @return false if the remodeled IBuffer is not full
   */
  bool is_full();

  /**
   * @brief Says if the remodeled IBuffer has reached the return instruction
   *
   * @return true if the remodeled IBuffer has reached the return instruction
   * @return false if the remodeled IBuffer has not reached the return
   * instruction
   */
  bool get_is_ret_reached();

  /**
   * @brief Sets the boolean that says if the remodeled IBuffer has reached the
   * return instruction
   *
   * @param is_ret_reached boolean that says if the remodeled IBuffer has reached
   * the return instruction
   */
  void set_is_ret_reached(bool is_ret_reached);

  /**
   * @brief Get the next pc address that the IBuffer wants to be requested by the
   * fetch stage
   *
   * @return address_type , the pc address that the IBuffer wants to be requested
   * by the fetch stage
   */
  address_type get_next_pc_to_fetch_request();
  
  /**
   * @brief Removes the entry of the remodeled IBuffer that has the same pc. It has to be the latest entry pushed
   * 
   * @param address_type pc , the pc address that the IBuffer wants to be removed
   */
  void remove_entry(address_type pc);

   /**
    * @brief Get the next pc to issue
    * 
    * @return address_type , the pc address that is going to be issued
    */
  address_type get_next_pc_to_issue();

  std::deque<IBuffer_Entry> &get_remodeled_ibuffer() { return m_remodeled_ibuffer; }

  /**
   * @brief Says if the oldest instruction/entry of the remodeled IBuffer is
   * valid
   *
   * @return true is the oldest instruction/entry is valid
   * @return false if the oldest instruction/entry is not valid. Case when is
   * empty
   */
  bool is_next_valid();

  /**
   * @brief Gets a pointer to the oldest instruction in the remodeled IBuffer
   *
   * @return const warp_inst_t* pointer of the oldest instruction in the
   * remodeled IBuffer
   */
  warp_inst_t *next_inst();

  /**
   * @brief Issues the oldest entry of the remodeled IBuffer
   *
   */
  void issued();

  /**
   * @brief Flushes the remodeled IBuffer and becomes empty with all the entries
   * as not valid
   * 
   * @param reset_pc_to_0 boolean that says if the pc address of the warp should be reset to 0
   *
   */
  void flush(bool reset_pc_to_0);

  /**
   * @brief Prints the information of the instructions and top entry (if exists)
   * inside the remodeled IBuffer
   *
   * @param fout Pointer to a file where the information will be printed
   */
  void print(FILE *fout);

 private:
  /**
   * @brief Boolean that indicates if the remodeled IBuffer is used during the
   * execution
   *
   */
  bool m_is_enabled;

  /**
   * @brief Number of current entries filled of the IBuffer
   *
   */
  unsigned int m_num_entries;

  /**
   * @brief Size of the IBuffer
   *
   */
  unsigned int m_num_max_entries;

  /**
   * @brief Number of instructions that can be fetched and decoded in one cycle
   *
   */
  unsigned int m_fetch_decode_width;

  /**
   * @brief Pointer to the warp owner of the extender IBuffer instance
   *
   */
  shd_warp_t *m_shd_warp;

  /**
   * @brief Pointer to the object that stores and tracks the stats of the
   * simulator
   *
   */
  shader_core_stats *m_stats;

  /**
   * @brief Queue that stores all the entries of the remodeled Instruction Buffer
   *
   */
  std::deque<IBuffer_Entry> m_remodeled_ibuffer;

  /**
   * @brief Says if the next pc address is initialized
   *
   */
  bool m_is_init_next_pc;

   /**
    * @brief Next pc address that the Remodeled IBuffer wants to be requested by the fetch stage
    *
    */
  address_type m_next_pc_to_fetch_request;

  /**
   * @brief Says if the remodeled IBuffer has reached the return instruction
   *
   */
  bool m_is_ret_reached;

  /**
   * @brief Pointer to the object that stores the configuration of the shader
   * core
   *
   */
  const shader_core_config *m_config;
};