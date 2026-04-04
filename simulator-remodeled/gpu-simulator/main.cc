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

#include <fstream>
#include <iostream>
#include <math.h>
#include <sstream>
#include <stdio.h>
#include <string>
#include <time.h>
#include <vector>

#include "gpgpu_context.h"
#include "abstract_hardware_model.h"
#include "cuda-sim/cuda-sim.h"
#include "gpgpu-sim/gpu-sim.h"
#include "gpgpu-sim/icnt_wrapper.h"
#include "gpgpusim_entrypoint.h"
#include "option_parser.h"
#include "../ISA_Def/trace_opcode.h"
#include "trace_driven.h"
#include "../trace-parser/trace_parser.h"
#include "accelsim_version.h"

#include <omp.h>

#include <signal.h>
#include <unistd.h>
#include <execinfo.h>
#include <stdlib.h>

gpgpu_sim *gpgpu_trace_sim_init_perf_model(int argc, const char *argv[],
                                           gpgpu_context *m_gpgpu_context,
                                           class trace_config *m_config, option_parser_t &opp);

trace_kernel_info_t *create_kernel_info( kernel_trace_t* kernel_trace_info,
		                      gpgpu_context *m_gpgpu_context, class trace_config *config,
							  trace_parser *parser);


void handler(int sig) {
  void *array[50];
  size_t size;

  fprintf(stderr, "Error: signal %d:\n", sig);
  fflush(stderr);

  // get void*'s for all entries on the stack
  size = backtrace(array, 50);

  // print out all the frames to stderr
  fprintf(stdout, "Error: signal %d:\n", sig);
  backtrace_symbols_fd(array, size, STDOUT_FILENO);
  fflush(stdout);
  exit(1);
}

int main(int argc, const char **argv) {
  std::cout << "Accel-Sim [build " << g_accelsim_version << "]";
  gpgpu_context *m_gpgpu_context = new gpgpu_context();
  trace_config tconfig;
  option_parser_t opp;
  gpgpu_sim *m_gpgpu_sim =
      gpgpu_trace_sim_init_perf_model(argc, argv, m_gpgpu_context, &tconfig, opp);
  m_gpgpu_sim->init();
  
  m_gpgpu_sim->m_current_omp_scheduler = omp_sched_t::omp_sched_static;
  omp_set_schedule(m_gpgpu_sim->m_current_omp_scheduler, 1);

  trace_parser tracer(tconfig.get_traces_filename(), tconfig.get_is_extra_traces_enabled(), m_gpgpu_sim->getShaderCoreConfig()->filter_first_kernel_id, m_gpgpu_sim->getShaderCoreConfig()->filter_last_kernel_id); // MOD. Improved tracer

  tconfig.parse_config(); 

  m_gpgpu_sim->parse_extra_trace_info(tracer.get_extra_trace_info_filename(), tconfig.get_is_extra_traces_enabled()); // MOD. Improved tracer

  // for each kernel
  // load file
  // parse and create kernel info
  // launch
  // while loop till the end of the end kernel execution
  // prints stats
  bool concurrent_kernel_sm =  m_gpgpu_sim->getShaderCoreConfig()->gpgpu_concurrent_kernel_sm;
  unsigned window_size = concurrent_kernel_sm ? m_gpgpu_sim->get_config().get_max_concurrent_kernel() : 1;
  assert(window_size > 0);
  std::vector<trace_command> commandlist = tracer.parse_commandlist_file();
  std::vector<unsigned long> busy_streams;
  std::vector<trace_kernel_info_t*> kernels_info;
  kernels_info.reserve(window_size);

  bool active = false;
  bool sim_cycles = false;
  unsigned finished_kernel_uid = 0;
  bool is_cta_max_hit = false;
  bool can_continue_simulation = true;

  unsigned i = 0;
  signal(SIGSEGV, handler);
  signal(SIGILL, handler);
  signal(SIGABRT, handler);
  signal(SIGTERM, handler);
  signal(SIGFPE, handler);
  while (true) {
    if((i >= commandlist.size() && kernels_info.empty()) || !can_continue_simulation){
      break;
    }
    //gulp up as many commands as possible - either cpu_gpu_mem_copy 
    //or kernel_launch - until the vector "kernels_info" has reached
    //the window_size or we have read every command from commandlist
    while (kernels_info.size() < window_size && i < commandlist.size()) {
      trace_kernel_info_t *kernel_info = NULL;
      if (commandlist[i].m_type == command_type::cpu_gpu_mem_copy) {
        size_t addre, Bcount;
        tracer.parse_memcpy_info(commandlist[i].command_string, addre, Bcount);
        std::cout << "launching memcpy command : " << commandlist[i].command_string << std::endl;
        m_gpgpu_sim->perf_memcpy_to_gpu(addre, Bcount);
        i++;
      } else if (commandlist[i].m_type == command_type::kernel_launch) {
        // Read trace header info for window_size number of kernels
        kernel_trace_t* kernel_trace_info = tracer.parse_kernel_info(commandlist[i].command_string, m_gpgpu_sim->get_extra_trace_info());
        kernel_info = create_kernel_info(kernel_trace_info, m_gpgpu_context, &tconfig, &tracer);
        kernels_info.push_back(kernel_info);
        std::cout << "Header info loaded for kernel command : " << commandlist[i].command_string << std::endl;
        i++;
      }
      else{
        //unsupported commands will fail the simulation
        assert(0 && "Undefined Command");
      }
    }

    // Launch all kernels within window that are on a stream that isn't already running
    for (auto k : kernels_info) {
      bool stream_busy = false;
      for (auto s: busy_streams) {
        if (s == k->get_cuda_stream_id())
          stream_busy = true;
      }
      if (!stream_busy && m_gpgpu_sim->can_start_kernel() && !k->was_launched()) {
        std::cout << "launching kernel name: " << k->get_name() << " uid: " << k->get_uid() << std::endl;
        m_gpgpu_sim->launch(k);
        k->set_launched();
        busy_streams.push_back(k->get_cuda_stream_id());
      }
    }
    active = m_gpgpu_sim->active();
    sim_cycles = false;
    finished_kernel_uid = 0;
    is_cta_max_hit = false;
    
    while (true) {
      if (active) {
        m_gpgpu_sim->cycle();
      }

      if (active) {
          sim_cycles = true;
          m_gpgpu_sim->deadlock_check();
      } else if (m_gpgpu_sim->cycle_insn_cta_max_hit()) {
          m_gpgpu_context->the_gpgpusim->g_stream_manager
              ->stop_all_running_kernels();
          is_cta_max_hit = true;
      }
      active = m_gpgpu_sim->active();
      finished_kernel_uid = m_gpgpu_sim->finished_kernel();

      if (!active || finished_kernel_uid || is_cta_max_hit) {
        break;
      }
    }

    // cleanup finished kernel
    if ( (finished_kernel_uid || m_gpgpu_sim->cycle_insn_cta_max_hit()
        || !m_gpgpu_sim->active()) && !kernels_info.empty() ) {
      trace_kernel_info_t* k = NULL;
      for (unsigned j = 0; j < kernels_info.size(); j++) {
        k = kernels_info.at(j);
        if (k->get_uid() == finished_kernel_uid || m_gpgpu_sim->cycle_insn_cta_max_hit()
            || !m_gpgpu_sim->active()) {
          for (std::size_t l = 0; l < busy_streams.size(); l++) {
            if (busy_streams.at(l) == k->get_cuda_stream_id()) {
              busy_streams.erase(busy_streams.begin()+l);
              break;
            }
          }
          tracer.kernel_finalizer(k->get_trace_info());
          delete k->entry();
          delete k;
          kernels_info.erase(kernels_info.begin()+j);
          if (!m_gpgpu_sim->cycle_insn_cta_max_hit() && m_gpgpu_sim->active())
            break;
        }
      }
      assert(k);
      m_gpgpu_sim->print_stats();
    }

    if (sim_cycles) {
      m_gpgpu_sim->update_stats();
      m_gpgpu_context->print_simulation_time();
    }
    if (m_gpgpu_sim->cycle_insn_cta_max_hit()) {
      printf("GPGPU-Sim: ** break due to reaching the maximum cycles (or "
            "instructions) **\n");
      fflush(stdout);
      can_continue_simulation = false;
    }
  }
  option_parser_destroy(opp);
  delete m_gpgpu_sim;
  delete m_gpgpu_context;
  // we print this message to inform the gpgpu-simulation stats_collect script
  // that we are done
  printf("GPGPU-Sim: *** simulation thread exiting ***\n");
  printf("GPGPU-Sim: *** exit detected ***\n");
  fflush(stdout);

  return 0;
}


trace_kernel_info_t *create_kernel_info( kernel_trace_t* kernel_trace_info,
		                      gpgpu_context *m_gpgpu_context, class trace_config *config,
							  trace_parser *parser){

  gpgpu_ptx_sim_info info;
  info.smem = kernel_trace_info->shmem;
  info.regs = kernel_trace_info->nregs;
  dim3 gridDim(kernel_trace_info->grid_dim_x, kernel_trace_info->grid_dim_y, kernel_trace_info->grid_dim_z);
  dim3 blockDim(kernel_trace_info->tb_dim_x, kernel_trace_info->tb_dim_y, kernel_trace_info->tb_dim_z);
  trace_function_info *function_info =
      new trace_function_info(info, m_gpgpu_context);
  function_info->set_name(kernel_trace_info->kernel_name.c_str());
  trace_kernel_info_t *kernel_info =
      new trace_kernel_info_t(gridDim, blockDim, function_info,
    		  parser, config, kernel_trace_info);
  kernel_info->function_unique_id = kernel_trace_info->func_unique_id;
  kernel_info->is_captured_from_binary = kernel_trace_info->is_cap_from_binary;
  return kernel_info;
}

gpgpu_sim *gpgpu_trace_sim_init_perf_model(int argc, const char *argv[],
                                           gpgpu_context *m_gpgpu_context,
                                           trace_config *m_config, option_parser_t &opp) {
  srand(1);
  print_splash();

  opp = option_parser_create();

  m_gpgpu_context->ptx_reg_options(opp);
  m_gpgpu_context->func_sim->ptx_opcocde_latency_options(opp);

  icnt_reg_options(opp);

  m_gpgpu_context->the_gpgpusim->g_the_gpu_config =
      new gpgpu_sim_config(m_gpgpu_context);
  m_gpgpu_context->the_gpgpusim->g_the_gpu_config->reg_options(
      opp); // register GPU microrachitecture options
  m_config->reg_options(opp);
  m_gpgpu_context->the_gpgpusim->g_trace_config = m_config;

  option_parser_cmdline(opp, argc, argv); // parse configuration options
  fprintf(stdout, "GPGPU-Sim: Configuration options:\n\n");
  option_parser_print(opp, stdout);
  // Set the Numeric locale to a standard locale where a decimal point is a
  // "dot" not a "comma" so it does the parsing correctly independent of the
  // system environment variables
  assert(setlocale(LC_NUMERIC, "C"));
  m_gpgpu_context->the_gpgpusim->g_the_gpu_config->init();

  m_gpgpu_context->the_gpgpusim->g_the_gpu_config->set_custom_options(true); // MOD. General parse options

  m_gpgpu_context->the_gpgpusim->g_the_gpu = new trace_gpgpu_sim(
      *(m_gpgpu_context->the_gpgpusim->g_the_gpu_config), m_gpgpu_context);

  m_gpgpu_context->the_gpgpusim->g_stream_manager =
      new stream_manager((m_gpgpu_context->the_gpgpusim->g_the_gpu),
                         m_gpgpu_context->func_sim->g_cuda_launch_blocking);

  m_gpgpu_context->the_gpgpusim->g_simulation_starttime = time((time_t *)NULL);
  
  return m_gpgpu_context->the_gpgpusim->g_the_gpu;
}
