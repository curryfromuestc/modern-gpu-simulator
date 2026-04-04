// Copyright (c) 2009-2011, Tor M. Aamodt, Wilson W.L. Fung, Ali Bakhoda,
// George L. Yuan
// The University of British Columbia
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

#include "mem_latency_stat.h"
#include "../abstract_hardware_model.h"
#include "../cuda-sim/ptx-stats.h"
#include "dram.h"
#include "gpu-cache.h"
#include "gpu-misc.h"
#include "gpu-sim.h"
#include "mem_fetch.h"
#include "shader.h"
#include "stat-tool.h"
#include "visualizer.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../../libcuda/gpgpu_context.h"

memory_stats_t::memory_stats_t(unsigned n_shader,
                               const shader_core_config *shader_config,
                               const memory_config *mem_config,
                               const class gpgpu_sim *gpu) {
  assert(mem_config->m_valid);
  assert(shader_config->m_valid);

  unsigned i, j;

  concurrent_row_access =
      (unsigned int **)calloc(mem_config->m_n_mem, sizeof(unsigned int *));
  num_activates =
      (unsigned int **)calloc(mem_config->m_n_mem, sizeof(unsigned int *));
  row_access =
      (unsigned int **)calloc(mem_config->m_n_mem, sizeof(unsigned int *));
  max_conc_access2samerow =
      (unsigned int **)calloc(mem_config->m_n_mem, sizeof(unsigned int *));
  max_servicetime2samerow =
      (unsigned int **)calloc(mem_config->m_n_mem, sizeof(unsigned int *));

  for (unsigned i = 0; i < mem_config->m_n_mem; i++) {
    concurrent_row_access[i] =
        (unsigned int *)calloc(mem_config->nbk, sizeof(unsigned int));
    row_access[i] =
        (unsigned int *)calloc(mem_config->nbk, sizeof(unsigned int));
    num_activates[i] =
        (unsigned int *)calloc(mem_config->nbk, sizeof(unsigned int));
    max_conc_access2samerow[i] =
        (unsigned int *)calloc(mem_config->nbk, sizeof(unsigned int));
    max_servicetime2samerow[i] =
        (unsigned int *)calloc(mem_config->nbk, sizeof(unsigned int));
  }

  m_n_shader = n_shader;
  m_memory_config = mem_config;
  m_gpu = gpu;
  total_n_access = 0;
  total_n_reads = 0;
  total_n_writes = 0;
  max_mrq_latency = 0;
  max_dq_latency = 0;
  max_mf_latency = 0;
  max_icnt2mem_latency = 0;
  max_icnt2sh_latency = 0;
  tot_icnt2mem_latency = 0;
  tot_icnt2sh_latency = 0;
  tot_mrq_num = 0;
  tot_mrq_latency = 0;
  memset(mrq_lat_table, 0, sizeof(unsigned) * 32);
  memset(dq_lat_table, 0, sizeof(unsigned) * 32);
  memset(mf_lat_table, 0, sizeof(unsigned) * 32);
  memset(icnt2mem_lat_table, 0, sizeof(unsigned) * 24);
  memset(icnt2sh_lat_table, 0, sizeof(unsigned) * 24);
  memset(mf_lat_pw_table, 0, sizeof(unsigned) * 32);
  mf_num_lat_pw = 0;
  max_warps =
      n_shader *
      (shader_config->n_thread_per_shader / shader_config->warp_size + 1);
  mf_tot_lat_pw = 0;  // total latency summed up per window. divide by
                      // mf_num_lat_pw to obtain average latency Per Window
  mf_total_lat = 0;
  num_mfs = 0;
  printf("*** Initializing Memory Statistics ***\n");
  totalbankreads =
      (unsigned int **)calloc(mem_config->m_n_mem, sizeof(unsigned int *));
  totalbankwrites =
      (unsigned int **)calloc(mem_config->m_n_mem, sizeof(unsigned int *));
  totalbankaccesses =
      (unsigned int **)calloc(mem_config->m_n_mem, sizeof(unsigned int *));
  mf_total_lat_table = (unsigned long long int **)calloc(
      mem_config->m_n_mem, sizeof(unsigned long long *));
  mf_max_lat_table =
      (unsigned **)calloc(mem_config->m_n_mem, sizeof(unsigned *));
  bankreads = (unsigned int ***)calloc(n_shader, sizeof(unsigned int **));
  bankwrites = (unsigned int ***)calloc(n_shader, sizeof(unsigned int **));
  num_MCBs_accessed = (unsigned int *)calloc(
      mem_config->m_n_mem * mem_config->nbk, sizeof(unsigned int));
  if (mem_config->gpgpu_frfcfs_dram_sched_queue_size) {
    position_of_mrq_chosen = (unsigned int *)calloc(
        mem_config->gpgpu_frfcfs_dram_sched_queue_size, sizeof(unsigned int));
  } else
    position_of_mrq_chosen = (unsigned int *)calloc(1024, sizeof(unsigned int));
  for (i = 0; i < n_shader; i++) {
    bankreads[i] =
        (unsigned int **)calloc(mem_config->m_n_mem, sizeof(unsigned int *));
    bankwrites[i] =
        (unsigned int **)calloc(mem_config->m_n_mem, sizeof(unsigned int *));
    for (j = 0; j < mem_config->m_n_mem; j++) {
      bankreads[i][j] =
          (unsigned int *)calloc(mem_config->nbk, sizeof(unsigned int));
      bankwrites[i][j] =
          (unsigned int *)calloc(mem_config->nbk, sizeof(unsigned int));
    }
  }

  for (i = 0; i < mem_config->m_n_mem; i++) {
    totalbankreads[i] =
        (unsigned int *)calloc(mem_config->nbk, sizeof(unsigned int));
    totalbankwrites[i] =
        (unsigned int *)calloc(mem_config->nbk, sizeof(unsigned int));
    totalbankaccesses[i] =
        (unsigned int *)calloc(mem_config->nbk, sizeof(unsigned int));
    mf_total_lat_table[i] = (unsigned long long int *)calloc(
        mem_config->nbk, sizeof(unsigned long long int));
    mf_max_lat_table[i] = (unsigned *)calloc(mem_config->nbk, sizeof(unsigned));
  }

  mem_access_type_stats =
      (unsigned ***)malloc(NUM_MEM_ACCESS_TYPE * sizeof(unsigned **));
  for (i = 0; i < NUM_MEM_ACCESS_TYPE; i++) {
    int j;
    mem_access_type_stats[i] =
        (unsigned **)calloc(mem_config->m_n_mem, sizeof(unsigned *));
    for (j = 0; (unsigned)j < mem_config->m_n_mem; j++) {
      mem_access_type_stats[i][j] =
          (unsigned *)calloc((mem_config->nbk + 1), sizeof(unsigned *));
    }
  }

  // AerialVision L2 stats
  L2_read_miss = 0;
  L2_write_miss = 0;
  L2_read_hit = 0;
  L2_write_hit = 0;

  L2_cbtoL2length =
      (unsigned int *)calloc(mem_config->m_n_mem, sizeof(unsigned int));
  L2_cbtoL2writelength =
      (unsigned int *)calloc(mem_config->m_n_mem, sizeof(unsigned int));
  L2_L2tocblength =
      (unsigned int *)calloc(mem_config->m_n_mem, sizeof(unsigned int));
  L2_dramtoL2length =
      (unsigned int *)calloc(mem_config->m_n_mem, sizeof(unsigned int));
  L2_dramtoL2writelength =
      (unsigned int *)calloc(mem_config->m_n_mem, sizeof(unsigned int));
  L2_L2todramlength =
      (unsigned int *)calloc(mem_config->m_n_mem, sizeof(unsigned int));
}

memory_stats_t::~memory_stats_t() {
  unsigned i, j;
  for (i = 0; i < m_memory_config->m_n_mem; i++) {
    free(concurrent_row_access[i]);
    free(row_access[i]);
    free(num_activates[i]);
    free(max_conc_access2samerow[i]);
    free(max_servicetime2samerow[i]);
    free(mf_max_lat_table[i]);
    free(mf_total_lat_table[i]);
    free(totalbankaccesses[i]);
    free(totalbankwrites[i]);
    free(totalbankreads[i]);
  }
  free(concurrent_row_access);
  free(row_access);
  free(num_activates);
  free(max_conc_access2samerow);
  free(max_servicetime2samerow);
  free(totalbankreads);
  free(totalbankwrites);
  free(totalbankaccesses);
  free(mf_total_lat_table);
  free(mf_max_lat_table);
  free(num_MCBs_accessed);
  free(position_of_mrq_chosen);
  for (i = 0; i < m_n_shader; i++) {
    for (j = 0; j < m_memory_config->m_n_mem; j++) {
      free(bankreads[i][j]);
      free(bankwrites[i][j]);
    }
    free(bankreads[i]);
    free(bankwrites[i]);
  }
  free(bankreads);
  free(bankwrites);
  for (i = 0; i < NUM_MEM_ACCESS_TYPE; i++) {
    for (j = 0; (unsigned)j < m_memory_config->m_n_mem; j++) {
      free(mem_access_type_stats[i][j]);
    }
    free(mem_access_type_stats[i]);
  }
  free(mem_access_type_stats);
  free(L2_cbtoL2length);
  free(L2_cbtoL2writelength);
  free(L2_L2tocblength);
  free(L2_dramtoL2length);
  free(L2_dramtoL2writelength);
  free(L2_L2todramlength);
}

// record the total latency
unsigned memory_stats_t::memlatstat_done(mem_fetch *mf) {
  unsigned mf_latency;
  mf_latency =
      (m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle) - mf->get_timestamp();
  mf_num_lat_pw++;
  mf_tot_lat_pw += mf_latency;
  unsigned idx = LOGB2(mf_latency);
  assert(idx < 32);
  mf_lat_table[idx]++;
  shader_mem_lat_log(mf->get_sid(), mf_latency);
  mf_total_lat_table[mf->get_tlx_addr().chip][mf->get_tlx_addr().bk] +=
      mf_latency;
  if (mf_latency > max_mf_latency) max_mf_latency = mf_latency;
  return mf_latency;
}

void memory_stats_t::memlatstat_read_done(mem_fetch *mf) {
  if (m_memory_config->gpgpu_memlatency_stat) {
    unsigned mf_latency = memlatstat_done(mf);
    if (mf_latency >
        mf_max_lat_table[mf->get_tlx_addr().chip][mf->get_tlx_addr().bk])
      mf_max_lat_table[mf->get_tlx_addr().chip][mf->get_tlx_addr().bk] =
          mf_latency;
    unsigned icnt2sh_latency;
    icnt2sh_latency = (m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle) -
                      mf->get_return_timestamp();
    tot_icnt2sh_latency += icnt2sh_latency;
    icnt2sh_lat_table[LOGB2(icnt2sh_latency)]++;
    if (icnt2sh_latency > max_icnt2sh_latency)
      max_icnt2sh_latency = icnt2sh_latency;
  }
}

void memory_stats_t::memlatstat_dram_access(mem_fetch *mf) {
  unsigned dram_id = mf->get_tlx_addr().chip;
  unsigned bank = mf->get_tlx_addr().bk;
  if (m_memory_config->gpgpu_memlatency_stat) {
    if (mf->get_is_write()) {
      if (mf->get_sid() < m_n_shader) {  // do not count L2_writebacks here
        bankwrites[mf->get_sid()][dram_id][bank]++;
        shader_mem_acc_log(mf->get_sid(), dram_id, bank, 'w');
      }
      totalbankwrites[dram_id][bank] +=
          ceil(mf->get_data_size() / m_memory_config->dram_atom_size);
    } else {
      bankreads[mf->get_sid()][dram_id][bank]++;
      shader_mem_acc_log(mf->get_sid(), dram_id, bank, 'r');
      totalbankreads[dram_id][bank] +=
          ceil(mf->get_data_size() / m_memory_config->dram_atom_size);
    }
    mem_access_type_stats[mf->get_access_type()][dram_id][bank] +=
        ceil(mf->get_data_size() / m_memory_config->dram_atom_size);
  }

  if (mf->get_pc() != (unsigned)-1)
    m_gpu->gpgpu_ctx->stats->ptx_file_line_stats_add_dram_traffic(
        mf->get_pc(), mf->get_data_size());
}

void memory_stats_t::memlatstat_icnt2mem_pop(mem_fetch *mf) {
  if (m_memory_config->gpgpu_memlatency_stat) {
    unsigned icnt2mem_latency;
    icnt2mem_latency =
        (m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle) - mf->get_timestamp();
    tot_icnt2mem_latency += icnt2mem_latency;
    icnt2mem_lat_table[LOGB2(icnt2mem_latency)]++;
    if (icnt2mem_latency > max_icnt2mem_latency)
      max_icnt2mem_latency = icnt2mem_latency;
  }
}

void memory_stats_t::memlatstat_lat_pw() {
  if (mf_num_lat_pw && m_memory_config->gpgpu_memlatency_stat) {
    assert(mf_tot_lat_pw);
    mf_total_lat += mf_tot_lat_pw;
    num_mfs += mf_num_lat_pw;
    mf_lat_pw_table[LOGB2(mf_tot_lat_pw / mf_num_lat_pw)]++;
    mf_tot_lat_pw = 0;
    mf_num_lat_pw = 0;
  }
}

void memory_stats_t::memlatstat_print(unsigned n_mem, unsigned gpu_mem_n_bk) {
  unsigned i, j, k, l, m;
  unsigned max_bank_accesses, min_bank_accesses, max_chip_accesses,
      min_chip_accesses;

  if (m_memory_config->gpgpu_memlatency_stat) {
    printf("maxmflatency = %d \n", max_mf_latency);
    printf("max_icnt2mem_latency = %d \n", max_icnt2mem_latency);
    printf("maxmrqlatency = %d \n", max_mrq_latency);
    // printf("maxdqlatency = %d \n", max_dq_latency);
    printf("max_icnt2sh_latency = %d \n", max_icnt2sh_latency);
    if (num_mfs) {
      printf("averagemflatency = %lld \n", mf_total_lat / num_mfs);
      printf("avg_icnt2mem_latency = %lld \n", tot_icnt2mem_latency / num_mfs);
      if (tot_mrq_num)
        printf("avg_mrq_latency = %lld \n", tot_mrq_latency / tot_mrq_num);

      printf("avg_icnt2sh_latency = %lld \n", tot_icnt2sh_latency / num_mfs);
    }
    printf("mrq_lat_table:");
    for (i = 0; i < 32; i++) {
      printf("%d \t", mrq_lat_table[i]);
    }
    printf("\n");
    printf("dq_lat_table:");
    for (i = 0; i < 32; i++) {
      printf("%d \t", dq_lat_table[i]);
    }
    printf("\n");
    printf("mf_lat_table:");
    for (i = 0; i < 32; i++) {
      printf("%d \t", mf_lat_table[i]);
    }
    printf("\n");
    printf("icnt2mem_lat_table:");
    for (i = 0; i < 24; i++) {
      printf("%d \t", icnt2mem_lat_table[i]);
    }
    printf("\n");
    printf("icnt2sh_lat_table:");
    for (i = 0; i < 24; i++) {
      printf("%d \t", icnt2sh_lat_table[i]);
    }
    printf("\n");
    printf("mf_lat_pw_table:");
    for (i = 0; i < 32; i++) {
      printf("%d \t", mf_lat_pw_table[i]);
    }
    printf("\n");

    /*MAXIMUM CONCURRENT ACCESSES TO SAME ROW*/
    printf("maximum concurrent accesses to same row:\n");
    for (i = 0; i < n_mem; i++) {
      printf("dram[%d]: ", i);
      for (j = 0; j < gpu_mem_n_bk; j++) {
        printf("%9d ", max_conc_access2samerow[i][j]);
      }
      printf("\n");
    }

    /*MAXIMUM SERVICE TIME TO SAME ROW*/
    printf("maximum service time to same row:\n");
    for (i = 0; i < n_mem; i++) {
      printf("dram[%d]: ", i);
      for (j = 0; j < gpu_mem_n_bk; j++) {
        printf("%9d ", max_servicetime2samerow[i][j]);
      }
      printf("\n");
    }

    /*AVERAGE ROW ACCESSES PER ACTIVATE*/
    int total_row_accesses = 0;
    int total_num_activates = 0;
    printf("average row accesses per activate:\n");
    for (i = 0; i < n_mem; i++) {
      printf("dram[%d]: ", i);
      for (j = 0; j < gpu_mem_n_bk; j++) {
        total_row_accesses += row_access[i][j];
        total_num_activates += num_activates[i][j];
        printf("%9f ", (float)row_access[i][j] / num_activates[i][j]);
      }
      printf("\n");
    }
    printf("average row locality = %d/%d = %f\n", total_row_accesses,
           total_num_activates,
           (float)total_row_accesses / total_num_activates);
    /*MEMORY ACCESSES*/
    k = 0;
    l = 0;
    m = 0;
    max_bank_accesses = 0;
    max_chip_accesses = 0;
    min_bank_accesses = 0xFFFFFFFF;
    min_chip_accesses = 0xFFFFFFFF;
    printf("number of total memory accesses made:\n");
    for (i = 0; i < n_mem; i++) {
      printf("dram[%d]: ", i);
      for (j = 0; j < gpu_mem_n_bk; j++) {
        l = totalbankaccesses[i][j];
        if (l < min_bank_accesses) min_bank_accesses = l;
        if (l > max_bank_accesses) max_bank_accesses = l;
        k += l;
        m += l;
        printf("%9d ", l);
      }
      if (m < min_chip_accesses) min_chip_accesses = m;
      if (m > max_chip_accesses) max_chip_accesses = m;
      m = 0;
      printf("\n");
    }
    printf("total accesses: %d\n", k);
    if (min_bank_accesses)
      printf("bank skew: %d/%d = %4.2f\n", max_bank_accesses, min_bank_accesses,
             (float)max_bank_accesses / min_bank_accesses);
    else
      printf("min_bank_accesses = 0!\n");
    if (min_chip_accesses)
      printf("chip skew: %d/%d = %4.2f\n", max_chip_accesses, min_chip_accesses,
             (float)max_chip_accesses / min_chip_accesses);
    else
      printf("min_chip_accesses = 0!\n");

    /*READ ACCESSES*/
    k = 0;
    l = 0;
    m = 0;
    max_bank_accesses = 0;
    max_chip_accesses = 0;
    min_bank_accesses = 0xFFFFFFFF;
    min_chip_accesses = 0xFFFFFFFF;
    printf("number of total read accesses:\n");
    for (i = 0; i < n_mem; i++) {
      printf("dram[%d]: ", i);
      for (j = 0; j < gpu_mem_n_bk; j++) {
        l = totalbankreads[i][j];
        if (l < min_bank_accesses) min_bank_accesses = l;
        if (l > max_bank_accesses) max_bank_accesses = l;
        k += l;
        m += l;
        printf("%9d ", l);
      }
      if (m < min_chip_accesses) min_chip_accesses = m;
      if (m > max_chip_accesses) max_chip_accesses = m;
      m = 0;
      printf("\n");
    }
    printf("total dram reads = %d\n", k);
    if (min_bank_accesses)
      printf("bank skew: %d/%d = %4.2f\n", max_bank_accesses, min_bank_accesses,
             (float)max_bank_accesses / min_bank_accesses);
    else
      printf("min_bank_accesses = 0!\n");
    if (min_chip_accesses)
      printf("chip skew: %d/%d = %4.2f\n", max_chip_accesses, min_chip_accesses,
             (float)max_chip_accesses / min_chip_accesses);
    else
      printf("min_chip_accesses = 0!\n");

    unsigned int total_reads = k;
    /*WRITE ACCESSES*/
    k = 0;
    l = 0;
    m = 0;
    max_bank_accesses = 0;
    max_chip_accesses = 0;
    min_bank_accesses = 0xFFFFFFFF;
    min_chip_accesses = 0xFFFFFFFF;
    printf("number of total write accesses:\n");
    for (i = 0; i < n_mem; i++) {
      printf("dram[%d]: ", i);
      for (j = 0; j < gpu_mem_n_bk; j++) {
        l = totalbankwrites[i][j];
        if (l < min_bank_accesses) min_bank_accesses = l;
        if (l > max_bank_accesses) max_bank_accesses = l;
        k += l;
        m += l;
        printf("%9d ", l);
      }
      if (m < min_chip_accesses) min_chip_accesses = m;
      if (m > max_chip_accesses) max_chip_accesses = m;
      m = 0;
      printf("\n");
    }
    printf("total dram writes = %d\n", k);
    if (min_bank_accesses)
      printf("bank skew: %d/%d = %4.2f\n", max_bank_accesses, min_bank_accesses,
             (float)max_bank_accesses / min_bank_accesses);
    else
      printf("min_bank_accesses = 0!\n");
    if (min_chip_accesses)
      printf("chip skew: %d/%d = %4.2f\n", max_chip_accesses, min_chip_accesses,
             (float)max_chip_accesses / min_chip_accesses);
    else
      printf("min_chip_accesses = 0!\n");

    unsigned int total_writes = k;
    
    double total_accesses = total_reads + total_writes;
    double dram_bw = ( (total_accesses * 32) / (m_gpu->dram_tot_sim_cycle * m_gpu->get_config().get_dram_period()) ) / 1000000000 ; 
    printf("DRAM_BW_total = %12.4lf GB/Sec\n", dram_bw);
    /*AVERAGE MF LATENCY PER BANK*/
    printf("average mf latency per bank:\n");
    for (i = 0; i < n_mem; i++) {
      printf("dram[%d]: ", i);
      for (j = 0; j < gpu_mem_n_bk; j++) {
        k = totalbankwrites[i][j] + totalbankreads[i][j];
        if (k)
          printf("%10lld", mf_total_lat_table[i][j] / k);
        else
          printf("    none  ");
      }
      printf("\n");
    }

    /*MAXIMUM MF LATENCY PER BANK*/
    printf("maximum mf latency per bank:\n");
    for (i = 0; i < n_mem; i++) {
      printf("dram[%d]: ", i);
      for (j = 0; j < gpu_mem_n_bk; j++) {
        printf("%10d", mf_max_lat_table[i][j]);
      }
      printf("\n");
    }
  }

  if (m_memory_config->gpgpu_memlatency_stat & GPU_MEMLATSTAT_MC) {
    printf(
        "\nNumber of Memory Banks Accessed per Memory Operation per Warp (from "
        "0):\n");
    unsigned long long accum_MCBs_accessed = 0;
    unsigned long long tot_mem_ops_per_warp = 0;
    for (i = 0; i < n_mem * gpu_mem_n_bk; i++) {
      accum_MCBs_accessed += i * num_MCBs_accessed[i];
      tot_mem_ops_per_warp += num_MCBs_accessed[i];
      printf("%d\t", num_MCBs_accessed[i]);
    }

    printf(
        "\nAverage # of Memory Banks Accessed per Memory Operation per "
        "Warp=%f\n",
        (float)accum_MCBs_accessed / tot_mem_ops_per_warp);

    // printf("\nAverage Difference Between First and Last Response from Memory
    // System per warp = ");

    printf("\nposition of mrq chosen\n");

    if (!m_memory_config->gpgpu_frfcfs_dram_sched_queue_size)
      j = 1024;
    else
      j = m_memory_config->gpgpu_frfcfs_dram_sched_queue_size;
    k = 0;
    l = 0;
    for (i = 0; i < j; i++) {
      printf("%d\t", position_of_mrq_chosen[i]);
      k += position_of_mrq_chosen[i];
      l += i * position_of_mrq_chosen[i];
    }
    printf("\n");
    printf("\naverage position of mrq chosen = %f\n", (float)l / k);
  }
}

void memory_stats_t::add(const memory_stats_t *other) {
  max_mrq_latency = std::max(max_mrq_latency, other->max_mrq_latency);
  max_dq_latency = std::max(max_dq_latency, other->max_dq_latency);
  max_mf_latency = std::max(max_mf_latency, other->max_mf_latency);
  max_icnt2mem_latency =
      std::max(max_icnt2mem_latency, other->max_icnt2mem_latency);
  tot_icnt2mem_latency += other->tot_icnt2mem_latency;
  tot_icnt2sh_latency += other->tot_icnt2sh_latency;
  tot_mrq_latency += other->tot_mrq_latency;
  tot_mrq_num += other->tot_mrq_num;
  max_icnt2sh_latency =
      std::max(max_icnt2sh_latency, other->max_icnt2sh_latency);

  for (unsigned int i = 0; i < 32; ++i) {
    mrq_lat_table[i] += other->mrq_lat_table[i];
    dq_lat_table[i] += other->dq_lat_table[i];
    mf_lat_table[i] += other->mf_lat_table[i];
    mf_lat_pw_table[i] += other->mf_lat_pw_table[i];
  }

  for (unsigned int i = 0; i < 24; ++i) {
    icnt2mem_lat_table[i] += other->icnt2mem_lat_table[i];
    icnt2sh_lat_table[i] += other->icnt2sh_lat_table[i];
  }

  mf_num_lat_pw += other->mf_num_lat_pw;
  max_warps = std::max(max_warps, other->max_warps);
  mf_tot_lat_pw += other->mf_tot_lat_pw;
  mf_total_lat += other->mf_total_lat;

  // Assuming dimensions match, perform element-wise addition for 2D arrays
  for (unsigned int i = 0; i < m_memory_config->m_n_mem; ++i) {
    for (unsigned int j = 0; j < m_memory_config->nbk; ++j) {
      mf_total_lat_table[i][j] += other->mf_total_lat_table[i][j];
      mf_max_lat_table[i][j] =
          std::max(mf_max_lat_table[i][j], other->mf_max_lat_table[i][j]);
      totalbankwrites[i][j] += other->totalbankwrites[i][j];
      totalbankreads[i][j] += other->totalbankreads[i][j];
      totalbankaccesses[i][j] += other->totalbankaccesses[i][j];
      concurrent_row_access[i][j] += other->concurrent_row_access[i][j];
      num_activates[i][j] += other->num_activates[i][j];
      row_access[i][j] += other->row_access[i][j];
      max_conc_access2samerow[i][j] += other->max_conc_access2samerow[i][j];
      max_servicetime2samerow[i][j] = std::max(
          max_servicetime2samerow[i][j], other->max_servicetime2samerow[i][j]);
    }
    L2_cbtoL2length[i] += other->L2_cbtoL2length[i];
    L2_cbtoL2writelength[i] += other->L2_cbtoL2writelength[i];
    L2_L2tocblength[i] += other->L2_L2tocblength[i];
    L2_dramtoL2length[i] += other->L2_dramtoL2length[i];
    L2_dramtoL2writelength[i] += other->L2_dramtoL2writelength[i];
    L2_L2todramlength[i] += other->L2_L2todramlength[i];
  }

  // Assuming dimensions match, perform element-wise addition for 3D arrays
  for (unsigned int i = 0; i < m_n_shader; ++i) {
    for (unsigned int j = 0; j < m_memory_config->m_n_mem; ++j) {
      for (unsigned int k = 0; k < m_memory_config->nbk; ++k) {
        bankwrites[i][j][k] += other->bankwrites[i][j][k];
        bankreads[i][j][k] += other->bankreads[i][j][k];
      }
    }
  }

  for(unsigned int i = 0; i < NUM_MEM_ACCESS_TYPE; i++) {
    for(unsigned int j = 0; j < m_memory_config->m_n_mem; j++) {
      for(unsigned int k = 0; k < m_memory_config->nbk; k++) {
        mem_access_type_stats[i][j][k] += other->mem_access_type_stats[i][j][k];
      }
    }
  }

  for (unsigned int i = 0; i < m_memory_config->m_n_mem * m_memory_config->nbk; ++i) {
    num_MCBs_accessed[i] += other->num_MCBs_accessed[i];
  }

  for (unsigned int i = 0; i < m_memory_config->gpgpu_frfcfs_dram_sched_queue_size; ++i) {
    position_of_mrq_chosen[i] += other->position_of_mrq_chosen[i];
  }

  L2_read_miss += other->L2_read_miss;
  L2_write_miss += other->L2_write_miss;
  L2_read_hit += other->L2_read_hit;
  L2_write_hit += other->L2_write_hit;
  total_n_access += other->total_n_access;
  total_n_reads += other->total_n_reads;
  total_n_writes += other->total_n_writes;
}

void memory_stats_t::reset() {
  tot_icnt2mem_latency = 0;
  tot_icnt2sh_latency = 0;
  tot_mrq_latency = 0;
  tot_mrq_num = 0;

  for (unsigned int i = 0; i < 32; ++i) {
    mrq_lat_table[i] = 0;
    dq_lat_table[i] = 0;
    mf_lat_table[i] = 0;
    mf_lat_pw_table[i] = 0;
  }

  for (unsigned int i = 0; i < 24; ++i) {
    icnt2mem_lat_table[i] = 0;
    icnt2sh_lat_table[i] = 0;
  }

  mf_num_lat_pw = 0;
  mf_tot_lat_pw = 0;
  mf_total_lat  = 0;

  // Assuming dimensions match, perform element-wise addition for 2D arrays
  for (unsigned int i = 0; i < m_memory_config->m_n_mem; ++i) {
    for (unsigned int j = 0; j < m_memory_config->nbk; ++j) {
      mf_total_lat_table[i][j] = 0;
      mf_max_lat_table[i][j] = 0;
      totalbankwrites[i][j] = 0;
      totalbankreads[i][j] = 0;
      totalbankaccesses[i][j] = 0;
      num_activates[i][j] = 0;
      row_access[i][j] = 0;
    }
    L2_cbtoL2length[i] = 0;
    L2_cbtoL2writelength[i] = 0;
    L2_L2tocblength[i] = 0;
    L2_dramtoL2length[i] = 0;
    L2_dramtoL2writelength[i] = 0;
    L2_L2todramlength[i] = 0;
  }

  // Assuming dimensions match, perform element-wise addition for 3D arrays
  for (unsigned int i = 0; i < m_n_shader; ++i) {
    for (unsigned int j = 0; j < m_memory_config->m_n_mem; ++j) {
      for (unsigned int k = 0; k < m_memory_config->nbk; ++k) {
        bankwrites[i][j][k] = 0;
        bankreads[i][j][k] = 0;
      }
    }
  }

  for(unsigned int i = 0; i < NUM_MEM_ACCESS_TYPE; i++) {
    for(unsigned int j = 0; j < m_memory_config->m_n_mem; j++) {
      for(unsigned int k = 0; k < m_memory_config->nbk; k++) {
        mem_access_type_stats[i][j][k] = 0;
      }
    }
  }

  for (unsigned int i = 0; i < m_memory_config->m_n_mem * m_memory_config->nbk; ++i) {
    num_MCBs_accessed[i] = 0;
  }

  for (unsigned int i = 0; i < m_memory_config->gpgpu_frfcfs_dram_sched_queue_size; ++i) {
    position_of_mrq_chosen[i] = 0;
  }

  L2_read_miss = 0;
  L2_write_miss = 0;
  L2_read_hit = 0;
  L2_write_hit = 0;
  total_n_access = 0;
  total_n_reads = 0;
  total_n_writes = 0;
}