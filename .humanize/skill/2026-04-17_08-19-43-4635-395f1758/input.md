# Ask Codex Input

## Question

I'm doing MAPE validation for an Accel-Sim based GPU simulator on the RTX 4070 Super (SM89, Ada Lovelace). Just discovered a data integrity issue I want your opinion on.

Overall MAPE: 14.04% (Pearson 0.99) across 10 Rodinia 2.0-ft benchmarks. Two outliers: lud (-58.7% APE) and nn (-27.8% APE) — both under-predicting.

I just discovered the reason: the sim_output.txt files for both show 'Error: signal 15' (SIGTERM). The user interrupted runs yesterday when stopping for the night, and the outputs were incomplete:

- lud: trace has 10 kernels (lud_diagonal x4, lud_perimeter x3, lud_internal x3), simulator only completed 4 kernels (reported ~89k cycles) before SIGTERM. HW ncu measured 10 kernels totaling ~216k cycles.
- nn: trace has 4 kernels (all euclid), simulator only completed 3 (reported ~25k cycles). HW ncu measured 4 kernels totaling ~34k cycles.

Both benchmarks have tiny block sizes (lud_diagonal is 16 threads/1 block, nn is 16 threads/938 blocks), so their simulation is slow because warps are serial within small blocks.

The other 8 benchmarks ran to completion and average ~7% APE.

My plan:
1. Create branch '4070super-tuning' for any changes
2. First, rerun only lud and nn to completion, recompute APE — I suspect the MAPE will drop significantly once the data is complete
3. If APE for lud/nn is still high after proper completion, THEN investigate:
   - Sub-warp (16-thread) block latency modeling
   - Kernel launch overhead (these benchmarks have many short kernel launches)
   - lud_diagonal is extremely serial: 1 block × 16 threads × O(N) dependent iterations
4. Only tune microarch parameters if real APE is still high

Question: do you agree the first step must be re-running the incomplete benchmarks rather than immediately tuning? Any specific advice for how to investigate short-kernel / sub-warp cases if the APE stays high after rerun? Any known Accel-Sim pitfalls for rodinia-2.0-ft's tiny kernels on modern GPUs (Ada/Ampere)?

## Configuration

- Model: gpt-5.4
- Effort: high
- Timeout: 3600s
- Timestamp: 2026-04-17_08-19-43
- Tool: codex
