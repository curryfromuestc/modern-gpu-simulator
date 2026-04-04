| Benchmark | HW_cycles | Sim_cycles | Error_Cycles % | AbsError_Cycles % | Sim_WallTime |
|:--|--:|--:|--:|--:|:--|
| backprop-rodinia-2.0-ft | 12830 | 16564 | 29.10 | 29.10 | 3h01m29s |
| bfs-rodinia-2.0-ft | 93392 | 117991 | 26.34 | 26.34 | 2h49m58s |
| heartwall-rodinia-2.0-ft | 17107 | 10243 | -40.12 | 40.12 | 2h57m39s |
| hotspot-rodinia-2.0-ft | 87947 | 87293 | -0.74 | 0.74 | 2h09m36s |
| lud-rodinia-2.0-ft | 208552 | 177512 | -14.88 | 14.88 | 4h41m12s |
| nn-rodinia-2.0-ft | 14697 | 26875 | 82.86 | 82.86 | 6h22m57s |
| nw-rodinia-2.0-ft | 121153 | 140587 | 16.04 | 16.04 | 7h13m30s |
| pathfinder-rodinia-2.0-ft | 20529 | 29715 | 44.75 | 44.75 | 6h53m29s |
| srad_v2-rodinia-2.0-ft | 26749 | 33591 | 25.58 | 25.58 | 7h06m27s |
| streamcluster-rodinia-2.0-ft | 1198133 | 186962 | -84.39 | 84.39 | 7h37m23s |
| **Average (MAPE)** | | | **8.45** | **36.48** | |
| Median | | | | 29.10 | |
| Pearson_Corr | | | 0.6683 | | |
| Std Dev | | | | 26.43 | |

## Config
- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition
- 188 SMs, 2430 MHz core, 12481 MHz DRAM, 512-bit, 128 MB L2
- Simulator config: SM120_RTXPRO6000 (based on SM120_RTX5070_TI)
- Traces: NVBit v1.7.5 + CUDA 12.9 on Device 0
- HW cycles: nsys profiling on Device 1, converted via Duration_ns * 2430 MHz / 1000
- Config NOT tuned (DRAM timing, L2 latency from RTX 5070 TI)
