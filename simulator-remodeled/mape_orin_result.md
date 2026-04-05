# Rodinia 2.0-ft MAPE on Jetson AGX Orin (SM87, MODE_30W, 8 SMs @ 612 MHz)

## MAPE 结果（10 个 benchmark，trace.config 已正确加载）

| Benchmark | HW cycles | Sim cycles | Error % | AbsError % | Sim time (s) |
|:--|--:|--:|--:|--:|--:|
| backprop-rodinia-2.0-ft | 98,554 | 97,696 | -0.87 | 0.87 | 15 |
| bfs-rodinia-2.0-ft | 127,290 | 102,244 | -19.68 | 19.68 | 10 |
| hotspot-rodinia-2.0-ft | 389,273 | 351,654 | -9.66 | 9.66 | 46 |
| heartwall-rodinia-2.0-ft | 30,107 | 25,374 | -15.72 | 15.72 | 5 |
| lud-rodinia-2.0-ft | 195,097 | 161,859 | -17.04 | 17.04 | 11 |
| nw-rodinia-2.0-ft | 142,299 | 120,574 | -15.27 | 15.27 | 8 |
| nn-rodinia-2.0-ft | 77,306 | 55,845 | -27.76 | 27.76 | 15 |
| pathfinder-rodinia-2.0-ft | 31,514 | 26,687 | -15.32 | 15.32 | 3 |
| srad_v2-rodinia-2.0-ft | 83,026 | 78,316 | -5.67 | 5.67 | 13 |
| streamcluster-rodinia-2.0-ft | 917,314 | 915,206 | -0.23 | 0.23 | 119 |

### 统计（10 个全部有效样本）

| 指标 | 值 |
|:--|--:|
| **MAPE (Average AbsError %)** | **12.72** |
| Average Error % (signed) | -12.72 |
| Median AbsError % | 15.32 |
| Pearson correlation | 0.9987 |

## 串行运行时间（wall-clock seconds）

```
backprop        15s
bfs             10s
hotspot         46s
nw               8s
heartwall        5s
lud             11s
nn              15s
pathfinder       3s
srad_v2         13s
streamcluster  119s
--------------------
TOTAL          245s  (~4.08 min)
```

## 根因分析（与首次跑数据差异的定位）

首次跑出 MAPE=34.07%、nn 崩溃的结果，根因是 **`trace.config` 没有被加载**：

- `run_sim_all.sh` 只把 `gpgpusim.config` 复制到运行目录，但 `trace.config`（放在同目录下，包含 `-trace_opcode_latency_initiation_*` 参数）没有被命令行载入。
- Accel-Sim 官方脚本 `util/job_launching/run_simulations.py:330` 的做法是把 `trace.config` 的内容 **append 到 gpgpusim.config 末尾** 作为单一配置文件使用。
- 由于没有加载 `trace.config`，模拟器使用了内置默认值（例如 `int_latency` 默认 4,1 而 trace.config 是 2,2；`branch_latency` 默认 1,1 而 trace.config 是 2,1）。

### nn SIGSEGV 的触发链

1. `gpgpusim.config` 里 `-branch_latency 2` 导致 BRANCH 流水线 `m_pipeline_depth = 2`（`m_pipeline_reg.size() = 2`）。
2. `trace.config` 未加载时 `int_latency = 4`（默认值）。`CALL_OPS` / `RET_OPS` 指令从 `trace_config::set_latency` 获得 `latency = int_latency = 4`。
3. 在 `functional_unit::cycle()` (`functional_unit.cc:294`) 里：
   ```cpp
   int start_stage = m_dispatch_reg->latency - 1;  // = 3
   assert(start_stage >= 0);                        // ✓（只查下界）
   if (m_pipeline_reg[start_stage]->empty()) {      // ✗ 越界 → SIGSEGV
   ```
4. 越界访问 `m_pipeline_reg[3]`（vector 大小为 2），SIGSEGV。

### 修复方式

**短期**：命令行加两个 `-config`，同时加载两个文件：
```bash
accel-sim.out -config ./gpgpusim.config -config ./trace.config -trace ./traces/dynamic_trace.pb
```

**长期**（建议）：在 `functional_unit.cc:294` 加上上界 assert：
```cpp
int start_stage = m_dispatch_reg->latency - 1;
assert(start_stage >= 0);
assert(start_stage < (int)m_pipeline_depth);  // NEW: 上界检查
```
或者在 `trace_config::set_latency` 里加一个后置 clamp，让 latency 不超过对应 FU 的 pipeline depth，从而即使 trace.config 未加载也不会崩。

## 观察

1. **streamcluster** 和 **backprop** 最准确（<1% 误差）。
2. **nn** 是误差最大的（-27.76%），表现为 sim 过于乐观；可能是缺 atomic / dependent memory 延迟建模。
3. 所有 signed error 仍然是负号——sim 普遍偏快，说明仍然缺捕捉某些 HW 变慢效应（unified memory 一致性、带宽争用、LPDDR5 调度等）。
4. **Pearson 0.9987** 说明 sim 与 HW 的大小排序非常一致；偏差以系统性比例缩放为主，后续可以考虑统一乘一个系数或调 `-dram_latency` / `-gpgpu_l2_rop_latency` 来消除。
5. 串行总运行时间 245s ≈ 4 分钟，和原先 219s 差不多（加载 trace.config 后 hotspot/srad_v2/nn 等 latency 变大，运行时略变长）。
