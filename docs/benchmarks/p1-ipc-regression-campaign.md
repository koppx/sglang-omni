# P1(GPU IPC handoff)c32 退化定位 campaign

任务书(2026-10-13,用户):暂停全量,优先定位并修复 P1 的 c32 -10.5% 退化。
流程:存档 → 清场 → 独立 worktree 加探针(不改同步/生命周期)→ 串行 c32
P0/P1 对照(预热、同输入、同口径)→ 最小修复 → 关探针验证 → 正式全量
(符合复用条件的旧结果保留)。跟踪:Task #8-#12。

## 存档(2026-10-13 上午)

### 代码与环境
- 分支 `perf/minicpm-o-thinker-talker-handoff` @ **`4823570c8`**(= upstream `76b3aebb5`
  + 我方 3 提交:`5f5ee7101` 测试 ack endpoints / `62b73637e` poll 120s /
  `4823570c8` driver 回收逃逸 GPU worker;均已 push origin)
- venv:`/data/enccache2284/handoff-cache/environments/4823570c8.../`(py3.12,
  sglang 0.5.20 系列 278 包,torch 2.13.0+cu130)
- 数据:measurement.lst/warmup.lst(seed 42 holdout 16)在两个 run 目录内,
  内容一致(driver 生成);meta 源 = /data/enccache2284/data/seed-tts-eval/en/meta.lst
- 模型:openbmb/MiniCPM-o-4_5 @ 503e7542(本地快照)
- 节点:scrp-node-21 物理 GPU 0(H100 NVL 94GB, driver 610.43.02)
- driver sbatch:/data/enccache2284/scripts/handoff_driver.sbatch(含 FFmpeg
  LD_LIBRARY_PATH、物理卡号解析)

### 已完成数据(保留,不无条件重跑)
- run `handoff-results/20261012-235705/`:**42/57**(r1/r2 全部 + r3 的 c1×3、c2×2)
  失败点:full-p1-c2-r3 admission(残留 worker,已修)
- run `handoff-results/20261013-065002/`:**19/57**(r1 至 c32:baseline-c32 完成)
  **中断点:full-p0-c32-r1 运行中被用户 scancel(09:52:23),该 trial 未完成不计入**
- 每个目录已生成 completed_trials.txt;report.json 含全部 metrics
- 初步结论(42 样本):p0 全档 +1~3%;**p1 c32 -10.5%(13.49 vs 15.07 qps,n=2)**

### 复用条件(正式全量时)
- arm=baseline/p0 且并发档 ≤ 该 run 已完成档:两 run 数据可复用
- p1 全档与 r3 缺口:重跑(修复后)
- 复用与重跑数据须同 SHA 尾缀记录(修复提交后 candidate 变化,严格对照只复用
  baseline/p0 于同代码基线——若修复触及 thinker/talker 路径则 p0 也须重跑)

## 诊断探针(p1-diag 分支第 1 提交,worktree handoff-p1-ws)
comm_trace 全事件注入 pid/tid;cuda_ipc_receive_done(queue_wait_ms/import_ms);
comm_payload_device_ready 增 clone_ms/sync_wait_ms;stage_ipc_receive(receive_ms);
minicpm_talker_prefill(batch_requests/prefill_rows/fenced_payloads/fence_wait_ms/
prefill_ms);minicpm_talker_request_done(codec_ids)。全部 env 门控(SGLANG_OMNI_COMM_TRACE),
不改同步/生命周期。

## 定位结论(c32 A/B,job 428567,2026-10-13)

性能: p0 16.19/16.35/16.61 qps vs p1 14.79/15.30/13.56(-8~18%,p95 更差)
—— 复现并解释全量数据的 p1 c32 -10.5%。

分解(每 512 样本,chunk TTS → 864 个 talker 请求,每 chunk 一次 hidden handoff):
- **IPC import 5ms/次 × 864 次 ≈ 4.3s 纯串行开销(talker 单 pid)**,占差异 ~80%
- import 内部: clone 0.13ms + event sync 0.75-1.07ms + queue 0.25ms +
  **~3.7ms 在每张量一次的 cudaIpcOpenMemHandle(pickle IPC handle 同步 driver 调用)**
- 张量中位仅 128KB:CPU 往返(p0)≈0.2ms,比 open 便宜 25 倍
- 排除项: fence 等待(0.02-0.03ms 两臂同)、prefill 批形态(同 bs=1)、
  线程池排队(0.25ms)、竞态
- direct_cuda_ipc_targets 的设计本意是"同卡零拷贝"——但零拷贝只在 open 成本
  可摊薄的大张量成立

## 最小修复(p1-diag 分支,COMMIT 待填验证后)

`fix(comm): gate direct CUDA IPC payloads by size`:
- runtime 决策链加 `payload_below_ipc_min_bytes` 门控;
  `DIRECT_CUDA_IPC_MIN_PAYLOAD_BYTES` 默认 8MiB(env SGLANG_OMNI_DIRECT_IPC_MIN_BYTES)
- 小张量 fallback 走池化 CUDA_IPC relay(槽位复用,同卡 D2D 微秒级),
  不是每张量一次 handle open;≥8MiB 保留零拷贝 direct IPC
- 辅助: stage_io.payload_cuda_tensor_bytes()

## 验证结果(关探针,job 430591,2026-10-13)

| 臂 | qps | p95 | p99 | 失败 |
|----|-----|-----|-----|------|
| p0_r1 | 17.53 | 2.55 | 3.62 | 0 |
| **p1_r1(修复后)** | **17.14** | 2.60 | 3.48 | 0 |

p1 与 p0 差 2.2%(噪音带内),修复前为 -8~-18% → **修复验证通过**。
r2 双臂因 /data 磁盘满中断(wav 写入 ENOSPC,非修复问题);已清理:
3 个旧 candidate venv(33G)+ 4 个 run 的 audio wav(~12G,metrics 已在
report.json)。旧 run 目录保留全部 log/report/事件。

## 上游化
- 修复已 cherry-pick 回 handoff 分支:`2f830f818`(已 push)
- p1-diag 分支保留探针提交(a5a1b82b6)作诊断工具

## 正式全量(job 430600,candidate 2f830f818)
- c32g8 QoS(避开 tmux job 的 c16 配额),全套 smoke+54 trial
- 复用策略:修复对 baseline/p0 语义无差(阈值仅作用于 p1 的 CUDA 张量
  分支;CPU 张量走原 no_cuda_tensor 路径),旧 42+19 trial 的 baseline/p0
  数据与新 run 交叉验证后可并入最终报告;p1 以新 run 为准
- 注意:磁盘 130G 余量,driver 会再产 ~10G wav + 新 venv(uv 缓存在,快)
