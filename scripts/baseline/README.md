# MiniCPM-o 4.5 Baseline (A100)

端到端推理基线脚本，方法论对齐 [issue #2273](https://github.com/sgl-project/sglang-omni/issues/2273)。它是为 #2284 的 **Encoder caches + Talker** 优化服务的"before/after 尺子"：改代码前跑一次存档，改完在同一张卡复跑，diff 结果就是优化收益。

## 用法

```bash
# 默认：GPU 0，端口 30000，并发扫 1/2/4/8/16/32，只测速度
bash scripts/baseline/run_minicpm_o_baseline.sh

# 指定某张 A100（单机 8 张卡中任选 1 张）
GPU=3 bash scripts/baseline/run_minicpm_o_baseline.sh

# 冒烟：只跑 16 条、单点并发（先确认链路通）
CONCURRENCY=1 MAX_SAMPLES=16 GPU=3 bash scripts/baseline/run_minicpm_o_baseline.sh

# 速度 + 质量（WER）：优化完用来证明没改坏音质
WITH_Q=1 GPU=3 bash scripts/baseline/run_minicpm_o_baseline.sh
```

国内首次跑需先设 HF 镜像（数据集自动下载）：
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## 产物（`results/minicpm_o_baseline_a100/`）

| 文件 | 内容 |
|---|---|
| `fingerprint.txt` | GPU/驱动/git commit/采样参数，以及**空闲 vs 加载后**显存对比 |
| `server.log` | 服务端日志（含启动显存、报错） |
| `c<c>/run.log` | 每个并发点的汇总（req/s、audio s/s、p50/p95、TTFT）——**before/after 就 diff 这个** |
| `c<c>/eval_results.json` | 仅 `WITH_Q=1` 时有：WER、ASR 速度 |

## 与 #2273 的对齐点

| 维度 | #2273 | 本脚本 |
|---|---|---|
| 模型 | `openbmb/MiniCPM-o-4_5` | 同（可 `MODEL_PATH` 覆盖） |
| 数据集 | `zhaochenyang20/seed-tts-eval`，英文 1088 条 | `zhaochenyang20/seed-tts-eval-arrow`（当前 main 标准 arrow 版，内容同为英文集） |
| 并发 | 1/2/4/8/16/32 | 同（`CONCURRENCY`） |
| 采样 | `max_new_tokens=256`、`temperature=0.7`、非流式、voice clone | 同 |
| 指标 | req/s、audio s/s、HTTP E2E 延迟、TTFT | 同（`--generate-only` 速度；`WITH_Q=1` 加 WER） |
| thinker 配置 | max_seq 8192、`mem_fraction_static=0.55` | 同 |
| talker 配置 | `mem_fraction_static=0.15` | 同 |
| 线程 | `OMP_NUM_THREADS=4` | 同 |
| 硬件 | **1× H100 SXM 80GB** | **1× A100 80GB**（单机 8 张卡中任选 1 张） |

## before/after 怎么用

1. **改代码前**：`GPU=3 bash ...`，把 `results/minicpm_o_baseline_a100/` 整个目录改名存档（如 `results/baseline_before/`）。
2. **改完代码**：同一张卡 `GPU=3 bash ...` 再跑一次。
3. **对比**：
   ```bash
   diff <(grep -E "throughput|p50|TTFT" results/baseline_before/c16/run.log) \
        <(grep -E "throughput|p50|TTFT" results/minicpm_o_baseline_a100/c16/run.log)
   ```
   吞吐涨了多少、p50 降了多少，一目了然。
4. **验证没改坏音质**：优化后跑一次 `WITH_Q=1`，对比 WER 是否持平。

## 重要说明

- **绝对数字不要直接和 #2273 比**：H100 比 A100 快，吞吐会低一截。价值在**你自己 before/after 同卡自洽**。
- **单卡 DP1**：默认只占一张卡。不要为了"用满 8 卡"改多卡——会引入 TP/DP 变量，和 #2273 不可比。
- **先冒烟再跑全量**：`CONCURRENCY=1 MAX_SAMPLES=16` 确认服务能起、请求能通，再放全量。
- **OOM**：#2273 里 c8 后曾 OOM，他们按点重启服务。本脚本一个服务跑全程；若 OOM，按并发点拆成多次跑。
- **Encoder cache 命中数不在本脚本里**：当前 `StageOutputCache` 没有 hit/miss 计数器。这是你做 Encoder caches 时要**在代码里加的埋点**（不是基线脚本的事）；加完后把计数从 `server.log` 摘出来，和本脚本的吞吐一起看。
