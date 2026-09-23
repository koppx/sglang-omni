# MiniCPM-o 4.5 Baseline (A100)

端到端推理基线脚本，方法论对齐 [issue #2273](https://github.com/sgl-project/sglang-omni/issues/2273)（BruceLoveDecimal 的 Runtime Profiling）。用于在改代码前固化一组「改前」数字，改完后用同一张卡复跑做 before/after 对比。

## 用法

```bash
# 默认：GPU 0，端口 30000，并发扫 1/2/4/8/16/32
bash scripts/baseline/run_minicpm_o_baseline.sh

# 指定某张 A100（你有 8 张，挑一张跑）
GPU=3 bash scripts/baseline/run_minicpm_o_baseline.sh

# 先小样本冒烟（只跑 16 条、单点并发）
CONCURRENCY=1 MAX_SAMPLES=16 bash scripts/baseline/run_minicpm_o_baseline.sh
```

产物落在 `results/minicpm_o_baseline_a100/`：

| 文件 | 内容 |
|---|---|
| `fingerprint.txt` | GPU 型号/显存/驱动、git commit、模型路径、采样参数（对应 #2273 的 Baseline Fingerprint） |
| `server.log` | 服务端日志（含启动时的 `avail_gpu_mem` 显存日志） |
| `c<c>/` | 每个并发点的结果（req/s、audio s/s、E2E p50/p95/p99） |

## 与 #2273 的对齐点

| 维度 | #2273 | 本脚本 |
|---|---|---|
| 模型 | `openbmb/MiniCPM-o-4_5` | 同（可 `MODEL_PATH` 覆盖） |
| 数据集 | `zhaochenyang20/seed-tts-eval`，英文 1088 条 | `zhaochenyang20/seed-tts-eval-arrow`（当前 main 的标准 arrow 版，内容同为英文集；首次运行自动从 HF 下载） |
| 并发 | 1/2/4/8/16/32 | 同（`CONCURRENCY`） |
| 采样 | `max_new_tokens=256`、`temperature=0.7`、非流式、voice clone | 同 |
| 指标 | req/s、audio s/s、HTTP E2E 延迟 | 同（`--generate-only`，跳过 WER） |
| thinker 配置 | max_seq 8192、`mem_fraction_static=0.55` | 同 |
| talker 配置 | `mem_fraction_static=0.15` | 同 |
| 线程 | `OMP_NUM_THREADS=4` | 同 |
| 硬件 | **1× H100 SXM 80GB** | **1× A100 80GB**（单机 8 张卡中任选 1 张） |

## 重要说明

- **绝对数字不要直接和 #2273 比**：H100 带宽/算力都高于 A100，吞吐会低一截。本脚本的价值是**你自己 before/after 在同一张 A100 上自洽对比**；趋势可与 #2273 对照。
- **单卡 DP1**：默认只占一张卡（`CUDA_VISIBLE_DEVICES=GPU`），和 #2273 的 DP1、all stages on GPU 0 对齐。不要为了"用满 8 卡"而改多卡部署——那会引入 TP/DP 变量，和 #2273 不可比。
- **先冒烟再跑全量**：首次跑建议先 `CONCURRENCY=1` 确认服务能起、请求能通，再放全量 1088 条 × 6 个并发点。
- **OOM 注意**：#2273 里 c8 之后曾 OOM，他们是每个点重启服务。本脚本默认一个服务跑全程；若遇到 OOM，按并发点拆成多次运行即可。
- **跑两次**：改代码前跑一次（存档），改完跑第二次，diff 两个 `c<c>/` 的汇总就是优化收益。
