# MiniCPM-o 单卡 A100 80GB 自动回归与验收

针对 PR #1879 和 run-004 暴露的问题。脚本在被测仓库上运行，记录实际 HEAD、未提交 diff、运行脚本副本、模型文件哈希和两套依赖清单。不会切换代码分支，不会自动修复模型代码，不需要在线 agent 值守。

本次修改**未在本地运行测试或 GPU 实验**；请在开发机验证。HF 独立环境和诊断 hook 是待开发机验证的实现，不代表已通过兼容性或精度验收。

## 开发机执行

先将整个 `scripts/minicpm_a100/` 同步到准备测试的最新仓库，包含新增文件和 `alignment_hook/` 子目录。只复制 run.py 不够；git pull 不会同步本地未提交改动。Linux、Python 3.10–3.12、单卡 A100 80GB，建议至少 100 GiB 可用空间；两个 venv 会增加依赖下载与磁盘占用。

已有完整模型目录：

```bash
python3 scripts/minicpm_a100/run.py \
  --mode bug-regression \
  --gpu 0 \
  --model-path /data/models/MiniCPM-o-4_5 \
  --cache-dir /data/minicpm-cache \
  --output /data/minicpm-runs/run-005
```

替换上面的目录为开发机实际目录。`--model-path` 跳过主模型的 Hub 查询和下载，仍校验权重与 `assets/token2wav` 并记录文件哈希；Whisper 和数据集仍按需下载。也可在 config 中设置 `asr_model_path` 复用本地 Whisper。未给 model-path 时自动下载主模型。

后台执行可用现有 Slurm 作业，或：

```bash
nohup python3 scripts/minicpm_a100/run.py --mode bug-regression --gpu 0 \
  --model-path /data/models/MiniCPM-o-4_5 --cache-dir /data/minicpm-cache \
  --output /data/minicpm-runs/run-005 > run-005-launcher.log 2>&1 &
```

输出目录必须是新目录，避免覆盖旧证据。跨运行严格固定模型版本：使用相同本地模型目录或在 config 中填入先前 resources.json 的模型 SHA，不使用漂移的 main。

## 两种模式

默认 `--mode bug-regression`：

1. 服务端依赖审计、资源准备；单独安装 HF 参考环境并先验证加载与最小生成。
2. 原有 MiniCPM 单元测试、四项 CPU 逻辑探针。已知 bug 未修时正常记录 FAIL，不放宽断言。
3. 空输出、取消恢复、标准多模态块、边界测试各自使用全新服务实例，前后检查 health 和实际文本请求。
4. 不分块与 chunk=256 两套服务执行同样的三个长 prompt；诊断 hook 保存 token 和 hidden，独立比较实际对齐。
5. 长语音并发 1/2/4/8 各使用新服务，每档在同一服务上连续 3 轮、每轮 32 请求，区分新启动容量与累积压力。记录全卡及进程显存；没有把增长直接认定为泄漏。
6. 独立执行功能、ASR、MMMU、TTS、参考请求采集。样本规模保持 clean/other/中文各 100、MMMU 100、TTS 50、参考每模态 10。
7. 停止服务后依次导出原生编码器结果、HF 参考结果、做精度比较与 TTS 双层评分。

`--mode full` 在以上关键回归全部通过后，继续原有 performance、1 小时 soak、profiling、分块语音完成性和 A/B、A/A。关键项失败时这些阶段标记 BLOCKED，避免把不正确或不稳定的运行作为正式性能验收。

总墙钟预算仍是 8 小时，包含安装、下载、独立服务重启和清理；不保证预算内所有阶段都能完成。未执行项保留 BLOCKED/INCOMPLETE，不能当作通过。

## 依赖策略

默认 `minicpmo-override`：核心依赖跟随被测仓库，`minicpmo-utils==1.0.6` 通过 `--no-deps` 安装，Token2wav 必需依赖单独补齐。`dependency_check.py` 同时检查基础依赖和该包的 tts extra，并保存 `results/dependency_audit.json`。

仅允许 minicpmo-utils 对 librosa、Pillow、Transformers、soundfile、onnxruntime 的已知版本覆盖，以及不在本轮 Token2wav 路径使用的 moviepy/decord 缺失；所有记录保留。缺少其他运行库、其他包的冲突或 Token2wav 导入失败仍阻止运行。onnxruntime 可能由仓库的 onnxruntime-gpu 提供实际模块。

environment PASS 表示安装符合选定覆盖策略且基本导入/CUDA 检查通过，**不表示原始包依赖完全一致**。查看 `dependency_consistent` 和具体 issues。

严格验证未经覆盖的安装方式：

```bash
python3 scripts/minicpm_a100/run.py --dependency-policy strict --gpu 0
```

当前声明有冲突时该方式应失败，不会自动悄悄降级。

HF 参考环境独立固定 Transformers 4.52.4，服务环境保持仓库版本。两个环境通过落盘 Tensor/JSON 交换结果，不同时占用 GPU。记录 `dependencies.txt` 和 `reference-dependencies.txt`；辅助库实际解析版本以清单为准。参考加载失败时后续对照标记 BLOCKED，其他独立实验继续。

## 精度与诊断的边界

- **四项逻辑探针**：针对真实生产函数做隔离验证，不能替代 GPU 集成测试。
- **alignment_compare**：检查实际中间 chunk 是否错误追加 hidden、结束时 token/hidden 数量、两种分块配置的 prompt/token/hidden。没有记录到中间 chunk、缺失诊断数据或 hook 加载失败均不算通过。不同 greedy token 轨迹时不能直接做 hidden 数值比较，记录为需要调查。诊断 hook 只在两次 alignment 服务启用，保存 CPU 副本，增加开销，不用于性能结论。
- **参考比较**：先在服务依赖环境导出原生预处理和编码器结果，再由独立 HF 环境导出同输入结果。检查 input IDs、形状、有限性和 cosine。生成文本的相似度仅为初筛，不等同于 logits 精度。
- **thinking/模板**：配置的 HF thinking 默认 True，与当前 MiniCPM 服务默认模板对齐；不向不支持该字段的服务假装传递开关。实际追踪 HF chat 的 processor 输入，必须与服务预处理输入一致；不一致时导出失败并阻止精度对照，不能算成模型精度退化。改变模型模板或 thinking 后必须重新验证该检查。
- **TTS 两层评分**：保存目标文本、服务返回文本、音频回转录文本。分别计算 `target_to_text_error`（是否逐字朗读，默认门槛 0.05）和 `text_to_audio_error`（文本—语音一致性，默认门槛 0.35）。记录两层 corpus error、评分错误和分层失败数。回转录依赖 Whisper 与文本归一化，不是人工 MOS/音色身份验证；不兼容缺少 target_text 的旧请求日志。
- **空输出**：允许明确的 2xx 空结果或 4xx，但不允许 500 或服务死亡；这项不检查空片段音质。
- **content blocks**：图片含内容语义与多轮顺序断言；标准音频块检查请求完成，音频转录精度另由 ASR 评测，不能把返回了文本当成正确理解音频。

## 失败与报告

- 每个独立实验使用新服务；失败记录不因下次启动成功而消失。启动不可用为 BLOCKED，运行错误为 FAIL；超时为 INCOMPLETE。
- 连续两次传输错误触发熔断，剩余请求记为 blocked，不再实际发送。500 后额外检查服务 health；不会自动重放已失败请求。
- `counts` 区分 sent、passed、failed、blocked、transport failure、响应断言失败及截断。性能 points 的正式测量失败数另列，避免混入 warmup。保留兼容字段 `failed` 作为阶段门槛汇总，**不能把它当作失败请求数**。
- `report.md` / `report.html`、`summary.json`、`junit.xml`、阶段日志、逐请求 JSONL、音频、原生/HF Tensor、alignment 记录、gpu.csv 和 gpu-processes.csv 均保留。大 JSON 仅摘要展示，完整内容通过链接查看。
- 退出码 0=PASS，1=存在 FAIL，2=只有未完成/阻塞。清理只停止本次拥有的进程组，不杀其他用户进程。
- 不采集 DCGM/nsys 时，不把 nvidia-smi 利用率当作 SM/Tensor Active。

中断后仅重建报告：

```bash
python3 scripts/minicpm_a100/run.py --recover-report /data/minicpm-runs/run-005
```

完整重跑用新输出目录并复用缓存。请回传整个运行目录，尤其是 `results/`、`requests/`、两套依赖清单和服务日志，而不仅是 report.md。
