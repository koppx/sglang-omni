# MiniCPM-o 单卡 A100 80GB 自动验收

面向 PR #1879（review 基线 `f3fedb25`）。不会修改被测模型代码，不需要在线 agent
持续值守，不会发消息、提交代码或创建 GitHub issue。当前仓库脚本可以在开发机上
直接运行；尚未提供开发机地址，因此没有远程部署或实测结果。

## 一次启动

在 Linux A100 80GB 开发机的本仓库根目录，用 Python **3.10–3.12** 运行：

```bash
python3 scripts/minicpm_a100/run.py --gpu 0
```

默认模型/数据缓存位于仓库 `.minicpm-cache/`，报告位于 `.minicpm-runs/<run-id>/`。
也可以指定有足够空间的目录：

```bash
python3 scripts/minicpm_a100/run.py --gpu 0 --cache-dir /data/minicpm-cache --output /data/minicpm-runs/run-001
```

远程 SSH 主机和目录不写死。启动后无需人工输入。希望断开 SSH 后继续运行时，可以
使用开发机现有作业系统，或者在 shell 中用 `nohup ... > launcher.log 2>&1 &` 启动。
HF 网络需要可用；依赖 gated/private 资源时应提前在环境中提供授权。本默认数据源为公开资源。

脚本自动创建独立 venv、安装仓库的固定核心依赖和 `minicpm-o` extra、执行 `pip check`、
记录完整 `pip freeze`。不安装系统 CUDA 驱动，不修改共享环境。驱动不兼容、空间不足、
GPU 忙、下载失败、依赖冲突均会明确失败并输出报告。建议至少预留 100 GiB。

## 默认实验

- 总墙钟预算 28,800 秒，包含环境准备与下载；保留清理/报告时间。阶段和单请求也有超时。
- LibriSpeech clean/other 各 100、FLEURS 中文 100；MMMU 按学科轮转抽取 100 个选择题；
  中英文 TTS 50；HF 文本/图像/音频参考各 10。
- 40 个基本功能请求和额外多轮标准 content block 检查，30 个异常/边界请求。
- 文本、音频输入、语音输出，短/长 workload，并发 1/2/4/8，每档预热后重复 3 次。
  每次 32 请求；混合稳定性 60 分钟。
- 3 次低频 py-spy 采样、request events、短 torch trace；独立文本配置、关闭 CUDA graph、
  原配置重启 A/A。三个对照的计时 workload 均为相同的短/长文本，额外运行对应功能回归。
- 单独缩小 thinker prefill chunk 到 256，运行 12 条长 prompt 语音请求，覆盖分块路径。
- 服务结束后顺序运行 HF encoder/生成对照和独立 Whisper-small 语音一致性评分，
  不让评测模型与服务争夺显存。

`config.json` 中质量阈值只是初始 smoke gate，不是官方性能承诺。样本不是完整 benchmark。
ASR 失败计为空转录，计入 deletion；MMMU 解析失败按错题，不随机猜答案。
文本相似度、ASR 回转不能代替数值一致性、人工音质或说话人身份评估。

下载时将 `main` 或转换分支解析成不可变 SHA，写入 `resources.json`，后续全程离线使用。
固定模型大文件 SHA256、样本 ID、随机种子及数据 SHA；下载中断由 HF cache 在下次运行复用。
FLEURS 使用其 `refs/convert/parquet`，避免依赖已废弃 dataset script。
要跨新运行严格复用模型版本，在 config 中把 `model_revision`/`asr_model_revision` 改成
已有报告的 SHA；数据版本可通过 `dataset_revisions` 映射固定。

## 失败处理与报告

- 首次默认配置失败后，仅尝试预设低显存诊断配置，原始失败不会消失。
- 运行中服务崩溃最多恢复一次。独立实验继续；失败和超时保留原始请求、返回及堆栈。
- 超时会终止本次子进程组，不使用 `pkill`，不终止其他用户的进程。
- `report.md` / `report.html`、`summary.json`、`junit.xml`、各阶段日志、逐请求 JSONL、
  GPU CSV、音频和 profile 证据均保留。所有阶段状态每次更新时落盘。
- 退出码：`0` = 必测项 PASS，`1` = 存在 FAIL，`2` = 仅存在 INCOMPLETE。
- 机器掉电 / SIGKILL 无法运行 finally。再次启动后可执行：

```bash
python3 scripts/minicpm_a100/run.py --recover-report /data/minicpm-runs/run-001
```

它从日志状态补出中断报告，不自动重新执行完成项，也不向旧 PID 发信号。完整重跑使用新
输出目录并复用 cache。禁止复用旧输出目录覆盖证据。

profile HTTP 接口为异步广播，脚本在停止后等待文件写出并检查 events/trace。
不存在 DCGM/nsys 硬件计数器时不会把 nvidia-smi 利用率叫作 SM/Tensor Active。
目前自动分析输出可追溯的阶段数据、CPU 热点、trace busy union、A/A 噪声和 A/B 方向，
不会声称已经证明某个内核优化有效。

## 本地检查与 review probes

```bash
python3 -m unittest discover -s scripts/minicpm_a100 -p test_runner.py -v
python3 scripts/minicpm_a100/run.py --print-plan
python3 scripts/minicpm_a100/review_probes.py --output /tmp/minicpm-review.json
```

前两个不需要 GPU 或下载。review probes 需要 CPU PyTorch，隔离执行生产函数以定位
逻辑问题，已知问题存在时**预期返回 1**，不会因为自动流程完结而伪装测试通过。
模型、SGLang/CUDA 集成和真实数据下载仍以开发机执行结果为准。
