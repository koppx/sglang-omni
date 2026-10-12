# MiniCPM-o handoff 一键远端验证

## 目的与交付

本节目的：用一个入口完成由小到大的远端验证，并明确报告能证明什么。

入口为 `scripts/test_minicpm_handoff.py`。脚本复用已经提交的 7 个回归测试文件，不复制其实现；收到远端报告后可以删除这个临时入口和本说明，常规回归测试继续保留。脚本不会自动删除测试代码或结果。

一次启动后无需逐步确认：小规模检查通过才进入大规模测试，失败或超时立即停止后续阶段，保留原始证据并更新 `report.md`、`report.json`。退出码 0 表示本次生成验证完成，不表示音频质量、完整模型张量一致性或性能收益已经验收。

## 执行顺序

本节目的：让昂贵的全量实验只在基本正确性和真实服务冒烟通过后开始。

| 阶段 | 默认工作量 | 进入下一阶段的条件 |
|---|---|---|
| 自动准备 | 创建 Python 3.12 隔离环境、安装项目依赖；补齐模型和数据；切出 16 条预热 | 安装／下载成功，资源清单落盘；缓存可供重跑复用 |
| 环境和输入 | Linux、单张 CUDA GPU、依赖、源码导入路径、数据与参考音频 | Torch 2.13.0、SGLang 0.5.21；测量集与预热集无 ID、文本或参考音频内容重叠 |
| CPU 小规模回归 | 21 个 handoff／projection 用例 | 全部执行通过 |
| GPU 核心回归 | 每轮 5 个展开用例，重复 3 轮；IPC 用例内部有 8 轮回收 | graph replay、三 stream conditioning、relay、跨进程 IPC 全部实际执行；跳过视为失败 |
| 真实服务小样本 | A/B/C 各 8 条，并发 1；每组独立预热 | 请求全部成功、音频时长有效、WAV 存在；C 组日志出现 direct CUDA IPC 接收完成事件 |
| 完整单测 | 7 个已有测试文件 | pytest 成功，JUnit 无失败或错误；通用可选后端的 skip 留在 XML 中，核心 CUDA 用例已由上一阶段严格检查 |
| 全量性能 | 切分后全部测量行；并发 1/2/4/8/16/32，每点每组 3 次 | 每条请求成功并产生有效结果，否则停止并报告 |
| 汇总 | 每阶段更新，结束生成完整表格 | 保存原始数据及相对变化，不自动声称统计显著或音频质量通过 |

GPU 测试保持默认异步执行。脚本移除子进程的 `CUDA_LAUNCH_BLOCKING`、`PYTHONOPTIMIZE` 和 `PYTEST_ADDOPTS`，防止同步模式、关闭断言或外部筛选参数改变验证范围。

默认一共启动 57 次服务：3 次冒烟和 54 次全量实验。如果原始英文数据是 1,088 条，切出 16 条预热且没有额外重叠，正式测量为 1,072 条，对应 57,912 条测量请求，另有每次 16 条预热。这是完整验收规模，不是几分钟的快测。启动超时默认 1,200 秒，每个测试／benchmark 子进程超时默认 7,200 秒，每个安装／资源准备子进程超时默认 14,400 秒，可用参数调整。

## 基准与其他 PR 的关系

本节目的：固定三组对照及测量定义，让收益能归因到具体改动。

| 组别 | 源码和开关 | 比较 |
|---|---|---|
| A：baseline | `7f78bf3dc5c3053d6eb270764eb19232b617dec4` | 优化前基线 |
| B：p0 | 启动脚本时的已提交 HEAD，`enable_gpu_handoff=false` | B/A：连续 Tensor 与投影优化 |
| C：p1 | 同一 HEAD，`enable_gpu_handoff=true` | C/B：GPU handoff；C/A：总体 |

脚本创建两个临时 detached worktree，不切换当前分支；各服务从相应源码目录启动。三组 benchmark 客户端都使用候选版本的同一个 SeedTTS 实现，避免客户端版本造成口径差异。三组共享自动创建的 Python 环境，或通过 `--use-current-env` 复用当前已安装环境。运行前要求已跟踪文件无未提交改动。浅克隆缺少基线对象时，脚本尝试从 `origin` 获取固定基线 SHA。

每次实验都重启服务、使用独立预热集。三轮依次采用 A/B/C、B/C/A、C/A/B 顺序；相同 checkpoint、数据顺序、seed 42、temperature 0.7、max_new_tokens 256、Thinker/Talker 显存比例 0.55/0.15、`OMP_NUM_THREADS=4`。其余模型设置沿用这两个版本的配置，本次 handoff 改动没有调整采样策略。

这套设置沿用旧 #2330 说明中引用的 #2273 的生成参数。自动下载模型固定为 `openbmb/MiniCPM-o-4_5@503e754207c94da6bb26850b4469f367c9ea3582`，与旧 #2273 的 checkpoint revision 相同。自动下载数据使用仓库固定的 `zhaochenyang20/seed-tts-eval-arrow@27f4c1adee83b5b29b7c4b375f6b976324bda308`；旧原始 dataset revision 为 `8f5e1aa2a35d42f42e940074c1983358b9491f89`，不能据此声称数据快照完全相同。

默认切出预热集后，测量集也不再是原始 1,088 条；报告记录真实条数。若要对齐原始 workload，需提供相同 checkpoint、原始全量 `--meta` 以及独立的 `--warmup-meta`。旧实验的依赖、预热和服务重启策略不同，其历史数字不作为本次 A 组。

脚本记录候选 SHA、固定基线、模型路径及 config SHA256、数据与实际参考音频内容的联合 SHA256、依赖、GPU 和每次命令。checkpoint 权重应使用不再修改的本地 snapshot；config 哈希不是完整权重哈希。数据行数按输入记录，脚本不会把任意 metadata 冒称为原始 1,088 条基准。

## 一次启动

本节目的：新机器只有项目仓库时，也能自动补齐 Python 依赖、模型和评测数据。

先拉取当前 fork 分支。机器需要 Linux、可运行的 Python 3.10+（启动器）、git、空闲 NVIDIA GPU、兼容项目 CUDA 13 栈的驱动，以及访问 Python 包源和 Hugging Face 的网络。没有 `uv` 时，脚本通过启动器的 `venv`／pip 安装它，再由 uv 创建 Python 3.12 环境并执行 `uv pip install --prerelease=allow -e '.[minicpm-o]'`。启动器的 `venv` 模块也需可用。

仓库根目录一条命令后台启动：

```bash
nohup python3 scripts/test_minicpm_handoff.py --gpu 0 > /tmp/minicpm-handoff.log 2>&1 &
```

默认结果保存在 `handoff-results/<时间戳>-<PID>/`，启动日志会打印完整路径。脚本自动下载固定模型和当前语言的固定 SeedTTS 数据，并持久化参考音频。用 seed 42 从原始行号确定性抽出 16 条预热，再保持原始顺序生成测量集；测量集中与预热 ID、目标文本或音频字节 SHA256 重叠的行会额外剔除，数量记录在 `resources.json`。条数不足会报错。

默认缓存为 `~/.cache/sglang-omni/handoff`，可用 `--cache-dir /data/cache/handoff` 改到大磁盘；设置了 `HF_HOME`／`HF_HUB_CACHE` 时沿用该模型缓存配置。下载使用 Hugging Face 的缓存与重试能力，失败重跑会复用已下载文件。按候选 SHA 隔离 Python 环境；同一缓存同时只允许一个运行。资源准备有独立日志和超时，可用 `--setup-timeout-seconds` 调整。首次运行会下载完整权重和依赖，需预留充足磁盘和时间。

脚本不会安装 NVIDIA 驱动或通过 sudo 配置 UCX 等系统依赖。相关原生运行库仍需由机器镜像提供，参见[安装说明](../get_started/installation.md)；缺失时会停止并保存失败日志。它能补齐项目级依赖和资源，不能把任意空白系统自动变成可用的 CUDA 主机。

已有资源时可显式复用，以下路径按机器修改；结果目录必须不存在：

```bash
nohup python scripts/test_minicpm_handoff.py \
  --checkpoint /data/models/MiniCPM-o-4_5 \
  --meta /data/seedtts/en/meta.lst \
  --warmup-meta /data/seedtts/warmup/meta.lst \
  --output /data/results/minicpm-handoff-run1 \
  --use-current-env \
  --gpu 0 \
  > /tmp/minicpm-handoff-run1.log 2>&1 &
echo $! > /tmp/minicpm-handoff-run1.pid
```

三个资源路径都可独立省略：缺 `--checkpoint` 就下载模型，缺 `--meta` 就下载数据，缺 `--warmup-meta` 就从测量来源切分。显式提供但不存在的本地路径会立即报错，不覆盖原文件。提供独立预热时保留原测量集，若两者重叠则报错；预热集至少 16 条，测量集大于 8 条。生成的 `measurement.lst`、`warmup.lst` 都使用绝对音频路径。

此命令自动完成所有阶段。只想先检查执行矩阵，用 `python3 scripts/test_minicpm_handoff.py --dry-run`；它不安装、不联网、不创建输出目录，也不检查 GPU 或加载模型，不属于验证通过。

可以使用 `--concurrencies 1 4 --repeats 1` 缩小性能矩阵检查流程，但报告中的一次重复不能替代默认三轮正式验收。可以通过 `--lang zh` 测试中文固定数据，三组必须一致。

需要中止时：

```bash
kill -TERM "$(cat /tmp/minicpm-handoff-run1.pid)"
```

脚本捕获终止信号，清理自己启动的服务／benchmark／监控进程组和临时 worktree，并记录失败状态。它不使用全局 `pkill`，不终止已有服务；端口被占用时失败。SIGKILL、机器断电或磁盘写满无法保证清理和最终报告。正常失败后使用新的结果目录重跑，不隐式拼接不同实验。

## 看报告与回传

本节目的：无需盯日志，就能区分运行进度、失败位置和可比较的性能结果。

查看输出目录的 `report.md` 或 `report.json`：`RUNNING` 是当前阶段，`FAILED` 表示停止，最终 `COMPLETED: generation validation; quality review pending` 表示生成验证完成。启动日志也会输出当前阶段。

| 文件 | 内容 |
|---|---|
| `report.md`、`report.json` | 每次 E2E p50/p95、req/s、audio seconds/s、平均音频长度；三轮中位数及 B/A、C/B、C/A 变化 |
| `environment.json`、`packages.log`、`gpu.log` | 输入指纹、实际导入源码、依赖版本、GPU／驱动信息 |
| `resources.json`、`measurement.lst`、`warmup.lst` | 模型／数据来源和 revision、切分策略、实际条数、额外剔除数量、样本 ID、与路径无关的内容 SHA256，以及本次实际使用的两份 metadata |
| `bootstrap-*.log`、`environment-*.log`、`resources.log`、`downloads.json` | 环境安装／检查、下载日志和原始资源路径；某些文件仅在需要准备时生成 |
| `unit-small.xml`、`gpu-core-*.xml`、`unit-full.xml` 及同名日志 | 真实执行、失败、skip 和错误详情 |
| `smoke-*`、`full-*` | 每次实验的服务与客户端日志、`speed_results.json`、逐请求 CSV、WAV |
| 各实验下 `gpu-memory.csv` | 每秒采样的所选 GPU 总显存 MiB 和利用率；包含预热，不能等同进程 allocator 峰值或精确瞬时峰值 |

小样本开启 COMM_TRACE 确认 P1 路径；全量关闭它。报告中的正式比较只使用 full 结果。吞吐变化为正更好，p95 变化为负更好；同时检查音频长度，避免把短输出当成加速。

这里的 E2E 是非流式请求完成延迟，不是流式音频首包延迟或文本 TTFT。脚本不测 handoff 的精确 GPU 完成耗时，也不自动运行 ASR、相似度、UTMOS 或全模型 hidden／codec 逐值比较。保留 WAV 后可以在生成结束后单独评估质量，避免 ASR 抢占测量 GPU。性能有提升也不能据此自动默认启用 P1。

先回传 `report.md`、`report.json`、环境文件和失败阶段日志；进一步分析需要原始 speed JSON、CSV 和相应 WAV。收到报告并完成分析后，再删除临时入口及本说明；结果文件和常规单测不会随入口一起删除。

## 本地交付验证

本节目的：区分已验证的自动化行为与尚未运行的 GPU／模型实验。

2026-10-12 在 macOS 的隔离环境中，临时控制测试 27 项通过：阶段排序、三组轮换、JUnit skip／失败门禁、benchmark 部分失败／空音频／缺文件／非法数值、输出目录保护、子进程超时、失败停止、报告落盘，以及真实 HTTP 子进程的启动／关闭。HTTP 用例使用轻量服务替身，只验证测试入口的进程控制，不代表模型推理通过。临时控制测试保存在本地临时目录，不随本次提交发布。

既有 CPU handoff／projection 子集再次执行：21 passed、10 deselected。入口的 `--dry-run` 和针对新增文件的 pre-commit 检查通过。全仓 pre-commit 的可执行检查通过，Rust hook 因本机缺少 `cargo` 返回 127，未把全仓检查标记为通过。本次没有确认新的业务源码缺陷，也没有修改推理代码或迁移原来的 7 个单测文件。

本轮资源兜底扩展另通过 14 项临时控制检查，覆盖：1,088 → 16 + 1,072 的确定性切分、音频内容去重、外部预热、坏 metadata／缺失或空音频、样本不足、下载子进程结束后的音频持久化、缓存与本地资源复用、环境安装分支、缓存互斥、失败报告和无资源 dry-run。下载器以本地 fixture 代替远程服务；安装分支检查命令编排，不代表真实 CUDA 依赖安装成功。通过 Hugging Face 元数据接口核实默认模型 revision 存在，未下载权重。

目标 Linux CUDA 环境、真实 checkpoint 和全量 SeedTTS 尚未在本机执行；远端脚本的实际报告才是这些项目的验收证据。
