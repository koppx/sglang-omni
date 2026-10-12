# MiniCPM-o handoff 一键远端验证

## 目的与交付

本节目的：用一个入口完成由小到大的远端验证，并明确报告能证明什么。

入口为 `scripts/test_minicpm_handoff.py`。脚本复用已经提交的 7 个回归测试文件，不复制其实现；收到远端报告后可以删除这个临时入口和本说明，常规回归测试继续保留。脚本不会自动删除测试代码或结果。

一次启动后无需逐步确认：小规模检查通过才进入大规模测试，失败或超时立即停止后续阶段，保留原始证据并更新 `report.md`、`report.json`。退出码 0 表示本次生成验证完成，不表示音频质量、完整模型张量一致性或性能收益已经验收。

## 执行顺序

本节目的：让昂贵的全量实验只在基本正确性和真实服务冒烟通过后开始。

| 阶段 | 默认工作量 | 进入下一阶段的条件 |
|---|---|---|
| 环境和输入 | Linux、单张 CUDA GPU、依赖、源码导入路径、数据与参考音频 | Torch 2.13.0、SGLang 0.5.21；测量集与预热集无 ID、文本或参考音频路径重叠 |
| CPU 小规模回归 | 21 个 handoff／projection 用例 | 全部执行通过 |
| GPU 核心回归 | 每轮 5 个展开用例，重复 3 轮；IPC 用例内部有 8 轮回收 | graph replay、三 stream conditioning、relay、跨进程 IPC 全部实际执行；跳过视为失败 |
| 真实服务小样本 | A/B/C 各 8 条，并发 1；每组独立预热 | 请求全部成功、音频时长有效、WAV 存在；C 组日志出现 direct CUDA IPC 接收完成事件 |
| 完整单测 | 7 个已有测试文件 | pytest 成功，JUnit 无失败或错误；通用可选后端的 skip 留在 XML 中，核心 CUDA 用例已由上一阶段严格检查 |
| 全量性能 | 全部 metadata 行；并发 1/2/4/8/16/32，每点每组 3 次 | 每条请求成功并产生有效结果，否则停止并报告 |
| 汇总 | 每阶段更新，结束生成完整表格 | 保存原始数据及相对变化，不自动声称统计显著或音频质量通过 |

GPU 测试保持默认异步执行。脚本移除子进程的 `CUDA_LAUNCH_BLOCKING`、`PYTHONOPTIMIZE` 和 `PYTEST_ADDOPTS`，防止同步模式、关闭断言或外部筛选参数改变验证范围。

默认一共启动 57 次服务：3 次冒烟和 54 次全量实验。原始英文 1,088 条数据对应 58,776 条测量请求，另有每次 16 条预热；这是完整验收规模，不是几分钟的快测。启动超时默认 1,200 秒，每个测试／benchmark 子进程超时默认 7,200 秒，可用参数调整。

## 基准与其他 PR 的关系

本节目的：固定三组对照及测量定义，让收益能归因到具体改动。

| 组别 | 源码和开关 | 比较 |
|---|---|---|
| A：baseline | `7f78bf3dc5c3053d6eb270764eb19232b617dec4` | 优化前基线 |
| B：p0 | 启动脚本时的已提交 HEAD，`enable_gpu_handoff=false` | B/A：连续 Tensor 与投影优化 |
| C：p1 | 同一 HEAD，`enable_gpu_handoff=true` | C/B：GPU handoff；C/A：总体 |

脚本创建两个临时 detached worktree，不切换当前分支；各服务从相应源码目录启动。三组 benchmark 客户端都使用候选版本的同一个 SeedTTS 实现，避免客户端版本造成口径差异。共享当前已安装的 Python 环境，脚本不安装依赖。运行前要求已跟踪文件无未提交改动。

每次实验都重启服务、使用独立预热集。三轮依次采用 A/B/C、B/C/A、C/A/B 顺序；相同 checkpoint、数据顺序、seed 42、temperature 0.7、max_new_tokens 256、Thinker/Talker 显存比例 0.55/0.15、`OMP_NUM_THREADS=4`。其余模型设置沿用这两个版本的配置，本次 handoff 改动没有调整采样策略。

这套设置沿用旧 #2330 说明中引用的 #2273 workload，但必须提供相同的原始英文 SeedTTS 1,088 条数据和 checkpoint 才能说 workload 对齐。旧 #2273 使用的 checkpoint revision 为 `503e754207c94da6bb26850b4469f367c9ea3582`，原始 dataset revision 为 `8f5e1aa2a35d42f42e940074c1983358b9491f89`。仓库默认 Arrow 数据集不一定是同一个快照。旧实验的依赖、预热和服务重启策略不同，其历史数字不作为本次 A 组。

脚本记录候选 SHA、固定基线、模型路径及 config SHA256、数据与实际参考音频内容的联合 SHA256、依赖、GPU 和每次命令。checkpoint 权重应使用不再修改的本地 snapshot；config 哈希不是完整权重哈希。数据行数按输入记录，脚本不会把任意 metadata 冒称为原始 1,088 条基准。

## 一次启动

本节目的：在已有远端项目环境中，仅填入本机路径即可后台运行。

先拉取当前 fork 分支并激活项目 Python 环境。选择一张空闲 NVIDIA GPU，准备本地 checkpoint、测量 `meta.lst` 和独立的预热 `meta.lst`。相对音频路径的解析遵循项目 SeedTTS loader；建议 metadata 使用绝对音频路径。预热集至少 16 条，测量集大于 8 条，完整验收应使用全量固定数据。

在仓库根目录执行，修改下面四个路径。结果目录必须不存在：

```bash
nohup python scripts/test_minicpm_handoff.py \
  --checkpoint /data/models/MiniCPM-o-4_5 \
  --meta /data/seedtts/en/meta.lst \
  --warmup-meta /data/seedtts/warmup/meta.lst \
  --output /data/results/minicpm-handoff-run1 \
  --gpu 0 \
  > /tmp/minicpm-handoff-run1.log 2>&1 &
echo $! > /tmp/minicpm-handoff-run1.pid
```

此命令自动完成所有阶段。只想先检查执行矩阵，在同样命令后加 `--dry-run` 并前台执行；它不启动服务，也不检查 GPU 或加载模型，不属于验证通过。

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

目标 Linux CUDA 环境、真实 checkpoint 和全量 SeedTTS 尚未在本机执行；远端脚本的实际报告才是这些项目的验收证据。
