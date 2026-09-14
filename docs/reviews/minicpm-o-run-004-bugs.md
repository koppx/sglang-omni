# MiniCPM-o run-004 实验问题分类报告

## 2026-09-14 更新：PR #1879 最新提交复核

复核提交：[`0eec570b3b6440088e5f4fca3cbc6c25ad8d164e`](https://github.com/sgl-project/sglang-omni/commit/0eec570b3b6440088e5f4fca3cbc6c25ad8d164e)。本轮检查 MiniCPM 相对此前版本的增量、相关公共调用路径及新增测试，并对最新源码重新运行 CPU 隔离探针；**没有重新运行 A100 实验**。后文 run-004 的结果继续作为历史证据，不能自动视为最新提交的实测结果。

### 最新状态

| 编号 | 最新提交状态 | 证据与边界 |
|---|---|---|
| BUG-01 空 Tensor 共享内存异常 | **代码缺陷仍在** | SHM 仍直接以实际字节数创建共享内存；打包层只对“没有 Tensor”填充 1 字节，没有覆盖“存在 Tensor 但全部为空”。本轮未重新触发服务崩溃 |
| BUG-02 中间 prefill hidden 累计 | **最新源码探针重新复现** | 中间 chunk 未生成 token，仍累计 1 个 hidden |
| BUG-03 标准多模态 content blocks | **最新源码探针重新复现** | 图片 content 数组仍进入文本模板，未分发媒体编码。PR 描述明确 inline image_url 尚不支持，应同时标记为已知功能缺口；请求处理异常仍需修复 |
| BUG-04 音频 padding mask | **最新源码探针重新复现** | 60 帧下采样为 30 帧后，额外 30 个 padding key 仍可见 |
| BUG-05 历史 EOS 误选 | **最新源码探针重新复现** | 当前片段应返回 `[10, 11]`，实际返回 `[]` |
| BUG-06 非法输入返回 500 | **保留历史发现，最新版本待端到端复测** | 尚未对 rebase 后完整 API/错误处理链执行边界测试 |
| ISSUE-01 并发 OOM | **最新版本待 GPU 复测** | 运行时及公共基础设施已随 rebase 更新，不能直接套用旧版本容量结论 |
| ISSUE-02 TTS 一致性不佳 | **最新版本待质量复测** | 保留旧分数作为历史记录，不代表最新版本成绩 |
| ISSUE-03 图像数值对齐失败 | **最新版本待数值复测** | 尚未在新运行时及 A100 上重跑原失败单测 |
| TOOL-01 HF 对照加载失败 | **最新版本待加载复测** | 不将旧环境中的加载错误直接认定为新版本实测结果 |
| 安装依赖冲突 | **声明仍冲突** | 最新 pyproject.toml 仍要求 minicpmo-utils[tts]==1.0.6、Transformers 5.12.1 和 librosa >=0.11；此前确认的包元数据冲突没有在仓库声明中解决 |

### 本次更新已处理的内容

- 修复 rebase 后数据集准备代码的空 `elif` 分支。
- 将 runner 的特殊 `super(...)` 初始化改为显式 `ModelRunner.__init__()`。
- 适配当前 SGLang 的基础设施返回值，移除过时 scheduler 参数。
- 更新 CUDA Graph 初始化、hidden capture 配置与失败后的配置恢复，并增加对应测试。

这些属于启动和运行时适配，**不是 BUG-02 至 BUG-05 的修复**。检查到的增量中暂未确认新的明显逻辑 bug，但这不等于完成新版全量验收。复核时 lint 已通过，部分运行测试 skipped、CI gate failure；没有将 CI 状态解释为已通过 MiniCPM GPU 验收。

**复核结论：REQUEST_CHANGES。** 优先修复仍已确认的问题，再对 OOM、质量、边界行为和 HF 对照执行最新版本复测。本文未将后文的历史问题条目删除或改写成最新提交实测结论。

## 实验范围与结论

- 实验 commit：`ff78467f3937766d3c8c66c8b705824fa9bffc15`。
- 硬件：单卡 NVIDIA A100-SXM4-80GB。
- 实验结果：**FAIL**。默认启动和部分推理成功，但发生两次服务崩溃，质量与稳定性验收未完成。
- 阶段耗时合计约 69 分钟，不是完成的 8 小时稳定性测试。
- 本报告分类：**6 项确定的产品代码 bug、3 项已复现但根因待查的问题、5 项实验工具与环境问题**。部分问题可能共享根因，不能机械相加为独立根因数量。
- P1/P2 为建议修复优先级，不表示问题一定由 PR #1879 新引入。
- 代码位置按当前本地源码核对；实验结论以 run-004 日志为准。后续代码变更可能使行号移动。

原始材料：[实验报告](/Users/bytedance/Downloads/run-004/report.md)、[运行目录](/Users/bytedance/Downloads/run-004/)。本文链接指向本机文件，分享时需要同时提供源码和实验产物。

## 一、确定是产品代码 bug

### BUG-01 · P1：空 Tensor 传输导致服务退出

**代码位置**：[shm.py:25](/Users/bytedance/sglang-omni/sglang_omni/relay/shm.py:25)，`shm_create_from_tensor()`。

代码直接使用 Tensor 的字节数创建共享内存，没有处理零元素 Tensor：

```python
size = t_np.nbytes
shm = _shm.SharedMemory(create=True, size=size)
```

**实验现象**：边界测试期间，Talker 传输输出时抛出 `ValueError: 'size' must be a positive number different from zero`。Talker 随后退出，服务不可用，后续 300 个 ASR 请求连接失败。

**解决方案**：传输层显式支持空 Tensor。最小方案可至少分配 1 字节，但保持真实 shape、dtype 和零元素语义，并验证接收与资源释放流程。同时隔离请求级传输异常，避免单个异常请求拖垮 Stage。

**验收**：空 Tensor 跨进程传输正确；触发空输出后，下一个正常请求仍成功；共享内存正常释放。

证据：[首次崩溃堆栈](/Users/bytedance/Downloads/run-004/default_startup.log:1394)。本轮没有证明空 Tensor 一定来自 BUG-05。

### BUG-02 · P1：中间 prefill chunk 被累计进 hidden 序列

**代码位置**：[thinker_model_runner.py:93](/Users/bytedance/sglang-omni/sglang_omni/models/minicpm_o/thinker_model_runner.py:93)，`post_process_outputs()`。

收到 `hidden_states` 后无条件追加到 `_pending_hidden`，没有排除中间 prefill chunk。下游却假定首项是完整 prompt 的最后位置，其余项依次对应 decode 输入位置。

**实验现象**：CPU 隔离探针确认，中间 chunk 未生成 token，却新增了一个 hidden。强制 chunk size 256 的长 prompt 语音请求 12/12 完成，仅证明能返回结果，不能证明对齐正确。

**解决方案**：依据实际调度状态排除中间 prefill，只累计最终 prefill 与有效 decode hidden，维护与 token 的位置关系；不要仅根据 hidden 是否存在判断。

**验收**：同一输入在不分块与小 chunk 配置下，Talker 的 token/hidden 对齐一致；截断和取消请求后正确清理累计状态。

证据：[逻辑探针](/Users/bytedance/Downloads/run-004/review_probes.log:11)。本轮尚未证明 TTS 低分由此导致。

### BUG-03 · P1：标准多模态 content blocks 没有被解析

**代码位置**：[preprocessor.py:113](/Users/bytedance/sglang-omni/sglang_omni/models/minicpm_o/components/preprocessor.py:113)，`__call__()`。

只提取顶层 `images`、`audio`、`audios`，没有解析 `messages[].content` 中的 `image_url/input_audio`。标准 content 数组可能进入纯文本模板路径。

**实验现象**：图片 content-block 探针确认未执行媒体解码、占位符注入或编码器分发；多轮 content-block 功能检查失败，日志出现 `can only concatenate str (not "list") to str`。音频块路径缺失由代码检查支持，本轮探针直接复现的是图片块。

**解决方案**：在分流前统一解析 content blocks，提取文字与媒体并保留原始顺序、所属消息轮次；兼容现有顶层媒体入口，避免把历史媒体全部移到最后一轮。无效媒体或不支持类型返回明确 4xx。

**验收**：标准图片、音频及多轮混合消息进入正确编码器；媒体与轮次对应正确。

证据：[探针日志](/Users/bytedance/Downloads/run-004/review_probes.log)、[运行错误](/Users/bytedance/Downloads/run-004/default_startup.log:359)。

### BUG-04 · P2：音频 padding mask 使用下采样前长度

**代码位置**：[audio_encoder.py:293](/Users/bytedance/sglang-omni/sglang_omni/models/minicpm_o/components/audio_encoder.py:293)，音频编码器 mask 构造逻辑。

卷积下采样后，mask 仍使用原始 mel 有效长度，导致部分 padding 位置被当作有效位置。

**实验现象**：60 个有效 mel 帧下采样后只有 30 个有效位置，探针发现另外 30 个 padding key 可见。

**解决方案**：使用卷积实际输出长度构造 attention mask，正确处理 stride、padding 和奇偶长度。

**验收**：不同长度混合 batch 的 padding 全部被遮蔽；比较同一样本单独运行与混合 batch 运行的有效输出。

证据：[探针日志](/Users/bytedance/Downloads/run-004/review_probes.log)。此前检查的 HF 参考实现也存在此问题，不能仅依赖与 HF 相等的测试发现它。

### BUG-05 · P2：当前语音截断时误用历史 EOS

**代码位置**：[request_builders.py:542](/Users/bytedance/sglang-omni/sglang_omni/models/minicpm_o/request_builders.py:542)，TTS BOS/EOS 片段提取逻辑。

以最后一个 BOS 确定起点，却从整个历史序列选择最后一个 EOS；没有要求 EOS 位于当前 BOS 之后。当前片段没有生成 EOS 时，结束位置可能落在起点之前。

**实验现象**：带历史语音标记且当前片段截断的探针中，应保留 `[10, 11]`，实际返回 `[]`。

**解决方案**：仅在当前起点之后查找 EOS；没有 EOS 时按明确的截断语义保留当前有效片段，保证 token 和 hidden 切片一致。

**验收**：覆盖历史 EOS、正常结束、当前截断和空片段，不误选历史结束位置。

证据：[探针日志](/Users/bytedance/Downloads/run-004/review_probes.log)。此前检查的 HF 参考逻辑也存在此问题；本轮没有证明它就是首次服务崩溃的上游原因。

### BUG-06 · P2：非法输入返回 500

**代码范围**：API 参数校验、MiniCPM 预处理、错误到 HTTP 状态码的映射。涉及多个入口，尚不能统一定位为单行错误。

**实验现象**：30 个边界用例中 16 个通过、14 个失败，其中 12 个预期返回 4xx 的用例返回 500。相关日志包含负数 token 参数、空消息和损坏媒体错误。

**解决方案**：对参数、消息结构及媒体内容进行入口校验；将明确的用户输入错误映射为 4xx，保留内部异常为 500。不要用通用捕获将所有异常都改成 4xx。

**验收**：非法输入返回可解释的 4xx，服务保持健康；真正内部故障仍保留可诊断日志。

证据：[边界结果](/Users/bytedance/Downloads/run-004/boundary.log)、[服务日志](/Users/bytedance/Downloads/run-004/default_startup.log)。

## 二、已复现，但根因待查

### ISSUE-01 · P1：长语音并发 OOM，随后服务退出

**现象**：长语音输出、并发 4、第二轮压测触发 CUDA OOM，额外 2 MiB 都无法分配，Talker 随后退出。整个语音输出压测 768 个请求中 612 个成功、156 个失败。

**报错位置**：实验堆栈中的 `model_runner/base.py::_sample_next_token_ids()`，最终在 SGLang `sampler.py` 的 `probs.sort()` 分配失败。这里是触发点，不等于显存占用的源头。

**待查**：Thinker/Talker/Code2Wav 整体显存预算、动态峰值、hidden 与缓存释放、并发准入。不能仅凭本轮断言为内存泄漏或显存碎片。

**解决方向**：先记录各进程和请求生命周期的显存，预留动态余量并限制过量并发，再针对实际持有内存的对象修复。OOM 后需受控恢复，不能将服务默认为仍可用。

**验收**：长语音并发重复运行不崩溃；容量不足时排队或明确拒绝；请求结束后显存不会无界增长。

证据：[OOM 堆栈](/Users/bytedance/Downloads/run-004/recovery.log:14640)。并发 8 的连接失败发生在服务已崩溃之后，不能独立证明并发 8 必然 OOM。

### ISSUE-02 · P1：TTS 文本与语音一致性不佳

**现象**：50 条语音全部生成成功，回转录评分只有 10 条通过。Whisper-small 评分的英文语料错误率约 44.3%，中文约 73.9%。

**待查**：token/hidden 对齐、文本或语音截断、模型实际生成文本是否等于目标朗读文本、参考音频处理，以及 Whisper 识别和中文繁简/归一化误差。

**解决方向**：对齐目标文本、实际生成文本、语音 token 和波形；建立可用 HF 对照；校验评分器与归一化。必要时听检有代表性的失败样本。

**验收**：相同输入和参考音频下，与可信基线比较文本—音频一致性。该指标不是人工 MOS，也不是说话人身份验证。

证据：[评分日志](/Users/bytedance/Downloads/run-004/tts_score.log)。不能直接将低分归因于 BUG-02、librosa 或依赖绕过。

### ISSUE-03 · P2：图像编码器数值对齐不达标

**现象**：94 个单测中 93 个通过、1 个失败。失败项 `test_golden_parity_vs_remote_code` 的最低余弦相似度为 0.966690，低于测试中的远端 BF16 基线 0.978266。

**待查**：实现差异、权重与输入一致性、精度设置、attention 后端和数值波动。

**解决方向**：固定权重、输入、精度与后端，逐层比较误差起点；不要先放宽阈值掩盖差异。

**验收**：解释数值差异并达到有依据的容差，随后验证图像任务效果。

证据：[单测失败](/Users/bytedance/Downloads/run-004/unit.log:73)。

## 三、实验工具与环境问题

这些问题影响结果可信度与自动化完整性，不应直接计为模型推理代码 bug。

| 编号 | 问题 | 实验依据 | 调整方案 |
|---|---|---|---|
| TOOL-01 | HF 对照加载失败 | `MiniCPMO` 缺少 `all_tied_weights_keys`，对照未执行 | 修复加载适配或采用独立兼容环境，并记录基线版本；不能把失败当作模型效果退化 |
| TOOL-02 | 环境 PASS 未体现依赖不一致 | 使用 `--no-deps` 安装 minicpmo-utils，日志仍有 decord/moviepy 缺失及 librosa/Pillow 冲突 | 分开记录可运行性和依赖一致性；明确允许覆盖的约束，不忽略其他冲突 |
| TOOL-03 | 服务崩溃后仍发起 ASR 批量请求 | 300 次连接失败，没有有效识别结果 | 发请求前检查健康；恢复后重跑或标记 BLOCKED，限制无效重试 |
| TOOL-04 | 失败数口径混合 | performance 顶层 `failed=212`，实际失败请求 156；functional 也有附加检查失败 | 分离请求失败、质量门槛失败、附加检查失败，并记录各自分母 |
| TOOL-05 | PASS 容易被理解为全面通过 | TTS PASS 是波形生成检查；chunked prefill PASS 是请求完成 | 报告分别标注完成性、质量和对齐验证，明确不包含的验证范围 |

证据：[HF 对照日志](/Users/bytedance/Downloads/run-004/reference.log:33)、[环境日志](/Users/bytedance/Downloads/run-004/environment.log)、[ASR 日志](/Users/bytedance/Downloads/run-004/asr.log)、[性能日志](/Users/bytedance/Downloads/run-004/performance.log)。

## 四、不能重复计为独立 bug 的结果

- **ASR 300 次连接失败**：服务崩溃的后果，不是识别准确率为零。
- **并发 8 连接失败**：此前 OOM 后服务已退出，不是独立有效的并发 8 容量测试。
- **soak、profiling 未完成**：属于证据缺失，不能据此判定有泄漏或某个算子是瓶颈。
- **MMMU 44%、仅 58 条解析出答案**：需要区分答错、格式不符和输出截断；不能直接列为某个实现 bug，也不是完整 benchmark 成绩。
- **依赖绕过**：本轮确实跑通了 Token2wav，但未证明整包兼容；也没有证据表明全部故障都由依赖覆盖造成。

## 五、建议修复与复测顺序

1. 修复 BUG-01 的空 Tensor 传输与异常隔离，定位 ISSUE-01 的 OOM；确认错误请求不会使服务退出。
2. 修复 BUG-02 至 BUG-06，补充对应回归测试；保留已有探针作为修复前后对照。
3. 修复实验健康检查、失败计数和环境状态展示；恢复 HF 对照。
4. 小规模复测图像数值对齐和 TTS 一致性，确认评分口径可靠。
5. 重跑有效的 ASR、MMMU、并发压测，最后执行长时间稳定性与 profiling。

当前最关键的四项为：**BUG-01 空输出服务崩溃、ISSUE-01 长语音并发 OOM、BUG-02 hidden 错位、BUG-03 标准多模态消息未解析**。其中 OOM 已确认是稳定性问题，但根因仍待定位。
