# PR #1879 review 与自动验收交付

Review 对象：`f3fedb250452b07d684a885567c84b172c83192a`。2026-09-13 经 GitHub
只读核对，PR #1879 head 与当前分支一致。以首次 MiniCPM commit 前的 `8f8b73d3`
为实现范围起点，34 个变更文件、约 4,700 行新增；不是拿最新 main 的所有变动作比较。
当前分支没有额外增量。结论：**REQUEST_CHANGES**。

已检查 MiniCPM runtime 文件、共享 model worker/runner 注册改动、ASR dataset loader
改动和对应单元测试。没有在 A100 上执行模型服务；以下证据为代码追踪及隔离生产函数
CPU 执行，不应理解成真实模型的准确率或性能测量。被测模型代码未修改。

## Findings

### [IMPORTANT / P1] 中间 prefill chunk 被累计进逐 token hidden 序列

位置：`sglang_omni/models/minicpm_o/thinker_model_runner.py:93–103`。

`post_process_outputs` 对所有含 hidden 的调度输出无条件 append，而共享输出处理器
不会过滤中间 prefill chunk；这些 chunk 不产生有效输出 token。下游
`build_talker_request` 则把每个序列位置当作一次有效生成步，假设第 0 项就是最后的
prompt 位置。分块 prefill 后，实际序列开头多出中间 prompt hidden，整段 TTS token
与 hidden 的对应关系向后错位。`make_thinker_stream_output_builder` 已过滤中间
chunk，但这个过滤不作用于 hidden 累计。

触发：启用 chunked prefill，长 prompt 被拆分；包括显式缩小 chunk size，或并发请求
耗尽当前 prefill token budget。语音输出最受影响，短 prompt 单请求不一定触发。

CPU 复现：`inflight_middle_chunks=1` 的请求仍累计了 1 项 hidden。建议根据该请求的
有效生成步过滤中间 chunk，并补“同 prompt、分块/不分块”的 token/hidden 对齐测试；
同时覆盖 retract/replay，避免重放 hidden 重复累计。

### [IMPORTANT / P1] 标准多模态 content blocks 没有被预处理

位置：`sglang_omni/models/minicpm_o/components/preprocessor.py:113–123`。

服务入口把 `messages[].content` 原样交给模型预处理器，只有独立顶层 `images` /
`audios` 才转成独立媒体输入。MiniCPM 预处理器不遍历消息中的 `image_url` /
`input_audio`，因此标准 OpenAI 多模态消息没有生成 encoder 输入，content 数组被直接
交给 tokenizer template。可能报模板错误，或把描述字符串交给纯文本模型，均没有
真正理解媒体。多轮历史里的媒体同样丢失位置关系。HF 的 `chat()` 则显式执行
`normalize_content` 并在所属消息处插入占位符。

CPU 复现：含 image_url 的 content 数组未经规范化，原样传入 tokenizer。
建议逐消息解析媒体并保留顺序及轮次，统一顶层媒体兼容路径；不能仅把所有媒体插在
最后一个 user message。自动套件既测现有顶层媒体 API，也包含标准 content-block
多轮回归，防止只测内部特有格式。

### [IMPORTANT / P2] 音频 padding mask 使用了错误尺度的长度（HF 继承问题）

位置：`sglang_omni/models/minicpm_o/components/audio_encoder.py:293–298`。

卷积输出长度是 `(mel_length - 1) // 2 + 1`，但 key-validity 比较仍使用原始 mel 长度。
例如短音频有效 mel=60、batch padded mel=200，卷积后只有 30 个有效帧，而 key mask
允许前 60 帧，额外 30 个 padding key 可被有效 query 关注。分块因果 mask 不一定屏蔽
同 chunk 的这些 padding，可能影响变长 batch/多段音频最后一段的有效 embedding。

CPU 复现执行真实 `MiniCPMOAudioEncoder.forward`，在 fake encoder 入口截获 mask，
验证额外可见 key 数量为 30。现有 `test_golden_parity_vs_remote_code` 是手动构造正确
mask 后比较底层 encoder，未调用出错的 wrapper mask 构造，所以无法覆盖本问题。

**来源区别**：下载核对的 HF `modeling_minicpmo.py` 也使用原始长度，此问题不属于
native 重写相对 HF 的数值偏差。建议先加“同一短音频单独/与长音频拼 batch”的
不变性测试，记录修正前后音质/ASR 变化后再决定兼容策略，不仅凭 HF parity 判断正确。

### [IMPORTANT / P2] 当前 TTS span 没有 EOS 时会误用历史 EOS（HF 继承问题）

位置：`sglang_omni/models/minicpm_o/request_builders.py:542–558`。

代码分别取整个 prompt+output 的最后 BOS 和最后 EOS，未验证 EOS 在当前 BOS 之后。
当历史 prompt 中保留 TTS 标记，当前输出被 max tokens/stop 截断而无新 EOS 时，选中
旧轮次 EOS，`end <= start`，返回空 codec 条件；有可朗读文本也会产生空音频。

CPU 复现：prompt `[BOS, 7, EOS, BOS]`，当前 output `[10, 11]` 且 hidden 足够，
返回 token span `[]`，预期 `[10, 11]`。仅适用于历史含 TTS 标记的输入（例如支持的
pre-tokenized/Python 路径）；普通无标记聊天历史不一定触发。

HF 参考 chat 的 BOS/EOS 搜索也有同样逻辑，列为继承问题。建议只在当前 BOS 后找 EOS，
找不到则使用实际可用 hidden 覆盖的尾部，并测试历史 EOS + 当前截断。

## 没有重复报告为 bug 的历史问题

- abort callback 已接到 `reset_request`，以前的 abort hidden 泄漏评论不再适用于此 head。
- vocoder reference cache 已按内容 key 更新，A→B→默认切换及单请求错误隔离已有测试。
- 当前 code2wav 已回到逐请求 `SimpleScheduler`，不能根据历史 commit 标题误称其支持
  真正批量 vocoding。顺序计算的吞吐上限属于需要测量的性能约束。
- thinker/encoder 初始化顺序和复用已有 TP=1 context 已有后续修正。

## 尚需 GPU 验证的风险

- 默认 thinker KV budget、talker pool、encoder、vocoder 同卡时的峰值显存，A100 80GB
  能否完整启动；没有实测，不把潜在 OOM 报成确定 bug。
- A100 attention/CUDA graph 实际后端支持；不能外推 PR 中的 H100 测试结论。
- native talker 与 HF 的 top-k/top-p 顺序差异、语音长输出和高并发稳定性。
- 标准音频/图片 API 接入、多模态长 prompt、取消后状态回收的真实端到端行为。

## 复现与验证

```bash
python3 scripts/minicpm_a100/review_probes.py --output /tmp/minicpm-review.json
python3 -m unittest discover -s scripts/minicpm_a100 -p test_runner.py -v
```

第一次命令在上述 head 隔离执行生产函数，复现 4 项不变量失败，退出码 **1 是预期结果**。
它用 AST 提取函数，fake stage dependency + 真实 CPU tensor；不声称跑过完整 SGLang。
第二次命令验证自动测试工具本身的超时、日志、报告、HTTP/SSE、失败判定及指标算法。
详细交付和开发机启动方式见 `scripts/minicpm_a100/README.md`。

## Sources

- PR: https://github.com/sgl-project/sglang-omni/pull/1879
- HF remote-code comparison: `openbmb/MiniCPM-o-4_5` revision
  `503e754207c94da6bb26850b4469f367c9ea3582`, `modeling_minicpmo.py`。
- Profiling tracking source: https://github.com/sgl-project/sglang-omni/issues/1798
- Methodology snapshot: `3060470a85c39dfd5afff9a16eabdbc85c179ea9`，
  `.claude/skills/model-profiling/METHODOLOGY.md`。
