#!/usr/bin/env python3
"""Execute isolated production functions on CPU; expected invariants expose PR defects.

AST extraction avoids importing CUDA-only SGLang on the review host. These probes
are not substitutes for end-to-end GPU tests. Exit 1 means a defect was reproduced.
"""
from __future__ import annotations
import argparse
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import torch

ROOT = Path(__file__).resolve().parents[2]


def function(file, name, cls=None, **globals_):
    tree = ast.parse((ROOT / file).read_text())
    nodes = tree.body
    if cls:
        nodes = next(n for n in nodes if isinstance(n, ast.ClassDef) and n.name == cls).body
    fn = next(n for n in nodes if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name)
    fn.decorator_list = []
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), fn], type_ignores=[])
    ns = {"torch": torch, **globals_}
    exec(compile(ast.fix_missing_locations(module), str(ROOT / file), "exec"), ns)
    return ns[name]


def probes():
    prefix = "sglang_omni/models/minicpm_o/"
    findings = []
    forward = function(prefix + "components/audio_encoder.py", "forward", "MiniCPMOAudioEncoder", _MASK_MIN=-1e9)
    captured = {}
    def apm(wave, mask):
        captured["mask"] = mask
        return torch.zeros(2, 100, 4)
    fake = SimpleNamespace(_device=torch.device("cpu"), _dtype=torch.float32, apm=apm,
        _cached_chunk_mask=lambda size: torch.ones(size, size, dtype=torch.bool),
        audio_projection_layer=lambda x: x, audio_avg_pooler=torch.nn.AvgPool1d(2),
        _feature_lens_after_pooling=lambda lens: ((lens - 1) // 2 + 1) // 2)
    forward(fake, audio_features=torch.zeros(2, 80, 200), audio_feature_lens=torch.tensor([200, 60]))
    leaked = int((captured["mask"][1, 0, 0, 30:60] == 0).sum())
    findings.append({"id": "audio_padding_mask", "passed": leaked == 0,
                     "evidence": f"For 60 valid mel frames, 30 valid conv frames: {leaked} padding keys are visible",
                     "scope": "Also present in downloaded HF reference; parity alone cannot detect it"})

    post = function(prefix + "thinker_model_runner.py", "post_process_outputs", "MiniCPMOThinkerModelRunner")
    fake = SimpleNamespace(_pending_hidden={})
    req = SimpleNamespace(request_id="r", data=SimpleNamespace(req=SimpleNamespace(inflight_middle_chunks=1)))
    post(fake, None, SimpleNamespace(requests=[req]), {"r": SimpleNamespace(extra={"hidden_states": torch.ones(1, 4)})})
    findings.append({"id": "middle_prefill_hidden", "passed": not fake._pending_hidden,
                     "evidence": f"Middle chunk emitted no generated token but accumulated {len(fake._pending_hidden.get('r', []))} hidden entry"})

    build = function(prefix + "request_builders.py", "build_talker_request")
    state = SimpleNamespace(prompt={"input_ids": torch.tensor([900, 7, 901, 900])}, thinker_out={
        "output_ids": [10, 11], "extra_model_outputs": {"hidden_states_seq": [torch.ones(4)] * 3}})
    result = build(state, tts_bos_token_id=900, tts_eos_token_id=901)
    findings.append({"id": "old_turn_tts_eos", "passed": result["tts_token_ids"].tolist() == [10, 11],
                     "evidence": f"Current span is truncated (no new EOS); old-turn EOS yields {result['tts_token_ids'].tolist()} instead of [10, 11]"})

    render = function(prefix + "components/preprocessor.py", "_render_chat_template", "MiniCPMOPreprocessor")
    received = {}
    def template(messages, **kwargs):
        received["messages"] = messages
        return "stub-template"
    messages = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,fixture"}}, {"type": "text", "text": "Describe"}]}]
    render(SimpleNamespace(tokenizer=SimpleNamespace(apply_chat_template=template)), messages)
    findings.append({"id": "unprocessed_content_blocks", "passed": isinstance(received["messages"][0]["content"], str),
                     "evidence": "Standard image_url content array passed verbatim to tokenizer; no image decoding, placeholder injection or encoder dispatch"})
    return findings


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    findings = probes()
    result = {"failed": sum(not f["passed"] for f in findings), "probes": findings,
              "method": "isolated production function execution with CPU tensors and fake stage dependencies"}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    sys.exit(1 if result["failed"] else 0)
