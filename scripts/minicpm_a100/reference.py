"""Sequential HF reference and independent TTS transcription after server teardown."""
from __future__ import annotations

import base64
import copy
import difflib
import io
import json

from run import save


def _audio(path):
    import librosa
    import soundfile as sf
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    return librosa.resample(data.mean(axis=1), orig_sr=sr, target_sr=16000)


async def run_reference(w):
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from transformers import AutoConfig, AutoModel, AutoProcessor
    from sglang_omni.models.minicpm_o.components.audio_encoder import MiniCPMOAudioEncoder
    from sglang_omni.models.minicpm_o.components.image_encoder import MiniCPMOImageEncoder
    from sglang_omni.models.minicpm_o.components.preprocessor import MiniCPMOPreprocessor
    from sglang_omni.proto import OmniRequest, StagePayload

    source = w.out / "requests/reference_collect.jsonl"
    if not source.exists():
        raise RuntimeError("No serving reference cases were collected")
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    model_path = w.res["model_path"]
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    cfg.init_tts = False
    cfg.init_audio = cfg.init_vision = True
    remote = AutoModel.from_pretrained(model_path, config=cfg, trust_remote_code=True,
                                      torch_dtype=torch.bfloat16, attn_implementation="sdpa").eval().cuda()
    # Existing repo reference tests document this HF-v5 2-tuple / remote 3-tuple adapter.
    for layer in remote.apm.layers:
        original = layer.self_attn.forward
        def adapted(*args, _original=original, **kwargs):
            result = _original(*args, **kwargs)
            return (*result, None) if isinstance(result, tuple) and len(result) == 2 else result
        layer.self_attn.forward = adapted
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    preprocessor = MiniCPMOPreprocessor(model_path, speech_enabled=False)
    native_image = MiniCPMOImageEncoder(model_path, dtype="bfloat16")
    native_audio = MiniCPMOAudioEncoder(model_path, dtype="bfloat16")
    result = {"failed": 0, "cases": [], "reference_revision": w.res["models"]["model"]["revision"],
              "compatibility_adapter": "HF audio attention 2-tuple padded to remote 3-tuple, as in repository parity test",
              "generation_gate": "difflib sequence ratio >= 0.8 is a triage gate, not numerical identity",
              "encoder_gate": "finite, identical shape and minimum per-row cosine >= 0.98; max/mean errors retained"}
    for row in rows:
        item = {"serving_id": row["id"], "sample_id": row.get("sample_id")}
        try:
            if not row["ok"]:
                raise RuntimeError("Serving request failed; no valid baseline")
            body = row["request"]
            messages = copy.deepcopy(body["messages"])
            imgs = [Image.open(io.BytesIO(base64.b64decode(x.split(",", 1)[1]))).convert("RGB") for x in body.get("images", [])]
            audios = [_audio(io.BytesIO(base64.b64decode(x.split(",", 1)[1]))) for x in body.get("audios", [])]
            raw = {"messages": messages, "images": imgs, "audios": audios}
            # This exercises production preprocessing, not a hand-built replacement mask.
            payload = StagePayload("reference", OmniRequest(raw, metadata={"output_modalities": ["text"]}), {})
            prepared = await preprocessor(payload)
            state = prepared.data
            prompt = state["prompt"]["prompt_text"]
            data = processor(prompt, images=[imgs] if imgs else None, audios=[audios] if audios else None, return_tensors="pt").to("cuda")
            same_tokens = torch.equal(state["prompt"]["input_ids"].cpu(), data["input_ids"][0].cpu())
            item["processor_input_ids_equal"] = same_tokens
            if not same_tokens:
                raise RuntimeError("Processor input_ids differ")
            with torch.inference_mode():
                for modality, native in (("image", native_image), ("audio", native_audio)):
                    key = modality + "_encoder"
                    if key not in state.get("encoder_inputs", {}):
                        continue
                    inputs = {k: v for k, v in state["encoder_inputs"][key].items() if k != "cache_key"}
                    got = native(**inputs)[modality + "_embeds"].float()
                    reference = remote.get_vision_embedding(data) if modality == "image" else remote.get_audio_embedding(data, chunk_length=remote.config.audio_chunk_length)
                    tensors = []
                    def flatten(value):
                        if isinstance(value, torch.Tensor):
                            tensors.append(value.reshape(-1, value.shape[-1]))
                        elif isinstance(value, (list, tuple)):
                            for x in value:
                                flatten(x)
                    flatten(reference)
                    gold = torch.cat(tensors).float()
                    if got.shape != gold.shape or not torch.isfinite(got).all():
                        raise RuntimeError(f"{modality} encoder shape or finite check failed")
                    cosine = F.cosine_similarity(got, gold, dim=-1).min().item()
                    item[modality] = {"cosine_min": cosine, "max_abs_error": (got - gold).abs().max().item(), "mean_abs_error": (got - gold).abs().mean().item()}
                    if cosine < .98:
                        raise RuntimeError(f"{modality} encoder cosine {cosine} below .98")
                messages[-1]["content"] = [*imgs, *audios, messages[-1]["content"]]
                answer = remote.chat(msgs=messages, processor=processor, tokenizer=processor.tokenizer,
                                     do_sample=False, generate_audio=False, enable_thinking=False,
                                     max_new_tokens=body.get("max_tokens", 256))
            if isinstance(answer, tuple):
                answer = answer[0]
            answer = str(answer)
            similarity = difflib.SequenceMatcher(None, row["text"].strip(), answer.strip(), autojunk=False).ratio()
            item.update(serving_text=row["text"], reference_text=answer, exact_match=row["text"].strip() == answer.strip(), sequence_ratio=similarity)
            item["passed"] = similarity >= .8
        except Exception as exc:
            import traceback
            item.update(passed=False, error=str(exc), traceback=traceback.format_exc())
        result["cases"].append(item)
        result["failed"] += not item["passed"]
        save(w.out / "results/reference.partial.json", result)
    return result


def score_tts(w):
    import jiwer
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
    from benchmarks.tasks.asr import normalize_text
    source = w.out / "requests/tts.jsonl"
    if not source.exists():
        raise RuntimeError("TTS generation phase has no records")
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    path = w.res["asr_model_path"]
    processor = AutoProcessor.from_pretrained(path)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(path, torch_dtype=torch.float16).eval().cuda()
    transcribe = pipeline("automatic-speech-recognition", model=model, tokenizer=processor.tokenizer,
                          feature_extractor=processor.feature_extractor, device=0, torch_dtype=torch.float16)
    result = {"failed": 0, "cases": [], "scorer": w.res["models"]["asr_model"]["repo"],
              "note": "Independent ASR text/audio consistency; not a human MOS or proof of speaker identity"}
    by_lang = {}
    for row in rows:
        item = {"id": row["id"]}
        try:
            if not row["ok"] or not row.get("audio"):
                raise RuntimeError("Missing valid generated audio")
            lang = row["lang"]
            audio = _audio(row["audio"])
            hyp = transcribe({"raw": audio, "sampling_rate": 16000}, generate_kwargs={"language": "chinese" if lang == "zh" else "english", "task": "transcribe"}, chunk_length_s=30)["text"]
            ref = normalize_text(row["text"], lang)
            normalized_hyp = normalize_text(hyp, lang)
            error = jiwer.wer(ref, normalized_hyp)
            item.update(reference=row["text"], hypothesis=hyp, error_rate=error,
                        passed=error <= w.cfg["quality_thresholds"]["tts_roundtrip_error_max"])
            by_lang.setdefault(lang, []).append((ref, normalized_hyp))
        except Exception as exc:
            item.update(passed=False, error=str(exc))
        result["cases"].append(item)
        result["failed"] += not item["passed"]
        save(w.out / "results/tts_score.partial.json", result)
    result["corpus_error"] = {lang: jiwer.wer([p[0] for p in pairs], [p[1] for p in pairs]) for lang, pairs in by_lang.items()}
    return result
