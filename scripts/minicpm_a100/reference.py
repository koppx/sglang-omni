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
    source = w.out / "requests/reference_collect.jsonl"
    if not source.exists():
        raise RuntimeError("No serving reference cases were collected")
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    result = {"failed": 0, "cases": [], "reference_revision": w.res["models"]["model"]["revision"],
              "generation_gate": "Sequence ratio >= 0.8 is triage, not logits equivalence",
              "encoder_gate": "Same input IDs, finite outputs, same shape, cosine >= 0.98"}
    if w.action == "reference_prepare":
        from PIL import Image
        from sglang_omni.models.minicpm_o.components.audio_encoder import MiniCPMOAudioEncoder
        from sglang_omni.models.minicpm_o.components.image_encoder import MiniCPMOImageEncoder
        from sglang_omni.models.minicpm_o.components.preprocessor import MiniCPMOPreprocessor
        from sglang_omni.proto import OmniRequest, StagePayload
        model_path = w.res["model_path"]
        preprocessor = MiniCPMOPreprocessor(model_path, speech_enabled=False)
        encoders = {"image": MiniCPMOImageEncoder(model_path, dtype="bfloat16"),
                    "audio": MiniCPMOAudioEncoder(model_path, dtype="bfloat16")}
        directory = w.out / "reference-native"
        directory.mkdir(exist_ok=True)
        inputs = []
        for row in rows:
            if not row["ok"]:
                result["failed"] += 1
                continue
            body = copy.deepcopy(row["request"])
            imgs = [Image.open(io.BytesIO(base64.b64decode(x.split(",", 1)[1]))).convert("RGB") for x in body.get("images", [])]
            audios = [_audio(io.BytesIO(base64.b64decode(x.split(",", 1)[1]))) for x in body.get("audios", [])]
            payload = StagePayload(row["id"], OmniRequest({"messages": body["messages"], "images": imgs, "audios": audios}, metadata={"output_modalities": ["text"]}), {})
            state = (await preprocessor(payload)).data
            captured = {"input_ids": state["prompt"]["input_ids"].cpu(), "text": row["text"]}
            with torch.inference_mode():
                for name, encoder in encoders.items():
                    key = name + "_encoder"
                    if key in state.get("encoder_inputs", {}):
                        args = {k: v for k, v in state["encoder_inputs"][key].items() if k != "cache_key"}
                        captured[name] = encoder(**args)[name + "_embeds"].cpu()
            torch.save(captured, directory / (row["id"] + ".pt"))
            inputs.append({"id": row["id"], "body": body, "prompt": state["prompt"]["prompt_text"]})
        save(w.out / "reference-inputs.json", inputs)
        result["prepared_cases"] = len(inputs)
        if len(inputs) != len(rows) or not inputs:
            result["failed"] += 1
        return result
    for row in rows:
        item = {"id": row["id"]}
        try:
            native = torch.load(w.out / "reference-native" / (row["id"] + ".pt"), weights_only=True)
            gold = torch.load(w.out / "reference-hf" / (row["id"] + ".pt"), weights_only=True)
            item["input_ids_equal"] = torch.equal(native["input_ids"], gold["input_ids"])
            passed = item["input_ids_equal"]
            for name in ("image", "audio"):
                if name not in native:
                    continue
                x, y = native[name].float(), gold[name].float()
                if x.shape != y.shape or not torch.isfinite(x).all() or not torch.isfinite(y).all():
                    raise RuntimeError(f"{name}: shape/finite mismatch")
                cosine = F.cosine_similarity(x, y, dim=-1).min().item()
                item[name] = {"cosine_min": cosine, "max_abs_error": (x-y).abs().max().item(), "mean_abs_error": (x-y).abs().mean().item()}
                passed &= cosine >= .98
            ratio = difflib.SequenceMatcher(None, native["text"].strip(), gold["text"].strip(), autojunk=False).ratio()
            item.update(serving_text=native["text"], reference_text=gold["text"], sequence_ratio=ratio, passed=bool(passed and ratio >= .8))
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
    target_by_lang = {}
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
            target = row.get("target_text")
            if target is None:
                raise RuntimeError("Missing target_text: old run cannot provide both TTS gates")
            target_normalized = normalize_text(target, lang)
            target_error = jiwer.wer(target_normalized, ref)
            threshold = w.cfg["quality_thresholds"]["tts_roundtrip_error_max"]
            item.update(target_text=target, serving_text=row["text"], hypothesis=hyp,
                        target_to_text_error=target_error, text_to_audio_error=error,
                        verbatim_passed=target_error <= w.cfg["quality_thresholds"].get("tts_verbatim_error_max", .05),
                        audio_consistency_passed=error <= threshold)
            item["passed"] = item["verbatim_passed"] and item["audio_consistency_passed"]
            target_by_lang.setdefault(lang, []).append((target_normalized, ref))
            by_lang.setdefault(lang, []).append((ref, normalized_hyp))
        except Exception as exc:
            item.update(passed=False, error=str(exc))
        result["cases"].append(item)
        result["failed"] += not item["passed"]
        save(w.out / "results/tts_score.partial.json", result)
    result["verbatim_failure_count"] = sum(x.get("verbatim_passed") is False for x in result["cases"])
    result["audio_consistency_failure_count"] = sum(x.get("audio_consistency_passed") is False for x in result["cases"])
    result["scoring_error_count"] = sum("error" in x for x in result["cases"])
    result["target_to_text_corpus_error"] = {lang: jiwer.wer([p[0] for p in pairs], [p[1] for p in pairs]) for lang, pairs in target_by_lang.items()}
    result["text_to_audio_corpus_error"] = {lang: jiwer.wer([p[0] for p in pairs], [p[1] for p in pairs]) for lang, pairs in by_lang.items()}
    return result
