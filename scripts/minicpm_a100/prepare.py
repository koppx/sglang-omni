#!/usr/bin/env python3
"""Resolve immutable HF revisions, download complete models, stage fixed samples."""
from __future__ import annotations

import argparse
import ast
import fnmatch
import hashlib
import io
import json
from pathlib import Path
import random
import time

from run import save


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def retry(fn):
    for attempt in range(3):
        try:
            return fn()
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def prepare(out):
    import numpy as np
    import pyarrow.parquet as pq
    import soundfile as sf
    from huggingface_hub import HfApi, snapshot_download
    from PIL import Image, ImageDraw
    cfg = json.loads((out / "config.json").read_text())
    rng = random.Random(cfg["seed"])
    api = HfApi()
    media = out / "media"
    media.mkdir(exist_ok=True)
    manifest = {"models": {}, "datasets": {}, "samples": {}}

    for key, repo, revision in [("model", cfg["model_id"], cfg["model_revision"]),
                                ("asr_model", cfg["asr_model_id"], cfg["asr_model_revision"])]:
        local = cfg.get(key + "_path")
        if local:
            model_path = str(Path(local).expanduser().resolve())
            if not Path(model_path).is_dir():
                raise RuntimeError(f"Local model directory missing: {model_path}")
            resolved_revision = "local-content-hashed"
        else:
            info = retry(lambda: api.model_info(repo, revision=revision))
            resolved_revision = info.sha
            model_path = retry(lambda: snapshot_download(repo, revision=resolved_revision))
        index = Path(model_path) / "model.safetensors.index.json"
        if index.exists():
            for name in set(json.loads(index.read_text())["weight_map"].values()):
                if not (Path(model_path) / name).is_file():
                    raise RuntimeError(f"Missing weight shard {name}")
        if not list(Path(model_path).glob("*.safetensors")) and not list(Path(model_path).glob("*.bin")):
            raise RuntimeError(f"No model weights in {model_path}")
        if key == "model" and not (Path(model_path) / "assets/token2wav").is_dir():
            raise RuntimeError("Checkpoint has no assets/token2wav")
        manifest[key + "_path"] = model_path
        manifest["models"][key] = {"repo": repo, "revision": resolved_revision, "source": "local" if local else "hub",
            "files": {str(p.relative_to(model_path)): {"bytes": p.stat().st_size, "sha256": sha(p)}
                      for p in sorted(Path(model_path).rglob("*")) if p.is_file()}}
        save(out / "resources.partial.json", manifest)

    specs = [
        ("librispeech_clean", "openslr/librispeech_asr", "main", "clean/test/*.parquet", cfg["samples_asr_per_split"], "en"),
        ("librispeech_other", "openslr/librispeech_asr", "main", "other/test/*.parquet", cfg["samples_asr_per_split"], "en"),
        ("fleurs_zh", "google/fleurs", "refs/convert/parquet", "cmn_hans_cn/test/*.parquet", cfg["samples_asr_per_split"], "zh"),
        ("mmmu", "MMMU/MMMU", "main", "*validation*.parquet", cfg["samples_mmmu"], None),
    ]
    for key, repo, ref, pattern, count, lang in specs:
        ref = cfg.get("dataset_revisions", {}).get(key, ref)
        info = retry(lambda: api.dataset_info(repo, revision=ref))
        names = sorted(s.rfilename for s in info.siblings if fnmatch.fnmatch(s.rfilename, pattern))
        if not names:
            raise RuntimeError(f"No files match {repo}@{info.sha}/{pattern}")
        path = Path(retry(lambda: snapshot_download(repo, repo_type="dataset", revision=info.sha, allow_patterns=names)))
        # Reservoir sampling does not hold the full audio dataset in RAM.
        reservoir, seen = [], 0
        buckets = {}
        for name in names:
            for batch in pq.ParquetFile(path / name).iter_batches(batch_size=16):
                for row in batch.to_pylist():
                    if key == "mmmu":
                        if row.get("question_type") != "multiple-choice":
                            continue
                        # Round-robin subjects below avoids a first-subject-only sample.
                        subject = name.split("/")[0]
                        buckets.setdefault(subject, []).append(row)
                    else:
                        seen += 1
                        if len(reservoir) < count:
                            reservoir.append(row)
                        else:
                            j = rng.randrange(seen)
                            if j < count:
                                reservoir[j] = row
        if key == "mmmu":
            for rows in buckets.values():
                rng.shuffle(rows)
            while len(reservoir) < count and any(buckets.values()):
                for subject in sorted(buckets):
                    if buckets[subject] and len(reservoir) < count:
                        row = buckets[subject].pop()
                        row["subject"] = subject
                        reservoir.append(row)
        if len(reservoir) != count:
            raise RuntimeError(f"{key}: expected {count} samples, found {len(reservoir)}")
        staged = []
        for i, row in enumerate(reservoir):
            item = {"id": str(row.get("id", row.get("sample_id", i))), "dataset": key}
            if lang:
                value = row["audio"]
                raw = value.get("bytes")
                if raw is None:
                    candidate = (path / value["path"]).resolve()
                    if not candidate.is_relative_to(path.resolve()):
                        raise RuntimeError("Audio path escapes downloaded snapshot")
                    raw = candidate.read_bytes()
                audio, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
                target = media / f"{key}-{i:04d}.wav"
                sf.write(target, audio.mean(axis=1), sr, subtype="PCM_16")
                item.update(audio=str(target), lang=lang, duration_s=len(audio) / sr,
                            target=row.get("text", row.get("transcription", "")), sha256=sha(target))
            else:
                images = []
                for n in range(1, 8):
                    value = row.get(f"image_{n}")
                    if value is None:
                        continue
                    image = Image.open(io.BytesIO(value["bytes"])).convert("RGB")
                    target = media / f"mmmu-{i:04d}-{n}.png"
                    image.save(target)
                    images.append(str(target))
                options = row["options"]
                if isinstance(options, str):
                    options = ast.literal_eval(options)
                prompt = row["question"] + "\n" + "\n".join(f"{chr(65+j)}. {s}" for j, s in enumerate(options))
                prompt += "\nAnswer with 'Answer: LETTER' on the final line."
                item.update(images=images, prompt=prompt, target=row["answer"], subject=row["subject"])
            staged.append(item)
        manifest["datasets"][key] = {"repo": repo, "revision": info.sha, "files": names, "count": len(staged), "seed": cfg["seed"]}
        manifest["samples"][key] = staged
        save(out / "resources.partial.json", manifest)
    # Locally generated, unambiguous visual fixtures and audio length boundaries.
    for color in ("red", "blue"):
        path = media / f"{color}.png"
        image = Image.new("RGB", (336, 336), color)
        ImageDraw.Draw(image).text((40, 150), color.upper(), fill="white")
        image.save(path)
    ref = manifest["samples"]["librispeech_clean"][0]
    audio, sr = sf.read(ref["audio"], dtype="float32")
    for seconds in (0.01, 1, 29, 31, 60):
        samples = np.resize(audio, max(1, int(sr * seconds)))
        sf.write(media / f"boundary-{seconds}.wav", samples, sr)
    sf.write(media / "silence.wav", np.zeros(16000, dtype=np.float32), 16000)
    save(out / "resources.json", manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    prepare(parser.parse_args().run_dir)
