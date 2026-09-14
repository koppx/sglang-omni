"""Run only HF reference code in the isolated Transformers 4.52 environment."""
import argparse
import base64
import io
import json
from pathlib import Path
import traceback
import torch
from PIL import Image
from transformers import AutoConfig, AutoModel, AutoProcessor
from run import save

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--preflight', action='store_true')
    a = p.parse_args()
    resources = json.loads((a.run_dir / 'resources.json').read_text())
    cfg = json.loads((a.run_dir / 'config.json').read_text())
    model_path = resources['model_path']
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config.init_tts = False
    config.init_audio = config.init_vision = True
    model = AutoModel.from_pretrained(model_path, config=config, trust_remote_code=True,
             torch_dtype=torch.bfloat16, attn_implementation='sdpa').eval().cuda()
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    # Remote audio encoder versions expect three attention outputs.
    for layer in model.apm.layers:
        original = layer.self_attn.forward
        def adapted(*args, _original=original, **kwargs):
            value = _original(*args, **kwargs)
            return (*value, None) if isinstance(value, tuple) and len(value) == 2 else value
        layer.self_attn.forward = adapted
    if a.preflight:
        answer = model.chat(msgs=[dict(role='user', content='Reply OK.')], processor=processor,
            tokenizer=processor.tokenizer, do_sample=False, generate_audio=False,
            enable_thinking=cfg['enable_thinking'], max_new_tokens=32)
        save(a.run_dir / 'results/reference_preflight.json', dict(failed=0, response=str(answer)))
        return
    inputs = json.loads((a.run_dir / 'reference-inputs.json').read_text())
    target = a.run_dir / 'reference-hf'
    target.mkdir(exist_ok=True)
    results = []
    for entry in inputs:
        item = dict(id=entry['id'])
        try:
            from reference import _audio
            body = entry['body']
            imgs = [Image.open(io.BytesIO(base64.b64decode(x.split(',', 1)[1]))).convert('RGB') for x in body.get('images', [])]
            audios = [_audio(io.BytesIO(base64.b64decode(x.split(',', 1)[1]))) for x in body.get('audios', [])]
            data = processor(entry['prompt'], images=[imgs] if imgs else None, audios=[audios] if audios else None, return_tensors='pt').to('cuda')
            captured = dict(input_ids=data['input_ids'][0].cpu())
            with torch.inference_mode():
                for name, enabled in [('image', bool(imgs)), ('audio', bool(audios))]:
                    if not enabled:
                        continue
                    value = model.get_vision_embedding(data) if name == 'image' else model.get_audio_embedding(data, chunk_length=model.config.audio_chunk_length)
                    tensors = []
                    def flatten(v):
                        if isinstance(v, torch.Tensor):
                            tensors.append(v.reshape(-1, v.shape[-1]).cpu())
                        elif isinstance(v, (tuple, list)):
                            for child in v:
                                flatten(child)
                    flatten(value)
                    captured[name] = torch.cat(tensors)
                messages = body['messages']
                messages[-1]['content'] = [*imgs, *audios, messages[-1]['content']]
                # Verify the prompt actually used by remote.chat, including thinking/template
                # defaults. A mismatch must not be reported as a model precision regression.
                calls = []
                processor_type = type(processor)
                original_call = processor_type.__call__
                def tracked(instance, *args, **kwargs):
                    output = original_call(instance, *args, **kwargs)
                    if "input_ids" in output:
                        calls.append(output["input_ids"].detach().cpu())
                    return output
                processor_type.__call__ = tracked
                try:
                    answer = model.chat(msgs=messages, processor=processor, tokenizer=processor.tokenizer,
                        do_sample=False, generate_audio=False, enable_thinking=cfg['enable_thinking'],
                        repetition_penalty=1.0, max_new_tokens=body.get('max_tokens', 256))
                finally:
                    processor_type.__call__ = original_call
                if not calls or not torch.equal(calls[-1].reshape(-1), captured['input_ids'].reshape(-1)):
                    raise RuntimeError('HF chat prompt differs from serving preprocessing (including thinking/template); comparison blocked')
            if isinstance(answer, tuple):
                answer = answer[0]
            captured['text'] = str(answer)
            torch.save(captured, target / (entry['id'] + '.pt'))
            item['passed'] = True
        except Exception:
            item.update(passed=False, error=traceback.format_exc())
        results.append(item)
        save(a.run_dir / 'results/reference_export.partial.json', dict(cases=results))
    result = dict(failed=sum(not x['passed'] for x in results), cases=results)
    save(a.run_dir / 'results/reference_export.json', result)
    if not results:
        raise RuntimeError('No reference cases prepared')
    raise SystemExit(bool(result['failed']))

if __name__ == '__main__':
    main()
