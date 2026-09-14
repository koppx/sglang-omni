"""Opt-in diagnostic instrumentation; never enabled for benchmark timing."""
import os
if os.environ.get('MINICPM_ALIGNMENT_DIR'):
    try:
        import json
        import hashlib
        from pathlib import Path
        import torch
        from sglang_omni.models.minicpm_o.thinker_model_runner import MiniCPMOThinkerModelRunner as Runner
        root = Path(os.environ['MINICPM_ALIGNMENT_DIR'])
        root.mkdir(parents=True, exist_ok=True)
        original_post = Runner.post_process_outputs
        original_finish = Runner.on_request_finished

        def post(self, result, scheduler_output, outputs):
            before = {r.request_id: len(self._pending_hidden.get(r.request_id, [])) for r in scheduler_output.requests}
            original_post(self, result, scheduler_output, outputs)
            for r in scheduler_output.requests:
                req = getattr(r.data, 'req', None)
                middle = getattr(req, 'inflight_middle_chunks', None)
                after = len(self._pending_hidden.get(r.request_id, []))
                with (root / f'events-{os.getpid()}.jsonl').open('a') as f:
                    f.write(json.dumps(dict(request_id=r.request_id, middle_chunks=middle, before=before[r.request_id], after=after)) + '\n')

        def finish(self, request_id, data):
            original_finish(self, request_id, data)
            sequence = data.extra_model_outputs.get('hidden_states_seq')
            req = getattr(data, 'req', None)
            prompt = getattr(req, 'origin_input_ids', None)
            if prompt is None:
                prompt = getattr(data, 'input_ids', None)
            if isinstance(prompt, torch.Tensor):
                prompt = prompt.tolist()
            # Scheduler copies req.output_ids to data only AFTER this callback.
            ids = list(req.output_ids) if req is not None else None
            key = hashlib.sha256(json.dumps(prompt).encode()).hexdigest()
            torch.save(dict(request_id=request_id, prompt_ids=prompt, output_ids=ids,
                            hidden=torch.stack(sequence) if sequence else None), root / f'{key}-{request_id}.pt')
        Runner.post_process_outputs = post
        Runner.on_request_finished = finish
    except Exception:
        import traceback
        from pathlib import Path
        root = Path(os.environ['MINICPM_ALIGNMENT_DIR'])
        root.mkdir(parents=True, exist_ok=True)
        (root / f'hook-error-{os.getpid()}.txt').write_text(traceback.format_exc())
