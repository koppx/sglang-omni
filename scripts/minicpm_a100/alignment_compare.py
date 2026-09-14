"""Check actual captured token/hidden alignment across prefill configurations."""
import argparse
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from run import save
p = argparse.ArgumentParser()
p.add_argument('--run-dir', type=Path, required=True)
a = p.parse_args()
cfg = json.loads((a.run_dir / 'config.json').read_text())
r = dict(failed=0, cases=[], violations=[], note='Diagnostic hook changes timing; no performance inference. Greedy token divergence blocks numerical alignment comparison.')
groups = []
for name in ('alignment_baseline', 'alignment_chunked'):
    directory = a.run_dir / name
    data = {}
    for path in directory.glob('*.pt'):
        x = torch.load(path, map_location='cpu', weights_only=True)
        if x['prompt_ids'] is None or x['hidden'] is None:
            r['violations'].append(f'{path.name}: missing prompt/hidden capture')
            continue
        key = tuple(x['prompt_ids'])
        # Ignore startup warmup; diagnostic cases have substantial prompts.
        if len(key) < 200:
            continue
        data[key] = x
        if len(x['hidden']) != len(x['output_ids']):
            r['violations'].append(f'{name}/{path.name}: hidden count != generated token count')
    events = [json.loads(line) for path in directory.glob('events-*.jsonl') for line in path.read_text().splitlines()]
    if not events or any(e['middle_chunks'] is None for e in events):
        r['violations'].append(f'{name}: missing scheduler chunk state instrumentation')
    if name.endswith('chunked') and not any((e['middle_chunks'] or 0) > 0 for e in events):
        r['violations'].append('chunked: no observed middle chunks; test did not exercise target path')
    for e in events:
        if (e['middle_chunks'] or 0) > 0 and e['after'] != e['before']:
            r['violations'].append(f'{name}/{e["request_id"]}: middle chunk appended hidden')
    for path in directory.glob('hook-error-*.txt'):
        r['violations'].append(path.read_text())
    groups.append(data)
if len(groups[0]) < 3 or groups[0].keys() != groups[1].keys():
    r['violations'].append('Missing or different prompt cases: require all three paired prompts')
for key in groups[0].keys() & groups[1].keys():
    x, y = groups[0][key], groups[1][key]
    same = x['output_ids'] == y['output_ids']
    item = dict(prompt_tokens=len(key), tokens_equal=same, shape_equal=x['hidden'].shape == y['hidden'].shape)
    if same and item['shape_equal']:
        item['cosine_min'] = F.cosine_similarity(x['hidden'].float(), y['hidden'].float(), dim=-1).min().item()
        item['passed'] = item['cosine_min'] >= cfg.get('alignment_cosine_min', .98)
    else:
        item.update(passed=False, reason='Cannot compare hidden numerically on different token trajectories/shapes')
    r['cases'].append(item)
r['failed'] = len(r['violations']) + sum(not x['passed'] for x in r['cases'])
save(a.run_dir / 'results/alignment_compare.json', r)
print(json.dumps(r, indent=2))
raise SystemExit(bool(r['failed']))
