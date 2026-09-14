"""Audit the narrowly authorized MiniCPM metadata overrides; reject anything else."""
import argparse
import importlib.metadata as md
import json
from pathlib import Path
import subprocess
import sys
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from run import save

p = argparse.ArgumentParser()
p.add_argument('--run-dir', type=Path, required=True)
a = p.parse_args()
cfg = json.loads((a.run_dir / 'config.json').read_text())
check = subprocess.run([sys.executable, '-m', 'pip', 'check'], capture_output=True, text=True)
allowed = {'librosa', 'pillow', 'decord', 'moviepy', 'transformers', 'soundfile', 'onnxruntime'}
issues = []
for dist in md.distributions():
    owner = canonicalize_name(dist.metadata['Name'])
    for raw in dist.requires or []:
        req = Requirement(raw)
        extras = ['', 'tts'] if owner == 'minicpmo-utils' else ['']
        if req.marker and not any(req.marker.evaluate({'extra': e}) for e in extras):
            continue
        try:
            version = md.version(req.name)
        except md.PackageNotFoundError:
            version = None
        if version is None or (req.specifier and not req.specifier.contains(version, prereleases=True)):
            accepted = cfg['dependency_policy'] == 'minicpmo-override' and owner == 'minicpmo-utils' and canonicalize_name(req.name) in allowed
            issues.append(dict(package=owner, requirement=str(req), installed=version, accepted_override=accepted))
result = dict(policy=cfg['dependency_policy'], dependency_consistent=not issues, issues=issues,
              pip_check_exit=check.returncode, pip_check_output=check.stdout + check.stderr,
              note='Accepted overrides do not prove whole-package compatibility; missing moviepy/decord are outside Token2wav scope.')
# pip may catch inconsistencies our marker walk did not explain: reject those too.
unexplained = [line for line in check.stdout.splitlines() if line and line != 'No broken requirements found.' and not (cfg['dependency_policy'] == 'minicpmo-override' and line.lower().startswith('minicpmo-utils '))]
result['unexplained_pip_errors'] = unexplained
result['failed'] = sum(not i['accepted_override'] for i in issues) + len(unexplained) + int(check.returncode not in (0, 1))
save(a.run_dir / 'results/dependency_audit.json', result)
print(json.dumps(result, indent=2))
if not result['failed']:
    from stepaudio2 import Token2wav
sys.exit(bool(result['failed']))
