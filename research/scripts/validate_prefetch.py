#!/usr/bin/env python3
"""Differential mock scheduling check. Does not validate model quality."""
import argparse
import contextlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
import torch
import runner
import decode.schedule as schedule


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--id', required=True)
    a = p.parse_args()
    if Path(a.id).name != a.id or a.id in ('.', '..'):
        p.error('invalid experiment id')
    d = ROOT / 'research/results' / a.id
    d.mkdir(parents=True, exist_ok=False)
    (d / 'stdout').mkdir(); (d / 'raw').mkdir()
    summary = {'experiment_id': a.id, 'status': 'running', 'seed': 42,
               'git_commit': subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
               'dirty_status': subprocess.check_output(['git','status','--short'],text=True),
               'weights': 'synthetic', 'routing': 'emergent; not forced',
               'context': 4096, 'batch_size': 1, 'requested_tokens': 2,
               'comparison': 'same seeded one-step mock fixtures; not checkpoint reference correctness',
               'comparisons': []}
    def save():
        (d / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    save()
    (d / 'commands.sh').write_text(shlex.join([sys.executable, str(Path(__file__).resolve()), '--id', a.id]) + '\n')
    (d / 'metadata.md').write_text('# ' + a.id + '\n\nHypothesis: moving input waits preserves finite mock outputs within baseline numerical noise.\n'
        'Expected: finite tensors, matching sampled token, error <=max(1e-5, 4*baseline-repeat error).\n'
        'Full configuration, tolerances, observations: summary.json; logs and tensors stay on server.\n')
    with (d / 'stdout/environment.log').open('w') as f:
        for cmd in [['nvidia-smi','-q'], [os.environ.get('NVCC','nvcc'),'--version'], [sys.executable,'-m','pip','freeze']]:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True)
    original = schedule._launch_nmc_decode
    snapshots = {}
    modes = [('baseline-1','baseline'), ('baseline-2','baseline'),
             ('after-input','after-input'), ('qkv','qkv'), ('router','router'),
             ('moe-upgate','moe-upgate'), ('moe-down','moe-down')]
    try:
        for label, mode in modes:
            os.environ['NMC_RESEARCH_PREFETCH'] = mode
            captured = {}
            def launch(**kwargs):
                out = original(**kwargs)
                state = kwargs['state']
                # These are materialized final-layer outputs, not unused padded slots.
                for name in ['lm_logits','x_raw','x_resid','q_out','o_proj_in','router_logits']:
                    captured[name] = state.tensors[name].detach().cpu().clone()
                captured['topk_experts'] = state.tensors['topk_experts'][1:].detach().cpu().clone()
                return out
            schedule._launch_nmc_decode = launch
            torch.manual_seed(42); torch.cuda.manual_seed_all(42)
            flags = ['--mk','--fast','--lib-path',str(ROOT/'build/libmk_release.so'),
                     '--batch-size','1','--fake-prompt-len','4096','--max-new-tokens','2',
                     '--num-sms','132','--frac-vram-utilization','0.9']
            args = runner._parse_args(flags)
            with (d / f'stdout/{label}.log').open('w') as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                report = runner._run_native_benchmark(args, runner.SamplingParams(2,0.0))
            captured['tokens'] = torch.tensor(report.result['token_ids_by_batch'])
            torch.save(captured, d / f'raw/{label}.pt')
            snapshots[label] = captured
            print('captured', label, flush=True)
        ref = snapshots['baseline-1']
        noise = {k: (snapshots['baseline-2'][k].float()-v.float()).abs().max().item()
                 for k,v in ref.items()}
        summary['baseline_repeat_max_abs'] = noise
        for label, tensors in snapshots.items():
            for k, v in tensors.items():
                finite = bool(torch.isfinite(v).all())
                err = (v.float()-ref[k].float()).abs().max().item()
                # Integer IDs/tokens must match exactly; floating results use
                # the observed repeated-baseline noise, never model quality tolerance.
                tol = max(1e-5, 4*noise[k]) if v.is_floating_point() else 0.0
                summary['comparisons'].append(dict(mode=label,tensor=k,finite=finite,max_abs=err,
                                                    tolerance=tol,passed=finite and err<=tol))
        summary['status'] = 'passed' if all(x['passed'] for x in summary['comparisons']) else 'failed'
        save()
        if summary['status'] != 'passed':
            raise RuntimeError('differential mock check failed; inspect retained tensors and baseline noise')
    except BaseException as exc:
        summary.update(status='failed', failure=repr(exc)); save(); raise
    finally:
        schedule._launch_nmc_decode = original
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
