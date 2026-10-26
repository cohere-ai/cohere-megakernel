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
import native
import prefill.torch_prefill as torch_prefill


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--id', required=True)
    p.add_argument('--stable-router', action='store_true',
                   help='use distinct signed router rows to remove random-weight top-k boundary ties; correctness fixture only')
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
               'comparison': 'one shared weights/KV/page mapping/bootstrap fixture across all modes; not checkpoint correctness',
               'comparisons': []}
    summary['stable_router_fixture'] = a.stable_router
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
    native.bind(library_path=str(ROOT/'build/libmk_release.so'), debug=False)
    torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    cfg = torch_prefill.NmcConfig.release()
    weights = torch_prefill.make_fast_nmc_weights(cfg=cfg, device=torch.device('cuda'),
                dtype=torch.bfloat16, upgate_chunk=64, moe_upgate_chunk=64)
    if a.stable_router:
        direction = torch.randn((cfg.num_hidden_layers,1,cfg.hidden_size),device='cuda') * 0.0002
        coefficients = torch.zeros((1,cfg.num_experts,1),device='cuda')
        coefficients[0,:8,0] = torch.arange(8,0,-1,device='cuda')
        coefficients[0,8:16,0] = -torch.arange(8,0,-1,device='cuda')
        weights['W_router'].copy_(direction*coefficients)
    prefill = torch_prefill.make_synthetic_kv_prefill(cfg=cfg, device=torch.device('cuda'),
                dtype=torch.bfloat16, batch=1, fake_prompt_len=4096, max_new_tokens=2,
                page_block=64, frac_vram_utilization=0.9, row_active=None)
    # Reserve the decode page once. Re-creating handles can remap logical KV
    # pages to different physical noise, even when the generator seed matches.
    prefill.kv_cache.step_decode_positions([4096], [1])
    fixed_page_table = prefill.kv_cache.page_table.detach().cpu().clone()
    summary['bootstrap_token'] = prefill.next_token.detach().cpu().tolist()
    summary['kv_pool_shape'] = list(prefill.kv_cache.k_pool.shape)
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
                # Top-k slot order is not a semantic invariant. Require exact
                # selected expert sets, matching the existing layer suite's metric.
                captured['topk_expert_sets'] = torch.sort(
                    state.tensors['topk_experts'][1:].detach().cpu(), dim=-1).values
                return out
            schedule._launch_nmc_decode = launch
            with (d / f'stdout/{label}.log').open('w') as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                result = schedule.run_mk_decode_from_prefill(prefill=prefill, weights=weights, cfg=cfg,
                    num_sms=132, max_new_tokens=2, temperature=0.0, gumbel_sampler=False,
                    max_attn_splits=32, min_attn_chunk=512, fast=True, decode_timeout_sec=30.0,
                    cpp_runtime=False, attn_drain=True, attn_drain_sms=None, moe_drain_sms=None,
                    profile_path=None, profile_max_events=65536, rr_dependency_affinity=False,
                    moe_combine_atomic_tma=False, ablation=False)
            if not torch.equal(prefill.kv_cache.page_table.detach().cpu(), fixed_page_table):
                raise RuntimeError('fixture page mapping changed')
            captured['tokens'] = torch.tensor(result['token_ids_by_batch'])
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
        prefill.close()
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
