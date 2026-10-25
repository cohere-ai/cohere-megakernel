#!/usr/bin/env python3
"""Repeat unmodified synthetic decode with same-process warmups per repeat."""
import argparse
import json
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def worker(a):
    sys.path.insert(0, str(ROOT / 'src'))
    import torch
    import runner
    flags = ['--mk', '--fast', '--lib-path', str(ROOT / 'build/libmk_release.so'),
             '--batch-size', '1', '--max-new-tokens', str(a.tokens),
             '--fake-prompt-len', str(a.context), '--num-sms', '132',
             '--frac-vram-utilization', '0.9', '--cpp-decode-runtime']
    args = runner._parse_args(flags)
    params = runner.SamplingParams(max_new_tokens=a.tokens, temperature=0.0)
    results = []
    for phase in ['warmup', 'timed']:
        torch.manual_seed(a.seed)
        torch.cuda.manual_seed_all(a.seed)
        report = runner._run_native_benchmark(args, params)
        r = report.result
        # These native fields exclude the bootstrap token; the display summary
        # includes it. C++ step timings are wall/n, not per-token samples.
        record = {k: r[k] for k in ['decode_wall_ms', 'tpot_ms',
                  'throughput_tok_s', 'num_output_tokens', 'prompt_len',
                  'num_context_buckets', 'num_unique_schedules', 'total_inst',
                  'max_inst_per_sm']}
        record['phase'] = phase
        results.append(record)
        print('RESEARCH_' + phase.upper() + '=' + json.dumps(record), flush=True)
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--id', required=True)
    p.add_argument('--context', type=int, default=8192)
    p.add_argument('--tokens', type=int, default=512)
    p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.context < 1 or a.tokens < 2 or a.repeats < 5:
        p.error('positive context, >=2 tokens and >=5 repeats required')
    if a.worker:
        worker(a)
        return
    if Path(a.id).name != a.id or a.id in ['.', '..']:
        p.error('id must be a single directory name')
    d = ROOT / 'research/results' / a.id
    d.mkdir(parents=True, exist_ok=False)
    (d / 'stdout').mkdir()
    (d / 'raw').mkdir()
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    status = subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True)
    metadata = {'experiment_id': a.id, 'git_commit': commit, 'dirty_status': status,
                'variant': 'baseline', 'weights': 'synthetic',
                'routing': 'synthetic weights, emergent routing; no forced uniform routing',
                'batch_size': 1, 'context_length': a.context, 'output_tokens': a.tokens,
                'executed_decode_steps_expected': a.tokens - 1,
                'seed': a.seed, 'warmup': 'one full same-process decode before each timed decode',
                'clock_control': 'none; query recorded', 'job_id': os.environ.get('SLURM_JOB_ID'),
                'repetitions': [], 'status': 'running', 'started_unix': time.time()}
    summary = d / 'summary.json'
    def save():
        summary.write_text(json.dumps(metadata, indent=2) + '\n')
    save()
    (d / 'metadata.md').write_text('# ' + a.id + '\n\nHypothesis: upstream synthetic decode executes repeatedly on the fixed driver.\n'
        'No prefetch behavior change; no bandwidth-interference claim.\n\n'
        + 'Commit: ' + commit + '\nDirty status:\n```\n' + status + '```\n'
        'Configuration and all repetitions: summary.json. Exact commands: commands.sh.\n')
    with (d / 'stdout/environment.log').open('w') as log:
        for cmd in [['nvidia-smi', '-q'], [os.environ.get('NVCC', 'nvcc'), '--version'],
                    [sys.executable, '--version'], [sys.executable, '-m', 'pip', 'freeze']]:
            log.write('$ ' + shlex.join(cmd) + '\n'); log.flush()
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
    commands = []
    try:
        for i in range(a.repeats):
            cmd = [sys.executable, str(Path(__file__).resolve()), '--id', a.id,
                   '--context', str(a.context), '--tokens', str(a.tokens),
                   '--repeats', str(a.repeats), '--seed', str(a.seed), '--worker']
            commands.append(shlex.join(cmd))
            (d / 'commands.sh').write_text('#!/usr/bin/env bash\n' + '\n'.join(commands) + '\n')
            logpath = d / f'stdout/repeat-{i + 1:02d}.log'
            with logpath.open('w') as log:
                subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            lines = logpath.read_text().splitlines()
            timed = [json.loads(x.split('=', 1)[1]) for x in lines if x.startswith('RESEARCH_TIMED=')]
            if len(timed) != 1 or timed[0]['num_output_tokens'] != a.tokens - 1:
                raise RuntimeError('missing metrics or unexpected executed step count')
            metadata['repetitions'].append(timed[0])
            save()
        vals = [r['tpot_ms'] for r in metadata['repetitions']]
        q = statistics.quantiles(vals, n=4)
        metadata.update(median_tpot_ms=statistics.median(vals),
                        min_tpot_ms=min(vals), max_tpot_ms=max(vals),
                        p25_tpot_ms=q[0], p75_tpot_ms=q[2], stdev_tpot_ms=statistics.stdev(vals),
                        median_throughput_tps=statistics.median(r['throughput_tok_s'] for r in metadata['repetitions']),
                        status='complete')
    except BaseException as exc:
        metadata.update(status='failed', failure=repr(exc))
        raise
    finally:
        metadata['finished_unix'] = time.time()
        save()
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
