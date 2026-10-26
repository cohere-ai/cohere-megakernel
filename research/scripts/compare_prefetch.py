#!/usr/bin/env python3
"""Balanced paired timing; each child performs a full same-process warmup."""
import argparse
import json
import os
from pathlib import Path
import random
import shlex
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--id',required=True)
    p.add_argument('--contexts',type=int,nargs='+',default=[4096,131072])
    p.add_argument('--mode',choices=['after-input','qkv','router','moe-upgate','moe-down'],default='after-input')
    p.add_argument('--pairs',type=int,default=6)
    p.add_argument('--tokens',type=int,default=512)
    a=p.parse_args()
    if Path(a.id).name!=a.id or a.id in ('.','..') or a.pairs<5 or a.tokens<2 or min(a.contexts)<1:
        p.error('safe id, >=5 pairs, >=2 tokens, positive contexts required')
    d=ROOT/'research/results'/a.id
    d.mkdir(parents=True,exist_ok=False); (d/'stdout').mkdir(); (d/'raw').mkdir()
    summary=dict(experiment_id=a.id,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                 dirty_status=subprocess.check_output(['git','status','--short'],text=True),
                 status='running',weights='synthetic',routing='emergent, not forced uniform',batch_size=1,
                 contexts=a.contexts,output_tokens=a.tokens,decode_steps=a.tokens-1,variant=a.mode,
                 pairs=a.pairs,seed=42,order='alternating AB/BA within each context; contexts shuffled per round (seed42)',
                 warmup='full same-process decode per child before timed decode',
                 clock_control='none',job_id=os.environ.get('SLURM_JOB_ID'),repetitions=[],started_unix=time.time())
    def save(): (d/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    save()
    (d/'metadata.md').write_text('# '+a.id+'\n\nHypothesis: the causal benefit of early prefetch changes with context.\n'
        'Baseline and selected opcode/all after-input variant; tiling, routing algorithm, precision and tasks unchanged.\n'
        'All raw repetitions retained. Expected sign is not assumed. Full configuration/results: summary.json.\n')
    with (d/'stdout/environment.log').open('w') as f:
        for cmd in [['nvidia-smi','-q'],[os.environ.get('NVCC','nvcc'),'--version'],[sys.executable,'-m','pip','freeze']]:
            subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
    commands=[]; rng=random.Random(42)
    try:
        for pair in range(a.pairs):
            contexts=list(a.contexts); rng.shuffle(contexts)
            for context in contexts:
                modes=['baseline',a.mode] if pair%2==0 else [a.mode,'baseline']
                for mode in modes:
                    cmd=[sys.executable,str(ROOT/'research/scripts/baseline.py'),'--id',a.id,'--context',str(context),
                         '--tokens',str(a.tokens),'--mode',mode,'--worker']
                    commands.append(shlex.join(cmd)); (d/'commands.sh').write_text('\n'.join(commands)+'\n')
                    log=d/f'stdout/s{context}-pair{pair+1:02d}-{mode}.log'
                    with log.open('w') as f:
                        subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
                    records=[json.loads(x.split('=',1)[1]) for x in log.read_text().splitlines() if x.startswith('RESEARCH_TIMED=')]
                    if len(records)!=1 or records[0]['num_output_tokens']!=a.tokens-1:
                        raise RuntimeError('invalid measurement')
                    records[0].update(context=context,pair=pair+1,mode=mode)
                    summary['repetitions'].append(records[0]); save()
                    print(context,pair+1,mode,records[0]['tpot_ms'],flush=True)
        stats=[]
        for context in a.contexts:
            rows=[x for x in summary['repetitions'] if x['context']==context]
            b=[x['tpot_ms'] for x in rows if x['mode']=='baseline']
            v=[x['tpot_ms'] for x in rows if x['mode']==a.mode]
            deltas=[(v0-b0)/b0*100 for b0,v0 in zip(b,v)]
            stats.append(dict(context=context,baseline_tpot_median=statistics.median(b),variant_tpot_median=statistics.median(v),
                              baseline_stdev=statistics.stdev(b),variant_stdev=statistics.stdev(v),
                              paired_prefetch_benefit_pct=deltas,median_paired_benefit_pct=statistics.median(deltas),
                              baseline_range=[min(b),max(b)],variant_range=[min(v),max(v)]))
        summary.update(status='complete',statistics=stats)
    except BaseException as exc:
        summary.update(status='failed',failure=repr(exc)); raise
    finally:
        summary['finished_unix']=time.time(); save()
    print(json.dumps(summary['statistics'],indent=2))


if __name__=='__main__': main()
