#!/usr/bin/env python3
"""Plot completed paired summaries; retain absolute and relative deltas."""
import argparse
import json
from pathlib import Path
import statistics
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def interval(values):
    x=np.asarray(values,dtype=float)
    rng=np.random.default_rng(42)
    med=np.median(rng.choice(x,size=(20000,len(x)),replace=True),axis=1)
    return np.percentile(med,[2.5,97.5])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('summary',type=Path)
    p.add_argument('--output',type=Path,required=True,help='output prefix')
    a=p.parse_args(); s=json.loads(a.summary.read_text())
    if s['status']!='complete':raise ValueError('refuse to plot incomplete/failed experiment')
    contexts=s['contexts']; labels=[f'{c//1024}K' for c in contexts]
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,
                         'svg.fonttype':'none','pdf.fonttype':42})
    fig,axes=plt.subplots(1,3,figsize=(11,3.6),gridspec_kw={'width_ratios':[1.2,1,1]})
    colors={'baseline':'#16697a',s['variant']:'#c56b2f'}
    absolute=[];relative=[]
    for i,c in enumerate(contexts):
        rows=[r for r in s['repetitions'] if r['context']==c]
        by={m:{r['pair']:r['tpot_ms'] for r in rows if r['mode']==m} for m in colors}
        pairs=sorted(by['baseline'])
        b=[by['baseline'][k] for k in pairs];v=[by[s['variant']][k] for k in pairs]
        absolute.append([(v0-b0)*1000 for b0,v0 in zip(b,v)])
        relative.append([(v0-b0)/b0*100 for b0,v0 in zip(b,v)])
        for mode,values,offset in [('baseline',b,-.15),(s['variant'],v,.15)]:
            x=i+offset
            axes[0].scatter(x+np.linspace(-.035,.035,len(values)),values,s=20,
                            color=colors[mode],alpha=.65,label=mode if i==0 else None)
            axes[0].plot([x-.07,x+.07],[statistics.median(values)]*2,color=colors[mode],lw=2.5)
    axes[0].set(ylabel='Decode TPOT (ms)',xticks=range(len(labels)),xticklabels=labels,
                title='Measured latency')
    axes[0].legend(frameon=False,fontsize=8)
    for ax,data,title,ylabel in [(axes[1],absolute,'Absolute prefetch benefit','After-input − baseline (µs/token)'),
                                 (axes[2],relative,'Relative prefetch benefit','(After-input − baseline) / baseline (%)')]:
        for i,values in enumerate(data):
            med=statistics.median(values);lo,hi=interval(values)
            ax.scatter(i+np.linspace(-.04,.04,len(values)),values,s=22,color='#607d8b',alpha=.65)
            ax.errorbar(i,med,yerr=[[med-lo],[hi-med]],fmt='D',color='#172a3a',capsize=5,ms=5)
        ax.axhline(0,color='#777777',ls='--',lw=.8)
        ax.set(title=title,ylabel=ylabel,xticks=range(len(labels)),xticklabels=labels)
        ax.set_xlim(-.5,len(labels)-.5)
    for ax in axes:
        ax.set_xlabel('Initial context length');ax.grid(axis='y',alpha=.18)
    fig.suptitle('Synthetic decode: sensitivity to delaying early weight loads',fontsize=12,y=.99)
    fig.text(.5,.065,f'H100 · BS1 · synthetic weights & KV · {s["pairs"]} paired repeats · '
             f'{s["decode_steps"]} decode steps after process-local warmup',ha='center',fontsize=8)
    fig.text(.5,.02,'Dots: all repetitions/pairs. Diamonds: median with 95% paired-bootstrap interval. '
             'Positive benefit favors early prefetch; no HBM mechanism is inferred.',ha='center',fontsize=7.3)
    fig.tight_layout(rect=(0,.12,1,.94))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    for suffix in ['.png','.svg','.pdf']:
        fig.savefig(a.output.with_suffix(suffix),dpi=220,bbox_inches='tight')


if __name__=='__main__': main()
