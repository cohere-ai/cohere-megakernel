#!/usr/bin/env python3
"""Commit-based source transport; diagnostics remain on the cluster."""
import argparse
from pathlib import Path
import re
import shlex
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]


def call(args, **kwargs):
    return subprocess.run(args, check=True, **kwargs)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host', default='ln301')
    p.add_argument('--remote-dir', default='hejy/cohere-megakernel',
                   help='path relative to remote home')
    p.add_argument('--job', type=int, help='current existing Slurm allocation')
    p.add_argument('action', choices=['push', 'status', 'run', 'pull-figures'])
    p.add_argument('command', nargs=argparse.REMAINDER)
    a = p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_./-]+', a.remote_dir) or '..' in Path(a.remote_dir).parts or a.remote_dir.startswith('/'):
        p.error('remote-dir must be a safe relative path below remote home')
    if a.host.startswith('-'):
        p.error('invalid host')
    remote = '"$HOME"/' + shlex.quote(a.remote_dir)
    ssh = ['ssh', '-o', 'BatchMode=yes', a.host]
    if a.action == 'push':
        dirty = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=normal'], cwd=ROOT, text=True)
        if dirty:
            p.error('commit source changes before push; transport never silently drops dirty files')
        branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip()
        if not branch:
            p.error('push requires a named branch')
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        with tempfile.TemporaryDirectory(prefix='cohere-sync-') as d:
            bundle = Path(d) / 'source.bundle'
            call(['git', 'bundle', 'create', str(bundle), 'HEAD'], cwd=ROOT)
            call(ssh + [f'mkdir -p {remote}/.transport'])
            call(['rsync', '-a', str(bundle), f'{a.host}:{a.remote_dir}/.transport/source.bundle'])
        qbranch = shlex.quote(branch)
        script = f'''set -eu
cd {remote}
if [ ! -d .git ]; then git init -q; fi
git bundle verify .transport/source.bundle
git fetch .transport/source.bundle HEAD
if git show-ref --verify --quiet refs/heads/{qbranch}; then
  git switch {qbranch}
  git merge --ff-only FETCH_HEAD
else
  git switch -c {qbranch} FETCH_HEAD
fi
test "$(git rev-parse HEAD)" = {commit}
git status --short
'''
        call(ssh + [script])
        print(f'Synchronized {commit}; dependency archives are installed separately.')
    elif a.action == 'status':
        if not a.job:
            p.error('--job is required')
        call(ssh + [f'squeue -j {a.job}; cd {remote} && git log -1 --oneline && git status --short'])
    elif a.action == 'run':
        if not a.job or a.job <= 0:
            p.error('run requires a positive --job')
        command = a.command[1:] if a.command[:1] == ['--'] else a.command
        if not command:
            p.error('supply a command after run --')
        inner = f'cd {remote} && source research/scripts/server_env.sh && exec {shlex.join(command)}'
        call(ssh + [f'srun --jobid={a.job} --overlap -N1 -n1 bash -lc {shlex.quote(inner)}'])
    else:
        dest = ROOT / 'research/figures'
        dest.mkdir(exist_ok=True)
        call(['rsync', '-av', '--include=*/', '--include=*.png', '--include=*.pdf',
              '--include=*.svg', '--exclude=*',
              f'{a.host}:{a.remote_dir}/research/figures/', str(dest) + '/'])


if __name__ == '__main__':
    main()
