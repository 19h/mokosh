#!/usr/bin/env python3
"""Compare every cell, segment, presynaptic endpoint and permanence bit live."""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile

parser = argparse.ArgumentParser()
parser.add_argument('--reference-root', type=Path, required=True)
parser.add_argument('--reference-build', type=Path, required=True)
args = parser.parse_args()
repo = Path(__file__).resolve().parents[2]
with tempfile.TemporaryDirectory(prefix='mokosh-full-graph-') as tmp:
    executable = Path(tmp) / 'oracle'
    fixture = Path(tmp) / 'full-graph.txt'
    command = ['c++', '-std=c++17', '-O2',
        '-I' + str(args.reference_root / 'src'),
        '-I' + str(args.reference_build / 'cereal-src/include'),
        '-I' + str(args.reference_build / 'eigen-src'),
        '-I' + str(args.reference_root / 'external/common/include'),
        str(repo / 'tests/reference/tm_campaign.cpp'),
        str(args.reference_build / 'src/libhtm_core.a'), '-o', str(executable)]
    subprocess.run(command, check=True)
    env = dict(os.environ, MOKOSH_FULL_GRAPH='1', MOKOSH_CAMPAIGN_FIXTURE=str(fixture))
    with fixture.open('w') as out:
        subprocess.run([str(executable)], env=env, stdout=out, check=True)
    subprocess.run(['cargo', 'test', '--all-features', '--test', 'tm_campaign', '--locked'],
                   cwd=repo, env=env, check=True)
