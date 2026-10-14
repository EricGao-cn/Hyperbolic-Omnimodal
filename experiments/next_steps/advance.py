"""Resume the ordered gates, refusing incomplete selection or missing evidence.

Run only after the separate smoke suite has completed. This controller does not
decide a hierarchy gate from missing measurements and does not retry failed runs.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import suite as workflow


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', required=True)
    parser.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    parser.add_argument('--port', type=int, default=29670)
    parser.add_argument('--hierarchy')
    parser.add_argument('--hierarchy-std', type=float)
    args = parser.parse_args()
    suite = Path(args.suite).resolve()
    protocol = workflow.read_json(suite / 'protocol.json')
    logs = Path(protocol['resources']['logs_dir'])
    logs.mkdir(parents=True, exist_ok=True)
    lock = (logs / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status_path = logs / 'controller_status.json'

    def status(phase, **extra):
        workflow.write_json(status_path, dict(pid=os.getpid(), phase=phase, updated=time.time(), **extra))

    def check_complete(target, phases=None):
        selected = [t for t in workflow.load_trials(target) if not phases or t['phase'] in phases]
        missing = [f"{t['name']}/{stage}" for t in selected for stage in workflow.stages_for(t)
                   if not workflow.done(target, t['name'], stage)]
        if not selected or missing:
            raise RuntimeError(f'incomplete stages in {target}: {missing or "no trials"}')

    command = [sys.executable, '-u', str(Path(__file__).with_name('suite.py')), '--suite', str(suite)]

    def call(*parts):
        print(time.strftime('%Y-%m-%dT%H:%M:%S'), ' '.join(parts), flush=True)
        subprocess.run(command + list(parts), check=True)

    def run(*phases):
        status('running', trial_phases=phases)
        call('run', '--gpus', args.gpus, '--port', str(args.port), '--phase', *phases, '--stop-on-failure')
        check_complete(suite, phases)
        call('summarize')

    try:
        check_complete(suite / 'smoke')
        call('add', 'gate_a')
        call('add', 'pmrl_screen')
        run('gate_a', 'gate_a_screen', 'pmrl_screen')
        call('add', 'gate_a_promote')
        run('gate_a')
        # Contrast selection must happen AFTER both promotions finish.
        call('add', 'gate_a_contrast')
        run('gate_a')
        call('gate-a')
        gate_a = workflow.read_json(suite / 'gate_a.json')
        if not gate_a['passed']:
            status('gate_a_failed_requires_fallback_proxy',
                   reason='PLAN requires K score with M Perron proxy; the supplied K fallback uses tangent_mean instead. No silent substitution.')
            return
        call('add', 'gate_b')
        run('gate_b')

        gate_a = workflow.read_json(suite / 'gate_a.json')
        base = workflow.holdout_view(suite, gate_a['winner'])
        eligible, missing_geometry = [], []
        for t in workflow.load_trials(suite):
            if t['phase'] != 'gate_b':
                continue
            geometry = workflow.train_geometry(suite, t['name'])
            cones = [v for k, v in geometry.items() if k.startswith('cone_satisfied_')]
            if not cones:
                missing_geometry.append(t['name'])
            held = workflow.holdout_view(suite, t['name'])
            if (cones and sum(cones) / len(cones) >= workflow.GATE_B['cone']
                    and held['top50'] - base['top50'] >= workflow.GATE_B['top50']):
                eligible.append(t['name'])
        if missing_geometry:
            status('awaiting_cone_diagnostics', trials=missing_geometry)
            return
        if eligible and not args.hierarchy:
            status('awaiting_hierarchy_metrics', eligible=eligible,
                   reason='Cone/retrieval conditions pass; hierarchy evidence is required to decide Gate B.')
            return
        if args.hierarchy and args.hierarchy_std is None:
            raise ValueError('--hierarchy-std must be the measured seed std when supplying hierarchy results')
        call('gate-b', *(['--hierarchy', args.hierarchy, '--hierarchy-std', str(args.hierarchy_std)]
                        if args.hierarchy else []))
        call('add', 'main')
        run('main')
        status('implemented_main_trials_complete',
               remaining=['proxy diagnostics and hierarchy evaluation',
                          'external GRAM/HyperGRAM protocol-matched baselines',
                          'k=4 implementation and width sweep'])
    except BaseException as error:
        status('stopped', error=repr(error))
        raise


if __name__ == '__main__':
    main()
