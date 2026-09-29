#!/usr/bin/env python3
"""Server-only diagnostics of frozen EXP004 calibration scores, never heldout scores."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def leaf(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def digest(payload):
    return hashlib.sha256(json.dumps(payload,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def read_verified(path):
    payload = json.loads(path.read_text())
    claimed = payload.pop('sha256')
    if digest(payload) != claimed:
        raise ValueError(f'payload hash mismatch: {path}')
    return payload, claimed


def write_result(path, payload):
    payload = dict(payload)
    payload['sha256'] = digest(payload)
    path.write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def bank_diagnostics(normal, events, calibration, ac):
    """Describe existing acceptance and both threshold policies without modifying them."""
    accepted = [e for e in events if e['accepted']]
    weights, _ = ac.stratified_cell_weights([e['device'] for e in accepted], [e['family'] for e in accepted],
                                           [e['strength'] for e in accepted],
                                           expected_device_ids=sorted({r['device'] for r in normal}))
    by_base = {r['base_window_id']:r for r in normal}
    composition = Counter()
    for event in events:
        p = event['parameters']
        key = (event['family'], event['strength'], '|'.join(p.get('sensor_channels',[])),
               p.get('operator','none'), event['accepted'])
        composition[key] += 1
    models = {}
    for model, values in calibration.items():
        primary = values['primary']
        scores = np.asarray([e['scores'][model] for e in accepted])
        nq, syn = primary['NQ']['tau'], primary['SYN']['tau']
        if syn < nq:
            raise ValueError('frozen threshold ordering violated')
        models[model] = {'NQ':{'tau':nq,'FPR_cal_Normal':primary['NQ']['achieved_normal_fpr'],
                               'TPR_cal_synthetic':float(weights[scores > nq].sum())},
                         'SYN':{'tau':syn,'FPR_cal_Normal':primary['SYN']['achieved_normal_fpr'],
                                'TPR_cal_synthetic':float(weights[scores > syn].sum())},
                         'exact_score_invariance_by_family':{}}
        for family in ac.V3_FAMILIES:
            es = [e for e in accepted if e['family']==family]
            same = sum(e['scores'][model] == by_base[e['base_window_id']]['scores'][model] for e in es)
            models[model]['exact_score_invariance_by_family'][family] = {'unchanged':same,'accepted':len(es),
                                                                       'unweighted_fraction':same/len(es)}
    return {'scope':'frozen_calibration_diagnostic_not_heldout_performance', 'models':models,
            'composition':[dict(family=k[0],strength=k[1],sensor_channels=k[2],operator=k[3],accepted=k[4],count=v)
                           for k,v in sorted(composition.items())],
            'generator_or_model_retuned':False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared',type=Path,required=True)
    parser.add_argument('--controls',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--replicates',type=int,default=20)
    parser.add_argument('--blocks',type=int,nargs='+',default=[12])
    parser.add_argument('--alphas',type=float,nargs='+',default=[.01])
    parser.add_argument('--models',nargs='+',choices=['B-NTC','B-MD','B-THERM','B-LATE'],
                        default=['B-NTC','B-MD','B-THERM','B-LATE'])
    args = parser.parse_args(argv)
    if (platform.machine() != 'x86_64' or Path('/etc/nv_tegra_release').exists()
            or Path(sys.prefix).resolve() != (Path.home()/'venvs/factory_training').resolve()
            or os.environ.get('PYTHONNOUSERSITE') != '1'):
        parser.error('frozen-score data analysis requires audited SERVER-TRAINING venv')
    if not 1 <= args.replicates <= 2000 or any(b not in [6,12,30] for b in args.blocks):
        parser.error('replicates must be 1..2000 and blocks predeclared 6/12/30')
    if any(a not in [.005,.01,.02] for a in args.alphas):
        parser.error('alphas must be predeclared .005/.01/.02')
    manifest, manifest_hash = read_verified(args.prepared/'split_manifest.json')
    controls, controls_hash = read_verified(args.controls/'controls.json')
    if manifest['data_gate'] != 'PASS' or controls['calibration_gate'] != 'PASS' or controls['heldout_scored']:
        raise ValueError('diagnostic requires successful frozen calibration and preparation')
    roles = manifest['folds'][str(controls['fold'])]['roles']
    normal = controls['calibration_Normal']; events = controls['calibration_events']
    expected = {w['base_window_id']:w for w in manifest['windows']
                if roles[w['device']]=='calibration' and w['hard_valid'] and w['pure_Normal']}
    if len(normal) != len(expected) or {r['base_window_id'] for r in normal} != set(expected):
        raise ValueError('frozen Normal scores differ from the complete calibration view')
    for row in normal:
        if any(row[k] != expected[row['base_window_id']][k] for k in ['device','session_id','start_index']):
            raise ValueError('calibration temporal identity mismatch')
    if controls['normalizer_sha256'] != manifest['folds'][str(controls['fold'])]['normalizer']['sha256']:
        raise ValueError('frozen scores and manifest normalizers disagree')
    out = args.out.expanduser().resolve()
    raw = Path(manifest['provenance']['data_root']).resolve()
    if out.exists() or out == raw or raw in out.parents or out in raw.parents:
        parser.error('output must be a new directory separate from raw data')
    out.mkdir(parents=True,exist_ok=False)
    ac = leaf('_diagnostic_calibration','src/evaluation/anomaly_calibration.py')
    u = leaf('_diagnostic_uncertainty','src/evaluation/anomaly_uncertainty.py')
    provenance = {'timestamp':datetime.now(timezone.utc).isoformat(), 'hostname':platform.node(),
                  'python_version':platform.python_version(),'numpy_version':np.__version__,
                  'environment_profile':'SERVER-TRAINING','git_dirty':True,
                  'git_commit_sha':manifest['provenance']['git_commit_sha'],
                  'prepared_sha256':manifest_hash,'frozen_controls_sha256':controls_hash,
                  'code_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                 [Path(__file__),ROOT/'src/evaluation/anomaly_uncertainty.py',
                                  ROOT/'src/evaluation/anomaly_calibration.py']},
                  'full_P0_gate':'NOT_EVALUATED','heldout_scored':False,'gradient_training_runs':0}
    write_result(out/'provenance.json',provenance)
    try:
        write_result(out/'bank_diagnostic.json',bank_diagnostics(normal,events,controls['calibration'],ac))
        stride = manifest['stride_ticks']
        summary = {'replicates':args.replicates,'planned_replicate_count_met':args.replicates==2000,
                   'run_resolution':{str(b):u.run_resolution(normal,stride=stride,block_starts=b) for b in [6,12,30]},
                   'completed':[], 'full_P0_gate':'NOT_EVALUATED', 'retuned_primary_threshold':False}
        accepted = [e for e in events if e['accepted']]
        for model in args.models:
            calibration_args = ([r['scores'][model] for r in normal], [r['device'] for r in normal],
                                [e['scores'][model] for e in accepted], [e['device'] for e in accepted],
                                [e['family'] for e in accepted], [e['strength'] for e in accepted])
            # Verify the optimisation against both exhaustive search and the frozen primary.
            fast = ac.calibrate_syn(*calibration_args)
            exhaustive = ac.calibrate_syn(*calibration_args,candidate_pruning=False)
            keys = ['tau','achieved_normal_fpr','achieved_synthetic_tpr','objective_value','tied_at_optimum']
            frozen = controls['calibration'][model]['primary']['SYN']
            if any(fast[k] != exhaustive[k] or fast[k] != frozen[k] for k in keys):
                raise ValueError('calibration optimisation or frozen primary equivalence failed')
            nq = ac.calibrate_nq(*calibration_args[:2])
            if nq['tau'] != controls['calibration'][model]['primary']['NQ']['tau']:
                raise ValueError('frozen NQ equivalence failed')
            write_result(out/f'{model}_search_equivalence.json',{'status':'PASS','compared_keys':keys,
                                                               'fast':fast,'exhaustive':exhaustive,
                                                               'frozen_primary_unchanged':True})
            sensitivity = u.fixed_subset_sensitivity(normal,events,model,ac,stride=stride)
            write_result(out/f'{model}_subset_sensitivity.json',sensitivity)
            for block in args.blocks:
                for alpha in args.alphas:
                    start = time.monotonic()
                    result = u.conditional_intervals(normal,events,model,ac,stride=stride,
                                                     replicates=args.replicates,block_starts=block,alpha=alpha)
                    elapsed = time.monotonic()-start
                    filename = f'{model}_block{block}_alpha{alpha}.json'
                    write_result(out/filename,{**result,'elapsed_seconds':elapsed})
                    record = {'file':filename,'elapsed_seconds':elapsed,'replicates_valid':result['replicates_valid'],
                              'failed_replicates':len(result['failed_replicates']),
                              'percentile_interval':result['percentile_interval']}
                    summary['completed'].append(record)
                    write_result(out/'summary.json',summary)
                    print(json.dumps(record),flush=True)
        return 0
    except Exception as exc:
        write_result(out/'failure.json',{'exception':type(exc).__name__,'message':str(exc)})
        raise


if __name__ == '__main__':
    raise SystemExit(main())
