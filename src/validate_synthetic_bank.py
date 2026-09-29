#!/usr/bin/env python3
"""Audit blind fit/dev banks without model fitting, calibration or heldout reads."""
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

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('_bank_controls', ROOT/'src/fit_anomaly_controls.py')
controls = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = controls
spec.loader.exec_module(controls)
sa = controls.sa


def validate_bank(windows, roles, normalizer, reader, *, fold, seed=42, ct_channel='CT1',
                  emit=None, progress=None):
    norm = dict(normalizer)
    claimed = norm.pop('sha256')
    if sa.digest_json(norm) != claimed:
        raise ValueError('normalizer hash mismatch')
    fit_ids = sorted({i for w in windows if roles[w['device']]=='fit' and w['hard_valid'] and w['pure_Normal']
                      for i in w['raw_base_ids']})
    if fit_ids != normalizer['fit_raw_ids'] or sa.digest_json(fit_ids) != normalizer['fit_ids_sha256']:
        raise ValueError('normalizer fit provenance mismatch')
    counts, composition, reasons, base_qc = Counter(), Counter(), Counter(), Counter()
    manifests, role_counts, seen, coverage = hashlib.sha256(), Counter(), set(), {}
    for role in ['fit','dev']:
        selected = [w for w in windows if roles[w['device']]==role and w['hard_valid'] and w['pure_Normal']]
        expected_devices = sorted(d for d,r in roles.items() if r==role)
        if not expected_devices or {w['device'] for w in selected} != set(expected_devices):
            raise ValueError('missing eligible pure-Normal equipment in '+role)
        for i,w in enumerate(selected):
            s,t,labels = reader(w)
            original_s, original_t = s.copy(), t.copy()
            base_qc[(role, bool(w['base_QC_flags']))] += 1
            for family in ['sensor-only','thermal-only','coupled']:
                for strength in [1,2,4]:
                    _,_,event = sa.inject_blind(s,t,tick_labels=labels,normalizer=normalizer,
                                                protocol=sa.PROTOCOL,fold=fold,role=role,
                                                base_window_id=w['base_window_id'],family=family,
                                                strength=strength,generator_seed=seed,ct_channel=ct_channel)
                    if event['base_QC_flags'] != w['base_QC_flags']:
                        raise ValueError('base QC flags disagree with preparation')
                    if event['event_instance_id'] in seen:
                        raise ValueError('duplicate synthetic event ID')
                    seen.add(event['event_instance_id'])
                    if event['accepted'] and (event['sensor_changed'],event['thermal_changed']) != {
                            'sensor-only':(True,False),'thermal-only':(False,True),'coupled':(True,True)}[family]:
                        raise ValueError('accepted event violates modality-change contract')
                    record = dict(base_window_id=w['base_window_id'],device=w['device'],role=role,**event)
                    payload = json.dumps(record,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)+'\n'
                    manifests.update(payload.encode())
                    if emit: emit(payload)
                    counts[(role,w['device'],family,strength,event['accepted'])] += 1
                    p = event['parameters']
                    composition[(role,family,strength,'|'.join(p['sensor_channels']),p['operator'],event['accepted'])] += 1
                    reasons.update((role,r) for r in event['rejection_reasons'])
            if not sa.np.array_equal(s,original_s) or not sa.np.array_equal(t,original_t):
                raise ValueError('injection mutated base input arrays')
            role_counts[role] += 1
            if progress and (i+1)%100==0: progress(role,i+1,len(selected))
        missing = [[d,f,s] for d in expected_devices for f in ['sensor-only','thermal-only','coupled']
                   for s in [1,2,4] if not counts[(role,d,f,s,True)]]
        coverage[role] = {'expected_cells':len(expected_devices)*9,'missing_accepted_cells':missing,
                          'bank_gate':'FAIL' if missing else 'PASS'}
    accepted = sum(v for k,v in counts.items() if k[-1])
    result = {'protocol_id':sa.PROTOCOL,'fold':fold,'seed':seed,'ct_channel':ct_channel,
              'normalizer_sha256':claimed,'event_manifest_sha256':manifests.hexdigest(),
              'normal_base_windows':dict(role_counts),'attempted':len(seen),'accepted':accepted,
              'rejected':len(seen)-accepted,'coverage':coverage,
              'bank_gate':'PASS' if all(v['bank_gate']=='PASS' for v in coverage.values()) else 'FAIL',
              'base_QC_accounting':[dict(role=k[0],soft_QC=k[1],windows=v) for k,v in sorted(base_qc.items())],
              'cells':[dict(role=k[0],device=k[1],family=k[2],strength=k[3],accepted=k[4],count=v)
                       for k,v in sorted(counts.items())],
              'composition':[dict(role=k[0],family=k[1],strength=k[2],sensor_channels=k[3],operator=k[4],accepted=k[5],count=v)
                             for k,v in sorted(composition.items())],
              'rejection_reasons':[dict(role=k[0],reason=k[1],count=v) for k,v in sorted(reasons.items())],
              'heldout_read':False,'calibration_read':False,'gradient_training_runs':0,
              'full_P0_gate':'NOT_EVALUATED'}
    result['sha256'] = sa.digest_json(result)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--fold',type=int,choices=range(4),required=True)
    p.add_argument('--seed',type=int,choices=[42,123,456],default=42)
    p.add_argument('--ct-channel',choices=['CT1','CT2'],default='CT1')
    args=p.parse_args(argv)
    if (platform.machine()!='x86_64' or Path('/etc/nv_tegra_release').exists()
            or Path(sys.prefix).resolve()!=(Path.home()/'venvs/factory_training').resolve()
            or os.environ.get('PYTHONNOUSERSITE')!='1'):
        p.error('actual bank validation requires audited SERVER-TRAINING venv')
    manifest=json.loads((args.prepared/'split_manifest.json').read_text());sha=manifest.pop('sha256')
    if sa.digest_json(manifest)!=sha or manifest['data_gate']!='PASS' or manifest['protocol_id']!=sa.PROTOCOL:
        raise ValueError('invalid prepared manifest')
    raw=json.loads((args.prepared/'raw_manifest.json').read_text())
    if sa.digest_json(raw)!=manifest['source_manifest_sha256']:
        raise ValueError('raw manifest hash mismatch')
    data_root=Path(manifest['provenance']['data_root']).resolve();out=args.out.resolve()
    if out.exists() or out==data_root or data_root in out.parents or out in data_root.parents:
        p.error('output must be new and separate from raw data')
    out.mkdir(parents=True,exist_ok=False)
    provenance={'timestamp':datetime.now(timezone.utc).isoformat(),'hostname':platform.node(),
                'environment_profile':'SERVER-TRAINING','python_version':platform.python_version(),
                'numpy_version':sa.np.__version__,'git_dirty':True,
                'git_dirty_basis':'conservative_declaration_uncommitted_diagnostic_packet_not_git_status_measurement',
                'git_commit_sha':manifest['provenance']['git_commit_sha'],'prepared_sha256':sha,
                'code_sha256':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in
                               [Path(__file__),ROOT/'src/fit_anomaly_controls.py',ROOT/'src/data/synthetic_anomaly.py']},
                'purpose':'P0_fit_dev_bank_diagnostic_only','gradient_training_runs':0}
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    start=time.monotonic()
    try:
        fold=manifest['folds'][str(args.fold)]
        with (out/'events.jsonl').open('x') as events:
            result=validate_bank(manifest['windows'],fold['roles'],fold['normalizer'],controls.make_reader(data_root,raw),
                                 fold=args.fold,seed=args.seed,ct_channel=args.ct_channel,emit=events.write,
                                 progress=lambda role,n,total:print(f'{role}: {n}/{total}',flush=True))
        (out/'bank.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        print(json.dumps({k:result[k] for k in ['bank_gate','attempted','accepted','rejected','coverage']}),flush=True)
        print(json.dumps({'elapsed_seconds':time.monotonic()-start}),flush=True)
        return 0 if result['bank_gate']=='PASS' else 2
    except Exception as exc:
        (out/'failure.json').write_text(json.dumps({'exception':type(exc).__name__,'message':str(exc)})+'\n')
        raise


if __name__=='__main__':
    raise SystemExit(main())
