#!/usr/bin/env python3
"""Build and verify the EXP004 fold-0 raw cache on SERVER-TRAINING.

The CLI binds frozen manifests, source files and independent fixture evidence before
constructing the canonical reader itself. It writes a new run directory, with the raw
cache in ``cache/`` and a completion record in ``result.json``. A cache whose parent
contains ``failure.json`` or lacks the completion record is not a completed CLI run.

No training, sampler, dev weighting, normalization fitting or raw modification occurs.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import site
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PARENT = Path('/mnt/data-ssd/keti_data/factory_safety/exp004_raw_cache')
CLI_SCHEMA = 'exp004-raw-cache-cli-fixtures-v1'
RESULT_SCHEMA = 'exp004-raw-cache-build-v1'
MODULE_FILES = (
    'src/data/anomaly_raw_cache.py', 'src/data/synthetic_anomaly.py',
    'src/fit_anomaly_controls.py', 'src/models/anomaly_baselines.py',
    'src/models/anomaly_detectors.py', 'src/models/v2_plus.py',
    'src/train_anomaly_pilot.py', 'tests/test_anomaly_detectors.py',
    'tests/test_anomaly_raw_cache.py',
)
SOURCE_FILES = (*MODULE_FILES, 'src/build_anomaly_raw_cache.py',
                'tests/test_build_anomaly_raw_cache.py')
BINDING_FIELDS = ('fold', 'source_manifest_file_sha256', 'source_manifest_payload_sha256',
                  'split_manifest_file_sha256', 'split_manifest_payload_sha256',
                  'roles_sha256', 'count', 'role_counts', 'base_order_sha256',
                  'role_order_sha256', 'window_order_sha256', 'declared_dtypes',
                  'array_payload_bytes')


def _leaf(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


rc = _leaf('_exp004_build_raw_cache', 'src/data/anomaly_raw_cache.py')
controls = _leaf('_exp004_build_raw_controls', 'src/fit_anomaly_controls.py')
sa = rc.sa


def source_hashes():
    return {rel: hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() for rel in SOURCE_FILES}


def _signed_digest(record, label):
    payload = {k: v for k, v in record.items() if k != 'sha256'}
    digest = sa.digest_json(payload)
    if record.get('sha256') != digest:
        raise ValueError(f'{label} payload hash mismatch')
    return digest


def _load_pinned(path, expected, label):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f'{label} file hash mismatch')
    return json.loads(raw)


def _write_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def server_environment():
    if (platform.machine() != 'x86_64' or Path('/etc/nv_tegra_release').exists()
            or Path(sys.prefix).resolve() != (Path.home() / 'venvs/factory_training').resolve()
            or os.environ.get('PYTHONNOUSERSITE') != '1' or site.ENABLE_USER_SITE is not False):
        raise ValueError('actual cache creation requires audited SERVER-TRAINING venv')
    return {'environment_profile': 'SERVER-TRAINING', 'hostname': platform.node(),
            'python_version': platform.python_version(), 'python_executable': sys.executable,
            'numpy_version': np.__version__, 'jetpack_l4t': None,
            'sensor_driver_versions': {}, 'user_site_enabled': site.ENABLE_USER_SITE}


def _check_output(out, data_root):
    out, data_root = out.resolve(), data_root.resolve()
    parent = OUTPUT_PARENT.resolve()
    if out.parent != parent:
        raise ValueError('output must be a new direct child of the SSD raw-cache parent')
    if out.exists():
        raise ValueError('output already exists; never overwrite or resume')
    if out == data_root or data_root in out.parents or out in data_root.parents:
        raise ValueError('output must be separate from raw data')
    return out


def _fixture_ok(summary, count):
    return (summary.get('gate') == 'PASS' and summary.get('tests_run') == count
            and summary.get('passed') == count
            and all(summary.get(key) == 0 for key in ('skipped', 'failures', 'errors')))


def validate_inputs(args, failure_context=None):
    """Read only metadata, binding both evidence files to the exact current code."""
    code = source_hashes()
    if failure_context is not None:
        failure_context['source_files'] = code
    review = _load_pinned(args.review, args.review_sha256, 'module review')
    review_sha = _signed_digest(review, 'module review')
    if (review.get('gate') != 'PASS_FIXTURE_AND_RC01' or review.get('RC-01') != 'CLOSED'
            or not _fixture_ok(review['server']['summary'], 50)):
        raise ValueError('raw-cache module lacks independent server fixture PASS')
    if {key: review['source_files'].get(key) for key in MODULE_FILES} != {key: code[key] for key in MODULE_FILES}:
        raise ValueError('module review source hashes do not match executing code')
    evidence = _load_pinned(args.cli_fixtures, args.cli_fixtures_sha256, 'CLI fixtures')
    _signed_digest(evidence, 'CLI fixtures')
    expected_tests = sum(isinstance(n, ast.FunctionDef) and n.name.startswith('test_')
                         for n in ast.walk(ast.parse((ROOT / SOURCE_FILES[-1]).read_text())))
    if (evidence.get('schema') != CLI_SCHEMA or expected_tests < 1
            or not _fixture_ok(evidence, expected_tests)
            or evidence.get('environment_profile') != 'SERVER-TRAINING'
            or evidence.get('source_files') != code):
        raise ValueError('CLI fixture evidence is not a complete source-bound server PASS')
    if (not isinstance(evidence.get('git_commit_sha'), str)
            or len(evidence['git_commit_sha']) != 40
            or any(c not in '0123456789abcdef' for c in evidence['git_commit_sha'])):
        raise ValueError('CLI fixture evidence lacks a source revision')
    if failure_context is not None:
        failure_context.update(git_commit_sha=evidence['git_commit_sha'],
                               git_revision_source='validated CLI fixture evidence',
                               cli_fixture_file_sha256=args.cli_fixtures_sha256)

    binding = {key: review['future_real_cache_binding'][key] for key in BINDING_FIELDS}
    split = _load_pinned(args.prepared / 'split_manifest.json',
                         binding['split_manifest_file_sha256'], 'split manifest')
    split_sha = _signed_digest(split, 'split manifest')
    if (split_sha != binding['split_manifest_payload_sha256']
            or split.get('data_gate') != 'PASS' or split.get('protocol_id') != sa.PROTOCOL):
        raise ValueError('split manifest is not the frozen protocol PASS')
    raw = _load_pinned(args.prepared / 'raw_manifest.json',
                       binding['source_manifest_file_sha256'], 'raw manifest')
    raw_sha = sa.digest_json(raw)
    if raw_sha != binding['source_manifest_payload_sha256'] or raw_sha != split['source_manifest_sha256']:
        raise ValueError('raw manifest payload hash mismatch')
    if len({r['path'] for r in raw}) != len(raw):
        raise ValueError('duplicate raw manifest paths')
    if args.fold != 0 or binding.get('fold') != args.fold:
        raise ValueError('this verified cache stage is fold 0 only')
    root = args.data_root.resolve()
    if root != Path(split['provenance']['data_root']).resolve():
        raise ValueError('data root differs from frozen split provenance')
    fold = split['folds'][str(args.fold)]
    roles, normalizer = fold['roles'], fold['normalizer']
    if sa.digest_json(roles) != binding['roles_sha256']:
        raise ValueError('fold roles differ from reviewed role map')
    selected = rc.select_windows(split['windows'], roles)
    rc.check_windows(selected, roles)
    identities = rc.plain_json([rc.window_identity(w, roles[w['device']]) for w in selected], 'window identity')
    observed = {'count': len(selected), 'role_counts': dict(Counter(roles[w['device']] for w in selected)),
                'base_order_sha256': sa.digest_json([w['base_window_id'] for w in selected]),
                'role_order_sha256': sa.digest_json([roles[w['device']] for w in selected]),
                'window_order_sha256': sa.digest_json(identities)}
    if any(value != binding.get(key) for key, value in observed.items()):
        raise ValueError('selected windows differ from reviewed fit/dev counts or order')
    normalizer_sha = _signed_digest(normalizer, 'normalizer')
    fit_ids = sorted({i for w in selected if roles[w['device']] == 'fit' for i in w['raw_base_ids']})
    if normalizer.get('fit_raw_ids') != fit_ids or normalizer.get('fit_ids_sha256') != sa.digest_json(fit_ids):
        raise ValueError('normalizer fit provenance differs from selected fit windows')
    for key, shape in (('sensor_mean', (8,)), ('sensor_scale', (8,)),
                       ('thermal_mean', ()), ('thermal_scale', ())):
        a = np.asarray(normalizer[key])
        if a.shape != shape or not np.isfinite(a).all() or ('scale' in key and (a <= 0).any()):
            raise ValueError('invalid frozen normalizer: ' + key)
    layout = rc.declared_layout(binding['declared_dtypes'])
    if rc.required_bytes(len(selected), layout) != binding['array_payload_bytes']:
        raise ValueError('declared dtype/size differs from reviewed cache binding')
    ledger = {r['path']: r['sha256'] for r in raw}
    for w in selected:
        rc.raw_sources(w, root, ledger)
    groups = {(roles[w['device']], w['device']) for w in selected}
    return {'split': split, 'raw': raw, 'windows': selected, 'roles': roles,
            'normalizer': normalizer, 'binding': binding, 'source_files': code,
            'normalizer_sha256': normalizer_sha, 'review_payload_sha256': review_sha,
            'git_commit_sha': evidence['git_commit_sha'], 'identities': identities,
            'cost_plan': {'canonical_build_window_reads': len(selected),
                          'canonical_verification_window_reads': len(selected),
                          'normal_tensor_pairs': len(selected) * 2,
                          'synthetic_sample_windows': len(groups),
                          'synthetic_paired_attempts': len(groups) * 2 * 3 * 3,
                          'synthetic_scope': 'first manifest-order window per fit/dev device; CT1/CT2, three families, strengths1/2/4, seed42',
                          'synthetic_exhaustive': False, 'wall_seconds_estimate': None}}


def _same_array(a, b, label):
    if a.dtype.str != b.dtype.str or a.shape != b.shape or a.tobytes() != b.tobytes():
        raise ValueError(f'{label} byte parity failed')


def _final_pair(sensor, thermal, normalizer, ct):
    result = tuple(np.ascontiguousarray(a, dtype=np.float32)
                   for a in controls.normalized_pair(sensor, thermal, normalizer, ct))
    if not all(np.isfinite(a).all() for a in result):
        raise ValueError('nonfinite final model input')
    return result


def verify_parity(cache, inputs, canonical):
    """Full raw/Normal parity; explicitly sampled synthetic parity, not performance."""
    read = cache.reader(verify_each_read=True)
    normal, attempts, accepted = 0, 0, 0
    groups, witness = set(), hashlib.sha256()
    for window in inputs['windows']:
        raw, stored = canonical(window), read(window)
        for name, a, b in zip(rc.ARRAYS, raw, stored):
            _same_array(a, b, name)
        for ct in ('CT1', 'CT2'):
            original = _final_pair(*raw[:2], inputs['normalizer'], ct)
            copied = _final_pair(*stored[:2], inputs['normalizer'], ct)
            for name, a, b in zip(('sensor', 'thermal'), original, copied):
                _same_array(a, b, 'Normal final ' + name)
                witness.update(a.tobytes())
            normal += 1
        role = inputs['roles'][window['device']]
        group = (role, window['device'])
        if group not in groups:
            groups.add(group)
            for ct in ('CT1', 'CT2'):
                for family in ('sensor-only', 'thermal-only', 'coupled'):
                    for strength in (1, 2, 4):
                        kwargs = dict(normalizer=inputs['normalizer'], protocol=sa.PROTOCOL,
                                      fold=0, role=role, base_window_id=window['base_window_id'],
                                      family=family, strength=strength, generator_seed=42, ct_channel=ct)
                        a = sa.inject_blind(*raw[:2], tick_labels=raw[2], **kwargs)
                        b = sa.inject_blind(*stored[:2], tick_labels=stored[2], **kwargs)
                        if sa.digest_json(a[2]) != sa.digest_json(b[2]):
                            raise ValueError('injection metadata parity failed')
                        for x, y in zip(a[:2], b[:2]):
                            _same_array(x, y, 'injected raw')
                        for x, y in zip(_final_pair(*a[:2], inputs['normalizer'], ct),
                                        _final_pair(*b[:2], inputs['normalizer'], ct)):
                            _same_array(x, y, 'injected final')
                            witness.update(x.tobytes())
                        witness.update(sa.digest_json(a[2]).encode())
                        attempts += 1
                        accepted += int(a[2]['accepted'])
    plan = inputs['cost_plan']
    if normal != plan['normal_tensor_pairs'] or attempts != plan['synthetic_paired_attempts']:
        raise ValueError('parity coverage differs from declared scope')
    return {'gate': 'PASS', 'raw_windows': len(inputs['windows']),
            'normal_tensor_pairs': normal, 'synthetic_sample_windows': len(groups),
            'synthetic_paired_attempts': attempts, 'synthetic_accepted': accepted,
            'synthetic_rejected': attempts - accepted, 'synthetic_exhaustive': False,
            'tensor_and_event_stream_sha256': witness.hexdigest()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('prepared', 'data-root', 'out', 'review', 'cli-fixtures'):
        parser.add_argument('--' + flag, type=Path, required=True)
    for flag in ('review-sha256', 'cli-fixtures-sha256'):
        parser.add_argument('--' + flag, required=True)
    parser.add_argument('--fold', type=int, choices=[0], default=0)
    args = parser.parse_args(argv)
    environment = server_environment()
    out = _check_output(args.out, args.data_root)
    out.mkdir(parents=True, exist_ok=False)
    stage, started, timings = 'inputs', time.perf_counter(), {}
    failure_context = {**environment, 'git_commit_sha': None, 'git_dirty': True,
                       'git_revision_source': 'not yet verified; input gate incomplete',
                       'source_files': {}, 'module_review_file_sha256_requested': args.review_sha256,
                       'cli_fixture_file_sha256_requested': args.cli_fixtures_sha256}
    try:
        before = time.perf_counter()
        inputs = validate_inputs(args, failure_context)
        timings['input_validation'] = time.perf_counter() - before
        provenance = {**environment, 'timestamp': datetime.now(timezone.utc).isoformat(),
                      'git_commit_sha': inputs['git_commit_sha'], 'git_dirty': True,
                      'git_dirty_basis': 'conservative diagnostic declaration; source snapshot hashes are authoritative',
                      'source_files': inputs['source_files'], 'fold': 0,
                      'binding': inputs['binding'], 'normalizer_sha256': inputs['normalizer_sha256'],
                      'module_review_file_sha256': args.review_sha256,
                      'module_review_payload_sha256': inputs['review_payload_sha256'],
                      'cli_fixture_file_sha256': args.cli_fixtures_sha256,
                      'cost_plan': inputs['cost_plan'], 'optimizer_steps': 0,
                      'paper_result_eligible': False}
        stage = 'provenance'
        _write_json(out / 'provenance.json', provenance)
        provenance_sha = hashlib.sha256((out / 'provenance.json').read_bytes()).hexdigest()
        stage, before = 'cache_build', time.perf_counter()
        raw_reader = controls.make_reader(args.data_root.resolve(), inputs['raw'])
        identities = {w['base_window_id']: w for w in inputs['identities']}
        reads, role_reads = Counter(), Counter()

        def canonical(window):
            role = inputs['roles'].get(window.get('device'))
            identity = rc.plain_json(rc.window_identity(window, role), 'window identity')
            if role not in rc.ALLOWED_ROLES or identity != identities.get(window['base_window_id']):
                raise ValueError('canonical reader refused a window outside the frozen fit/dev view')
            arrays = raw_reader(window)
            reads[window['base_window_id']] += 1
            role_reads[role] += 1
            return arrays
        cache = rc.RawNormalCache.build(out / 'cache', inputs['windows'], inputs['roles'], canonical,
                                       data_root=args.data_root, raw_manifest=inputs['raw'],
                                       source_manifest_sha256=inputs['split']['source_manifest_sha256'],
                                       dtypes=inputs['binding']['declared_dtypes'], provenance=provenance)
        timings['cache_build'] = time.perf_counter() - before
        stage, before = 'attach', time.perf_counter()
        cache = rc.RawNormalCache.attach(cache.out).open()
        timings['attach'] = time.perf_counter() - before
        stage, before = 'parity', time.perf_counter()
        parity = verify_parity(cache, inputs, canonical)
        timings['parity'] = time.perf_counter() - before
        cache._data = None
        stage = 'final_checks'
        # Re-read the pinned input files and code after the run; never bless a moving input.
        final_inputs = validate_inputs(args)
        if final_inputs['source_files'] != inputs['source_files']:
            raise ValueError('source changed during execution')
        if set(reads) != set(identities) or any(n != 2 for n in reads.values()):
            raise ValueError('canonical read counts differ from the two-pass plan')
        record = {'schema': RESULT_SCHEMA, 'gate': 'PASS', 'fold': 0,
                  'cache_manifest_sha256': cache.manifest['sha256'], 'binding': inputs['binding'],
                  'provenance_file_sha256': provenance_sha,
                  'source_files': inputs['source_files'], 'parity': parity,
                  'phase_wall_seconds': timings,
                  'wall_until_final_record_seconds': time.perf_counter() - started,
                  'canonical_window_reads': sum(reads.values()), 'canonical_role_reads': dict(role_reads),
                  'raw_roles_read': sorted(role_reads), 'optimizer_steps': 0,
                  'paper_result_eligible': False, 'full_P0_gate': 'NOT_EVALUATED',
                  'performance_measured': False}
        record['sha256'] = sa.digest_json(record)
        stage = 'final_record'
        _write_json(out / 'result.json', record)
        print(json.dumps({'gate': 'PASS', 'out': str(out), 'parity': parity}), flush=True)
        return 0
    except BaseException as exc:
        failure = {**failure_context, 'stage': stage, 'exception': type(exc).__name__, 'message': str(exc),
                   'timestamp': datetime.now(timezone.utc).isoformat(),
                   'partial_files_preserved': [],
                   'run_usable': False}
        try:
            failure['partial_files_preserved'] = sorted(
                str(p.relative_to(out)) for p in out.rglob('*') if p.is_file())
        except BaseException as listing_error:
            failure['partial_file_listing_error'] = {
                'exception': type(listing_error).__name__, 'message': str(listing_error)}
        try:
            _write_json(out / 'failure.json', failure)
        except BaseException as recording_error:
            print(json.dumps({'failure_record_write_error': repr(recording_error),
                              'original_failure': failure}), file=sys.stderr, flush=True)
        raise


if __name__ == '__main__':
    raise SystemExit(main())
