"""Actual main-path fixtures; raw files and evidence below are synthetic test data only."""
import argparse
import ast
from collections import Counter
import contextlib
import copy
from datetime import datetime
import hashlib
import importlib.util
import io
import json
import itertools
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def leaf(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


fixtures = leaf('_build_raw_fixtures', 'tests/test_anomaly_raw_cache.py')
cli = leaf('_build_raw_cli', 'src/build_anomaly_raw_cache.py')
ORIGINAL_ENVIRONMENT = cli.server_environment


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, record, signed=False):
    record = copy.deepcopy(record)
    if signed:
        record.pop('sha256', None)
        record['sha256'] = cli.sa.digest_json(record)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    return record


class BuildCliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.shared = tempfile.TemporaryDirectory()
        cls.fixture = fixtures.Fixture(cls.shared.name, np.float64, fixtures.F64)

    @classmethod
    def tearDownClass(cls):
        cls.shared.cleanup()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.prepared = self.root / 'prepared'
        self.prepared.mkdir()
        self.out = self.root / 'outputs/run'
        self.review_path = self.root / 'module-review.json'
        self.evidence_path = self.root / 'cli-evidence.json'
        roles, selected = fixtures.ROLES, self.fixture.selected
        normalizer = copy.deepcopy(fixtures.NORMALIZER)
        normalizer.pop('sha256')
        fit_ids = sorted({i for w in selected if roles[w['device']] == 'fit' for i in w['raw_base_ids']})
        normalizer.update(fit_raw_ids=fit_ids, fit_ids_sha256=cli.sa.digest_json(fit_ids))
        normalizer['sha256'] = cli.sa.digest_json(normalizer)
        self.raw = copy.deepcopy(self.fixture.ledger)
        write_json(self.prepared / 'raw_manifest.json', self.raw)
        self.split = write_json(self.prepared / 'split_manifest.json', {
            'protocol_id': cli.sa.PROTOCOL, 'data_gate': 'PASS',
            'source_manifest_sha256': cli.sa.digest_json(self.raw),
            'provenance': {'data_root': str(self.fixture.data_root)},
            'windows': self.fixture.windows,
            'folds': {'0': {'roles': roles, 'normalizer': normalizer}},
        }, signed=True)
        identities = [cli.rc.window_identity(w, roles[w['device']]) for w in selected]
        binding = {
            'fold': 0, 'source_manifest_file_sha256': file_sha(self.prepared / 'raw_manifest.json'),
            'source_manifest_payload_sha256': cli.sa.digest_json(self.raw),
            'split_manifest_file_sha256': file_sha(self.prepared / 'split_manifest.json'),
            'split_manifest_payload_sha256': self.split['sha256'],
            'roles_sha256': cli.sa.digest_json(roles), 'count': len(selected),
            'role_counts': {'fit': 4, 'dev': 2},
            'base_order_sha256': cli.sa.digest_json([w['base_window_id'] for w in selected]),
            'role_order_sha256': cli.sa.digest_json([roles[w['device']] for w in selected]),
            'window_order_sha256': cli.sa.digest_json(identities),
            'declared_dtypes': fixtures.F64,
            'array_payload_bytes': cli.rc.required_bytes(len(selected), fixtures.F64),
        }
        self.review = {'gate': 'PASS_FIXTURE_AND_RC01', 'RC-01': 'CLOSED',
                       'purpose': 'fabricated evidence fixture, not experimental evidence',
                       'server': {'summary': dict(gate='PASS', tests_run=50, passed=50,
                                                   skipped=0, failures=0, errors=0)},
                       'source_files': {k: v for k, v in cli.source_hashes().items() if k in cli.MODULE_FILES},
                       'future_real_cache_binding': binding}
        n = sum(isinstance(node, ast.FunctionDef) and node.name.startswith('test_')
                for node in ast.walk(ast.parse(Path(__file__).read_text())))
        self.evidence = dict(schema=cli.CLI_SCHEMA, gate='PASS', tests_run=n, passed=n,
                             skipped=0, failures=0, errors=0, environment_profile='SERVER-TRAINING',
                             source_files=cli.source_hashes(), git_commit_sha='a' * 40,
                             purpose='fabricated evidence fixture, not experimental evidence')
        self.seal()
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(mock.patch.object(cli, 'OUTPUT_PARENT', self.root / 'outputs'))
        self.stack.enter_context(mock.patch.object(cli, 'server_environment', return_value={
            'environment_profile': 'SYNTHETIC_FIXTURE', 'hostname': 'fixture',
            'python_version': sys.version.split()[0], 'numpy_version': np.__version__,
            'jetpack_l4t': None, 'sensor_driver_versions': {}, 'user_site_enabled': False}))

    def seal(self):
        self.review = write_json(self.review_path, self.review, signed=True)
        self.evidence = write_json(self.evidence_path, self.evidence, signed=True)

    def update_split(self):
        self.split = write_json(self.prepared / 'split_manifest.json', self.split, signed=True)
        self.review['future_real_cache_binding'].update(
            split_manifest_file_sha256=file_sha(self.prepared / 'split_manifest.json'),
            split_manifest_payload_sha256=self.split['sha256'])
        self.seal()

    def args(self):
        return argparse.Namespace(prepared=self.prepared, data_root=self.fixture.data_root,
                                  out=self.out, review=self.review_path, review_sha256=file_sha(self.review_path),
                                  cli_fixtures=self.evidence_path,
                                  cli_fixtures_sha256=file_sha(self.evidence_path), fold=0)

    def argv(self):
        args = self.args()
        return [part for key, value in vars(args).items() for part in ('--' + key.replace('_', '-'), str(value))]

    def run_main(self, argv=None):
        with contextlib.redirect_stdout(io.StringIO()):
            return cli.main(argv or self.argv())

    def assert_failed(self, stage, exception, match, patch=None):
        with contextlib.ExitStack() as stack:
            if patch:
                stack.enter_context(patch)
            try:
                with self.assertRaisesRegex(exception, match):
                    self.run_main()
            except BaseException as exc:
                if isinstance(exc, Exception):
                    raise
                self.fail(f'unexpected {type(exc).__name__} escaped main: {exc}')
        failure = json.loads((self.out / 'failure.json').read_text())
        self.assertEqual(failure['stage'], stage)
        self.assertFalse(failure['run_usable'])
        self.assertIsNotNone(datetime.fromisoformat(failure['timestamp']).utcoffset())
        self.assertFalse((self.out / 'result.json').exists())
        self.assertIsInstance(failure['partial_files_preserved'], list)
        if 'partial_file_listing_error' not in failure:
            expected = sorted(str(p.relative_to(self.out)) for p in self.out.rglob('*')
                              if p.is_file() and p != self.out / 'failure.json')
            self.assertEqual(failure['partial_files_preserved'], expected)
        return failure

    def test_actual_main_success_constructs_canonical_reader_and_writes_completion(self):
        real = cli.controls.make_reader
        parity_results = []
        real_parity = cli.verify_parity
        real_inject = cli.sa.inject_blind
        injection_outcomes = {}

        def observe_injection(*args, **kwargs):
            result = real_inject(*args, **kwargs)
            key = tuple(kwargs[k] for k in ('role', 'base_window_id', 'ct_channel', 'family', 'strength'))
            injection_outcomes.setdefault(key, []).append(bool(result[2]['accepted']))
            return result

        def measured(*args):
            result = real_parity(*args)
            parity_results.append(copy.deepcopy(result))
            return result

        with mock.patch.object(cli.controls, 'make_reader', wraps=real) as factory, \
                mock.patch.object(cli.sa, 'inject_blind', new=observe_injection), \
                mock.patch.object(cli, 'verify_parity', side_effect=measured):
            self.assertEqual(self.run_main(), 0)
        factory.assert_called_once_with(self.fixture.data_root.resolve(), self.raw)
        result = json.loads((self.out / 'result.json').read_text())
        self.assertEqual(cli._signed_digest(result, 'result'), result['sha256'])
        self.assertEqual(result['canonical_window_reads'], 12)
        self.assertEqual(result['canonical_role_reads'], {'fit': 8, 'dev': 4})
        self.assertEqual(result['raw_roles_read'], ['dev', 'fit'])
        self.assertEqual(result['parity']['raw_windows'], 6)
        self.assertEqual(result['parity']['normal_tensor_pairs'], 12)
        self.assertEqual(result['parity']['synthetic_paired_attempts'], 54)
        self.assertEqual(result['parity'], parity_results[0])
        self.assertEqual(len(injection_outcomes), 54)
        for outcomes in injection_outcomes.values():
            self.assertEqual(len(outcomes), 2)
            self.assertEqual(outcomes[0], outcomes[1])
        accepted = sum(outcomes[0] for outcomes in injection_outcomes.values())
        rejected = len(injection_outcomes) - accepted
        self.assertGreater(accepted, 0)
        self.assertGreater(rejected, 0)
        self.assertEqual(result['parity']['synthetic_accepted'], accepted)
        self.assertEqual(result['parity']['synthetic_rejected'], rejected)
        self.assertEqual(result['parity']['synthetic_sample_windows'], 3)
        self.assertEqual(result['parity']['synthetic_paired_attempts'], accepted + rejected)
        self.assertFalse(result['parity']['synthetic_exhaustive'])
        self.assertFalse(result['performance_measured'])
        self.assertFalse(result['paper_result_eligible'])
        self.assertEqual(result['optimizer_steps'], 0)
        self.assertEqual(result['provenance_file_sha256'], file_sha(self.out / 'provenance.json'))
        manifest = json.loads((self.out / 'cache/manifest.json').read_text())
        self.assertEqual(result['cache_manifest_sha256'], manifest['sha256'])
        provenance = json.loads((self.out / 'provenance.json').read_text())
        self.assertTrue(provenance['git_dirty'])
        self.assertEqual(provenance['git_commit_sha'], self.evidence['git_commit_sha'])
        self.assertEqual(provenance['source_files'], cli.source_hashes())
        self.assertEqual(result['source_files'], cli.source_hashes())
        self.assertEqual(result['binding'], self.review['future_real_cache_binding'])
        self.assertFalse(provenance['paper_result_eligible'])
        self.assertIsNotNone(datetime.fromisoformat(provenance['timestamp']).utcoffset())
        self.assertEqual(provenance['environment_profile'], 'SYNTHETIC_FIXTURE')
        self.assertTrue(set(('git_commit_sha', 'git_dirty', 'timestamp', 'hostname', 'jetpack_l4t',
                             'python_version', 'environment_profile', 'sensor_driver_versions')) <= set(provenance))
        self.assertFalse((self.out / 'failure.json').exists())

    def test_environment_is_rejected_before_any_output_or_reader(self):
        with mock.patch.object(cli, 'server_environment', ORIGINAL_ENVIRONMENT), \
                mock.patch.object(cli.platform, 'machine', return_value='aarch64'), \
                mock.patch.object(cli.controls, 'make_reader') as reader:
            with self.assertRaisesRegex(ValueError, 'SERVER-TRAINING'):
                self.run_main()
        reader.assert_not_called()
        self.assertFalse(self.out.exists())

    def test_existing_output_is_untouched(self):
        self.out.mkdir(parents=True)
        sentinel = self.out / 'sentinel'
        sentinel.write_text('keep')
        with self.assertRaisesRegex(ValueError, 'already exists'):
            self.run_main()
        self.assertEqual(list(self.out.iterdir()), [sentinel])
        self.assertEqual(sentinel.read_text(), 'keep')

    def test_nested_success_and_failed_runs_are_refused_without_touching_parents(self):
        for state, child in (('success', 'cache/run2'), ('failed', 'retry')):
            with self.subTest(state=state):
                parent = self.root / 'outputs' / state
                (parent / 'cache').mkdir(parents=True)
                marker = parent / ('result.json' if state == 'success' else 'failure.json')
                marker.write_text(state)
                before = sorted(str(p.relative_to(parent)) for p in parent.rglob('*'))
                self.out = parent / child
                with self.assertRaisesRegex(ValueError, 'direct child'):
                    self.run_main()
                self.assertEqual(marker.read_text(), state)
                self.assertEqual(sorted(str(p.relative_to(parent)) for p in parent.rglob('*')), before)

    def test_output_must_be_below_ssd_parent_and_separate_from_raw(self):
        with self.assertRaisesRegex(ValueError, 'SSD'):
            cli._check_output(self.root / 'elsewhere', self.fixture.data_root)
        with self.assertRaisesRegex(ValueError, 'separate'):
            cli._check_output(self.out, self.root / 'outputs')

    def test_module_review_file_pin_is_enforced_before_reader(self):
        argv = self.argv()
        argv[argv.index('--review-sha256') + 1] = '0' * 64
        with mock.patch.object(cli.controls, 'make_reader') as factory:
            with self.assertRaisesRegex(ValueError, 'module review file hash'):
                self.run_main(argv)
        factory.assert_not_called()
        self.assertEqual(json.loads((self.out / 'failure.json').read_text())['stage'], 'inputs')

    def test_module_review_self_digest_is_enforced(self):
        self.review['RC-01'] = 'OPEN'
        write_json(self.review_path, self.review)
        self.assert_failed('inputs', ValueError, 'module review payload hash')

    def test_module_review_cannot_be_for_old_source(self):
        self.review['source_files']['src/data/anomaly_raw_cache.py'] = '0' * 64
        self.seal()
        self.assert_failed('inputs', ValueError, 'module review source hashes')

    def test_module_review_requires_actual_server_pass(self):
        self.review['server']['summary']['skipped'] = 1
        self.seal()
        self.assert_failed('inputs', ValueError, 'server fixture PASS')

    def test_resigned_negative_module_review_gate_and_rc_status_are_rejected(self):
        baseline = copy.deepcopy(self.review)
        for key, value in (('gate', 'FAIL'), ('RC-01', 'OPEN')):
            with self.subTest(key=key):
                self.review = {**baseline, key: value}
                self.seal()
                with self.assertRaisesRegex(ValueError, 'server fixture PASS'):
                    cli.validate_inputs(self.args())

    def test_cli_evidence_file_pin_is_enforced(self):
        argv = self.argv()
        argv[argv.index('--cli-fixtures-sha256') + 1] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'CLI fixtures file hash'):
            self.run_main(argv)

    def test_cli_evidence_requires_all_tests_source_and_server(self):
        original = copy.deepcopy(self.evidence)
        for key, value in (('source_files', {}), ('environment_profile', 'JETSON-RUNTIME'),
                           ('skipped', 1), ('tests_run', 0), ('passed', 0)):
            with self.subTest(key=key):
                self.evidence = {**original, key: value}
                self.seal()
                with self.assertRaisesRegex(ValueError, 'source-bound server PASS'):
                    cli.validate_inputs(self.args())

    def test_cli_evidence_schema_digest_revision_and_exact_count_are_enforced(self):
        baseline = copy.deepcopy(self.evidence)
        for change, expected in (({'schema': 'wrong'}, 'source-bound'),
                                 ({'git_commit_sha': 'x' * 40}, 'source revision'),
                                 ({'git_commit_sha': 'a' * 39}, 'source revision'),
                                 ({'git_commit_sha': 'a' * 41}, 'source revision'),
                                 ({'git_commit_sha': None}, 'source revision'),
                                 ({'tests_run': baseline['tests_run'] + 1}, 'source-bound'),
                                 ({'tests_run': baseline['tests_run'] + 1,
                                   'passed': baseline['passed'] + 1}, 'source-bound')):
            with self.subTest(change=change):
                self.evidence = {**baseline, **change}
                self.seal()
                with self.assertRaisesRegex(ValueError, expected):
                    cli.validate_inputs(self.args())
        self.evidence = copy.deepcopy(baseline)
        self.seal()
        self.evidence['git_commit_sha'] = 'b' * 40
        write_json(self.evidence_path, self.evidence)  # re-pin bytes, keep stale payload digest
        with self.assertRaisesRegex(ValueError, 'CLI fixtures payload hash'):
            cli.validate_inputs(self.args())

    def test_resigned_negative_split_gate_protocol_and_payload_are_rejected(self):
        baseline = copy.deepcopy(self.split)
        self.review['future_real_cache_binding']['split_manifest_payload_sha256'] = '0' * 64
        self.seal()
        with self.assertRaisesRegex(ValueError, 'frozen protocol PASS'):
            cli.validate_inputs(self.args())
        for key, value in (('data_gate', 'FAIL'), ('protocol_id', 'different')):
            with self.subTest(key=key):
                self.split = {**baseline, key: value}
                self.update_split()
                with self.assertRaisesRegex(ValueError, 'frozen protocol PASS'):
                    cli.validate_inputs(self.args())
        self.split = copy.deepcopy(baseline)
        self.split['data_gate'] = 'FAIL'
        write_json(self.prepared / 'split_manifest.json', self.split)
        self.review['future_real_cache_binding']['split_manifest_file_sha256'] = file_sha(self.prepared / 'split_manifest.json')
        self.seal()
        with self.assertRaisesRegex(ValueError, 'split manifest payload hash'):
            cli.validate_inputs(self.args())

    def test_raw_payload_and_raw_to_split_binding_each_reject_resigned_negatives(self):
        self.review['future_real_cache_binding']['source_manifest_payload_sha256'] = '0' * 64
        self.seal()
        with self.assertRaisesRegex(ValueError, 'raw manifest payload hash'):
            cli.validate_inputs(self.args())
        self.review['future_real_cache_binding']['source_manifest_payload_sha256'] = cli.sa.digest_json(self.raw)
        self.split['source_manifest_sha256'] = '0' * 64
        self.update_split()
        with self.assertRaisesRegex(ValueError, 'raw manifest payload hash'):
            cli.validate_inputs(self.args())

    def test_normalizer_payload_digest_is_checked_after_repinning_split(self):
        self.split['folds']['0']['normalizer']['thermal_mean'] += 1
        self.update_split()
        self.assert_failed('inputs', ValueError, 'normalizer payload hash')

    def test_split_file_change_is_preserved_before_reader(self):
        with (self.prepared / 'split_manifest.json').open('a') as handle:
            handle.write(' ')
        with mock.patch.object(cli.controls, 'make_reader') as factory:
            self.assert_failed('inputs', ValueError, 'split manifest file hash')
        factory.assert_not_called()
        self.assertFalse((self.out / 'cache').exists())

    def test_raw_manifest_file_change_is_preserved(self):
        with (self.prepared / 'raw_manifest.json').open('a') as handle:
            handle.write(' ')
        self.assert_failed('inputs', ValueError, 'raw manifest file hash')

    def test_duplicate_raw_ledger_paths_are_refused_even_with_consistent_hashes(self):
        self.raw.append(copy.deepcopy(self.raw[0]))
        write_json(self.prepared / 'raw_manifest.json', self.raw)
        sha = cli.sa.digest_json(self.raw)
        self.review['future_real_cache_binding'].update(
            source_manifest_file_sha256=file_sha(self.prepared / 'raw_manifest.json'),
            source_manifest_payload_sha256=sha)
        self.split['source_manifest_sha256'] = sha
        self.update_split()
        self.assert_failed('inputs', ValueError, 'duplicate raw manifest')

    def test_rehashed_split_cannot_change_reviewed_roles(self):
        self.split['folds']['0']['roles']['agv01'] = 'calibration'
        self.update_split()
        self.assert_failed('inputs', ValueError, 'roles differ')

    def test_rehashed_split_cannot_change_reviewed_order(self):
        self.split['windows'].reverse()
        self.update_split()
        self.assert_failed('inputs', ValueError, 'counts or order')

    def test_canonical_root_is_bound_to_the_split(self):
        argv = self.argv()
        argv[argv.index('--data-root') + 1] = str(self.root / 'different_raw')
        with mock.patch.object(cli.controls, 'make_reader') as factory:
            with self.assertRaisesRegex(ValueError, 'data root differs'):
                self.run_main(argv)
        factory.assert_not_called()

    def test_normalizer_fit_ids_are_checked(self):
        norm = self.split['folds']['0']['normalizer']
        norm['fit_raw_ids'] = norm['fit_raw_ids'][1:]
        norm.pop('sha256')
        norm['sha256'] = cli.sa.digest_json(norm)
        self.update_split()
        self.assert_failed('inputs', ValueError, 'normalizer fit provenance')

    def test_normalizer_zero_scale_is_refused(self):
        norm = self.split['folds']['0']['normalizer']
        norm['thermal_scale'] = 0.0
        norm.pop('sha256')
        norm['sha256'] = cli.sa.digest_json(norm)
        self.update_split()
        self.assert_failed('inputs', ValueError, 'invalid frozen normalizer')

    def test_payload_geometry_is_bound_to_dtypes(self):
        self.review['future_real_cache_binding']['array_payload_bytes']['thermal'] -= 1
        self.seal()
        self.assert_failed('inputs', ValueError, 'dtype/size')

    def test_fold_mismatch_is_refused(self):
        self.review['future_real_cache_binding']['fold'] = 1
        self.seal()
        self.assert_failed('inputs', ValueError, 'fold 0 only')

    def test_actual_reader_failure_leaves_wrapper_and_cache_failures(self):
        failure = self.assert_failed('cache_build', OSError, 'injected read failure',
            mock.patch.object(cli.controls, 'make_reader', return_value=mock.Mock(
                side_effect=OSError('injected read failure'))))
        self.assertIn('cache/failure.json', failure['partial_files_preserved'])
        self.assertTrue((self.out / 'cache/thermal.npy').exists())

    def test_flush_failure_preserves_cache_and_wrapper_records(self):
        self.assert_failed('cache_build', OSError, 'injected flush failure',
                           mock.patch.object(np.memmap, 'flush', side_effect=OSError('injected flush failure')))
        child = json.loads((self.out / 'cache/failure.json').read_text())
        self.assertEqual(child['stage'], 'flush')

    def test_interrupt_in_reader_is_preserved(self):
        self.assert_failed('cache_build', KeyboardInterrupt, '',
                           mock.patch.object(cli.controls, 'make_reader', return_value=mock.Mock(
                               side_effect=KeyboardInterrupt)))
        self.assertEqual(json.loads((self.out / 'cache/failure.json').read_text())['exception'], 'KeyboardInterrupt')

    def test_final_json_failure_is_preserved_after_real_build_attach_and_parity(self):
        real = cli._write_json

        def writer(path, value):
            if path.name == 'result.json':
                raise OSError('injected final write failure')
            return real(path, value)

        self.assert_failed('final_record', OSError, 'injected final write failure',
                           mock.patch.object(cli, '_write_json', side_effect=writer))
        self.assertTrue((self.out / 'cache/manifest.json').exists())
        self.assertTrue((self.out / 'provenance.json').exists())

    def test_inputs_changed_during_verification_prevent_completion(self):
        real = cli.verify_parity

        def change_after_verification(*args):
            result = real(*args)
            with (self.prepared / 'split_manifest.json').open('a') as handle:
                handle.write(' ')
            return result

        self.assert_failed('final_checks', ValueError, 'split manifest file hash',
                           mock.patch.object(cli, 'verify_parity', side_effect=change_after_verification))

    def test_injection_metadata_mismatch_prevents_cli_completion(self):
        real = cli.sa.inject_blind
        calls = []

        def change_metadata(*args, **kwargs):
            result = real(*args, **kwargs)
            calls.append(1)
            if len(calls) == 2:
                result[2]['independent_test_change'] = True
            return result

        self.assert_failed('parity', ValueError, 'injection metadata parity',
                           mock.patch.object(cli.sa, 'inject_blind', side_effect=change_metadata))

    def test_verification_rejects_cache_normal_tensor_changes(self):
        inputs = cli.validate_inputs(self.args())
        one = self.fixture.selected[:1]
        inputs = {**inputs, 'windows': one,
                  'cost_plan': {'normal_tensor_pairs': 2, 'synthetic_paired_attempts': 18}}
        with tempfile.TemporaryDirectory() as tmp:
            cache = self.fixture.build(one, out=Path(tmp) / 'cache').open()
            real = cli._final_pair
            calls = []

            def changed(*args):
                result = real(*args)
                calls.append(1)
                if len(calls) == 2:
                    result[0][0, 0] += 1
                return result

            with mock.patch.object(cli, '_final_pair', side_effect=changed):
                with self.assertRaisesRegex(ValueError, 'Normal final sensor byte parity'):
                    cli.verify_parity(cache, inputs, self.fixture.canonical)
            cache._data = None

    def test_failed_record_write_preserves_original_exception(self):
        def failing(path, value):
            raise OSError('disk write unavailable')

        self.review['future_real_cache_binding']['fold'] = 1
        self.seal()
        with mock.patch.object(cli, '_write_json', side_effect=failing), \
                contextlib.redirect_stderr(io.StringIO()) as stderr:
            with self.assertRaisesRegex(ValueError, 'fold 0 only'):
                self.run_main()
        self.assertIn('failure_record_write_error', stderr.getvalue())
        self.assertIn('original_failure', stderr.getvalue())

    def test_failure_listing_errors_do_not_hide_original_error_or_metadata(self):
        for i, error in enumerate((PermissionError('listing denied'), KeyboardInterrupt())):
            with self.subTest(error=type(error).__name__):
                self.out = self.root / 'outputs' / f'listing{i}'
                self.review['future_real_cache_binding']['fold'] = 1
                self.seal()
                failure = self.assert_failed('inputs', ValueError, 'fold 0 only',
                                            mock.patch.object(Path, 'rglob', side_effect=error))
                self.assertEqual(failure['partial_file_listing_error']['exception'], type(error).__name__)
                self.assertEqual(failure['partial_files_preserved'], [])
                self.assertEqual(failure['git_commit_sha'], 'a' * 40)
                self.assertEqual(failure['git_revision_source'], 'validated CLI fixture evidence')
                self.assertEqual(failure['cli_fixture_file_sha256'], file_sha(self.evidence_path))
                self.assertTrue(failure['git_dirty'])
                self.assertEqual(failure['source_files'], cli.source_hashes())
                self.assertTrue(set(('timestamp', 'hostname', 'jetpack_l4t', 'python_version',
                                     'environment_profile', 'sensor_driver_versions')) <= set(failure))

    def test_unverified_evidence_never_supplies_failure_revision(self):
        review, evidence = copy.deepcopy(self.review), copy.deepcopy(self.evidence)
        cases = ('review_pin', 'cli_pin', 'cli_digest', 'cli_schema', 'cli_sources',
                 'git_nonhex', 'git_short', 'git_null')
        for case in cases:
            with self.subTest(case=case):
                self.out = self.root / 'outputs' / case
                self.review, self.evidence = copy.deepcopy(review), copy.deepcopy(evidence)
                if case == 'cli_schema':
                    self.evidence['schema'] = 'wrong'
                elif case == 'cli_sources':
                    self.evidence['source_files'] = {}
                elif case.startswith('git_'):
                    self.evidence['git_commit_sha'] = {'git_nonhex': 'x' * 40,
                                                       'git_short': 'a' * 39, 'git_null': None}[case]
                self.seal()
                if case == 'cli_digest':
                    self.evidence['git_commit_sha'] = 'b' * 40
                    write_json(self.evidence_path, self.evidence)  # pin bytes, stale payload digest
                argv = self.argv()
                if case in ('review_pin', 'cli_pin'):
                    flag = '--review-sha256' if case == 'review_pin' else '--cli-fixtures-sha256'
                    argv[argv.index(flag) + 1] = '0' * 64
                with mock.patch.object(cli.controls, 'make_reader') as factory:
                    with self.assertRaises(ValueError):
                        self.run_main(argv)
                factory.assert_not_called()
                failure = json.loads((self.out / 'failure.json').read_text())
                self.assertEqual(failure['stage'], 'inputs')
                self.assertIsNone(failure['git_commit_sha'])
                self.assertEqual(failure['git_revision_source'], 'not yet verified; input gate incomplete')
                self.assertTrue(failure['git_dirty'])
                self.assertEqual(failure['source_files'], cli.source_hashes())
                for field, flag in (('module_review_file_sha256_requested', '--review-sha256'),
                                    ('cli_fixture_file_sha256_requested', '--cli-fixtures-sha256')):
                    self.assertEqual(failure[field], argv[argv.index(flag) + 1])
                self.assertNotIn('cli_fixture_file_sha256', failure)
                self.assertFalse((self.out / 'provenance.json').exists())
                self.assertFalse((self.out / 'result.json').exists())

    def test_failure_writer_interrupt_keeps_the_original_exception(self):
        self.review['future_real_cache_binding']['fold'] = 1
        self.seal()
        with mock.patch.object(cli, '_write_json', side_effect=KeyboardInterrupt), \
                contextlib.redirect_stderr(io.StringIO()) as stderr:
            try:
                with self.assertRaisesRegex(ValueError, 'fold 0 only'):
                    self.run_main()
            except BaseException as exc:
                if isinstance(exc, Exception):
                    raise
                self.fail(f'unexpected {type(exc).__name__} escaped failure writer: {exc}')
        self.assertIn('KeyboardInterrupt', stderr.getvalue())

    def test_each_environment_guard_independently_rejects_an_otherwise_valid_environment(self):
        exists = Path.exists

        def no_tegra(path):
            return False if str(path) == '/etc/nv_tegra_release' else exists(path)

        with mock.patch.object(cli.platform, 'machine', return_value='x86_64'), \
                mock.patch.object(cli.sys, 'prefix', str(Path.home() / 'venvs/factory_training')), \
                mock.patch.dict(os.environ, {'PYTHONNOUSERSITE': '1'}), \
                mock.patch.object(cli.site, 'ENABLE_USER_SITE', False), \
                mock.patch.object(Path, 'exists', autospec=True, side_effect=no_tegra):
            self.assertEqual(ORIGINAL_ENVIRONMENT()['environment_profile'], 'SERVER-TRAINING')
            cases = (mock.patch.object(cli.platform, 'machine', return_value='aarch64'),
                     mock.patch.object(Path, 'exists', return_value=True),
                     mock.patch.object(cli.sys, 'prefix', '/wrong/venv'),
                     mock.patch.dict(os.environ, {'PYTHONNOUSERSITE': '0'}),
                     mock.patch.object(cli.site, 'ENABLE_USER_SITE', True))
            for i, patch in enumerate(cases):
                with self.subTest(guard=i), patch, self.assertRaisesRegex(ValueError, 'SERVER-TRAINING'):
                    ORIGINAL_ENVIRONMENT()

    def test_exact_array_comparison_rejects_ulp_dtype_shape_and_signed_zero(self):
        f4, f8 = np.array([1.0], dtype='f4'), np.array([1.0], dtype='f8')
        pairs = [(f4, np.nextafter(f4, np.float32(np.inf))),
                 (f8, np.nextafter(f8, np.float64(np.inf))), (f4, f8),
                 (np.array([0.0]), np.array([-0.0])),
                 (np.zeros(2), np.zeros((1, 2))),
                 (np.zeros(30, dtype='i8'), np.zeros(30, dtype='i4'))]
        for dtype in ('<f8', '>i8', '<u8'):
            a, b = np.zeros(30, dtype='<i8'), np.zeros(30, dtype=dtype)
            self.assertEqual(a.tobytes(), b.tobytes())
            self.assertEqual(a.shape, b.shape)
            pairs.append((a, b))
        for i, (a, b) in enumerate(pairs):
            with self.subTest(case=i), self.assertRaisesRegex(ValueError, 'byte parity'):
                cli._same_array(a, b, 'independent probe')
        cli._same_array(f4, f4.copy(), 'equal copy')

    def test_canonical_second_pass_ulp_dtype_and_shape_changes_fail_main_parity(self):
        real = cli.controls.make_reader
        target = self.fixture.selected[1]['base_window_id']  # unsampled second window
        for case in ('thermal_ulp', 'labels_dtype', 'sensor_shape'):
            self.out = self.root / 'outputs' / case
            seen = Counter()

            def factory(*args):
                reader = real(*args)

                def altered(window):
                    arrays = list(reader(window))
                    identifier = window['base_window_id']
                    seen[identifier] += 1
                    if identifier == target and seen[identifier] == 2:
                        if case == 'thermal_ulp':
                            before = cli._final_pair(*arrays[:2], self.split['folds']['0']['normalizer'], 'CT1')[1]
                            arrays[1].flat[0] = np.nextafter(arrays[1].flat[0], np.inf)
                            after = cli._final_pair(*arrays[:2], self.split['folds']['0']['normalizer'], 'CT1')[1]
                            self.assertEqual(before.tobytes(), after.tobytes())
                        elif case == 'labels_dtype':
                            arrays[2] = arrays[2].astype('i4')
                        else:
                            arrays[0] = arrays[0].reshape(8, 30)
                    return tuple(arrays)
                return altered

            with self.subTest(case=case):
                self.assert_failed('parity', ValueError, 'byte parity',
                                   mock.patch.object(cli.controls, 'make_reader', side_effect=factory))

    def test_cache_changed_after_attach_is_caught_by_per_read_verification(self):
        real = cli.rc.RawNormalCache.attach

        def corrupt_after_attach(out):
            cache = real(out)
            data = np.load(Path(out) / 'thermal.npy', mmap_mode='r+')
            data[0, 0, 0, 0] = np.nextafter(data[0, 0, 0, 0], np.inf)
            data.flush()
            del data
            return cache

        self.assert_failed('parity', ValueError, 'changed since',
                           mock.patch.object(cli.rc.RawNormalCache, 'attach', side_effect=corrupt_after_attach))

    def test_factor_role_seed_and_ct_calls_cover_the_declared_design(self):
        inject, final = cli.sa.inject_blind, cli._final_pair
        with mock.patch.object(cli.sa, 'inject_blind', wraps=inject) as injections, \
                mock.patch.object(cli, '_final_pair', wraps=final) as tensors:
            self.run_main()
        expected_windows = [next(w for w in self.fixture.selected if w['device'] == device)
                            for device in ('agv01', 'oht01', 'agv02')]
        expected = Counter({(w['base_window_id'], fixtures.ROLES[w['device']], ct, family, strength, 42, 0,
                             cli.sa.PROTOCOL): 2
                            for w in expected_windows
                            for ct, family, strength in itertools.product(
                                ('CT1', 'CT2'), ('sensor-only', 'thermal-only', 'coupled'), (1, 2, 4))})
        keys = ('base_window_id', 'role', 'ct_channel', 'family', 'strength', 'generator_seed', 'fold', 'protocol')
        self.assertEqual(Counter(tuple(call.kwargs[k] for k in keys) for call in injections.call_args_list), expected)
        self.assertEqual(Counter(call.args[3] for call in tensors.call_args_list), {'CT1': 66, 'CT2': 66})

    def test_two_reads_are_required_for_each_window_not_only_the_total(self):
        for mode in ('one_pass_only', 'same_total_wrong_distribution', 'one_window_overread'):
            self.out = self.root / 'outputs' / mode

            def incomplete(cache, inputs, canonical):
                if mode == 'same_total_wrong_distribution':
                    for w in [inputs['windows'][0], inputs['windows'][0], *inputs['windows'][2:]]:
                        canonical(w)
                elif mode == 'one_window_overread':
                    for w in [*inputs['windows'], inputs['windows'][0]]:
                        canonical(w)
                return {'gate': 'PASS', 'fixture_incomplete_parity': True}

            with self.subTest(mode=mode):
                self.assert_failed('final_checks', ValueError, 'canonical read counts',
                                   mock.patch.object(cli, 'verify_parity', side_effect=incomplete))

    @contextlib.contextmanager
    def one_window_parity(self):
        inputs = cli.validate_inputs(self.args())
        inputs = {**inputs, 'windows': self.fixture.selected[:1],
                  'cost_plan': {'normal_tensor_pairs': 2, 'synthetic_paired_attempts': 18}}
        with tempfile.TemporaryDirectory() as tmp:
            cache = self.fixture.build(inputs['windows'], out=Path(tmp) / 'cache').open()
            try:
                yield cache, inputs
            finally:
                cache._data = None

    def test_parity_coverage_checks_normal_and_synthetic_counts(self):
        with self.one_window_parity() as (cache, inputs):
            for key in ('normal_tensor_pairs', 'synthetic_paired_attempts'):
                wrong = {**inputs, 'cost_plan': {**inputs['cost_plan'], key: inputs['cost_plan'][key] + 1}}
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'coverage differs'):
                    cli.verify_parity(cache, wrong, self.fixture.canonical)

    def test_source_drift_is_rechecked_after_parity(self):
        real = cli.source_hashes
        calls = []

        def drift():
            current = real()
            calls.append(1)
            if len(calls) > 1:
                current['src/build_anomaly_raw_cache.py'] = '0' * 64
            return current

        self.assert_failed('final_checks', ValueError, 'source-bound server PASS',
                           mock.patch.object(cli, 'source_hashes', side_effect=drift))

    def test_normal_and_injected_comparisons_use_both_independent_reader_outputs(self):
        with self.one_window_parity() as (cache, inputs):
            origins, final_calls, injection_calls = {}, [], []
            real_cached, real_final, real_inject = cache.reader, cli._final_pair, cli.sa.inject_blind

            def canonical(window):
                result = self.fixture.canonical(window)
                origins[(id(result[0]), id(result[1]))] = ('normal_canonical',)
                return result

            def cached_factory(**kwargs):
                read = real_cached(**kwargs)

                def cached(window):
                    result = read(window)
                    origins[(id(result[0]), id(result[1]))] = ('normal_cache',)
                    return result
                return cached

            def inject(sensor, thermal, **kwargs):
                origin = origins[(id(sensor), id(thermal))][0]
                factor = (kwargs['ct_channel'], kwargs['family'], kwargs['strength'])
                injection_calls.append((origin, *factor))
                result = real_inject(sensor, thermal, **kwargs)
                origins[(id(result[0]), id(result[1]))] = ('injected_' + origin.removeprefix('normal_'), *factor)
                return result

            def final(sensor, thermal, normalizer, ct):
                final_calls.append((*origins[(id(sensor), id(thermal))], ct))
                return real_final(sensor, thermal, normalizer, ct)

            with mock.patch.object(cache, 'reader', side_effect=cached_factory), \
                    mock.patch.object(cli.sa, 'inject_blind', side_effect=inject), \
                    mock.patch.object(cli, '_final_pair', side_effect=final):
                cli.verify_parity(cache, inputs, canonical)
            factors = list(itertools.product(('CT1', 'CT2'), ('sensor-only', 'thermal-only', 'coupled'), (1, 2, 4)))
            self.assertEqual(Counter(injection_calls), Counter((origin, *factor)
                             for origin in ('normal_canonical', 'normal_cache') for factor in factors))
            expected = Counter((origin, ct) for origin in ('normal_canonical', 'normal_cache') for ct in ('CT1', 'CT2'))
            expected.update((origin, *factor, factor[0]) for origin in ('injected_canonical', 'injected_cache')
                            for factor in factors)
            self.assertEqual(Counter(final_calls), expected)

    def test_witness_includes_normal_tensors_even_when_synthetic_results_are_unchanged(self):
        with self.one_window_parity() as (cache, inputs):
            baseline = cli.verify_parity(cache, inputs, self.fixture.canonical)
            normal_ids = set()
            real_read, real_final = cache.reader, cli._final_pair

            def canonical(window):
                result = self.fixture.canonical(window)
                normal_ids.add((id(result[0]), id(result[1])))
                return result

            def reader_factory(**kwargs):
                read = real_read(**kwargs)

                def read_normal(window):
                    result = read(window)
                    normal_ids.add((id(result[0]), id(result[1])))
                    return result
                return read_normal

            def change_both_normal_sides(sensor, thermal, normalizer, ct):
                result = real_final(sensor, thermal, normalizer, ct)
                if (id(sensor), id(thermal)) in normal_ids:
                    for array in result:
                        array += np.float32(1)
                return result

            with mock.patch.object(cache, 'reader', side_effect=reader_factory), \
                    mock.patch.object(cli, '_final_pair', side_effect=change_both_normal_sides):
                changed = cli.verify_parity(cache, inputs, canonical)
            self.assertNotEqual(baseline['tensor_and_event_stream_sha256'], changed['tensor_and_event_stream_sha256'])
            self.assertEqual(baseline['synthetic_accepted'], changed['synthetic_accepted'])

    def test_witness_includes_injected_metadata_and_tensor_changes(self):
        with self.one_window_parity() as (cache, inputs):
            baseline = cli.verify_parity(cache, inputs, self.fixture.canonical)
            real_inject = cli.sa.inject_blind
            for mode in ('metadata', 'tensor'):
                def change_both_injected_sides(*args, **kwargs):
                    result = real_inject(*args, **kwargs)
                    if mode == 'metadata':
                        result[2]['witness_fixture_tag'] = 'changed on both sides'
                    else:
                        result[0][0, 0] += 1  # final float32 changes; raw Normal arrays are untouched
                    return result

                with self.subTest(mode=mode), mock.patch.object(cli.sa, 'inject_blind', new=change_both_injected_sides):
                    changed = cli.verify_parity(cache, inputs, self.fixture.canonical)
                self.assertNotEqual(baseline['tensor_and_event_stream_sha256'], changed['tensor_and_event_stream_sha256'])
                self.assertEqual(baseline['synthetic_accepted'], changed['synthetic_accepted'])
                self.assertEqual(baseline['synthetic_rejected'], changed['synthetic_rejected'])

    def test_output_symlinks_are_resolved_before_parent_and_raw_separation(self):
        parent = self.root / 'outputs'
        parent.mkdir()
        parent_link = self.root / 'parent_link'
        parent_link.symlink_to(parent, target_is_directory=True)
        with mock.patch.object(cli, 'OUTPUT_PARENT', parent_link):
            self.assertEqual(cli._check_output(parent_link / 'new', self.fixture.data_root), parent / 'new')
        old_run = parent / 'old_run'
        old_run.mkdir()
        marker = old_run / 'result.json'
        marker.write_bytes(b'keep prior run')
        out_link = parent / 'alias'
        out_link.symlink_to(old_run / 'retry', target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'direct child'):
            cli._check_output(out_link, self.fixture.data_root)
        self.assertEqual(marker.read_bytes(), b'keep prior run')
        self.assertFalse((old_run / 'retry').exists())

    def test_proposed_report_status_is_not_promoted_into_the_completed_run_binding(self):
        self.review['future_real_cache_binding']['status'] = 'PROPOSED_NOT_CREATED'
        self.seal()
        result = cli.validate_inputs(self.args())
        self.assertNotIn('status', result['binding'])

    def test_injected_raw_ulp_difference_is_rejected_before_float32_rounding(self):
        with self.one_window_parity() as (cache, inputs):
            real = cli.sa.inject_blind
            calls = []

            def change_one_side(*args, **kwargs):
                result = real(*args, **kwargs)
                calls.append(1)
                if len(calls) == 2:
                    before = cli._final_pair(*result[:2], inputs['normalizer'], kwargs['ct_channel'])[1]
                    result[1].flat[0] = np.nextafter(result[1].flat[0], np.inf)
                    after = cli._final_pair(*result[:2], inputs['normalizer'], kwargs['ct_channel'])[1]
                    self.assertEqual(before.tobytes(), after.tobytes())
                return result

            with mock.patch.object(cli.sa, 'inject_blind', side_effect=change_one_side), \
                    self.assertRaisesRegex(ValueError, 'injected raw byte parity'):
                cli.verify_parity(cache, inputs, self.fixture.canonical)

    def test_canonical_reader_refuses_changed_identity_and_role_before_raw_read(self):
        real = cli.controls.make_reader
        for case in ('start_index', 'heldout_device'):
            self.out = self.root / 'outputs' / case
            raw_calls = []

            def factory(*args):
                read = real(*args)

                def tracked(window):
                    raw_calls.append(window['base_window_id'])
                    return read(window)
                return tracked

            def bad_request(cache, inputs, canonical):
                changed = copy.deepcopy(inputs['windows'][0])
                if case == 'start_index':
                    changed['start_index'] += 1
                else:
                    changed['device'] = 'outside_frozen_roles'
                return canonical(changed)

            with self.subTest(case=case), \
                    mock.patch.object(cli.controls, 'make_reader', side_effect=factory), \
                    mock.patch.object(cli, 'verify_parity', side_effect=bad_request):
                self.assert_failed('parity', ValueError, 'outside the frozen fit/dev view')
            self.assertEqual(len(raw_calls), len(self.fixture.selected))

    def test_output_cannot_contain_or_equal_the_raw_root(self):
        for raw_root in (self.out, self.out / 'raw'):
            argv = self.argv()
            argv[argv.index('--data-root') + 1] = str(raw_root)
            with self.subTest(raw_root=raw_root), self.assertRaisesRegex(ValueError, 'separate from raw'):
                self.run_main(argv)
            self.assertFalse(self.out.exists())

    def test_json_write_preserves_existing_temporary_artifact(self):
        target = self.root / 'record.json'
        temporary = self.root / 'record.json.tmp'
        temporary.write_bytes(b'preserved incomplete record')
        with self.assertRaises(FileExistsError):
            cli._write_json(target, {'replacement': True})
        self.assertEqual(temporary.read_bytes(), b'preserved incomplete record')
        self.assertFalse(target.exists())


if __name__ == '__main__':
    unittest.main()
