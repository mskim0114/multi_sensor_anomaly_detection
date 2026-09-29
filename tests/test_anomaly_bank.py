"""Fit/dev injection coverage, reproducibility and role-boundary integration checks."""
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest import mock

ROOT=Path(__file__).resolve().parents[1]


def leaf(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module


bank=leaf('_bank_test','src/validate_synthetic_bank.py')
fixtures=leaf('_bank_fixtures','tests/test_anomaly_controls.py')


class BankTests(unittest.TestCase):
    def test_fit_dev_only_replay_and_manifest_digest(self):
        import hashlib
        windows,roles,norm,reader,read=fixtures.IntegratedControlsTests().fixture()
        records=[]
        result=bank.validate_bank(windows,roles,norm,reader,fold=0,emit=records.append)
        self.assertEqual(set(read),{'agv01','agv02'})
        self.assertEqual(result['attempted'],36)
        self.assertEqual(result['bank_gate'],'PASS')
        self.assertEqual(result['event_manifest_sha256'],hashlib.sha256(''.join(records).encode()).hexdigest())
        self.assertEqual(result,bank.validate_bank(windows,roles,norm,reader,fold=0))

    def test_rejected_family_fails_coverage_without_silent_reweighting(self):
        windows,roles,norm,reader,_=fixtures.IntegratedControlsTests().fixture()
        actual=bank.sa.inject_blind
        def reject(*args,**kwargs):
            s,t,event=actual(*args,**kwargs)
            if event['family']=='thermal-only':
                event['accepted']=False;event['rejection_reasons']=['fixture_rejection']
            return s,t,event
        with mock.patch.object(bank.sa,'inject_blind',side_effect=reject):
            result=bank.validate_bank(windows,roles,norm,reader,fold=0)
        self.assertEqual(result['bank_gate'],'FAIL')
        self.assertEqual(len(result['coverage']['fit']['missing_accepted_cells']),3)
        self.assertEqual(result['attempted'],result['accepted']+result['rejected'])

    def test_actual_non_Normal_tick_is_refused(self):
        windows,roles,norm,reader,_=fixtures.IntegratedControlsTests().fixture()
        def contaminated(w):
            s,t,labels=reader(w);labels[-1]=1;return s,t,labels
        with self.assertRaisesRegex(ValueError,'pure Normal'):
            bank.validate_bank(windows,roles,norm,contaminated,fold=0)


if __name__=='__main__':unittest.main()
