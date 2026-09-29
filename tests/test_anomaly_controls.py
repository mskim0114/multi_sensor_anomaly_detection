"""Cross-module contract checks: raw fit/dev/cal views through frozen controls."""
import copy
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('anomaly_controls_test',ROOT/'src/fit_anomaly_controls.py')
controls=importlib.util.module_from_spec(spec);sys.modules[spec.name]=controls;spec.loader.exec_module(controls)
sa=controls.sa


class IntegratedControlsTests(unittest.TestCase):
    def fixture(self):
        roles={'agv01':'fit','agv02':'dev','agv03':'calibration','agv04':'heldout'}
        ticks=[]
        for j,device in enumerate(roles):
            for i in range(40):
                v=np.array([20+i*.1,10+i*.01,20+i*.02,30+i*.03,5+i*.01,6,7,8.])
                ticks.append(sa.Tick(f'Training/{device}_0901_{i:06d}',device,'0901',i,0,
                                     v,35+i*.02,0.))
        windows,_=sa.make_windows(ticks)
        by_id={t.base_id:t for t in ticks}
        normalizer=sa.fit_normalizer(ticks,windows,roles)
        read=[]
        def reader(w):
            read.append(w['device'])
            if roles[w['device']]=='heldout': raise AssertionError('heldout was read')
            rows=[by_id[i] for i in w['raw_base_ids']]
            return np.stack([t.sensor for t in rows]),np.broadcast_to(
                np.array([t.thermal_mean for t in rows])[:,None,None],(30,120,160)).copy(),np.array([t.label for t in rows])
        return windows,roles,normalizer,reader,read

    def test_fit_dev_cal_connect_without_heldout_reads(self):
        windows,roles,norm,reader,read=self.fixture()
        r=controls.fit_controls(windows,roles,norm,reader,fold=0)
        self.assertEqual(set(read),{'agv01','agv02','agv03'})
        self.assertEqual(r['event_accounting']['attempted'],18)
        self.assertEqual(r['event_accounting']['accepted'],18)
        self.assertIn(r['calibration_gate'],['PASS','DEGENERATE'])
        self.assertFalse(r['heldout_scored'])
        self.assertEqual(r['full_P0_gate'],'NOT_EVALUATED')
        self.assertEqual(set(r['calibration']),{'B-NTC','B-MD','B-THERM','B-LATE'})
        for row in r['calibration'].values():
            self.assertLessEqual(row['primary']['NQ']['achieved_normal_fpr'],.01)
            self.assertLessEqual(row['primary']['SYN']['achieved_normal_fpr'],.01)
        for record in r['calibration_events']:
            self.assertEqual(record['namespace'][2],'calibration')

    def test_changed_normalizer_or_fit_ids_fail_before_read(self):
        windows,roles,norm,reader,read=self.fixture()
        altered=copy.deepcopy(norm);altered['sensor_mean'][0]+=1
        with self.assertRaisesRegex(ValueError,'hash mismatch'):
            controls.fit_controls(windows,roles,altered,reader,fold=0)
        self.assertEqual(read,[])
        altered=copy.deepcopy(norm);altered['fit_raw_ids']=altered['fit_raw_ids'][1:]
        altered.pop('sha256');altered['sha256']=sa.digest_json(altered)
        with self.assertRaisesRegex(ValueError,'provenance'):
            controls.fit_controls(windows,roles,altered,reader,fold=0)
        self.assertEqual(read,[])

    def test_actual_labels_refuse_a_misflagged_Normal_view(self):
        windows,roles,norm,reader,read=self.fixture()
        def bad_reader(w):
            s,t,labels=reader(w);labels[0]=1;return s,t,labels
        with self.assertRaisesRegex(ValueError,'actual reader labels'):
            controls.fit_controls(windows,roles,norm,bad_reader,fold=0)

    def test_degenerate_calibration_is_not_a_plain_PASS(self):
        windows,roles,norm,reader,read=self.fixture()
        class DegenerateCal:
            @staticmethod
            def calibrate_with_sensitivity(*args,**kwargs):
                return {'primary':{'NQ':{},'SYN':{'degenerate_always_normal':True}}}
        with mock.patch.object(controls,'leaf',return_value=DegenerateCal):
            result=controls.fit_controls(windows,roles,norm,reader,fold=0)
        self.assertEqual(result['calibration_gate'],'DEGENERATE')
        self.assertTrue(result['calibration_diagnostics']['B-THERM']['primary_degeneracy_flags']['SYN']['degenerate_always_normal'])


if __name__=='__main__':unittest.main()
