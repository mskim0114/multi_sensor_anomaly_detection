"""EXP004 leakage, raw/QC, split and injection regression tests using fixtures."""
import ast
import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("synthetic_anomaly_test_leaf", ROOT / "src/data/synthetic_anomaly.py")
sa = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = sa
SPEC.loader.exec_module(sa)


def fold_map():
    # Execute exactly the existing pure function, without importing GPU training.
    tree = ast.parse((ROOT / "src/train_kfold.py").read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "assign_folds")
    scope = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "assign_folds_fixture", "exec"), scope)
    return scope["assign_folds"]([f"{kind}{i:02d}" for kind in ("agv", "oht") for i in range(1, 17)], 4)


def ticks(device="agv01", count=40, label=0):
    return [sa.Tick(f"Training/{device}_0901_{i:06d}", device, "0901", i, label,
                    np.array([20+i, 1, 2, 3, 4, 5, 6, 7.], dtype=float), 35.0+i, 4.0)
            for i in range(count)]


class ManifestTests(unittest.TestCase):
    def test_fold_roles_exact_and_each_device_heldout_once(self):
        maps = [sa.partition_devices(fold_map(), f) for f in range(4)]
        for roles in maps:
            self.assertEqual(sa.Counter(roles.values()), dict(fit=16, dev=4, calibration=4, heldout=8))
            for kind in ("agv", "oht"):
                self.assertEqual(sum(d.startswith(kind) and r == "dev" for d,r in roles.items()), 2)
        for d in fold_map():
            self.assertEqual(sum(m[d] == "heldout" for m in maps), 1)
        bad = fold_map(); bad.pop("agv01")
        with self.assertRaises(ValueError): sa.partition_devices(bad, 0)

    def test_plurality_Normal_is_not_pure_Normal(self):
        ts = ticks(count=30)
        for t,l in zip(ts, [0]*8+[1]*6+[2]*8+[3]*8): t.label=l
        ws,_ = sa.make_windows(ts)
        self.assertEqual(ws[0]["majority_label"], 0)
        self.assertFalse(ws[0]["pure_Normal"])
        self.assertTrue(ws[0]["mixed"])

    def test_gap_and_hard_invalid_are_shared_exclusions(self):
        ts=ticks()
        for t in ts[20:]: t.second+=1
        ts[5].hard_errors=("nonfinite",)
        ws,_=sa.make_windows(ts)
        self.assertTrue(all(not w["hard_valid"] for w in ws))
        self.assertIn("non_1Hz_window",ws[0]["hard_errors"])
        self.assertIn("nonfinite",ws[0]["hard_errors"])

    def test_sessions_not_crossed_and_duplicates_fail(self):
        ts=ticks(count=60)
        for t in ts[30:]: t.second+=200
        ws,ss=sa.make_windows(ts)
        self.assertEqual((len(ws),len(ss)),(2,2))
        with self.assertRaises(ValueError): sa.make_windows(ts+[ts[0]])

    def test_soft_QC_stays_eligible_and_deduplicates_fit_ticks(self):
        ts=ticks()
        ts[12].soft_flags=("thermal_negative_C","thermal_outside_legacy_range")
        ws,_=sa.make_windows(ts)
        self.assertTrue(all(w["hard_valid"] for w in ws))
        norm=sa.fit_normalizer(ts,ws,{"agv01":"fit"})
        self.assertEqual(norm["unique_raw_ticks"],40)
        self.assertAlmostEqual(norm["sensor_mean"][0],39.5)
        self.assertAlmostEqual(norm["thermal_std"],np.sqrt(np.var(np.arange(40))+4))
        self.assertEqual(norm["sensor_std"][1],0)
        self.assertEqual(norm["sensor_scale"][1],1)

    def test_heldout_values_do_not_change_normalizer(self):
        ts=ticks()+ticks("agv02")
        ws,_=sa.make_windows(ts)
        roles={"agv01":"fit","agv02":"heldout"}
        expected=sa.fit_normalizer(ts,ws,roles)
        for t in ts[40:]:
            t.sensor[:]=1e12; t.thermal_mean=1e12
        self.assertEqual(sa.fit_normalizer(ts,ws,roles),expected)

    def test_exact_decimal_constant_and_real_small_variation(self):
        ts=ticks()
        for t in ts: t.sensor[:]=.1
        ts[0].sensor[1]=np.nextafter(.1,1.)
        ws,_=sa.make_windows(ts)
        norm=sa.fit_normalizer(ts,ws,{'agv01':'fit'})
        self.assertEqual(norm['sensor_std'][0],0.)
        self.assertEqual(norm['sensor_scale'][0],1.)
        self.assertGreater(norm['sensor_std'][1],0.)
        self.assertLess(norm['sensor_scale'][1],1e-12)

    def test_no_Normal_device_fails_gate_without_borrowing(self):
        ts=[]
        for d in fold_map(): ts.extend(ticks(d,30,1 if d=="agv01" else 0))
        m=sa.prepare_manifests(ts,fold_map())
        self.assertEqual(m["data_gate"],"FAIL")
        self.assertEqual(m["full_P0_gate"],"NOT_EVALUATED")
        self.assertEqual(len(m["data_gate_failures"]),3)
        self.assertFalse(m["heldout_model_performance_computed"])


class RawReadTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.source=self.root/"Training/원천데이터"; self.source.mkdir(parents=True)
        self.labels=self.root/"Training/라벨링데이터"; self.labels.mkdir()
        self.name="agv01_0901_120000"
        self.csv=self.source/(self.name+".csv")
        self.csv.write_text(','.join(sa.CHANNELS)+'\n20,1,2,3,4,5,6,7\n')
        self.bin=self.source/(self.name+".bin")
        self.thermal(np.ones((120,160))*35)
        self.doc={"meta_info":[{"device_id":"agv01","collection_date":"09-01","collection_time":"12:00:00","duration_time":"1"}],
                  "annotations":[{"tagging":[{"state":0}]}]}
        self.label=self.labels/(self.name+".json"); self.label.write_text(json.dumps(self.doc))

    def thermal(self,arr):
        with self.bin.open('wb') as f: np.save(f,arr,allow_pickle=False)

    def read(self): return sa.read_tick(self.source,self.labels,self.name,"Training")

    def test_contents_hashed_and_raw_unmodified(self):
        before={p:p.read_bytes() for p in [self.csv,self.bin,self.label]}
        tick,ledger=self.read()
        self.assertEqual(tick.hard_errors,())
        self.assertEqual(len(ledger),3)
        for p,b in before.items(): self.assertEqual(p.read_bytes(),b)

    def test_negative_thermal_is_soft_not_clipped(self):
        arr=np.ones((120,160))*35;arr[1,1]=-20;self.thermal(arr)
        t,_=self.read()
        self.assertFalse(t.hard_errors)
        self.assertIn('thermal_negative_C',t.soft_flags)
        self.assertAlmostEqual(t.thermal_mean,arr.mean())

    def test_nonfinite_and_shape_and_timestamp_are_hard_invalid(self):
        self.thermal(np.ones((2,2)))
        self.assertIn('thermal_shape',self.read()[0].hard_errors)
        self.thermal(np.full((120,160),np.nan))
        self.assertIn('nonfinite',self.read()[0].hard_errors)
        self.doc['meta_info'][0]['collection_time']='12:00:01';self.label.write_text(json.dumps(self.doc))
        self.assertIn('timestamp_or_device_metadata',self.read()[0].hard_errors)

    def test_missing_label_or_invalid_schema_fails_closed(self):
        self.doc['annotations'][0]['tagging'][0]['state']=4;self.label.write_text(json.dumps(self.doc))
        with self.assertRaises(ValueError): self.read()
        self.label.unlink()
        with self.assertRaises(ValueError): sa.scan_split(self.root,'Training')


class InjectionTests(unittest.TestCase):
    def setUp(self):
        self.sensor=np.tile([20,10,20,30,5,6,7,8.],(30,1))
        self.thermal=np.full((30,120,160),35.)
        self.kw=dict(tick_labels=np.zeros(30,dtype=int),normalizer={'sensor_std':[.2]*8,'thermal_std':2.,'sha256':'fixture','fit_ids_sha256':'fixture'},
                     protocol=sa.PROTOCOL,fold=0,role='fit',base_window_id='fixture:30',strength=2,generator_seed=42)

    def event(self,family,**changes):
        return sa.inject_blind(self.sensor,self.thermal,family=family,**dict(self.kw,**changes))

    def test_replay_no_mutation_and_no_cross_modality_change(self):
        original_s=self.sensor.copy();original_t=self.thermal.copy()
        for family in ['sensor-only','thermal-only','coupled']:
            s,t,m=self.event(family);s2,t2,m2=self.event(family)
            np.testing.assert_array_equal(s,s2);np.testing.assert_array_equal(t,t2);self.assertEqual(m,m2)
            self.assertTrue(m['accepted'])
            if family=='sensor-only': np.testing.assert_array_equal(t,original_t)
            if family=='thermal-only': np.testing.assert_array_equal(s,original_s)
        np.testing.assert_array_equal(self.sensor,original_s);np.testing.assert_array_equal(self.thermal,original_t)

    def test_Namespace_and_CT_sensitivity_pairing(self):
        _,_,a=self.event('coupled')
        _,_,b=self.event('coupled',role='calibration')
        self.assertNotEqual(a['event_id'],b['event_id'])
        _,t,c=self.event('coupled',ct_channel='CT2')
        _,t1,_=self.event('coupled',ct_channel='CT1')
        np.testing.assert_array_equal(t,t1)
        self.assertEqual(a['parameters']['start'],c['parameters']['start'])
        self.assertEqual(a['event_id'],c['event_id'])
        self.assertEqual(a['event_id_scope'],'paired_CT_sensitivity')
        self.assertNotEqual(a['event_instance_id'],c['event_instance_id'])

    def test_non_Normal_and_outside_grid_rejected(self):
        bad=np.zeros(30);bad[3]=1
        with self.assertRaises(ValueError): self.event('sensor-only',tick_labels=bad)
        with self.assertRaises(ValueError): self.event('sensor-only',strength=128)

    def test_zero_scale_and_base_constraint_violations_accounted(self):
        zero={'sensor_std':[0.]*8,'thermal_std':0.}
        _,_,m=self.event('thermal-only',normalizer=zero)
        self.assertFalse(m['accepted']);self.assertIn('zero_thermal_scale',m['rejection_reasons'])
        self.sensor[:,1]=50
        _,_,m=self.event('thermal-only')
        self.assertFalse(m['accepted']);self.assertIn('PM_nonnegative_order',m['base_constraint_errors'])
        self.assertFalse(m['introduced_constraint_errors'])

    def test_thermal_QC_flag_inherited_not_filtered(self):
        self.thermal[0,0,0]=-1
        _,_,m=self.event('thermal-only')
        self.assertTrue(m['accepted']);self.assertIn('thermal_negative_C',m['base_QC_flags'])

    def test_coupled_moves_NTC_CT_and_grows_thermal_in_time(self):
        s,t,m=self.event('coupled')
        self.assertEqual(m['parameters']['sensor_channels'],['NTC','CT1'])
        self.assertTrue(np.any(s[:,0]!=self.sensor[:,0]))
        self.assertTrue(np.any(s[:,4]!=self.sensor[:,4]))
        np.testing.assert_array_equal(s[:,1:4],self.sensor[:,1:4])
        p=m['parameters'];start=p['start']+p['lag'];duration=p['duration'];cy,cx=p['center_yx']
        delta=t[start:start+duration,cy,cx]-self.thermal[start:start+duration,cy,cx]
        np.testing.assert_allclose(delta,np.linspace(0,4,duration),atol=1e-14)


if __name__ == '__main__': unittest.main()
