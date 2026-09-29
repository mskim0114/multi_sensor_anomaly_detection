"""Block and device clustering contracts independent of any score distribution."""
import importlib.util
from pathlib import Path
import sys
import unittest
from collections import Counter

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('uncertainty_test',ROOT/'src/evaluation/anomaly_uncertainty.py')
u=importlib.util.module_from_spec(spec);sys.modules[spec.name]=u;spec.loader.exec_module(u)


def fixture():
    normal=[];events=[]
    for device in ['agv01','agv02','oht01','oht02']:
        for i in range(20):
            base=f'{device}:{i}'
            normal.append(dict(base_window_id=base,device=device,session_id=device+'_s',start_index=i*10,scores={'M':float(i)}))
            for f in ['sensor-only','thermal-only','coupled']:
                for s in [1,2,4]:
                    events.append(dict(base_window_id=base,device=device,event_id=base+f+str(s),accepted=True,
                                       family=f,strength=s,scores={'M':float(i+s)}))
    return normal,events


class BlockPlanTests(unittest.TestCase):
    def test_replay_pairing_and_stratum_counts(self):
        n,e=fixture();p=u.make_resampling_plan(n,e,stride=10)
        self.assertEqual(p,u.make_resampling_plan(n,e,stride=10))
        self.assertEqual(Counter(d['source_device'][:3] for d in p['device_draws']),{'agv':2,'oht':2})
        self.assertEqual(len(p['normal_rows']),80);self.assertEqual(len(p['synthetic_rows']),720)
        counts=Counter(r['cluster'] for r in p['synthetic_rows']);self.assertEqual(set(counts.values()),{9})
        for row in p['synthetic_rows']:
            parent=p['normal_rows'][row['cluster']]
            self.assertEqual(e[row['source_index']]['base_window_id'],n[parent['source_index']]['base_window_id'])
            self.assertEqual(row['device'],parent['device'])
        self.assertNotEqual(p['sha256'],u.make_resampling_plan(n,e,stride=10,replicate=1)['sha256'])

    def test_repeated_equipment_draws_keep_separate_weighting_ids(self):
        n,e=fixture()
        repeated=None
        for rep in range(20):
            p=u.make_resampling_plan(n,e,stride=10,replicate=rep)
            if len({r['source_device'] for r in p['device_draws']})<4:
                repeated=p;break
        self.assertIsNotNone(repeated)
        self.assertEqual(len({r['resampled_device'] for r in repeated['device_draws']}),4)

    def test_normal_gaps_split_runs_and_short_runs_are_preserved(self):
        n,e=fixture();n=[r for r in n if r['start_index']!=100];bases={r['base_window_id'] for r in n};e=[r for r in e if r['base_window_id'] in bases]
        runs=u.grouped_normal_runs(n,stride=10)
        self.assertEqual([len(r) for r in runs['agv01']],[10,9])
        p=u.make_resampling_plan(n,e,stride=10,block_starts=12)
        self.assertEqual(len(p['short_runs']),8)
        self.assertEqual(len(p['normal_rows']),76)

    def test_role_or_mismatched_event_is_refused(self):
        n,e=fixture()
        with self.assertRaises(ValueError):u.make_resampling_plan(n,e,stride=10,role='heldout')
        e[0]['device']='oht99'
        with self.assertRaises(ValueError):u.make_resampling_plan(n,e,stride=10)

    def test_interval_probe_preserves_frozen_scores_and_is_not_full_run(self):
        import copy
        spec=importlib.util.spec_from_file_location('uncertainty_calibration_test',ROOT/'src/evaluation/anomaly_calibration.py')
        calibration=importlib.util.module_from_spec(spec);sys.modules[spec.name]=calibration;spec.loader.exec_module(calibration)
        n,e=fixture();original=copy.deepcopy((n,e))
        r=u.conditional_intervals(n,e,'M',calibration,stride=10,replicates=4)
        self.assertEqual(r['replicates_valid'],4)
        self.assertIsNotNone(r['percentile_interval'])
        self.assertFalse(r['planned_replicate_count_met'])
        self.assertFalse(r['retuned_primary_threshold'])
        self.assertEqual((n,e),original)

    def test_failed_cells_invalidate_the_interval(self):
        class FailingCalibration:
            @staticmethod
            def calibrate_nq(*args,**kwargs): return {'tau':1.}
            @staticmethod
            def calibrate_syn(*args,**kwargs): raise ValueError('missing cells')
        n,e=fixture();r=u.conditional_intervals(n,e,'M',FailingCalibration,stride=10,replicates=3)
        self.assertEqual(len(r['failed_replicates']),3)
        self.assertIsNone(r['percentile_interval'])

    def test_stride_grid_is_required_and_mismatch_fails(self):
        n,e=fixture()
        with self.assertRaises(TypeError):u.make_resampling_plan(n,e)
        with self.assertRaises(ValueError):u.make_resampling_plan(n,e,stride=30)
        for stride in [0,-1,True,10.5]:
            with self.assertRaises(ValueError):u.grouped_normal_runs(n,stride=stride)
        selected=[r for r in n if r['start_index']%30==10]
        runs=u.grouped_normal_runs(selected,stride=30,offset=10)
        self.assertEqual([len(x) for x in runs['agv01']],[7])

    def test_whole_circular_run_has_no_multiset_variation(self):
        n,e=fixture()
        resolution=u.run_resolution(n,stride=10,block_starts=20)
        self.assertEqual(resolution['whole_run_fixed_multiset_window_fraction'],1.)
        p=u.make_resampling_plan(n,e,stride=10,block_starts=20)
        for draw in p['device_draws']:
            scores=[n[r['source_index']]['scores']['M'] for r in p['normal_rows'] if r['device']==draw['resampled_device']]
            self.assertEqual(sorted(scores),list(range(20)))

    def test_alpha_forwarded_and_partial_valid_quantiles_are_not_CI(self):
        class PartialCalibration:
            seen=[]
            @classmethod
            def calibrate_nq(cls,*args,**kwargs):
                cls.seen.append(kwargs['alpha']);return {'tau':1.}
            @classmethod
            def calibrate_syn(cls,*args,**kwargs):
                if len(cls.seen)==2:raise ValueError('missing cells')
                return {'tau':2.,'degenerate_always_normal':False}
        n,e=fixture();r=u.conditional_intervals(n,e,'M',PartialCalibration,stride=10,alpha=.02,replicates=3)
        self.assertEqual(PartialCalibration.seen,[.02]*3)
        self.assertEqual(r['failure_fraction'],1/3)
        self.assertIsNone(r['percentile_interval'])
        self.assertEqual(r['valid_only_quantiles_diagnostic_not_CI']['SYN'],[2.,2.])

    def test_predeclared_subsets_keep_base_derivatives_and_do_not_retune(self):
        spec=importlib.util.spec_from_file_location('subset_calibration_test',ROOT/'src/evaluation/anomaly_calibration.py')
        calibration=importlib.util.module_from_spec(spec);sys.modules[spec.name]=calibration;spec.loader.exec_module(calibration)
        n,e=fixture();r=u.fixed_subset_sensitivity(n,e,'M',calibration,stride=10)
        self.assertEqual(len(r['scenarios']),7)
        self.assertFalse(r['retuned_primary_threshold'])
        for name,case in r['scenarios'].items():
            self.assertEqual(case['status'],'PASS',name)
            self.assertEqual(case['accepted_events'],case['normal_windows']*9)
            self.assertGreaterEqual(case['SYN']['tau'],case['NQ']['tau'])
            self.assertLessEqual(case['SYN']['achieved_synthetic_tpr'],case['synthetic_TPR_at_NQ']+1e-12)
        self.assertEqual(r['scenarios']['leave_out_agv01']['normal_windows'],60)
        self.assertEqual(r['scenarios']['stride30_offset20']['normal_windows'],24)


if __name__=='__main__':unittest.main()
