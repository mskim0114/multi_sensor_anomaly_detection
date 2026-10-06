"""Independent numeric oracles and failure gates for the separate S/F pilot.

Every array and model in these tests is artificial. No project dataset, cache,
calibration or held-out window is opened. CPU is a fixture device only.
"""
import argparse
import ast
from collections import Counter
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('_test_matched_lr',ROOT/'src/train_anomaly_matched_lr.py')
ml=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=ml
spec.loader.exec_module(ml)
spec2=importlib.util.spec_from_file_location('_matched_lr_artificial',ROOT/'tests/test_anomaly_bin_stability.py')
aux=importlib.util.module_from_spec(spec2)
sys.modules[spec2.name]=aux
spec2.loader.exec_module(aux)
try:
    import torch
except ImportError:
    torch=None
requires_torch=unittest.skipUnless(torch is not None,'SERVER-TRAINING torch required')
requires_cuda=unittest.skipUnless(torch is not None and torch.cuda.is_available(),'SERVER-TRAINING CUDA required')


def write_signed(path,value):
    return aux.write_signed(path,value)


def plan_recipe():
    return {**copy.deepcopy(ml.FROZEN_VALUES),'reference_epochs':[7,30],
            'branch_observation_policy':{
                'permutation':'fixed_cyclic_shift_by_one_over_given_example_order_no_rng',
                'representation_std_collapse_max':1e-6,'permutation_max_logit_delta_negligible_max':1e-6,
                'zero_effect_alone_is_not_learned_signal':True,'S_thermal_diagnostics':'not_applicable',
                'dev_selection_is_exploratory_and_biased':True}}


class MetadataTests(unittest.TestCase):
    def make_gate(self,root,task='T-dev-reference'):
        inputs=aux.artificial_inputs()
        original={'sha256':'b'*64,'runtime_reference':dict(aux.aux.ARTIFICIAL_RUNTIME),
                  'input_file_sha256':{'fixture':'c'*64},'health_probes':[]}
        inputs.update(plan=original,evidence={'artificial':True},probe_pairs=())
        thermal_dir=root/'T';thermal_dir.mkdir()
        history=[];states=[]
        for epoch in range(1,31):
            state=thermal_dir/f'epoch-{epoch:02}.pt'
            state.write_bytes(f'NOT A CHECKPOINT: artificial metadata {epoch}'.encode())
            digest=hashlib.sha256(str(epoch).encode()).hexdigest()
            history.append({'epoch':epoch,'dev_loss':0.6 if epoch==7 else 0.75,
                            'fit_order_manifest_sha256':digest,'cumulative_fit_example_stream_sha256':digest,
                            'terminal_state_tensor_sha256':digest})
            states.append({'epoch':epoch,'file':state.name,'file_sha256':ml.p1b.file_sha(state),'tensor_sha256':digest})
        (thermal_dir/'epochs.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in history))
        (thermal_dir/'provenance.json').write_text('{"artificial":true}\n')
        thermal={'schema':ml.st.SCHEMA,'gate':'PASS','arm':'T','objective':'BIN','initial_lr':1e-4,
                 'epochs_completed':30,'real_data_optimizer_steps':7830,'source_files':ml.st.source_hashes(),
                 'plan_payload_sha256':original['sha256'],'input_evidence':inputs['evidence'],
                 'runtime_identity':original['runtime_reference'],'epochs':history,'terminal_states':states,
                 'actual_fit_event_stream_sha256':history[-1]['cumulative_fit_example_stream_sha256'],
                 'artifact_file_sha256':{'epochs.jsonl':ml.p1b.file_sha(thermal_dir/'epochs.jsonl')},
                 'provenance_file_sha256':ml.p1b.file_sha(thermal_dir/'provenance.json'),
                 'dev_weight_manifest':ml.bs.dev_weight_manifest(inputs['indexes']['dev']),
                 'initial_manifest':{'initial_state_sha256':{'S':'1'*64,'F':'2'*64},
                     'shared_branch_sha256':{'sensor':'3'*64,'thermal':'4'*64}}}
        thermal_sha,thermal=write_signed(thermal_dir/'result.json',thermal)
        expected=sum(isinstance(n,ast.FunctionDef) and n.name.startswith('test_') for p in ml.TEST_FILES
                     for n in ast.walk(ast.parse((ROOT/p).read_text())))
        fixture={'schema':ml.FIXTURE_SCHEMA,'gate':'PASS','source_files':ml.source_hashes(),
                 'test_files':list(ml.TEST_FILES),'unit_tests':dict(tests_run=expected,failures=0,errors=0,skipped=0),
                 'fixture_optimizer_steps':1,'real_data_optimizer_steps':0,
                 'runtime_identity':original['runtime_reference'],'git_commit_sha':'a'*40}
        fixture_sha,fixture=write_signed(root/'fixtures.json',fixture)
        plan={**plan_recipe(),'candidate_source_files':ml.source_hashes(),
              'original_stability_source_files':ml.st.source_hashes(),
              'fixture_evidence':dict(file_sha256=fixture_sha,payload_sha256=fixture['sha256']),
              **{k:original[k] for k in ('runtime_reference','input_file_sha256','health_probes')},
              'T_result_file_sha256':thermal_sha,'T_result_payload_sha256':thermal['sha256'],
              'T_epochs_file_sha256':ml.p1b.file_sha(thermal_dir/'epochs.jsonl'),
              'epoch_order_reference':ml.epoch_order_reference(thermal),
              'initial_model_state_tensor_sha256':thermal['initial_manifest']['initial_state_sha256'],
              'shared_branch_tensor_sha256':thermal['initial_manifest']['shared_branch_sha256']}
        plan_sha,plan=write_signed(root/'plan.json',plan)
        args=argparse.Namespace(task=task,plan=root/'plan.json',plan_sha256=plan_sha,
            stability_plan=root/'stability.json',stability_plan_sha256='9'*64,
            thermal_result=thermal_dir/'result.json',thermal_result_sha256=thermal_sha,
            comparison_fixtures=root/'fixtures.json',comparison_fixtures_sha256=fixture_sha,
            reference_predictions=None,reference_predictions_sha256=None,sensor_result=None,sensor_result_sha256=None)
        return inputs,args,plan,fixture,thermal

    def add_predictions(self,args,plan,thermal):
        root=args.plan.parent/'predictions';root.mkdir()
        predictions=[];artifacts={}
        for epoch in (7,30):
            path=root/f'dev-predictions-epoch-{epoch:02}.jsonl';path.write_text('{"artificial":true}\n')
            state=thermal['terminal_states'][epoch-1]
            artifacts[path.name]=ml.p1b.file_sha(path)
            predictions.append({'epoch':epoch,'gate':'PASS_EXACT_NATIVE_DEV_BCE',
                'dev_loss':thermal['epochs'][epoch-1]['dev_loss'],'state_tensor_sha256':state['tensor_sha256'],
                'state_file_sha256':state['file_sha256'],'file':path.name,'file_sha256':artifacts[path.name]})
        (root/'provenance.json').write_text('{"artificial":true}\n')
        record={'schema':ml.SCHEMA,'gate':'PASS','task':'T-dev-reference','source_files':ml.source_hashes(),
                'plan_payload_sha256':plan['sha256'],'real_data_optimizer_steps':0,'runtime_identity':plan['runtime_reference'],
                'T_result_payload_sha256':thermal['sha256'],'actual_roles_read':['dev'],'reference_epochs':[7,30],
                'micro_batch':2,'dev_weight_manifest':thermal['dev_weight_manifest'],'predictions':predictions,
                'artifact_file_sha256':artifacts,'provenance_file_sha256':ml.p1b.file_sha(root/'provenance.json'),
                **{k:False for k in ('calibration_window_read','heldout_window_read','threshold_generation','paper_result_eligible')}}
        args.reference_predictions=root/'result.json'
        args.reference_predictions_sha256,record=write_signed(args.reference_predictions,record)
        return record

    def validate(self,inputs,args):
        with mock.patch.object(ml.st,'validate_inputs',side_effect=lambda a:copy.deepcopy(inputs)):
            return ml.validate_inputs(args)

    def test_valid_gate_is_metadata_only_and_binds_all26_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            inputs,args,_,_,_=self.make_gate(Path(tmp))
            with mock.patch.object(ml.p1b,'_training',side_effect=AssertionError('no torch or tensor load allowed')):
                result=self.validate(inputs,args)
            self.assertEqual(len(result['candidate_source_files']),26)
            self.assertEqual(len(result['epoch_order_reference']),30)

    def test_recipe_budget_and_branch_policy_cannot_drift(self):
        changes={'initial_lr':1e-3,'epochs':True,'epochs2':29,'micro_batch':1,'training_seed':43,
                 'retry':True,'resume':True,'execution_sequence':['F','S'],'paper_result_eligible':True,
                 'probe_steps':[0,7830],'branch_rules_selected_after_T_inspection':False,
                 'epoch7_anchor_is_posthoc_T_selected':False,'reference_epochs':[30],
                 'branch_observation_policy':{}}
        for key,value in changes.items():
            with self.subTest(key=key),self.assertRaises(ValueError):
                ml.check_recipe({**plan_recipe(),('epochs' if key=='epochs2' else key):value})

    def test_cli_exposes_no_training_budget_or_cpu_fallback(self):
        flags={option for action in ml.parser()._actions for option in action.option_strings}
        self.assertFalse(flags.intersection({'--lr','--epochs','--resume','--retry','--device','--cpu','--micro-batch'}))
        self.assertIn('--stability-plan',flags)

    def test_old_validator_receives_original_plan_instead_of_comparison_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            inputs,args,_,_,_=self.make_gate(Path(tmp))
            with mock.patch.object(ml.st,'validate_inputs',return_value=inputs) as old:
                ml.validate_inputs(args)
            delegated=old.call_args.args[0]
            self.assertEqual(delegated.plan,args.stability_plan)
            self.assertEqual(delegated.plan_sha256,args.stability_plan_sha256)

    def test_foreign_skipped_or_real_data_fixture_is_rejected(self):
        for change in ('source','skip','real','count','runtime'):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as tmp:
                inputs,args,plan,fixture,_=self.make_gate(Path(tmp))
                if change=='source':fixture['source_files'][ml.NEW_FILES[0]]='f'*64
                if change=='skip':fixture['unit_tests']['skipped']=1
                if change=='real':fixture['real_data_optimizer_steps']=1
                if change=='count':fixture['fixture_optimizer_steps']=True
                if change=='runtime':fixture['runtime_identity']={}
                args.comparison_fixtures_sha256,fixture=write_signed(args.comparison_fixtures,fixture)
                plan['fixture_evidence']=dict(file_sha256=args.comparison_fixtures_sha256,payload_sha256=fixture['sha256'])
                args.plan_sha256,_=write_signed(args.plan,plan)
                with self.assertRaisesRegex(ValueError,'source/fixture/input'):
                    self.validate(inputs,args)

    def test_source_input_runtime_and_order_pins_fail_closed(self):
        for key in ('candidate_source_files','original_stability_source_files','runtime_reference',
                    'input_file_sha256','health_probes','T_epochs_file_sha256','epoch_order_reference',
                    'initial_model_state_tensor_sha256','shared_branch_tensor_sha256'):
            with self.subTest(key=key),tempfile.TemporaryDirectory() as tmp:
                inputs,args,plan,_,_=self.make_gate(Path(tmp))
                plan[key]='f'*64 if key=='T_epochs_file_sha256' else {}
                args.plan_sha256,_=write_signed(args.plan,plan)
                with self.assertRaises(ValueError):self.validate(inputs,args)

    def test_T_history_terminal_and_provenance_corruption_is_rejected(self):
        for name in ('epochs.jsonl','epoch-30.pt','provenance.json'):
            with self.subTest(name=name),tempfile.TemporaryDirectory() as tmp:
                inputs,args,_,_,_=self.make_gate(Path(tmp))
                (args.thermal_result.parent/name).write_text('changed\n')
                with self.assertRaises((ValueError,json.JSONDecodeError)):self.validate(inputs,args)

    def test_T_terminal_list_cannot_be_missing_or_shuffled(self):
        for changed in ([],list(range(30,0,-1))):
            with self.subTest(changed=changed),tempfile.TemporaryDirectory() as tmp:
                inputs,args,plan,_,thermal=self.make_gate(Path(tmp))
                thermal['terminal_states']=[thermal['terminal_states'][i-1] for i in changed]
                args.thermal_result_sha256,thermal=write_signed(args.thermal_result,thermal)
                plan.update(T_result_file_sha256=args.thermal_result_sha256,T_result_payload_sha256=thermal['sha256'])
                args.plan_sha256,_=write_signed(args.plan,plan)
                with self.assertRaisesRegex(ValueError,'all30'):self.validate(inputs,args)

    def test_S_requires_T_native_prediction_success_and_exact_values(self):
        for change in ('missing','empty','dev_rounding','state','role','micro','artifact','provenance'):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as tmp:
                inputs,args,plan,_,thermal=self.make_gate(Path(tmp),'S')
                if change=='missing':
                    with self.assertRaisesRegex(ValueError,'require successful'):self.validate(inputs,args)
                    continue
                record=self.add_predictions(args,plan,thermal)
                if change=='empty':record['predictions']=[]
                if change=='dev_rounding':record['predictions'][0]['dev_loss']=math.nextafter(0.6,math.inf)
                if change=='state':record['predictions'][0]['state_tensor_sha256']='f'*64
                if change=='role':record['actual_roles_read']=['dev','heldout']
                if change=='micro':record['micro_batch']=1
                if change=='artifact':(args.reference_predictions.parent/record['predictions'][0]['file']).write_text('changed')
                if change=='provenance':args.reference_predictions.with_name('provenance.json').write_text('changed')
                args.reference_predictions_sha256,_=write_signed(args.reference_predictions,record)
                with self.assertRaises(ValueError):self.validate(inputs,args)

    def test_F_cannot_run_before_a_successful_complete_source_bound_S(self):
        with tempfile.TemporaryDirectory() as tmp:
            inputs,args,plan,_,thermal=self.make_gate(Path(tmp),'F')
            self.add_predictions(args,plan,thermal)
            with self.assertRaisesRegex(ValueError,'preceding successful S'):self.validate(inputs,args)
            args.sensor_result=Path(tmp)/'failed-S.json'
            args.sensor_result_sha256,_=write_signed(args.sensor_result,{'schema':ml.SCHEMA,'gate':'FAILED','arm':'S'})
            with self.assertRaisesRegex(ValueError,'sequence gate'):self.validate(inputs,args)

    def test_F_requires_artifact_verification_even_for_otherwise_valid_S(self):
        for verification in (None,{}, {'gate':'FAILED'}):
            with self.subTest(verification=verification),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);inputs,args,plan,_,thermal=self.make_gate(root,'F')
                reference=self.add_predictions(args,plan,thermal)
                sensor_dir=root/'S';sensor_dir.mkdir()
                (sensor_dir/'epochs.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in thermal['epochs']))
                (sensor_dir/'provenance.json').write_text('{"artificial":true}\n')
                sensor={'schema':ml.SCHEMA,'gate':'PASS','arm':'S','real_data_optimizer_steps':7830,
                    'epochs_completed':30,'source_files':ml.source_hashes(),'plan_payload_sha256':plan['sha256'],
                    'actual_fit_event_stream_sha256':thermal['actual_fit_event_stream_sha256'],
                    'runtime_identity':plan['runtime_reference'],'input_evidence':inputs['evidence'],
                    'initial_state_tensor_sha256':plan['initial_model_state_tensor_sha256']['S'],
                    'reference_prediction_payload_sha256':reference['sha256'],
                    'dev_weight_manifest':thermal['dev_weight_manifest'],
                    'epoch_order_gates':[{'epoch':e,'gate':'PASS_EXACT_T_ORDER_STREAM','before_dev_forward':True} for e in range(1,31)],
                    'epochs':copy.deepcopy(thermal['epochs']),'terminal_states':copy.deepcopy(thermal['terminal_states']),
                    'actual_roles_read':['fit','dev'],'artifact_verification':{'gate':'PASS_FULL_TRAINING_ARTIFACTS'},
                    'artifact_file_sha256':{'epochs.jsonl':ml.p1b.file_sha(sensor_dir/'epochs.jsonl')},
                    'provenance_file_sha256':ml.p1b.file_sha(sensor_dir/'provenance.json'),
                    **{k:False for k in ('calibration_window_read','heldout_window_read','threshold_generation','paper_result_eligible')}}
                args.sensor_result=sensor_dir/'result.json'
                args.sensor_result_sha256,_=write_signed(args.sensor_result,sensor)
                with mock.patch.object(ml.p1b,'_training',side_effect=AssertionError('metadata only')):
                    self.validate(inputs,args)  # Positive control before altering only the verification gate.
                    if verification is None:sensor.pop('artifact_verification')
                    else:sensor['artifact_verification']=verification
                    args.sensor_result_sha256,_=write_signed(args.sensor_result,sensor)
                    with self.assertRaisesRegex(ValueError,'sequence gate'):self.validate(inputs,args)


def toy_factory(arm,dropout=0.2):
    def factory(seed):
        class MeanEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer=torch.nn.Linear(1,128)
                self.drop=torch.nn.Dropout(dropout)
            def forward(self,x):
                return self.drop(torch.tanh(self.layer(x.mean(tuple(range(1,x.ndim))).unsqueeze(1))))
        class ToyBin(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.arm,self.objective=arm,'BIN'
                if arm in ('S','F'):self.sensor_encoder=MeanEncoder()
                if arm in ('T','F'):self.thermal_encoder=MeanEncoder()
                if arm=='F':self.fusion=torch.nn.Linear(256,128)
                self.head=torch.nn.Linear(128,1)
            def forward(self,**tensors):
                z=[getattr(self,k+'_encoder')(tensors[k]) for k in ml.p1b.detectors.ARM_MODALITIES[arm]]
                if arm=='F':hidden=self.fusion(torch.cat(z,-1))
                else:hidden=z[0]
                return {'logit':self.head(hidden).squeeze(-1)}
        torch.manual_seed(seed)
        return ToyBin(),{'fixture':'artificial mean encoder, not a project model'}
    return factory


@requires_torch
class NumericTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.inputs=aux.artificial_inputs(2)
        ml.p1b._training()

    def run_fixture(self,arm='S',**options):
        values=dict(arm=arm,device=torch.device('cpu'),scope='synthetic_fixture',epochs=3,
                    nominal_batch=4,micro_batch=2,probe_steps=(),model_factory=toy_factory(arm))
        values.update(options)
        return ml.train_loop(self.inputs,aux.ArtificialLoader(),**values)

    def oracle(self,arm):
        """Own AdamW/cosine/reduction, independent of candidate train_loop/evaluate_dev."""
        _,helpers,models=ml.p1b._training();helpers.enable_determinism()
        model,_=toy_factory(arm)(42);model.train()
        optimizer=torch.optim.AdamW(model.parameters(),lr=0.0001,betas=(0.9,0.999),eps=1e-8,weight_decay=0.01)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=30)
        sampler=ml.bs.BasePairedSampler(self.inputs['indexes']['fit'],42,2,30)
        loader=aux.ArtificialLoader();history=[];states=[]
        for epoch in range(1,4):
            model.train();terms=[];count=0
            for pairs in sampler.epoch_batches(epoch):
                refs=ml.p1b.paired_examples(pairs);optimizer.zero_grad(set_to_none=True);total=0
                for start in range(0,len(refs),2):
                    chunk=refs[start:start+2];loaded=[loader.load('fit',r) for r in chunk]
                    tensors={k:torch.from_numpy(np.stack([v[i] for v in loaded]))
                             for i,k in enumerate(('sensor','thermal')) if k in ml.p1b.detectors.ARM_MODALITIES[arm]}
                    loss=torch.nn.functional.binary_cross_entropy_with_logits(model(**tensors)['logit'],
                        torch.tensor([r.label for r in chunk],dtype=torch.float32))*len(chunk)/len(refs)
                    total+=float(loss.detach());loss.backward()
                optimizer.step();terms.append(total*len(refs));count+=len(refs)
            lr=scheduler.get_last_lr()[0];scheduler.step()
            states.append(models.state_tensor_sha256(model));dev=[];model.eval()
            refs=ml.bs.dev_examples(self.inputs['indexes']['dev'])
            with torch.no_grad():
                for start in range(0,len(refs),2):
                    chunk=refs[start:start+2];loaded=[loader.load('dev',r) for r in chunk]
                    tensors={k:torch.from_numpy(np.stack([v[i] for v in loaded]))
                             for i,k in enumerate(('sensor','thermal')) if k in ml.p1b.detectors.ARM_MODALITIES[arm]}
                    bce=torch.nn.functional.binary_cross_entropy_with_logits(model(**tensors)['logit'],
                        torch.tensor([r.label for r in chunk],dtype=torch.float32),reduction='none')
                    dev.append(float((bce.double()*torch.tensor([r.weight for r in chunk],dtype=torch.float64)).sum()))
            history.append({'train_loss':math.fsum(terms)/count,'dev_loss':math.fsum(dev),'lr_used_this_epoch':lr})
        return history,states

    def test_S_and_F_updates_equal_independent_optimizer_and_weighted_BCE_oracle(self):
        for arm in ('S','F'):
            with self.subTest(arm=arm):
                expected,states=self.oracle(arm);actual,_=self.run_fixture(arm)
                for reference,epoch,state in zip(expected,actual['epochs'],states):
                    self.assertEqual({k:epoch[k] for k in reference},reference)
                    self.assertEqual(epoch['terminal_state_tensor_sha256'],state)
                self.assertEqual(actual['real_data_optimizer_steps'],0)
                self.assertEqual(actual['fixture_optimizer_steps'],3)

    def test_each_epoch_gate_precedes_dev_and_preserves_failing_terminal(self):
        baseline,_=self.run_fixture(epochs=2)
        refs=ml.epoch_order_reference(baseline);refs[1]['fit_order_manifest_sha256']='f'*64
        terminal=[];dev_epochs=[];original=ml.evaluate_dev
        def evaluate(*args,**kwargs):
            dev_epochs.append(args[8]['epoch']);return original(*args,**kwargs)
        with mock.patch.object(ml,'evaluate_dev',side_effect=evaluate):
            with self.assertRaises(ml.p1b.P1bFailure) as caught:
                self.run_fixture(epochs=2,order_reference=refs,on_terminal=lambda s,e:terminal.append(e.copy()))
        self.assertEqual([e['epoch'] for e in terminal],[1,2])
        self.assertEqual(dev_epochs,[1])
        self.assertTrue(caught.exception.context['before_dev_forward'])

    def test_exact_all_epoch_streams_accept_and_current_stream_changes_fail(self):
        baseline,_=self.run_fixture()
        accepted,_=self.run_fixture(order_reference=ml.epoch_order_reference(baseline))
        self.assertEqual(len(accepted['epoch_order_gates']),3)
        bad=copy.deepcopy(accepted['epochs'][2]);bad['cumulative_fit_example_stream_sha256']='0'*64
        with self.assertRaises(ml.p1b.P1bFailure):ml.check_epoch_order(bad,ml.epoch_order_reference(baseline))

    def test_observers_cannot_change_dropout_updates_mode_rng_or_terminal_selection(self):
        no_observer,_=self.run_fixture()
        seen=[]
        def observer(model,step):
            seen.append((step,model.training))
            model(sensor=torch.ones(2,30,5))
            return {'native_output_responsiveness':'responsive_at_probe','groups':{'sensor-only':{}}}
        observed,_=self.run_fixture(probe_steps=(0,1,2,3),observer=observer)
        self.assertEqual([e['terminal_state_tensor_sha256'] for e in observed['epochs']],
                         [e['terminal_state_tensor_sha256'] for e in no_observer['epochs']])
        self.assertEqual(observed['selected_state_tensor_sha256'],no_observer['selected_state_tensor_sha256'])
        self.assertEqual(seen,[(i,False) for i in range(4)])

    def test_equal_dev_minimum_selects_earliest_and_keeps_every_terminal(self):
        terminals=[];selected=[];boundaries=[]
        with mock.patch.object(ml,'evaluate_dev',return_value=0.7):
            record,_=self.run_fixture(on_terminal=lambda s,e:terminals.append(e['epoch']),
                on_improvement=lambda s,e,v:selected.append(e),on_boundary=boundaries.append)
        self.assertEqual(selected,[1]);self.assertEqual(terminals,[1,2,3]);self.assertEqual(boundaries,[1,2,3])
        self.assertEqual(record['dev_checkpoint_ties'],[2,3])

    def test_initial_full_model_and_shared_branch_pins_are_checked_before_update(self):
        _,_,models=ml.p1b._training();arms,manifest=models.build_arm_set('BIN',seed=42)
        references={'initial_model_state_tensor_sha256':{a:models.state_tensor_sha256(arms[a]) for a in ('S','F')},
                    'shared_branch_tensor_sha256':manifest['shared_branch_sha256']}
        for arm in ('S','F'):
            self.assertEqual(ml.check_initial(arms[arm],manifest,arm,references),references['initial_model_state_tensor_sha256'][arm])
        with torch.no_grad():arms['S'].head.bias.add_(0.1)
        with self.assertRaises(ml.p1b.P1bFailure):ml.check_initial(arms['S'],manifest,'S',references)
        manifest=copy.deepcopy(manifest);manifest['shared_branch_sha256']['sensor']='f'*64
        with self.assertRaises(ml.p1b.P1bFailure):ml.check_initial(arms['F'],manifest,'F',references)

    def test_real_scope_rejects_budget_and_fixture_factory_before_training(self):
        with mock.patch.object(ml.p1b,'_training',side_effect=AssertionError('must fail before GPU setup')):
            with self.assertRaises(ValueError):self.run_fixture(scope='P1c_matched_lr_fit_dev')

    def test_S_native_probe_checks_visible_inputs_and_thermal_negative_control(self):
        pairs=ml.onset.resolve_probe_pairs(self.inputs,ml.onset.selected_probes(self.inputs))
        model,_=toy_factory('S')(42);model.train()
        # CPU toy dense GEMM produced a row-position-dependent 1 ulp difference
        # for identical inputs in r1. A single dyadic coefficient avoids that
        # fixture artifact. Exact toy replay is not evidence for real heads;
        # the separate CUDA test uses unmodified real S/F heads after updates.
        with torch.no_grad():
            model.head.weight.zero_();model.head.weight[0,0]=0.125;model.head.bias.zero_()
        state_before=ml.p1b._training()[2].state_tensor_sha256(model)
        measured=ml.observe_fit(model,aux.ArtificialLoader(),pairs,arm='S',device=torch.device('cpu'))
        self.assertEqual(measured['groups']['thermal-only']['contrast_BCE_gain'],0)
        self.assertEqual(measured['groups']['thermal-only']['max_abs_logit_gap'],0)
        self.assertGreater(measured['groups']['sensor-only']['max_abs_logit_gap'],0)
        self.assertEqual(measured['branch_controls']['branches']['thermal'],{'status':'not_applicable'})
        self.assertTrue(measured['branch_controls']['head_replay']['exact_zero'])
        thermal=measured['branch_family_controls']['sensor']['within_pair_swap']['thermal-only']
        self.assertTrue(thermal['exact_zero']);self.assertEqual(thermal['contrast_BCE_gain'],0)
        self.assertEqual(len(measured['unchanged_input_controls']),len(pairs))
        self.assertEqual(ml.p1b._training()[2].state_tensor_sha256(model),state_before);self.assertTrue(model.training)

    def test_F_pair_controls_change_only_the_swapped_branch_and_reject_hidden_leakage(self):
        model,_=toy_factory('F',dropout=0)(42);model.eval()
        sensor,thermal=torch.zeros(4,128),torch.zeros(4,128)
        sensor[1,0]=2.;thermal[3,0]=4.
        pairs=[SimpleNamespace(family='sensor-only'),SimpleNamespace(family='thermal-only')]
        with torch.no_grad():
            model.fusion.weight.zero_();model.fusion.bias.zero_();model.head.weight.zero_();model.head.bias.zero_()
            model.fusion.weight[0,0]=1.;model.fusion.weight[1,128]=1.
            model.head.weight[0,0]=0.5;model.head.weight[0,1]=0.25
            logits=model.head(model.fusion(torch.cat([sensor,thermal],-1))).squeeze(-1).tolist()
            result=ml.branch_family_controls(model,{'sensor':sensor,'thermal':thermal},logits,pairs,torch)
            self.assertTrue(result['sensor']['within_pair_swap']['thermal-only']['exact_zero'])
            self.assertTrue(result['thermal']['within_pair_swap']['sensor-only']['exact_zero'])
            self.assertEqual(result['sensor']['within_pair_swap']['sensor-only']['max_abs_native_logit_delta'],1.)
            self.assertEqual(result['thermal']['within_pair_swap']['thermal-only']['max_abs_native_logit_delta'],1.)
            same={'sensor':torch.ones(4,128),'thermal':torch.ones(4,128)}
            native=model.head(model.fusion(torch.cat(list(same.values()),-1))).squeeze(-1).tolist()
            identical=ml.branch_family_controls(model,same,native,pairs,torch)
            for branch in identical.values():
                for control in ('fixed_cyclic_shift','within_pair_swap'):
                    self.assertTrue(all(g['exact_zero'] for g in branch[control].values()))
            bad=thermal.clone();bad[1,0]=1.
            with self.assertRaisesRegex(ValueError,'unchanged native thermal'):
                ml.branch_family_controls(model,{'sensor':sensor,'thermal':bad},logits,pairs,torch)

    def test_S_native_thermal_control_rejects_output_leakage_before_head_replay(self):
        pairs=ml.onset.resolve_probe_pairs(self.inputs,ml.onset.selected_probes(self.inputs))
        model,_=toy_factory('S')(42)
        original=model.forward
        def corrupted(**values):
            output=original(**values)
            output['logit'][1]+=0.01
            return output
        model.forward=corrupted
        with self.assertRaisesRegex(ValueError,'S thermal-only native logit/contrast'):
            ml.observe_fit(model,aux.ArtificialLoader(),pairs,arm='S',device=torch.device('cpu'))

    def test_prediction_replay_uses_native_microbatch_BCE_without_rounding(self):
        model,_=toy_factory('S')(42)
        refs=ml.bs.dev_examples(self.inputs['indexes']['dev']);_,helpers,models=ml.p1b._training()
        rows=[];value=ml.evaluate_dev(model,aux.ArtificialLoader(),refs,torch.device('cpu'),torch,helpers,
                                    Counter(),Counter(),{},2,arm='S',on_prediction=rows.append)
        self.assertEqual(len(rows),len(refs));self.assertIn('sensor_changed',rows[-1])
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);path=out/'fixture.pt'
            state={k:v.detach().clone() for k,v in model.state_dict().items()}
            ml.st.save_state(path,state,7,selection='ARTIFICIAL_ONLY')
            source={'path':path,'file_sha256':ml.p1b.file_sha(path),'tensor_sha256':models.state_tensor_sha256(state),'dev_loss':value}
            with mock.patch.object(models,'build_arm_set',side_effect=lambda *a,**k:({'S':toy_factory('S')(42)[0]},{})), \
                    mock.patch.object(torch.optim.AdamW,'step',side_effect=AssertionError('predictions must never optimize')):
                reports=ml.predict_states(self.inputs,aux.ArtificialLoader(),arm='S',states=[(7,source)],out=out,device=torch.device('cpu'),progress={})
                self.assertEqual(reports[0]['dev_loss'],value)
                self.assertEqual([json.loads(l) for l in (out/reports[0]['file']).read_text().splitlines()],rows)
                bad=dict(source,dev_loss=math.nextafter(value,math.inf))
                with self.assertRaisesRegex(ml.p1b.P1bFailure,'differs exactly'):
                    ml.predict_states(self.inputs,aux.ArtificialLoader(),arm='S',states=[(7,bad)],out=out,device=torch.device('cpu'),progress={})

    def artifact_fixture(self,out):
        (out/'terminal-states').mkdir();terminals=[]
        def terminal(state,entry):
            path=out/'terminal-states'/f"epoch-{entry['epoch']:02}.pt"
            ml.st.save_state(path,state,entry['epoch'],selection='EPOCH_TERMINAL_NOT_DEV_SELECTED')
            terminals.append({'epoch':entry['epoch'],'file':str(path.relative_to(out)),
                              'file_sha256':ml.p1b.file_sha(path),'tensor_sha256':entry['terminal_state_tensor_sha256']})
        record,state=self.run_fixture(epochs=2,probe_steps=(0,1,2),
            observer=lambda m,s:{'native_output_responsiveness':'responsive_at_probe','groups':{}},
            on_terminal=terminal,on_epoch=lambda e:ml.onset._append_json(out/'epochs.jsonl',e),
            on_probe=lambda e:ml.onset._append_json(out/'probes.jsonl',e),
            on_step=lambda e:ml.onset._append_json(out/'steps.jsonl',e),
            on_improvement=lambda s,e,v:ml.st.save_state(out/'checkpoint.pt',s,e,selection='DEV_EARLIEST_MINIMUM',dev_loss=v))
        return record,state,terminals

    def test_training_artifact_reloads_match_selected_state_and_every_terminal(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);record,state,terminals=self.artifact_fixture(out)
            gate=ml.verify_training_artifacts(out,record,state,terminals,expected_epochs=2,expected_steps=2,expected_probe_steps=(0,1,2))
            self.assertEqual(gate['gate'],'PASS_FULL_TRAINING_ARTIFACTS')
            self.assertEqual(gate['terminal_states_reloaded'],2)
            self.assertEqual(gate['selected_state_tensor_sha256'],record['selected_state_tensor_sha256'])
            self.assertTrue(gate['before_result_PASS'])

    def test_selected_checkpoint_must_match_earliest_disk_minimum_and_its_terminal(self):
        # Internally consistent checkpoint/record corruption used to pass reload.
        for corruption in ('later_tie','worse_epoch','wrong_tensor'):
            with self.subTest(corruption=corruption),tempfile.TemporaryDirectory() as tmp:
                out=Path(tmp)
                with mock.patch.object(ml,'evaluate_dev',side_effect=(0.2,0.2) if corruption=='later_tie' else (0.2,0.3)):
                    record,state,terminals=self.artifact_fixture(out)
                self.assertEqual(record['dev_checkpoint_epoch'],1)
                ml.verify_training_artifacts(out,record,state,terminals,expected_epochs=2,expected_steps=2,expected_probe_steps=(0,1,2))
                replacement=torch.load(out/terminals[1]['file'],map_location='cpu',weights_only=True)['model_state_dict']
                self.assertNotEqual(ml.p1b._training()[2].state_tensor_sha256(state),
                                    ml.p1b._training()[2].state_tensor_sha256(replacement))
                epoch=1 if corruption=='wrong_tensor' else 2
                record.update(dev_checkpoint_epoch=epoch,dev_checkpoint_loss=record['epochs'][epoch-1]['dev_loss'],
                              selected_state_tensor_sha256=ml.p1b._training()[2].state_tensor_sha256(replacement))
                ml.st.save_state(out/'checkpoint.pt',replacement,epoch,selection='DEV_EARLIEST_MINIMUM',dev_loss=record['dev_checkpoint_loss'])
                with self.assertRaises(ValueError):
                    ml.verify_training_artifacts(out,record,replacement,terminals,expected_epochs=2,expected_steps=2,expected_probe_steps=(0,1,2))

    def test_corrupted_or_incomplete_actual_training_artifacts_fail_before_PASS(self):
        for corruption in ('history','checkpoint','terminal','probes','steps','budget','missing_terminal'):
            with self.subTest(corruption=corruption),tempfile.TemporaryDirectory() as tmp:
                out=Path(tmp);record,state,terminals=self.artifact_fixture(out)
                if corruption=='history':
                    history=[json.loads(l) for l in (out/'epochs.jsonl').read_text().splitlines()]
                    # Change a non-selection field: the independent argmin gate
                    # must not mask removal of the disk-vs-memory history check.
                    history[0]['train_loss']+=0.125
                    (out/'epochs.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in history))
                if corruption in ('checkpoint','terminal'):
                    path=out/'checkpoint.pt' if corruption=='checkpoint' else out/terminals[-1]['file']
                    data=torch.load(path,weights_only=True,map_location='cpu')
                    data['model_state_dict']['head.bias'].add_(0.125);torch.save(data,path)
                if corruption=='probes':
                    values=[json.loads(l) for l in (out/'probes.jsonl').read_text().splitlines()]
                    values[1]['classification']='false_health'
                    (out/'probes.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in values))
                if corruption=='steps':(out/'steps.jsonl').write_text('{}\n')
                if corruption=='budget':record['optimizer_steps_total']=1
                if corruption=='missing_terminal':terminals.pop()
                with self.assertRaises(ValueError):
                    ml.verify_training_artifacts(out,record,state,terminals,expected_epochs=2,expected_steps=2,expected_probe_steps=(0,1,2))

    @requires_cuda
    def test_real_S_F_cuda_native_probes_preserve_next_training_update(self):
        """Real project encoders on artificial arrays, observer on/off exact states."""
        _,_,models=ml.p1b._training();device=torch.device('cuda:0')
        pairs=ml.onset.resolve_probe_pairs(self.inputs,ml.onset.selected_probes(self.inputs))
        for arm in ('S','F'):
            def factory(seed):
                arms,manifest=models.build_arm_set('BIN',seed=seed)
                return arms[arm],manifest
            with self.subTest(arm=arm):
                baseline,_=self.run_fixture(arm,device=device,epochs=2,model_factory=factory)
                seen=[]
                def observer(model,step):
                    record=ml.observe_fit(model,aux.ArtificialLoader(),pairs,arm=arm,device=device)
                    seen.append(record);return record
                observed,_=self.run_fixture(arm,device=device,epochs=2,model_factory=factory,
                                           probe_steps=(0,1,2),observer=observer)
                self.assertEqual([e['terminal_state_tensor_sha256'] for e in observed['epochs']],
                                 [e['terminal_state_tensor_sha256'] for e in baseline['epochs']])
                self.assertTrue(all(e['branch_controls']['head_replay']['exact_zero'] for e in seen))
                if arm=='F':self.assertTrue(all(e['thermal_parity']['parity_sequence']['passed'] for e in seen))


class SummaryAndPersistenceTests(unittest.TestCase):
    def test_conditional_BCE_and_bounds_use_original_weights_and_explicit_mass(self):
        rows=[{'base_window_id':'b','device':'d','label':0,'family':None,'weight':0.5,'logit':0.,
               'native_float32_BCE':math.log(2),'sensor_changed':False,'thermal_changed':False,'strength':None},
              {'base_window_id':'b','device':'d','label':1,'family':'thermal-only','weight':0.2,'logit':0.,
               'native_float32_BCE':math.log(2),'sensor_changed':False,'thermal_changed':True,'strength':4},
              {'base_window_id':'b','device':'d','label':1,'family':'sensor-only','weight':0.3,'logit':0.,
               'native_float32_BCE':math.log(2),'sensor_changed':True,'thermal_changed':False,'strength':2}]
        s,t,f=(ml.prediction_summary(rows,a) for a in ('S','T','F'))
        self.assertEqual(s['original_weight_mass'],1.)
        self.assertEqual(s['normal_original_weighted_contribution'],0.5*math.log(2))
        thermal=s['families']['thermal-only'];self.assertEqual(thermal['conditional_weight_mass'],0.7)
        self.assertAlmostEqual(thermal['conditional_BCE'],math.log(2))
        sbound=0.5*math.log(0.7/0.5)+0.2*math.log(0.7/0.2)
        tbound=0.5*math.log(0.8/0.5)+0.3*math.log(0.8/0.3)
        self.assertAlmostEqual(s['unchanged_input_aggregate_lower_bound'],sbound)
        self.assertAlmostEqual(t['unchanged_input_aggregate_lower_bound'],tbound)
        self.assertAlmostEqual(thermal['conditional_unchanged_input_lower_bound'],sbound/0.7)
        self.assertIsNone(f['unchanged_input_aggregate_lower_bound'])
        self.assertIsNone(f['families']['thermal-only']['conditional_unchanged_input_lower_bound'])
        self.assertEqual(s['family_strength']['thermal-only:4']['conditional_weight_mass'],0.7)

    def test_probe_loader_rejects_signed_zero_dtype_shape_and_wrong_adjacency(self):
        pair=SimpleNamespace(event_id='e',family='thermal-only')
        normal=SimpleNamespace(label=0,base_window_id='b');positive=SimpleNamespace(label=1,base_window_id='b',event_id='e')
        for change in ('signed_zero','dtype','shape','adjacency'):
            a=np.zeros((2,3),np.float32);b=a.copy()
            if change=='signed_zero':b[0,0]=-0.
            if change=='dtype':b=b.astype(np.float64)
            if change=='shape':b=b.reshape(3,2)
            loader=SimpleNamespace(load=mock.Mock(side_effect=[(a,a),(b,a)]),begin_phase=lambda:None)
            checked=ml.ProbeControlLoader(loader,[pair],'S')
            with self.subTest(change=change),self.assertRaises(ValueError):
                if change!='adjacency':checked.load('fit',normal)
                checked.load('fit',positive)

    @requires_torch
    def test_failure_state_preserves_original_parity_error_even_when_save_fails(self):
        model,_=toy_factory('F')(42)
        for save_fail in (False,True):
            with tempfile.TemporaryDirectory() as tmp:
                error=ml.p1b.P1bFailure('frozen parity failed',{'last_microbatch_parity':{'max_abs':0.2}})
                patch=mock.patch.object(ml.st,'save_state',side_effect=OSError('full disk')) if save_fail else contextlib.nullcontext()
                with mock.patch.object(ml,'observe_fit',side_effect=error),patch:
                    with self.assertRaises(ml.p1b.P1bFailure) as caught:
                        ml.observe_and_preserve(model,9,Path(tmp),None,None,arm='F',device=torch.device('cpu'))
                self.assertIs(caught.exception,error)
                self.assertEqual(error.context['last_microbatch_parity']['max_abs'],0.2)
                if not save_fail:self.assertTrue((Path(tmp)/'failure-probe-state-step-0009.pt').is_file())
                else:self.assertIn('save failed',error.context['probe_failure_state'])

    def test_failure_marker_has_own_schema_and_keeps_partial_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);(out/'partial.json').write_text('{}')
            ml.record_failure(out,KeyboardInterrupt(),{'stage':'training','optimizer_steps_completed':2})
            failure=json.loads((out/'failure.json').read_text())
            self.assertEqual(failure['schema'],ml.SCHEMA);self.assertFalse(failure['run_usable'])
            self.assertEqual(failure['context']['optimizer_steps_completed'],2)
            self.assertIn('partial.json',failure['partial_files_preserved']);self.assertFalse((out/'result.json').exists())


class MainWiringTests(unittest.TestCase):
    def fake_main(self,root,task='T-dev-reference',prediction_error=None,broken_stdout=False,artifact_failure=False,
                  successful_training=False,prediction_tamper=False):
        out=root/'new-output';meta=root/'inputs';meta.mkdir()
        args=argparse.Namespace(task=task,out=out,plan=meta/'plan.json',stability_plan=meta/'stability.json',
            thermal_result=meta/'result.json',comparison_fixtures=meta/'fixtures.json',reference_predictions=None,
            sensor_result=None,plan_sha256='a'*64)
        sources={'fixture':'a'*64};runtime={'fixture':True}
        states=[{'file':f'epoch-{i:02}.pt','tensor_sha256':'b'*64,'file_sha256':'c'*64} for i in range(1,31)]
        inputs={'plan':{'sha256':'a'*64,'output_paths':{task:str(out)},'runtime_reference':runtime},
            'candidate_source_files':sources,'candidate_git_commit_sha':'a'*40,'comparison_fixture_payload_sha256':'d'*64,
            'thermal_reference':{'sha256':'e'*64,'terminal_states':states,'epochs':[{'dev_loss':0.7}]*30,'dev_weight_manifest':{}},
            'thermal_result_path':meta/'result.json','cache_path':meta/'cache','cache_manifest':{'sha256':'f'*64},
            'evidence':{'artificial':True},'epoch_order_reference':[{'epoch':i} for i in range(1,31)],'probe_pairs':(),
            'reference_predictions':{'sha256':'1'*64}}
        cache=SimpleNamespace(manifest={'sha256':'f'*64},reader=mock.Mock(),close=mock.Mock())
        attach=mock.Mock();attach.open.return_value=cache
        loader=SimpleNamespace(counts=Counter(),seconds=Counter())
        prediction=mock.Mock(return_value=[{'epoch':7},{'epoch':30}],side_effect=prediction_error)
        training=mock.Mock(side_effect=ml.p1b.P1bFailure('artificial abort',{'wire_check':True}))
        if artifact_failure:
            training=mock.Mock(return_value=({'artificial':True},None))
        if successful_training:
            # Only test main's wiring. Training, model serialization and prediction
            # are stubs; no tensor, dataset or optimizer is used in these paths.
            def complete_training(*a,**kw):
                epochs=[{'epoch':e,'dev_loss':0.7,'terminal_state_tensor_sha256':'b'*64} for e in range(1,31)]
                for entry in epochs:
                    kw['on_terminal'](None,entry);kw['on_epoch'](entry)
                return {'artificial':True,'real_data_optimizer_steps':0,'dev_checkpoint_epoch':7,'epochs':epochs},None
            training=mock.Mock(side_effect=complete_training)
            def verify_stub(*a,**kw):
                return {'gate':'PASS_FULL_TRAINING_ARTIFACTS',
                        'artifact_file_sha256':{'epochs.jsonl':ml.p1b.file_sha(out/'epochs.jsonl')}}
            def predict_stub(*a,**kw):
                if prediction_tamper:(out/'epochs.jsonl').write_text('{"changed_during_prediction":true}\n')
                return [{'epoch':7},{'epoch':30}]
            prediction=mock.Mock(side_effect=predict_stub)
        with contextlib.ExitStack() as stack:
            for obj,name,value in ((ml,'validate_inputs',mock.Mock(return_value=inputs)),(ml,'source_hashes',lambda:sources),
                (ml.p1b,'_check_output',lambda a:out),(ml.p1b,'runtime_environment',lambda:(SimpleNamespace(type='cuda'),{},'')),
                (ml.onset,'runtime_identity',lambda d:runtime),(ml.onset,'check_runtime_identity',lambda *a:{'gate':'PASS'}),
                (ml.st,'gpu_activity',lambda uuid:{}),(ml.p1b.rc.RawNormalCache,'attach',mock.Mock(return_value=attach)),
                (ml.p1b,'PairLoader',mock.Mock(return_value=loader)),(ml,'predict_states',prediction),(ml,'train_loop',training)):
                stack.enter_context(mock.patch.object(obj,name,value))
            runtime['gpu_uuid']='fixture'
            if artifact_failure:stack.enter_context(mock.patch.object(ml,'verify_training_artifacts',side_effect=ValueError('persisted checkpoint/history differs')))
            if successful_training:
                stack.enter_context(mock.patch.object(ml,'verify_training_artifacts',side_effect=verify_stub))
                stack.enter_context(mock.patch.object(ml.st,'save_state',side_effect=lambda path,*a,**kw:Path(path).write_bytes(b'artificial state sentinel')))
            stack.enter_context(mock.patch.object(ml,'parser',return_value=SimpleNamespace(parse_args=lambda a:args)))
            if broken_stdout:stack.enter_context(mock.patch('builtins.print',side_effect=BrokenPipeError('closed')))
            else:stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            error=None
            try:ml.main([])
            except BaseException as caught:error=caught
        return out,prediction,training,cache,error

    def test_T_reference_main_selects_only_saved7_and30_and_never_calls_train(self):
        with tempfile.TemporaryDirectory() as tmp:
            out,predict,train,cache,error=self.fake_main(Path(tmp))
            self.assertIsNone(error);train.assert_not_called();cache.close.assert_called_once()
            self.assertEqual(predict.call_args.kwargs['arm'],'T')
            self.assertEqual([e for e,s in predict.call_args.kwargs['states']],[7,30])
            record=json.loads((out/'result.json').read_text())
            self.assertEqual(record['actual_roles_read'],['dev']);self.assertEqual(record['real_data_optimizer_steps'],0)
            self.assertFalse((out/'failure.json').exists())

    def test_S_main_passes_observer_all_epoch_order_and_terminal_callbacks(self):
        with tempfile.TemporaryDirectory() as tmp:
            out,predict,train,cache,error=self.fake_main(Path(tmp),'S')
            self.assertIsInstance(error,ml.p1b.P1bFailure);predict.assert_not_called();cache.close.assert_called_once()
            options=train.call_args.kwargs
            self.assertEqual(options['scope'],'P1c_matched_lr_fit_dev');self.assertEqual(options['arm'],'S')
            self.assertEqual(len(options['order_reference']),30)
            for name in ('observer','on_probe','on_step','on_epoch','on_terminal','on_improvement','on_boundary'):
                self.assertTrue(callable(options[name]),name)
            self.assertFalse((out/'result.json').exists());self.assertTrue((out/'failure.json').exists())

    def test_prediction_failure_produces_only_FAILED_and_broken_success_pipe_keeps_only_PASS(self):
        for broken in (False,True):
            with self.subTest(broken=broken),tempfile.TemporaryDirectory() as tmp:
                out,_,_,_,error=self.fake_main(Path(tmp),prediction_error=None if broken else ValueError('exact BCE differs'),broken_stdout=broken)
                self.assertIsInstance(error,BrokenPipeError if broken else ValueError)
                self.assertEqual((out/'result.json').exists(),broken)
                self.assertEqual((out/'failure.json').exists(),not broken)

    def test_main_persisted_artifact_failure_prevents_prediction_and_PASS(self):
        with tempfile.TemporaryDirectory() as tmp:
            out,predict,train,cache,error=self.fake_main(Path(tmp),'S',artifact_failure=True)
            self.assertIsInstance(error,ValueError);self.assertIn('persisted checkpoint/history',str(error))
            predict.assert_not_called();cache.close.assert_called_once()
            self.assertFalse((out/'result.json').exists())
            failure=json.loads((out/'failure.json').read_text())
            self.assertEqual(failure['stage'],'artifact_verification')

    def test_main_rechecks_verified_training_file_after_prediction_before_PASS(self):
        for tamper in (False,True):
            with self.subTest(tamper=tamper),tempfile.TemporaryDirectory() as tmp:
                out,predict,train,cache,error=self.fake_main(Path(tmp),'S',successful_training=True,prediction_tamper=tamper)
                train.assert_called_once();predict.assert_called_once();cache.close.assert_called_once()
                self.assertEqual((out/'result.json').exists(),not tamper)
                self.assertEqual((out/'failure.json').exists(),tamper)
                if tamper:
                    self.assertIsInstance(error,ValueError);self.assertIn('changed during prediction/finalization',str(error))
                    self.assertEqual(json.loads((out/'failure.json').read_text())['stage'],'final_bindings')
                else:self.assertIsNone(error)


if __name__=='__main__':unittest.main()
