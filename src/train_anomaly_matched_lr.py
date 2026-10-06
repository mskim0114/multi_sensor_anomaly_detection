#!/usr/bin/env python3
"""Separately frozen S/F LR1e-4 comparison and immutable T-dev prediction replay.

This candidate preserves original source20 and completed R3 source22. S/F require
all30 order/stream gates before dev. The T task only reads saved states and dev
inputs; it never resumes or trains T. Full cal/heldout/threshold remain outside
this pilot. Each invocation has a new output, no retry, sweep or CPU fallback.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('_matched_lr_stability', ROOT / 'src/diagnose_anomaly_bin_stability.py')
st = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = st
spec.loader.exec_module(st)
onset, p1b, bs, sa = st.onset, st.p1b, st.bs, st.sa
NEW_FILES = ('src/train_anomaly_matched_lr.py', 'tests/test_anomaly_matched_lr.py',
             'src/models/anomaly_branch_controls.py', 'tests/test_anomaly_branch_controls.py')
SOURCE_FILES = (*st.SOURCE_FILES, *NEW_FILES)
TEST_FILES = (*st.TEST_FILES, *(p for p in NEW_FILES if p.startswith('tests/')))
SCHEMA = 'exp004-matched-lr-result-v1'
PLAN_SCHEMA = 'exp004-matched-lr-plan-v1'
FIXTURE_SCHEMA = 'exp004-matched-lr-fixtures-v1'
RECIPE = 'exp004-sfbin-lr1e4-controlled-v1'
PROBE_STEPS = st.PROBE_STEPS
FROZEN_VALUES = {'schema': PLAN_SCHEMA, 'recipe_id': RECIPE, 'status': 'FROZEN_MATCHED_LR',
                 'objective':'BIN','arms':['S','F'],'execution_sequence':['S','F'],
                 'fold':0,'ct_channel':'CT1','generator_seed':42,'training_seed':42,
                 'initial_lr':1e-4,'epochs':30,'steps_per_epoch':261,'effective_batch':16,'micro_batch':2,
                 'optimizer':dict(st.FROZEN_VALUES['optimizer']), 'scheduler':dict(st.FROZEN_VALUES['scheduler']),
                 'proposed_optimizer_steps_per_arm':7830,'proposed_new_optimizer_steps':15660,
                 'probe_steps':list(PROBE_STEPS),'probe_parity_tolerance':dict(st.PROBE_PARITY),
                 'epoch_order_gate_before_dev':True,'resume':False,'retry':False,'collapse_early_stop':False,
                 'actual_roles_read':['fit','dev'],'calibration_window_read':False,'heldout_window_read':False,
                 'threshold_generation':False,'paper_result_eligible':False,
                 'dev_report_epochs':[7,30],'epoch7_anchor_is_posthoc_T_selected':True,
                 'branch_rules_selected_after_T_inspection':True}


def source_hashes():
    return {p:p1b.file_sha(ROOT/p) for p in SOURCE_FILES}


def check_recipe(plan):
    for key,value in FROZEN_VALUES.items():
        if plan.get(key)!=value or type(plan.get(key)) is not type(value):
            raise ValueError('matched-LR frozen recipe differs: '+key)
    if plan.get('reference_epochs') != [7,30]:
        raise ValueError('immutable T reference must cover epoch7 and30')
    if plan.get('branch_observation_policy') != {
            'permutation':'fixed_cyclic_shift_by_one_over_given_example_order_no_rng',
            'representation_std_collapse_max':1e-6,'permutation_max_logit_delta_negligible_max':1e-6,
            'zero_effect_alone_is_not_learned_signal':True,
            'S_thermal_diagnostics':'not_applicable',
            'dev_selection_is_exploratory_and_biased':True}:
        raise ValueError('branch observation policy differs')


def epoch_order_reference(thermal):
    return [{k:e[k] for k in ('epoch','fit_order_manifest_sha256','cumulative_fit_example_stream_sha256')}
            for e in thermal['epochs']]


def check_epoch_order(entry, references):
    e=entry['epoch']
    if type(e) is not int or e<1 or e>len(references):
        raise ValueError('epoch order gate index out of range')
    expected=references[e-1]
    actual={k:entry[k] for k in expected}
    if actual!=expected:
        raise p1b.P1bFailure('matched-LR epoch order/stream differs from immutable T',
                             {'epoch':e,'actual':actual,'expected':expected,'before_dev_forward':True})
    return {'epoch':e,'gate':'PASS_EXACT_T_ORDER_STREAM','before_dev_forward':True}


def check_initial(model, manifest, arm, references):
    _,_,models=p1b._training()
    actual=models.state_tensor_sha256(model)
    if actual!=references['initial_model_state_tensor_sha256'][arm] or manifest['shared_branch_sha256']!=references['shared_branch_tensor_sha256']:
        raise p1b.P1bFailure('matched-LR initial model/shared branch mismatch',{'arm':arm,'actual_initial_state_tensor_sha256':actual})
    return actual


def validate_inputs(args):
    """File/metadata only: no torch, array attachment, model or GPU initialization."""
    old_args=SimpleNamespace(**vars(args))
    old_args.plan,old_args.plan_sha256=args.stability_plan,args.stability_plan_sha256
    inputs=st.validate_inputs(old_args)
    stability_plan=inputs['plan']
    plan=p1b._pinned(args.plan,args.plan_sha256,'matched-LR plan')
    check_recipe(plan)
    live=source_hashes()
    fixture=p1b._pinned(args.comparison_fixtures,args.comparison_fixtures_sha256,'matched-LR fixtures')
    expected=sum(isinstance(n,ast.FunctionDef) and n.name.startswith('test_') for p in TEST_FILES
                 for n in ast.walk(ast.parse((ROOT/p).read_text())))
    if (plan.get('candidate_source_files')!=live or fixture.get('source_files')!=live
            or fixture.get('schema')!=FIXTURE_SCHEMA or fixture.get('gate')!='PASS'
            or fixture.get('test_files')!=list(TEST_FILES)
            or fixture.get('unit_tests')!=dict(tests_run=expected,failures=0,errors=0,skipped=0)
            or type(fixture.get('fixture_optimizer_steps')) is not int or fixture['fixture_optimizer_steps']<1
            or fixture.get('real_data_optimizer_steps')!=0
            or fixture.get('runtime_identity')!=stability_plan['runtime_reference']
            or plan.get('fixture_evidence')!={'file_sha256':args.comparison_fixtures_sha256,'payload_sha256':fixture['sha256']}
            or plan.get('original_stability_source_files')!=st.source_hashes()
            or plan.get('runtime_reference')!=stability_plan['runtime_reference']
            or plan.get('input_file_sha256')!=stability_plan['input_file_sha256']
            or plan.get('health_probes')!=stability_plan['health_probes']):
        raise ValueError('matched-LR source/fixture/input binding differs')
    thermal=p1b._pinned(args.thermal_result,args.thermal_result_sha256,'completed T stability result')
    epoch_path=args.thermal_result.with_name('epochs.jsonl')
    history=[json.loads(l) for l in epoch_path.read_text().splitlines() if l.strip()]
    if (thermal.get('schema')!=st.SCHEMA or thermal.get('gate')!='PASS' or thermal.get('arm')!='T'
            or thermal.get('objective')!='BIN' or thermal.get('initial_lr')!=1e-4
            or thermal.get('epochs_completed')!=30 or thermal.get('real_data_optimizer_steps')!=7830
            or thermal.get('source_files')!=st.source_hashes()
            or thermal.get('plan_payload_sha256')!=stability_plan['sha256']
            or thermal.get('input_evidence')!=inputs['evidence']
            or thermal.get('runtime_identity')!=plan['runtime_reference']
            or history!=thermal['epochs'] or p1b.file_sha(epoch_path)!=thermal['artifact_file_sha256']['epochs.jsonl']
            or plan.get('T_result_file_sha256')!=args.thermal_result_sha256
            or plan.get('T_result_payload_sha256')!=thermal['sha256']
            or plan.get('T_epochs_file_sha256')!=p1b.file_sha(epoch_path)
            or plan.get('epoch_order_reference')!=epoch_order_reference(thermal)
            or plan.get('initial_model_state_tensor_sha256')!={a:thermal['initial_manifest']['initial_state_sha256'][a] for a in ('S','F')}
            or plan.get('shared_branch_tensor_sha256')!=thermal['initial_manifest']['shared_branch_sha256']):
        raise ValueError('immutable T result/history/initial/order reference differs')
    if ([e.get('epoch') for e in history]!=list(range(1,31))
            or [s.get('epoch') for s in thermal.get('terminal_states',[])]!=list(range(1,31))
            or any(s.get('tensor_sha256')!=history[i]['terminal_state_tensor_sha256']
                   for i,s in enumerate(thermal['terminal_states']))
            or thermal.get('actual_fit_event_stream_sha256')!=history[-1]['cumulative_fit_example_stream_sha256']):
        raise ValueError('immutable T must have ordered all30 epochs and terminal states')
    for item in thermal['terminal_states']:
        if p1b.file_sha(args.thermal_result.parent/item['file'])!=item['file_sha256']:
            raise ValueError('immutable T terminal artifact changed')
    if p1b.file_sha(args.thermal_result.with_name('provenance.json'))!=thermal['provenance_file_sha256']:
        raise ValueError('immutable T provenance artifact changed')
    reference_predictions=None
    if args.task!='T-dev-reference':
        if args.reference_predictions is None or args.reference_predictions_sha256 is None:
            raise ValueError('S/F require successful source-bound T native dev predictions')
        reference_predictions=p1b._pinned(args.reference_predictions,args.reference_predictions_sha256,'T native predictions')
        if (reference_predictions.get('schema')!=SCHEMA or reference_predictions.get('gate')!='PASS'
                or reference_predictions.get('task')!='T-dev-reference'
                or reference_predictions.get('source_files')!=live or reference_predictions.get('plan_payload_sha256')!=plan['sha256']
                or reference_predictions.get('real_data_optimizer_steps')!=0
                or reference_predictions.get('runtime_identity')!=plan['runtime_reference']
                or reference_predictions.get('T_result_payload_sha256')!=thermal['sha256']
                or reference_predictions.get('actual_roles_read')!=['dev']
                or reference_predictions.get('micro_batch')!=2
                or reference_predictions.get('dev_weight_manifest')!=thermal['dev_weight_manifest']
                or reference_predictions.get('reference_epochs')!=[7,30]
                or [r['epoch'] for r in reference_predictions.get('predictions',[])]!=[7,30]
                or any(r.get('dev_loss')!=thermal['epochs'][r['epoch']-1]['dev_loss']
                       or r.get('gate')!='PASS_EXACT_NATIVE_DEV_BCE'
                       or r.get('state_tensor_sha256')!=thermal['terminal_states'][r['epoch']-1]['tensor_sha256']
                       or r.get('state_file_sha256')!=thermal['terminal_states'][r['epoch']-1]['file_sha256']
                       or r.get('file_sha256')!=reference_predictions.get('artifact_file_sha256',{}).get(r.get('file'))
                       for r in reference_predictions['predictions'])
                or any(reference_predictions.get(k) is not False for k in
                       ('calibration_window_read','heldout_window_read','threshold_generation','paper_result_eligible'))):
            raise ValueError('T native dev prediction replay gate differs')
        for name,h in reference_predictions['artifact_file_sha256'].items():
            if p1b.file_sha(args.reference_predictions.parent/name)!=h:
                raise ValueError('T prediction artifact changed')
        if p1b.file_sha(args.reference_predictions.with_name('provenance.json'))!=reference_predictions['provenance_file_sha256']:
            raise ValueError('T prediction provenance changed')
    if args.task=='F':
        if args.sensor_result is None or args.sensor_result_sha256 is None:
            raise ValueError('F requires preceding successful S run')
        sensor=p1b._pinned(args.sensor_result,args.sensor_result_sha256,'preceding S result')
        if (sensor.get('schema')!=SCHEMA or sensor.get('gate')!='PASS' or sensor.get('arm')!='S'
                or sensor.get('real_data_optimizer_steps')!=7830 or sensor.get('epochs_completed')!=30
                or sensor.get('source_files')!=live or sensor.get('plan_payload_sha256')!=plan['sha256']
                or sensor.get('actual_fit_event_stream_sha256')!=thermal['actual_fit_event_stream_sha256']
                or sensor.get('runtime_identity')!=plan['runtime_reference']
                or sensor.get('input_evidence')!=inputs['evidence']
                or sensor.get('initial_state_tensor_sha256')!=plan['initial_model_state_tensor_sha256']['S']
                or sensor.get('reference_prediction_payload_sha256')!=reference_predictions['sha256']
                or sensor.get('dev_weight_manifest')!=thermal['dev_weight_manifest']
                or sensor.get('epoch_order_gates')!=[{'epoch':e,'gate':'PASS_EXACT_T_ORDER_STREAM','before_dev_forward':True} for e in range(1,31)]
                or [e.get('epoch') for e in sensor.get('epochs',[])]!=list(range(1,31))
                or epoch_order_reference(sensor)!=plan['epoch_order_reference']
                or [e.get('epoch') for e in sensor.get('terminal_states',[])]!=list(range(1,31))
                or sensor.get('actual_roles_read')!=['fit','dev']
                or sensor.get('artifact_verification',{}).get('gate')!='PASS_FULL_TRAINING_ARTIFACTS'
                or any(sensor.get(k) is not False for k in
                       ('calibration_window_read','heldout_window_read','threshold_generation','paper_result_eligible'))):
            raise ValueError('preceding S run does not satisfy sequence gate')
        for name,h in sensor['artifact_file_sha256'].items():
            if p1b.file_sha(args.sensor_result.parent/name)!=h:
                raise ValueError('preceding S artifact changed')
        if p1b.file_sha(args.sensor_result.with_name('provenance.json'))!=sensor['provenance_file_sha256']:
            raise ValueError('preceding S provenance changed')
    inputs.update(plan=plan,candidate_source_files=live,thermal_reference=thermal,
                  thermal_result_path=args.thermal_result,epoch_order_reference=plan['epoch_order_reference'],
                  initialization_reference=plan,comparison_fixture_payload_sha256=fixture['sha256'],
                  candidate_git_commit_sha=fixture['git_commit_sha'],reference_predictions=reference_predictions)
    return inputs


def _tensors(loader,role,refs,arm,device,th,helpers,timing):
    pairs=[loader.load(role,ref) for ref in refs]
    tick=time.perf_counter()
    arrays={name:np.stack([p[i] for p in pairs]) for i,name in enumerate(('sensor','thermal'))
            if name in p1b.detectors.ARM_MODALITIES[arm]}
    timing[role+'_stack']+=time.perf_counter()-tick
    tick=time.perf_counter()
    tensors={name:th.from_numpy(a).to(device=device,dtype=th.float32) for name,a in arrays.items()}
    helpers._sync(device)
    timing[role+'_transfer']+=time.perf_counter()-tick
    return tensors


class ProbeControlLoader:
    """Check unchanged visible inputs from actual normalized pairs, by bytes."""
    def __init__(self,loader,pairs,arm):
        self.loader,self.arm=loader,arm
        self.by_event={p.event_id:p for p in pairs}
        self.previous=None; self.rows=[]
    def begin_phase(self):
        self.previous=None; self.loader.begin_phase()
    def load(self,role,ref):
        if role!='fit':
            raise ValueError('fixed probes must read fit only')
        pair=self.loader.load(role,ref)
        if any(np.asarray(a).dtype!=np.dtype('float32') for a in pair):
            raise ValueError('native normalized probe arrays must be float32')
        if ref.label==0:
            self.previous=(ref.base_window_id,pair)
        else:
            if self.previous is None or self.previous[0]!=ref.base_window_id:
                raise ValueError('probe positive is not preceded by its Normal')
            frozen=self.by_event[ref.event_id]
            normal=self.previous[1]
            for index,name in enumerate(('sensor','thermal')):
                if (name=='sensor' and frozen.family=='thermal-only') or (name=='thermal' and frozen.family=='sensor-only'):
                    a,b=np.ascontiguousarray(normal[index]),np.ascontiguousarray(pair[index])
                    if a.dtype!=b.dtype or a.shape!=b.shape or a.tobytes()!=b.tobytes():
                        raise ValueError('unchanged probe '+name+' input differs')
                    self.rows.append({'base_window_id':ref.base_window_id,'event_id':ref.event_id,'modality':name,
                                      'dtype':str(a.dtype),'shape':list(a.shape),'normal_sha256':hashlib.sha256(a.tobytes()).hexdigest(),
                                      'positive_sha256':hashlib.sha256(b.tobytes()).hexdigest()})
            self.previous=None
        return pair


def branch_family_controls(model,features,logits,pairs,th):
    """Deterministic head-only pair substitution and cyclic permutation by family.

    The other branch remains native. No encoder runs or RNG are used. These are
    diagnostics on trained fit events, not measurements of held-out benefit.
    """
    def replay(values):
        chunks=[]
        for start in range(0,len(logits),2):
            part={k:z[start:start+2] for k,z in values.items()}
            z=(model.fusion(th.cat([part['sensor'],part['thermal']],-1))
               if model.arm=='F' else next(iter(part.values())))
            chunks.extend(model.head(z).squeeze(-1).detach().cpu().tolist())
        if not np.isfinite(chunks).all():
            raise ValueError('nonfinite branch family replay')
        return np.asarray(chunks,dtype=np.float64)
    native=np.asarray(logits,dtype=np.float64)
    record={}
    for branch,z in features.items():
        unchanged='thermal-only' if branch=='sensor' else 'sensor-only'
        indices=[i for i,p in enumerate(pairs) if p.family==unchanged]
        if any(not th.equal(z[2*i],z[2*i+1]) for i in indices):
            raise ValueError('unchanged native '+branch+' branch features differ')
        branch_record={'unchanged_input_native_z_pairs_exact':len(indices)}
        controls={'fixed_cyclic_shift':th.roll(z,shifts=1,dims=0),
                  'within_pair_swap':z.reshape(-1,2,z.shape[1]).flip(1).reshape_as(z)}
        for name,replacement in controls.items():
            changed=replay(dict(features,**{branch:replacement}))
            groups={}
            for family in ('sensor-only','thermal-only'):
                ids=np.array([i for i,p in enumerate(pairs) if p.family==family],dtype=int)
                example_ids=np.column_stack((ids*2,ids*2+1)).reshape(-1)
                delta=np.abs(changed[example_ids]-native[example_ids])
                groups[family]={'max_abs_native_logit_delta':float(delta.max()),
                    'mean_abs_native_logit_delta':float(delta.mean()),'exact_zero':bool((delta==0).all()),
                    **st.contrast_metrics(changed[2*ids],changed[2*ids+1])}
            branch_record[name]=groups
        record[branch]=branch_record
    return record


def observe_fit(model,loader,pairs,*,arm,device):
    th,helpers,_=p1b._training()
    controls=p1b._leaf('_matched_lr_branch_controls','src/models/anomaly_branch_controls.py')
    checked=ProbeControlLoader(loader,pairs,arm)
    checked.begin_phase()
    examples=p1b.paired_examples(pairs)
    logits=[]; features={'sensor':[],'thermal':[]}; parity_parts=[]
    timing=Counter()
    with onset.observation_mode(model,th):
        for start in range(0,len(examples),2):
            tensors=_tensors(checked,'fit',examples[start:start+2],arm,device,th,helpers,timing)
            captured={}; handles=[]
            for name in p1b.detectors.ARM_MODALITIES[arm]:
                handles.append(getattr(model,name+'_encoder').register_forward_hook(
                    lambda module,args,output,n=name:captured.__setitem__(n,output.detach().clone())))
            try:
                output=model(**tensors)['logit']
            finally:
                for handle in handles:handle.remove()
            if output.shape!=(len(examples[start:start+2]),) or not th.isfinite(output).all():
                raise ValueError('probe native logit shape/nonfinite mismatch')
            logits.extend(output.cpu().tolist())
            for name,z in captured.items():
                if z.shape!=(len(output),128) or z.dtype!=th.float32 or not th.isfinite(z).all():
                    raise ValueError('probe native branch z shape/nonfinite mismatch')
                features[name].extend(z.cpu().tolist())
            if arm=='F':
                diag=p1b._leaf(*st.DIAGNOSTICS_MODULE)
                measured=diag.thermal_lstm_diagnostics(model,tensors['thermal'],**st.PROBE_PARITY)
                if any(measured[p]['passed'] is not True for p in ('parity_sequence','parity_encoder_output')):
                    raise p1b.P1bFailure('probe thermal parity exceeded frozen tolerance',
                        {'probe_parity_tolerance':dict(st.PROBE_PARITY),
                         'last_microbatch_parity':{p:measured[p] for p in ('parity_sequence','parity_encoder_output')}})
                parity_parts.append(measured)
        groups={}
        for family in ('sensor-only','thermal-only'):
            ids=[i for i,p in enumerate(pairs) if p.family==family]
            n=np.array([logits[2*i] for i in ids],dtype=np.float64)
            p=np.array([logits[2*i+1] for i in ids],dtype=np.float64)
            groups[family]={**st.contrast_metrics(n,p),'pair_count':len(ids),
                            'max_abs_logit_gap':float(np.abs(p-n).max()),
                            'mean_abs_logit_gap':float(np.abs(p-n).mean()),
                            'positive_higher_logit_pairs':int((p>n).sum())}
            invisible=(arm=='S' and family=='thermal-only')
            if invisible and (np.any(p!=n) or groups[family]['contrast_BCE_gain']!=0):
                raise ValueError('S thermal-only native logit/contrast negative control failed')
        branch_kwargs={name+'_z':th.tensor(z,dtype=th.float32,device=device) for name,z in features.items() if z}
        branch=controls.analyze_branch_controls(model,th.tensor(logits,dtype=th.float32,device=device),
                                               micro_batch=2,**branch_kwargs)
        by_family=branch_family_controls(model,{k[:-2]:v for k,v in branch_kwargs.items()},logits,pairs,th)
        contribution=None
        if arm=='F':
            sensor=branch['branches']['sensor']['permutation_logit_delta']['max_abs']
            thermal=branch['branches']['thermal']['permutation_logit_delta']['max_abs']
            contribution={'thermal_to_sensor_permutation_max_abs_ratio':thermal/sensor if sensor else None,
                          'sensor_denominator_max_abs':sensor,'thermal_numerator_max_abs':thermal,
                          'scope':'descriptive fit-probe dependence; a false near-zero category is not evidence of branch health'}
    parity=None if arm=='S' else {k:diag.combine_diagnostics(parity_parts)[k] for k in ('parity_sequence','parity_encoder_output')}
    return {'arm':arm,'pair_count':len(pairs),'logits_normal_then_positive':logits,
            'native_branch_z':{k:v for k,v in features.items() if v},'groups':groups,'branch_controls':branch,
            'branch_family_controls':by_family,
            'relative_branch_contribution':contribution,
            'thermal_parity':{'status':'not_applicable'} if parity is None else parity,
            'unchanged_input_controls':checked.rows,
            'native_output_responsiveness':'responsive_at_probe' if max(g['max_abs_logit_gap'] for g in groups.values())>1e-6 else 'low_response_at_probe',
            'scope':'fit-only trained events; output response is not branch health or generalization'}


def observe_and_preserve(model,step,out,loader,pairs,*,arm,device):
    try:
        return observe_fit(model,loader,pairs,arm=arm,device=device)
    except Exception as error:
        context=dict(getattr(error,'context',{}))
        try:
            state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            path=Path(out)/f'failure-probe-state-step-{step:04}.pt'
            st.save_state(path,state,None,selection='PROBE_FAILURE_STATE_NOT_DEV_SELECTED',optimizer_step=step)
            context['probe_failure_state']=str(path.relative_to(out))
        except Exception as save_error:
            context['probe_failure_state']='save failed: '+repr(save_error)
        if isinstance(error,p1b.P1bFailure):
            error.context.update(context); raise
        raise p1b.P1bFailure(str(error),context) from error


def evaluate_dev(model,loader,refs,device,th,helpers,timing,exposures,progress,micro_batch,*,arm,on_prediction=None):
    terms=[]
    loader.begin_phase()
    with onset.observation_mode(model,th):
        for position in range(0,len(refs),micro_batch):
            chunk=refs[position:position+micro_batch]
            progress.update(phase='dev',dev_position=position)
            loader.context=dict(progress)
            tensors=_tensors(loader,'dev',chunk,arm,device,th,helpers,timing)
            tick=time.perf_counter()
            logits=model(**tensors)['logit']
            labels=th.tensor([r.label for r in chunk],dtype=th.float32,device=device)
            if logits.shape!=labels.shape:
                raise ValueError('dev native logit shape differs')
            loss=th.nn.functional.binary_cross_entropy_with_logits(logits,labels,reduction='none')
            weights=th.tensor([r.weight for r in chunk],dtype=th.float64,device=device)
            if not th.isfinite(loss).all() or not th.isfinite(weights).all() or (weights<=0).any():
                raise p1b.P1bFailure('nonfinite dev BCE/invalid weights',dict(progress))
            terms.append(float((loss.to(th.float64)*weights).sum()))
            helpers._sync(device); timing['dev_compute']+=time.perf_counter()-tick
            for ref,logit,bce in zip(chunk,logits.detach().cpu().tolist(),loss.detach().cpu().tolist()):
                exposures['dev_positive' if ref.label else 'dev_Normal']+=1
                if on_prediction:
                    on_prediction({'base_window_id':ref.base_window_id,'device':ref.device,'event_id':ref.event_id,
                        'event_instance_id':ref.event_instance_id,'family':ref.family,'strength':ref.strength,
                        'sensor_changed':ref.sensor_changed,'thermal_changed':ref.thermal_changed,
                        'label':ref.label,'weight':ref.weight,'logit':logit,'native_float32_BCE':bce})
    value=math.fsum(terms)
    p1b._finite(value,'nonfinite dev weighted mean',progress)
    return value


def trajectory_summary(probes,*,arm):
    return {'sampled_native_output_categories':[{'step':p['step'],'classification':p['classification']} for p in probes],
            'sparse_sampling':True,'prevention_established':False,'arm':arm,
            'scope':'Native output responsiveness does not establish both branches are used; inspect separate branch controls. Trained fit events may be memorized.'}


def train_loop(inputs, loader, *, arm, device, scope, epochs=30, nominal_batch=16, micro_batch=2,
               seed=42, probe_steps=PROBE_STEPS, observer=None, on_probe=None, on_step=None,
               on_epoch=None, on_terminal=None, on_improvement=None, on_boundary=None,
               progress=None, model_factory=None, order_reference=None):
    """Artificial fixtures may reduce budget; main always uses the exact frozen budget."""
    for value, name in ((epochs, "epochs"), (nominal_batch, "nominal_batch"), (micro_batch, "micro_batch")):
        p1b._positive_int(value, name)
    if scope not in ("synthetic_fixture", "P1c_matched_lr_fit_dev") or arm not in ("S","F") or nominal_batch % 2 or seed != 42:
        raise ValueError("explicit stability scope, seed42 and complete pairs required")
    if scope == "P1c_matched_lr_fit_dev" and (epochs != 30 or nominal_batch != 16 or micro_batch != 2
                                            or model_factory is not None or observer is None or order_reference is None
                                            or tuple(probe_steps) != PROBE_STEPS):
        raise ValueError("real stability budget/observer/replay must match frozen recipe")
    if tuple(sorted(set(probe_steps))) != tuple(probe_steps) or any(type(s) is not int or s < 0 for s in probe_steps):
        raise ValueError("invalid probe steps")
    th, helpers, models_module = p1b._training()
    if th.get_default_dtype() != th.float32:
        raise ValueError("FP32 default dtype required")
    determinism = helpers.enable_determinism()
    th.manual_seed(seed)
    if model_factory is None:
        models, manifest = models_module.build_arm_set("BIN", seed=seed)
        model = models[arm].to(device)
        del models
    else:
        model, manifest = model_factory(seed)
        model = model.to(device)
    initial_sha = (check_initial(model, manifest, arm, inputs["initialization_reference"])
                   if model_factory is None else models_module.state_tensor_sha256(model))
    optimizer = th.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, amsgrad=False)
    scheduler = th.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    sampler = bs.BasePairedSampler(inputs["indexes"]["fit"], seed, nominal_batch//2, 30)
    per_epoch = len(tuple(sampler.epoch_batches(1)))
    if not per_epoch or any(s > per_epoch*epochs for s in probe_steps):
        raise ValueError("probe step outside stability budget")
    dev_refs = bs.dev_examples(inputs["indexes"]["dev"])
    if not dev_refs:
        raise ValueError("dev examples required")
    progress = {} if progress is None else progress
    progress.update(phase="model_setup", optimizer_steps_completed=0)
    timing, exposures, batch_sizes = Counter(), Counter(), Counter()
    stream, history, probes, baseline, best = hashlib.sha256(), [], [], {}, None
    started, peak = time.perf_counter(), 0
    model.train()

    def callback(fn, *args, phase="record_io"):
        if fn:
            tick = time.perf_counter()
            fn(*args)
            timing[phase] += time.perf_counter() - tick

    def measure(step):
        if observer is None or step not in probe_steps:
            return
        progress.update(phase="probe", probe_step=step)
        tick = time.perf_counter()
        with onset.observation_mode(model, th):
            measured = observer(model, step)
        helpers._sync(device)
        entry = {"step": step, "measurement": measured, "classification": measured["native_output_responsiveness"],
                 "wall_seconds": time.perf_counter() - tick}
        timing["probe"] += entry["wall_seconds"]
        for family, group in measured["groups"].items():
            if step == 0:
                baseline[family] = dict(group)
            for key in ("balanced_probe_BCE", "mean_abs_logit_gap", "contrast_BCE_gain"):
                if key in group and key in baseline.get(family, {}):
                    group[key + "_change_from_step0"] = group[key] - baseline[family][key]
        probes.append(entry)
        callback(on_probe, entry)

    measure(0)
    order_gates = []
    for epoch in range(1, epochs+1):
        epoch_start = time.perf_counter()
        timing_before, loader_before = timing.copy(), loader.seconds.copy()
        progress.update(epoch=epoch, phase="train")
        if device.type == "cuda":
            th.cuda.reset_peak_memory_stats(device)
        model.train()
        loader.begin_phase()
        terms, train_count = [], 0
        for batch, pairs in enumerate(sampler.epoch_batches(epoch), 1):
            refs = p1b.paired_examples(pairs)
            progress.update(phase="train", epoch=epoch, batch=batch)
            loader.context = dict(progress)
            optimizer.zero_grad(set_to_none=True)
            projection = {"count": 0, "abs_sum": 0.0, "abs_max": 0.0}
            observation_before = timing["train_projection_observation"]
            def read_projection(module, args, output):
                tick = time.perf_counter()
                value = output.detach()
                if not th.isfinite(value).all():
                    raise ValueError("nonfinite actual train fc_thermal output")
                projection["count"] += value.numel()
                projection["abs_sum"] += float(value.to(th.float64).abs().sum())
                projection["abs_max"] = max(projection["abs_max"], float(value.abs().max()))
                timing["train_projection_observation"] += time.perf_counter() - tick
            fc = getattr(getattr(model, "thermal_encoder", None), "fc_thermal", None)
            handle = fc.register_forward_hook(read_projection) if fc is not None else None
            total, actual = 0.0, len(refs)
            batch_sizes[actual] += 1
            try:
                for start, stop, weight in helpers.partition_microbatches(actual, micro_batch):
                    chunk = refs[start:stop]
                    progress["micro"] = [start, stop]
                    tensors = _tensors(loader, "fit", chunk, arm, device, th, helpers, timing)
                    tick = time.perf_counter()
                    logits = model(**tensors)["logit"]
                    labels = th.tensor([r.label for r in chunk], dtype=th.float32, device=device)
                    if logits.shape != labels.shape:
                        raise ValueError("train BIN logit shape differs")
                    loss = th.nn.functional.binary_cross_entropy_with_logits(logits, labels) * weight
                    value = float(loss.detach())
                    p1b._finite(value, "nonfinite train loss", progress)
                    loss.backward()
                    helpers._sync(device)
                    timing["training_compute"] += time.perf_counter() - tick
                    total += value
                    for ref in chunk:
                        exposures["fit_positive" if ref.label else "fit_Normal"] += 1
                        stream.update((json.dumps({"epoch": epoch, "base": ref.base_window_id, "event": ref.event_id,
                                                   "label": ref.label}, sort_keys=True, separators=(",", ":")) + "\n").encode())
            finally:
                if handle is not None:
                    handle.remove()
            timing["training_compute"] -= timing["train_projection_observation"] - observation_before
            projection["abs_mean"] = projection["abs_sum"] / projection["count"] if projection["count"] else None
            progress["phase"] = "gradient_observation"
            tick = time.perf_counter()
            norms = onset.gradient_norms(model, th)
            helpers._sync(device)
            timing["gradient_observation"] += time.perf_counter() - tick
            progress["phase"] = "optimizer_step"
            tick = time.perf_counter()
            optimizer.step()
            progress["optimizer_steps_completed"] += 1
            helpers._sync(device)
            timing["training_compute"] += time.perf_counter() - tick
            terms.append(total * actual)
            train_count += actual
            callback(on_step, {"step": progress["optimizer_steps_completed"], "epoch": epoch, "batch": batch,
                               "train_loss": total, "actual_batch": actual, "lr_used": scheduler.get_last_lr()[0],
                               "gradient_norms": norms, "train_fc_thermal": projection,
                               "measurement_timing": "forward/backward before this update; fixed probes after update",
                               "train_batch_input_scope": "Changes by step; auxiliary trace is not a fixed-probe onset"})
            measure(progress["optimizer_steps_completed"])
        lr_used = scheduler.get_last_lr()[0]
        scheduler.step()
        tick = time.perf_counter()
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        terminal_sha = models_module.state_tensor_sha256(state)
        timing["terminal_state_snapshot"] += time.perf_counter() - tick
        entry = {"epoch": epoch, "optimizer_steps": batch, "train_loss": math.fsum(terms)/train_count,
                 "lr_used_this_epoch": lr_used, "next_epoch_lr": scheduler.get_last_lr()[0],
                 "fit_order_manifest_sha256": sampler.epoch_manifest(epoch)["sha256"],
                 "cumulative_fit_example_stream_sha256": stream.hexdigest(), "terminal_state_tensor_sha256": terminal_sha}
        progress["phase"] = "terminal_persistence"
        callback(on_terminal, state, entry, phase="terminal_persistence")
        if order_reference is not None:
            progress["phase"] = "epoch_order_gate"
            order_gates.append(check_epoch_order(entry, order_reference))
        entry["dev_loss"] = evaluate_dev(model, loader, dev_refs, device, th, helpers, timing, exposures, progress, micro_batch, arm=arm)
        if best is None or entry["dev_loss"] < best["dev_loss"]:
            best = {"epoch": epoch, "dev_loss": entry["dev_loss"], "state": state}
            progress["phase"] = "selected_checkpoint_persistence"
            callback(on_improvement, state, epoch, entry["dev_loss"], phase="checkpoint_persistence")
        if device.type == "cuda":
            entry["peak_gpu_allocated_bytes"] = int(th.cuda.max_memory_allocated(device))
            peak = max(peak, entry["peak_gpu_allocated_bytes"])
        entry["wall_seconds"] = time.perf_counter() - epoch_start
        entry["phase_seconds"] = {k: v-timing_before[k] for k,v in timing.items()}
        entry["loader_phase_seconds"] = {k: v-loader_before[k] for k,v in loader.seconds.items()}
        history.append(entry)
        progress["phase"] = "history_persistence"
        callback(on_epoch, entry)
        progress["phase"] = "epoch_boundary"
        callback(on_boundary, epoch, phase="gpu_boundary_record")
    record = {"schema": SCHEMA, "recipe_id": RECIPE, "gate": "PASS", "scope": scope,
              "arm": arm, "objective": "BIN", "initial_lr": 1e-4, "training_seed": seed,
              "epochs_completed": len(history), "planned_epochs": epochs, "epochs": history,
              "optimizer_steps_total": progress["optimizer_steps_completed"],
              "real_data_optimizer_steps": progress["optimizer_steps_completed"] if scope == "P1c_matched_lr_fit_dev" else 0,
              "fixture_optimizer_steps": progress["optimizer_steps_completed"] if scope == "synthetic_fixture" else 0,
              "effective_batch": nominal_batch, "micro_batch": micro_batch, "cosine_T_max": 30,
              "initial_manifest": manifest, "initial_state_tensor_sha256": initial_sha,
              "final_state_tensor_sha256": terminal_sha, "selected_state_tensor_sha256": models_module.state_tensor_sha256(best["state"]),
              "dev_checkpoint_epoch": best["epoch"], "dev_checkpoint_loss": best["dev_loss"],
              "dev_checkpoint_ties": [e["epoch"] for e in history if e["dev_loss"] == best["dev_loss"] and e["epoch"] != best["epoch"]],
              "checkpoint_rule": "strict_less_than; earliest_epoch_on_exact_tie",
              "dev_reduction": "sum(weight * unreduced_BCE); no second mean",
              "dev_weight_manifest": bs.dev_weight_manifest(inputs["indexes"]["dev"]),
              "epoch_order_gates": order_gates, "actual_fit_event_stream_sha256": stream.hexdigest(),
              "exposures": dict(exposures), "actual_batch_sizes": {str(k):v for k,v in sorted(batch_sizes.items())},
              "determinism": determinism, "probe_steps_completed": [p["step"] for p in probes],
              "probe_record_payload_sha256":sa.digest_json(probes),
              "phase_seconds": dict(timing), "loader_counts": dict(loader.counts), "loader_phase_seconds": dict(loader.seconds),
              "train_loop_wall_seconds": time.perf_counter()-started, "peak_gpu_allocated_bytes_run_max": peak,
              "trajectory": trajectory_summary(probes, arm=arm) if probes else None,
              "actual_roles_read": ["fit", "dev"], "calibration_window_read": False, "heldout_window_read": False,
              "threshold_generation": False, "paper_result_eligible": False, "full_P0_gate": "NOT_EVALUATED"}
    return record, best["state"]


def verify_training_artifacts(out,record,selected,terminals,*,expected_epochs=30,
                              expected_steps=7830,expected_probe_steps=PROBE_STEPS):
    """Reload actual files before PASS; smaller expectations are test fixtures only."""
    if (record['optimizer_steps_total']!=expected_steps or record['epochs_completed']!=expected_epochs
            or record['probe_steps_completed']!=list(expected_probe_steps)):
        raise ValueError('incomplete frozen comparison budget/probes')
    out=Path(out)
    persisted=[json.loads(l) for l in (out/'epochs.jsonl').read_text().splitlines()]
    probes=[json.loads(l) for l in (out/'probes.jsonl').read_text().splitlines()]
    steps=[json.loads(l) for l in (out/'steps.jsonl').read_text().splitlines()]
    if (persisted!=record['epochs'] or [e.get('epoch') for e in persisted]!=list(range(1,expected_epochs+1))
            or [s.get('epoch') for s in terminals]!=list(range(1,expected_epochs+1))
            or [p.get('step') for p in probes]!=list(expected_probe_steps)
            or sa.digest_json(probes)!=record['probe_record_payload_sha256']
            or [s.get('step') for s in steps]!=list(range(1,expected_steps+1))):
        raise ValueError('persisted history/probes/steps/terminal coverage differs')
    # min keeps the first tied row in persisted epoch order, matching strict <.
    earliest=min(persisted,key=lambda e:e['dev_loss'])
    if (record['dev_checkpoint_epoch']!=earliest['epoch']
            or record['dev_checkpoint_loss']!=earliest['dev_loss']):
        raise ValueError('selected checkpoint is not the earliest persisted dev minimum')
    th,_,models=p1b._training()
    checkpoint=th.load(out/'checkpoint.pt',map_location='cpu',weights_only=True)
    selected_sha=models.state_tensor_sha256(selected)
    if (checkpoint['epoch']!=record['dev_checkpoint_epoch'] or checkpoint['dev_loss']!=record['dev_checkpoint_loss']
            or checkpoint['selection']!='DEV_EARLIEST_MINIMUM'
            or models.state_tensor_sha256(checkpoint['model_state_dict'])!=selected_sha
            or selected_sha!=persisted[record['dev_checkpoint_epoch']-1]['terminal_state_tensor_sha256']
            or selected_sha!=record['selected_state_tensor_sha256']):
        raise ValueError('persisted selected checkpoint differs')
    del checkpoint
    for item,epoch in zip(terminals,persisted):
        path=out/item['file']
        state=th.load(path,map_location='cpu',weights_only=True)
        if (state['epoch']!=item['epoch'] or state['selection']!='EPOCH_TERMINAL_NOT_DEV_SELECTED'
                or p1b.file_sha(path)!=item['file_sha256']
                or models.state_tensor_sha256(state['model_state_dict'])!=item['tensor_sha256']
                or item['tensor_sha256']!=epoch['terminal_state_tensor_sha256']):
            raise ValueError('persisted terminal state differs')
        del state
    return {'gate':'PASS_FULL_TRAINING_ARTIFACTS','epochs_reloaded':len(persisted),'terminal_states_reloaded':len(terminals),
            'probes_reloaded':len(probes),'steps_reloaded':len(steps),'selected_state_tensor_sha256':selected_sha,
            'checkpoint_file_sha256':p1b.file_sha(out/'checkpoint.pt'),'before_result_PASS':True,
            'artifact_file_sha256':{name:p1b.file_sha(out/name) for name in
                                   ('checkpoint.pt','epochs.jsonl','probes.jsonl','steps.jsonl',*(s['file'] for s in terminals))}}


def prediction_summary(rows,arm):
    """Original weighted contributions plus conditional metrics with explicit masses.

    A family conditional metric includes Normal and that family's positives with
    their original weights, then divides by its mass. It is not class rebalanced.
    Normal contribution appears once in the aggregate, separately in each
    conditional comparison. Bounds use the identical conditional definition.
    """
    if not rows or any(not math.isfinite(r['logit']) or not math.isfinite(r['native_float32_BCE'])
                       or not math.isfinite(r['weight']) or r['weight']<=0 for r in rows):
        raise ValueError('invalid native dev predictions')
    normal=[r for r in rows if r['label']==0]
    positive=[r for r in rows if r['label']==1]
    def weighted(values):
        return math.fsum(r['weight']*r['native_float32_BCE'] for r in values)
    def bound(values):
        by_base={}
        for r in values:by_base.setdefault((r['device'],r['base_window_id']),[]).append(r)
        terms=[]
        for base in by_base.values():
            a=math.fsum(r['weight'] for r in base if r['label']==0)
            b=math.fsum(r['weight'] for r in base if r['label']==1 and (
                (arm=='S' and r['sensor_changed'] is False) or (arm=='T' and r['thermal_changed'] is False)))
            if a and b:terms.append(a*math.log((a+b)/a)+b*math.log((a+b)/b))
        return math.fsum(terms) if arm!='F' else None
    families={}
    for family in sorted({r['family'] for r in positive}):
        selected=[r for r in positive if r['family']==family]
        population=normal+selected
        mass=math.fsum(r['weight'] for r in population)
        lower=bound(population)
        families[family]={'positive_examples':len(selected),'original_weighted_positive_contribution':weighted(selected),
            'conditional_population':'all Normal plus this family positives; original weights renormalized by mass, not class balanced',
            'conditional_weight_mass':mass,'conditional_BCE':weighted(population)/mass,
            'conditional_unchanged_input_lower_bound':None if lower is None else lower/mass}
    devices={}
    for device in sorted({r['device'] for r in rows}):
        selected=[r for r in rows if r['device']==device]
        mass=math.fsum(r['weight'] for r in selected)
        devices[device]={'examples':len(selected),'original_weight_mass':mass,'conditional_BCE':weighted(selected)/mass}
    strength_groups={}
    for family,strength in sorted({(r['family'],r['strength']) for r in positive}):
        selected=[r for r in positive if (r['family'],r['strength'])==(family,strength)]
        population=normal+selected;mass=math.fsum(r['weight'] for r in population)
        lower=bound(population)
        strength_groups[family+':'+str(strength)]={'positive_examples':len(selected),
            'original_weighted_positive_contribution':weighted(selected),
            'conditional_population':'all Normal plus this family/strength positives; original weights renormalized by mass',
            'conditional_weight_mass':mass,'conditional_BCE':weighted(population)/mass,
            'conditional_unchanged_input_lower_bound':None if lower is None else lower/mass}
    return {'examples':len(rows),'original_weight_mass':math.fsum(r['weight'] for r in rows),
            'normal_original_weighted_contribution':weighted(normal),'families':families,'family_strength':strength_groups,'devices':devices,
            'unchanged_input_aggregate_lower_bound':bound(rows),
            'scope':'Native float32 BCE with original weights; source per-microbatch aggregate is primary. Cross-base duplicates not measured. F bound uncomputed.'}


def predict_states(inputs,loader,*,arm,states,out,device,progress,boundary=None):
    """Fresh inference model, exact same native microbatch dev reduction as training."""
    th,helpers,models=p1b._training()
    helpers.enable_determinism()
    th.manual_seed(42)
    arms,_=models.build_arm_set('BIN',seed=42)
    model=arms[arm].to(device)
    del arms
    refs=bs.dev_examples(inputs['indexes']['dev'])
    reports=[]
    for epoch,source in states:
        progress.update(phase='dev_prediction_state_load',prediction_epoch=epoch)
        saved=th.load(source['path'],map_location='cpu',weights_only=True)
        if (saved['epoch']!=epoch or models.state_tensor_sha256(saved['model_state_dict'])!=source['tensor_sha256']
                or p1b.file_sha(source['path'])!=source['file_sha256']):
            raise ValueError('prediction model state/metadata differs')
        model.load_state_dict(saved['model_state_dict'])
        del saved
        rows=[]; timing=Counter(); exposures=Counter()
        value=evaluate_dev(model,loader,refs,device,th,helpers,timing,exposures,progress,2,arm=arm,on_prediction=rows.append)
        if value!=source['dev_loss']:
            raise p1b.P1bFailure('native dev prediction replay BCE differs exactly',
                                 {'arm':arm,'epoch':epoch,'actual':value,'reference':source['dev_loss']})
        path=Path(out)/f'dev-predictions-epoch-{epoch:02}.jsonl'
        with path.open('x',encoding='utf-8') as stream:
            for row in rows:stream.write(json.dumps(row,sort_keys=True,separators=(',', ':'),allow_nan=False)+'\n')
            stream.flush();os.fsync(stream.fileno())
        reports.append({'epoch':epoch,'dev_loss':value,'gate':'PASS_EXACT_NATIVE_DEV_BCE',
                        'state_tensor_sha256':source['tensor_sha256'],'state_file_sha256':source['file_sha256'],
                        'file':path.name,'file_sha256':p1b.file_sha(path),'summary':prediction_summary(rows,arm),
                        'phase_seconds':dict(timing),'exposures':dict(exposures)})
        if boundary:boundary(epoch)
    return reports


def parser():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--task',choices=('T-dev-reference','S','F'),required=True)
    paths=('prepared','bank-result','cache-run','sampler-fixtures','runner-fixtures','diagnostic-fixtures',
           'onset-plan','baseline-result','replay-result','stability-fixtures','stability-plan',
           'thermal-result','comparison-fixtures','plan','out')
    pins=('cache-result','sampler-fixtures','runner-fixtures','diagnostic-fixtures','onset-plan','baseline-result',
          'replay-result','stability-fixtures','stability-plan','thermal-result','comparison-fixtures','plan')
    for flag in paths:cli.add_argument('--'+flag,type=Path,required=True)
    for flag in pins:cli.add_argument('--'+flag+'-sha256',required=True)
    for flag in ('reference-predictions','sensor-result'):
        cli.add_argument('--'+flag,type=Path)
        cli.add_argument('--'+flag+'-sha256')
    return cli


def record_failure(out, exception, progress):
    failure = {"schema": SCHEMA, "gate": "FAILED", "run_usable": False,
               "timestamp": datetime.now(timezone.utc).isoformat(), "hostname": platform.node(),
               "python_version": platform.python_version(), "git_dirty": True,
               "exception": type(exception).__name__, "message": str(exception),
               "stage": progress.get("stage"), "context": {**progress, **getattr(exception,"context",{})},
               "retried": False, "learning_rate_adjusted": False, "paper_result_eligible": False}
    try:
        failure["partial_files_preserved"] = sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file())
        p1b._write_json(out / "failure.json", failure)
    except BaseException as error:
        try:
            print(json.dumps({"original_failure": failure,"failure_record_write_error":repr(error)},
                             default=repr,allow_nan=False),file=sys.stderr,flush=True)
        except BaseException:
            pass


def main(argv=None):
    args=parser().parse_args(argv)
    out=p1b._check_output(args)
    for name in ('plan','stability_plan','thermal_result','comparison_fixtures','reference_predictions','sensor_result'):
        value=getattr(args,name)
        if value is not None:
            guarded=value.resolve().parent
            if out==guarded or guarded in out.parents or out in guarded.parents:
                raise ValueError('output overlaps immutable comparison evidence')
    out.mkdir(parents=True,exist_ok=False)
    cache=None;progress={'stage':'inputs','optimizer_steps_completed':0};started=time.perf_counter()
    try:
        inputs=validate_inputs(args)
        if str(out)!=inputs['plan']['output_paths'][args.task]:
            raise ValueError('comparison output is not frozen for this task')
        progress['stage']='environment'
        device,environment,_=p1b.runtime_environment()
        identity=onset.runtime_identity(device)
        runtime_gate=onset.check_runtime_identity(identity,inputs['plan']['runtime_reference'])
        progress['runtime_identity']=identity
        provenance={'schema':SCHEMA,'task':args.task,'timestamp':datetime.now(timezone.utc).isoformat(),
            'hostname':platform.node(),'python_version':platform.python_version(),'environment_profile':'SERVER-TRAINING',
            'environment':environment,'git_commit_sha':inputs['candidate_git_commit_sha'],'git_dirty':True,
            'jetpack_l4t':environment.get('jetpack_l4t'),
            'sensor_driver_versions':environment.get('sensor_driver_versions',{}),
            'source_files':inputs['candidate_source_files'],'plan_file_sha256':args.plan_sha256,
            'plan_payload_sha256':inputs['plan']['sha256'],'runtime_identity':identity,'runtime_gate':runtime_gate,
            'comparison_fixture_payload_sha256':inputs['comparison_fixture_payload_sha256'],
            'T_result_payload_sha256':inputs['thermal_reference']['sha256'],
            'command':list(argv) if argv is not None else sys.argv,'paper_result_eligible':False}
        p1b._write_json(out/'provenance.json',provenance)
        def boundary(epoch):
            onset.check_runtime_identity(onset.runtime_identity(device),identity)
            if source_hashes()!=inputs['candidate_source_files']:
                raise ValueError('candidate source changed at comparison boundary')
            row=st.gpu_activity(identity['gpu_uuid'])
            row.update(epoch=epoch,runtime_identity=identity,source_files_unchanged=True)
            onset._append_json(out/'gpu-activity.jsonl',row)
        boundary(0)
        progress['stage']='cache_attach'
        cache=p1b.rc.RawNormalCache.attach(inputs['cache_path']).open()
        if cache.manifest['sha256']!=inputs['cache_manifest']['sha256']:
            raise ValueError('comparison cache manifest changed after gate')
        loader=p1b.PairLoader(inputs,cache.reader(verify_each_read=False))
        terminals=[]
        if args.task=='T-dev-reference':
            arm='T';states=[]
            for epoch in (7,30):
                state=inputs['thermal_reference']['terminal_states'][epoch-1]
                states.append((epoch,{'path':inputs['thermal_result_path'].parent/state['file'],
                    'tensor_sha256':state['tensor_sha256'],'file_sha256':state['file_sha256'],
                    'dev_loss':inputs['thermal_reference']['epochs'][epoch-1]['dev_loss']}))
            progress['stage']='reference_prediction'
            predictions=predict_states(inputs,loader,arm='T',states=states,out=out,device=device,progress=progress,boundary=boundary)
            record={'schema':SCHEMA,'recipe_id':RECIPE,'gate':'PASS','task':args.task,'arm':'T',
                    'objective':'BIN','real_data_optimizer_steps':0,'fixture_optimizer_steps':0,
                    'micro_batch':2,'dev_weight_manifest':inputs['thermal_reference']['dev_weight_manifest'],
                    'dev_reduction':'sum(weight * unreduced_BCE); no second mean',
                    'reference_epochs':[7,30],'predictions':predictions,'actual_roles_read':['dev']}
        else:
            arm=args.task
            probe_loader=p1b.PairLoader(inputs,cache.reader(verify_each_read=False))
            (out/'terminal-states').mkdir()
            def terminal(state,entry):
                path=out/'terminal-states'/f"epoch-{entry['epoch']:02}.pt"
                st.save_state(path,state,entry['epoch'],selection='EPOCH_TERMINAL_NOT_DEV_SELECTED')
                terminals.append({'epoch':entry['epoch'],'file':str(path.relative_to(out)),
                                  'file_sha256':p1b.file_sha(path),'tensor_sha256':entry['terminal_state_tensor_sha256']})
            progress['stage']='training'
            record,selected=train_loop(inputs,loader,arm=arm,device=device,scope='P1c_matched_lr_fit_dev',
                order_reference=inputs['epoch_order_reference'],progress=progress,
                observer=lambda m,s:observe_and_preserve(m,s,out,probe_loader,inputs['probe_pairs'],arm=arm,device=device),
                on_probe=lambda e:onset._append_json(out/'probes.jsonl',e),on_step=lambda e:onset._append_json(out/'steps.jsonl',e),
                on_epoch=lambda e:onset._append_json(out/'epochs.jsonl',e),on_terminal=terminal,
                on_improvement=lambda state,epoch,loss:st.save_state(out/'checkpoint.pt',state,epoch,
                                                    selection='DEV_EARLIEST_MINIMUM',dev_loss=loss),
                on_boundary=boundary)
            progress['stage']='artifact_verification'
            record['artifact_verification']=verify_training_artifacts(out,record,selected,terminals)
            del selected
            epochs=sorted({7,30,record['dev_checkpoint_epoch']})
            states=[]
            for epoch in epochs:
                state=terminals[epoch-1]
                states.append((epoch,{'path':out/state['file'],'tensor_sha256':state['tensor_sha256'],
                    'file_sha256':state['file_sha256'],'dev_loss':record['epochs'][epoch-1]['dev_loss']}))
            progress['stage']='selected_and_fixed_predictions'
            predictions=predict_states(inputs,loader,arm=arm,states=states,out=out,device=device,progress=progress,boundary=boundary)
            record.update(task=args.task,predictions=predictions,terminal_states=terminals,
                          selected_dev_is_exploratory_and_selection_biased=True,epoch7_is_posthoc_T_selected_anchor=True,
                          no_equal_optimizer_trajectory_claim_across_modalities=True,
                          reference_prediction_payload_sha256=inputs['reference_predictions']['sha256'],
                          probe_loader_counts=dict(probe_loader.counts),probe_loader_phase_seconds=dict(probe_loader.seconds))
        progress['stage']='final_bindings'
        refreshed=validate_inputs(args)
        if refreshed['candidate_source_files']!=inputs['candidate_source_files'] or refreshed['plan']['sha256']!=inputs['plan']['sha256']:
            raise ValueError('comparison inputs changed during execution')
        onset.check_runtime_identity(onset.runtime_identity(device),identity)
        artifact_hashes={str(p.relative_to(out)):p1b.file_sha(p) for p in out.rglob('*') if p.is_file() and p.name!='provenance.json'}
        if args.task!='T-dev-reference' and any(artifact_hashes.get(name)!=h for name,h in record['artifact_verification']['artifact_file_sha256'].items()):
            raise ValueError('verified training artifact changed during prediction/finalization')
        record.update(source_files=inputs['candidate_source_files'],plan_payload_sha256=inputs['plan']['sha256'],
            input_evidence=inputs['evidence'],runtime_identity=identity,runtime_gate=runtime_gate,
            comparison_fixture_payload_sha256=inputs['comparison_fixture_payload_sha256'],
            T_result_payload_sha256=inputs['thermal_reference']['sha256'],
            calibration_window_read=False,heldout_window_read=False,threshold_generation=False,paper_result_eligible=False,
            full_P0_gate='NOT_EVALUATED',branch_rules_selected_after_T_inspection=True,
            provenance_file_sha256=p1b.file_sha(out/'provenance.json'),
            artifact_file_sha256=artifact_hashes,
            total_loader_counts=dict(loader.counts),total_loader_phase_seconds=dict(loader.seconds),
            wall_until_final_record_seconds=time.perf_counter()-started)
        record['sha256']=sa.digest_json(record)
        p1b._write_json(out/'result.json',record)
    except BaseException as error:
        record_failure(out,error,progress)
        raise
    finally:
        if cache is not None:
            try:cache.close()
            except BaseException:pass
    # Success printing is outside the failure-recording try: an output pipe error
    # cannot append FAILED beside an already completed PASS result (review51-B).
    print(json.dumps({'gate':'PASS','task':args.task,'out':str(out),'optimizer_steps':record['real_data_optimizer_steps']}))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
