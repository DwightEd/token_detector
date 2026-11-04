"""Finite orchestration checks: original rows, frozen inputs and failure reuse."""
import argparse
import json
from pathlib import Path

import numpy as np
import pytest

from experiments import head_carrier_observation as observation
from experiments.data import file_hash, read_json
from experiments import head_carrier_frontend


@pytest.fixture
def prepared(tmp_path,monkeypatch):
    graphs=tmp_path/'graphs'; graphs.mkdir()
    samples=tmp_path/'samples'; samples.mkdir()
    for case in ('00012','00013'):
        root=graphs/case; root.mkdir()
        a=np.zeros((32,32,5,5),dtype=np.float32)
        for h in range(32):
            a[:,h,:,0]=.1+.02*(h%4)
            a[:,h,:,1]=.3-.02*(h%4)
        v=np.zeros((32,5,8,128),dtype=np.float32)
        v[1:,0]=1.; v[1:,1]=2.; v[1:,2]=3.
        np.save(root/'attention.npy',a)
        np.save(root/'values.npy',v)
        np.save(root/'target_ids.npy',np.array([3,17],dtype=np.int64))
    metadata=tmp_path/'input.json'; metadata.write_text('{}',encoding='utf-8')
    stat=metadata.stat()
    frontend=dict(schema='test-only-frontend',gate_passed=True,source_id='14375',F=2,Q=2,
        tuple_count=32*48*2*2,
        cases=[dict(case_id=case,P=4,T=2,N=5,all_targets=[
            dict(t=0,q=3,target_id=3,selected=False),
            dict(t=1,q=4,target_id=17,selected=True)]) for case in ('00012','00013')],
        eligible_pairs=[dict(pair_id='duplicate',kind='EQUIV',positions=[0,1],token_id=4520),
                        dict(pair_id='different',kind='SEM',positions=[0,2],token_id=4520)],
        input_files=[dict(path=str(metadata),bytes=stat.st_size,mtime_ns=stat.st_mtime_ns,
                          sha256=file_hash(metadata),hash_mode='actual')])
    for name in ('licence','qualification'):
        (tmp_path/f'{name}.json').write_text('{}',encoding='utf-8')
    bindings={}
    for index in range(3):
        path=tmp_path/f'policy-{index}.md'; path.write_text(str(index),encoding='utf-8')
        bindings[str(path)]=file_hash(path)
    protocol=dict(schema='h019-head-carrier-association-v1',limits=observation.LIMITS,
        source_id='14375',cases=['00012','00013'],counters=observation.COUNTERS,
        m2_admitted=False,detector_admitted=False,scientific_bindings=bindings)
    protocol_path=tmp_path/'protocol.json'; protocol_path.write_text(json.dumps(protocol),encoding='utf-8')
    args=argparse.Namespace(graphs=graphs,samples=samples,licence=tmp_path/'licence.json',
        qualification=tmp_path/'qualification.json',protocol=protocol_path,
        output=tmp_path/'plan',prepare_only=True,plan=None)
    monkeypatch.setattr(head_carrier_frontend,'prepare_frontend',lambda *args:frontend)
    monkeypatch.setattr(observation,'research_identity',lambda:{'fixture':'frozen'})
    monkeypatch.setattr(observation,'completed_identity',lambda identity:identity)
    monkeypatch.setattr(observation,'_private_bytes',lambda:64*1024**2)
    return args,frontend,metadata


def execute(args,output):
    observation.run(args)
    args.plan=args.output
    args.output=output
    args.prepare_only=False
    return observation.run(args)


def test_all_original_rows_independent_pairs_and_factorized_messages(prepared,tmp_path):
    args,front,_=prepared
    result=execute(args,tmp_path/'observations')
    assert result['completed'] and result['accounting']['completed_tuples']==front['tuple_count']
    assert result['actual_counts']==observation.COUNTERS
    assert result['M2']=='not_admitted_unknown' and result['auroc'] is None
    assert result['independent_interventions_not_summed']
    for case in ('00012','00013'):
        root=args.output/('case-'+case)
        assert len(list(root.glob('layer-*-pair-*-chunk-*.npz')))==64
        with np.load(root/'layer-01-pair-00-chunk-0000.npz') as saved:
            assert saved['query'].tolist()==[4]*48
            assert saved['target'].tolist()==[1]*48
            assert saved['target_id'].tolist()==[17]*48
            assert np.unique(np.stack((saved['head1'],saved['head2']),axis=1),axis=0).shape==(48,2)
            assert np.all(saved['head1']//4==saved['kv_group'])
            assert np.all(saved['head2']//4==saved['kv_group'])
            assert saved['concat_norm'].max()>0
            assert 'delta' not in saved.files and 'value_difference' not in saved.files
            with np.load(root/'layer-01-value-differences.npz') as factors:
                difference=factors['value_difference'][saved['pair_index'],saved['kv_group']]
            delta=saved['delta_coefficients'][...,None]*difference[:,None,:]
            assert np.allclose(np.linalg.norm(delta.reshape(48,-1),axis=1),saved['concat_norm'])
        with np.load(root/'layer-00-pair-00-chunk-0000.npz') as saved:
            assert np.all(saved['concat_norm']==0)
    assert not list(args.output.rglob('*prediction*'))


def test_frontend_rejection_cannot_start_m1(prepared,tmp_path):
    args,front,_=prepared
    front.update(gate_passed=False,F=0,tuple_count=0)
    result=observation.run(args)
    assert result['m1_numeric_tuples']==0 and result['gate_passed'] is False
    args.plan=args.output; args.output=tmp_path/'rejected'; args.prepare_only=False
    with pytest.raises(ValueError,match='M0 did not admit'):
        observation.run(args)
    assert not args.output.exists()


def test_changed_input_stops_before_any_numeric_output(prepared,tmp_path):
    args,_,metadata=prepared
    observation.run(args)
    metadata.write_text('{"changed":true}',encoding='utf-8')
    args.plan=args.output; args.output=tmp_path/'changed'; args.prepare_only=False
    with pytest.raises(ValueError,match='Prepared input stat changed'):
        observation.run(args)
    assert not args.output.exists()


def test_numeric_failure_keeps_first_raw_chunk_without_retry(prepared,tmp_path,monkeypatch):
    from experiments.research import head_carrier_association as kernel
    args,_,_=prepared
    real=kernel.erase_head_carrier_association; calls=[]
    def failed(*values,**options):
        calls.append(1)
        result=real(*values,**options); result['passed']=False
        return result
    monkeypatch.setattr(kernel,'erase_head_carrier_association',failed)
    with pytest.raises(ValueError,match='numeric gate failed'):
        execute(args,tmp_path/'failed')
    failure=read_json(args.output/'failure.json')
    assert len(calls)==1 and failure['accounting']['attempted_tuples']==48
    assert failure['accounting']['completed_tuples']==0
    assert len(list(args.output.rglob('*chunk-*.npz')))==1
    assert (args.output/'seal.json').exists() and not (args.output/'result.json').exists()


def test_input_identity_is_rechecked_after_numeric_work(prepared,tmp_path,monkeypatch):
    args,_,metadata=prepared
    real=observation._case
    def changed_after_case(*values,**options):
        result=real(*values,**options)
        metadata.write_text('{"changed_during_run":true}',encoding='utf-8')
        return result
    monkeypatch.setattr(observation,'_case',changed_after_case)
    with pytest.raises(ValueError,match='Prepared input stat changed'):
        execute(args,tmp_path/'changed-during-run')
    failure=read_json(args.output/'failure.json')
    assert failure['completed_cases']==['00012','00013']
    assert failure['accounting']['completed_tuples']==32*48*2*2
    assert not (args.output/'result.json').exists()


def test_central_cli_exposes_finite_stage(tmp_path):
    from experiments.run import parser
    args=parser().parse_args(['transfer','observe-head-carrier-association',
        *[item for name in ('graphs','samples','licence','qualification','protocol','output')
          for item in ('--'+name,str(tmp_path/name))],'--prepare-only'])
    assert args.prepare_only and args.transfer_stage=='observe-head-carrier-association'
