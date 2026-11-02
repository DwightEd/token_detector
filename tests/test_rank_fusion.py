"""Independent rank oracle and score-only integration checks."""
import json

import numpy as np
import pytest

from experiments.data import file_hash, text_hash, verify_seal
from experiments.rank_fusion import AXES, RULE, _load, score
from experiments.research.score_fusion import equal_rank_fusion
from experiments.run import main


def oracle(values):
    values=np.asarray(values)
    return np.array([(np.sum(values<x)+.5*np.sum(values==x))/len(values) for x in values])


def test_ties_permutation_and_monotone_scaling():
    a=np.array([3.,-2.,3.,7.,0.])
    b=np.array([1.,8.,0.,1.,-4.])
    expected=.5*(oracle(a)+oracle(b))
    np.testing.assert_allclose(equal_rank_fusion(a,b),expected,atol=1e-15)
    np.testing.assert_array_equal(equal_rank_fusion(3*a+10,np.exp(b)),equal_rank_fusion(a,b))
    order=np.array([3,1,4,0,2])
    np.testing.assert_array_equal(equal_rank_fusion(a[order],b[order]),expected[order])


def test_constant_channel_retains_other_order_and_empty():
    a=np.array([-5.,2.,2.,1.])
    fused=equal_rank_fusion(a,np.ones(4))
    np.testing.assert_array_equal(np.sign(fused[:,None]-fused),np.sign(a[:,None]-a))
    assert equal_rank_fusion([],[]).shape==(0,)
    for a,b in [([1],[np.nan]),([1,2],[1]),([[1]],[[2]]),(['1'],[1]),([1+2j],[1])]:
        with pytest.raises(ValueError):equal_rank_fusion(a,b)


@pytest.mark.parametrize('dtype,base',[(np.int64,2**53),(np.uint64,2**63)])
def test_integer_rank_precision(dtype,base):
    values=np.array([base+1,base,base+1,base+2],dtype=dtype)
    constant=np.ones(4,dtype=dtype)
    expected=.5*(oracle(values)+oracle(constant))
    np.testing.assert_array_equal(equal_rank_fusion(values,constant),expected)
    np.testing.assert_array_equal(equal_rank_fusion(constant,values),expected)


def inputs(tmp_path):
    common={'sample_id':np.array(['a','a','b']),
            'source_id':np.array(['来源一','来源一','source-two']),
            'response_sha256':np.array([text_hash('first response')]*2+[text_hash('second response')]),
            'token_index':np.array([0,2,0]),'token_id':np.array([10,12,10]),
            'start':np.array([0,4,0]),'end':np.array([2,6,2]),
            'observer_id':np.array('fixture-observer')}
    paths=[]
    for method,values in [('r4',[3.,1.,2.]),('witness',[0.,5.,1.])]:
        path=tmp_path/(method+'.npz')
        # Object labels would fail with allow_pickle=False if the runner reads
        # them. Including them here independently checks score-only access.
        np.savez(path,**common,score=np.array(values),method=np.array(method),
                 labels=np.array([{'do_not_read':True}]*3,dtype=object))
        paths.append(path)
    protocol=tmp_path/'protocol.json'
    population=tmp_path/'population.npz'
    np.savez(population,**{key:common[key] for key in AXES})
    protocol.write_text(json.dumps({**RULE,'task':'fixture','observer_id':'fixture-observer',
                                   'population_manifest':population.name,
                                   'population_sha256':file_hash(population)}))
    return paths,protocol,common


def test_central_cli_seals_scores_without_reading_labels(tmp_path,monkeypatch):
    paths,protocol,common=inputs(tmp_path)
    monkeypatch.chdir(tmp_path)
    output=tmp_path/'outputs/fusion'
    monkeypatch.setattr('sys.argv',['token-experiment','transfer','fuse-ranks',
                                   '--r4',str(paths[0]),'--witness',str(paths[1]),
                                   '--protocol',str(protocol),'--output',str(output)])
    main()
    verify_seal(output,expected={'scores.npz','run.json'})
    report=json.loads((output/'run.json').read_text())
    assert report['annotation_reads']==0 and report['metrics_computed'] is False
    population=tmp_path/'population.npz'
    assert report['input_sha256'][str(population)]==file_hash(population)
    with np.load(output/'scores.npz',allow_pickle=False) as result:
        for key in AXES:np.testing.assert_array_equal(result[key],common[key])
        np.testing.assert_array_equal(result['score'],.5*(oracle([3,1,2])+oracle([0,5,1])))
    with pytest.raises(FileExistsError):score(*paths,protocol,output)


def test_rejects_misaligned_tokens_or_unfrozen_rule(tmp_path,monkeypatch):
    paths,protocol,common=inputs(tmp_path)
    monkeypatch.chdir(tmp_path)
    common['token_id']=np.array([10,99,10])
    np.savez(paths[1],**common,score=np.array([0.,5.,1.]),method=np.array('witness'))
    output=tmp_path/'outputs/rejected'
    with pytest.raises(ValueError,match='axes differ'):score(*paths,protocol,output)
    assert not output.exists()
    protocol.write_text(json.dumps({**RULE,'r4_weight':.7,'task':'fixture','observer_id':'fixture-observer'}))
    with pytest.raises(ValueError,match='Protocol'):score(*paths,protocol,output)


def test_utf8_byte_scalar_and_array_identities_are_normalized(tmp_path,monkeypatch):
    paths,protocol,common=inputs(tmp_path)
    byte_common={key:(np.char.encode(value,'utf-8') if value.dtype.kind=='U' else value)
                 for key,value in common.items()}
    np.savez(paths[0],**byte_common,score=np.array([3.,1.,2.]),method=np.array(b'r4'))
    loaded=_load(paths[0],'r4')
    assert loaded['method']=='r4' and loaded['observer_id']=='fixture-observer'
    for key in ['sample_id','source_id','response_sha256']:
        assert loaded[key].dtype.kind=='U'
        np.testing.assert_array_equal(loaded[key],common[key])
    monkeypatch.chdir(tmp_path)
    score(*paths,protocol,tmp_path/'outputs/bytes')


@pytest.mark.parametrize('key',['sample_id','source_id','response_sha256','observer_id'])
def test_empty_byte_identities_are_rejected(tmp_path,key):
    paths,_,common=inputs(tmp_path)
    common[key]=np.array(b'') if key=='observer_id' else np.array([b'',b'',b''])
    np.savez(paths[0],**common,score=np.array([3.,1.,2.]),method=np.array(b'r4'))
    with pytest.raises(ValueError,match='Nonempty UTF-8 text'):
        _load(paths[0],'r4')


def test_invalid_utf8_identity_is_rejected(tmp_path):
    paths,_,common=inputs(tmp_path)
    common['observer_id']=np.array(b'\xff')
    np.savez(paths[0],**common,score=np.array([3.,1.,2.]),method=np.array(b'r4'))
    with pytest.raises(ValueError,match='Nonempty UTF-8 text'):
        _load(paths[0],'r4')


@pytest.mark.parametrize('digest',['h1','a'*63,'a'*65,'A'*64,'g'*64])
def test_response_digest_requires_lowercase_sha256(tmp_path,digest):
    paths,_,common=inputs(tmp_path)
    common['response_sha256']=np.array([digest]*3)
    np.savez(paths[0],**common,score=np.array([3.,1.,2.]),method=np.array('r4'))
    with pytest.raises(ValueError,match='64 lowercase hexadecimal'):
        _load(paths[0],'r4')


def test_matching_score_subsets_cannot_drop_population_tokens(tmp_path,monkeypatch):
    paths,protocol,common=inputs(tmp_path)
    for path,method,values in zip(paths,['r4','witness'],[[3.,1.],[0.,5.]],strict=True):
        subset={key:(value[:2] if key in AXES else value) for key,value in common.items()}
        np.savez(path,**subset,score=np.array(values),method=np.array(method))
    monkeypatch.chdir(tmp_path)
    output=tmp_path/'outputs/subset'
    with pytest.raises(ValueError,match='coverage differs'):
        score(*paths,protocol,output)
    assert not output.exists()


def test_population_manifest_hash_is_frozen(tmp_path,monkeypatch):
    paths,protocol,common=inputs(tmp_path)
    population=tmp_path/'population.npz'
    common['token_id']=np.array([10,12,99])
    np.savez(population,**{key:common[key] for key in AXES})
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError,match='Frozen population manifest changed'):
        score(*paths,protocol,tmp_path/'outputs/changed-population')


def test_population_manifest_must_be_label_free(tmp_path,monkeypatch):
    paths,protocol,common=inputs(tmp_path)
    population=tmp_path/'population.npz'
    np.savez(population,**{key:common[key] for key in AXES},
             labels=np.array([{'do_not_read':True}]*3,dtype=object))
    declared=json.loads(protocol.read_text())
    declared['population_sha256']=file_hash(population)
    protocol.write_text(json.dumps(declared))
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError,match='only the original token axes'):
        score(*paths,protocol,tmp_path/'outputs/labelled-population')


def test_protocol_requires_population_binding(tmp_path,monkeypatch):
    paths,protocol,_=inputs(tmp_path)
    declared=json.loads(protocol.read_text())
    declared.pop('population_sha256')
    protocol.write_text(json.dumps(declared))
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError,match='independent label-free population manifest'):
        score(*paths,protocol,tmp_path/'outputs/unbound-population')
