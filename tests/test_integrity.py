from dataclasses import replace
import numpy as np
import pytest
import torch
from failure_lab.cli import load_config
from failure_lab.data import registry, render, FACTORS, quality
from failure_lab.engine import select, factory, materialize, predict
from failure_lab.metrics import metrics

def test_determinism_split_integrity():
    cfg=load_config('configs/smoke.yaml'); rows=registry(cfg)
    assert len({o.seed for o in rows})==len(rows)
    assert len({o.base_object_id for o in rows})==len(rows)
    a=render(rows[1],64); b=render(rows[1],64)
    assert all(np.array_equal(x,y) for x,y in zip(a,b))
    assert rows[1].label==1 and np.asarray(a[2]).sum()>0

def test_intervention_boundaries_visibility():
    for obj in registry(load_config('configs/smoke.yaml'))[:32]:
        for factor in FACTORS: assert quality(obj,factor,64)['valid']

def test_budget_and_label_isolation():
    scores=[dict(base_object_id=str(i),score=i/20) for i in range(20)]
    assert len(set(select(scores,16,'targeted',11)))==4
    assert select(scores,16,'targeted',11)==['19','18','17','16']
    assert select(scores,16,'random_nuisance',11)==select(scores,16,'random_standard',11)
    with pytest.raises(ValueError): select(scores,5,'targeted',11)
    with pytest.raises(ValueError): select([dict(base_object_id='x',score=.2,label=1)],4,'targeted',1)
    with pytest.raises(ValueError): select(scores*2,4,'targeted',1)

def test_metrics_known_values():
    m=metrics([0,1,0,1],[.1,.9,.8,.7],['a','a','b','b'])
    assert m['accuracy']==.75 and m['worst_group_accuracy']==.5
    assert m['groups']['a']['count']==2
    assert abs(m['brier']-.1875)<1e-6

def test_checkpoint_roundtrip(tmp_path):
    cfg=load_config('configs/smoke.yaml'); model,prep=factory('tiny_cnn',False)
    x,_=materialize(registry(cfg)[:2],64,prep)
    expected=predict(model,x); torch.save(model.state_dict(),tmp_path/'model.pt')
    reloaded,_=factory('tiny_cnn',False); reloaded.load_state_dict(torch.load(tmp_path/'model.pt',weights_only=True))
    np.testing.assert_allclose(expected,predict(reloaded,x))

def test_end_to_end(tmp_path):
    from failure_lab.engine import run
    torch.set_num_threads(2)
    cfg=load_config('configs/smoke.yaml'); cfg['train_steps']=2; cfg['repair_steps']=1; cfg['head_steps']=0; cfg['bootstrap_samples']=2
    results=run(cfg,tmp_path)
    assert len(results)==5
    assert all(len(r['selected_parents'])*4==r['budget'] for r in results)
    assert all(len(r['evaluation']['decorrelated_test']['raw']['groups'])==16 for r in results)
    assert (tmp_path/'report.md').exists()


def test_cached_renderer_matches_uncached():
    cfg=load_config('configs/smoke.yaml')
    for obj in registry(cfg)[:8]:
        cached=render(obj,64)
        independent=render.__wrapped__(obj,64)
        assert all(np.array_equal(a,b) for a,b in zip(cached,independent))

def test_returned_images_do_not_mutate_cached_dataset():
    obj=registry(load_config('configs/smoke.yaml'))[1]
    first=render(obj,64)
    expected=np.asarray(first[0]).copy()
    first[0].putpixel((0,0),(255,0,255))
    np.testing.assert_array_equal(np.asarray(render(obj,64)[0]),expected)
