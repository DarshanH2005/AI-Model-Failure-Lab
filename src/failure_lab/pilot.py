"""Validation-only tuning and decorrelated learnability controls; never read tests."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from .cli import load_config
from .data import registry
from .engine import factory, materialize, fit, group_ids, predict
from .metrics import metrics

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/pilot.yaml'); p.add_argument('--root',required=True); args=p.parse_args()
    cfg=load_config(args.config); root=Path(args.root); root.mkdir(parents=True,exist_ok=True); torch.set_num_threads(4)
    rows=registry(cfg); splits={s:[o for o in rows if o.split==s] for s in ('train','challenge_val','matched_val')}
    results=[]; controls=[]
    for architecture in ('resnet18.a1_in1k','deit_tiny_patch16_224.fb_in1k'):
        model,prep=factory(architecture,True)
        train=materialize(splits['train'],cfg['image_size'],prep); vx,vy=materialize(splits['challenge_val'],cfg['image_size'],prep); mx,my=materialize(splits['matched_val'],cfg['image_size'],prep)
        val=(vx,vy.numpy(),group_ids(splits['challenge_val']),mx,my.numpy())
        for lr in (1e-4,3e-4):
            for wd in (.01,.1):
                torch.manual_seed(11); model,_=factory(architecture,True)
                candidate={**cfg,'learning_rate':lr,'weight_decay':wd}
                info=fit(model,*train,candidate,cfg['train_steps'],11,val,warmup=True)
                probs=1/(1+np.exp(-predict(model,vx)))
                m=metrics(vy,probs,val[2])
                row=dict(architecture=architecture,lr=lr,weight_decay=wd,challenge_worst_group=m['worst_group_accuracy'],challenge_accuracy=m['accuracy'],matched_accuracy=float(((predict(model,mx)>=0)==my.numpy()).mean()),training=info)
                results.append(row); (root/'tuning.json').write_text(json.dumps(results,indent=2))
                print(f"Tuning {architecture} lr={lr} wd={wd}: WGA={row['challenge_worst_group']:.3f}",flush=True)
        best=max([r for r in results if r['architecture']==architecture],key=lambda r:(r['challenge_worst_group'],r['matched_accuracy']))
        control={**cfg,'training_correlation':.5,'learning_rate':best['lr'],'weight_decay':best['weight_decay']}
        cr=[o for o in registry(control) if o.split=='train']
        torch.manual_seed(11); model,_=factory(architecture,True)
        info=fit(model,*materialize(cr,cfg['image_size'],prep),control,120,11,val,warmup=True)
        probs=1/(1+np.exp(-predict(model,vx))); m=metrics(vy,probs,val[2])
        controls.append(dict(architecture=architecture,validation=m,training=info))
        (root/'learnability.json').write_text(json.dumps(controls,indent=2))
        print(f"Decorrelated learnability {architecture}: accuracy={m['accuracy']:.3f}, WGA={m['worst_group_accuracy']:.3f}",flush=True)
    (root/'selected-settings.json').write_text(json.dumps({a:max([r for r in results if r['architecture']==a],key=lambda r:(r['challenge_worst_group'],r['matched_accuracy'])) for a in ('resnet18.a1_in1k','deit_tiny_patch16_224.fb_in1k')},indent=2))

if __name__=='__main__': main()

