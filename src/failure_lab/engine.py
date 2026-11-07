import copy
import hashlib
import json
import platform
import os
import importlib.metadata
import random
import subprocess
import time
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import torch
from torch import nn
from PIL import ImageOps, ImageEnhance
from .data import FACTORS, registry, render, intervene, quality, standard_variant
from .metrics import metrics, temperature, bootstrap, paired_bootstrap

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
torch.backends.cudnn.benchmark=False
torch.backends.cudnn.deterministic=True
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def factory(name,pretrained):
    if name=='tiny_cnn':
        return nn.Sequential(nn.Conv2d(3,16,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(16,32,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(32,1)),dict(mean=[.485,.456,.406],std=[.229,.224,.225],interpolation='bicubic')
    import timm
    model=timm.create_model(name,pretrained=pretrained,num_classes=1)
    cfg=timm.data.resolve_model_data_config(model)
    pretraining={}
    if pretrained:
        from huggingface_hub import try_to_load_from_cache
        repo=model.pretrained_cfg.get('hf_hub_id')
        cached=try_to_load_from_cache(repo,'model.safetensors') if repo else None
        if isinstance(cached,str):
            pretraining={'repo':repo,'revision':Path(cached).parent.name,'weights_sha256':hashlib.sha256(Path(cached).read_bytes()).hexdigest()}
    return model,dict(mean=list(cfg['mean']),std=list(cfg['std']),interpolation=cfg['interpolation'],pretrained_cfg=model.pretrained_cfg,pretraining=pretraining)

def tensor(image,preprocessing):
    a=np.asarray(image,dtype=np.float32).copy()/255
    a=(a-np.array(preprocessing['mean']))/np.array(preprocessing['std'])
    return torch.from_numpy(a.transpose(2,0,1).astype('float32'))

_TENSOR_CACHE={}
def materialize(rows,size,preprocessing,variant=None):
    key=(tuple(rows),size,tuple(preprocessing['mean']),tuple(preprocessing['std']),variant)
    cacheable=len(rows)>=1000
    if cacheable and key in _TENSOR_CACHE: return _TENSOR_CACHE[key]
    def prepare(obj):
        im=render(obj,size)[0]
        if variant is not None:
            if variant in FACTORS:
                if not quality(obj,variant,size)['valid']: raise ValueError('Invalid repair')
                im=render(intervene(obj,variant),size)[0]
            else: im=standard_variant(obj,variant,size)
        return tensor(im,preprocessing)
    with ThreadPoolExecutor(max_workers=int(os.environ.get('FAILURE_LAB_WORKERS','4'))) as pool:
        xs=list(pool.map(prepare,rows))
    result=(torch.stack(xs),torch.tensor([o.label for o in rows],dtype=torch.float32))
    if cacheable: _TENSOR_CACHE[key]=result
    return result

def predict(model,x,batch=32):
    model.eval(); out=[]
    with torch.inference_mode():
        for chunk in x.split(batch): out.extend(model(chunk.to(next(model.parameters()).device)).flatten().cpu().tolist())
    return np.array(out)

def group_ids(rows):
    return [f'{o.label}{o.background}{o.watermark}{o.lighting}' for o in rows]

def fit(model,x,y,cfg,steps,seed,validation,extra=None,warmup=False):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    rng=np.random.default_rng(seed); model.to(DEVICE)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    history=[]; best=None; best_score=(-1,-1); started=time.perf_counter()
    head=model.get_classifier() if hasattr(model,'get_classifier') else model[-1]
    head_ids={id(p) for p in head.parameters()}
    scaler=torch.amp.GradScaler('cuda',enabled=DEVICE.type=='cuda')
    if DEVICE.type=='cuda': torch.cuda.reset_peak_memory_stats()
    for step in range(steps):
        for p in model.parameters(): p.requires_grad_(not warmup or step>=cfg['head_steps'] or id(p) in head_ids)
        model.train()
        count=cfg['batch_size']; original=count if extra is None else count//2
        idx=rng.integers(len(x),size=original); xb=x[idx]; yb=y[idx]
        if extra is not None:
            ex,ey=extra; ix=rng.integers(len(ex),size=count-original)
            xb=torch.cat((xb,ex[ix])); yb=torch.cat((yb,ey[ix]))
        optimizer.zero_grad()
        for group in optimizer.param_groups: group['lr']=cfg['learning_rate']*.5*(1+np.cos(np.pi*step/steps))
        with torch.autocast(device_type=DEVICE.type,enabled=DEVICE.type=='cuda'):
            loss=nn.functional.binary_cross_entropy_with_logits(model(xb.to(DEVICE)).flatten(),yb.to(DEVICE))
        scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        if (step+1)%max(1,steps//3)==0 or step==steps-1:
            vx,vy,vg,mx,my=validation
            probs=1/(1+np.exp(-predict(model,vx)))
            score=(metrics(vy,probs,vg)['worst_group_accuracy'],float(((predict(model,mx)>=0)==np.asarray(my)).mean()))
            print(f'Train step {step+1}/{steps}: challenge WGA={score[0]:.3f}, matched={score[1]:.3f}',flush=True)
            history.append(dict(step=step+1,loss=float(loss.detach()),challenge_worst_group=score[0],matched_accuracy=score[1]))
            if score>best_score: best_score=score; best=copy.deepcopy(model.state_dict())
    model.load_state_dict(best)
    for p in model.parameters(): p.requires_grad_(True)
    return dict(history=history,steps=steps,duration_seconds=time.perf_counter()-started,precision='cuda-amp' if DEVICE.type=='cuda' else 'float32',peak_gpu_bytes=torch.cuda.max_memory_allocated() if DEVICE.type=='cuda' else None)

def audit(model,rows,cfg,prep):
    if any(o.split not in ('audit_dev','candidates') for o in rows): raise ValueError('Audit split is protected')
    base=predict(model,materialize(rows,cfg['image_size'],prep)[0]); p=1/(1+np.exp(-base))
    deltas={}; edited={}
    for f in FACTORS:
        with ThreadPoolExecutor(max_workers=int(os.environ.get('FAILURE_LAB_WORKERS','4'))) as pool:
            q=list(pool.map(lambda o:quality(o,f,cfg['image_size']),rows))
        if not all(v['valid'] for v in q): raise ValueError('Invalid audit')
        edited[f]=1/(1+np.exp(-predict(model,materialize(rows,cfg['image_size'],prep,f)[0])))
        deltas[f]=abs(edited[f]-p)
    # Selection contract contains no labels or test metadata.
    scores=[dict(base_object_id=o.base_object_id,score=float(np.mean([deltas[f][i] for f in FACTORS]))) for i,o in enumerate(rows)]
    records=[dict(base_object_id=o.base_object_id,label=o.label,correct=bool((p[i]>=.5)==o.label),probability=float(p[i]),edits={f:dict(probability=float(edited[f][i]),correct=bool((edited[f][i]>=.5)==o.label),delta=float(deltas[f][i]),valid=True,quality=quality(o,f,cfg['image_size'])) for f in FACTORS}) for i,o in enumerate(rows)]
    summary={f:dict(mean_probability_change=float(deltas[f].mean()),class_flip_rate=float(((p>=.5)!=(edited[f]>=.5)).mean()),harmful_flips=int((((p>=.5)==np.array([o.label for o in rows])) & ((edited[f]>=.5)!=np.array([o.label for o in rows]))).sum()),beneficial_flips=int((((p>=.5)!=np.array([o.label for o in rows])) & ((edited[f]>=.5)==np.array([o.label for o in rows]))).sum())) for f in FACTORS}
    return scores,records,summary

def select(scores,budget,method,seed):
    if budget<=0 or budget%4: raise ValueError('Budget must be positive and divisible by four')
    if any(set(s)!={'base_object_id','score'} for s in scores): raise ValueError('Selector accepts only IDs and sensitivity scores')
    if len({s['base_object_id'] for s in scores})!=len(scores): raise ValueError('Duplicate candidates')
    n=budget//4
    if n>len(scores): raise ValueError('Insufficient candidates')
    if method=='targeted': chosen=sorted(scores,key=lambda s:(-s['score'],s['base_object_id']))[:n]
    else: chosen=[scores[i] for i in np.random.default_rng(seed).choice(len(scores),n,replace=False)]
    return [s['base_object_id'] for s in chosen]

def run(config,root):
    root=Path(root); root.mkdir(parents=True,exist_ok=True)
    source_hash=hashlib.sha256(b''.join(p.read_bytes() for p in sorted(Path(__file__).parent.glob('*.py')))).hexdigest()
    protocol=json.dumps(config,sort_keys=True); config_hash=hashlib.sha256(protocol.encode()).hexdigest()
    frozen=root/'protocol.json'
    if frozen.exists() and json.loads(frozen.read_text())!=config: raise ValueError('Protocol changed: use a fresh experiment ID')
    source_file=root/'source-hash.txt'
    if source_file.exists() and source_file.read_text()!=source_hash: raise ValueError('Source changed: use a fresh experiment ID')
    source_file.write_text(source_hash)
    frozen.write_text(json.dumps(config,indent=2))
    rows=registry(config); splits={s:[o for o in rows if o.split==s] for s in config['splits']}
    try: revision=subprocess.check_output(['git','rev-parse','HEAD'],stderr=subprocess.DEVNULL).decode().strip()
    except subprocess.CalledProcessError: revision='uncommitted'
    meta=dict(data_workers=int(os.environ.get('FAILURE_LAB_WORKERS','4')),render_cache_capacity=int(os.environ.get('FAILURE_LAB_RENDER_CACHE','256')),generator_source_sha256=hashlib.sha256(Path(__file__).with_name('data.py').read_bytes()).hexdigest(),dependencies={n:importlib.metadata.version(n) for n in ('torch','torchvision','timm','numpy','Pillow','PyYAML')},source_hash=source_hash,config_hash=config_hash,source_revision=revision,python=platform.python_version(),torch=torch.__version__,device=str(DEVICE),hardware=platform.processor(),profile=config['profile'],timestamp=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    (root/'environment.json').write_text(json.dumps(meta,indent=2))
    results=[]
    for architecture in config['architectures']:
        _TENSOR_CACHE.clear()
        train_cfg={**config,**config.get('model_settings',{}).get(architecture,{})}
        for seed in config['seeds']:
            torch.manual_seed(seed)
            model,prep=factory(architecture,config['pretrained']); model.to(DEVICE)
            data={s:materialize(rs,config['image_size'],prep) for s,rs in splits.items() if not s.endswith('_test') and s!='candidates'}
            vx,vy=data['challenge_val']; mx,my=data['matched_val']
            validation=(vx,vy.numpy(),group_ids(splits['challenge_val']),mx,my.numpy())
            base_checkpoint=root/f'{architecture}-seed{seed}-baseline-selected.pt'
            original_result=root/f'{architecture}-seed{seed}-original-B0-result.json'
            if original_result.exists():
                previous=json.loads(original_result.read_text())
                saved=torch.load(root/Path(previous['checkpoint']).name,map_location=DEVICE,weights_only=False)
                model.load_state_dict(saved['state_dict']); info=previous['training']
            elif base_checkpoint.exists():
                saved=torch.load(base_checkpoint,map_location=DEVICE,weights_only=False)
                model.load_state_dict(saved['state_dict']); info=saved['training']
            else:
                info=fit(model,*data['train'],train_cfg,config['train_steps'],seed,validation,warmup=True)
                torch.save(dict(state_dict=model.state_dict(),training=info,preprocessing=prep,architecture=architecture),base_checkpoint)
            original=copy.deepcopy(model.state_dict())
            audit_start=time.perf_counter()
            scores,_,_=audit(model,splits['candidates'],config,prep)
            candidate_scoring_seconds=time.perf_counter()-audit_start
            _,records,summary=audit(model,splits['audit_dev'],config,prep)
            prefix=f'{architecture}-seed{seed}'
            (root/f'{prefix}-audit.json').write_text(json.dumps(dict(records=records,summary=summary),indent=2))
            (root/f'{prefix}-candidate-scores.json').write_text(json.dumps(scores,indent=2))
            conditions=[('original',0),('continuation',0)]+[(m,b) for b in config['budgets'] for m in ('random_standard','random_nuisance','targeted')]
            predictions={}
            for method,budget in conditions:
                existing_result=root/f'{prefix}-{method}-B{budget}-result.json'
                if existing_result.exists():
                    result=json.loads(existing_result.read_text())
                    checkpoint_path=root/Path(result['checkpoint']).name
                    if hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()!=result['checkpoint_hash']: raise ValueError('Checkpoint hash mismatch')
                    predictions[(method,budget)]=json.loads((root/f'{prefix}-{method}-B{budget}-decorrelated_test-predictions.json').read_text())
                    results.append(result)
                    print(f'Resumed {result["run_id"]}',flush=True)
                    continue
                model.load_state_dict(original); selected=[]; repair_manifest=[]; training=info
                if method not in ('original','continuation'):
                    selected=select(scores,budget,method,seed)
                    parents=[o for o in splits['candidates'] if o.base_object_id in selected]
                    policy=('flip','rotate','contrast') if method=='random_standard' else FACTORS
                    packs=[materialize(parents,config['image_size'],prep,v) for v in (None,*policy)]
                    extra=(torch.cat([p[0] for p in packs]),torch.cat([p[1] for p in packs]))
                    assert len(extra[0])==budget
                    repair_manifest=[dict(image_id=f"{o.base_object_id}:{v or 'original'}",base_object_id=o.base_object_id,label=o.label,render_seed=o.seed,parent_image_id=o.base_object_id,intervention_factor=v,intervention_parameters={'policy':method},attributes=asdict(intervene(o,v) if v in FACTORS else o)) for v in (None,*policy) for o in parents]
                    training=fit(model,*data['train'],train_cfg,config['repair_steps'],seed,validation,extra=extra)
                elif method=='continuation':
                    training=fit(model,*data['train'],train_cfg,config['repair_steps'],seed,validation)
                cx,cy=data['calibration_val']; t=temperature(predict(model,cx),cy.numpy())
                run_id=f'{prefix}-{method}-B{budget}'; ckpt=root/f'{run_id}.pt'
                torch.save(dict(state_dict=model.state_dict(),architecture=architecture,preprocessing=prep,temperature=t,config=config,run_id=run_id),ckpt)
                checkpoint_hash=hashlib.sha256(ckpt.read_bytes()).hexdigest()
                evaluation={}
                for split in ('matched_test','decorrelated_test','inverted_test','unfamiliar_test'):
                    tx,ty=materialize(splits[split],config['image_size'],prep)
                    start=time.perf_counter(); logits=predict(model,tx); latency=(time.perf_counter()-start)/len(tx)
                    raw=1/(1+np.exp(-logits)); calibrated=1/(1+np.exp(-logits/t)); groups=group_ids(splits[split])
                    evaluation[split]=dict(raw=metrics(ty,raw,groups),calibrated=metrics(ty,calibrated,groups),bootstrap_note='Object bootstrap, discarding resamples missing predefined observed groups; intervals exclude training variability',bootstrap_worst_group_95=bootstrap(ty,raw,groups,config['bootstrap_samples'],seed),latency_seconds_per_image=latency)
                    prediction=dict(base_object_ids=[o.base_object_id for o in splits[split]],labels=ty.tolist(),raw_probabilities=raw.tolist(),calibrated_probabilities=calibrated.tolist(),groups=groups)
                    (root/f'{run_id}-{split}-predictions.json').write_text(json.dumps(prediction))
                    if split=='decorrelated_test': predictions[(method,budget)]=prediction
                result=dict(run_id=run_id,architecture=architecture,seed=seed,method=method,budget=budget,selected_parents=selected,repair_manifest=repair_manifest,checkpoint=ckpt.name,checkpoint_hash=checkpoint_hash,temperature=t,training=training,evaluation=evaluation,candidate_scoring_seconds=candidate_scoring_seconds,preprocessing=prep,training_settings=train_cfg,metadata=meta)
                (root/f'{run_id}-result.json').write_text(json.dumps(result,indent=2))
                results.append(result)
                (root/'results.json').write_text(json.dumps(results,indent=2))
                report(root)
                print(f'Completed {run_id}',flush=True)
            paired=[]
            for budget in config['budgets']:
                a=predictions[('targeted',budget)]; b=predictions[('random_nuisance',budget)]
                paired.append(dict(budget=budget,seed=seed,architecture=architecture,difference=metrics(a['labels'],a['raw_probabilities'],a['groups'])['worst_group_accuracy']-metrics(b['labels'],b['raw_probabilities'],b['groups'])['worst_group_accuracy'],bootstrap_95=paired_bootstrap(a['labels'],a['raw_probabilities'],b['raw_probabilities'],a['groups'],config['bootstrap_samples'],seed)))
            (root/f'{prefix}-paired.json').write_text(json.dumps(paired,indent=2))
    (root/'results.json').write_text(json.dumps(results,indent=2))
    report(root)
    return results

def report(root):
    root=Path(root); results=json.loads((root/'results.json').read_text())
    lines=['# Experiment results','',f"Profile: **{results[0]['metadata']['profile']}**. Only runs recorded below were completed.",'','Bootstrap intervals describe object sampling, not training variability. Smoke results are pipeline verification, not evidence for the core research claim.','','| Architecture | Seed | Method | Budget | Matched accuracy | Decorrelated WGA | Inverted accuracy | Unfamiliar accuracy |','|---|---:|---|---:|---:|---:|---:|---:|']
    for r in results:
        e=r['evaluation']; lines.append(f"| {r['architecture']} | {r['seed']} | {r['method']} | {r['budget']} | {e['matched_test']['raw']['accuracy']:.3f} | {e['decorrelated_test']['raw']['worst_group_accuracy']:.3f} | {e['inverted_test']['raw']['accuracy']:.3f} | {e['unfamiliar_test']['raw']['accuracy']:.3f} |")
    lines.extend(['','## Seed aggregates','','| Architecture | Method | Budget | Mean WGA | Sample SD | Seeds |','|---|---|---:|---:|---:|---:|'])
    keys=sorted({(r['architecture'],r['method'],r['budget']) for r in results})
    for a,m,b in keys:
        values=[r['evaluation']['decorrelated_test']['raw']['worst_group_accuracy'] for r in results if (r['architecture'],r['method'],r['budget'])==(a,m,b)]
        sd=f'{np.std(values,ddof=1):.3f}' if len(values)>1 else 'not estimable'
        lines.append(f'| {a} | {m} | {b} | {np.mean(values):.3f} | {sd} | {len(values)} |')
    lines.extend(['','## Primary paired comparison','','Positive differences favour targeted nuisance selection. These intervals do not capture training variability.',''])
    for paired_file in sorted(root.glob('*-paired.json')):
        for row in json.loads(paired_file.read_text()):
            lines.append(f"- {row['architecture']}, seed {row['seed']}, B={row['budget']}: targeted minus random nuisance WGA = {row['difference']:+.3f}; object bootstrap 95% interval {row['bootstrap_95']}.")
    lines.extend(['','## Interpretation','','Sensitivity measures response to controlled nuisance edits, not complete internal reasoning. Inspect all seed differences and ordinary-accuracy trade-offs before claiming a repair gain. Synthetic generalisation does not establish real industrial performance.'])
    (root/'report.md').write_text('\n'.join(lines),encoding='utf-8')


