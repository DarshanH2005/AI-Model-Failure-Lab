"""Separate, frozen-protocol Waterbirds group-robustness validation."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import sys, json, hashlib, random, copy
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import torch
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from failure_lab.engine import factory, tensor, fit, predict
from failure_lab.metrics import metrics, temperature, bootstrap
ROOT=Path(os.environ['LOCALAPPDATA'])/'ShortcutDetective/waterbirds-v1'
PROJECT=Path(__file__).resolve().parents[1]
ARCHS=['resnet18.a1_in1k','deit_tiny_patch16_224.fb_in1k']
SEEDS=[11,22,33]
CFG=dict(learning_rate=1e-4,weight_decay=.01,batch_size=32,head_steps=100)
torch.set_num_threads(4)

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,value): p.write_text(json.dumps(value,indent=2),encoding='utf-8')
def validation_partition(frame):
    rng=np.random.default_rng(20261005); selected=[]; calibrated=[]
    for _, rows in frame.groupby(['y','place']):
        ids=rows.index.to_numpy().copy(); rng.shuffle(ids)
        selected.extend(ids[:len(ids)//2]); calibrated.extend(ids[len(ids)//2:])
    return sorted(map(int,selected)), sorted(map(int,calibrated))
def groups(frame): return [str(int(y))+str(int(p)) for y,p in zip(frame.y,frame.place)]
def arrays(frame,data,prep):
    def convert(name):
        with Image.open(data/name) as im: return tensor(im.convert('RGB').resize((224,224),Image.Resampling.BICUBIC),prep)
    with ThreadPoolExecutor(max_workers=4) as pool: images=list(pool.map(convert,frame.img_filename))
    return torch.stack(images),torch.tensor(frame.y.to_numpy(),dtype=torch.float32)
def evaluate(frame,logits,t,train_weights,seed):
    gs=groups(frame); y=frame.y.to_numpy(); p=1/(1+np.exp(-logits)); cp=1/(1+np.exp(-logits/t))
    raw=metrics(y,p,gs); calibrated=metrics(y,cp,gs)
    for obj in (raw,calibrated):
        obj['waterbird_recall']=obj.pop('defect_recall')
        obj['adjusted_accuracy']=sum(train_weights[g]*obj['groups'][g]['accuracy'] for g in train_weights)
    return {'raw':raw,'calibrated':calibrated,'worst_group_bootstrap_95':bootstrap(y,p,gs,1000,seed)},p,cp

def main():
    data=ROOT/'waterbird_complete95_forest2water2'
    frame=pd.read_csv(data/'metadata.csv')
    assert set(frame.split)=={0,1,2} and set(frame.y)=={0,1} and set(frame.place)=={0,1}
    assert frame.img_filename.is_unique
    train=frame[frame.split==0]; val=frame[frame.split==1]; test=frame[frame.split==2]
    selection_ids,calibration_ids=validation_partition(val)
    selection=frame.loc[selection_ids]; calibration=frame.loc[calibration_ids]
    assert not set(selection_ids)&set(calibration_ids)
    assert len(selection)+len(calibration)==len(val)
    protocol=PROJECT/'docs/waterbirds_protocol.md'
    code_files=[Path(__file__),PROJECT/'src/failure_lab/engine.py',PROJECT/'src/failure_lab/metrics.py']
    frozen={'protocol_sha256':sha(protocol),'source_sha256':{str(p.relative_to(PROJECT)):sha(p) for p in code_files},'metadata_sha256':sha(data/'metadata.csv'),'archive_sha256':sha(ROOT/'waterbirds.tar.gz'),'architectures':ARCHS,'seeds':SEEDS,'steps':1500,'config':CFG,'selection_ids':selection_ids,'calibration_ids':calibration_ids}
    freeze_path=ROOT/'frozen-protocol.json'
    if freeze_path.exists(): assert json.loads(freeze_path.read_text())==frozen,'Protocol or source changed; use a new study version.'
    else: save(freeze_path,frozen)
    (ROOT/'protocol.md').write_bytes(protocol.read_bytes())
    integrity_path=ROOT/'dataset-integrity.json'
    if not integrity_path.exists():
        def check(name):
            path=data/name
            with Image.open(path) as image: image.verify()
            return sha(path)
        with ThreadPoolExecutor(max_workers=4) as pool: hashes=list(pool.map(check,frame.img_filename))
        seen={}
        for split,digest in zip(frame.split,hashes):
            assert digest not in seen or seen[digest]==int(split),'Exact duplicate across official splits'
            seen[digest]=int(split)
        save(integrity_path,{'objects':len(frame),'split_counts':{str(k):len(v) for k,v in frame.groupby('split')},'selection_count':len(selection),'calibration_count':len(calibration),'distinct_paths':True,'cross_split_exact_hash_overlap':False,'images_verified':len(frame),'group_counts':{str(s):{g:gs.count(g) for g in sorted(set(gs))} for s,rows in frame.groupby('split') for gs in [groups(rows)]}})
        save(ROOT/'image-hashes.json',dict(zip(frame.img_filename,hashes)))
    train_groups=groups(train); weights={g:train_groups.count(g)/len(train_groups) for g in set(train_groups)}
    results=[]
    for architecture in ARCHS:
        prep=None; loaded=None
        for seed in SEEDS:
            rid=f'{architecture}-seed{seed}'; result_path=ROOT/f'{rid}-result.json'
            if result_path.exists():
                row=json.loads(result_path.read_text()); assert sha(ROOT/row['checkpoint'])==row['checkpoint_sha256']; results.append(row); continue
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            model,prep=factory(architecture,True)
            if loaded is None:
                print('Preparing photographic tensors:',architecture,flush=True)
                loaded=[arrays(part,data,prep) for part in (train,selection,calibration,test)]
            (tx,ty),(vx,vy),(cx,cy),(ex,ey)=loaded
            history=fit(model,tx,ty,CFG,1500,seed,(vx,vy,groups(selection),vx,vy),warmup=True)
            t=temperature(predict(model,cx),cy.numpy())
            checkpoint=ROOT/f'{rid}.pt'
            torch.save({'architecture':architecture,'state_dict':copy.deepcopy(model.cpu().state_dict()),'preprocessing':prep,'temperature':t,'training':history,'seed':seed,'task':'waterbird versus landbird','protocol_sha256':frozen['protocol_sha256']},checkpoint)
            model.to('cuda'); logits=predict(model,ex)
            evaluation,p,cp=evaluate(test,logits,t,weights,seed)
            predictions=[{'img_filename':str(name),'label':int(y),'background':int(place),'raw_probability':float(a),'calibrated_probability':float(b)} for name,y,place,a,b in zip(test.img_filename,test.y,test.place,p,cp)]
            save(ROOT/f'{rid}-predictions.json',predictions)
            row={'run_id':rid,'architecture':architecture,'seed':seed,'checkpoint':checkpoint.name,'checkpoint_sha256':sha(checkpoint),'preprocessing':prep,'temperature':t,'training':history,'evaluation':evaluation}
            save(result_path,row); results.append(row); save(ROOT/'results.json',results)
            print('Completed',rid,'test WGA',evaluation['raw']['worst_group_accuracy'],flush=True)
            del model
            torch.cuda.empty_cache()
        del loaded
    save(ROOT/'results.json',results)
    assert len(results)==6
    lines=['# Waterbirds external validation results','','Separate photographic-composite bird classification using official splits. This does not validate industrial defect detection or intervention-repair transfer. No foreground edits were applied.','','| Architecture | Seed | Unweighted accuracy | Adjusted accuracy | Worst-group accuracy | Raw ECE | Calibrated ECE |','|---|---:|---:|---:|---:|---:|---:|']
    for r in results:
        e=r['evaluation']; a=e['raw']; b=e['calibrated']
        lines.append(f"| {r['architecture']} | {r['seed']} | {a['accuracy']:.4f} | {a['adjusted_accuracy']:.4f} | {a['worst_group_accuracy']:.4f} | {a['ece']:.4f} | {b['ece']:.4f} |")
    lines+=['','## Training-seed variability','']
    for arch in ARCHS:
        subset=[r for r in results if r['architecture']==arch]
        for name in ('accuracy','adjusted_accuracy','worst_group_accuracy'):
            values=[r['evaluation']['raw'][name] for r in subset]
            lines.append(f'- {arch}, {name}: mean {np.mean(values):.4f}, sample SD {np.std(values,ddof=1):.4f} (3 seeds).')
    lines+=['','## Group counts and uncertainty','']
    for r in results:
        e=r['evaluation']; lines.append(f"- {r['run_id']}: groups {e['raw']['groups']}; worst-group object-bootstrap 95% interval {e['worst_group_bootstrap_95']}.")
    lines+=['','Adjusted accuracy weights test-group accuracies by official training-group proportions. Unweighted accuracy uses the actual test distribution. Bootstrap intervals describe test-object sampling, not all training variability. All six seeds/architectures were fixed before test evaluation. The finite 1500-step training budget and lack of valid masks limit conclusions; these are not published benchmark replications. See frozen protocol for preprocessing, known label issues and license sources.']
    (ROOT/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    import subprocess
    environment=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True)
    (ROOT/'environment.txt').write_text(environment,encoding='utf-8')
    save(ROOT/'release-verification.json',{'runs':6,'seeds':SEEDS,'architectures':ARCHS,'verified_checkpoint_hashes':6,'test_objects_per_run':len(test),'all_four_groups_present':all(len(r['evaluation']['raw']['groups'])==4 for r in results),'passed':True})
    print('External study complete',flush=True)
if __name__=='__main__': main()

