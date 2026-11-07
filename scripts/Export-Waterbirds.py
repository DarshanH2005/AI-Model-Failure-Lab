"""Read-only figures from a completed external study; no training or selection."""
import json, os, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(os.environ['LOCALAPPDATA'])/'ShortcutDetective/waterbirds-v1'
results=json.loads((root/'results.json').read_text())
assert len(results)==6
metadata=pd.read_csv(root/'waterbird_complete95_forest2water2/metadata.csv')
test_metadata=metadata[metadata.split==2].set_index('img_filename')
train=metadata[metadata.split==0]
train_weights={str(int(y))+str(int(place)):len(rows)/len(train) for (y,place),rows in train.groupby(['y','place'])}
for row in results:
    assert hashlib.sha256((root/row['checkpoint']).read_bytes()).hexdigest()==row['checkpoint_sha256']
    predictions=json.loads((root/(row['run_id']+'-predictions.json')).read_text())
    assert len(predictions)==5794 and len({p['img_filename'] for p in predictions})==5794
    assert len(row['evaluation']['raw']['groups'])==4
    assert {p['img_filename'] for p in predictions}==set(test_metadata.index)
    for p in predictions:
        expected=test_metadata.loc[p['img_filename']]
        assert p['label']==int(expected.y) and p['background']==int(expected.place)
        assert np.isfinite(p['raw_probability']) and np.isfinite(p['calibrated_probability'])
    correct=np.array([(p['raw_probability']>=.5)==p['label'] for p in predictions])
    actual_groups=np.array([str(p['label'])+str(p['background']) for p in predictions])
    group_scores={g:float(correct[actual_groups==g].mean()) for g in train_weights}
    raw=row['evaluation']['raw']
    assert np.isclose(raw['accuracy'],correct.mean())
    assert np.isclose(raw['worst_group_accuracy'],min(group_scores.values()))
    assert np.isclose(raw['adjusted_accuracy'],sum(train_weights[g]*group_scores[g] for g in train_weights))
    for g,score in group_scores.items():
        assert raw['groups'][g]['count']==int((actual_groups==g).sum())
        assert np.isclose(raw['groups'][g]['accuracy'],score)
out=root/'media'; out.mkdir(exist_ok=True)
architectures=sorted({r['architecture'] for r in results})
fig,ax=plt.subplots(figsize=(11,5))
for i,metric in enumerate(('accuracy','adjusted_accuracy','worst_group_accuracy')):
    values=[[r['evaluation']['raw'][metric] for r in results if r['architecture']==a] for a in architectures]
    ax.bar(np.arange(2)+(i-1)*.23,[np.mean(v) for v in values],width=.23,yerr=[np.std(v,ddof=1) for v in values],capsize=4,label=metric.replace('_',' '))
ax.set_xticks(range(2),['DeiT-Tiny','ResNet-18']); ax.set_ylim(0,1.05); ax.set_ylabel('Official test accuracy'); ax.set_title('Waterbirds: mean and sample SD over three training seeds'); ax.legend(); fig.tight_layout(); fig.savefig(out/'group-robustness.png',dpi=150); plt.close(fig)
row=results[0]
predictions=json.loads((root/(row['run_id']+'-predictions.json')).read_text())
failures=[p for p in predictions if (p['raw_probability']>=.5)!=p['label']]
chosen=[]
for group in ((0,0),(0,1),(1,0),(1,1)):
    chosen.extend([p for p in failures if (p['label'],p['background'])==group][:2])
for p in failures:
    if len(chosen)>=8: break
    if p not in chosen: chosen.append(p)
font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',25)
small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',19)
gallery=Image.new('RGB',(1280,800),(245,247,250)); draw=ImageDraw.Draw(gallery)
draw.text((25,20),'Observed Waterbirds errors: official test, first saved model',font=font,fill='black')
draw.text((25,60),f"{row['run_id']} | illustrative examples, not a random sample",font=small,fill='black')
for i,p in enumerate(chosen):
    x=25+(i%4)*315; y=110+(i//4)*310
    with Image.open(root/'waterbird_complete95_forest2water2'/p['img_filename']) as im:
        im=im.convert('RGB'); im.thumbnail((285,215)); gallery.paste(im,(x,y))
    bird='waterbird' if p['label'] else 'landbird'; background='water' if p['background'] else 'land'
    predicted='waterbird' if p['raw_probability']>=.5 else 'landbird'
    draw.text((x,y+225),f'True: {bird}; bg: {background}',font=small,fill='black')
    draw.text((x,y+250),f'Pred: {predicted}; p={p["raw_probability"]:.3f}',font=small,fill='black')
draw.text((25,750),'Local research/educational use. No interventions or test-driven retraining.',font=small,fill='black')
gallery.save(out/'observed-errors.png')
(out/'gallery-manifest.json').write_text(json.dumps({'run_id':row['run_id'],'split':'official test','selection':'illustrative observed misclassifications','examples':chosen},indent=2))
print('External checkpoints, test counts and figures verified.')


interpretation=['','## Interpretation','']
for architecture in architectures:
    subset=[r for r in results if r['architecture']==architecture]
    overall=np.mean([r['evaluation']['raw']['accuracy'] for r in subset])
    worst=np.mean([r['evaluation']['raw']['worst_group_accuracy'] for r in subset])
    interpretation.append(f'- {architecture}: mean unweighted test accuracy {overall:.2%}, mean worst-group accuracy {worst:.2%}; gap {overall-worst:.2%}.')
interpretation.append('The lower group-specific performance demonstrates that high average performance can conceal failures under a changed background distribution. The scores alone do not prove an internal causal mechanism. This study did not test the targeted repair method on photographic images because valid foreground interventions were unavailable. The next repair-transfer study requires validated bird masks and a new protected evaluation protocol; no new training is selected from these test outcomes.')
report=root/'report.md'
text=report.read_text(encoding='utf-8').split('\n## Interpretation')[0]
report.write_text(text+'\n'.join(interpretation),encoding='utf-8')
(root/'artifact-verification.json').write_text(json.dumps({'verified_checkpoints':6,'prediction_rows':6*5794,'official_metadata_and_group_metrics_match':True,'gallery_source':'official test; after frozen evaluation','passed':True},indent=2))
