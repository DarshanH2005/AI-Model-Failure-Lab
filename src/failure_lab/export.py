"""Export reproducible development galleries and a narrated-by-captions demo."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from .data import registry,render,intervene,FACTORS
from .cli import load_config
from .engine import factory,materialize,predict
import torch

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--experiment',required=True); parser.add_argument('--output',required=True); parser.add_argument('--video',action='store_true'); args=parser.parse_args()
    root=Path(args.experiment); out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((root/'protocol.json').read_text()); rows=[o for o in registry(cfg) if o.split=='audit_dev']
    results=json.loads((root/'results.json').read_text())
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',28); small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',22)
    gallery=Image.new('RGB',(1280,720),(245,247,250)); d=ImageDraw.Draw(gallery); d.text((30,20),'Controlled tile defects and nuisance interventions',fill=(20,30,45),font=font)
    for i,obj in enumerate(rows[:8]):
        x=30+(i%4)*310; y=85+(i//4)*300
        gallery.paste(render(obj,224)[0],(x,y)); d.text((x,y+235),f'{obj.split}:{obj.base_object_id.split(":")[-1]} | label {obj.label}',fill=(20,30,45),font=small)
    gallery.save(out/'dataset-gallery.png')
    original=next(r for r in results if r['method']=='original')
    repaired=next(r for r in results if r['method']=='targeted' and r['architecture']==original['architecture'] and r['seed']==original['seed'])
    repair_ck=torch.load(root/Path(repaired['checkpoint']).name,map_location='cpu',weights_only=False)
    repair_model,_=factory(repair_ck['architecture'],False); repair_model.load_state_dict(repair_ck['state_dict'])
    ck=torch.load(root/Path(original['checkpoint']).name,map_location='cpu',weights_only=False)
    model,_=factory(ck['architecture'],False); model.load_state_dict(ck['state_dict']); torch.set_num_threads(4)
    pairs=Image.new('RGB',(1280,720),(245,247,250)); d=ImageDraw.Draw(pairs)
    d.text((30,20),'One development object; only its nuisance cue changes',fill=(20,30,45),font=font)
    obj=rows[1]
    for i,(factor,value) in enumerate([('original',obj)]+[(f,intervene(obj,f)) for f in FACTORS]):
        x=30+i*310; im=render(value,224)[0]; pairs.paste(im,(x,120))
        logit=predict(model,materialize([value],cfg['image_size'],ck['preprocessing'])[0])[0]
        p=1/(1+np.exp(-logit))
        repaired_logit=predict(repair_model,materialize([value],cfg['image_size'],repair_ck['preprocessing'])[0])[0]
        repaired_p=1/(1+np.exp(-repaired_logit))
        d.text((x,370),factor,fill=(20,30,45),font=font); d.text((x,415),f'Original raw p: {p:.3f}',fill=(20,30,45),font=small); d.text((x,450),f'Repaired raw p: {repaired_p:.3f}',fill=(20,30,45),font=small)
    d.text((30,510),f"Run: {original['run_id']}",fill=(20,30,45),font=small)
    d.text((30,555),'Sensitivity alone is not evidence of an error; true label = 1.',fill=(20,30,45),font=small)
    pairs.save(out/'intervention-gallery.png')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,len(cfg['architectures']),figsize=(12,5),squeeze=False)
    for ax,architecture in zip(axes[0],cfg['architectures']):
        for method in ('random_standard','random_nuisance','targeted'):
            means=[]; deviations=[]
            for budget in cfg['budgets']:
                values=[r['evaluation']['decorrelated_test']['raw']['worst_group_accuracy'] for r in results if r['architecture']==architecture and r['method']==method and r['budget']==budget]
                means.append(np.mean(values)); deviations.append(np.std(values,ddof=1) if len(values)>1 else 0)
            ax.errorbar(cfg['budgets'],means,yerr=deviations,marker='o',capsize=3,label=method)
        ax.set_title(architecture); ax.set_xlabel('Additional labelled images'); ax.set_ylim(-.05,1.05); ax.grid(alpha=.2); ax.legend(fontsize=8)
    axes[0][0].set_ylabel('Decorrelated worst-group accuracy'); fig.suptitle(f"{cfg['profile']} study: mean and sample SD over seeds")
    fig.tight_layout(); fig.savefig(out/'budget-curves.png',dpi=150); plt.close(fig)
    if args.video:
        import imageio.v2 as imageio
        frames=[]
        texts=[
            ('Shortcut Detective','A controlled laboratory for vision-model shortcut failures.'),
            ('The true signal','Scratches, cracks and chips determine defect labels.'),
            ('Misleading cues','Training correlates background, watermark and brightness with labels.'),
            ('Controlled interventions','Change one cue while preserving object and defect identity.'),
            ('Four repair conditions','Continuation, random standard, random nuisance and targeted nuisance.'),
            ('Fair comparison','Targeted and random nuisance use identical edits and image budgets.'),
            ('Measured results',f"{cfg['profile'].upper()} profile; {len(results)} saved runs. See the accompanying report."),
            ('Limits and unresolved failures','Synthetic results do not establish industrial reliability. No guaranteed repair gain.')]
        for title,description in texts:
            frame=Image.new('RGB',(1280,720),(245,247,250)); dr=ImageDraw.Draw(frame)
            dr.text((50,60),title,fill=(20,30,45),font=font); dr.text((50,120),description,fill=(20,30,45),font=small)
            if title in ('The true signal','Misleading cues'): frame.paste(gallery.resize((960,540)),(50,170))
            elif title in ('Controlled interventions','Fair comparison'): frame.paste(pairs.resize((960,540)),(50,170))
            elif title=='Measured results': frame.paste(Image.open(out/'budget-curves.png').resize((1100,460)),(50,200))
            else:
                yy=240
                for r in results[:5]:
                    dr.text((50,yy),f"{r['method']} B={r['budget']}: matched {r['evaluation']['matched_test']['raw']['accuracy']:.3f}, decorrelated WGA {r['evaluation']['decorrelated_test']['raw']['worst_group_accuracy']:.3f}",fill=(20,30,45),font=small); yy+=60
            frames.append(np.asarray(frame))
        with imageio.get_writer(out/'demo.mp4',fps=2,codec='libx264',quality=7,macro_block_size=16) as writer:
            for frame in frames:
                for _ in range(40): writer.append_data(frame)
if __name__=='__main__': main()


