from dataclasses import dataclass, asdict, replace
from pathlib import Path
import hashlib
import os
from functools import lru_cache
import json
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageOps

FACTORS = ('background','watermark','lighting')
@dataclass(frozen=True)
class Object:
    base_object_id: str
    split: str
    seed: int
    label: int
    background: int
    watermark: int
    lighting: int
    unfamiliar: bool = False

def registry(config):
    rows=[]
    for si,(split,count) in enumerate(config['splits'].items()):
        rng=np.random.default_rng(config.get('registry_seed',0)+1000+si)
        for i in range(count):
            label=i%2
            if split in ('decorrelated_test','challenge_val','unfamiliar_test'):
                attrs=[(i//(2**(j+1)))%2 for j in range(3)]
            else:
                corr=.1 if split=='inverted_test' else config.get('training_correlation',.9)
                attrs=[label if rng.random()<corr else 1-label for _ in FACTORS]
            rows.append(Object(f'{config.get("registry_seed",0)}:{split}:{i}',split,config.get('registry_seed',0)+(si+1)*1000000+i,label,*attrs,split=='unfamiliar_test'))
    return rows

@lru_cache(maxsize=int(os.environ.get('FAILURE_LAB_RENDER_CACHE','256')))
def _render_bytes(obj,size=224):
    rng=np.random.default_rng(obj.seed)
    yy,xx=np.mgrid[:size,:size]
    colour=np.array((55,100,165) if obj.background else (165,95,50),dtype=float)
    texture=np.sin((xx+yy)*.7)*8 if obj.unfamiliar else rng.normal(0,3,(size,size))
    image=Image.fromarray(np.clip(colour+texture[:,:,None],0,255).astype('uint8'))
    foreground=Image.new('L',(size,size)); defect=foreground.copy()
    dx,dy=rng.uniform(-.03,.03,2)
    ImageDraw.Draw(foreground).rounded_rectangle((int(size*(.23+dx)),int(size*(.22+dy)),int(size*(.78+dx)),int(size*(.79+dy))),radius=int(size*rng.uniform(.04,.09)),fill=255)
    tile=np.clip(180+rng.normal(0,5,(size,size)),0,255).astype('uint8')
    image.paste(Image.fromarray(np.repeat(tile[:,:,None],3,axis=2)),mask=foreground)
    draw=ImageDraw.Draw(image)
    for _ in range(3):
        x,y=rng.integers(int(size*.32),int(size*.65),2)
        draw.line((int(x),int(y),int(x+size*.08),int(y+size*.03)),fill=(151,151,151),width=1)
    if obj.label:
        x=int(rng.uniform(.35,.55)*size)
        points=[(x,int(size*.33)),(x+int(size*.07),int(size*.46)),(x-int(size*.03),int(size*.58)),(x+int(size*.08),int(size*.68))]
        width=max(2,int(size*.018))
        kind=obj.seed%3
        if kind==0:
            points=[(x,int(size*.35)),(x+int(size*.10),int(size*.65))]
        if kind==2:
            edge=int(size*(.76+dx)); y=int(size*.55)
            points=[(edge,y-int(size*.07)),(edge-int(size*.08),y),(edge,y+int(size*.07))]
            draw.polygon(points,fill=(35,35,35))
            ImageDraw.Draw(defect).polygon(points,fill=255)
        else:
            draw.line(points,fill=(35,35,35),width=width)
            ImageDraw.Draw(defect).line(points,fill=255,width=width)
    angle=float(np.random.default_rng(obj.seed+917).uniform(-12,12))
    surface=Image.new('RGB',(size,size),(180,180,180)); surface.paste(image,mask=foreground)
    surface=surface.rotate(angle,resample=3)
    foreground=foreground.rotate(angle,resample=0); defect=defect.rotate(angle,resample=0)
    image.paste(surface,mask=foreground); draw=ImageDraw.Draw(image)
    corner=(int(size*.03),int(size*.03),int(size*.15),int(size*.15))
    colour=(245,240,225) if obj.watermark else (20,25,35)
    (draw.ellipse if obj.unfamiliar else draw.rectangle)(corner,fill=colour)
    image=ImageEnhance.Brightness(image).enhance((1.16 if obj.lighting else .84) if obj.unfamiliar else (1.12 if obj.lighting else .88))
    return image.tobytes(),foreground.tobytes(),defect.tobytes()

def _decode(values,size):
    return tuple(Image.frombytes(mode,(size,size),value) for mode,value in zip(('RGB','L','L'),values))

def render(obj,size=224):
    return _decode(_render_bytes(obj,size),size)

def _render_uncached(obj,size=224):
    return _decode(_render_bytes.__wrapped__(obj,size),size)

render.__wrapped__=_render_uncached
render.cache_clear=_render_bytes.cache_clear
render.cache_info=_render_bytes.cache_info

def intervene(obj,factor):
    if factor not in FACTORS: raise ValueError(f'Unsupported factor: {factor}')
    return replace(obj,**{factor:1-getattr(obj,factor)})

@lru_cache(maxsize=65536)
def quality(obj,factor,size):
    before,fg,dm=render(obj,size); after,_,_=render(intervene(obj,factor),size)
    a,b=np.asarray(before),np.asarray(after); mask=np.asarray(fg)>0; d=np.asarray(dm)>0
    preserved=np.array_equal(a[mask],b[mask]) if factor!='lighting' else True
    visible=not obj.label or (d.sum()>=8 and b[mask & ~d].mean()-b[d].mean()>70)
    clipped=float(((b[mask]<=1)|(b[mask]>=254)).mean())
    return dict(valid=bool(preserved and visible and clipped<.01),foreground_preserved=bool(preserved),defect_visible=bool(visible),clipping_fraction=clipped)

def validate(config):
    rows=registry(config)
    if len({o.seed for o in rows})!=len(rows): raise ValueError('Split overlap')
    checks=[quality(o,f,config['image_size']) for o in rows for f in FACTORS]
    if not all(q['valid'] for q in checks): raise ValueError('Invalid intervention')
    return dict(objects=len(rows),quality_checks=len(checks),valid=True)

def generate(config,root):
    root=Path(root); root.mkdir(parents=True,exist_ok=True)
    manifest=[]; hashes=set()
    for obj in registry(config):
        images=render(obj,config['image_size'])
        folder=root/obj.split; folder.mkdir(exist_ok=True)
        stem=obj.base_object_id.replace(':','-')
        paths=[folder/f'{stem}{s}.png' for s in ('','-foreground','-defect')]
        for image,path in zip(images,paths): image.save(path)
        digest=hashlib.sha256(images[0].tobytes()).hexdigest()
        if digest in hashes: raise ValueError('Duplicate image')
        hashes.add(digest)
        manifest.append({**asdict(obj),'image_id':stem,'image_path':str(paths[0]),'foreground_mask_path':str(paths[1]),'defect_mask_path':str(paths[2]),'image_hash':digest,'generator_version':'tiles-v3','defect_type':('scratch','crack','chip')[obj.seed%3] if obj.label else 'none','defect_parameters':{'seed':obj.seed},'render_seed':obj.seed,'background_attribute':obj.background,'background_family':'sinusoidal' if obj.unfamiliar else 'noise','watermark_attribute':obj.watermark,'watermark_design':'ellipse' if obj.unfamiliar else 'square','lighting_attribute':obj.lighting,'lighting_parameters':{'brightness':(1.16 if obj.lighting else .84) if obj.unfamiliar else (1.12 if obj.lighting else .88)},'intervention_parameters':{},'parent_image_id':None,'intervention_factor':None})
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2))
    return manifest



def standard_variant(obj,variant,size):
    image,foreground,defect=render(obj,size)
    if variant=='flip':
        image,foreground,defect=map(ImageOps.mirror,(image,foreground,defect))
    elif variant=='rotate':
        image=image.rotate(5,resample=3)
        foreground=foreground.rotate(5,resample=0)
        defect=defect.rotate(5,resample=0)
    elif variant=='contrast': image=ImageEnhance.Contrast(image).enhance(1.1)
    else: raise ValueError('Unknown standard transformation')
    a=np.asarray(image); fg=np.asarray(foreground)>0; dm=np.asarray(defect)>0
    if obj.label and (dm.sum()<8 or a[fg & ~dm].mean()-a[dm].mean()<70): raise ValueError('Standard augmentation hides defect')
    return image


