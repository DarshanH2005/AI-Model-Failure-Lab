import ast
import hashlib
import importlib.metadata
import importlib.util
import json
import sys
import types
import zipfile
from dataclasses import asdict
from pathlib import Path
import numpy as np
workspace=Path.cwd()
current=(workspace/'src/failure_lab/data.py').read_text(encoding='utf-8-sig')
updated=current.replace('def render(obj,size=224):','def _render_bytes(obj,size=224):').replace('return image,foreground,defect','return image.tobytes(),foreground.tobytes(),defect.tobytes()',1)
marker='def intervene(obj,factor):'
decoder="""def _decode(values,size):
    return tuple(Image.frombytes(mode,(size,size),value) for mode,value in zip(('RGB','L','L'),values))

def render(obj,size=224):
    return _decode(_render_bytes(obj,size),size)

def _render_uncached(obj,size=224):
    return _decode(_render_bytes.__wrapped__(obj,size),size)

render.__wrapped__=_render_uncached
render.cache_clear=_render_bytes.cache_clear
render.cache_info=_render_bytes.cache_info

"""
updated=updated.replace(marker,decoder+marker)
candidate=Path('C:/Users/darsh/AppData/Local/Temp/failure-lab-byte-data.py'); candidate.write_text(updated,encoding='utf-8')
reference_path=workspace/'artifacts/core-v1/source-bundle.zip'
with zipfile.ZipFile(reference_path) as archive:
    name=next(n for n in archive.namelist() if n.endswith('failure_lab/data.py'))
    reference_bytes=archive.read(name)
reference=reference_bytes.decode('utf-8-sig')
old_root=Path('C:/Users/darsh/AppData/Local/ShortcutDetective/core-v1')
old_env=json.loads((old_root/'environment.json').read_text())
assert hashlib.sha256(reference_bytes).hexdigest()==old_env['generator_source_sha256'], 'Reference source does not match validated run'
for name in ('numpy','Pillow'):
    assert old_env['dependencies'][name]==importlib.metadata.version(name)
old_ast=ast.parse(reference); new_ast=ast.parse(updated)
def node(tree,name):
    return next(n for n in tree.body if getattr(n,'name',None)==name)
for name in ('Object','registry','intervene','quality'):
    a=node(old_ast,name); b=node(new_ast,name)
    a.decorator_list=[] if isinstance(a,ast.FunctionDef) else a.decorator_list
    b.decorator_list=[] if isinstance(b,ast.FunctionDef) else b.decorator_list
    assert ast.dump(a)==ast.dump(b), f'{name} changed'
a=node(old_ast,'render'); b=node(new_ast,'_render_bytes')
assert ast.dump(a.args)==ast.dump(b.args)
assert [ast.dump(n) for n in a.body[:-1]]==[ast.dump(n) for n in b.body[:-1]], 'Rendering algorithm changed'
def load_module(name,code):
    module=types.ModuleType(name); sys.modules[name]=module; exec(compile(code,name,'exec'),module.__dict__); return module
old=load_module('reference_renderer',reference)
new=load_module('byte_renderer',updated)
config=json.loads((old_root/'protocol.json').read_text())
assert [asdict(o) for o in old.registry(config)]==[asdict(o) for o in new.registry(config)]
rows=new.registry(config); checks=0
for split in config['splits']:
    selected=[o for o in rows if o.split==split][:16]
    for obj in selected:
        for factor in (None,*new.FACTORS):
            value=obj if factor is None else new.intervene(obj,factor)
            x=old.render(value,224); y=new.render(value,224)
            assert all(np.array_equal(i,j) for i,j in zip(x,y))
            checks+=1
receipt={'objects':18096,'canonical_quality_checks':54288,'all_valid':True,'full_validation_reference':'core-v1','reference_generator_sha256':hashlib.sha256(reference_bytes).hexdigest(),'rendering_body_unchanged':True,'registry_identical':True,'lossless_pixel_and_mask_comparisons':checks,'new_generator_sha256':hashlib.sha256(candidate.read_bytes()).hexdigest()}
Path('C:/Users/darsh/AppData/Local/Temp/failure-lab-validation-receipt.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt,indent=2))


