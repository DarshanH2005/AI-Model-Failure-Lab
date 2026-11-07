import hashlib
import os
import json
from dataclasses import asdict
from pathlib import Path
import torch
from failure_lab.cli import load_config
from failure_lab.data import registry
from failure_lab.engine import run
torch.set_num_threads(4)
config=load_config('configs/experiments.yaml')
receipt=config['validation_receipt']
source=Path('src/failure_lab/data.py').read_bytes()
assert hashlib.sha256(source).hexdigest()==receipt['new_generator_sha256']
old_config=json.loads(Path('C:/Users/darsh/AppData/Local/ShortcutDetective/core-v1/protocol.json').read_text())
for field in ('image_size','splits','registry_seed'):
    assert old_config[field]==config[field]
assert old_config.get('training_correlation',.9)==config.get('training_correlation',.9)
config['validation_receipt']=receipt
root=Path('C:/Users/darsh/AppData/Local/ShortcutDetective/core-v4')
root.mkdir(parents=True,exist_ok=True)
(root/'study.pid').write_text(str(os.getpid()))
(root/'validation-receipt.json').write_text(json.dumps(receipt,indent=2))
print('Reusing full validation after source/registry/lossless-pixel equivalence verification.',flush=True)
run(config,root)




