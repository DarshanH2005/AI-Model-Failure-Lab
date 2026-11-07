import argparse
import json
from pathlib import Path
import torch
import yaml
from .data import generate, validate
from .engine import run, report

def load_config(path):
    cfg=yaml.safe_load(Path(path).read_text(encoding='utf-8-sig'))
    required=('experiment_id','profile','image_size','architectures','seeds','budgets','splits','train_steps','repair_steps','batch_size','learning_rate','weight_decay','head_steps','bootstrap_samples','pretrained')
    missing=set(required)-set(cfg)
    if missing: raise ValueError(f'Missing configuration: {sorted(missing)}')
    if cfg['image_size']<64 or cfg['batch_size']<2: raise ValueError('Image size >=64 and batch size >=2 required')
    if any(v<16 or v%2 for v in cfg['splits'].values()): raise ValueError('Splits must have at least 16 objects and balanced labels')
    if any(b<=0 or b%4 or b//4>cfg['splits']['candidates'] for b in cfg['budgets']): raise ValueError('Invalid repair budget')
    if cfg['profile']=='final' and cfg.get('registry_seed',0)==0: raise ValueError('Final study requires fresh registry_seed after pilot evaluation')
    if cfg['profile']=='final' and ('tiny_cnn' in cfg['architectures'] or not cfg['pretrained']): raise ValueError('Final study requires pretrained models')
    return cfg

def main():
    parser=argparse.ArgumentParser(description='Shortcut Detective reproducible laboratory')
    parser.add_argument('command',choices=['generate','validate-data','run-experiments','report'])
    parser.add_argument('--config',default='configs/smoke.yaml')
    parser.add_argument('--root',default=None)
    parser.add_argument('--threads',type=int,default=4)
    args=parser.parse_args(); torch.set_num_threads(args.threads)
    cfg=load_config(args.config)
    root=Path(args.root or f"artifacts/{cfg['experiment_id']}")
    if args.command=='generate': generate(cfg,Path('data')/cfg['experiment_id'])
    elif args.command=='validate-data': print(json.dumps(validate(cfg),indent=2))
    elif args.command=='run-experiments':
        print(json.dumps(validate(cfg))); run(cfg,root)
    else: report(root)

if __name__=='__main__': main()

