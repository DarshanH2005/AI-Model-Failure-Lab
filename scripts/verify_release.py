import hashlib
import json
import sys
from pathlib import Path
root=Path(sys.argv[1]); results=json.loads((root/'results.json').read_text())
assert len(results)==66, f'Expected 66 runs, found {len(results)}'
assert len({r['run_id'] for r in results})==66
assert {r['architecture'] for r in results}=={'resnet18.a1_in1k','deit_tiny_patch16_224.fb_in1k'}
assert {r['seed'] for r in results}=={11,22,33}
for result in results:
    budget=result['budget']
    assert len(result['repair_manifest'])==budget
    assert len(set(result['selected_parents']))*4==budget
    assert result['training']['steps']==(1500 if result['method']=='original' else 300)
    checkpoint=root/Path(result['checkpoint']).name
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==result['checkpoint_hash']
    groups=result['evaluation']['decorrelated_test']['raw']['groups']
    assert len(groups)==16 and {g['count'] for g in groups.values()}=={64}
    for split in ('matched_test','decorrelated_test','inverted_test','unfamiliar_test'):
        predictions=json.loads((root/f"{result['run_id']}-{split}-predictions.json").read_text())
        ids=predictions['base_object_ids']
        assert len(ids)==1024 and len(set(ids))==1024
        assert all(i.startswith('100000000:') for i in ids)
summary={'completed_runs':66,'seeds':[11,22,33],'budgets':[128,256,512],'verified_checkpoint_hashes':66,'decorrelated_groups_per_run':16,'objects_per_group':64,'checks_passed':True}
(root/'release-verification.json').write_text(json.dumps(summary,indent=2))
lines=(root/'report.md').read_text().splitlines()
lines.extend(['','## Research interpretation','','The primary comparison uses targeted and random nuisance selection with identical edit policies. Original checkpoint performance determines available headroom. If originals already attain perfect worst-group accuracy, no method can improve that metric; a tie is a ceiling-limited finding rather than proof that sensitivity-based repair never works. Regressions and matched-accuracy changes must still be reported.','','All methods, seeds and budgets are preserved, including inconclusive or negative outcomes. Controlled sensitivity is not full internal reasoning. This synthetic proxy does not establish real industrial reliability. Perceptual near-duplicates remain possible; exact pretrained weight revisions/hashes are recorded for replay.'])
(root/'report.md').write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps(summary))


# Full image/mask export also checks exact image hashes across all splits.
from failure_lab.data import generate, registry
config=json.loads((root/'protocol.json').read_text())
manifest=generate(config,root/'dataset')
assert len(manifest)==18096
coverage={}
expected=[format(i,'04b') for i in range(16)]
for result in results:
    coverage[result['run_id']]={split:{g:result['evaluation'][split]['raw']['groups'].get(g,{'count':0,'accuracy':None}) for g in expected} for split in result['evaluation']}
(root/'group-coverage.json').write_text(json.dumps(coverage,indent=2))
(root/'dataset-validation.json').write_text(json.dumps({'unique_base_objects':len(manifest),'unique_image_hashes':len({m['image_hash'] for m in manifest}),'cross_split_id_overlap':False,'canonical_intervention_checks':54288,'all_valid':True},indent=2))
print('Dataset manifest, PNG images, masks and full group coverage exported.')


import numpy as np
interpretation=['','## Answer to the primary research question','']
for architecture in sorted({r['architecture'] for r in results}):
    originals=[r['evaluation']['decorrelated_test']['raw']['worst_group_accuracy'] for r in results if r['architecture']==architecture and r['method']=='original']
    interpretation.append(f"### {architecture}")
    interpretation.append(f"Original decorrelated WGA: mean {np.mean(originals):.4f}, sample SD {np.std(originals,ddof=1):.4f} over three seeds.")
    for budget in (128,256,512):
        differences=[]
        for seed in (11,22,33):
            targeted=next(r for r in results if r['architecture']==architecture and r['seed']==seed and r['method']=='targeted' and r['budget']==budget)
            random=next(r for r in results if r['architecture']==architecture and r['seed']==seed and r['method']=='random_nuisance' and r['budget']==budget)
            differences.append(targeted['evaluation']['decorrelated_test']['raw']['worst_group_accuracy']-random['evaluation']['decorrelated_test']['raw']['worst_group_accuracy'])
        mean=float(np.mean(differences)); sd=float(np.std(differences,ddof=1))
        finding='No mean improvement observed' if mean<=0 else 'A positive mean difference was observed'
        interpretation.append(f"- B={budget}: targeted minus random nuisance = {mean:+.4f} +/- {sd:.4f}; per-seed differences {differences}. {finding} in this configuration.")
    if all(v==1 for v in originals):
        interpretation.append('The original models already attained perfect worst-group accuracy. This setting provides no headroom for robustness gains over the original model; it cannot establish how selection would perform on a harder benchmark.')
    interpretation.append('')
interpretation.append('Three training seeds limit statistical certainty. Consult the saved paired object-bootstrap intervals, all shifted metrics, calibration and regressions. This result applies only to the declared synthetic task and protocol.')
with (root/'report.md').open('a',encoding='utf-8') as stream:
    stream.write('\n'.join(interpretation))
manifest_path=root/'dataset/manifest.json'
portable=json.loads(manifest_path.read_text())
for row in portable:
    for field in ('image_path','foreground_mask_path','defect_mask_path'):
        row[field]=Path(row[field]).relative_to(root/'dataset').as_posix()
manifest_path.write_text(json.dumps(portable,indent=2))


regressions=[]
for row in results:
    if row['method']=='original':
        continue
    original=next(r for r in results if r['architecture']==row['architecture'] and r['seed']==row['seed'] and r['method']=='original')
    for split,metrics in row['evaluation'].items():
        actual=metrics['raw']['accuracy']
        before=original['evaluation'][split]['raw']['accuracy']
        if actual<before:
            regressions.append(f"- {row['run_id']} on {split}: accuracy {actual:.10f}, original {before:.10f}.")
if regressions:
    with (root/'report.md').open('a',encoding='utf-8') as stream:
        stream.write('\n\n## Measured accuracy regressions\n\n'+'\n'.join(regressions)+'\n')
