import json
import hashlib
import sys
from dataclasses import replace
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
import streamlit as st
import torch
import plotly.express as px
from failure_lab.data import registry, render, FACTORS, quality
from failure_lab.engine import factory, materialize, predict

torch.set_num_threads(4)
st.set_page_config(page_title='Shortcut Detective',layout='wide')
st.title('Shortcut Detective')
st.write('Surface defects determine the label. Background, watermark and lighting may mislead models trained on correlated cues.')
generator_hash=hashlib.sha256((Path(__file__).resolve().parents[1]/'src/failure_lab/data.py').read_bytes()).hexdigest()
files=[]
for candidate in sorted(Path('artifacts').glob('*/results.json')):
    candidate_results=json.loads(candidate.read_text())
    if candidate_results and candidate_results[0].get('metadata',{}).get('generator_source_sha256')==generator_hash:
        files.append(candidate)
if not files:
    st.info('Run Start-Lab.ps1 -Mode smoke to create results.'); st.stop()
path=st.sidebar.selectbox('Experiment',files,format_func=lambda p:p.parent.name)
results=json.loads(path.read_text()); config=json.loads((path.parent/'protocol.json').read_text())
st.caption(f"{config['profile'].upper()} | {len(results)} completed runs | {config['experiment_id']}")
if config['profile']!='final': st.warning('Development study. These results do not complete the planned final research matrix.')
overview,investigate,compare,report=st.tabs(['Overview','Investigation','Repair comparison','Report'])
with overview:
    st.write('Saved artifact metrics; interactive examples come only from the development audit split.')
    st.json(config)
@st.cache_resource
def load_checkpoint(filename):
    ck=torch.load(filename,map_location='cpu',weights_only=False)
    model,_=factory(ck['architecture'],False); model.load_state_dict(ck['state_dict']); model.eval()
    return model,ck
with investigate:
    architecture=st.selectbox('Architecture',sorted({r['architecture'] for r in results}))
    choices=[r for r in results if r['architecture']==architecture]
    original=st.selectbox('Original',[r for r in choices if r['method']=='original'],format_func=lambda r:r['run_id'])
    repair_choices=[r for r in choices if r['method']!='original']
    repair=st.selectbox('Repair',repair_choices,format_func=lambda r:r['run_id']) if repair_choices else None
    if repair is None: st.info('Repair conditions have not completed yet.')
    objects=[o for o in registry(config) if o.split=='audit_dev']
    obj=objects[st.slider('Development object',0,len(objects)-1,0)]
    factor=st.radio('Change cue',FACTORS,horizontal=True)
    edited=replace(obj,**{factor:1-getattr(obj,factor)})
    columns=st.columns(2); predictions={}
    for col,value,label in zip(columns,(obj,edited),('Original image','Edited image')):
        image,fg,dm=render(value,config['image_size'])
        col.image(image,caption=f'{label}: true label {value.label}',width=320)
        for r in (r for r in (original,repair) if r is not None):
            model,ck=load_checkpoint(str(path.parent/Path(r['checkpoint']).name))
            logit=predict(model,materialize([value],config['image_size'],ck['preprocessing'])[0])[0]
            p=float(1/(1+np.exp(-logit/ck['temperature'])))
            raw=float(1/(1+np.exp(-logit)))
            predictions[(r['method'],label)]=raw
            col.metric(r['method']+' raw defect probability',f'{raw:.6f}')
            col.caption(f'Calibrated probability: {p:.6f}')
        if col.checkbox('Show masks',key=label): col.image([fg,dm],caption=['Foreground','Defect'],width=150)
    for r in (r for r in (original,repair) if r is not None):
        delta=predictions[(r['method'],'Edited image')]-predictions[(r['method'],'Original image')]
        st.write(f"{r['method']} raw probability change: {delta:+.6f}")
    checks=quality(obj,factor,config['image_size']).copy()
    checks.pop('foreground_preserved',None)
    checks['foreground_pixels_unchanged']=factor!='lighting'
    checks['geometry_and_masks_unchanged']=True
    st.json(checks)
with compare:
    records=[]
    for r in results:
        e=r['evaluation']
        records.append(dict(architecture=r['architecture'],seed=r['seed'],method=r['method'],budget=r['budget'],worst_group_accuracy=e['decorrelated_test']['raw']['worst_group_accuracy'],matched_accuracy=e['matched_test']['raw']['accuracy'],inverted_accuracy=e['inverted_test']['raw']['accuracy'],unfamiliar_accuracy=e['unfamiliar_test']['raw']['accuracy']))
    df=pd.DataFrame(records); st.dataframe(df,hide_index=True)
    st.plotly_chart(px.line(df[df.budget>0],x='budget',y='worst_group_accuracy',color='method',facet_col='architecture',markers=True),width='stretch')
    st.dataframe(df.groupby(['architecture','method','budget']).worst_group_accuracy.agg(['mean','std','count']))
    selected=st.selectbox('Group counts and calibration',results,format_func=lambda r:r['run_id'])
    st.json(selected['evaluation']['decorrelated_test'])
with report:
    content=(path.parent/'report.md').read_text()
    st.markdown(content); st.download_button('Download report',content,'report.md')
    st.download_button('Download results',path.read_text(),'results.json')




