"""Read-only external photographic benchmark results."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
st.set_page_config(page_title='Waterbirds external validation',layout='wide')
st.title('Waterbirds: external validation')
st.write('Bird photographs composited onto land or water backgrounds. This is a separate bird-classification task; it does not establish industrial defect reliability or transfer of the intervention-repair method.')
root=Path(__file__).resolve().parents[2]/'artifacts/waterbirds-v1'
if not (root/'results.json').exists():
    st.info('The external study is running. Verified results will appear after release packaging.'); st.stop()
results=json.loads((root/'results.json').read_text())
st.caption(f'{len(results)} completed conditions | two architectures | three training seeds | official held-out test split')
records=[]
for row in results:
    a=row['evaluation']['raw']; b=row['evaluation']['calibrated']
    records.append({'architecture':row['architecture'],'seed':row['seed'],'accuracy':a['accuracy'],'adjusted_accuracy':a['adjusted_accuracy'],'worst_group_accuracy':a['worst_group_accuracy'],'raw_ece':a['ece'],'calibrated_ece':b['ece']})
df=pd.DataFrame(records)
st.dataframe(df,hide_index=True)
st.write('Adjusted accuracy weights each test-group accuracy by its training-group proportion. Unweighted accuracy uses the actual test distribution.')
st.dataframe(df.groupby('architecture')[['accuracy','adjusted_accuracy','worst_group_accuracy']].agg(['mean','std','count']))
selected=st.selectbox('Group counts, calibration and uncertainty',results,format_func=lambda r:r['run_id'])
st.json(selected['evaluation'])
st.caption('Object-bootstrap intervals capture test sampling; sample SD across seeds describes observed training variability. No test-driven tuning or foreground edits were performed.')
st.markdown((root/'report.md').read_text(encoding='utf-8'))
st.download_button('Download external report',(root/'report.md').read_text(encoding='utf-8'),'waterbirds-report.md')
st.download_button('Download results',json.dumps(results,indent=2),'waterbirds-results.json')

if (root/'media/group-robustness.png').exists():
    st.image(str(root/'media/group-robustness.png'),caption='Mean and sample SD across three training seeds.')
if (root/'media/observed-errors.png').exists():
    st.image(str(root/'media/observed-errors.png'),caption='Illustrative held-out errors from the first saved model. Test inspection takes place after the fixed study; these images did not enter tuning. Local research and educational use.')
