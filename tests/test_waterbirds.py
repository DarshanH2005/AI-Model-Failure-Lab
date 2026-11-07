import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
module_path=Path(__file__).resolve().parents[1]/'scripts/run_waterbirds.py'
spec=importlib.util.spec_from_file_location('waterbirds_runner',module_path)
wb=importlib.util.module_from_spec(spec)
spec.loader.exec_module(wb)

def test_validation_partition_is_deterministic_and_disjoint():
    frame=pd.DataFrame({'y':[0]*10+[1]*10,'place':[0,1]*10},index=range(100,120))
    a,b=wb.validation_partition(frame)
    assert set(a).isdisjoint(b) and set(a+b)==set(frame.index)
    assert (a,b)==wb.validation_partition(frame)
    assert all(isinstance(i,int) for i in a+b)
    assert len(set(wb.groups(frame.loc[a])))==4
    assert len(set(wb.groups(frame.loc[b])))==4

def test_adjusted_accuracy_uses_training_weights():
    frame=pd.DataFrame({'y':[0,0,1,1]*8,'place':[0,1,0,1]*8})
    logits=np.array([-5,5,5,5]*8)
    weights={'00':.7,'01':.1,'10':.1,'11':.1}
    result,raw,calibrated=wb.evaluate(frame,logits,1,weights,11)
    assert result['raw']['accuracy']==.75
    assert abs(result['raw']['adjusted_accuracy']-.9)<1e-12
    assert result['raw']['worst_group_accuracy']==0
    assert result['raw']['waterbird_recall']==1
    assert 'defect_recall' not in result['raw']
    assert np.array_equal(raw,calibrated)

def test_photographic_preprocessing_is_rgb_and_model_normalized(tmp_path):
    Image.new('L',(19,37),128).save(tmp_path/'image.png')
    frame=pd.DataFrame({'img_filename':['image.png'],'y':[1]})
    prep={'mean':[.5,.4,.3],'std':[.1,.2,.3]}
    x,y=wb.arrays(frame,tmp_path,prep)
    assert tuple(x.shape)==(1,3,224,224) and y.tolist()==[1]
    expected=(np.array([128/255]*3)-np.array(prep['mean']))/np.array(prep['std'])
    assert np.allclose(x[0,:,0,0].numpy(),expected,atol=1e-6)

