import numpy as np
import torch

def metrics(labels,probs,groups):
    y=np.asarray(labels); p=np.clip(np.asarray(probs),1e-7,1-1e-7); correct=(p>=.5)==y
    scores={}
    for group in sorted(set(groups)):
        mask=np.array([g==group for g in groups])
        scores[group]={'count':int(mask.sum()),'accuracy':float(correct[mask].mean())}
    recalls=[float(correct[y==k].mean()) if (y==k).any() else None for k in (0,1)]
    confidence=np.maximum(p,1-p); ece=0.
    bins=[]
    for i in range(10):
        mask=(confidence>=i/10)&(confidence<(i+1)/10 if i<9 else confidence<=1)
        count=int(mask.sum())
        gap=abs(float(confidence[mask].mean()-correct[mask].mean())) if count else 0.
        ece+=count/len(y)*gap; bins.append({'count':count,'gap':gap})
    return dict(accuracy=float(correct.mean()),balanced_accuracy=float(np.mean([r for r in recalls if r is not None])),defect_recall=recalls[1],worst_group_accuracy=min(v['accuracy'] for v in scores.values()),groups=scores,nll=float(-(y*np.log(p)+(1-y)*np.log(1-p)).mean()),brier=float(((p-y)**2).mean()),ece=ece,calibration_bins=bins)

def temperature(logits,labels):
    x=torch.tensor(logits,dtype=torch.float32); y=torch.tensor(labels,dtype=torch.float32)
    log_t=torch.zeros((),requires_grad=True)
    optimizer=torch.optim.LBFGS([log_t],max_iter=30)
    def closure():
        optimizer.zero_grad()
        loss=torch.nn.functional.binary_cross_entropy_with_logits(x/log_t.exp().clamp(.05,20),y)
        loss.backward(); return loss
    optimizer.step(closure)
    return float(log_t.exp().clamp(.05,20).detach())

def bootstrap_wga(labels,probs,groups,repetitions,seed,other=None):
    rng=np.random.default_rng(seed)
    group_names=sorted(set(groups)); codes=np.array([group_names.index(g) for g in groups])
    correct=(np.asarray(probs)>=.5)==np.asarray(labels)
    different=None if other is None else ((np.asarray(other)>=.5)==np.asarray(labels))
    values=[]
    for _ in range(repetitions):
        idx=rng.integers(0,len(labels),len(labels))
        count=np.bincount(codes[idx],minlength=len(group_names))
        if np.any(count==0): continue
        first=(np.bincount(codes[idx],weights=correct[idx],minlength=len(group_names))/count).min()
        if different is None: values.append(first)
        else:
            second=(np.bincount(codes[idx],weights=different[idx],minlength=len(group_names))/count).min()
            values.append(first-second)
    if not values: return [None,None]
    return [float(x) for x in np.quantile(values,[.025,.975])]

def bootstrap(labels,probs,groups,repetitions,seed):
    return bootstrap_wga(labels,probs,groups,repetitions,seed)
def paired_bootstrap(labels,a,b,groups,repetitions,seed):
    return bootstrap_wga(labels,a,groups,repetitions,seed,b)

