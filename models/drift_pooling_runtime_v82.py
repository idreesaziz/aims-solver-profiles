"""Reproduced Drift prior + fixed cross-fitted probability pooling, API-free."""
from collections import OrderedDict
from pathlib import Path
import threading
import numpy as np
import drift_prior_model as prior_model
from models.subject_adaptation_v79 import Adaptation

def identity(value):return tuple(sorted((str(k),str(v)) for k,v in value.items()))
def item_identity(item):return tuple(str(item.get(k,'') or '') for k in ['item_content','item_features','interactors'])

class Predictor:
    def __init__(self,folder):
        meta,arrays=prior_model._load_art(str(Path(folder)))
        self.prior=prior_model.Predictor(meta,arrays)
        self.fits=OrderedDict();self.lock=threading.RLock()

    def probability(self,subject,item):
        # Exact published prior, with numeric admission floor matching research.
        z=self.prior.b+self.prior.zs(subject)+self.prior.zi(item)
        p=1/(1+np.exp(-np.clip(z,-30,30)))
        return float(np.clip(p,1e-5,1-1e-5))

    def predict(self,target,labeled=None):
        with self.lock:return self._predict(target,labeled)

    def _predict(self,target,labeled):
        subject,item=target;sk=identity(subject);ik=item_identity(item);bid=str(item.get('benchmark_id','') or '')
        rows=[]
        for (s,i),y in labeled or []:
            if y not in [0,1]:continue
            other=str(i.get('benchmark_id','') or '')
            if bid:
                if other!=bid:continue
            elif other or identity(s)!=sk:continue
            rows.append((identity(s),item_identity(i),float(y)))
        base=self.probability(subject,item)
        if not rows:return float(np.clip(base,.001,.999))
        subjects=list(dict.fromkeys(r[0] for r in rows));items=list(dict.fromkeys(r[1] for r in rows))
        unseen=sk not in subjects
        if unseen:subjects.append(sk)
        key=(bid,tuple(rows),sk if unseen else None)
        if key not in self.fits:
            si={s:j for j,s in enumerate(subjects)};ii={i:j for j,i in enumerate(items)}
            support=[[] for _ in subjects];ys=[[] for _ in subjects]
            for s,i,y in rows:support[si[s]].append(ii[i]);ys[si[s]].append(y)
            support=[np.array(s,int) for s in support];ys=[np.array(y,float) for y in ys]
            inputs=[dict(zip(['item_content','item_features','interactors'],i)) for i in items]
            p0=np.array([[self.probability(dict(s),i) for i in inputs] for s in subjects])
            fitted=Adaptation(p0,support,ys,.25,'pseudo4')
            self.fits[key]=(fitted,si,ii)
            if len(self.fits)>8:self.fits.popitem(last=False)
        else:self.fits.move_to_end(key)
        fitted,si,ii=self.fits[key]
        value=float(fitted.predict(np.array([base]),np.array([ii.get(ik,-1)]),si[sk])[0])
        assert np.isfinite(value) and 0<=value<=1
        return value
