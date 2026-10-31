"""Portable source-trained forest + cross-fitted probability pooling.

Uses only supplied inputs/acquired labels and bundled fitted arrays. No network,
downloads, private-input disk cache, or annotation lookup. The Gaussian fallback
extends Team Drift's MIT hierarchical update; standard statistical components
and the upstream contribution are documented in THIRD_PARTY_NOTICES.md.
"""
from collections import OrderedDict
from pathlib import Path
import hashlib,json,threading
import numpy as np
from scipy.special import expit,logit
from models.manual_text_features_v1 import extract
from models.calibration_priors_v73 import fit_shift
from models.statistical_adaptation_v70 import intercept
from models.correlated_profile_adaptation_v67 import fit as fit_current,predict as predict_current


def local_features(text):
    """Preserve normal features; unsupported/extreme syntax gets a valid prior."""
    fallback=np.zeros(52,dtype=np.float32)
    fallback[0]=int('Options:' not in text)
    if len(text)>1_000_000:return fallback
    try:
        values,_=extract(text)
        values=np.asarray(values,dtype=np.float32)
        if values.shape!=(52,) or not np.isfinite(values).all():return fallback
        return values
    except (ValueError,AssertionError,SyntaxError,RecursionError,OverflowError):
        # Python bounds integer conversion / AST depth, and exact interpreters
        # may reject ambiguous scope. A new question must not crash evaluation.
        return fallback


class Forest:
    def __init__(self,path):
        with np.load(path,allow_pickle=False) as f:self.arrays={k:f[k] for k in f.files}
        assert len(self.arrays['roots'])==160 and self.arrays['values'].shape[1]==192

    def predict(self,features):
        x=np.asarray(features,dtype=np.float32).reshape(1,-1);f=self.arrays
        profile=np.zeros(192,dtype=np.float64)
        for root in f['roots']:
            node=int(root)
            while f['left'][node]>=0:
                node=int(f['left'][node] if x[0,f['feature'][node]]<=f['threshold'][node] else f['right'][node])
            profile+=f['values'][f['leaf'][node]]
        return float(np.clip(profile/len(f['roots']),1e-4,1-1e-4).mean())


class BetaFit:
    """Arbitrary counts per subject, including empty rows and repeated trials."""
    def __init__(self,p0,support,labels,kappa,setting):
        self.kappa=kappa;self.setting=setting
        iid=np.concatenate(support);ys=np.concatenate(labels)
        keys=np.unique(iid);self.lookup={int(i):j for j,i in enumerate(keys)}
        self.counts=np.array([np.sum(iid==i) for i in keys],float)
        self.success=np.array([ys[iid==i].sum() for i in keys],float)
        u=len(keys)
        if u:
            frac=self.success/self.counts;z=logit(p0[keys])
            self.global_shift=float(fit_shift(z[None,:],frac[None,:],setting)[0])
            if u>1:
                keep=~np.eye(u,dtype=bool)
                off=np.broadcast_to(z,(u,u))[keep].reshape(u,u-1)
                y=np.broadcast_to(frac,(u,u))[keep].reshape(u,u-1)
                self.item_shift=fit_shift(off,y,setting)
            else:self.item_shift=np.zeros(u)
        else:self.global_shift=0.;self.item_shift=np.empty(0)
        prior=np.clip(expit(logit(p0[keys])+self.item_shift),1e-5,1-1e-5)
        self.solver_shift=[]
        for ids,y in zip(support,labels):
            at=np.array([self.lookup[int(i)] for i in ids],int)
            base=(kappa*prior[at]+self.success[at]-y)/(kappa+self.counts[at]-1)
            z=logit(np.clip(base,1e-5,1-1e-5))
            self.solver_shift.append(float(intercept(z[None,:],np.asarray(y,float)[None,:],16.)[0]))

    def predict(self,base,ids,subject):
        base=np.clip(np.asarray(base,float),1e-5,1-1e-5)
        shift=np.array([self.item_shift[self.lookup[int(i)]] if int(i) in self.lookup else self.global_shift for i in ids])
        k=np.array([self.success[self.lookup[int(i)]] if int(i) in self.lookup else 0. for i in ids])
        n=np.array([self.counts[self.lookup[int(i)]] if int(i) in self.lookup else 0. for i in ids])
        m=np.clip(expit(logit(base)+shift),1e-5,1-1e-5)
        p=(self.kappa*m+k)/(self.kappa+n)
        return np.clip(expit(logit(np.clip(p,1e-5,1-1e-5))+self.solver_shift[subject]),.001,.999)


def current_fit(p0,support,labels):
    size=len(support);ids=np.concatenate(support)
    subjects=np.concatenate([np.repeat(j,len(s)) for j,s in enumerate(support)])
    cov=(.1*np.eye(size)+.9*np.ones((size,size)))[None,:,:]
    return fit_current(logit(p0[ids]),np.concatenate(labels),subjects,ids,cov)


class Adaptation:
    def __init__(self,p0,support,labels,kappa=.25,setting='pseudo4'):
        p0=np.clip(np.asarray(p0,float),1e-5,1-1e-5)
        self.old=current_fit(p0,support,labels)
        self.beta=BetaFit(p0,support,labels,kappa,setting)
        aa=[];bb=[];truth=[]
        # Same3-position-fold policy as v71/v73 for equal counts. A common fold
        # assignment also supports unequal acquired counts without dropping data.
        folds=min(3,max(map(len,support),default=0))
        for fold in range(folds):
            held=[np.arange(len(s))%folds==fold for s in support]
            ss=[s[~h] for s,h in zip(support,held)]
            yy=[y[~h] for y,h in zip(labels,held)]
            qq=[s[h] for s,h in zip(support,held)]
            a=current_fit(p0,ss,yy);b=BetaFit(p0,ss,yy,kappa,setting)
            for j,q in enumerate(qq):
                if not len(q):continue
                assert set(ss[j]).isdisjoint(q) or len(set(support[j]))<len(support[j])
                aa.extend(predict_current(p0[q],q,j,a)[0]);bb.extend(b.predict(p0[q],q,j))
                truth.extend(labels[j][held[j]])
        aa=np.array(aa);bb=np.array(bb);truth=np.array(truth);d=bb-aa
        self.weight=float(np.clip((d@(truth-aa)+1.)/(d@d+1.),0,1))

    def predict(self,base,ids,subject):
        a=predict_current(base,ids,subject,self.old)[0]
        b=self.beta.predict(base,ids,subject)
        return (1-self.weight)*a+self.weight*b


def identity(value):return tuple(sorted((str(k),str(v)) for k,v in value.items()))
def item_identity(item):return tuple(str(item.get(k,'')) for k in ['item_content','item_features','interactors'])


class Predictor:
    def __init__(self,folder,bank='structural52',kappa=.25,setting='pseudo4'):
        assert bank in ['structural52','combined70']
        folder=Path(folder);self.forest=Forest(folder/(bank+'.npz'))
        self.feature_count=52 if bank=='structural52' else 70
        self.kappa=kappa;self.setting=setting
        self.priors=OrderedDict();self.fits=OrderedDict()
        self._lock=threading.RLock()

    def features(self,text):return local_features(text)

    def prior(self,text):
        with self._lock:
            digest=hashlib.sha256(text.encode('utf-8',errors='surrogatepass')).digest()
            if digest not in self.priors:
                features=self.features(text)
                assert len(features)==self.feature_count and np.isfinite(features).all()
                self.priors[digest]=self.forest.predict(features)
                if len(self.priors)>4096:self.priors.popitem(last=False)
            else:self.priors.move_to_end(digest)
            return self.priors[digest]

    def predict(self,input,labeled=None):
        with self._lock:return self._predict(input,labeled)

    def _predict(self,input,labeled=None):
        subject,item=input;subject_key=identity(subject);bid=item.get('benchmark_id','')
        rows=[]
        for entry in labeled or []:
            (s,i),y=entry
            if y not in [0,1]:continue
            if bid:
                if i.get('benchmark_id','')!=bid:continue
            elif i.get('benchmark_id','') or identity(s)!=subject_key:continue
            rows.append((identity(s),item_identity(i),float(y)))
        base=self.prior(str(item.get('item_content','')))
        if not rows:return float(np.clip(base,.001,.999))
        subjects=list(dict.fromkeys(r[0] for r in rows));items=list(dict.fromkeys(r[1] for r in rows))
        if subject_key not in subjects:subjects.append(subject_key)
        cache_key=(bid,tuple(rows),subject_key if not any(r[0]==subject_key for r in rows) else None)
        if cache_key not in self.fits:
            si={s:j for j,s in enumerate(subjects)};ii={i:j for j,i in enumerate(items)}
            support=[[] for _ in subjects];ys=[[] for _ in subjects]
            for s,i,y in rows:support[si[s]].append(ii[i]);ys[si[s]].append(y)
            support=[np.array(s,dtype=int) for s in support];ys=[np.array(y,float) for y in ys]
            p0=np.array([self.prior(i[0]) for i in items])
            self.fits[cache_key]=(Adaptation(p0,support,ys,self.kappa,self.setting),si,ii)
            if len(self.fits)>8:self.fits.popitem(last=False)
        else:self.fits.move_to_end(cache_key)
        fit,si,ii=self.fits[cache_key]
        iid=ii.get(item_identity(item),-1)
        p=float(fit.predict(np.array([base]),np.array([iid]),si[subject_key])[0])
        assert np.isfinite(p) and 0<=p<=1
        return p
