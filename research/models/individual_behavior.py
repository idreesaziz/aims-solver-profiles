"""One subject's labels in, one subject's correctness probabilities out.

No learned cross-subject parameters, initialization or outcome features.
"""
import warnings
import time

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit,logit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from models.capability_graphs import balanced_weights


def fit_head(x,y,weights,kind,c,regularization=None):
    x=np.asarray(x);y=np.asarray(y)
    if y.ndim!=1 or len(y)!=len(x) or not np.isin(y,[0,1]).all():raise ValueError('Exactly one subject label vector is required')
    weights=np.asarray(weights,dtype=float);weights=weights/weights.sum()
    rate=float(np.clip((weights@y*len(y)+1)/(len(y)+2),1e-6,1-1e-6))
    active=np.ptp(x,axis=0)>0
    if kind=='mean' or not active.any() or len(np.unique(y))<2:
        return dict(kind='constant',rate=rate,active=active,estimator=None)
    a=np.ascontiguousarray(x[:,active],dtype=float);sample_weight=weights*len(y)
    if kind=='logistic':
        estimator=LogisticRegression(C=1/(regularization*sample_weight.sum()),solver='lbfgs',
            max_iter=c['logistic_max_iterations'],tol=c['logistic_tolerance'],random_state=c['seed'])
        with warnings.catch_warnings():
            warnings.simplefilter('error',ConvergenceWarning)
            estimator.fit(a,y,sample_weight=sample_weight)
    elif kind=='tree':
        estimator=HistGradientBoostingClassifier(max_iter=c['tree_iterations'],learning_rate=c['tree_learning_rate'],
            max_leaf_nodes=c['tree_max_leaves'],min_samples_leaf=c['tree_min_leaf'],l2_regularization=c['tree_l2'],
            early_stopping=False,random_state=c['seed'])
        estimator.fit(a,y,sample_weight=sample_weight)
    else:raise ValueError('Unknown learner')
    return dict(kind=kind,active=active,estimator=estimator,rate=rate,regularization=regularization)


def predict_head(head,x):
    if head['estimator'] is None:return np.full(len(x),head['rate'])
    p=head['estimator'].predict_proba(np.ascontiguousarray(x[:,head['active']],dtype=float))[:,1]
    if not np.isfinite(p).all():raise ValueError('Nonfinite predictions')
    return p


def fit_calibration(p,y,weights,penalty):
    """Fixed regularized Platt map fitted on a separate calibration group set."""
    z=logit(np.clip(p,1e-6,1-1e-6));y=np.asarray(y);w=np.asarray(weights);w=w/w.sum()
    def objective(v):
        a,b=v;eta=a*z+b;r=(expit(eta)-y)*w
        return (float(w@(np.logaddexp(0,eta)-y*eta)+.5*penalty*((a-1)**2+b*b)),
                np.array([r@z+penalty*(a-1),r.sum()+penalty*b]))
    fit=minimize(objective,[1.,0.],jac=True,method='L-BFGS-B',bounds=[(.1,5),(-5,5)],options=dict(gtol=1e-9,maxiter=100))
    if not fit.success:
        # L-BFGS can report a line-search failure at machine precision. Retry
        # the identical strictly convex bounded objective once, never silently
        # substitute an identity map or a changed statistical model.
        fit=minimize(objective,fit.x,jac=True,method='SLSQP',bounds=[(.1,5),(-5,5)],options=dict(ftol=1e-12,maxiter=100))
    if not fit.success:raise RuntimeError('Calibration did not converge: '+str(fit.message))
    return fit.x


def calibrate(p,parameters):
    return expit(parameters[0]*logit(np.clip(p,1e-6,1-1e-6))+parameters[1])


def score(p,y,weights):
    p=np.asarray(p);y=np.asarray(y);w=np.asarray(weights);w=w/w.sum()
    brier=float(w@((p-y)**2));clipped=np.clip(p,1e-7,1-1e-7)
    nll=float(-w@(y*np.log(clipped)+(1-y)*np.log1p(-clipped)))
    auc=float(roc_auc_score(y,p,sample_weight=w)) if len(np.unique(y))==2 else None
    ece=0.
    for j in range(10):
        mask=(p>=j/10)&((p<(j+1)/10) if j<9 else (p<=1))
        if mask.any():ece+=abs(float(w[mask]@(p[mask]-y[mask])))
    return dict(brier=brier,log_loss=nll,auc=auc,ece_10_equal_width=float(ece))


def fit_subject(x,y,split,families,groups,objective_width,c,tree_enabled):
    """Fit/tune/calibrate/refit exclusively on one historical subject's data."""
    if np.asarray(y).ndim!=1:raise ValueError('Other subjects cannot enter this fit')
    tr=np.asarray(split['fit']);tu=np.asarray(split['tune']);ca=np.asarray(split['calibrate']);rf=np.asarray(split['refit']);te=np.asarray(split['test'])
    for left,right in [(tr,tu),(tr,ca),(tr,te),(tu,ca),(tu,te),(ca,te),(rf,te)]:
        if set(groups[left])&set(groups[right]):raise ValueError('Underlying question leakage')
    weights={key:balanced_weights(families[ix],groups[ix]) for key,ix in [('fit',tr),('tune',tu),('calibrate',ca),('refit',rf)]}
    predictions={};heads={};logs={};calibration={};tuning={}
    definitions=[('own_mean','mean',x.shape[1]),('objective_logistic','logistic',objective_width),('concept_logistic','logistic',x.shape[1])]
    if tree_enabled:definitions.append(('concept_tree','tree',x.shape[1]))
    for name,kind,width in definitions:
        started=time.perf_counter()
        features=x[:,:width]
        grid=c['regularization_grid'] if kind=='logistic' else [None]
        candidates=[]
        for reg in grid:
            head=fit_head(features[tr],y[tr],weights['fit'],kind,c,reg)
            p=predict_head(head,features[tu]);value=float(weights['tune']@((p-y[tu])**2))
            candidates.append((value,reg,head))
        best=min(candidates,key=lambda v:v[0]);value,reg,inner=best;tuning[name]=value
        cp=predict_head(inner,features[ca]);parameters=fit_calibration(cp,y[ca],weights['calibrate'],c['calibration_penalty'])
        final=fit_head(features[rf],y[rf],weights['refit'],kind,c,reg)
        p=predict_head(final,features[te]);predictions[name]=p
        if kind!='mean':predictions[name+'_calibrated']=calibrate(p,parameters)
        heads[name]=dict(head=final,width=width,calibration=parameters);calibration[name]=parameters.tolist()
        logs[name]=dict(regularization=reg,tune_brier=value,active_features=int(final['active'].sum()),
                        fit_iterations=int(np.max(final['estimator'].n_iter_)) if kind=='logistic' and final['estimator'] is not None else None,
                        seconds=time.perf_counter()-started)
    chosen=min(tuning,key=tuning.get)
    if chosen=='own_mean':predictions['inner_selected_calibrated']=calibrate(predictions[chosen],calibration[chosen])
    else:predictions['inner_selected_calibrated']=predictions[chosen+'_calibrated']
    predictions['constant_0.5']=np.full(len(te),.5)
    if split['protocol']=='within_benchmark':
        fallback=heads['own_mean']['head']['rate'];p=np.full(len(te),fallback)
        for f in np.unique(families[te]):
            ix=rf[families[rf]==f]
            if len(ix):
                w=balanced_weights(families[ix],groups[ix]);rate=(w@y[ix]*len(ix)+1)/(len(ix)+2);p[families[te]==f]=rate
        predictions['own_benchmark_mean_diagnostic']=p
    assert all(np.isfinite(p).all() and np.all((p>=0)&(p<=1)) for p in predictions.values())
    return predictions,dict(chosen=chosen,models=logs,calibration=calibration),heads
