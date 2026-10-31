"""Descriptive common-panel capability analysis; no API calls or neural models."""
import hashlib

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit, logsumexp


def group_half(group, seed):
    return hashlib.sha256((str(seed)+'|'+str(group)).encode()).digest()[0] % 2


def balanced_weights(families, groups):
    """Equal families, equal underlying groups per family; keep rows together."""
    families=np.asarray(families);groups=np.asarray(groups)
    weights=np.zeros(len(families),dtype=float);unique=np.unique(families)
    if not len(unique):raise ValueError('Empty data')
    for family in unique:
        ix=np.flatnonzero(families==family)
        _,inv,counts=np.unique(groups[ix],return_inverse=True,return_counts=True)
        weights[ix]=1/(len(unique)*len(counts)*counts[inv])
    return weights


def fit_profiles(x,y,families,groups,c,progress=None):
    """Fixed Gaussian MAP logistic head, joint shared and subject deviations.

    The offset is each model's smoothed family/group-balanced solve rate.
    Every response is a single likelihood term. Ancestors are covariates,
    never independent copies of an outcome. This is descriptive full-data MAP.
    """
    x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float)
    n,m=y.shape;weights=balanced_weights(families,groups)
    if x.shape[0]!=n or not np.isin(y,[0,1]).all():raise ValueError('Invalid panel')
    active=np.any(x!=0,axis=0);a=x[:,active];d=a.shape[1]
    prior=np.clip(weights@y,1e-4,1-1e-4);offset=logit(prior)
    lg=c['global_penalty'];ls=c['model_deviation_penalty'];calls=0
    def objective(v):
        nonlocal calls
        w=v[:d];dev=v[d:].reshape(d,m);z=a@(w[:,None]+dev)+offset
        loss=np.sum(weights*np.mean(np.logaddexp(0,z)-y*z,axis=1))+.5*lg*np.sum(w*w)+.5*ls*np.sum(dev*dev)/m
        gradient=a.T@((expit(z)-y)*weights[:,None])/m
        calls+=1
        if progress is not None and calls%10==0:progress(calls,float(loss))
        return loss,np.concatenate([gradient.sum(axis=1)+lg*w,(gradient+ls*dev/m).ravel()])
    result=minimize(objective,np.zeros(d*(m+1)),jac=True,method='L-BFGS-B',
                    options=dict(maxiter=c['max_iterations'],gtol=c['optimizer_gtol'],ftol=1e-11))
    w=np.zeros(x.shape[1]);dev=np.zeros((x.shape[1],m))
    w[active]=result.x[:d];dev[active]=result.x[d:].reshape(d,m)
    # Each particle is one historical subject in the same coordinate system.
    theta=dev.T.copy();theta[:,0]+=offset
    return dict(shared=w,theta=theta,prior=prior,active=active,
                log=dict(success=bool(result.success),message=str(result.message),iterations=int(result.nit),
                         evaluations=int(result.nfev),objective=float(result.fun),gradient_max=float(np.abs(result.jac).max()),
                         features=int(d),models=m,items=n))


def predict_profiles(x,fit):
    return expit(np.asarray(x)@(fit['shared'][:,None]+fit['theta'].T))


def weighted_capability_rates(y,membership,weights,prior_strength=2.):
    """Rates on identical tagged reference items for every model; not mastery."""
    y=np.asarray(y,dtype=float);member=np.asarray(membership,dtype=float)
    # Scale the family-balanced weights to preserve a modest pseudo-count.
    scaled=np.asarray(weights)*len(y)
    counts=member.T@scaled;success=member.T@(y*scaled[:,None])
    base=np.asarray(weights)@y
    return (success+prior_strength*base)/(counts[:,None]+prior_strength)


def remove_ability(profiles,ability):
    """Across-subject linear adjustment, including an intercept, descriptive."""
    a=np.column_stack([np.ones(len(ability)),np.asarray(ability)])
    p=np.asarray(profiles,dtype=float)
    return p-a@np.linalg.lstsq(a,p,rcond=None)[0]


def column_correlation(a,b=None):
    a=np.asarray(a,dtype=float);b=a if b is None else np.asarray(b,dtype=float)
    a=a-a.mean(axis=0);b=b-b.mean(axis=0)
    numerator=a.T@b;denominator=np.outer(np.linalg.norm(a,axis=0),np.linalg.norm(b,axis=0))
    result=np.divide(numerator,denominator,out=np.zeros_like(numerator),where=denominator>1e-14)
    return np.clip(result,-1,1)


def rasch_residual_profiles(y,membership,weights,iterations=60):
    """Descriptive item + model logistic adjustment, then topic score offsets.

    Item intercepts use observed answers, so this is an oracle descriptive
    control, not a question-only predictor. Weak fixed ridge prevents infinity.
    """
    y=np.asarray(y,dtype=float);weights=np.asarray(weights,dtype=float)
    ability=logit(np.clip(weights@y,1e-4,1-1e-4))
    item=logit((y.sum(axis=1)+.5)/(y.shape[1]+1));ridge=.01
    for iteration in range(iterations):
        p=expit(item[:,None]+ability);v=p*(1-p)
        step=(np.sum(y-p,axis=1)-ridge*item)/(v.sum(axis=1)+ridge)
        item+=np.clip(step,-1,1)
        p=expit(item[:,None]+ability);v=p*(1-p)
        astep=(weights@(y-p)-.0001*ability)/(weights@v+.0001)
        ability+=np.clip(astep,-1,1)
        center=ability.mean();ability-=center;item+=center
        if max(np.max(np.abs(step)),np.max(np.abs(astep)))<1e-6:break
    p=expit(item[:,None]+ability);scaled=weights*len(y)
    score=np.asarray(membership).T@((y-p)*scaled[:,None])
    information=np.asarray(membership).T@(p*(1-p)*scaled[:,None])
    return (score/(information+4)).T,dict(iterations=iteration+1,max_final_step=float(max(np.max(np.abs(step)),np.max(np.abs(astep)))))


def empirical_update(theta,probabilities,label,weights=None):
    """Exact Bayes for the stated finite historical-profile mixture.

    This is not an exact posterior for an unknown population or a Gaussian
    logistic model. The covariance uses population weights, not n-1 scaling.
    """
    theta=np.asarray(theta,dtype=float);q=np.asarray(probabilities,dtype=float)
    if theta.ndim!=2 or q.shape!=(len(theta),) or label not in [0,1]:raise ValueError('Shape/label')
    if not np.isfinite(theta).all() or not np.isfinite(q).all() or np.any((q<0)|(q>1)):raise ValueError('Invalid particles')
    weights=np.ones(len(theta))/len(theta) if weights is None else np.asarray(weights,dtype=float)
    if weights.shape!=q.shape or np.any(weights<0) or not np.isfinite(weights).all() or weights.sum()<=0:raise ValueError('Invalid weights')
    weights=weights/weights.sum();p=float(weights@q)
    if not 0<p<1:raise ValueError('Identity requires 0 < prior predictive p < 1')
    mean=weights@theta;cov=(weights*(q-p))@(theta-mean)
    likelihood=q if label else 1-q;posterior=weights*likelihood;posterior/=posterior.sum()
    updated=posterior@theta;identity=cov*(label-p)/(p*(1-p))
    return dict(weights=posterior,mean=updated,delta=updated-mean,identity_delta=identity,
                covariance=cov,probability=p,identity_error=float(np.max(np.abs(updated-mean-identity))))


def mixture_path(q,y,order,budgets,exclude=None):
    """Historical-type mixture predictions on unused questions; log stable."""
    q=np.clip(np.asarray(q,dtype=float),1e-8,1-1e-8);y=np.asarray(y)
    logw=np.zeros(q.shape[1]);
    if exclude is not None:logw[exclude]=-np.inf
    result={};previous=0
    for budget in budgets:
        if budget<previous:raise ValueError('Budgets must be nested')
        for ix in order[previous:budget]:logw+=np.log(q[ix] if y[ix] else 1-q[ix])
        normalized=np.exp(logw-logsumexp(logw))
        result[budget]=(q@normalized,float(1/(normalized@normalized)))
        previous=budget
    return result
