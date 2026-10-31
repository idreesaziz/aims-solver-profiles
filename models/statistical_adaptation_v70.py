"""Probability pooling, posterior averaging, and low-rank solver interactions.

Research prototypes. Beta-binomial conjugacy and proper predictive means,
Laplace GLM prediction, and regularized binary matrix factorization are standard
methods. No claim of inventing these statistical techniques.
"""
import numpy as np
from scipy.special import expit,logit,betaln,logsumexp
from scipy.optimize import minimize

KAPPAS=np.array([.25,.5,2.,8.,32.])


def intercept(offset,y,penalty):
    """Vectorized independent scalar logistic MAP fits, fractional y permitted."""
    z=np.asarray(offset,float);y=np.asarray(y,float)
    assert z.shape==y.shape and z.ndim==2
    b=np.zeros(len(z))
    for _ in range(20):
        p=expit(z+b[:,None]);v=p*(1-p)
        step=np.clip(((y-p).sum(1)-penalty*b)/(v.sum(1)+penalty),-1.5,1.5)
        b+=step
        if np.max(abs(step),initial=0)<1e-10:break
    return b


def beta_posterior(prior,successes,counts,kappa):
    p=np.clip(np.asarray(prior,float),1e-5,1-1e-5)
    return (kappa*p+successes)/(kappa+counts)


def beta_mixture(prior,k,n,kappas=KAPPAS):
    """Independent item-level mixture of Beta priors, exact component updates."""
    p=np.clip(np.asarray(prior,float),1e-5,1-1e-5)
    a=kappas[:,None]*p[None,:];b=kappas[:,None]*(1-p[None,:])
    evidence=betaln(a+k,b+n-k)-betaln(a,b)
    weights=np.exp(evidence-logsumexp(evidence,axis=0,keepdims=True))
    return np.sum(weights*(a+k)/(kappas[:,None]+n),axis=0)


def beta_predictions(p0,supports,observed,queries,global_penalty=1.,solver_penalty=16.):
    """Cross-fitted two-way pooling. No query outcomes accepted.

    Each item has one vote in global calibration regardless of repeated models.
    Global prior for an observed item excludes ALL outcomes on that item.
    A solver's calibration residual excludes its own outcome from item pooling.
    """
    p0=np.clip(np.asarray(p0,float),1e-5,1-1e-5);subjects=len(supports)
    sid=np.concatenate([np.repeat(s,len(q)) for s,q in enumerate(supports)])
    iid=np.concatenate(supports);ys=np.concatenate(observed)
    keys=np.unique(iid);counts=np.array([np.sum(iid==i) for i in keys],float)
    successes=np.array([ys[iid==i].sum() for i in keys],float)
    lookup={int(i):j for j,i in enumerate(keys)}
    u=len(keys)
    # One fractional binomial observation per question avoids counting strongly
    # correlated same-question answers as independent global difficulty evidence.
    if u:
        offsets=np.broadcast_to(logit(p0[keys]),(u+1,u)).copy()
        fraction=np.broadcast_to(successes/counts,(u+1,u)).copy()
        # Fit exact leave-item-out models by zeroing excluded contribution. The
        # equivalent rectangular arrays have u-1 columns for each LOO fit.
        if u>1:
            mask=~np.eye(u,dtype=bool)
            b_item=intercept(offsets[:u][mask].reshape(u,u-1),fraction[:u][mask].reshape(u,u-1),global_penalty)
        else:b_item=np.zeros(u)
        b_all=intercept(offsets[-1:],fraction[-1:],global_penalty)[0]
    else:b_item=np.empty(0);b_all=0.
    item_prior=expit(logit(p0[keys])+b_item)
    candidates={f'beta{k:g}':float(k) for k in KAPPAS};candidates['beta_mixture']=None
    predictions={}
    def pool(m,k,n,kappa):
        return beta_mixture(m,k,n) if kappa is None else beta_posterior(m,k,n,kappa)
    for name,kappa in candidates.items():
        row_bases=[]
        for s,items in enumerate(supports):
            where=np.array([lookup[int(i)] for i in items],int)
            row_bases.append(pool(item_prior[where],successes[where]-observed[s],counts[where]-1,kappa))
        if len(iid):
            # Equal observations per model in these cohorts. Empty shapes also
            # handled explicitly; row calibration sees no self-label base.
            n=len(supports[0]);assert all(len(s)==n for s in supports)
            raw=np.clip(np.array(row_bases),1e-5,1-1e-5)
            model_b=intercept(logit(raw),np.array(observed,float),solver_penalty)
        else:model_b=np.zeros(subjects)
        out=[];plain=[]
        for s,q in enumerate(queries):
            count=np.array([counts[lookup[int(i)]] if int(i) in lookup else 0. for i in q])
            success=np.array([successes[lookup[int(i)]] if int(i) in lookup else 0. for i in q])
            shift=np.array([b_item[lookup[int(i)]] if int(i) in lookup else b_all for i in q])
            prior=expit(logit(p0[q])+shift)
            probability=pool(prior,success,count,kappa)
            plain.append(np.clip(probability,.001,.999))
            out.append(np.clip(expit(logit(np.clip(probability,1e-5,1-1e-5))+model_b[s]),.001,.999))
        predictions[name]=out
        if name=='beta_mixture':predictions['beta_mixture_no_solver']=plain
    # A coherent shared-concentration model with unshifted source means:
    # integrate kappa across all CURRENT labels, then exact leave-one-label-out
    # component weights for calibration. No held query outcomes are used.
    if u:
        a=KAPPAS[:,None]*p0[keys];b=KAPPAS[:,None]*(1-p0[keys])
        loge=betaln(a+successes,b+counts-successes)-betaln(a,b)
        total=loge.sum(1)
    else:
        a=np.empty((len(KAPPAS),0));b=a.copy();loge=a.copy();total=np.zeros(len(KAPPAS))
    full_weight=np.exp(total-logsumexp(total))
    rowbase=[]
    for s,items in enumerate(supports):
        vals=[]
        for i,y in zip(items,observed[s]):
            j=lookup[int(i)];k=successes[j]-y;n=counts[j]-1
            revised=betaln(a[:,j]+k,b[:,j]+n-k)-betaln(a[:,j],b[:,j])
            lw=total-loge[:,j]+revised;w=np.exp(lw-logsumexp(lw))
            vals.append(float(w@((a[:,j]+k)/(KAPPAS+n))))
        rowbase.append(vals)
    n=len(supports[0])
    rb=np.clip(np.array(rowbase).reshape(subjects,n),1e-5,1-1e-5)
    shift=0.;delta=np.zeros(subjects)
    for _ in range(25):
        ps=expit(logit(rb)+shift+delta[:,None]);v=ps*(1-ps)
        step=np.clip(((np.array(observed)-ps).sum()-shift/(1.5**2+.8**2*.9))/(v.sum()+1/(1.5**2+.8**2*.9)),-1.5,1.5)
        shift+=step
        ps=expit(logit(rb)+shift+delta[:,None]);v=ps*(1-ps)
        step=np.clip(((np.array(observed)-ps).sum(1)-delta/(.8**2*.1))/(v.sum(1)+1/(.8**2*.1)),-1.5,1.5)
        delta+=step
    calibrated=[];uncalibrated=[]
    for s,q in enumerate(queries):
        k=np.array([successes[lookup[int(i)]] if int(i) in lookup else 0. for i in q])
        nq=np.array([counts[lookup[int(i)]] if int(i) in lookup else 0. for i in q])
        means=(KAPPAS[:,None]*p0[q]+k)/(KAPPAS[:,None]+nq)
        prob=full_weight@means
        uncalibrated.append(np.clip(prob,.001,.999))
        calibrated.append(np.clip(expit(logit(np.clip(prob,1e-5,1-1e-5))+shift+delta[s]),.001,.999))
    predictions['evidence_bma']=calibrated
    predictions['evidence_bma_no_postcal']=uncalibrated
    return predictions


def marginal_offset(p0,variance):
    """Invert logistic-normal mean so the no-data predictive mean stays p0."""
    from numpy.polynomial.hermite import hermgauss
    nodes,weights=hermgauss(48);weights=weights/np.sqrt(np.pi)
    p=np.clip(np.asarray(p0,float),1e-5,1-1e-5)
    a=logit(p)*np.sqrt(1+np.pi*variance/8)
    for _ in range(30):
        q=expit(a[:,None]+np.sqrt(2*variance)*nodes[None,:])
        error=q@weights-p
        step=error/np.maximum((q*(1-q))@weights,1e-12)
        a-=np.clip(step,-2,2)
        if np.max(abs(error))<1e-12:break
    return a


def logistic_fit(p0,supports,observed,item_sd=1.5,anchor=False):
    """Full-convergence convex Gaussian-ridge logistic model.

    Equivalent strong-sharing priors: beta variance1.5²+.8²*.9;
    solver contrast variance.8²*.1. Same model as equicorrelated update, new
    parameterization/convergence; contrasts not forced sum-zero.
    """
    sid=np.concatenate([np.repeat(s,len(q)) for s,q in enumerate(supports)])
    iid=np.concatenate(supports);y=np.concatenate(observed)
    subjects=len(supports);keys=np.unique(iid);lookup={int(i):j for j,i in enumerate(keys)}
    item=np.array([lookup[int(i)] for i in iid],int)
    penalty=np.r_[1/(1.5**2+.8**2*.9),np.repeat(1/(.8**2*.1),subjects),np.repeat(1/item_sd**2,len(keys))]
    offset_all=(marginal_offset(p0,1.5**2+.8**2+item_sd**2) if anchor else logit(np.clip(p0,1e-5,1-1e-5)))
    z=offset_all[iid];size=len(penalty)
    def objective(beta):
        eta=z+beta[0]+beta[1+sid]+beta[1+subjects+item]
        residual=expit(eta)-y
        gradient=penalty*beta
        gradient[0]+=residual.sum()
        gradient[1:1+subjects]+=np.bincount(sid,weights=residual,minlength=subjects)
        gradient[1+subjects:]+=np.bincount(item,weights=residual,minlength=len(keys))
        value=np.sum(np.logaddexp(0,eta)-y*eta)+.5*np.sum(penalty*beta*beta)
        return float(value),gradient
    fit=minimize(objective,np.zeros(size),jac=True,method='L-BFGS-B',options={'gtol':1e-8,'ftol':1e-12,'maxiter':350})
    assert np.max(abs(objective(fit.x)[1]))<2e-4
    beta=fit.x
    design=np.zeros((len(y),size));design[:,0]=1.
    design[np.arange(len(y)),1+sid]=1.;design[np.arange(len(y)),1+subjects+item]=1.
    eta=z+design@beta;v=expit(eta)*(1-expit(eta))
    hessian=(design.T*v)@design+np.diag(penalty)
    covariance=np.linalg.inv(hessian)
    return dict(beta=beta,covariance=covariance,keys=keys,lookup=lookup,subjects=subjects,offset_all=offset_all,
                prior_item_variance=item_sd**2,sid=sid,iid=iid,y=y,item=item,
                gradient_max=float(np.max(abs(objective(fit.x)[1]))))


def logistic_predict(p0,queries,fit):
    from numpy.polynomial.hermite import hermgauss
    nodes,weights=hermgauss(48);weights=weights/np.sqrt(np.pi)
    beta=fit['beta'];cov=fit['covariance'];subjects=fit['subjects']
    out={key:[] for key in ['map','posterior_mean']}
    for s,q in enumerate(queries):
        design=np.zeros((len(q),len(beta)));design[:,0]=1.;design[:,1+s]=1.
        unseen=np.array([int(i) not in fit['lookup'] for i in q])
        for j,i in enumerate(q):
            if not unseen[j]:design[j,1+subjects+fit['lookup'][int(i)]]=1.
        mean=fit['offset_all'][q]+design@beta
        variance=np.einsum('qi,ij,qj->q',design,cov,design)+unseen*fit['prior_item_variance']
        integrated=expit(mean[:,None]+np.sqrt(2*np.maximum(variance,0))[:,None]*nodes[None,:])@weights
        out['map'].append(np.clip(expit(mean),.01,.99))
        out['posterior_mean'].append(np.clip(integrated,.001,.999))
    return out


def factor_predictions(p0,supports,observed,queries,base_fit,rank=1,penalty=2.,_gradient_check=False):
    """Regularized1-bit matrix completion with additive baseline and low-rank residual.

    New item with no observed answers gets zero interaction; no text oracle.
    Two deterministic initializations, choose smaller current-support objective.
    """
    subjects=len(supports);sid=base_fit['sid'];iid=base_fit['iid'];y=base_fit['y'];item=base_fit['item']
    keys=base_fit['keys'];items=len(keys);base_size=len(base_fit['beta'])
    if not len(y):return logistic_predict(p0,queries,base_fit)['map']
    prior=np.r_[1/(1.5**2+.8**2*.9),np.repeat(1/(.8**2*.1),subjects),np.repeat(1/1.5**2,items)]
    z=logit(np.clip(p0[iid],1e-5,1-1e-5))
    def unpack(v):return v[:base_size],v[base_size:base_size+subjects*rank].reshape(subjects,rank),v[base_size+subjects*rank:].reshape(items,rank)
    def objective(v):
        beta,u,w=unpack(v)
        eta=z+beta[0]+beta[1+sid]+beta[1+subjects+item]+np.sum(u[sid]*w[item],axis=1)
        residual=expit(eta)-y
        gb=prior*beta
        gb[0]+=residual.sum();gb[1:1+subjects]+=np.bincount(sid,weights=residual,minlength=subjects)
        gb[1+subjects:]+=np.bincount(item,weights=residual,minlength=items)
        gu=penalty*u;gw=penalty*w
        np.add.at(gu,sid,residual[:,None]*w[item]);np.add.at(gw,item,residual[:,None]*u[sid])
        value=np.sum(np.logaddexp(0,eta)-y*eta)+.5*np.sum(prior*beta*beta)+.5*penalty*(np.sum(u*u)+np.sum(w*w))
        return float(value),np.r_[gb,gu.ravel(),gw.ravel()]
    if _gradient_check:
        from scipy.optimize import check_grad
        rng=np.random.default_rng(700)
        initial=np.r_[base_fit['beta'],rng.normal(0,.15,subjects*rank),rng.normal(0,.15,items*rank)]
        return float(check_grad(lambda v:objective(v)[0],lambda v:objective(v)[1],initial))
    best=None
    for seed in [701,702]:
        rng=np.random.default_rng(seed)
        initial=np.r_[base_fit['beta'],rng.normal(0,.15,subjects*rank),rng.normal(0,.15,items*rank)]
        opt=minimize(objective,initial,jac=True,method='L-BFGS-B',options={'maxiter':180,'gtol':1e-6,'ftol':1e-10})
        if best is None or opt.fun<best.fun:best=opt
    beta,u,w=unpack(best.x);pred=[]
    for s,q in enumerate(queries):
        eta=logit(np.clip(p0[q],1e-5,1-1e-5))+beta[0]+beta[1+s]
        for j,i in enumerate(q):
            if int(i) in base_fit['lookup']:
                k=base_fit['lookup'][int(i)];eta[j]+=beta[1+subjects+k]+u[s]@w[k]
        pred.append(np.clip(expit(eta),.001,.999))
    return pred
