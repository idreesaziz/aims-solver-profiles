"""Numerical primitives for descriptive geometry and inductive prediction audits."""
import numpy as np
from scipy.special import expit, logit
from scipy.spatial.distance import cdist
from scipy.linalg import eigh
from scipy.optimize import minimize


def fit_rasch(y, families, iterations=50, penalty=2.0):
    """Descriptive benchmark-specific subject abilities plus common item difficulty.

    Full observed historical panel only. This fit must never enter held-out
    predictions, where target outcomes are unavailable.
    """
    y = np.asarray(y, dtype=float)
    family_names, fi = np.unique(families, return_inverse=True)
    ability = np.stack([logit(np.clip(y[fi == j].mean(0), .005, .995))
                        for j in range(len(family_names))])
    item = np.zeros(len(y))
    for _ in range(iterations):
        p = expit(ability[fi] + item[:, None])
        step = ((y-p).sum(1) - penalty*item) / ((p*(1-p)).sum(1)+penalty)
        item += np.clip(step, -1, 1)
        p = expit(ability[fi] + item[:, None])
        for j in range(len(family_names)):
            mask = fi == j
            step = ((y[mask]-p[mask]).sum(0)-penalty*ability[j]) / ((p[mask]*(1-p[mask])).sum(0)+penalty)
            ability[j] += np.clip(step, -1, 1)
    p = expit(ability[fi]+item[:, None])
    return p.astype(np.float32), ability, item


def graph_association(signal, anchors, neighbors, weights):
    """Weighted neighbor covariance ratio; input signal should be centered."""
    a = signal[anchors]
    near = signal[neighbors].mean(1)
    numerator = (weights[:, None]*a*near).sum(0)
    denominator = (weights[:, None]*a*a).sum(0)
    return np.divide(numerator, denominator, out=np.full(signal.shape[1],np.nan), where=denominator>1e-10)


def family_center(y, families):
    r = np.asarray(y,dtype=np.float32).copy()
    for f in np.unique(families):
        mask = families == f
        r[mask] -= r[mask].mean(0)
    return r


def make_strata(families, difficulty, bins=10):
    strata = []
    for f in np.unique(families):
        ids = np.flatnonzero(families == f)
        # Equal-count strata; stable tie handling makes each row explicit.
        order = ids[np.argsort(difficulty[ids],kind='stable')]
        strata.extend([v for v in np.array_split(order,bins) if len(v)])
    return strata


def permute_strata(strata, n, rng):
    permutation=np.arange(n)
    for ids in strata: permutation[ids]=rng.permutation(ids)
    return permutation


def fit_diffusion_basis(x_train, landmarks=384, dimensions=48, seed=42):
    """Landmark diffusion eigenfunctions fitted on source features only.

    Density-normalized Gaussian kernel, alpha=1, diffusion time=1. New
    questions are extended through their transition probabilities to fixed
    training landmarks. No target feature distribution is used to fit geometry.
    """
    x_train=np.asarray(x_train,dtype=float)
    unique=np.unique(x_train,axis=0)
    rng=np.random.default_rng(seed)
    land=unique[rng.choice(len(unique),min(landmarks,len(unique)),replace=False)]
    d=cdist(land,land,'sqeuclidean')
    near=np.partition(d,min(10,len(land)-1),axis=1)[:,min(10,len(land)-1)]
    scale=max(float(np.median(near)),1e-5)
    kernel=np.exp(-d/(2*scale))
    q=kernel.sum(1)
    affinity=kernel/(q[:,None]*q[None,:])
    degree=affinity.sum(1)
    symmetric=affinity/np.sqrt(degree[:,None]*degree[None,:])
    values,vectors=eigh(symmetric)
    order=np.argsort(values)[::-1][1:dimensions+1]
    values=values[order]; psi=vectors[:,order]/np.sqrt(degree[:,None])
    basis=dict(landmarks=land,scale=scale,q=q,psi=psi,eigenvalues=values)
    z=transform_diffusion(basis,x_train,standardize=False)
    basis['mean']=z.mean(0);basis['std']=np.maximum(z.std(0),1e-7)
    return basis


def transform_diffusion(basis,x,standardize=True):
    answer=[]
    for start in range(0,len(x),256):
        kernel=np.exp(-cdist(x[start:start+256],basis['landmarks'],'sqeuclidean')/(2*basis['scale']))
        # q_x cancels in row normalization of alpha=1 affinity.
        transition=kernel/basis['q'][None,:]
        transition/=np.maximum(transition.sum(1,keepdims=True),1e-300)
        answer.append(transition@basis['psi'])
    z=np.concatenate(answer)
    if standardize: z=(z-basis['mean'])/basis['std']
    return z


def fit_logistic_multi(x,y,weights,penalty=.1):
    """Independent logistic heads in one optimizer; no cross-subject penalty."""
    x=np.c_[np.ones(len(x)),np.asarray(x,dtype=float)]
    y=np.asarray(y,dtype=float)
    if y.ndim==1:y=y[:,None]
    w=np.asarray(weights,dtype=float);w/=w.sum()
    initial=np.zeros((x.shape[1],y.shape[1]))
    initial[0]=logit(np.clip(w@y,.001,.999))
    def objective(flat):
        b=flat.reshape(initial.shape);eta=x@b;p=expit(eta)
        value=(w[:,None]*(np.logaddexp(0,eta)-y*eta)).sum()+.5*penalty*(b[1:]**2).sum()
        gradient=x.T@(w[:,None]*(p-y));gradient[1:]+=penalty*b[1:]
        return value,gradient.ravel()
    result=minimize(objective,initial.ravel(),jac=True,method='L-BFGS-B',
                    options=dict(maxiter=250,ftol=1e-11,gtol=1e-6))
    if not result.success:raise RuntimeError('Logistic optimization failed: '+str(result.message))
    return result.x.reshape(initial.shape),dict(iterations=int(result.nit),gradient_max=float(np.max(np.abs(result.jac))))


def logistic_predictions(x,b):return expit(np.c_[np.ones(len(x)),x]@b)


def laplace_update(mean,covariance,features,outcome):
    p=float(expit(features@mean));v=covariance@features
    updated=covariance-(p*(1-p)/(1+p*(1-p)*(features@v)))*np.outer(v,v)
    new_mean=mean+updated@features*(outcome-p)
    return new_mean,(updated+updated.T)/2


def persistence_statistics(x):
    from ripser import ripser
    x=np.unique(x,axis=0)
    if len(x)<4:return dict(points=len(x),h0_max=0.,h1_max=0.,h1_total=0.,h1_count=0)
    diagrams=ripser(x,maxdim=1)['dgms']
    def lives(d):
        finite=d[np.isfinite(d[:,1])]
        return finite[:,1]-finite[:,0]
    h0,h1=map(lives,diagrams)
    return dict(points=len(x),h0_max=float(np.max(h0,initial=0)),h1_max=float(np.max(h1,initial=0)),
                h1_total=float(h1.sum()),h1_count=int(np.sum(h1>1e-7)))
