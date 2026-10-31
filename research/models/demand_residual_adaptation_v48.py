"""Small probability-space adaptation functions; no dataset or runtime side effects."""
import numpy as np
from scipy.special import logsumexp

BUDGETS = [0, 1, 3, 7, 15, 31]
ALC = np.array([.1, .2, .2, .2, .2, .1])


def expert_prediction(support_experts, query_experts, labels, budget, method):
    """Batch shapes E,S,H / E,Q,H / E,S; model columns already excluded."""
    ps=np.clip(support_experts[:,:budget],1e-4,1-1e-4)
    if method=='bayes':
        logw=(labels[:,:budget,None]*np.log(ps)+(1-labels[:,:budget,None])*np.log1p(-ps)).sum(1)
    elif method in ('brier_mean','brier_strong'):
        logw=-2*np.square(ps-labels[:,:budget,None]).sum(1)
    else:
        raise ValueError(method)
    w=np.exp(logw-logsumexp(logw,axis=1,keepdims=True))
    output=np.empty(query_experts.shape[:2],dtype=float)
    for begin in range(0,query_experts.shape[1],128):
        pq=np.clip(query_experts[:,begin:begin+128],1e-4,1-1e-4)
        if method=='brier_strong':
            a0=np.einsum('eqh,eh->eq',np.exp(-2*pq*pq),w)
            a1=np.einsum('eqh,eh->eq',np.exp(-2*(1-pq)**2),w)
            output[:,begin:begin+128]=np.clip(.5+.25*np.log(a1/a0),0,1)
        else:output[:,begin:begin+128]=np.einsum('eqh,eh->eq',pq,w)
    return output


def ridge_features(us,uq,base_s,base_q,labels,budget,global_penalty,local_penalty):
    """Primal ridge, batched episodes; local features can be empty."""
    if budget==0:
        return base_q.copy()
    ds=np.concatenate([np.ones((*us.shape[:1],budget,1)),us[:,:budget]],axis=2)
    size=ds.shape[2]
    penalty=np.full(size,local_penalty,dtype=float);penalty[0]=global_penalty
    assert np.isfinite(penalty).all() and (penalty>0).all()
    residual=labels[:,:budget]-base_s[:,:budget]
    if size<=budget:
        lhs=np.einsum('esd,esf->edf',ds,ds)+np.diag(penalty)
        rhs=np.einsum('esd,es->ed',ds,residual)
        beta=np.linalg.solve(lhs,rhs[:,:,None])[:,:,0]
    else:
        weighted=ds/penalty
        lhs=weighted@ds.transpose(0,2,1)+np.eye(budget)
        alpha=np.linalg.solve(lhs,residual[:,:,None])[:,:,0]
        beta=np.einsum('esd,es->ed',weighted,alpha)
    return np.clip(base_q+beta[:,0,None]+np.einsum('eqd,ed->eq',uq,beta[:,1:]),0,1)


def ridge_kernel(kss,kqs,base_s,base_q,labels,budget,global_penalty,local_penalty):
    if budget==0:
        return base_q.copy()
    ks=kss[:,:budget,:budget]/local_penalty+1/global_penalty
    kq=kqs[:,:,:budget]/local_penalty+1/global_penalty
    alpha=np.linalg.solve(ks+np.eye(budget), (labels[:,:budget]-base_s[:,:budget])[:,:,None])[:,:,0]
    return np.clip(base_q+np.einsum('eqs,es->eq',kq,alpha),0,1)


def rbf_blocks(xs,xq,length):
    scale=2*length*length*xs.shape[-1]
    sn=np.square(xs).sum(-1);qn=np.square(xq).sum(-1)
    kss=np.exp(-np.maximum(sn[:,:,None]+sn[:,None,:]-2*(xs@xs.transpose(0,2,1)),0)/scale)
    kqs=np.exp(-np.maximum(qn[:,:,None]+sn[:,None,:]-2*(xq@xs.transpose(0,2,1)),0)/scale)
    return kss,kqs


def make_network(rank):
    import torch
    from torch import nn
    class FeatureMap(nn.Module):
        def __init__(self):
            super().__init__()
            self.net=nn.Sequential(nn.Linear(18,32),nn.GELU(),nn.Linear(32,rank),nn.Tanh())
            self.log_global=nn.Parameter(torch.tensor(np.log(64.),dtype=torch.float32))
            self.log_local=nn.Parameter(torch.tensor(np.log(16.),dtype=torch.float32))

        def features(self,x,anchors):
            center=self.net(anchors).mean(0)
            return (self.net(x)-center)/np.sqrt(rank)

        def penalties(self):
            return self.log_global.clamp(0,np.log(4096)).exp(), self.log_local.clamp(0,np.log(4096)).exp()

        def predict(self,xs,xq,anchors,base_s,base_q,labels,budget):
            if budget==0:return base_q
            batch=xs.shape[0]
            u=self.features(torch.cat([xs[:,:budget],xq],dim=1),anchors)
            us,uq=u[:,:budget],u[:,budget:]
            ds=torch.cat([torch.ones((batch,budget,1),dtype=u.dtype),us],dim=2)
            dq=torch.cat([torch.ones((*uq.shape[:2],1),dtype=u.dtype),uq],dim=2)
            lg,ll=self.penalties()
            penalty=torch.diag(torch.cat([lg.reshape(1),ll.expand(rank)]))
            lhs=ds.transpose(1,2)@ds+penalty
            rhs=ds.transpose(1,2)@(labels[:,:budget]-base_s[:,:budget]).unsqueeze(-1)
            beta=torch.linalg.solve(lhs,rhs)
            return (base_q+(dq@beta).squeeze(-1)).clamp(0,1)
    return FeatureMap()


def numpy_features(x,state,anchor_center):
    from scipy.special import erf
    z=x@state['net.0.weight'].T+state['net.0.bias']
    z=.5*z*(1+erf(z/np.sqrt(2)))
    z=np.tanh(z@state['net.2.weight'].T+state['net.2.bias'])
    return (z-anchor_center)/np.sqrt(z.shape[-1])


def feature_center(anchors,state):
    # numpy_features subtracts the supplied center after the final tanh.
    return numpy_features(anchors,state,np.zeros(len(state['net.2.bias']))).mean(0)*np.sqrt(len(state['net.2.bias']))
