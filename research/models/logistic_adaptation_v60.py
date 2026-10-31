"""Tiny identity-regularized logistic calibrators. No data or runtime side effects."""
import numpy as np
from scipy.special import expit, logsumexp

EPS = 1e-4


def logits(p):
    p = np.clip(np.asarray(p, float), EPS, 1-EPS)
    return np.log(p)-np.log1p(-p)


def objective(z, y, beta, intercept_penalty, slope_penalty):
    eta = z + beta[:, :1] + beta[:, 1:] * z
    return (np.logaddexp(0., eta)-y*eta).sum(1) + .5*(
        intercept_penalty*beta[:, 0]**2+slope_penalty*beta[:, 1]**2)


def fit(z, y, kind, intercept_penalty, slope_penalty=1.):
    """Sum Bernoulli NLL + ridge toward identity, constrained total slope >=0.

    z,y: episode x observed-item. Returned beta=(shift,slope_change), so
    calibrated_logit = z + shift + slope_change*z. Empty observations return 0.
    Active-set quadratic steps with Armijo backtracking preserve convex descent.
    """
    z, y = np.asarray(z, float), np.asarray(y, float)
    assert z.shape == y.shape and z.ndim == 2 and np.isfinite(z).all()
    assert np.isin(y, [0., 1.]).all()
    assert kind in ('intercept', 'affine') and intercept_penalty > 0 and slope_penalty > 0
    beta = np.zeros((len(z), 2))
    if z.shape[1] == 0:
        return beta
    for _ in range(80):
        eta = z+beta[:, :1]+beta[:, 1:]*z
        p = expit(eta)
        residual, w = p-y, p*(1-p)
        ga = residual.sum(1)+intercept_penalty*beta[:, 0]
        ha = w.sum(1)+intercept_penalty
        gb = (residual*z).sum(1)+slope_penalty*beta[:, 1]
        if kind == 'intercept':
            gb[:] = 0.
            direction = np.c_[-ga/ha, np.zeros(len(z))]
            violation = np.abs(ga)
        else:
            hb = (w*z*z).sum(1)+slope_penalty
            hab = (w*z).sum(1)
            det = ha*hb-hab*hab
            assert np.all(det > 0)
            da = (-hb*ga+hab*gb)/det
            db = (hab*ga-ha*gb)/det
            boundary = beta[:, 1]+db < -1.
            db[boundary] = -1.-beta[boundary, 1]
            da[boundary] = -(ga[boundary]+hab[boundary]*db[boundary])/ha[boundary]
            direction = np.c_[da, db]
            projected_gb = np.where(beta[:, 1] <= -1.+1e-10, np.minimum(gb, 0.), gb)
            violation = np.maximum(abs(ga), abs(projected_gb))
        active = violation > 1e-8
        if not active.any():
            break
        direction[~active] = 0.
        old = objective(z, y, beta, intercept_penalty, slope_penalty)
        directional_derivative = ga*direction[:, 0]+gb*direction[:, 1]
        assert np.max(directional_derivative) < 1e-9
        step = np.ones(len(z))
        for _ in range(40):
            candidate = beta+step[:, None]*direction
            value = objective(z, y, candidate, intercept_penalty, slope_penalty)
            bad = active & (value > old+1e-4*step*directional_derivative+1e-12)
            if not bad.any():
                break
            step[bad] *= .5
        else:
            raise AssertionError('Logistic calibration line search failed')
        beta = candidate
    eta = z+beta[:, :1]+beta[:, 1:]*z
    r = expit(eta)-y
    ga = r.sum(1)+intercept_penalty*beta[:, 0]
    gb = (r*z).sum(1)+slope_penalty*beta[:, 1]
    gb = np.where(beta[:, 1] <= -1.+1e-10, np.minimum(gb, 0.), gb)
    violation = np.max(abs(ga))
    if kind == 'affine':
        violation = max(violation, np.max(abs(gb)))
    assert violation < 2e-6, violation
    assert np.isfinite(beta).all() and (beta[:, 1] >= -1.-1e-12).all()
    return beta


def predict(ps, pq, labels, cfg):
    if cfg['kind'] == 'identity' or ps.shape[1] == 0:
        return np.array(pq, dtype=float, copy=True)
    beta = fit(logits(ps), labels, cfg['kind'], cfg['intercept_penalty'], cfg['slope_penalty'])
    zq = logits(pq)
    return expit(zq+beta[:, :1]+beta[:, 1:]*zq)


def bases(ps, pq, labels, base):
    """Fit-support bases exclude that support label for Bayesian mixtures.

    ps: episode x support x experts; pq can be episode x query x experts or
    query x experts shared across episodes. Inputs already clipped once.
    """
    episodes, n, _ = ps.shape
    if base == 'uniform':
        support = ps.mean(2)
        query = pq.mean(-1)
        if pq.ndim == 2:
            query = np.broadcast_to(query, (episodes, len(query)))
        return support, query
    assert base == 'bayes_loo' and labels.shape == (episodes, n)
    likelihood = labels[:, :, None]*np.log(ps)+(1-labels[:, :, None])*np.log1p(-ps)
    total = likelihood.sum(1)
    weights = np.exp(total-logsumexp(total, axis=1, keepdims=True))
    if pq.ndim == 2:
        query = weights @ pq.T
    else:
        query = np.einsum('eh,eqh->eq', weights, pq)
    if n:
        loo = total[:, None, :]-likelihood
        loo_weights = np.exp(loo-logsumexp(loo, axis=2, keepdims=True))
        support = np.sum(loo_weights*ps, axis=2)
    else:
        support = np.empty((episodes, 0))
    return support, query


def support_cv(ps, pq, labels, grid, fallback):
    """EXACT support-only LOO Brier selection for support-independent uniform bases.

    Never call this with self-label-dependent posterior bases. Before 7 answers
    use the SOURCE-frozen fallback. Candidate zero is identity, tie priority first.
    """
    episodes, n = ps.shape
    if n < 7:
        return predict(ps, pq, labels, fallback), np.full(episodes, -1)
    keep = ~np.eye(n, dtype=bool)
    z = logits(ps)
    ztrain = np.broadcast_to(z[:, None, :], (episodes, n, n))[:, keep].reshape(episodes*n, n-1)
    ytrain = np.broadcast_to(labels[:, None, :], (episodes, n, n))[:, keep].reshape(episodes*n, n-1)
    risks = []
    for cfg in grid:
        if cfg['kind'] == 'identity':
            pred = ps
        else:
            beta = fit(ztrain, ytrain, cfg['kind'], cfg['intercept_penalty'], cfg['slope_penalty'])
            zv = z.reshape(-1)
            pred = expit(zv+beta[:, 0]+beta[:, 1]*zv).reshape(episodes, n)
        risks.append(np.square(pred-labels).mean(1))
    choice = np.argmin(np.asarray(risks), axis=0)
    output = np.empty_like(pq, dtype=float)
    for k in np.unique(choice):
        use = choice == k
        output[use] = predict(ps[use], pq[use], labels[use], grid[int(k)])
    return output, choice
