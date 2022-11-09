#!/usr/bin/env python

import functools
import numpy as np
from scipy.linalg import lstsq
from scipy.optimize import minimize
from scipy.stats import multivariate_normal



from .mcmc import MCMC



class lreg(object):
    def __init__(self):
        self.cf = None
        self.cf_cov = None
        self.datavar = 0.0
        self.fitted = False
        return

    def fit(self, Amat, y):
        raise NotImplementedError


    def print_coefs(self):
        assert(self.fitted)
        print(self.cf)

        return

    def predict(self, Amat, msc=0, pp=True):
        raise NotImplementedError

    # def predict(self, Amat, msc=0, pp=True):
    #     assert(self.fitted)
    #     if pp:
    #         factor = 1.0
    #     else:
    #         factor = 0.0
    #     ypred = Amat @ self.cf
    #     if msc==2:
    #         ypred_cov = (Amat @ self.cf_cov) @ Amat.T + factor*self.datavar*np.eye(Amat.shape[0])
    #         ypred_var = np.diag(ypred_cov)
    #     elif msc==1:
    #         ypred_cov = None
    #         try:
    #             ypred_var = self.compute_stdev(Amat, method='chol')**2 +factor*self.datavar
    #         except np.linalg.LinAlgError:
    #             ypred_var = self.compute_stdev(Amat, method='svd')**2 +factor*self.datavar
    #     elif msc==0:
    #         ypred_cov = None #np.zeros((Amat.shape[1], Amat.shape[1]))
    #         ypred_var = None #np.zeros((Amat.shape[1],))
    #     else:
    #         print(f"msc={msc}, but needs to be 0,1, or 2. Exiting.")
    #         sys.exit()

    #     return ypred, ypred_var, ypred_cov

    # def compute_stdev(self, Amat, method="chol"):
    #     assert(self.cf_cov is not None)
    #     if method == "chol":
    #         chol = np.linalg.cholesky(self.cf_cov)
    #         mat = Amat @ chol
    #         pf_stdev = np.linalg.norm(mat, axis=1)
    #     elif method == "choleye":
    #         eigvals = np.linalg.eigvalsh(self.cf_cov)
    #         chol = np.linalg.cholesky(self.cf_cov+(abs(eigvals[0]) + 1e-14) * np.eye(self.cf_cov.shape[0]))
    #         mat = Amat @ chol
    #         pf_stdev = np.linalg.norm(mat, axis=1)
    #     elif method == "svd":
    #         u, s, vh = np.linalg.svd(self.cf_cov, hermitian=True)
    #         mat = (Amat @ u) @ np.sqrt(np.diag(s))
    #         pf_stdev = np.linalg.norm(mat, axis=1)
    #     elif method == "loop":
    #         tmp = np.dot(Amat, self.cf_cov)
    #         pf_stdev = np.empty(Amat.shape[0])
    #         for ipt in range(Amat.shape[0]):
    #             pf_stdev[ipt] = np.sqrt(np.dot(tmp[ipt, :], Amat[ipt, :]))
    #     elif method == "fullcov":
    #         pf_stdev = np.sqrt(np.diag((Amat @ self.cf_cov) @ Amat.T))
    #     else:
    #         pf_stdev = np.zeros(Amat.shape[0])

    #     return pf_stdev

# Bare minimum least squares solution
# (note: scipy's lstsq uses svd under the hood)
class lsq(lreg):
    def __init__(self):
        super(lsq, self).__init__()

        return


    def fit(self, Amat, y):

        self.cf, residues, rank, s = lstsq(Amat, y, 1.0e-13)
        self.cf_cov = np.zeros((Amat.shape[1], Amat.shape[1]))
        self.fitted = True

        return



def logpost_emb(x, aw=None, bw=None, ind_sig=None, datavar=0.0, multiplicative=False, merr_method='abc', cfs=None):
    assert(aw is not None and bw is not None)
    npt, nbas = aw.shape

    if cfs is None:
        cfs = x[:nbas]
        sig_cfs = x[nbas:]
    else:
        sig_cfs = x.copy()

    # if(np.min(sig_cfs)<=0.0):
    #     return -1.e+80

    if ind_sig is None:
        ind_sig = range(nbas)

    if multiplicative:
        sig_cfs = np.abs(cfs[ind_sig]) * sig_cfs

    #print(sig_cfs.shape[0], len(ind_sig))
    assert(sig_cfs.shape[0] == len(ind_sig))
    ss = aw[:, ind_sig] * sig_cfs

    # #### FULL COVARIANCE
    if merr_method == 'full':
        cov = ss @ ss.T + datavar * np.eye(npt) #self.datavar is a small nugget was crucial for MCMC sanity!
        #return sgn*(multivariate_normal.logpdf(aw @ cfs, mean=bw, cov=np.diag(cov), allow_singular=True)-np.sum(np.log(np.abs(sig_cfs))))
        val = multivariate_normal.logpdf(aw @ cfs, mean=bw, cov=np.diag(cov), allow_singular=False)

    # #### IID
    elif merr_method == 'iid':
        err = aw @ cfs - bw
        stds = np.linalg.norm(ss, axis=1)
        stds = np.sqrt(stds**2+datavar)
        val = -0.5 * np.sum((err/stds)**2)
        val -= 0.5 * npt * np.log(2.*np.pi)
        val -= np.sum(np.log(stds))

    #### ABC
    elif merr_method == 'abc':
        abceps=0.1
        abcalpha=1.0
        err = aw @ cfs - bw
        stds = np.linalg.norm(ss, axis=1)
        stds = np.sqrt(stds**2+datavar)
        err2 = abcalpha*np.abs(err)-stds
        val = -0.5 * np.sum((err/abceps)**2)
        val = -0.5 * np.sum((err2/abceps)**2)
        val -= 0.5 * np.log(2.*np.pi)
        val -= np.log(abceps)

    else:
        print(f"Merr type {merr_method} unknown. Exiting.")
        sys.exit()

    #print(val)

    # Prior?
    #val -= np.sum(np.log(np.abs(sig_cfs)))

    return val



class lreg_merr(lreg):
    def __init__(self, ind_embed=None, datavar=0.0, multiplicative=False, merr_method='abc', method='bfgs', cfs_fixed=None):
        super(lreg_merr, self).__init__()

        self.ind_embed = ind_embed
        self.datavar = datavar
        self.multiplicative = multiplicative
        self.merr_method = merr_method
        self.method = method
        self.cfs_fixed = cfs_fixed
        return

    def fit(self, A, y):
        npts, nbas = A.shape
        assert(y.shape[0] == npts)

        if self.ind_embed is None:
            self.ind_embed = range(nbas)

        nbas_emb = len(self.ind_embed)

        logpost_params = {'aw': A, 'bw':y, 'ind_sig':self.ind_embed, 'datavar':self.datavar, 'multiplicative':self.multiplicative, 'merr_method':self.merr_method, 'cfs':self.cfs_fixed}

        if self.cfs_fixed is None:
            params_ini = np.random.rand(nbas+nbas_emb)
            #params_ini[:nbas], residues, rank, s = lstsq(A, y, 1.0e-13)
            invptp = np.linalg.inv(np.dot(A.T, A)+1.e-6*np.diag(np.ones((nbas,))))
            params_ini[:nbas] = np.dot(invptp, np.dot(A.T, y))
        else:
            params_ini = np.random.rand(nbas_emb)

        if self.method == 'mcmc':

            # res = minimize((lambda x, fcn, p: -fcn(x, **p)), params_ini, args=(logpost_emb,logpost_params), method='BFGS', options={'gtol': 1e-16})
            # print(res)
            # params_ini = res.x

            covini = 0.1 * np.ones((params_ini.shape[0], params_ini.shape[0]))
            nmcmc = 10000
            gamma = 0.5
            t0 = 100
            tadapt = 100
            calib_params = {'param_ini': params_ini, 'cov_ini': covini,
                            't0': t0, 'tadapt' : tadapt,
                            'gamma' : gamma, 'nmcmc' : nmcmc}
            calib = AMCMC()
            calib.setParams(**calib_params)
            #samples, cmode, pmode, acc_rate, acc_rate_all, pmode_all = amcmc([nmcmc, params_ini, gamma, t0, tadapt, covini], logpost_emb, A, y, ind_sig=ind_embed, sgn=1.0)
            calib_results = calib.run(logpost_emb, **logpost_params)
            samples, cmode, pmode, acc_rate = calib_results['chain'],  calib_results['mapparams'],calib_results['maxpost'], calib_results['accrate']

            np.savetxt('chn.txt', samples)
            np.savetxt('mapparam.txt', cmode)

            solution = cmode.copy()

        elif self.method == 'bfgs':
            #params_ini[nbas:] = np.random.rand(nbas_emb,)
            res = minimize((lambda x, fcn, p: -fcn(x, **p)), params_ini, args=(logpost_emb, logpost_params), method='BFGS', options={'gtol': 1e-3})
            solution = res.x.copy()

        if self.cfs_fixed is None:
            coeffs = solution[:nbas]
            coefs_sig = solution[nbas:]
        else:
            coeffs = self.cfs_fixed.copy()
            coefs_sig = solution.copy()

        self.cf = coeffs
        coefs_sig_all = np.zeros((nbas,))
        if self.multiplicative:
            coefs_sig_all[self.ind_embed] = np.abs(self.cf[self.ind_embed]) * coefs_sig
        else:
            coefs_sig_all[self.ind_embed] = coefs_sig
        self.cf_cov = np.diag(coefs_sig_all**2)
        self.fitted = True

        return
