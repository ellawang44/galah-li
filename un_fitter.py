import numpy as np
import ultranest


class UNFitter():

    def __init__(self, wl_obs, flux_obs, flux_err, fitter, constraints, opt=None, nwalkers=8):
        self.wl_obs = wl_obs
        self.flux_obs = flux_obs
        self.flux_err = flux_err
        self.fitter = fitter
        self.c = 299792.458
        # define variables
        self.constraints = constraints 
        self.ndim = len(self.constraints)
        # randomise initial position of walkers
        p0 = np.zeros((nwalkers, self.ndim))
        for i in range(nwalkers):
            for j in range(self.ndim):
                minP, maxP = constraints[j]
                p0[i][j] = np.random.uniform(minP, maxP)
        # names of columns
        if self.ndim == 4:
            param_names = ['A(Li)', 'vbroad', 'rv', 'const']
        elif self.ndim == 7:
            param_names = ['A(Li)', 'C/N1', 'Fe', 'C/N2', 'Ce/V', 'C/N3', 'const']
        else:
            param_names = ['A(Li)', 'vbroad', 'C/N1', 'Fe', 'C/N2', 'Ce/V', 'C/N3', 'const']
        self.sampler = ultranest.ReactiveNestedSampler(param_names, self.like, self.transform)
        self.results = self.sampler.run(viz_callback=False, show_status=False)

    def prob(self, param):
        '''
        param: the parameters of the current iteration
        combines the prior with the likelihood function
        '''

        prior = self.prior(param)
        if not np.isfinite(prior):
            return -np.inf
        return self.prior(param) + self.like(param)

    def prior(self, param):
        '''
        param: the parameters of the current iteration
        prior function, uniform with limits currently
        '''
        for i, p in enumerate(param):
            if (p < self.constraints[i][0]) or (self.constraints[i][1] < p):
                return -np.inf
        return 0

    def transform(self, param):
        '''from 0-1 to original parameter space
        '''

        new_params = []
        for i, p in enumerate(param):
            scale = self.constraints[i][1] - self.constraints[i][0]
            new_params.append((p * scale) + self.constraints[i][0])
        return new_params

    def like(self, param):
        '''
        param: the parameters of the current iteration
        The likelihood function, logged so you can sum
        '''
        if self.ndim == 4:
            rv = param[-2]
            std = self.fitter.std_galah
        else:
            rv = self.fitter.rv
            std = self.fitter.std
        wl_left = 6706.730*(1+rv/self.c)-std*2
        wl_right = 6708.961*(1+rv/self.c)+std*2
        mask = (wl_left <= self.wl_obs) & (self.wl_obs <= wl_right)
        return -(np.sum(np.square((self.fitter.model(self.wl_obs[mask], param) - self.flux_obs[mask])/self.flux_err[mask])))

