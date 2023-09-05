from fit import FitBroad, FitG, FitGFixed, FitB, FitBFixed, iter_fit, amp_to_init, pred_amp, cc_rv, _wl, bline, gline
from synth import _spectra, Grid
import numpy as np
import matplotlib.pyplot as plt
from breidablik.interpolate.spectra import Spectra
from breidablik.analysis import read, tools
from config import *
from scipy.interpolate import interp1d, CubicSpline
from astro_tools import vac_to_air
from un_fitter import UNFitter
import corner
import time

class FitSpec:
    '''Fitting 1 spectrum, contains plotting, and save/load. 
    '''

    def __init__(self, std_galah, stdl, stdu, stdue, rv_lim, e_vbroad, e_rv, snr, sid, teff, logg, feh):
        '''
        Parameters
        ----------
        std_galah : float
            std from galah, in \AA. Used as the initial std for fitting
        stdl : float
            lower limit on std for gaussians (not used for Breidablik), in \AA. Corresponds to 10 km/s for this project, because instrumental profile. 
        stdu : float
            upper limit on std for both gaussians and Breidablik, in \AA. Corresponds to galah std + some error in terms of R=22000
        stdue : float
            upper limit on std for both gaussians and Breidablik, in \AA. Corresponds to galah std + galah error in std + some error in terms of R=22000
        rv_lim : float
            The limit on rv, mirrored limit on either side, it is the same limit as stdu, except in km/s.
        e_vbroad : float
            The error on std
        e_rv : float
            The error on rv
        snr : float
            The SNR per pixel of the spectrum.
        sid : int
            The sobject_id of the spectrum.
        teff : float
            The GALAH DR# of teff for the spectrum
        logg : float
            The GALAH DR# of logg for the spectrum
        feh : float
            The GALAH DR# of feh for the spectrum
        '''

        self.std_galah = std_galah
        self.stdl = stdl
        self.stdu = stdu
        self.stdue = stdue
        self.rv_lim = rv_lim
        self.e_vbroad = e_vbroad
        self.e_rv = e_rv
        self.snr = snr
        self.sid = sid
        self.teff = teff
        self.logg = logg
        self.feh = feh
        self.li_center = 6707.8139458 # weighted mean of smith98 Li7 line centers
        self.c = 299792.458 # speed of light in km/s
        if self.grid_check(np.array([teff]), np.array([logg]), np.array([feh]))[0]: # valid sp from galah
            self.mode = 'Breidablik'
        else:
            self.mode = 'Gaussian'
        # make model to translate between rew and abundance
        # can't run breidablik with nans
        if self.mode == 'Breidablik':
            self.gen_ew_to_abund()

    def grid_check(self, teffs, loggs, fehs):
        with open('grid_snapshot.txt', 'r') as f:
            t_step, m_step = np.float_(f.readline().split())
            grid = np.loadtxt(f)
        scaled_sp = np.array([teffs*t_step, loggs, fehs*m_step]).T
        tile = np.array([np.tile(sp, (grid.shape[0], 1)) for sp in scaled_sp])
        dist = np.sqrt(np.sum(np.square(grid - tile), axis = 2))
        min_dist = np.min(dist, axis=1)
        in_grid = np.sqrt(3*0.25**2) > min_dist
        return in_grid
    
    def gen_ew_to_abund(self):
        '''Generate the function converting rew to abundances

        Returns
        -------
        func : function
            function which takes rew and gives the corresponding abundance
        '''

        # calculate corresponding ews
        abunds = list(np.arange(-0.5, 5.1, 0.18)) # extrapolating 1 dex
        ews = np.array([self.li_center*10**tools.rew(_wl, spec, center=6709.659, upper=100, lower=100) for spec in _spectra._predict_flux(self.teff, self.logg, self.feh, abunds)])
        # set values
        self.min_ew = ews[0]
        self.max_ew = ews[-1]
        self.ew_to_abund = lambda x: CubicSpline(ews, abunds)(x)

    def fit_broad(self, spectra, center=np.array([6696.085, 6698.673, 6703.565, 6705.101, 6710.317, 6711.819, 6713.095, 6713.742, 6717.681])):
        '''Fit the broad region of the spectrum, ews, std, and rv simultaneously. None if the star is metal-poor (less than 3 lines with amplitudes above noise).

        Parameters
        ----------
        spectra : dict
            Dictionary containing spectrum, from read (keys: wave_norm, sob_norm, uob_norm) 
        center : 1darray
            The centers of the lines used in the fitting. Default values:
            np.array([6696.085, 6698.673, 6703.565, 6705.101, 6710.317, 6711.819, 6713.095, 6713.742, 6717.681])
        '''

        res = iter_fit(spectra['wave_norm'], spectra['sob_norm'], spectra['uob_norm'], center=center, stdl=self.stdl, stdu=self.stdu, std_init=self.std_galah, rv_lim=self.rv_lim)
        if res is None: # metal-poor
            self.metal_poor = True
            self.broad_fit = None
        else: # non-metal-poor
            self.metal_poor = False
            self.broad_fit = {'amps':res[:-2], 'std':res[-2], 'rv':res[-1]}
        # save the centers used
        self.broad_center = center
    
    def mp_init(self, spectra, perc):
        '''Init values for metal poor fit, involves a cross correlation for better rv estimate.
        
        Parameters
        ----------
        spectra : dict
            Dictionary containing spectrum, from read (keys: wave_norm, sob_norm, uob_norm) 
        '''
        
        # use cross correlated initial rv
        amps, _, _ = pred_amp(spectra['wave_norm'], spectra['sob_norm'], spectra['uob_norm'], centers=[self.li_center], rv=0, perc=perc)
        init = amp_to_init(amps, self.std_galah, 0, 1)[:-1]
        init_rv = cc_rv(spectra['wave_norm'], spectra['sob_norm'], [self.li_center], init[:-1], init[-1], self.rv_lim)
        # predict amps again with cross correlated initial rv
        amps, _, const = pred_amp(spectra['wave_norm'], spectra['sob_norm'], spectra['uob_norm'], [self.li_center], rv=init_rv, perc=perc)
        init = amp_to_init(amps, self.std_galah, init_rv, const)
        return init

    def fit_li(self, spectra, center=np.array([6707.8139458, 6706.730, 6707.433, 6707.545, 6708.096, 6708.810, 6709.011])):
        '''Fit the Li region of the spectrum, fits ew, std, and rv if metal-poor (less than 3 lines with amplitudes above noise), or else fixes std and rv and only fits ews.
        
        Parameters
        ----------
        spectra : dict
            Dictionary containing spectrum, from read (keys: wave_norm, sob_norm, uob_norm) 
        center : 1darray
            The centers of the lines used in the fitting. First value is Li, not really used in the fitting, but needed for the initial guess. Default values:
            np.array([6707.8139458, 6706.730, 6707.433, 6707.545, 6708.096, 6708.810, 6709.01]))
        '''
        
        self.narrow_center = center # save centers used
        
        # if any sp is nan, then don't do fit, because Breidablik breaks 
        # these stars will be taken out later down the line anyway when EW -> A(Li)
        # so it doesn't really matter what is saved in here
        if self.mode == 'Gaussian':
            self.fit_gaussian(spectra)
            return None

        if self.metal_poor:
            # if metal poor, no CN, because otherwise it's uncontrained again
            fitter = FitB(self.teff, self.logg, self.feh, self.ew_to_abund, self.min_ew, max_ew=self.max_ew, stdu=self.stdu, rv_lim=self.rv_lim, std_galah=self.std_galah)
            init = self.mp_init(spectra, perc=95)
            # fit 
            res, minchisq = fitter.fit(spectra['wave_norm'], spectra['sob_norm'], spectra['uob_norm'], init=init)
            # turn results into consistent format
            std_li = res[1]
            rv = res[2]
            const = res[3]
            amps = [0,0,0,0,0]
        else:
            fitter = FitBFixed(self.narrow_center[1:], self.broad_fit['std'], self.broad_fit['rv'], self.teff, self.logg, self.feh, self.ew_to_abund, self.min_ew, max_ew=self.max_ew, stdu=self.stdu)
            # initial guess from the amplitudes in the spectrum (a little bit overestimated)
            amps, _, const = pred_amp(spectra['wave_norm'], spectra['sob_norm'], spectra['uob_norm'], centers=self.narrow_center, rv=self.broad_fit['rv'], perc=95)
            init = amp_to_init(amps, self.broad_fit['std'], self.broad_fit['rv'], const) 
            init = [min(max(init[0], self.min_ew), self.max_ew), self.broad_fit['std'], *init[1:-3], const] # reformat
            # calculate ratio
            pred = fitter.model(spectra['wave_norm'], init)
            pred_amps, _, const = pred_amp(spectra['wave_norm'], pred, spectra['uob_norm'], centers=self.narrow_center, rv=self.broad_fit['rv'], perc=95)
            ratio = pred_amps/amps
            # better amps
            amps = amps/ratio
            init = amp_to_init(amps, self.broad_fit['std'], self.broad_fit['rv'], const)
            init = [min(max(init[0], self.min_ew), self.max_ew), self.broad_fit['std'], *init[1:-3], const]
            # fit 
            res, minchisq = fitter.fit(spectra['wave_norm'], spectra['sob_norm'], spectra['uob_norm'], init=init)
            # turn results into consistent format
            std_li = res[1]
            rv = self.broad_fit['rv']
            amps = res[2:-1]
            const = res[-1]
        # save Li fit
        self.li_init_fit = {'amps':[res[0], *amps], 'std':std_li, 'rv':rv, 'const':const, 'minchisq':minchisq}
    
    def fit_gaussian(self, spectra, center=np.array([6707.8139458, 6706.730, 6707.433, 6707.545, 6708.096, 6708.810, 6709.011])):
        '''Fit the Li region of the spectrum, fits ew, std, and rv if metal-poor (less than 3 lines with amplitudes above noise), or else fixes std and rv and only fits ews.
        
        Parameters
        ----------
        spectra : dict
            Dictionary containing spectrum, from read (keys: wave_norm, sob_norm, uob_norm) 
        center : 1darray
            The centers of the lines used in the fitting. First value is Li, not really used in the fitting, but needed for the initial guess. Default values:
            np.array([6707.8139458, 6706.730, 6707.433, 6707.545, 6708.096, 6708.810, 6709.011]))
        '''

        if self.metal_poor:
            # if metal poor, no CN, because otherwise it's uncontrained again
            fitter = FitG(stdl=self.stdl, stdu=self.stdu, rv_lim=self.rv_lim, std_galah=self.std_galah)
            init = self.mp_init(spectra, perc=95)
            # fit
            res, minchisq = fitter.fit(spectra['wave_norm'], spectra['sob_norm'], spectra['uob_norm'], init=init)
            # turn results into consistent format
            std_li = res[1]
            rv = res[2]
            const = res[3]
            amps = [0,0,0,0,0]
        else:
            fitter = FitGFixed(self.narrow_center, self.broad_fit['std'], self.broad_fit['rv'])
            # initial guess from the amplitudes in the spectrum (a little bit overestimated)
            amps, _, const = pred_amp(spectra['wave_norm'], spectra['sob_norm'], spectra['uob_norm'], centers=self.narrow_center, rv=self.broad_fit['rv'], perc=95)
            init = amp_to_init(amps, self.broad_fit['std'], self.broad_fit['rv'], const) 
            # calculate ratio
            pred = fitter.model(spectra['wave_norm'], init)
            pred_amps, _, const = pred_amp(spectra['wave_norm'], pred, spectra['uob_norm'], centers=self.narrow_center, rv=self.broad_fit['rv'], perc=95)
            ratio = pred_amps/amps
            # better amps
            amps = amps/ratio
            init = amp_to_init(amps, self.broad_fit['std'], self.broad_fit['rv'], const)
            # fit 
            res, minchisq = fitter.fit(spectra['wave_norm'], spectra['sob_norm'], spectra['uob_norm'], init=[*init[:-3], init[-1]])
            # turn results into consistent format
            std_li = self.broad_fit['std']
            rv = self.broad_fit['rv']
            amps = res[1:-1]
            const = res[-1]
        # save initial Li fit
        self.li_init_fit = {'amps':[res[0], *amps], 'const':const, 'std':std_li, 'rv':rv, 'minchisq':minchisq}

    def bad_spec(self, spectra, std):
        '''Identify bad spectra.
        '''
        # below 0 or extremely above 1 spectra
        lower, upper = np.percentile(spectra['sob_norm'], [5, 95])
        if lower < 0 or upper > 1.5:
            return True
        # extremely high vsini
        if std > 0.665: # 70 km/s
            return True
        return False    

    def get_err(self, cdelt):
        '''error from cayrel and norris formula'''
        error_factor = np.sqrt(3*np.pi)/(np.pi**(1/4))
        # metal-poor stars use galah std
        if self.broad_fit is None:
            std = self.std_galah
        else:
            std = self.broad_fit['std']
        self.cayrel = error_factor*(1/self.snr)*(std*cdelt)**0.5
        R = 25500
        npix = 5
        self.norris = self.li_center*npix**0.5/(R*self.snr)

    def posterior_setup(self, li_factor=5, blend_factor=5, const_range=0.1):
        '''Set up the fitter, bounds, grid required to sample posterior'''

        if self.metal_poor and self.mode == 'Breidablik':
            fitter = FitB(self.teff, self.logg, self.feh, self.ew_to_abund, self.min_ew, max_ew=self.max_ew, stdu=self.stdu, rv_lim=self.rv_lim, std_galah=self.std_galah)
            opt = [self.li_init_fit['amps'][0], self.li_init_fit['std'], self.li_init_fit['rv'], self.li_init_fit['const']]
            bounds = [(max(opt[0]-self.norris*li_factor, -self.max_ew), min(opt[0]+self.norris*li_factor, self.max_ew)),
                    (5e-4, self.stdue),
                    (-self.rv_lim, self.rv_lim),
                    (opt[-1]-const_range, opt[-1]+const_range)
                    ]
            grid = Grid(bounds[0], bounds[1], teff=self.teff, logg=self.logg, feh=self.feh, ew_to_abund=self.ew_to_abund, min_ew=self.min_ew)
        elif not self.metal_poor and self.mode == 'Breidablik':
            fitter = FitBFixed(self.narrow_center[1:], self.broad_fit['std'], self.broad_fit['rv'], self.teff, self.logg, self.feh, self.ew_to_abund, self.min_ew, max_ew=self.max_ew, stdu=self.stdu)
            opt = [self.li_init_fit['amps'][0], self.li_init_fit['std'], *self.li_init_fit['amps'][1:], self.li_init_fit['const']]
            bounds = [(max(opt[0]-self.norris*li_factor, -self.max_ew), min(opt[0]+self.norris*li_factor, self.max_ew)),
                    (5e-4, self.broad_fit['std']),
                    (max(0, opt[2]-self.norris*blend_factor), opt[2]+self.norris*blend_factor),
                    (max(0, opt[3]-self.norris*blend_factor), opt[3]+self.norris*blend_factor),
                    (max(0, opt[4]-self.norris*blend_factor), opt[4]+self.norris*blend_factor),
                    (max(0, opt[5]-self.norris*blend_factor), opt[5]+self.norris*blend_factor),
                    (max(0, opt[6]-self.norris*blend_factor), opt[6]+self.norris*blend_factor),
                    (max(0, opt[7]-self.norris*blend_factor), opt[7]+self.norris*blend_factor),
                    (opt[-1]-const_range, opt[-1]+const_range)
                    ]
            grid = Grid(bounds[0], bounds[1], teff=self.teff, logg=self.logg, feh=self.feh, ew_to_abund=self.ew_to_abund, min_ew=self.min_ew)
        elif self.metal_poor and self.mode == 'Gaussian':
            fitter = FitG(stdl=self.stdl, stdu=self.stdu, rv_lim=self.rv_lim, std_galah=self.std_galah)
            opt = [self.li_init_fit['amps'][0], self.li_init_fit['std'], self.li_init_fit['rv'], self.li_init_fit['const']]
            bounds = [(opt[0]-self.norris*li_factor, opt[0]+self.norris*li_factor),
                    (self.stdl, self.stdue),
                    (-self.rv_lim, self.rv_lim),
                    (opt[-1]-const_range, opt[-1]+const_range)
                    ]
            grid = None
        elif not self.metal_poor and self.mode == 'Gaussian':
            fitter = FitGFixed(self.narrow_center, self.broad_fit['std'], self.broad_fit['rv'])
            opt = [*self.li_init_fit['amps'], self.li_init_fit['const']]
            bounds = [(opt[0]-self.norris*li_factor, opt[0]+self.norris*li_factor),
                    (max(0, opt[1]-self.norris*blend_factor), opt[1]+self.norris*blend_factor),
                    (max(0, opt[2]-self.norris*blend_factor), opt[2]+self.norris*blend_factor),
                    (max(0, opt[3]-self.norris*blend_factor), opt[3]+self.norris*blend_factor),
                    (max(0, opt[4]-self.norris*blend_factor), opt[4]+self.norris*blend_factor),
                    (max(0, opt[5]-self.norris*blend_factor), opt[5]+self.norris*blend_factor),
                    (max(0, opt[6]-self.norris*blend_factor), opt[6]+self.norris*blend_factor),
                    (opt[-1]-const_range, opt[-1]+const_range)
                    ]
            grid = None

        return fitter, bounds, grid

    def posterior(self, spectra):
        '''run ultranest to get posteriors'''
        # set up variables
        self.get_err(spectra['CDELT1'])

        # spectra is bad so we skip mcmc
        if self.bad_spec(spectra, self.li_init_fit['std']):
            self.sample = None
            self.li_fit = None
            self.time = None
            self.posterior_good = False
            self.edge_ind = None
            return None

        # set up bounds and fitters
        fitter, bounds, grid = self.posterior_setup()

        # run mcmc
        start = time.time()
        un_fitter = UNFitter(spectra['wave_norm'], spectra['sob_norm'], spectra['uob_norm'], fitter, bounds, mode=self.mode, metal_poor=self.metal_poor, e_vbroad=self.e_vbroad, e_rv=self.e_rv, grid=grid)#, run=False)
        end = time.time()
        self.sample = un_fitter.results
        self.time = end - start

        # check if on edge
        _, argmax = self.get_map()
        is_on_edge, due_to_const, _ = self.on_edge(argmax, self.metal_poor, self.mode, bounds, self.max_ew)
        self.edge_ind = None

        if is_on_edge:
            if due_to_const:
                const_range = 0.5
            else:
                const_range = 0.1

            # make new bounds
            fitter, bounds, grid = self.posterior_setup(li_factor=20, blend_factor=20, const_range=const_range)

            # run mcmc
            start = time.time()
            un_fitter = UNFitter(spectra['wave_norm'], spectra['sob_norm'], spectra['uob_norm'], fitter, bounds, mode=self.mode, metal_poor=self.metal_poor, e_vbroad=self.e_vbroad, e_rv=self.e_rv, grid=grid)
            end = time.time()
            self.sample_old = self.sample
            self.sample = un_fitter.results
            self.time = end - start

            # check edge again
            _, argmax = self.get_map()
            is_on_edge, _, edge_ind = self.on_edge(argmax, self.metal_poor, self.mode, bounds, self.max_ew)
            self.edge_ind = edge_ind
        self.posterior_good = not is_on_edge

        # parse results
        self.err = np.percentile(self.sample['samples'][:,0], [50-68/2, 50+68/2])
        MAP, _ = self.get_map()
        if self.metal_poor:
            li_ew, std_li, rv, const = MAP
            amps = [0]*5
        elif self.mode == 'Gaussian':
            li_ew, *amps, const = MAP
            rv = self.broad_fit['rv']
            std_li = self.broad_fit['std']
        elif self.mode == 'Breidablik':
            li_ew, std_li, *amps, const = MAP
            rv = self.broad_fit['rv']

        self.li_fit = {'amps':[li_ew, *amps], 'const':const, 'std':std_li, 'rv':rv}

    def get_map(self, bins=100):
        params = []
        inds = []
        for i in range(self.sample['samples'].shape[1]):
            sample = self.sample['samples'][:,i]
            hist, edges = np.histogram(sample, bins=bins)
            centers = np.mean([edges[:-1], edges[1:]], axis=0)
            best = centers[np.argmax(hist)]
            params.append(best)
            inds.append(np.argmax(hist))
        return np.array(params), np.array(inds)

    def on_edge(self, argmax, metal_poor, mode, bounds, max_ew):
        '''
        Returns
        -------
        edge, cont : bool, bool, int
            If on the edge, if it's due to the continuum placement, which index triggered it.
        '''

        # check cont
        if argmax[-1] < 10 or argmax[-1] > 89:
            return True, True
        # indicies for ew
        if metal_poor:
            inds = [0]
        elif not metal_poor and mode == 'Gaussian':
            inds = list(range(len(argmax)-1))
        elif not metal_poor and mode == 'Breidablik':
            inds = list(range(len(argmax)-1))
            del inds[1]
        # check all ews
        for ind in inds:
            # lower bound, if not hitting 0
            if argmax[ind] < 5 and bounds[ind][0] > 0:
                return True, False, ind
            elif argmax[ind] > 94:
                if ind == 0 and bounds[ind][1] >= max_ew:
                    return False, False, 0
                return True, False, ind
        return False, False, None

    def plot_broad(self, spectra, show=True, path=None, ax=None):
        '''Plot the broad region and the fits. Meant to be a convenience function for quickly checking the fits are working
        
        Parameters
        ----------
        spectra : dict
            Dictionary containing spectrum, from read (keys: wave_norm, sob_norm, uob_norm) 
        show : bool, optional
            Toggle showing the plot, default True.
        path : str, optional
            Path to save fig, if None then it won't save.
        ax : matplotlib.axes, optional
            The axis to plot on, if None, then it will create one to plot on
        '''
        
        if ax is None:
            axes = plt
        else:
            axes = ax

        # observed spec
        axes.errorbar(spectra['wave_norm'], spectra['sob_norm'], yerr=spectra['uob_norm'], color='black', alpha=0.5, label='observed')
        # fit if not metal-poor (no fit if metal-poor)
        if self.broad_fit is not None:
            if ax is None:
                plt.title(f'{self.sid} {self.broad_fit["std"]:.4f} {self.broad_fit["rv"]:.4f} {self.snr:.2f}')
            fitter = FitBroad(center=self.broad_center, stdl=self.stdl, stdu=self.stdu, rv_lim=self.rv_lim)
            fitter.model(spectra['wave_norm'], [*self.broad_fit['amps'], self.broad_fit['std'], self.broad_fit['rv']], plot=True, ax=axes)
        
        if ax is None:
            plt.xlim(6695, 6719)
            plt.xlabel(r'wavelengths ($\AA$)')
            plt.ylabel('normalised flux')
        axes.legend()
        if path is not None:
            plt.savefig(path, bbox_inches='tight')
        if show:
            plt.show()

    def plot_li(self, spectra, mode='posterior', show=True, path=None, ax=None):
        '''Plot the Li region and the fits..
        
        Parameters
        ----------
        spectra : dict
            Dictionary containing spectrum, from read (keys: wave_norm, sob_norm, uob_norm) 
        mode : str, optional
            Plot posterior fits or scipy.minimize fits. Values: posterior, minimize.
        show : bool, optional
            Toggle showing the plot, default True.
        path : str, optional
            Path to save fig, if None then it won't save.
        ax : matplotlib.axes, optional
            The axis to plot on, if None, then it will create one to plot on
        '''
        
        if ax is None:
            axes = plt
        else:
            axes = ax

        if mode == 'posterior':
            fit = self.li_fit
        elif mode == 'minimize':
            fit = self.li_init_fit

        # observation
        axes.errorbar(spectra['wave_norm'], spectra['sob_norm'] * fit['const'], yerr=spectra['uob_norm'], label='observed', color='black', alpha=0.5)
        if ax is None:
            plt.title(f'{fit["amps"][0]:.4f} {fit["amps"][1]:.4f} {fit["std"]:.1f}')
    
        # Breidablik
        if self.mode == 'Breidablik':
            if not self.metal_poor:
                fitter = FitBFixed(center=self.narrow_center[1:], std=self.broad_fit['std'], rv=self.broad_fit['rv'], teff=self.teff, logg=self.logg, feh=self.feh, ew_to_abund=self.ew_to_abund, min_ew=self.min_ew)
                fit_list = [fit['amps'][0], fit['std'], *fit['amps'][1:], fit['const']]
            elif self.metal_poor:
                fitter = FitB(self.teff, self.logg, self.feh, self.ew_to_abund, self.min_ew)
                fit_list = [fit['amps'][0], fit['std'], fit['rv'], fit['const']]
            # error region
            if mode == 'posterior':
                lower = bline(spectra['wave_norm'], self.err[0], fit['std'], fit['rv'], teff=self.teff, logg=self.logg, feh=self.feh, ew_to_abund=self.ew_to_abund, min_ew=self.min_ew) 
                upper = bline(spectra['wave_norm'], self.err[1], fit['std'], fit['rv'], teff=self.teff, logg=self.logg, feh=self.feh, ew_to_abund=self.ew_to_abund, min_ew=self.min_ew) 
        # Gaussian
        elif self.mode == 'Gaussian':
            if not self.metal_poor:
                fitter = FitGFixed(center=self.narrow_center, std=self.broad_fit['std'], rv=self.broad_fit['rv'])
                fit_list = [*fit['amps'], fit['const']]
            elif self.metal_poor:
                fitter = FitG()
                fit_list = [fit['amps'][0], fit['std'], fit['rv'], fit['const']]
            # error region
            if mode == 'posterior':
                lower = gline(spectra['wave_norm'], self.err[0], fit['std'], fit['rv'], center=self.li_center) 
                upper = gline(spectra['wave_norm'], self.err[1], fit['std'], fit['rv'], center=self.li_center) 

        # plot fit
        fitter.model(spectra['wave_norm'], fit_list, plot=True, plot_all=True, ax=axes)
        
        # error shaded region
        if mode == 'posterior':
            axes.fill_between(spectra['wave_norm'], lower, y2=upper, alpha=0.5)

        # show chisq region
        if self.metal_poor:
            std = self.std_galah
        else:
            std = self.broad_fit['std']
        axes.axvline(self.narrow_center[1]*(1+fit['rv']/self.c)-std*2)
        axes.axvline(self.narrow_center[-1]*(1+fit['rv']/self.c)+std*2)
        
        axes.legend()
        if ax is None:
            plt.xlabel(r'wavelengths ($\AA$)')
            plt.ylabel('normalised flux')
            plt.xlim(6706, 6709.5)
        else:
            axes.set_xlim(6706, 6709.5)
        if path is not None:
            plt.savefig(path, bbox_inches='tight')
        if show:
            plt.show()

    def plot_corner(self, show=True, path=None):
        '''Corner plot for sampled posterior
        
        Parameters
        ----------
        path : str, optional
            Path to save fig, if None, then will show fig instead.
        '''

        paramnames = self.sample['paramnames']
        data = np.array(self.sample['weighted_samples']['points'])
        weights = np.array(self.sample['weighted_samples']['weights'])
        cumsumweights = np.cumsum(weights)

        mask = cumsumweights > 1e-4

        fig = corner.corner(data[mask,:], weights=weights[mask],
                      labels=paramnames, show_titles=True, quiet=True, quantiles=[0.5])
        
        # you've got to be kidding me, the version of corner on avatar doesn't have overplot_lines
        def _get_fig_axes(fig, K):
            if not fig.axes:
                return fig.subplots(K, K), True
            try:
                return np.array(fig.axes).reshape((K, K)), False
            except ValueError:
                raise ValueError(
                    (
                        "Provided figure has {0} axes, but data has "
                        "dimensions K={1}"
                    ).format(len(fig.axes), K)
                )
        
        none = [None]*(self.sample['samples'].shape[1] - 1)
        xs = [self.li_init_fit['amps'][0], *none]
        K = len(xs)
        axes, _ = _get_fig_axes(fig, K)
        axes[0,0].axvline(xs[0], alpha=0.5)

        if path is not None:
            plt.savefig(path, bbox_inches='tight')
        if show:
            plt.show()
        
        return fig

    def save(self, filepath):
        '''Save the fitted results in a dictionary.

        Parameters
        ----------
        filepath : str
            Filepath to saved results
        '''
        
        names = ['broad_fit', 'broad_center', # broad 
                'metal_poor',
                'li_init_fit',
                'li_fit', 'narrow_center',
                'mode',
                'cayrel',
                'norris',
                'sample',
                'posterior_good',
                'edge_ind',
                'time']
        dic = {}
        for name in names:
            dic[name] = getattr(self, name)
        np.save(filepath, dic)
   
    def load(self, filepath):
        '''Load the fitted results into the class.

        Parameters
        ----------
        filepath : str
            Filepath to saved results
        '''

        dic = np.load(filepath, allow_pickle=True).item()
        for name, value in dic.items():
            setattr(self, name, value)
        if self.sample is not None:
            self.err = np.percentile(self.sample['samples'][:,0], [50-68/2, 50+68/2])
