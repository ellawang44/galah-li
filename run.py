from fit import FitBroad, FitG, FitGFixed, FitB, FitBFixed, iter_fit, amp_to_init, pred_amp, line, cc_rv, _spectra, _wl
import numpy as np
import matplotlib.pyplot as plt
from breidablik.interpolate.spectra import Spectra
from breidablik.analysis import read, tools
from config import *
from scipy.interpolate import interp1d, CubicSpline
from astro_tools import vac_to_air

class FitSpec:
    '''Fitting 1 spectrum, contains plotting, and save/load. 
    '''

    def __init__(self, std_galah, stdl, stdu, rv_lim, snr, sid, teff, logg, feh):
        '''
        Parameters
        ----------
        std_galah : float
            std from galah, in \AA. Used as the initial std for fitting
        stdl : float
            lower limit on std for gaussians (not used for Breidablik), in \AA. Corresponds to 10 km/s for this project, because instrumental profile. 
        stdu : float
            upper limit on std for both gaussians and Breidablik, in \AA. Corresponds to galah std + some error in terms of R=22000
        rv_lim : float
            The limit on rv, mirrored limit on either side, it is the same limit as stdu, except in km/s.
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
        self.rv_lim = rv_lim
        self.snr = snr
        self.sid = sid
        self.teff = teff
        self.logg = logg
        self.feh = feh
        self.li_center = 6707.8139458 # weighted mean of smith98 Li7 line centers
        self.c = 299792.458 # speed of light in km/s
        # make model to translate between rew and abundance
        # can't run breidablik with nans
        if ~np.isnan(teff) and ~np.isnan(logg) and ~np.isnan(feh):
            self.rew_to_abund = self.gen_rew_to_abund()

    def gen_rew_to_abund(self):
        '''Generate the function converting rew to abundances

        Returns
        -------
        func : function
            function which takes rew and gives the corresponding abundance
        '''

        # calculate corresponding rews
        abunds = list(np.arange(-0.5, 5.05, 0.5)) # extrapolating 1 dex
        rews = np.array([tools.rew(_wl, spec, center=6709.659, upper=100, lower=100) for spec in _spectra._predict_flux(self.teff, self.logg, self.feh, abunds)])
        # remove all abunds and rews below the point that is not monotonically increasing
        inds = np.where(rews[1:] - rews[:-1] < 0)[0]
        if len(inds) > 0:
            ind = inds[-1] + 1
            abunds = abunds[ind:]
            rews = rews[ind:]
            if ind > 2: # there shouldn't be too many points removed, if the >4 abundance extrapolates incorrectly, I want to know
                print(f'cog removed {ind} points, sp {self.teff} {self.logg} {self.feh}')
        # create interlation models
        cs = CubicSpline(rews, abunds) # cog
        grad, inter = np.polyfit(abunds[:2], rews[:2], deg=1) # linear extrapolate
        # set maximum and minimum allowed ews
        self.max_ew = 10**rews[-1]*6707.814 # high abundances/errors might be incorrect due to restriction
        self.min_ew = 10**(-5*grad+inter)*6707.814 # technically don't need this to make the code work, but breidablik does eventually give fully looking profiles when abundance is too low
        return lambda rew: (rew-inter)/grad if rew < rews[0] else cs(rew)

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

    def fit_li(self, spectra, center=np.array([6707.8139458, 6706.730, 6707.433, 6707.545, 6708.096, 6708.961])):
        '''Fit the Li region of the spectrum, fits ew, std, and rv if metal-poor (less than 3 lines with amplitudes above noise), or else fixes std and rv and only fits ews.
        
        Parameters
        ----------
        spectra : dict
            Dictionary containing spectrum, from read (keys: wave_norm, sob_norm, uob_norm) 
        center : 1darray
            The centers of the lines used in the fitting. First value is Li, not really used in the fitting, but needed for the initial guess. Default values:
            np.array([6707.8139458, 6706.730, 6707.433, 6707.545, 6708.096, 6708.961]))
        '''
        
        self.narrow_center = center # save centers used
        
        # if any sp is nan, then don't do fit, because Breidablik breaks 
        # these stars will be taken out later down the line anyway when EW -> A(Li)
        # so it doesn't really matter what is saved in here
        if np.isnan([self.teff, self.logg, self.feh]).any():
            self.fit_gaussian(spectra)
            return None

        #TODO: has a tendency to fit poorly on broad spectra, need to modify initial conditions
        if self.metal_poor:
            # if metal poor, no CN, because otherwise it's uncontrained again
            fitter = FitB(self.teff, self.logg, self.feh, self.rew_to_abund, self.max_ew, self.min_ew, stdu=self.stdu, rv_lim=self.rv_lim, std_galah=self.std_galah)
            # use cross correlated initial rv
            amps, _ = pred_amp(spectra['wave_norm'], spectra['sob_norm'], spectra['uob_norm'], centers=[self.li_center], rv=0)
            init = amp_to_init(amps, self.std_galah, 0)
            init_rv = cc_rv(spectra['wave_norm'], spectra['sob_norm'], [self.li_center], init[:-1], init[-1], self.rv_lim)
            # predict amps again with cross correlated initial rv
            amps, _ = pred_amp(spectra['wave_norm'], spectra['sob_norm'], spectra['uob_norm'], [self.li_center], rv=init_rv)
            init = amp_to_init(amps, self.std_galah, init_rv)
            # fit and turn results into consistent format
            res, minchisq = fitter.fit(spectra['wave_norm'], spectra['sob_norm'], spectra['uob_norm'], init=[*init,1])
            std_li = res[1]
            rv = res[2]
            const = res[3]
            amps = [0,0,0,0,0]
        else:
            fitter = FitBFixed(self.narrow_center[1:], self.broad_fit['std'], self.broad_fit['rv'], self.teff, self.logg, self.feh, self.rew_to_abund, self.max_ew, self.min_ew, stdu=self.stdu)
            # initial guess from the amplitudes in the spectrum (a little bit overestimated)
            amps, _ = pred_amp(spectra['wave_norm'], spectra['sob_norm'], spectra['uob_norm'], centers=self.narrow_center, rv=self.broad_fit['rv'])
            init = amp_to_init(amps, self.broad_fit['std'], self.broad_fit['rv'])[:-2] # remove std and rv, fixed
            init = [min(max(init[0], self.min_ew), self.max_ew), self.broad_fit['std'], *init[1:]] # reformat
            # fit and turn results into consistent format
            res, minchisq = fitter.fit(spectra['wave_norm'], spectra['sob_norm'], spectra['uob_norm'], init=[*init,1])
            std_li = res[1]
            rv = self.broad_fit['rv']
            amps = res[2:-1]
            const = res[-1]
        # save Li fit
        self.li_fit = {'amps':[res[0], *amps], 'std':std_li, 'rv':rv, 'const':const, 'minchisq':minchisq}
        self.mode = 'Breidablik'
    
    def fit_gaussian(self, spectra, center=np.array([6707.8139458, 6706.730, 6707.433, 6707.545, 6708.096, 6708.961])):
        '''Fit the Li region of the spectrum, fits ew, std, and rv if metal-poor (less than 3 lines with amplitudes above noise), or else fixes std and rv and only fits ews.
        
        Parameters
        ----------
        spectra : dict
            Dictionary containing spectrum, from read (keys: wave_norm, sob_norm, uob_norm) 
        center : 1darray
            The centers of the lines used in the fitting. First value is Li, not really used in the fitting, but needed for the initial guess. Default values:
            np.array([6707.8139458, 6706.730, 6707.433, 6707.545, 6708.096, 6708.961]))
        '''

        if self.metal_poor:
            # if metal poor, no CN, because otherwise it's uncontrained again
            fitter = FitG(stdl=self.stdl, stdu=self.stdu, rv_lim=self.rv_lim, std_galah=self.std_galah)
            # use cross correlated initial rv
            amps, _ = pred_amp(spectra['wave_norm'], spectra['sob_norm'], spectra['uob_norm'], centers=[self.li_center], rv=0)
            init = amp_to_init(amps, self.std_galah, 0)
            init_rv = cc_rv(spectra['wave_norm'], spectra['sob_norm'], [self.li_center], init[:-1], init[-1], self.rv_lim)
            # predict amps again with cross correlated initial rv
            amps, _ = pred_amp(spectra['wave_norm'], spectra['sob_norm'], spectra['uob_norm'], [self.li_center], rv=init_rv)
            init = amp_to_init(amps, self.std_galah, init_rv)
            # fit and turn results into consistent format
            res, minchisq = fitter.fit(spectra['wave_norm'], spectra['sob_norm'], spectra['uob_norm'], init=[*init, 1])
            std_li = res[1]
            rv = res[2]
            const = res[3]
            amps = [0,0,0,0,0]
        else:
            fitter = FitGFixed(self.narrow_center, self.broad_fit['std'], self.broad_fit['rv'])
            # initial guess from the amplitudes in the spectrum (a little bit overestimated)
            amps, _ = pred_amp(spectra['wave_norm'], spectra['sob_norm'], spectra['uob_norm'], centers=self.narrow_center, rv=self.broad_fit['rv'])
            init = amp_to_init(amps, self.broad_fit['std'], self.broad_fit['rv'])[:-2] # remove std and rv, fixed
            # fit and turn results into consistent format
            res, minchisq = fitter.fit(spectra['wave_norm'], spectra['sob_norm'], spectra['uob_norm'], init=[*init,1])
            std_li = self.broad_fit['std']
            rv = self.broad_fit['rv']
            amps = res[1:-1]
            const = res[-1]
        # save Li fit
        self.li_fit = {'amps':[res[0], *amps], 'const':const, 'std':std_li, 'rv':rv, 'minchisq':minchisq}
        self.mode = 'Gaussian'

    def get_err(self, cdelt):
        '''error from cayrel formula'''
        #TODO 
        error_factor = np.sqrt(3*np.pi)/(np.pi**(1/4))
        # metal-poor stars use galah std
        if self.broad_fit is None:
            std = self.std_galah
        else:
            std = self.broad_fit['std']
        self.delta_ew = error_factor*(1/self.snr)*(std*cdelt)**0.5

    def plot_broad(self, spectra):
        '''Plot the broad region and the fits. Meant to be a convenience function for quickly checking the fits are working
        
        Parameters
        ----------
        spectra : dict
            Dictionary containing spectrum, from read (keys: wave_norm, sob_norm, uob_norm) 
        '''
        
        # observed spec
        plt.errorbar(spectra['wave_norm'], spectra['sob_norm'], yerr=spectra['uob_norm'], color='black', alpha=0.5, label='observed')
        # fit if not metal-poor (no fit if metal-poor)
        if self.broad_fit is not None:
            plt.title(f'{self.sid} {self.broad_fit["std"]:.4f} {self.broad_fit["rv"]:.4f} {self.snr:.2f}')
            fitter = FitBroad(center=self.broad_center, stdl=self.stdl, stdu=self.stdu, rv_lim=self.rv_lim)
            fitter.model(spectra['wave_norm'], [*self.broad_fit['amps'], self.broad_fit['std'], self.broad_fit['rv']], plot=True)
        
        plt.xlim(6695, 6719)
        plt.xlabel(r'wavelengths ($\AA$)')
        plt.ylabel('normalised flux')
        plt.legend()
        plt.show()

    def plot_li(self, spectra):
        '''Plot the Li region and the fits. Meant to be a convenience function for quickly checking the fits are working.
        
        Parameters
        ----------
        spectra : dict
            Dictionary containing spectrum, from read (keys: wave_norm, sob_norm, uob_norm) 
        '''
       
        # observation
        plt.errorbar(spectra['wave_norm'], spectra['sob_norm'] * self.li_fit['const'], yerr=spectra['uob_norm'], label='observed', color='black', alpha=0.5)
        plt.title(f'{self.li_fit["amps"][0]:.4f} {self.li_fit["amps"][1]:.4f} {self.li_fit["std"]:.1f} {self.delta_ew:.4f}')
        
        # metal-poor stars
        if self.metal_poor: 
            self.broad_fit = {'std':self.li_fit['std'], 'rv':self.li_fit['rv']}
        
        # Breidablik
        if self.mode == 'Breidablik':
            fitter = FitBFixed(center=self.narrow_center[1:], std=self.broad_fit['std'], rv=self.broad_fit['rv'], teff=self.teff, logg=self.logg, feh=self.feh, rew_to_abund=self.rew_to_abund, max_ew=self.max_ew, min_ew=self.min_ew)
            fitter.model(spectra['wave_norm'], [self.li_fit['amps'][0], self.li_fit['std'], *self.li_fit['amps'][1:], self.li_fit['const']], plot=True, plot_all=True)
        # Gaussian
        elif self.mode == 'Gaussian':
            fitter = FitGFixed(center=self.narrow_center, std=self.li_fit['std'], rv=self.li_fit['rv'])
            fitter.model(spectra['wave_norm'], [*self.li_fit['amps'], self.li_fit['const']], plot=True, plot_all=True)
        
        #TODO: plot errors
        # show chisq region
        if self.metal_poor:
            std = self.std_galah
        else:
            std = self.broad_fit['std']
        plt.axvline(self.narrow_center[1]*(1+self.broad_fit['rv']/self.c)-std*2)
        plt.axvline(self.narrow_center[-1]*(1+self.broad_fit['rv']/self.c)+std*2)
        
        plt.legend()
        plt.xlabel(r'wavelengths ($\AA$)')
        plt.ylabel('normalised flux')
        plt.xlim(6706, 6709.5)
        plt.show()

    def save(self, filepath):
        '''Save the fitted results in a dictionary.

        Parameters
        ----------
        filepath : str
            Filepath to saved results
        '''

        dic = {'broad_fit':self.broad_fit, 'broad_center':self.broad_center, # broad region results
                'metal_poor':self.metal_poor,
                'li_fit':self.li_fit, 'narrow_center':self.narrow_center, # li region results
                'mode':self.mode,
                'delta_ew':self.delta_ew} 
        np.save(filepath, dic)

    def load(self, filepath):
        '''Load the fitted results into the class.

        Parameters
        ----------
        filepath : str
            Filepath to saved results
        '''

        dic = np.load(filepath, allow_pickle=True).item()
        # broad region results
        self.broad_fit = dic['broad_fit']
        self.broad_center = dic['broad_center']
        # properties
        self.metal_poor = dic['metal_poor']
        # li region results
        self.li_fit = dic['li_fit']
        self.narrow_center = dic['narrow_center']
        self.mode = dic['mode']
        self.delta_ew = dic['delta_ew']

