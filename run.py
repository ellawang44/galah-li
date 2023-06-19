from fit import FitBroad, FitFixed, FitSat, FitSatFixed, iter_fit, amp_to_init, pred_amp, line
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from breidablik.interpolate.spectra import Spectra
from breidablik.analysis import read, tools
from config import *
from scipy.interpolate import interp1d, CubicSpline
from astro_tools import vac_to_air

# optimised from 8s for 100 spectra to 2s - cut and gaussian_broaden using matrix
_spectra = Spectra()
# cut to 6703 - 6712 (a little bit extra for rv shift)
_spectra.cut_models = _spectra.models[136:298]
_wl = vac_to_air(read.get_wavelengths()[136:298]*10)/10

def read_breidablik():
    '''Need a way of predicting the initial abundance given the amplitude and teff. This is for initial values when saturated.
    This is used to create the data file that will be read in to make the interpolation model.'''
    breidablik = np.load('breidablik/3D.npy', allow_pickle=True).item()
    # get useful data, teff (rounded to nearest 500), amp (min), abund 
    data = []
    for key in breidablik.keys():
        teff = round(key[0]/500)*500
        for abund in breidablik[key].keys():
            amp = np.min(breidablik[key][abund]['flux'])
            data.append([teff, amp, abund])
    data = np.array(data)
    
    # group this data into teff dicts
    abunds = np.arange(-0.5, 4.1, 0.5)
    dic = {}
    teffs = set(data[:,0])
    for teff in teffs:
        mask = data[:,0] == teff
        xs = []
        for abund in abunds:
            mask2 = data[:,2][mask] == abund
            x = np.mean(data[:,1][mask][mask2])
            xs.append(x)
        dic[teff] = xs
    # save
    np.save(f'{info_directory}/amp_to_ali.npy', dic, allow_pickle=True)
    
def amp_to_ali(amp, teff):
    '''Get the A(Li) given the amplitude and teff'''
    abunds = np.arange(-0.5, 4.1, 0.5)
    # read in dic
    if not os.path.exists(f'{info_directory}/amp_to_ali.npy'):
        read_breidablik()
    dic = np.load(f'{info_directory}/amp_to_ali.npy', allow_pickle=True).item()
    # round teff, if teff too large or small snap to closest value
    teff = round(teff/500)*500
    if teff < 4000:
        teff = 4000
    if teff > 7000:
        teff = 7000
    # use linear interpolation, extrapolation simply gives back max/min abund
    amps = dic[teff]
    pred_abund = float(interp1d(amps, abunds, fill_value='extrapolate')(amp)) # float to get rid of the funny array, doesn't make a difference tho
    if pred_abund < -0.5:
        return -0.5
    elif pred_abund > 4:
        return 4
    else:
        return pred_abund


class FitSpec:
    '''Object for fitting 1 spectrum. Fits broad region, then narrow region. Deals with metal-poor and saturation. Handles plotting and saving. 
    '''

    def __init__(self, std_galah, stdl, stdu, rv_lim, snr, sid, teff, logg, feh):
        self.std_galah = std_galah
        self.stdl = stdl
        self.stdu = stdu
        self.rv_lim = rv_lim
        self.snr = snr
        self.sid = sid
        self.teff = teff
        self.logg = logg
        self.feh = feh
        self.li_center = 6707.8139458
        self.c = 299792.458
        # can't run breidablik with nans, shouldn't need this if nan anyway, coz we skip
        if ~np.isnan(teff) and ~np.isnan(logg) and ~np.isnan(feh):
            self.interp = self.gen_interp()

    def gen_interp(self):
        # calculate corresponding rews
        abunds = list(np.arange(-0.5, 5.05, 0.5))
        #spectra = Spectra()
        rews = [tools.rew(_wl, spec) for spec in _spectra._predict_flux(self.teff, self.logg, self.feh, abunds)]
        # make linear interpolation and extrapolate using ali = -0.5, 0
        #interp = interp1d(rews, abunds, fill_value='extrapolate')
        try:
            interp = CubicSpline(rews, abunds)
        except ValueError:
            rews = rews[1:]
            abunds = abunds[1:]
            interp = CubicSpline(rews, abunds)
        cutoff_rew = rews[0]
        #ind = np.argmin(np.abs(abunds)) + 1
        ind=1
        grad, inter = np.polyfit(abunds[:ind+1], rews[:ind+1], deg=1)
        #interp1d([rews[0], rews[ind]], [abunds[0], abunds[ind]], fill_value='extrapolate')
        def func(rew):
            if rew < cutoff_rew:
                return (rew-inter)/grad
            else:
                return interp(rew)
        self.max_ew = 10**rews[-1]*6707.814
        self.min_ew = 10**(-5*grad+inter)*6707.814
        return func#, CubicSpline(rews, abunds) #cubicspline is causing too many issues with increasing x value 

    def fit_broad(self, spectra, center=np.array([6696.085, 6698.673, 6703.565, 6705.101, 6710.317, 6711.819, 6713.095, 6713.742, 6717.681])):
        res = iter_fit(spectra['wave_norm'], spectra['sob_norm'], spectra['uob_norm'], center=center, stdl=self.stdl, stdu=self.stdu, std_init=self.std_galah, rv_lim=self.rv_lim, sobject_id=self.sid)
        # metal poor star
        if res is None:
            self.metal_poor = True
            self.broad_fit = None
        else:
            self.metal_poor = False
            self.broad_fit = {'amps':res[:-2], 'std':res[-2], 'rv':res[-1]}
        self.broad_center = center

    def fit_li(self, spectra, center=np.array([6707.8139458, 6706.730, 6707.433, 6707.545, 6708.096, 6708.961])):
        self.narrow_center = center
        # deal with metal poor first
        if self.metal_poor:
            fit_all = FitBroad(center=np.array([self.li_center]), stdl=self.stdl, stdu=self.stdu, rv_lim=self.rv_lim)
            res = iter_fit(spectra['wave_norm'], spectra['sob_norm'], spectra['uob_norm'], center=np.array([self.li_center]), stdl=self.stdl, stdu=self.stdu, std_init=self.std_galah, rv_lim=self.rv_lim, Li=True, sobject_id=self.sid)
            minchisq = np.sum(np.square((fit_all.model(spectra['wave_norm'], res) - spectra['sob_norm'])/spectra['uob_norm']))
            amps = [res[0], 0, 0, 0, 0, 0]
            std = res[1]
            rv = res[2]
        else:
            # I don't think the Si line adds much value to this
            fitter = FitFixed(center=self.narrow_center, std=self.broad_fit['std'], rv=self.broad_fit['rv'])
            amps, _ = pred_amp(spectra['wave_norm'], spectra['sob_norm'], spectra['uob_norm'], centers=self.narrow_center, rv=self.broad_fit['rv'])
            init = amp_to_init(amps, std=self.broad_fit['std'], rv=self.broad_fit['rv'])
            res, minchisq = fitter.fit(spectra['wave_norm'], spectra['sob_norm'], spectra['uob_norm'], init=init[:-2])
            std = self.broad_fit['std']
            rv = self.broad_fit['rv']
            amps = res
        # save Li fit
        self.li_fit = {'amps':amps, 'std':std, 'rv':rv, 'minchisq':minchisq}

    def fit_sat(self, spectra):
        self.sat_fit = {}

        self.sat = True
        # saturation
        #lambda0 = self.li_center*(1+self.li_fit['rv']/self.c)
        #if np.log10((self.li_fit['amps'][0]+0.12)/lambda0) > -4.6: # eyeballed value from breidablik saturation curves of growths
        #    self.sat = True
        #else:
        #    self.sat = False

        # if any sp is nan, then just don't do sat, bc Breidablik breaks
        # these stars will be taken out later down the line anyway when EW -> A(Li)
        # so it doesn't really matter what is saved in here
        if np.isnan([self.teff, self.logg, self.feh]).any():
            self.sat = False

        # fit Breidablik
        if self.sat:
            if self.metal_poor:
                # if metal poor, no CN, because otherwise it's uncontrained again
                fitter = FitSat(self.li_center, self.teff, self.logg, self.feh, self.interp, self.max_ew, self.min_ew, stdu=self.stdu, rv_lim=self.rv_lim)
                amps, _ = pred_amp(spectra['wave_norm'], spectra['sob_norm'], spectra['uob_norm'], centers=[self.li_center], rv=0)
                #init = [amp_to_ali(1-amps[0], self.teff), self.std_galah, 0]
                init = [max(amps[0]*self.std_galah*np.sqrt(2*np.pi), self.min_ew), self.std_galah, 0]
                res, minchisq = fitter.fit(spectra['wave_norm'], spectra['sob_norm'], spectra['uob_norm'], init=init)
                std_li = res[1]
                rv = res[2]
                amps = [0,0,0,0,0]
            else:
                fitter = FitSatFixed(self.narrow_center[1:], self.broad_fit['std'], self.broad_fit['rv'], self.teff, self.logg, self.feh, self.interp, self.max_ew, self.min_ew, stdu=self.stdu)
                # pred all amps, or else underestimated too hard
                # remove li prediction and tag on other proper things
                amps, _ = pred_amp(spectra['wave_norm'], spectra['sob_norm'], spectra['uob_norm'], centers=self.narrow_center, rv=self.broad_fit['rv'])
                #init = [amp_to_ali(1-amps[0], self.teff), self.li_fit['std'], *amps[1:]]
                ew = amps[0]*self.li_fit['std']*np.sqrt(2*np.pi)
                init = [min(max(ew, self.min_ew), self.max_ew), self.li_fit['std'], *amps[1:]]
                res, minchisq = fitter.fit(spectra['wave_norm'], spectra['sob_norm'], spectra['uob_norm'], init=init)
                std_li = res[1]
                rv = self.broad_fit['rv']
                amps = res[2:]
            # write results
            self.sat_fit = {'amps':[res[0], *amps], 'std':std_li, 'rv':rv, 'minchisq':minchisq}

            # determine if still saturated
            rew = np.log10(amps[0]/self.li_center)
            if rew < -4.6:
                self.sat = False

    def get_err(self, cdelt):
        '''error from cayrel formula'''
        error_factor = np.sqrt(3*np.pi)/(np.pi**(1/4))
        # metal-poor stars use galah std
        if self.broad_fit is None:
            std = self.std_galah
        else:
            std = self.broad_fit['std']
        self.delta_ew = error_factor*(1/self.snr)*(std*cdelt)**0.5

    def plot_broad(self, spectra, center=None):
        fit_all = FitBroad(center=center or self.broad_center, stdl=self.stdl, stdu=self.stdu, rv_lim=self.rv_lim)
        if self.broad_fit is not None:
            plt.title(f'{self.sid} {self.broad_fit["std"]:.4f} {self.broad_fit["rv"]:.4f} {self.snr:.2f}')
        plt.errorbar(spectra['wave_norm'], spectra['sob_norm'], yerr=spectra['uob_norm'], color='black', alpha=0.5, label='observed')
        if self.broad_fit is not None:
            fit_all.model(spectra['wave_norm'], [*self.broad_fit['amps'], self.broad_fit['std'], self.broad_fit['rv']], plot=True)
        plt.xlim(6695, 6719)
        plt.xlabel(r'wavelengths ($\AA$)')
        plt.ylabel('normalised flux')
        plt.legend()
        plt.show()

    def plot_li(self, spectra):
        # breidablik doesn't handle nans
        plot_sat = ~np.isnan(self.teff) and ~np.isnan(self.logg) and ~np.isnan(self.feh)

        # if nans, then no sat fit, simply plot li fit
        if plot_sat:
            res = self.sat_fit
        else:
            res = self.li_fit
        
        # metal poor std
        if not self.metal_poor:
            std = self.broad_fit['std']
        else:
            std = res['std']
        
        # set up plot
        plt.errorbar(spectra['wave_norm'], spectra['sob_norm'], yerr=spectra['uob_norm'], label='observed', color='black', alpha=0.5)
        plt.title(f'{res["amps"][0]:.4f} {res["amps"][1]:.4f} {res["std"]:.1f} {self.delta_ew:.4f}')
        # set up errors
        err = None #TODO
        # different plots
        if plot_sat:
            fitter = FitSatFixed(center=self.narrow_center[1:], std=std, rv=res['rv'], teff=self.teff, logg=self.logg, feh=self.feh, rew_to_abund=self.interp, max_ew=self.max_ew, min_ew=self.min_ew)
            fitter.model(spectra['wave_norm'], [res['amps'][0], res['std'], *res['amps'][1:]], plot=True)
            #upper = res['amps'][0] + err
            #lower = res['amps'][0] - err
            #if lower < self.min_ew: # reflect lower error
            #    lower = abs(lower - self.min_ew) + self.min_ew
            #    if lower > self.max_ew:
            #        print('lower reflected is larger than the max ew! Setting to min ew')
            #        lower = self.min_ew
            #    l = line(spectra['wave_norm'], lower, self.li_center, res['std'], res['rv'], breidablik=True, teff=self.teff, logg=self.logg, feh=self.feh, rew_to_abund=self.interp)
            #    l = -(l - 1)+1
            #else:
            #    l = line(spectra['wave_norm'], lower, self.li_center, res['std'], res['rv'], breidablik=True, teff=self.teff, logg=self.logg, feh=self.feh, rew_to_abund=self.interp)
            #if upper > self.max_ew:
            #    print('upper is higher than max ew! upper', upper, 'max_ew', self.max_ew)
            #    print('setting to max ew')
            #    upper = min(upper, self.max_ew)
            #u = line(spectra['wave_norm'], upper, self.li_center, res['std'], res['rv'], breidablik=True, teff=self.teff, logg=self.logg, feh=self.feh, rew_to_abund=self.interp)
            #plt.fill_between(spectra['wave_norm'], u, l, facecolor='C0', alpha=0.5)
        else:
            fitter = FitFixed(center=self.narrow_center, std=res['std'], rv=res['rv'])
            fitter.model(spectra['wave_norm'], res['amps'], plot=True)
            l = (self.li_fit['amps'][0]-err)*norm.pdf(spectra['wave_norm'], self.li_center*(1+self.li_fit['rv']/self.c), self.li_fit['std'])
            u = (self.li_fit['amps'][0]+err)*norm.pdf(spectra['wave_norm'], self.li_center*(1+self.li_fit['rv']/self.c), self.li_fit['std'])
            plt.fill_between(spectra['wave_norm'], 1-u, 1-l, facecolor='C0', alpha=0.5)

        plt.legend()
        plt.xlabel(r'wavelengths ($\AA$)')
        plt.ylabel('normalised flux')
        plt.xlim(6706, 6709.5)
        plt.show()

    def save(self, filepath):
        dic = {'halpha_fit':None,#self.halpha_fit, # H alpha region results
                'broad_fit':self.broad_fit, 'broad_center':self.broad_center, # broad region results
                'metal_poor':self.metal_poor,
                'sat':self.sat,
                'li_fit':self.li_fit, 'narrow_center':self.narrow_center, # li region results
                'sat_fit':self.sat_fit, # saturated fits
                'delta_ew':self.delta_ew} 
        np.save(filepath, dic)

    def load(self, filepath):
        dic = np.load(filepath, allow_pickle=True).item()
        # H alpha region results
        self.halpha_fit = dic['halpha_fit']
        # broad region results
        self.broad_fit = dic['broad_fit']
        self.broad_center = dic['broad_center']
        # properties
        self.metal_poor = dic['metal_poor']
        self.sat = dic['sat']
        # li region results
        self.li_fit = dic['li_fit']
        self.narrow_center = dic['narrow_center']
        self.sat_fit = dic['sat_fit']
        self.delta_ew = dic['delta_ew']

