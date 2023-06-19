from astro_tools import vac_to_air, air_to_vac
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from breidablik.interpolate.spectra import Spectra
from breidablik.analysis import read, tools
from astro_tools import SpecAnalysis
from scipy.interpolate import CubicSpline

_c = 299792.458 # speed of light in km s^-1 
# optimised from 8s for 100 spectra to 2s - cut and gaussian_broaden using matrix
_spectra = Spectra()
# cut to 6703 - 6712 (a little bit extra for rv shift)
_spectra.cut_models = _spectra.models[136:298]
_wl = vac_to_air(read.get_wavelengths()*10)[136:298]

def line(x, ew, center, std, rv, breidablik=False, teff=None, logg=None, feh=None, rew_to_abund=None, plot=False, ax=None):
    '''Create a spectral line, either gaussian or from breidablik. There are some overlapping values.
    
    Parameters
    ----------
    x : 1darray
        The wavelengths to evaluate the spectral line at
    ew : float
        The EW of the line
    center : float
        The center that the line is at. Parameter ignored if breidablik=True.
    std : float
        The standard deviation of the line. If breidablik=True, this is the amount that the std that goes into the Gaussian convolution.
    rv : float
        The radial velocity. 
    breidablik : bool
        If true, uses breidablik line profiles
    teff : float, optional
        Used in breidablik, teff of star
    logg : float, optional
        Used in breidablik, logg of star 
    feh : float, optional
        Used in breidablik, feh of star 
    rew_to_abund : object
        Converting REW to A(Li), used in Breidablik since the input there is A(Li), but the input to this function is EW
    plot : bool
        Plot the individual lines. 
    ax : matplotlib.axes, optional
        The axis to plot on, if None, then it's the default one. 
    '''

    if breidablik: # breidablik line profile
        lambda0 = 6707.814*(1+rv/299792.458) # line center is shifted
        ali = rew_to_abund(np.log10(ew/lambda0)) # convert EW to REW
        flux = _spectra._predict_flux(teff, logg, feh, [ali])[0] # this won't produce warnings anymore
        # rv shift
        wl = _wl*(1+rv/299792.458)
        y = CubicSpline(wl, flux)(x)
        # gaussian broaden
        spec = SpecAnalysis(x, y)
        _, y = spec._gaussian_broaden(center=6707.814, sigma=std*2.35482*_c/6707.814)
    else: # Gaussian
        y = 1-ew*norm.pdf(x, center*(1+rv/299792.458), std)

    if plot:
        if ax is None: # set axes
            ax = plt
        if breidablik:
            plt.plot(x, y, label='Li')
        else:
            plt.plot(x, y)

    return y


class FitBroad:
    '''Fit based on centers, also fits std and rv'''
    def __init__(self, center, stdl=None, stdu=None, rv_lim=None):
        # surrounding lines:
        # np.array([6696.085, 6698.673, 6703.565, 6705.101, 6710.317, 6711.819, 6713.095, 6713.742, 6717.681])
        # Li line:
        # np.array([6707.8139458]

        self.center = center
        # don't need if using model
        self.stdl = stdl
        self.stdu = stdu
        self.rv_lim = rv_lim

    def fit(self, wl_obs, flux_obs, flux_err, init):
        '''Fit std and rv of observed spectrum.

        wl_obs : np.array
            observed wavelengths
        flux_obs : np.array
            observed flux
        flux_err : np.array
            observed flux error
        init : list
            The initial amplitudes and std, rv. 
        '''
        
        # metal poor star
        if init is None:
            return None, None, None

        bounds = [(0, np.inf) for _ in range(len(init)-2)]
        bounds.append((self.stdl, self.stdu))
        bounds.append((-self.rv_lim, self.rv_lim))
        bounds = np.array(bounds).T
        
        func = lambda x: self.chisq(wl_obs, flux_obs, flux_err, x, bounds)
        res = minimize(func, init, method='Nelder-Mead')
        fit = res.x

        return res.x, res.fun

    def chisq(self, wl_obs, flux_obs, flux_err, params, bounds):
        '''calculate the chisq
        '''
        for p, (l, r) in zip(params, bounds.T):
            if (p < l) or (r < p):
                return np.inf
        return np.sum(np.square((self.model(wl_obs, params) - flux_obs)/flux_err))

    def model(self, wl_obs, params, plot=False, ax=None, plot_all=False):
        '''Gaussian mixture model with std and rv

        wl_obs : np.array
            observed wavelengths
        flux_err : np.array
            observed flux error
        params : np.array
            Parameters to be fitted, amplitudes, std, and rv in this case
        plot : bool
            Plot overall fit
        plot_all : bool
            Plot each gaussian
        '''
        
        *amps, std, offset = params
        y = np.ones(len(wl_obs))
        
        for a, c in zip(amps, self.center):
            y1 = line(wl_obs, a, c, std, offset, plot=plot_all, ax=ax)
            y *= y1
        
        # plot
        if plot:
            if ax is None:
                ax = plt
            ax.plot(wl_obs, y, label='fit')
        
        return y


class FitFixed:
    '''Fit based on centers, fixed std and rv'''
    def __init__(self, center, std, rv):
        self.center = center
        self.std = std
        self.rv = rv

    def fit(self, wl_obs, flux_obs, flux_err, init):
        '''Fit std and rv of observed spectrum.

        wl_obs : np.array
            observed wavelengths
        flux_obs : np.array
            observed flux
        flux_err : np.array
            observed flux error
        init : list
            The initial amplitudes and std, rv. 
        '''
        
        bounds = [(0, np.inf) for _ in range(len(init))]
        bounds = np.array(bounds).T
        
        func = lambda x: self.chisq(wl_obs, flux_obs, flux_err, x, bounds)
        res = minimize(func, init, method='Nelder-Mead')

        return res.x, res.fun

    def chisq(self, wl_obs, flux_obs, flux_err, params, bounds):
        '''calculate the chisq
        '''
        for p, (l, r) in zip(params, bounds.T):
            if (p < l) or (r < p):
                return np.inf
        return np.sum(np.square((self.model(wl_obs, params) - flux_obs)/flux_err))

    def model(self, wl_obs, params, plot=False, ax=None):
        '''Gaussian mixture model with std and rv

        wl_obs : np.array
            observed wavelengths
        flux_err : np.array
            observed flux error
        params : np.array
            Parameters to be fitted, amplitudes
        plot : bool
            Plot overall fit
        plot_all : bool
            Plot each gaussian
        '''

        y = np.ones(len(wl_obs))
       
        for a, c in zip(params, self.center):
            y1 = line(wl_obs, a, c, self.std, self.rv, plot=plot, ax=ax)
            y *= y1
        
        # plot
        if plot:
            if ax is None:
                ax = plt
            ax.plot(wl_obs, y, label='fit')
        
        return y


class FitSat:
    '''Fit based on centers, also fits std rv, breidablik line profile'''
    def __init__(self, center, teff, logg, feh, rew_to_abund, max_ew, min_ew, stdu=None, rv_lim=None):

        self.center = center
        self.teff = teff
        self.logg = logg
        self.feh = feh
        self.rew_to_abund = rew_to_abund
        # don't need if using model
        self.max_ew = max_ew
        self.min_ew = min_ew
        self.stdu = stdu
        self.rv_lim = rv_lim

    def fit(self, wl_obs, flux_obs, flux_err, init):
        '''Fit std and rv of observed spectrum.

        wl_obs : np.array
            observed wavelengths
        flux_obs : np.array
            observed flux
        flux_err : np.array
            observed flux error
        init : list
            The initial amplitudes and std, rv. 
        '''
        
        bounds = [(self.min_ew, self.max_ew), (0, self.stdu), (-self.rv_lim, self.rv_lim)] # -5 spectra is within 1e-5 of 1 for 4000, 1, 0
        bounds = np.array(bounds).T
        
        func = lambda x: self.chisq(wl_obs, flux_obs, flux_err, x, bounds)
        res = minimize(func, init, method='Nelder-Mead')

        return res.x, res.fun

    def chisq(self, wl_obs, flux_obs, flux_err, params, bounds):
        '''calculate the chisq
        '''
        for p, (l, r) in zip(params, bounds.T):
            if (p < l) or (r < p):
                return np.inf
        return np.sum(np.square((self.model(wl_obs, params) - flux_obs)/flux_err))

    def model(self, wl_obs, params, plot=False, ax=None):
        '''Gaussian mixture model with std and rv

        wl_obs : np.array
            observed wavelengths
        flux_err : np.array
            observed flux error
        params : np.array
            Parameters to be fitted, amplitudes, std, and rv in this case
        plot : bool
            Plot overall fit
        plot_all : bool
            Plot each gaussian
        '''
    
        amps, std, offset = params
        y = line(wl_obs, amps, self.center, std, offset, breidablik=True, teff=self.teff, logg=self.logg, feh=self.feh, rew_to_abund=self.rew_to_abund, plot=plot, ax=ax)
        
        return y


class FitSatFixed:
    '''Fit based on centers, fixed std rv, breidablik line profile'''
    def __init__(self, center, std, rv, teff, logg, feh, rew_to_abund, max_ew, min_ew, stdu=None):
        # surrounding lines:
        # np.array([6696.085, 6698.673, 6703.565, 6705.101, 6710.317, 6711.819, 6713.095, 6713.742, 6717.681])
        # Li line:
        # np.array([6707.8139458]

        self.center = center
        self.std = std
        self.rv = rv
        self.teff = teff
        self.logg = logg
        self.feh = feh
        self.rew_to_abund = rew_to_abund
        # don't need if using model
        self.max_ew = max_ew
        self.min_ew = min_ew
        self.stdu = stdu

    def fit(self, wl_obs, flux_obs, flux_err, init):
        '''Fit std and rv of observed spectrum.

        wl_obs : np.array
            observed wavelengths
        flux_obs : np.array
            observed flux
        flux_err : np.array
            observed flux error
        init : list
            The initial amplitudes and std, rv. 
        '''

        bounds = [(self.min_ew, self.max_ew), (0, self.stdu)] # -5 spectra is within 1e-5 of 1 for 4000, 1, 0
        bounds.extend([(0, np.inf) for _ in range(len(init)-2)])
        bounds = np.array(bounds).T

        func = lambda x: self.chisq(wl_obs, flux_obs, flux_err, x, bounds)
        res = minimize(func, init, method='Nelder-Mead')

        return res.x, res.fun

    def chisq(self, wl_obs, flux_obs, flux_err, params, bounds):
        '''calculate the chisq
        '''
        for p, (l, r) in zip(params, bounds.T):
            if (p < l) or (r < p):
                return np.inf
        return np.sum(np.square((self.model(wl_obs, params) - flux_obs)/flux_err))

    def model(self, wl_obs, params, plot=False, ax=None):
        '''Gaussian mixture model with std and rv

        wl_obs : np.array
            observed wavelengths
        flux_err : np.array
            observed flux error
        params : np.array
            Parameters to be fitted, amplitudes, std, and rv in this case
        plot : bool
            Plot overall fit
        plot_all : bool
            Plot each gaussian
        '''

        ali, std_li, *amps = params
        y = line(wl_obs, ali, 6707.814, std_li, self.rv, breidablik=True, teff=self.teff, logg=self.logg, feh=self.feh, rew_to_abund=self.rew_to_abund, plot=plot, ax=ax) 
        
        for a, c in zip(amps, self.center):
            y1 = line(wl_obs, a, c, self.std, self.rv, plot=plot, ax=ax)
            y *= y1
        
        # plot
        if plot:
            if ax is None:
                ax = plt
            ax.plot(wl_obs, y, label='fit')
        
        return y


def pred_amp(wl_obs, flux_obs, flux_err, centers, rv=0):
    '''make a prediction for initial conditions
    '''
    # predict amplitudes
    inds = np.array([np.argmin(np.abs(wl_obs - i*(1+rv/299792.458))) for i in centers])
    amps = (1 - flux_obs[inds])*1.01 # bit bigger because sampling
    amps[amps < 0] = 0 # set negative amp to 0, chisq is inf otherwise
    err = flux_err[inds]
    return amps, err

def check_mp(amps, err):
    '''check if metal poor star'''
    mask = amps > err
    if np.sum(mask) < 3: # 3 lines detection is arbitrary
        return True
    else:
        return False

def cross_correlate(wl, flux, center, tparams, rv):
    '''Calculate the cross correlation between template and obs flux.

    tparams : list
        Parameters of template - Gaussian mixture model
    '''
    
    fit_all = FitBroad(center=center)
    template = fit_all.model(wl, [*tparams, rv])
    cc = np.sum(template*flux)
    return cc

def cc_rv(wl, flux, center, tparams, rv_init, rv_lim):
    # rv 10 km/s is shifting about the line width. 
    rvs = np.linspace(rv_init-10, rv_init+10, 2000) # accurate to 2nd dp
    rvs = rvs[np.abs(rvs)<rv_lim]# filter out values beyond rv_lim
    ccs = [cross_correlate(wl, flux, center, tparams, rv) for rv in rvs]
    return rvs[np.argmax(ccs)]

def filter_spec(spec, sigma=5):
    '''filter weird parts of the spectrum out.'''

    # filter 0 flux error
    mask = spec['sob_norm'] > 0
    # filter negative flux
    mask = mask & (spec['sob_norm'] >= 0)
    # filter sigma too small, if too small change to medium
    medium_sig = np.nanmedian(spec['uob_norm'])
    mask_medium = spec['uob_norm'] < medium_sig/10 # allow 1 order of magnitude
    spec['uob_norm'][mask_medium] = medium_sig
    # filter flux which sigma*error above 1
    #mask = mask & (spec['sob_norm'] < (1 + spec['uob_norm']*sigma))
    # this filter is a terrible idea
    # write
    spec['uob_norm'] = spec['uob_norm'][mask]
    spec['sob_norm'] = spec['sob_norm'][mask]
    spec['wave_norm'] = spec['wave_norm'][mask]
    return spec

def amp_to_init(amps, std, rv):
    '''convert amplitudes to initial conditions'''
    init = list(np.array(amps)*np.sqrt(2*np.pi)*std) # amp to ew
    init.extend([std, rv])
    return np.array(init)

def iter_fit(wl, flux, flux_err, center, stdl, stdu, std_init, rv_lim, Li=False, sobject_id=None):
    # get initial rv
    fitter = FitBroad(center=center, stdl=stdl, stdu=stdu, rv_lim=rv_lim)
    amps, _ = pred_amp(wl, flux, flux_err, center)
    res0 = amp_to_init(amps, std_init, 0)
    init_rv = cc_rv(wl, flux, center, res0[:-1], res0[-1], rv_lim)
    # get initial amp
    amps, err = pred_amp(wl, flux, flux_err, center, rv=init_rv)
    #plt.scatter(center*(1+init_rv/3e5), 1-amps, color='red')
    init = amp_to_init(amps, std_init, init_rv)
    # not just fitting Li, check metal-poor star
    if (not Li) and check_mp(amps, err):
        return None

    # get good initial fit
    res0, _ = fitter.fit(wl, flux, flux_err, init)
    if res0 is None: # metal poor star
        return None
    initial_res = np.copy(res0)
    
    # iterate
    iterations = 1
    while np.abs(init_rv - res0[-1]) > 0.1:
        *tparams, rv = res0
        init_rv = cc_rv(wl, flux, center, tparams, rv, rv_lim)
        init_params = [*tparams, init_rv]
        res0,  _ = fitter.fit(wl, flux, flux_err, init=init_params)
        iterations += 1
        if iterations >= 5:
            res0 = initial_res
            print('iterations over 5', sobject_id)
            break
    
    # check metal-poor again because some fits are bad
    if (not Li) and check_mp(res0[:-2], err):
        return None

    return res0

