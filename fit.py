from astro_tools import vac_to_air
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from breidablik.interpolate.spectra import Spectra
from breidablik.analysis import read, tools
from astro_tools import SpecAnalysis
from scipy.interpolate import CubicSpline

_c = 299792.458 # speed of light in km s^-1 
# optimised from 8s for 100 spectra to 2s - cut mainly, gaussian broadening versions don't make too much of a difference
_spectra = Spectra()
# cut to 6703 - 6712 (a little bit extra for rv shift)
_spectra.cut_models = _spectra.models[136:298]
_wl = vac_to_air(read.get_wavelengths()*10)[136:298]

def line(x, ew, std, rv, center=None, breidablik=False, teff=None, logg=None, feh=None, rew_to_abund=None, plot=False, ax=None):
    '''Create a spectral line, either gaussian or from breidablik. There are some overlapping values.
    
    Parameters
    ----------
    x : 1darray
        The wavelengths to evaluate the spectral line at
    ew : float
        The EW of the line
    std : float
        The standard deviation of the line. If breidablik=True, this is the amount that the std that goes into the Gaussian convolution.
    rv : float
        The radial velocity. 
    center : float
        The center that the line is at. Parameter ignored if breidablik=True.
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

    Returns
    -------
    y : 1darray
        The spectral line, flux, at the input x wavelengths.
    '''

    lambda0 = 6707.814*(1+rv/299792.458) # line center is shifted
    if breidablik: # breidablik line profile
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
        plt.axvline(lambda0, linestyle='--')

    return y

def chisq(wl_obs, flux_obs, flux_err, model, params, bounds, wl_left=None, wl_right=None):
    '''Calculate the chisq with bounds. If value is out of bounds, then chisq is inf. Note parameter and bounds need to have same ordering. 
    wl_left and wl_right need to be both given, or else ignored, it is the region in which to compute the chisq value. Needed for continuum normalisation constant, set to the extreme narrow region centers +- std, including metal-poor stars for consistency. 
    
    Parameters
    ----------
    wl_obs : 1darray
        observed wavelengths
    flux_obs : 1darray
        observed flux
    flux_err : 1darray
        observed flux error
    model : object
        Model to evaluate 
    params : 1darray
        Parameters to the model
    bounds : 1darray
        Bounds for the parameters
    wl_left : float, optional
        Left wl bound to compute chisq over. 
    wl_right : float, optional
        Right wl bound to compute chisq over. 

    Returns
    -------
    chisq : float
        The chisq value 
    '''
    
    for p, (l, r) in zip(params, bounds):
        if (p < l) or (r < p):
            return np.inf
    if (wl_left is not None) and (wl_right is not None):
        mask = (wl_left <= wl_obs) & (wl_obs <= wl_right)
        wl_obs = wl_obs[mask]
        flux_obs = flux_obs[mask]
        flux_err = flux_err[mask]

    return np.sum(np.square((model(wl_obs, params) - flux_obs)/flux_err))


class FitG:
    '''Fits std and rv simultaneously and a EW for each center given. 
    For Li only when sp is nan.
    '''

    def __init__(self, stdl=None, stdu=None, rv_lim=None, std_galah=None):
        '''Optional parameters are not needed if only using the model and not fitting. 
        
        Parameters
        ----------
        stdl : float, optional
            The lower limit on std, this is 0.09 \AA for this project.
        stdu : float, optional
            The upper limit on std in \AA, this is based on the broadening for GALAH (roughly R=22000)
        rv_lim : float, optional
            The limit on rv, mirrored limit on either side, it is the same limit as stdu, except in km/s. 
        std_galah : float, optional
            Used for the chisq region. rotational broadening + instrumental broadening (in quadrature). Units: \AA  
        '''

        # don't need if using model
        self.stdl = stdl
        self.stdu = stdu
        self.rv_lim = rv_lim
        self.std_galah = std_galah

    def fit(self, wl_obs, flux_obs, flux_err, init):
        '''Fit std and rv of observed spectrum.
        
        Parameters
        ----------
        wl_obs : 1darray
            observed wavelengths
        flux_obs : 1darray
            observed flux
        flux_err : 1darray
            observed flux error
        init : list
            The initial EW and std, rv; in that order. 

        Returns
        -------
        fit, minchisq : 1darray, float
            Fitted parameters: *EW, std, rv, const; minimum chisq value at best fit.
        '''
        
        # metal poor star
        if init is None:
            return None, None

        # construct bounds
        bounds = [(0, np.inf) for _ in range(len(init)-2)] # positive finite EW
        bounds.append((self.stdl, self.stdu)) # given in init
        bounds.append((-self.rv_lim, self.rv_lim)) # given in init
        bounds.append((0.5, 1.5)) # continuum normalisation constant
        
        # fit
        func = lambda x: chisq(wl_obs, flux_obs, flux_err, self.model, x, bounds, wl_left=6706.730*(1+x[-2]/_c)-self.std_galah*2, wl_right=6708.961*(1+x[-2]/_c)+self.std_galah*2)
        res = minimize(func, init, method='Nelder-Mead')
        fit = res.x

        return res.x, res.fun

    def model(self, wl_obs, params, plot=False, ax=None, plot_all=False):
        '''Multiplying Gaussians together with a common std and rv. 

        wl_obs : np.array
            observed wavelengths
        params : np.array
            Parameters to the model: *EWs, std, rv
        plot : bool
            If True, turns on plotting.
        ax : matplotlib.axes, optional
            The axis to plot on, if None, then it's the default one. 
        plot_all : bool
            If True, plot each gaussian. Or else plot only final model.

        Returns
        -------
        y : 1darray
            The model evaluated at given parameters. All gaussians multiplied together.
        '''
        
        *ews, std, offset, const = params

        y = line(wl_obs, a, std, offset, center=6707.814, plot=plot_all, ax=ax)
        
        # plot
        if plot:
            if ax is None:
                ax = plt
            ax.plot(wl_obs, y, label='fit')
        
        y /= const
       
        return y


class FitGFixed:
    '''Fits EW for each center given. std and rv are fixed from broad region.
    For when Breidablik fails with nan sp.
    '''

    def __init__(self, center, std, rv):
        '''Optional parameters are not needed if only using the model and not fitting. 
        
        Parameters
        ----------
        center : 1darray
            The line centers to be fitted. This should be np.array([6707.814, 6706.730, 6707.433, 6707.545, 6708.096, 6708.961])
        std : float
            The std found from the broad region.
        rv : float
            The rv found from the broad region.
        '''

        self.center = center
        self.std = std
        self.rv = rv

    def fit(self, wl_obs, flux_obs, flux_err, init):
        '''Fit std and rv of observed spectrum.
        
        Parameters
        ----------
        wl_obs : 1darray
            observed wavelengths
        flux_obs : 1darray
            observed flux
        flux_err : 1darray
            observed flux error
        init : list
            The initial EW; in the order of centers. 

        Returns
        -------
        fit, minchisq : 1darray, float
            Fitted parameters: *EW, const; minimum chisq value at best fit.
        '''

        # construct bounds
        bounds = [(0, np.inf) for _ in range(len(init))] # positive finite EW
        bounds.append((0.5, 1.5)) # continuum normalisation constant
        
        # fit
        func = lambda x: chisq(wl_obs, flux_obs, flux_err, self.model, x, bounds, wl_left=6706.730*(1+self.rv/_c)-self.std*2, wl_right=6708.961*(1+self.rv/_c)+self.std*2)
        res = minimize(func, init, method='Nelder-Mead')
        fit = res.x

        return res.x, res.fun

    def model(self, wl_obs, params, plot=False, ax=None, plot_all=False):
        '''Multiplying Gaussians together with a common std and rv. 

        wl_obs : np.array
            observed wavelengths
        params : np.array
            Parameters to the model: *EWs, std, rv
        plot : bool
            If True, turns on plotting.
        ax : matplotlib.axes, optional
            The axis to plot on, if None, then it's the default one. 
        plot_all : bool
            If True, plot each gaussian. Or else plot only final model.

        Returns
        -------
        y : 1darray
            The model evaluated at given parameters. All gaussians multiplied together.
        '''
        
        *ews, const = params
        y = np.ones(len(wl_obs))
        
        for a, c in zip(ews, self.center):
            y1 = line(wl_obs, a, self.std, self.rv, center=c, plot=plot_all, ax=ax)
            y *= y1
       
        # plot
        if plot:
            if ax is None:
                ax = plt
            ax.plot(wl_obs, y, label='fit')
        
        y /= const

        return y


class FitB:
    '''Fits Li EW, std, and rv simultaneously. 
    For metal-poor stars
    '''

    def __init__(self, teff, logg, feh, rew_to_abund, max_ew, min_ew, stdu=None, rv_lim=None, std_galah=None):
        '''Optional parameters are not needed if only using the model and not fitting.
        
        Parameters
        ----------
        teff : float
            Used in breidablik, teff of star
        logg : float
            Used in breidablik, logg of star 
        feh : float
            Used in breidablik, feh of star 
        rew_to_abund : object
            Converting REW to A(Li), used in Breidablik since the input there is A(Li), but the input to this function is EW
        max_ew : float, optional
            The maximum ew that the line can have, limit based on rew_to_abund
        min_ew : float, optional 
            The minimum ew that the line can have, limit based on rew_to_abund
        stdu : float, optional
            The upper limit on std in \AA, this is based on the broadening for GALAH (roughly R=22000)
        rv_lim : float, optional
            The limit on rv, mirrored limit on either side, it is the same limit as stdu, except in km/s. 
        std_galah : float, optional
            Used for the chisq region. rotational broadening + instrumental broadening (in quadrature). Units: \AA
        '''

        self.teff = teff
        self.logg = logg
        self.feh = feh
        self.rew_to_abund = rew_to_abund
        # don't need if using model
        self.max_ew = max_ew
        self.min_ew = min_ew
        self.stdu = stdu
        self.rv_lim = rv_lim
        self.std_galah = std_galah

    def fit(self, wl_obs, flux_obs, flux_err, init):
        '''Fit Li EW, Li std, and rv of observed spectrum.

        Parameters
        ----------
        wl_obs : np.array
            observed wavelengths
        flux_obs : np.array
            observed flux
        flux_err : np.array
            observed flux error
        init : list
            The initial EW, std, and rv; in that order. 
        
        Returns
        -------
        fit, minchisq : 1darray, float
            Fitted parameters: *EW, std, rv, const; minimum chisq value at best fit.
        '''
        
        # construct bounds
        bounds = [(self.min_ew, self.max_ew), # based on rew_to_abund
                (5e-4, self.stdu), # lower limit is sigma in \AA, corresponds to 0.05 FWHM in km/s 
                (-self.rv_lim, self.rv_lim), # based on stdu, except in km/s
                (0.5, 1.5)] # continuum normalisation constant

        func = lambda x: chisq(wl_obs, flux_obs, flux_err, self.model, x, bounds, wl_left=6706.730*(1+x[2]/_c)-self.std_galah*2, wl_right=6708.961*(1+x[2]/_c)+self.std_galah*2)
        res = minimize(func, init, method='Nelder-Mead')

        return res.x, res.fun

    def model(self, wl_obs, params, plot=False, ax=None):
        '''Breidablike ilne profile, with Gaussian broadening and rv shift.

        wl_obs : np.array
            observed wavelengths
        params : np.array
            Parameters to be fitted, Li EW, Li std, and rv.
        plot : bool
            If True, turns on plotting.
        ax : matplotlib.axes, optional
            The axis to plot on, if None, then it's the default one. 
        
        Returns
        -------
        y : 1darray
            The model evaluated at given parameters. Breidablik line profile.
        '''
    
        ews, std, offset, const = params
        y = line(wl_obs, ews, std, offset, breidablik=True, teff=self.teff, logg=self.logg, feh=self.feh, rew_to_abund=self.rew_to_abund, plot=plot, ax=ax)
        
        y /= const

        return y


class FitBFixed:
    '''Fits Li EW, Li std, other ews simltaneously based on the centers given. std and rv are fixed from broad region.
    For non-metal-poor stars
    '''
    
    def __init__(self, center, std, rv, teff, logg, feh, rew_to_abund, max_ew, min_ew, stdu=None):
        '''Optional parameters are not needed if only using the model and not fitting.
        
        Parameters
        ----------
        center : float
            The center that the blended lines (no Li) is at. The Li line center is already given by Breidablik. Input for this project should be np.array([6706.730, 6707.433, 6707.545, 6708.096, 6708.961])
        std : float
            The std found from the broad region, used for Gaussians (non-Li lines).
        rv : float
             The rv found from the broad region, used for the whole model.
        teff : float
            Used in breidablik, teff of star
        logg : float
            Used in breidablik, logg of star 
        feh : float
            Used in breidablik, feh of star 
        rew_to_abund : object
            Converting REW to A(Li), used in Breidablik since the input there is A(Li), but the input to this function is EW
        max_ew : float, optional
            The maximum ew that the line can have, limit based on rew_to_abund
        min_ew : float, optional 
            The minimum ew that the line can have, limit based on rew_to_abund
        stdu : float, optional
            The upper limit on std in \AA, this is based on the broadening for GALAH (roughly R=22000)
        '''
        
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
        '''Fit ews of observed spectrum.

        Parameters
        ----------
        wl_obs : np.array
            observed wavelengths
        flux_obs : np.array
            observed flux
        flux_err : np.array
            observed flux error
        init : list
            The initial Li EW, Li std, other ews, in order of centers. 
        
        Returns
        -------
        fit, minchisq : 1darray, float
            Fitted parameters: Li EW, Li std, *ews, const; minimum chisq value at best fit.
        '''

        bounds = [(self.min_ew, self.max_ew), # based on rew_to_abund
                (5e-4, self.stdu)] # lower limit is sigma in \AA, corresponds to 0.05 FWHM in km/s
        bounds.extend([(0, np.inf) for _ in range(len(init)-2)]) # positive finite EW
        bounds.append((0.5, 1.5)) # continuum normalisation constant

        func = lambda x: chisq(wl_obs, flux_obs, flux_err, self.model, x, bounds, wl_left=6706.730*(1+self.rv/_c)-self.std*2, wl_right=6708.961*(1+self.rv/_c)+self.std*2)
        res = minimize(func, init, method='Nelder-Mead')

        return res.x, res.fun

    def model(self, wl_obs, params, plot=False, ax=None, plot_all=False):
        '''Gaussians multiplied together with Breidablik line profile.

        wl_obs : np.array
            observed wavelengths
        params : np.array
            Parameters to be fitted, Li EW, Li std, and rv.
        plot : bool
            If True, turns on plotting.
        ax : matplotlib.axes, optional
            The axis to plot on, if None, then it's the default one. 
        plot_all : bool
            If True, plot each gaussian. Or else plot only final model.
        
        Returns
        -------
        y : 1darray
            The model evaluated at given parameters. Gaussians multiplied with Breidablik line profile.
        '''

        ali, std_li, *ews, const = params
        y = line(wl_obs, ali, std_li, self.rv, breidablik=True, teff=self.teff, logg=self.logg, feh=self.feh, rew_to_abund=self.rew_to_abund, plot=plot, ax=ax) 
        
        for a, c in zip(ews, self.center):
            y1 = line(wl_obs, a, self.std, self.rv, center=c, plot=plot_all, ax=ax)
            y *= y1
        
        # plot
        if plot:
            if ax is None:
                ax = plt
            ax.plot(wl_obs, y, label='fit')
        
        y /= const
        
        return y


class FitBroad:
    '''Fits std and rv simultaneously and a EW for each center given. 
    For broad region.
    '''

    def __init__(self, center, stdl=None, stdu=None, rv_lim=None):
        '''Optional parameters are not needed if only using the model and not fitting. 
        
        Parameters
        ----------
        center : 1darray
            The line centers to be fitted. This should be np.array([6696.085, 6698.673, 6703.565, 6705.101, 6710.317, 6711.819, 6713.095, 6713.742, 6717.681])
        stdl : float, optional
            The lower limit on std, this is 0.09 \AA for this project.
        stdu : float, optional
            The upper limit on std in \AA, this is based on the broadening for GALAH (roughly R=22000)
        rv_lim : float, optional
            The limit on rv, mirrored limit on either side, it is the same limit as stdu, except in km/s. 
        '''

        self.center = center
        # don't need if using model
        self.stdl = stdl
        self.stdu = stdu
        self.rv_lim = rv_lim

    def fit(self, wl_obs, flux_obs, flux_err, init):
        '''Fit std and rv of observed spectrum.
        
        Parameters
        ----------
        wl_obs : 1darray
            observed wavelengths
        flux_obs : 1darray
            observed flux
        flux_err : 1darray
            observed flux error
        init : list
            The initial EW and std, rv; in that order. 

        Returns
        -------
        fit, minchisq : 1darray, float
            Fitted parameters: *EW, std, rv; minimum chisq value at best fit.
        '''
        
        # metal poor star
        if init is None:
            return None, None

        # construct bounds
        bounds = [(0, np.inf) for _ in range(len(init)-2)] # positive finite EW
        bounds.append((self.stdl, self.stdu)) # given in init
        bounds.append((-self.rv_lim, self.rv_lim)) # given in init
        
        # fit
        func = lambda x: chisq(wl_obs, flux_obs, flux_err, self.model, x, bounds)
        res = minimize(func, init, method='Nelder-Mead')
        fit = res.x

        return res.x, res.fun

    def model(self, wl_obs, params, plot=False, ax=None, plot_all=False):
        '''Multiplying Gaussians together with a common std and rv. 

        wl_obs : np.array
            observed wavelengths
        params : np.array
            Parameters to the model: *EWs, std, rv
        plot : bool
            If True, turns on plotting.
        ax : matplotlib.axes, optional
            The axis to plot on, if None, then it's the default one. 
        plot_all : bool
            If True, plot each gaussian. Or else plot only final model.

        Returns
        -------
        y : 1darray
            The model evaluated at given parameters. All gaussians multiplied together.
        '''
        
        *ews, std, offset = params
        y = np.ones(len(wl_obs))
        
        for a, c in zip(ews, self.center):
            y1 = line(wl_obs, a, std, offset, center=c, plot=plot_all, ax=ax)
            y *= y1
        
        # plot
        if plot:
            if ax is None:
                ax = plt
            ax.plot(wl_obs, y, label='fit')
        
        return y


def pred_amp(wl_obs, flux_obs, flux_err, centers, rv=0):
    '''Get the amplitudes for the initial guess. 

    Parameters
    ----------
    wl_obs : np.array
        observed wavelengths
    flux_obs : np.array
        observed flux
    flux_err : np.array
        observed flux error
    centers : 1darray
        The centers the lines are at -- these are the wls used to find the amplitudes
    rv : float
        rv shift, used to shift the centers

    Returns
    -------
    amps, err : 1darray, 1darray
        Amplitudes of observed spectra at rv shifted centers, set to 0 if negative; the flux errors at those amplitudes.
    '''

    # predict amplitudes
    inds = np.array([np.argmin(np.abs(wl_obs - i*(1+rv/299792.458))) for i in centers])
    amps = (1 - flux_obs[inds])*1.01 # bit bigger because sampling
    amps[amps < 0] = 0 # set negative amp to 0, chisq is inf otherwise
    err = flux_err[inds]
    return amps, err

def check_mp(amps, err):
    '''check if metal poor star. Criteria is <3 amplitudes above error.

    Parameters
    ----------
    amps : 1darray
        amplitudes 
    err : 1darray
        errors at the amplitudes

    Returns
    -------
    mp : bool
        If True, metal-poor star (less than 3 lines detected)
    '''

    mask = amps > err
    if np.sum(mask) < 3: # 3 lines detection is arbitrary
        return True
    else:
        return False

def cross_correlate(wl, flux, centers, params, rv):
    '''Calculate the cross correlation between template and obs flux.
    
    Parameters
    ----------
    wl : 1darray
        observed wavelengths
    flux : 1darray
        observed flux
    centers : float
        centers the lines are at
    params : 1darray
        Parameters of model (multiplying Gaussians together). *ews, std
    rv : float
        radial velocity shift
    
    Returns
    -------
    cc : float
        The cross correlation between the observed spectrum and the model spectrum
    '''
    
    fit_all = FitBroad(center=centers)
    template = fit_all.model(wl, [*params, rv])
    cc = np.sum(template*flux)
    return cc

def cc_rv(wl, flux, centers, params, rv_init, rv_lim):
    '''Get best rv from cross correlation. Searches 10 km/s to either side of rv_init.
    
    Parameters
    ----------
    wl : 1darray
        observed wavelengths
    flux : 1darray
        observed flux
    centers : 1darray
        centers of the lines
    params : 1darray
        Parameters of the model (mulitplying Gaussians together). *ews, std
    rv_init : float
        rv to search around
    rv_lim : float
        limit to the rvs searched through

    Returns
    -------
    rv : float
        rv from cross correlation (2dp accuracy)
    '''

    # rv 10 km/s is shifting about the line width. 
    rvs = np.linspace(rv_init-10, rv_init+10, 2000) # accurate to 2nd dp
    rvs = rvs[np.abs(rvs)<rv_lim]# filter out values beyond rv_lim
    ccs = [cross_correlate(wl, flux, centers, params, rv) for rv in rvs]
    return rvs[np.argmax(ccs)]

def filter_spec(spec, sigma=5):
    '''filter weird parts of the spectrum out.

    Parameters
    ----------
    spec : dict
        Dictionary containing spectrum, from read (keys: wave_norm, sob_norm, uob_norm)

    Returns
    -------
    spec : dict
        Filtered spectrum, in same dictionary as input.
    '''

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
    '''convert amplitudes to initial guess (ew & include std, rv)
    
    Parameters
    ----------
    amps : 1darray
        amplitudes
    std : float
        std of the gaussians
    rv : float
        radial velocity of the star

    Returns
    -------
    init : 1darray
        The initial guess. [*ews, std, rv], where ews are from amps and std. 
    '''
    
    init = list(np.array(amps)*np.sqrt(2*np.pi)*std) # amp to ew
    init.extend([std, rv])
    return np.array(init)

def iter_fit(wl, flux, flux_err, center, stdl, stdu, std_init, rv_lim):
    '''Iteratively fit the broad region. The fit is only as good as the rv, which is only as good as the ews. Hence iteratively fit between cross correlated rv and ews. Gives up after 5 iterations and returns the initial fit.

    Parameters
    ----------
    wl : 1darray
        observed wavelengths
    flux : 1darray
        observed flux
    flux_err : 1darray
        observed flux error
    center : 1darray
        centers of the lines to be fitted. Should be np.array([6696.085, 6698.673, 6703.565, 6705.101, 6710.317, 6711.819, 6713.095, 6713.742, 6717.681])).
    stdl : float
        The lower limit on std, this is 0.09 \AA for this project.
    stdu : float
        The upper limit on std in \AA, this is based on the broadening for GALAH (roughly R=22000)
    std_init : float
        The initial std from GALAH in \AA/
    rv_lim : float
        The limit on rv, mirrored limit on either side, it is the same limit as stdu, except in km/s. 
    
    Returns
    -------
    res : 1darray
        Iteratively fitted ews, std, rv
    '''

    # get initial rv
    fitter = FitBroad(center=center, stdl=stdl, stdu=stdu, rv_lim=rv_lim)
    amps, _ = pred_amp(wl, flux, flux_err, center)
    res = amp_to_init(amps, std_init, 0)
    init_rv = cc_rv(wl, flux, center, res[:-1], res[-1], rv_lim)
    # get initial amp
    amps, err = pred_amp(wl, flux, flux_err, center, rv=init_rv)
    init = amp_to_init(amps, std_init, init_rv)
    # check metal-poor star
    if check_mp(amps, err):
        return None

    # get good initial fit
    res, _ = fitter.fit(wl, flux, flux_err, init)
    if res is None: # metal poor star
        return None
    initial_res = np.copy(res)
    
    # iterate
    iterations = 1
    while np.abs(init_rv - res[-1]) > 0.1:
        *tparams, rv = res
        init_rv = cc_rv(wl, flux, center, tparams, rv, rv_lim)
        init_params = [*tparams, init_rv]
        res,  _ = fitter.fit(wl, flux, flux_err, init=init_params)
        iterations += 1
        if iterations >= 5:
            res = initial_res
            print('iterations over 5')
            break
    
    # check metal-poor again because some fits are bad
    if check_mp(res[:-2], err):
        return None

    return res

