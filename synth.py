import numpy as np
from breidablik.interpolate.spectra import Spectra
from breidablik.analysis import read
from astro_tools import SpecAnalysis
from scipy.stats import norm
from astro_tools import vac_to_air
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

_c = 299792.458 # speed of light in km s^-1 
# optimised from 8s for 100 spectra to 2s - cut mainly, gaussian broadening versions don't make too much of a difference
_spectra = Spectra()
# cut to 6703 - 6712 (a little bit extra for rv shift)
_spectra.cut_models = _spectra.models[136:298]
_wl = vac_to_air(read.get_wavelengths()*10)[136:298]

def bline(x, ew, std, rv, teff, logg, feh, ew_to_abund, min_ew, grid=None):
    '''Li line profiles from breidablik or interpolation grid.

    Parameters
    ----------
    x: 1darray
        The wavelengths to evaluate the spectral line at
    ew : float
        The EW of the line
    std : float
        The standard deviation of the line. If breidablik=True, this is the amount that the std that goes into the Gaussian convolution.
    rv : float
        The radial velocity. 
    teff : float, optional
        Used in breidablik, teff of star
    logg : float, optional
        Used in breidablik, logg of star 
    feh : float, optional
        Used in breidablik, feh of star 
    ew_to_abund : object, optional
        Converting EW to A(Li), used in Breidablik since the input there is A(Li), but the input to this function is EW. 
    min_ew : float, optional
        The EW corresponding to A(Li) = -0.5, used for mirroring to emission.
    grid : object, optional
        Grid of cubicsplines to interpolate spectra on, faster than computing from scratch. Default None which doesn't use grid.
    
    Returns 
    -------
    flux : 1darray
        The flux from Breidablik (3D NLTE Li profile).
    ''' 
    
    if (grid is None) or (grid.grid is None):
        flux = bprof(ew, std, teff, logg, feh, ew_to_abund, min_ew)
    else:
        flux = grid.interpolate(ew, std)
    wl = _wl*(1+rv/_c)
    return CubicSpline(wl, flux)(x)

def bprof(ew, std, teff, logg, feh, ew_to_abund, min_ew):
    '''Li line profiles from breidablik

    Parameters
    ----------
    ew : float
        The EW of the line
    std : float
        The standard deviation of the line. If breidablik=True, this is the amount that the std that goes into the Gaussian convolution.
    teff : float, optional
        Used in breidablik, teff of star
    logg : float, optional
        Used in breidablik, logg of star 
    feh : float, optional
        Used in breidablik, feh of star 
    ew_to_abund : object, optional
        Converting EW to A(Li), used in Breidablik since the input there is A(Li), but the input to this function is EW. 
    min_ew : float, optional
        The EW corresponding to A(Li) = -0.5, used for mirroring to emission.
    
    Returns 
    -------
    flux : 1darray
        The flux from Breidablik (3D NLTE Li profile).
    '''
    
    if ew >= min_ew:
        ali = ew_to_abund(ew)
        flux = _spectra._predict_flux(teff, logg, feh, [ali])[0]
    elif ew <= -min_ew:
        ali = ew_to_abund(np.abs(ew))
        flux = 2-_spectra._predict_flux(teff, logg, feh, [ali])[0]
    else:
        grid_ews = np.array([-min_ew, min_ew])
        flux = _spectra._predict_flux(teff, logg, feh, -0.5)[0]
        fluxes = np.array([2-flux, flux])
        grads = (fluxes[1] - fluxes[0])/(grid_ews[1] - grid_ews[0])
        intercepts = fluxes[1] - grads*grid_ews[1]
        flux = ew*grads+intercepts
    # gaussian broaden
    spec = SpecAnalysis(_wl, flux)
    _, flux = spec._gaussian_broaden(center=6707.814, sigma=std*2.35482*_c/6707.814)
    return flux

def gline(x, ew, std, rv, center):
    '''Create a Gaussian spectral line.
    
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
        The center that the line is at.

    Returns
    -------
    y : 1darray
        The spectral line, flux, at the input x wavelengths.
    '''

    y = 1-ew*norm.pdf(x, center*(1+rv/_c), std)
    return y

class Grid:
    '''Interpolation grid for Li over ews and std.
    '''

    def __init__(self, ewrange, stdrange, cutoff=4000, **kwargs):
        '''
        Parameters
        ----------
        ewrange : [float, float]
            min ew, max ew
        stdrange : [float, float]
            min std, max std
        cutoff : int, optional
            Any value higher will not have a grid computed for it, takes too long. 
        **kwargs
            kwargs that go into the breidablik profile function.
        '''
        
        self.ewnum = max(int(np.ceil((ewrange[1]-ewrange[0])/1e-3)), 2)
        self.ews = np.linspace(ewrange[0], ewrange[1], self.ewnum)
        self.stdnum = max(int(np.ceil((stdrange[1]-stdrange[0])/1e-2)), 2)
        self.stds = np.linspace(stdrange[0], stdrange[1], self.stdnum)
        if self.ewnum * self.stdnum > cutoff:
            self.grid = None
        else:
            self.grid = self.make_grid(**kwargs)

    def make_grid(self, **kwargs):
        '''Make grid of splines for interpolation.

        Parameters
        ----------
        **kwargs
            kwargs that go into the breidablik profile function.

        Returns
        -------
        grid : list of list of splines
            Grid of splines for interpolation.
        '''

        if self.ewnum >= self.stdnum:
            grid = []
            for std in self.stds:
                fluxes = [bprof(ew, std, **kwargs) for ew in self.ews]
                grid.append([CubicSpline(self.ews, f) for f in np.array(fluxes).T])
        
        if self.stdnum > self.ewnum:
            grid = []
            for ew in self.ews:
                fluxes = [bprof(ew, std, **kwargs) for std in self.stds]
                grid.append([CubicSpline(self.stds, f) for f in np.array(fluxes).T])
            
        return grid

    def interpolate(self, ew, std):
        '''Interpolate to profile on grid.

        Parameters
        ----------
        ew : float
            EW of the line 
        std : float
            broadening of the line
        
        Returns
        -------
        int_flux : 1darray
            The interpolated flux.
        '''

        if self.grid is None:
            return None
        
        if self.ewnum >= self.stdnum:
            flux = [[cs(ew) for cs in csarray] for csarray in self.grid]
            int_flux = [CubicSpline(self.stds, f)(std) for f in np.array(flux).T]

        if self.stdnum > self.ewnum:
            flux = [[cs(std) for cs in csarray] for csarray in self.grid]
            int_flux = [CubicSpline(self.ews, f)(ew) for f in np.array(flux).T]

        return np.array(int_flux)

