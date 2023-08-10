import numpy as np
from breidablik.interpolate.spectra import Spectra
from breidablik.analysis import read
from astro_tools import SpecAnalysis
from scipy.stats import norm
from astro_tools import vac_to_air

_c = 299792.458 # speed of light in km s^-1 
# optimised from 8s for 100 spectra to 2s - cut mainly, gaussian broadening versions don't make too much of a difference
_spectra = Spectra()
# cut to 6703 - 6712 (a little bit extra for rv shift)
_spectra.cut_models = _spectra.models[136:298]
_wl = vac_to_air(read.get_wavelengths()*10)[136:298]

def bline(ew, std, teff, logg, feh, ew_to_abund, min_ew):
    '''Breidablik line profiles

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

