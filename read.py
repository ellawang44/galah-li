import glob
import numpy as np
import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
from config import *

def read_meta(directory=info_directory):
    '''Get metadata required to run.'''
    # get the columns that are useful and save it as npy format - done for speed
    # file is smaller, and np.load should be quicker than fits
    with pyfits.open(f'{directory}/{DR}') as hdul:
        data = hdul[1].data
        x = np.rec.array([
            data['sobject_id'],
            data['star_id'],
            data['flag_repeat'],
            data['snr_c3_iraf'],
            data['rv_galah'],
            data['e_rv_galah'],
            data['vbroad'],
            data['e_vbroad'],
            data['teff'],
            data['e_teff'],
            data['logg'],
            data['e_logg'],
            data['fe_h'],
            data['e_fe_h'],
            data['Li_fe'],
            data['e_Li_fe'],
            data['flag_sp'],
            data['flag_fe_h'],
            data['ind_flag_li6708']
            ],
            dtype=[
                ('sobject_id', int), 
                ('star_id', '<U16'),
                ('flag_repeat', int),
                ('snr_c3_iraf', np.float64),
                ('rv_galah', np.float64),
                ('e_rv_galah', np.float64),
                ('vbroad', np.float64),
                ('e_vbroad', np.float64),
                ('teff', np.float64),
                ('e_teff', np.float64),
                ('logg', np.float64),
                ('e_logg', np.float64),
                ('fe_h', np.float64),
                ('e_fe_h', np.float64),
                ('Li_fe', np.float64),
                ('e_Li_fe', np.float64),
                ('flag_sp', int),
                ('flag_fe_h', int),
                ('flag_li_fe', int)
                ]
            )
    np.save(f'{directory}/DR3_Li.npy', x)
    # make a hashmap between the sobject_ids and their indicies - done for speed
    # need this when analysing results
    objectid = data['sobject_id']
    hashmap = {}
    for ind, oid in enumerate(objectid):
        hashmap[oid] = ind
    np.save(f'{directory}/hashmap.npy', hashmap)

def read_spectra(sobject_id):
    """
    Read in all available in CCD3 and give back a dictionary
    """
    
    DR3 = np.load(f'{info_directory}/DR3_Li.npy')
    if not (sobject_id in DR3['sobject_id']):
        return None # some stars aren't published. politics, we ignore these stars anyway
        
    # Check if FITS files already available in working directory
    fits_files = f'{working_directory}/{sobject_id}3.fits'

    spectrum = dict()
    fits = pyfits.open(fits_files)
            
    # Extension 0: Reduced spectrum
    # Extension 1: Relative error spectrum
    # Extension 4: Normalised spectrum, NB: cut for CCD4

    # Extract wavelength grid for the reduced spectrum
    start_wavelength = fits[0].header["CRVAL1"]
    dispersion       = fits[0].header["CDELT1"]
    nr_pixels        = fits[0].header["NAXIS1"]
    reference_pixel  = fits[0].header["CRPIX1"]
    if reference_pixel == 0:
        reference_pixel = 1
    spectrum['wave_red'] = ((np.arange(0,nr_pixels)--reference_pixel+1)*dispersion+start_wavelength)

    # Extract flux and flux error of reduced spectrum
    spectrum['sob_red']  = np.array(fits[0].data)
    spectrum['uob_red']  = np.array(fits[0].data * fits[1].data)
    
    try:
        # Extract wavelength grid for the normalised spectrum
        start_wavelength = fits[4].header["CRVAL1"]
        dispersion       = fits[4].header["CDELT1"]
        nr_pixels        = fits[4].header["NAXIS1"]
        reference_pixel  = fits[4].header["CRPIX1"]
        if reference_pixel == 0:
            reference_pixel=1
        spectrum['wave_norm'] = ((np.arange(0,nr_pixels)--reference_pixel+1)*dispersion+start_wavelength)
        
        # Extract flux and flux error of reduced spectrum
        spectrum['sob_norm'] = np.array(fits[4].data)
        spectrum['uob_norm'] = np.array(fits[4].data * fits[1].data)
    except IndexError:
        print('no normalised spectra', sobject_id)
        DR3_rvs = DR3['rv_galah'][np.where(DR3['sobject_id'] == sobject_id)[0][0]]
        if np.isnan(DR3_rvs):
            print(sobject_id, 'nan')
            return None
        spectrum['wave_norm'] = spectrum['wave_red']*(1 + DR3_rvs/299792.458)
        spectrum['sob_norm'] = spectrum['sob_red']/np.median(spectrum['sob_red'])
        spectrum['uob_norm'] = spectrum['uob_red']/np.median(spectrum['sob_red'])
    
    spectrum['CDELT1'] = dispersion
    fits.close()

    return spectrum

def cut(spectrum, lower, upper):
    '''Cut the spectrum
    
    spectrum : dictionary
        Output from read_spectra
    lower : float
        The lower value to cut the spectrum to, in Angstroms
    upper : float
        The upper value to cut the specturm to, in Angstroms
    '''
    wl_mask = (lower <= spectrum['wave_norm']) & (spectrum['wave_norm'] <= upper)
    spectrum['wave_norm'] = spectrum['wave_norm'][wl_mask]
    spectrum['sob_norm'] = spectrum['sob_norm'][wl_mask]
    spectrum['uob_norm'] = spectrum['uob_norm'][wl_mask]
    return spectrum

if __name__ == '__main__':
    read_meta()
