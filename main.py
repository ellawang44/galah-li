# if a file doesn't exist, run setup.sh

from read import read_spectra, read_meta, cut
import os
import numpy as np
import matplotlib.pyplot as plt
from fit import filter_spec, line
from scipy.stats import norm
import argparse
from run import FitSpec
from config import *
import copy

# set up plotting and saving
save_fit = True
load_fit = True
plot = True
save = False

# argparse to change keys easily
parser = argparse.ArgumentParser(description='options for running')
parser.add_argument('-k', '--key', metavar='key', type=str, default='test', help='change the observations which are run, year/month. Needs to match id_dict contents')
args = parser.parse_args()
key = args.key

# if this key has already been run, don't run again
if os.path.exists(f'{output_directory}/{key}.npy') and save:
    print(f'Please remove results file for key {key} to re-run results')
    objectids = []
# testing purposes
elif key == 'test':
    #objectids = list(set([int(i[:-6]) for i in os.listdir('data/spec/com') if i[-4:] != '.txt'])) # my testing dataset, first 2 nights
    #objectids = [170121002201384, 170121002201396, 170104002901059, 170108002201155, 170508006401346, 140812004401119, 140823001401041, 160418005601330, 180621002901320, 160519005201183, 160520002601357, 150607003602126, 150601003201221, 171228003702082, 160130003101273] # a range of chisq to test new fitting
    #objectids = [140808000901102, 131216003201003, 140209005201151] # metal poor stars
    #objectids = np.load('data/benchmark.npy') # benchmark stars
    #objectids = [150112002502282] # young excited lil star
    #objectids = [171228003702082, 160130003101273] # high chisq, flag=0 -- cont norm, idk what
    #objectids = [140810002202184] # chisq min is correct but looks perhaps shifted?
    #objectids = [170516004101176, 150901002401298, 160420003301307, 150208005201271] # std and rv still drawing a line for flag=0 -- old results, not sure if still on line now # currently using a non-giant test set for multiple gaussians
    #objectids = [131120002001376] # high Li GC star
    #objectids = [140314002601106, 140607000701111, 160403002501179] # saturated stars
    #objectids = [131119001701221, 131216002601003, 150607003602126, 160530002201097] # Ce/V strong star, benchmark 0, 1 might be good edge case
   # objectids = [140710001701284] # test pcov
    #objectids = [161116002201392] # high feh, giant, check CN is fine, 
    #objectids = [180604003701205, 180604003701233, 170412004902165, 140710000801284] # SNR > 1200, check CN is fine
    #objectids = [160813005101117] # highest rot, 98 km/s. looks valid, all flags good, detection
    #objectids = [170912001201076] # breidablik ali is lower than gaussian, idk
    #objectids = [140808000901102, 150607003602126, 170506003401371, 160403002501179] # paper examples
    #objectids = [140607000701111, 160403002501179, 170506003401371, 140808000901102] # mcmc test
    #objectids = [170510005801366] # large sat pcov
    objectids = [150903002901344] # low detection
    #objectids = np.load('rerun.npy')
    # inherant broadening FWHM 10 km/s
    # initialise on pixel depth
    # min, 10, 100, 1000, 10000, 100000, max
    # removing cont norm helps for ~10000 chisq
# actual run
else:
    objectids = np.load(f'{info_directory}/id_dict.npy', allow_pickle=True).item()[key]

# calculate factors and get metadata values
error_factor = np.sqrt(3*np.pi)/(np.pi**(1/4))
data = np.load(f'{info_directory}/DR3_Li.npy')
sobject_id = data['sobject_id']
SNR = data['snr_c3_iraf']
DR3_rvs = data['rv_galah']
teff = data['teff']
logg = data['logg']
feh = data['fe_h'] 
vbroad = data['vbroad']
flags = np.array([data['flag_sp'], data['flag_fe_h'], data['flag_li_fe']]).T
factor = 6707.814/(2*np.sqrt(2*np.log(2)))/299792.458 # convert from km/s to \AA for std, might have a FWHM too

if save:
    data = []

for i in objectids:
    print(i)
    spectra = read_spectra(i)
    if spectra is None:
        continue

    # H alpha
    #spectra_halpha = copy.deepcopy(spectra)
    #spectra_halpha = cut(spectra_halpha, 6558, 6568)
    
    # cut to region to determine rv
    spectra = cut(spectra, 6695, 6719)
    spectra = filter_spec(spectra)
    
    spectra_broad = copy.deepcopy(spectra)
    if len(spectra['wave_norm']) == 0: # some spectra broken
        continue

    # identify object
    ind = np.where(i==sobject_id)[0][0]
    
    stdu = np.sqrt(vbroad[ind]**2 + (299792.458/22000)**2)*factor
    rv_lim = stdu/factor
    stdl = 0.09 #10km/s FWHM based on intrinsic broadening #np.sqrt(vbroad[ind]**2 + (299792.458/32000)**2)*factor
    std_galah = np.sqrt(vbroad[ind]**2 + (299792.458/25500)**2)*factor # eyeballed at R=25500
    if np.isnan(std_galah): # some DR3 have nan std
        continue

    # get std and rv
    fitspec = FitSpec(std_galah=std_galah, stdl=stdl, stdu=stdu, rv_lim=rv_lim, snr=SNR[ind], sid=i, teff=teff[ind], logg=logg[ind], feh=feh[ind])
    if os.path.exists(f'{info_directory}/fits/{i}.npy') and load_fit:
        fitspec.load(f'{info_directory}/fits/{i}.npy')
        spectra = cut(spectra, 6704, 6711)
        if save_fit:
            fitspec.save(f'{info_directory}/fits/{i}.npy')
    else:
        # fit h alpha region
        #fitspec.fit_halpha(spectra_halpha)

        # fit broad region
        fitspec.fit_broad(spectra_broad)

        # fit li region
        spectra = cut(spectra, 6704, 6711)
        fitspec.fit_li(spectra) # deals with metal poor
        fitspec.fit_sat(spectra) # deals with saturation
        fitspec.get_err(spectra['CDELT1']) # calculates delta_ew
        
        if save_fit:
            fitspec.save(f'{info_directory}/fits/{i}.npy')
    
    #print(fitspec.broad_fit)
    #print(fitspec.li_fit)
    #print(fitspec.sat_fit)
    #print(np.sqrt(np.diag(fitspec.sat_fit['pcov'])))
    #print(fitspec.li_fit['amps'][0], fitspec.li_fit['e_amps'][0], fitspec.delta_ew)
    
    if save:
        res = fitspec.sat_fit
        if len(res) == 0:
            res = {'amps':[np.nan], 'minchisq':np.nan, 'std':np.nan, 'rv':np.nan, 'pcov':[np.nan]}
        # save saturated or not
        #if fitspec.sat:
        #    res = fitspec.sat_fit # idk, pcov here is following the ALi, eali doesn't make sense
        #else:
        #    res = fitspec.li_fit
        
        std = res['std']
        #if fitspec.metal_poor:
        #    std = np.nan
        #else:
        #    std = fitspec.broad_fit['std']
        data_line = [i, res['amps'][0], res['minchisq'], std, res['rv'], fitspec.delta_ew, np.sqrt(np.diag(fitspec.li_fit['pcov'])[0]), np.sqrt(np.diag(res['pcov'])[0])]
        data.append(data_line)

    # plot halpha
    #if plot:
    #    fitspec.plot_halpha(spectra_halpha)
    
    # plot std and rv
    if plot:
        fitspec.plot_broad(spectra_broad)

    # plot results 
    if plot:
        fitspec.plot_li(spectra)

# need length check to make sure data isn't overwritten
if save and len(data) != 0: 
    data = np.array(data)
    x = np.rec.array([
        data[:,0],
        data[:,1],
        data[:,2],
        data[:,3],
        data[:,4],
        data[:,5],
        data[:,6],
        data[:,7]
        ], 
        dtype=[
            ('sobject_id', int),
            ('ew_li', np.float64),
            ('minchisq', np.float64),
            ('std', np.float64),
            ('rv', np.float64),
            ('delta_ew', np.float64),
            ('li_pcov', np.float64),
            ('pcov', np.float64)
            ]
        )
    np.save(f'{output_directory}/{key}.npy', x)
