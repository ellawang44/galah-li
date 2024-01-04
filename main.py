# if a file doesn't exist, run setup.sh and config.py

from read import read_spectra, read_meta, cut
import os
import numpy as np
import matplotlib.pyplot as plt
from fit import filter_spec
from scipy.stats import norm
import argparse
from run import FitSpec
from config import *
import copy

# argparse to change keys easily
parser = argparse.ArgumentParser(description='options for running')
parser.add_argument('-k', '--key', metavar='key', type=str, default='test', help='change the observations which are run, year/month. Needs to match id_dict contents')
parser.add_argument('--sid', metavar='sid', type=int, default=131120002001376, help='The sobject_id of the star to be run, default star is a "quick" test case')
parser.add_argument('--save_fit', action='store_true', help='save individual fits, 1 file per fit (all info)')
parser.add_argument('--load_fit', action='store_true', help='load individual fits, 1 file per fit  (all info)')
parser.add_argument('--plot', action='store_true', help='plot results')
parser.add_argument('--save', action='store_true', help='save simplified fit results, compiled into 1 file')
args = parser.parse_args()
load_fit = args.load_fit

# if this key has already been run, don't run again
if os.path.exists(f'{output_directory}/{args.key}.npy') and args.save:
    print(f'Please remove results file for key {args.key} to re-run results')
    objectids = []
# testing purposes
elif args.key == 'test':
    objectids = [args.sid]
    # inherant broadening FWHM 10 km/s
# actual run
else:
    objectids = np.load(f'{info_directory}/id_dict.npy', allow_pickle=True).item()[args.key]

# calculate factors and get metadata values
data = np.load(f'{info_directory}/DR3_Li.npy')
sobject_id = data['sobject_id']
SNR = data['snr_c3_iraf']
e_rv = data['e_rv_galah']
teff = data['teff']
logg = data['logg']
feh = data['fe_h'] 
vbroad = data['vbroad']
e_vbroad = data['e_vbroad']
factor = 6707.814/(2*np.sqrt(2*np.log(2)))/299792.458 # convert from km/s to \AA for std, might have a FWHM too

if args.save:
    data = []

for i in objectids:
    print(i)

    spectra = read_spectra(i)
    if spectra is None:
        continue

    # cut to broad region for std and rv fitting
    spectra = cut(spectra, 6695, 6719)
    spectra = filter_spec(spectra) 
    spectra_broad = copy.deepcopy(spectra)
    if len(spectra['wave_norm']) == 0: # some spectra broken
        continue

    # cut to Li region for detailed fitting
    spectra = cut(spectra, 6704, 6711)

    # identify object
    ind = np.where(i==sobject_id)[0][0]
    if np.isnan(e_vbroad[ind]):
        e_vbroad[ind] = 10 # this is where 99% of the values are below
    if np.isnan(e_rv[ind]):
        e_rv[ind] = 3 # 99% of values are below
    stdu = np.sqrt(vbroad[ind]**2 + (299792.458/22000)**2)*factor # max std based on R=22000
    stdue = np.sqrt((vbroad[ind]+e_vbroad[ind])**2 + (299792.458/22000)**2)*factor
    rv_lim = stdu/factor
    stdl = 0.09 #10km/s FWHM based on intrinsic broadening 
    #stdl = np.sqrt(vbroad[ind]**2 + (299792.458/32000)**2)*factor # min std based on R=32000
    std_galah = np.sqrt(vbroad[ind]**2 + (299792.458/25500)**2)*factor # eyeballed at R=25500
    if np.isnan(std_galah): # some DR3 have nan std
        continue

    # fitting
    fitspec = FitSpec(std_galah=std_galah, stdl=stdl, stdu=stdu, stdue=stdue, rv_lim=rv_lim, e_vbroad=e_vbroad[ind]*factor, e_rv=e_rv[ind], snr=SNR[ind], sid=i, teff=teff[ind], logg=logg[ind], feh=feh[ind])
    # load fit
    if os.path.exists(f'{info_directory}/fits/{i}.npy') and args.load_fit:
        fitspec.load(f'{info_directory}/fits/{i}.npy')
        if args.save_fit:
            fitspec.save(f'{info_directory}/fits/{i}.npy')
    # fit
    else:
        # fit broad region
        fitspec.fit_broad(spectra_broad)

        # fit li region
        fitspec.fit_li(spectra) 

        # get error
        fitspec.posterior(spectra) # calculates the error approx and posterior

        if args.save_fit:
            fitspec.save(f'{info_directory}/fits/{i}.npy')

    if args.save:
        li_fit = fitspec.li_fit
        # if no posterior fit was done
        if li_fit is None:
            li_fit = fitspec.li_init_fit

        broad_fit = fitspec.broad_fit
        # if poorly constrained star - no broad fit
        if broad_fit is None:
            broad_fit = {'std':np.nan}
        
        data_line = [i, li_fit['amps'][0], broad_fit['std'], li_fit['std'], li_fit['rv'], *fitspec.err, fitspec.edge_ind, fitspec.area, fitspec.stone_good, fitspec.norris] 
        data.append(data_line)
    
    if args.plot:
        # plot broad region
        fitspec.plot_broad(spectra_broad)
        # plot Li region
        fitspec.plot_li(spectra, mode='minimize')
        if fitspec.sample is not None:
            # plot cornerplot
            fitspec.plot_corner()
            # plot Li region
            fitspec.plot_li(spectra, mode='posterior')
        
if args.save: 
    data = np.array(data)
    x = np.rec.array([
        data[:,0],
        data[:,1],
        data[:,2],
        data[:,3],
        data[:,4],
        data[:,5],
        data[:,6],
        data[:,7],
        data[:,8],
        data[:,9],
        data[:,10]
        ], 
        dtype=[
            ('sobject_id', int),
            ('ew_li', np.float64),
            ('std', np.float64),
            ('li_std', np.float64),
            ('rv', np.float64),
            ('err_low', np.float64),
            ('err_upp', np.float64),
            ('post_ind', int),
            ('area', np.float64),
            ('stone', bool),
            ('norris', np.float64)
            ]
        )
    np.save(f'{output_directory}/{args.key}.npy', x)

