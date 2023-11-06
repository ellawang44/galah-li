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

# set up plotting and saving
save_fit = True # individual fits, 1 file per fit (all info)
load_fit = False # individual fits, 1 file per fit  (all info)
plot = False
save = True # simplified fit results, compiled into 1 file

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
    objectids = []
    objectids.extend([170121002201384, 170121002201396, 170104002901059, 170108002201155, 170508006401346, 140812004401119, 140823001401041, 160418005601330, 180621002901320, 160519005201183, 160520002601357, 150607003602126, 150601003201221, 171228003702082, 160130003101273]) # a range of chisq to test new fitting
    objectids.extend([140808000901102, 131216003201003, 140209005201151]) # metal poor stars
    objectids.extend(list(np.load('data/benchmark.npy'))) # benchmark stars
    objectids.append(150112002502282) # young excited lil star
    objectids.extend([171228003702082]) # cont norm is bad 
    objectids.extend([170516004101176, 150901002401298, 160420003301307, 150208005201271]) # std and rv still drawing a line for flag=0 -- old results, not sure if still on line now # currently using a non-giant test set for multiple gaussians
    objectids.append(131120002001376) # high Li GC star
    objectids.extend([140314002601106, 140607000701111, 160403002501179]) # saturated stars
    objectids.extend([131119001701221, 131216002601003, 150607003602126, 160530002201097]) # Ce/V strong star, benchmark 0, 1 might be good edge case
    objectids.append(161116002201392) # high feh, giant, check CN is fine, 
    objectids.extend([180604003701205, 180604003701233, 170412004902165, 140710000801284]) # SNR > 1200, check CN is fine
    objectids.extend([160813005101117, 170912001201076]) # the spectrum is too blended, the initial guess is very hard to do correctly, current fit is bad, needs better initial guess
    objectids.extend([140808000901102, 150607003602126, 170506003401371, 160403002501179]) # paper examples
    objectids.extend([140607000701111, 160403002501179, 170506003401371, 140808000901102]) # mcmc test
    objectids.append(170510005801366) # very blended, very weak
    objectids.append(150903002901344) # low detection
    # inherant broadening FWHM 10 km/s
    #objectids = [131120002001376] # my lovely "quick" test case after implementing changes
    #objectids = [140314005201392] # TiO & CN filled star
    #objectids = [180620001701391] # bad continuum const, check later
    #objectids = [150210002701226, 160524002701187, 181224002101375, 170106003601064] # err_upp < ew_li, example cases
    # strange ali < 0.8 detections 
    #objectids = [140414002601139, 150410003301377, 170710002201028] # RHS falls down too hard, causing upper error to be lower than MLE, should be fixed with new errors
# actual run
else:
    objectids = np.load(f'{info_directory}/id_dict.npy', allow_pickle=True).item()[key]

# calculate factors and get metadata values
data = np.load(f'{info_directory}/DR3_Li.npy')
sobject_id = data['sobject_id']
SNR = data['snr_c3_iraf']
DR3_rvs = data['rv_galah']
e_rv = data['e_rv_galah']
teff = data['teff']
logg = data['logg']
feh = data['fe_h'] 
vbroad = data['vbroad']
e_vbroad = data['e_vbroad']
factor = 6707.814/(2*np.sqrt(2*np.log(2)))/299792.458 # convert from km/s to \AA for std, might have a FWHM too

if save:
    data = []

for i in objectids:
    print(i)
    #if save_fit and os.path.exists(f'{info_directory}/fits/{i}.npy'):
    #    continue

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
    stdue = np.sqrt((vbroad[ind]+3*e_vbroad[ind])**2 + (299792.458/22000)**2)*factor
    rv_lim = stdu/factor
    stdl = 0.09 #10km/s FWHM based on intrinsic broadening 
    #stdl = np.sqrt(vbroad[ind]**2 + (299792.458/32000)**2)*factor # min std based on R=32000
    std_galah = np.sqrt(vbroad[ind]**2 + (299792.458/25500)**2)*factor # eyeballed at R=25500
    if np.isnan(std_galah): # some DR3 have nan std
        continue

    # fitting
    fitspec = FitSpec(std_galah=std_galah, stdl=stdl, stdu=stdu, stdue=stdue, rv_lim=rv_lim, e_vbroad=e_vbroad[ind]*factor, e_rv=e_rv[ind], snr=SNR[ind], sid=i, teff=teff[ind], logg=logg[ind], feh=feh[ind])
    # load fit
    if os.path.exists(f'{info_directory}/fits/{i}.npy') and load_fit:
        fitspec.load(f'{info_directory}/fits/{i}.npy')
        if save_fit:
            fitspec.save(f'{info_directory}/fits/{i}.npy')
    # fit
    else:
        # fit broad region
        fitspec.fit_broad(spectra_broad)

        # fit li region
        fitspec.fit_li(spectra) 

        # get error
        fitspec.posterior(spectra) # calculates the error approx and posterior

        if save_fit:
            fitspec.save(f'{info_directory}/fits/{i}.npy')
    
    if save:
        li_fit = fitspec.li_fit
        # if no posterior fit was done
        if li_fit is None:
            li_fit = fitspec.li_init_fit

        broad_fit = fitspec.broad_fit
        # if metal poor star - no broad fit
        if broad_fit is None:
            broad_fit = {'std':np.nan}

        # Nones break things
        if fitspec.edge_ind is None:
            fitspec.edge_ind = 99

        data_line = [i, li_fit['amps'][0], broad_fit['std'], li_fit['std'], li_fit['rv'], *fitspec.err, fitspec.edge_ind, fitspec.norris] 
        data.append(data_line)
    
    if plot:
        # plot broad region
        fitspec.plot_broad(spectra_broad)
        # plot Li region
        fitspec.plot_li(spectra, mode='minimize')
        # plot cornerplot
        fitspec.plot_corner()
        # plot Li region
        fitspec.plot_li(spectra, mode='posterior')

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
        data[:,7],
        data[:,8]
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
            ('norris', np.float64)
            ]
        )
    np.save(f'{output_directory}/{key}.npy', x)

