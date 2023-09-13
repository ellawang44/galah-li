# diagnostic plots

import numpy as np
import matplotlib.pyplot as plt
from run import FitSpec
from read import read_spectra, read_meta, cut
from PIL import Image
from config import *
from fit import filter_spec
import copy
from un_fitter import UNFitter

data = np.load(f'{info_directory}/DR3_Li.npy')                  
sobject_id = data['sobject_id']
SNR = data['snr_c3_iraf']
DR3_rvs = data['rv_galah']
teff = data['teff']
logg = data['logg']
feh = data['fe_h'] 
vbroad = data['vbroad']
e_vbroad = data['e_vbroad']
e_rv = data['e_rv_galah']
flags = np.array([data['flag_sp'], data['flag_fe_h'], data['flag_li_fe']]).T
factor = 6707.814/(2*np.sqrt(2*np.log(2)))/299792.458 # convert from km/s to \AA for std, might have a FWHM too

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
# initialise on pixel depth
# min, 10, 100, 1000, 10000, 100000, max
# removing cont norm helps for ~10000 chisq

for i in objectids:
    spectra = read_spectra(i)
    if spectra is None:
        continue

    spectra = cut(spectra, 6695, 6719)
    spectra = filter_spec(spectra) 
    spectra_broad = copy.deepcopy(spectra)
    if len(spectra['wave_norm']) == 0:
        continue

    spectra = cut(spectra, 6704, 6711)

    # identify object
    ind = np.where(i==sobject_id)[0][0]
    if np.isnan(e_vbroad[ind]):
        e_vbroad[ind] = 10
    if np.isnan(e_rv[ind]):
        e_rv[ind] = 3
    stdu = np.sqrt(vbroad[ind]**2 + (299792.458/22000)**2)*factor # max std based on R=22000
    stdue = np.sqrt((vbroad[ind]+3*e_vbroad[ind])**2 + (299792.458/22000)**2)*factor # max std based on R=22000
    rv_lim = stdu/factor
    stdl = 0.09 #10km/s FWHM based on intrinsic broadening 
    #stdl = np.sqrt(vbroad[ind]**2 + (299792.458/32000)**2)*factor # min std based on R=32000
    std_galah = np.sqrt(vbroad[ind]**2 + (299792.458/25500)**2)*factor # eyeballed at R=25500
    if np.isnan(std_galah): # some DR3 have nan std
        continue

    fitspec = FitSpec(std_galah=std_galah, stdl=stdl, stdu=stdu, stdue=stdue, rv_lim=rv_lim, e_vbroad=e_vbroad[ind]*factor, e_rv=e_rv[ind], snr=SNR[ind], sid=i, teff=teff[ind], logg=logg[ind], feh=feh[ind])
    if os.path.exists(f'{info_directory}/fits/{i}.npy'):
        fitspec.load(f'{info_directory}/fits/{i}.npy')
    else:
        continue

    # plot all things together
    fitspec.plot_broad(spectra_broad, show=False, path=f'view_temp/{i}_broad.png')
    plt.close()
    fitspec.plot_li(spectra, mode='minimize', show=False, path=f'view_temp/{i}_init.png')
    plt.close()
    if fitspec.sample is not None:
        fitspec.plot_li(spectra, mode='posterior', show=False, path=f'view_temp/{i}_li.png')
        plt.close()
        fitspec.plot_corner(show=False, path=f'view_temp/{i}_corner.png')
        plt.close()

    broad = Image.open(f'view_temp/{i}_broad.png')
    init = Image.open(f'view_temp/{i}_init.png')
    if fitspec.sample is not None:
        li = Image.open(f'view_temp/{i}_li.png')
        corner = Image.open(f'view_temp/{i}_corner.png')

    fig = plt.figure(figsize=(12,12), constrained_layout=True)
    gs = fig.add_gridspec(ncols=3, nrows=2, height_ratios=[1,3])
    ax0 = fig.add_subplot(gs[0,0])
    ax0.imshow(broad)
    ax0.axis('off')
    ax1 = fig.add_subplot(gs[0,1])
    ax1.imshow(init)
    ax1.axis('off')
    if fitspec.sample is not None:
        ax2 = fig.add_subplot(gs[0,2])
        ax2.imshow(li)
        ax2.axis('off')
        ax3 = fig.add_subplot(gs[1,:])
        ax3.imshow(corner)
        ax3.axis('off')
    title = f'{fitspec.mode}'
    if not fitspec.posterior_good:
        title = title + ' ' + str(fitspec.edge_ind)
    plt.title(title)
    plt.savefig(f'view/{i}.png')
    plt.close()


