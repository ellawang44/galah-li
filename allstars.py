import numpy as np
import matplotlib.pyplot as plt
from scalar import Scalar
import os
import time
import pandas as pd
import matplotlib as mpl
from load import FFNN
from config import *
from read import read_spectra, read_meta, cut
from astropy.table import Table

# define variables and functions required for the analysis
output_directory = 'results'
def grid_check(teffs, loggs, fehs):
    '''Parallelised version of grid check from breidablik'''
    with open('grid_snapshot.txt', 'r') as f:
        t_step, m_step = np.float_(f.readline().split())
        grid = np.loadtxt(f)
    grid[:,0] = np.round(grid[:,0]*2)/2
    scaled_sp = np.array([teffs*t_step, loggs, fehs*m_step]).T
    tile = np.array([np.tile(sp, (grid.shape[0], 1)) for sp in scaled_sp])
    dist = np.sqrt(np.sum(np.square(grid - tile), axis = 2))
    min_dist = np.min(dist, axis=1)
    in_grid = np.sqrt(3*0.25**2) > min_dist
    return in_grid

##
## Convert EW into abundances. 
##

sigma_num = 2 # sigma that we make the detection cutoff at

start = time.time()
# read in data and combine into 1 array - it doesn't take too long to run.
dtype = [('sobject_id', '<f8'), ('ew_li', '<f8'), ('std', '<f8'), ('li_std', '<f8'), ('rv', '<f8'), ('err_low', '<f8'), ('err_upp', '<f8'), ('post_ind', '<f8'), ('area', '<f8'), ('stone', '<f8'), ('norris', '<f8')] # need this to grab array quickly
dtype2 = [('sobject_id', '<i8'), ('ew_li', '<f8'), ('std', '<f8'), ('li_std', '<f8'), ('rv', '<f8'), ('err_low', '<f8'), ('err_upp', '<f8'), ('post_ind', '<i8'), ('area', '<f8'), ('stone', '?'), ('norris', '<f8')] # this is the actual one
keys = os.listdir(output_directory)
data = []
for key in keys:
    data_key = np.load(f'{output_directory}/{key}')
    length = data_key.shape[0]
    nparray = data_key.astype(dtype).view('<f8').reshape(length, len(dtype))
    data.append(nparray)
data = np.concatenate(data, axis=0)
data = np.rec.array([
    data[:,i] for i in range(len(dtype))
    ], dtype=dtype2)
# rewrite errors as differences
data['err_low'] = data['ew_li'] - data['err_low']
data['err_upp'] = data['err_upp'] - data['ew_li']

# check that upper error and lower error bound the measured value
check = (data['err_low'][~np.isnan(data['err_low'])] > 0).all() and (data['err_upp'][~np.isnan(data['err_upp'])] > 0).all()
assert check, 'errors do not bound MLE'

# check posterior stuff
# flag bad continuum, boundary expansion is correct, it is possible for sample to get driven to the edge after the boundary expansion
flag_cont = data['post_ind'] == -1 
print('no. with cont broken', np.sum(flag_cont)) # this is small enough that it doesn't matter
# flag posterior not converged for any parameter
flag_post = ~(data['post_ind'] == 99) 
print('no. with bad posterior (all params)', np.sum(flag_post))
# flag posterior not converged for A(Li) 
flag_ali_err = data['post_ind'] == 0 
print('no. with bad posterior (A(Li))', np.sum(flag_ali_err))

print('no. of spectra', data.shape)

# read in model
ffnn = FFNN(model=model_path+'/rew')
scalar = Scalar()
scalar.load(model_path+'/rew/scalar.npy')

# crossmatch between DR3
DR3 = pd.DataFrame(np.load('data/DR3_Li.npy'))
data = pd.DataFrame(data)
data = data.merge(DR3, on='sobject_id')
data = data.to_records()

# grid check
in_grid = grid_check(data['teff'], data['logg'], data['fe_h'])
print('inside grid', np.sum(in_grid))
# check if there exists any teff in grid, but both errors outside of grid
lgrid = grid_check(data['teff']-data['e_teff'], data['logg'], data['fe_h'])
ugrid = grid_check(data['teff']+data['e_teff'], data['logg'], data['fe_h'])
mask_grid = in_grid & ~lgrid & ~ugrid & ~np.isnan(data['e_teff'])
print('value in grid err out grid', np.sum(mask_grid))
# because circle

# chisq cut
# no chisq cut is done because it will throw out some high S/N stars as well as bad fits

# define detections and upper limits
upp_lim = np.copy(data['err_low'])*sigma_num 
upp_lim[np.isnan(upp_lim)] = data['norris'][np.isnan(upp_lim)]*sigma_num # where nan replace with norris
non_det = data['ew_li'] < upp_lim 
print('no. non det', np.sum(non_det))
norris_non_det = data['ew_li'] < data['norris']*sigma_num
print('norris no. non det', np.sum(norris_non_det))

# calculate REW
center = 6707.8139458*(1 + data['rv']/299792.458) 
REW = np.log10(data['ew_li'] / center)
REW_l = np.log10((data['ew_li']-data['err_low']) / center)
REW_u = np.log10((data['ew_li']+data['err_upp']) / center)
REW_upp_lim = np.log10(upp_lim / center)

# convert to Li abundance
# MLE
X_test = np.array([data['teff'], data['logg'], data['fe_h'], REW]).T
y_test = ffnn.forward(scalar.transform(X_test)).flatten()
# upper lim
X_test_upp_lim = np.array([data['teff'], data['logg'], data['fe_h'], REW_upp_lim]).T
y_test_upp_lim = ffnn.forward(scalar.transform(X_test_upp_lim)).flatten()
# error in REW
X_test_lr = np.array([data['teff'], data['logg'], data['fe_h'], REW_l]).T
X_test_ur = np.array([data['teff'], data['logg'], data['fe_h'], REW_u]).T
y_test_lr = ffnn.forward(scalar.transform(X_test_lr)).flatten()
y_test_ur = ffnn.forward(scalar.transform(X_test_ur)).flatten()
# set to nan where required
y_test[non_det] = np.nan
y_test_lr[non_det] = np.nan
y_test_ur[non_det] = np.nan
y_test_upp_lim[~non_det] = np.nan
# error in teff
X_test_lt = np.array([data['teff']-data['e_teff'], data['logg'], data['fe_h'], REW]).T
X_test_ut = np.array([data['teff']+data['e_teff'], data['logg'], data['fe_h'], REW]).T
y_test_lt = ffnn.forward(scalar.transform(X_test_lt)).flatten()
y_test_ut = ffnn.forward(scalar.transform(X_test_ut)).flatten()
# set errors outside of grid to nan
y_test_lt[~lgrid] = np.nan
y_test_ut[~ugrid] = np.nan
# mirror errors in teff where nan
mask_lt = np.isnan(y_test_lt)
y_test_lt[mask_lt] = (y_test - (y_test_ut - y_test))[mask_lt]
mask_ut = np.isnan(y_test_ut)
y_test_ut[mask_ut] = (y_test + (y_test - y_test_lt))[mask_ut]
# calc mean err
y_test_err = np.sqrt(np.sum(np.square(np.array([y_test_ur - y_test_lr, y_test_ut - y_test_lt])/2), axis = 0))

print('mean new err', np.nanmean(y_test_err))
print('mean old err', np.nanmean(data['e_Li_fe']))
end = time.time()
print('time', end - start)

# write allspec catalogue
x = [
    data['sobject_id'],
    data['star_id'].astype(str),
    data['std']*2.35482*299792.458/6707.814, # convert from \AA to km/s
    data['li_std']*2.35482*299792.458/6707.814,
    data['vbroad'], 
    data['rv'],
    data['rv_galah'],
    data['ew_li']*1000, # AA to mAA
    data['err_low']*1000,
    data['err_upp']*1000,
    data['norris']*1000,
    y_test,
    y_test_upp_lim,
    y_test - y_test_lr,
    y_test_ur - y_test,
    ((y_test - y_test_lt) + (y_test_ut - y_test))/2,
    np.int_(non_det) + 2*np.int_(~in_grid) + 4*np.int_(flag_ali_err) + 8*np.int_(flag_cont) + 16*np.int_(data['flag_sp']),
    data['Li_fe'] + data['fe_h'] + 1.05, # solar Li abundance in AS09, ALi_DR3
    data['e_Li_fe'],
    data['flag_li_fe'],
    data['teff'],
    data['e_teff'],
    data['logg'],
    data['e_logg'],
    data['flag_sp'],
    data['fe_h'],
    data['e_fe_h'],
    data['flag_fe_h'],
    data['snr_c3_iraf']
    ]
names = [
        'sobject_id',
        'star_id',
        'sigma_broad',
        'sigma_Li',
        'vbroad_DR3',
        'delta_rv_6708',
        'rv_DR3',
        'EW',
        'e_EW_low',
        'e_EW_upp', 
        'e_EW_norris', 
        'ALi', 
        'ALi_upp_lim',
        'e_ALi_low', 
        'e_ALi_upp', 
        'e_ALi_teff',
        'flag_ALi', 
        'ALi_DR3', 
        'e_ALi_DR3', 
        'flag_ALi_DR3', 
        'teff_DR3', 
        'e_teff_DR3', 
        'logg_DR3', 
        'e_logg_DR3', 
        'flag_sp_DR3', 
        'fe_h_DR3', 
        'e_fe_h_DR3',
        'flag_fe_h_DR3',
        'snr_DR3'
        ]

dat = Table(x, names=names)
if os.path.exists(f'{main_directory}/GALAH_DR3_VAC_li_allspec_v2.fits'):
    print('allspec fits exists')
else:
    print('writing allspec')
    dat.write(f'{main_directory}/GALAH_DR3_VAC_li_allspec_v2.fits', format='fits')

# remove repeated observations of same star
# some stacked spectra will be broken due to single observations being broken - this will propogate through
mask = ~np.bool_(data['flag_repeat']) 
#mask = np.ones(len(data['sobject_id']), dtype='bool') # simply keep all observations
data = data[mask]
print('no. of stars', np.sum(mask))

# write allstar catalogue
x = [
    data['sobject_id'],
    data['star_id'].astype(str),
    data['std']*2.35482*299792.458/6707.814, # convert from \AA to km/s
    data['li_std']*2.35482*299792.458/6707.814,
    data['vbroad'], 
    data['rv'],
    data['rv_galah'],
    data['ew_li']*1000, # AA to mAA
    data['err_low']*1000,
    data['err_upp']*1000,
    data['norris']*1000,
    y_test[mask],
    y_test_upp_lim[mask],
    (y_test - y_test_lr)[mask],
    (y_test_ur - y_test)[mask],
    (((y_test - y_test_lt) + (y_test_ut - y_test))/2)[mask],
    (np.int_(non_det) + 2*np.int_(~in_grid) + 4*np.int_(flag_ali_err) + 8*np.int_(flag_cont))[mask] + 16*np.int_(data['flag_sp']),
    data['Li_fe'] + data['fe_h'] + 1.05, # solar Li abundance in AS09, ALi_DR3
    data['e_Li_fe'],
    data['flag_li_fe'],
    data['teff'],
    data['e_teff'],
    data['logg'],
    data['e_logg'],
    data['flag_sp'],
    data['fe_h'],
    data['e_fe_h'],
    data['flag_fe_h'],
    data['snr_c3_iraf']
    ]
        
dat = Table(x, names=names)
if os.path.exists(f'{main_directory}/GALAH_DR3_VAC_li_allstar_v2.fits'):
    print('allstar fits exists')
else:
    print('writing allstar')
    dat.write(f'{main_directory}/GALAH_DR3_VAC_li_allstar_v2.fits', format='fits')
