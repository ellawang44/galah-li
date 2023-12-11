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

#TODO: check posteriors of high abundances
#TODO: check high abundance stuff after deciding what to do for errors

output_directory = 'results'
def grid_check(teffs, loggs, fehs):
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

sigma_num = 2

#TODO: how do upper limits correlate to anything? S/N? Teff? Abundance? 

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
# rewrite errors
data['err_low'] = data['ew_li'] - data['err_low']
data['err_upp'] = data['err_upp'] - data['ew_li']

# check that upper error and lower error bound the measured value
check = (data['err_low'][~np.isnan(data['err_low'])] > 0).all() and (data['err_upp'][~np.isnan(data['err_upp'])] > 0).all()
assert check, 'errors do not bound MLE'
# norris errors are within 0.015 of mcmc errors, mostly. can get up to 0.1. 

# check posterior stuff
# throw out the bad continuum, boundary expansion is correct, it is possible for sample to get driven to the edge after the boundary expansion
flag_cont = data['post_ind'] == -1 
print('no. with cont broken', np.sum(flag_cont)) # this is small enough that it doesn't matter
# print numbers for all parameter posteriors
flag_post = ~(data['post_ind'] == 99) 
print('no. with bad posterior (all params)', np.sum(flag_post))
# print number for A(Li) posteriors
flag_ali_err = data['post_ind'] == 0 
print('no. with bad posterior (A(Li))', np.sum(flag_ali_err))

print('no. of spectra', data.shape)

# manually remove binaries, bad cont norm
'''
# TODO: move this to somewhere else, we should still provide measured results
remove_ids = [170602002201196, 140310002701035, 140311008101002]
remove_ids.extend([140808001101091, 161006005401161]) # binary
remove_inds = [np.where(data['sobject_id'] == i)[0][0] for i in remove_ids]
data = np.delete(data, remove_inds, axis=0)

print('remove binary', 'no. of spectra', data.shape)
'''

# read in model
model_path = '/priv/avatar/ellawang/galah-li/model'
ffnn = FFNN(model=model_path+'/rew')
scalar = Scalar()
scalar.load(model_path+'/rew/scalar.npy')

# crossmatch between DR3
DR3 = pd.DataFrame(np.load('data/DR3_Li.npy'))
data = pd.DataFrame(data)
data = data.merge(DR3, on='sobject_id')
data = data.to_records()

# remove repeated observations of same star
# some stacked spectra will be broken due to single observations being broken - this will propogate through
mask = ~np.bool_(data['flag_repeat']) #np.ones(len(data['sobject_id']), dtype='bool') # simply keep all observations
data = data[mask]
print('no. of stars', np.sum(mask))

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
# no chisq cut is done because it is dependent on S/N as opposed to how the fitting actually looks 

# define detection limit 
upp_lim = np.copy(data['err_low'])*sigma_num # sig
upp_lim[np.isnan(upp_lim)] = data['norris'][np.isnan(upp_lim)]*sigma_num # where nan replace with norris
non_det = data['ew_li'] < upp_lim 
print('no. non det', np.sum(non_det))
norris_non_det = data['ew_li'] < data['norris']*sigma_num
print('norris no. non det', np.sum(norris_non_det))

# calculate REW
# report EW measurements, and detection and upper limits
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
# set to nan if non det
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
# other error stuff
y_test_err = np.sqrt(np.sum(np.square(np.array([y_test_ur - y_test_lr, y_test_ut - y_test_lt])/2), axis = 0))

print('mean new err', np.nanmean(y_test_err))
print('mean old err', np.nanmean(data['e_Li_fe']))
end = time.time()
print('time', end - start)

# write allstar catalogue
x = np.rec.array([
    data['sobject_id'],
    data['star_id'],
    data['teff'],
    data['e_teff'],
    data['logg'],
    data['e_logg'],
    data['fe_h'],
    data['e_fe_h'],
    data['ew_li']*1000,
    data['err_low']*1000,
    data['err_upp']*1000,
    data['norris']*1000,
    y_test,
    y_test_upp_lim,
    y_test - y_test_lr,
    y_test_ur - y_test,
    ((y_test - y_test_lt) + (y_test_ut - y_test))/2,
    data['Li_fe'] + data['fe_h'] + 1.05, # solar Li abundance in AS09, ALi_DR3
    data['e_Li_fe'],
    data['flag_sp'],
    data['flag_fe_h'],
    np.int_(non_det) + 2*np.int_(~in_grid) + 4*np.int_(flag_ali_err[mask]) + 8*np.int_(flag_cont[mask]),
    data['flag_li_fe'],
    data['std']*2.35482*299792.458/6707.814, # convert from \AA to km/s
    data['li_std']*2.35482*299792.458/6707.814,
    np.sqrt(np.square(data['vbroad']) + (299792.458/25500)**2), # add in instrumental profile
    data['rv'],
    data['rv_galah'],
    data['snr_c3_iraf']
    ],
    dtype=[
        ('sobject_id', int),
        ('star_id', '<U16'),
        ('teff_DR3', np.float64),
        ('e_teff_DR3', np.float64),
        ('logg_DR3', np.float64),
        ('e_logg_DR3', np.float64),
        ('fe_h_DR3', np.float64),
        ('e_fe_h_DR3', np.float64),
        ('EW', np.float64),
        ('e_EW_low', np.float64),
        ('e_EW_upp', np.float64),
        ('e_EW_norris', np.float64),
        ('ALi', np.float64),
        ('ALi_upp_lim', np.float64),
        ('e_ALi_low', np.float64),
        ('e_ALi_upp', np.float64),
        ('e_ALi_teff', np.float64),
        ('ALi_DR3', np.float64),
        ('e_ALi_DR3', np.float64),
        ('flag_sp_DR3', int),
        ('flag_fe_h_DR3', int),
        ('flag_ALi', int),
        ('flag_ALi_DR3', int),
        ('vbroad_broad', np.float64),
        ('vbroad', np.float64),
        ('vbroad_DR3', np.float64),
        ('delta_rv_6708', np.float64),
        ('rv_DR3', np.float64),
        ('snr_DR3', np.float64),
        ]
    )
np.save(f'{main_directory}/allstars.npy', x)
