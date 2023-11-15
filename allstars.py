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

output_directory = 'results'
def grid_check(teffs, loggs, fehs):
    with open('grid_snapshot.txt', 'r') as f:
        t_step, m_step = np.float_(f.readline().split())
        grid = np.loadtxt(f)
    scaled_sp = np.array([teffs*t_step, loggs, fehs*m_step]).T
    tile = np.array([np.tile(sp, (grid.shape[0], 1)) for sp in scaled_sp])
    dist = np.sqrt(np.sum(np.square(grid - tile), axis = 2))
    min_dist = np.min(dist, axis=1)
    in_grid = np.sqrt(3*0.25**2) > min_dist
    return in_grid

##
## Convert EW into abundances. 
##

start = time.time()
# read in data and combine into 1 array - it doesn't take too long to run.
dtype = [('sobject_id', '<f8'), ('ew_li', '<f8'), ('std', '<f8'), ('li_std', '<f8'), ('rv', '<f8'), ('err_low', '<f8'), ('err_upp', '<f8'), ('post_ind', '<f8'), ('norris', '<f8')] # need this to grab array quickly
dtype2 = [('sobject_id', '<i8'), ('ew_li', '<f8'), ('std', '<f8'), ('li_std', '<f8'), ('rv', '<f8'), ('err_low', '<f8'), ('err_upp', '<f8'), ('post_ind', '<i8'), ('norris', '<f8')] # this is the actual one
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
# this is EXTREMELY bad, like very broken bad
flag_bad = np.zeros(len(data), dtype=bool)

# check that upper error and lower error bound the measured value
mask = (data['err_low'] > data['ew_li']) & ~np.isnan(data['err_low']) & (data['post_ind'] == 99)
print('no. lower error > ew:', np.sum(mask))
# shape is expected, the MAP happens to be close to 5 but above. posteriors cut off by bounds
# the lower errors are inaccurate. 
# some of the numbers are detections. 
# proposed fix: remove lower errors. Would need to rerun with larger area if you wanted better results
# I think for DR4, condition for convergence needs to be 10 bins for EW Li parameter
mask = (data['ew_li'] > data['err_upp']) & ~np.isnan(data['err_upp']) & (data['post_ind'] == 99)
print('no. ew > upper error:', np.sum(mask))
# this is weirder, this one is not because of the bounds
# the posteriors simply have a sharp dropoff
# proposed fix: maybe use norris errors instead? TODO: check how good norris errors are as estimates

# check posterior stuff
# throw out the bad continuum, boundary expansion is correct, it is possible for sample to get driven to the edge after the boundary expansion
flag_cont = data['post_ind'] == -1 
print('no. with cont broken', np.sum(flag_cont)) # this is small enough that it doesn't matter
flag_bad = flag_bad & flag_cont
# print numbers for all parameter posteriors
flag_post = ~(data['post_ind'] == 99) 
print('no. with bad posterior (all params)', np.sum(flag_post))
# print number for A(Li) posteriors
flag_ali_err = data['post_ind'] == 0 
print('no. with bad posterior (A(Li))', np.sum(flag_ali_err))

# TODO: double check how abundance extrapolates, on the high end, (I think low end already works), and if this is fine, then report the higher error 

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
model_path = '/g/data1a/y89/xw5841/galah-li/model'
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

# define upper limit
#TODO: check that this is how we want to define upper limits
upper_lim = data['ew_li'] < (data['err_upp']-data['ew_li'])*2 # 2 sig
print('no. upper lim', np.sum(upper_lim))
norris_upper_lim = data['ew_li'] < data['norris']*2 
print('norris no. upper lim', np.sum(norris_upper_lim))
#TODO: check that errors from mcmc is similar enough to norris, for good stars
#TODO: report 2*upper error instead of the mean 
#TODO: if there is a star with massive abundance but also massive upper error, then it's an upper limit and that's ok, but check the spectra because it might be interesting. 


# grid check
in_grid = grid_check(data['teff'], data['logg'], data['fe_h'])
print('inside grid', np.sum(in_grid))

# chisq cut
# no chisq cut is done because it is dependent on S/N as opposed to how the fitting actually looks 

# calculate REW
# report EW measurements, but report A(Li) upper limits
ew_reassigned = np.copy(data['ew_li'])
ew_reassigned[upper_lim] = 2*delta_ew[upper_lim] # reassign to upper lims
center = 6707.8139458*(1 + data['rv']/299792.458) 
REW = np.log10(ew_reassigned / center)
REW_l = np.log10((ew_reassigned-delta_ew) / center)
REW_u = np.log10((ew_reassigned+delta_ew) / center)

# convert to Li abundance
X_test = np.array([data['teff'], data['logg'], data['fe_h'], REW]).T
X_test_lr = np.array([data['teff'], data['logg'], data['fe_h'], REW_l]).T
X_test_ur = np.array([data['teff'], data['logg'], data['fe_h'], REW_u]).T
X_test_lt = np.array([data['teff']-data['e_teff'], data['logg'], data['fe_h'], REW]).T
X_test_ut = np.array([data['teff']+data['e_teff'], data['logg'], data['fe_h'], REW]).T
y_test = ffnn.forward(scalar.transform(X_test)).flatten()
y_test_lr = ffnn.forward(scalar.transform(X_test_lr)).flatten()
y_test_ur = ffnn.forward(scalar.transform(X_test_ur)).flatten()
y_test_lt = ffnn.forward(scalar.transform(X_test_lt)).flatten()
y_test_ut = ffnn.forward(scalar.transform(X_test_ut)).flatten()
y_test_err = np.sqrt(np.sum(np.square(np.array([y_test_ur - y_test_lr, y_test_ut - y_test_lt])/2), axis = 0))
# needs to fall in li grid, lower extrapolation is ok, upper is not due to non-linearity
in_grid = in_grid & (y_test <= 4) 
# set error in teff to nan if error outside grid
eteff_l = grid_check(data['teff']-data['e_teff'], data['logg'], data['fe_h'])
# TODO: assume lower error is the same as the upper error in these cases, instead of doing nan
#TODO: double check that this works if the temperature is outside of the grid
#TODO: what if both errors are outside of the grid?? 
y_test_lt[~eteff_l] = np.nan
eteff_u = grid_check(data['teff']+data['e_teff'], data['logg'], data['fe_h'])
y_test_ut[~eteff_u] = np.nan
# set error in abundance above 4 to be nan
y_test_ur[y_test_ur > 4] = np.nan
# set error in abundance to nan if upper limit
y_test_lr[upper_lim] = np.nan
y_test_ur[upper_lim] = np.nan

print('mean new err', np.nanmean(y_test_err))
print('mean old err', np.nanmean(data['e_Li_fe']))
end = time.time()
print('time', end - start)

# write allstar catalogue
#TODO: review this, asymmetric errors?
#TODO: report all 3 stds: 1 from galah, 2 from my analysis
#TODO: new EW error needs to be asymmetric 
x = np.rec.array([
    data['sobject_id'],
    data['star_id'],
    data['teff'],
    data['e_teff'],
    data['logg'],
    data['e_logg'],
    data['fe_h'],
    data['e_fe_h'],
    data['ew_li'], 
    delta_ew,
    y_test,
    y_test - y_test_lr,
    y_test_ur - y_test,
    ((y_test - y_test_lt) + (y_test_ut - y_test))/2,
    data['Li_fe'] + data['fe_h'] + 1.05, # solar Li abundance in AS09, ALi_DR3
    data['e_Li_fe'],
    data['flag_sp'],
    data['flag_fe_h'],
    2*np.int_(~in_grid) + np.int_(upper_lim),
    data['flag_li_fe'],
    data['std']*2.35482*299792.458/6707.814, # convert from \AA to km/s
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
        ('e_EW', np.float64),
        ('ALi', np.float64),
        ('e_ALi_low', np.float64),
        ('e_ALi_upp', np.float64),
        ('e_ALi_teff', np.float64),
        ('ALi_DR3', np.float64),
        ('e_ALi_DR3', np.float64),
        ('flag_sp_DR3', int),
        ('flag_fe_h_DR3', int),
        ('flag_ALi', int),
        ('flag_ALi_DR3', int),
        ('vbroad', np.float64),
        ('vbroad_DR3', np.float64),
        ('delta_rv_6708', np.float64),
        ('rv_DR3', np.float64),
        ('snr_DR3', np.float64)
        ]
    )
np.save(f'{main_directory}/allstars.npy', x)
