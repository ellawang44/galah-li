import os

# all the paths

# path to the spectra fits files. Only need the ones with CCD3, but it's fine if others are there as well
working_directory = '/priv/avatar/thomasn/GALAH/com'

# directory which has access to the code, things will be launched from here
main_directory = '/priv/avatar/ellawang/galah-li'

# the intermediate directory to dump files needed to run the code 
info_directory = '/priv/avatar/ellawang/galah-li/data'
if not os.path.isdir(f'{info_directory}'):
    os.mkdir(f'{info_directory}')
# need to have a folder named fits inside it
if not os.path.isdir(f'{info_directory}/fits'):
    os.mkdir(f'{info_directory}/fits')

# info_directory also contains the GALAH DR file -- needed for sp
DR = 'GALAH_DR3_main_allspec_v2.fits'

# results directory
output_directory = '/priv/avatar/ellawang/galah-li/results'
