# script to split the sobject_ids apart. 
# The run results are saved at the end, so if something crashes, chunking the objects means not everything is lost
# Currently chunked on year + month, then broken down further
import os
import numpy as np
from collections import defaultdict
from config import *

alphabet = 'abcdefghijklmn'

data = [int(i[:-6]) for i in os.listdir(f'{working_directory}') if i[-4:] == 'fits']
data = np.array(list(set(data)))

# keep 4 values, splits data into chunks of year/month
dates = set(np.floor(data/100000000000))
digits = len(str(list(dates)[0]).split('.')[0])

# like this is slow, but it only takes minutes and is done once so who cares
split_dates = defaultdict(list)
for date in dates:
    for d in data: 
        if str(int(date)) == str(d)[:digits]:
            split_dates[str(int(date))].append(d)

# max chunk is 45270 which is very large, split further
threshold = 40000 # can change this if you want, each chunk will be smaller
keys = list(split_dates.keys())
for key in keys:
    l = len(split_dates[key])
    if l > threshold:
        # figure out the spitting
        chunks = int(np.ceil(l/threshold))
        chunk_size = int(np.ceil(l/chunks))
        inds = [chunk_size*i for i in range(chunks+1)]
        inds = zip(inds[:-1], inds[1:])
        # add to dictionary
        values = split_dates[key]  
        for i, (l, r) in enumerate(inds):
            split_dates[key+alphabet[i]] = values[l:r]
        # remove old key from dictionary 
        del split_dates[key]
print('no. of chunks', len(split_dates.keys()))
np.save(f'{info_directory}/id_dict.npy', split_dates, allow_pickle=True)

# write qsub file
qsub_string = f'''#!/bin/bash                                                                                                           
#PBS -N GALAH_Li
#PBS -l select=1:ncpus=56
##PBS -l place=scatter:excl
#PBS -q smallmem

cd {main_directory}
export PATH=/pkg/linux/anaconda3/bin:$PATH
/home/thomasn/bin/parallel python3 main.py '''
with open('qsub', 'w') as f:
    f.write(qsub_string + '-k {1} ::: ' + ' '.join(split_dates.keys()))
