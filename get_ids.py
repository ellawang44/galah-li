# script to split the sobject_ids apart. 
# The run results are saved at the end, so if something crashes, chunking the objects means not everything is lost
# Currently chunked on year + month, then broken down further
import os
import numpy as np
from collections import defaultdict
from config import *
from string import Template as template

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
with open('qsub_template', 'r') as f:
    raw = template(f.read())
    ncpu = len(split_dates.keys())
    filled = raw.safe_substitute(ncpu=ncpu-1, main_directory=main_directory, keys=' '.join(split_dates.keys()))
with open('qsub', 'w') as f:
    f.write(filled)
