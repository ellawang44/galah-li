# galah-li
analysing Li in GALAH

# setup 
1. Change directories in config.py to the ones being used, then run config.py (to generate the data and fits folder), or you can put it in manually

2. Download the GALAH DR# file into the info directory (make sure name matches in config.py). e.g. for DR3:
`cd data`
`wget https://cloud.datacentral.org.au/teamdata/GALAH/public/GALAH_DR3/GALAH_DR3_main_allspec_v2.fits`

3. Run setup.sh: `bash setup.sh`

4. Download `https://github.com/ellawang44/astro_tools/blob/main/ffnn.py` and put it in the main directory. If you already have `astro_tools` (my own repository with a ffnn) installed then you can skip this step 

5. check qsub parameters - the ncpu is default, but the code is parallelised over the keys, these numbers are not compared. Reduce keys or increase ncpu as required.
If you need to regen the qsub file since the keys are messed up, `run get_id.py`

6. run qsub

7. allstars.py is the script using Breidablik converting EW to abundances. Creates a separate file containing stars with no duplicate spectra.

