These are exploratory simulations:

simulation0: Attempt at full run fails but runs with rmax=1000m giving 111 stations

python ../../simulate_low_rfi_visibility.py --rmax 1000 --npixel 512 \
--station_skip 1 --noise False --attenuation 1.0 --use_agg True \
--declination -45 --ngroup_visibility 32 \
--ntimes 16 --nchannels 128 

simulation1: Try all stations within 2km of core
simulation2: Increase use of memory
simulation3: Increase use of memory

After simulation 8, switched to calculate RA, Dec equivalent of emitter for each integration

simulation9: use_pole=False, station_ship=2, high SNR
simulation10: use_pole=False, low SNR
simulation11: station_skip=1
simulation12: station_skip=1, more integration