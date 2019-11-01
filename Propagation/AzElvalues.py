from collections import namedtuple, OrderedDict
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from pycraf import pathprof
from pycraf import conversions as cnv
#import xlrd
import time
from SKA_low_pycraf_prop import calc_prop_atten

import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
date_parser = pd.to_datetime


############
## Inputs ##
############


#filtered = 'Filtered_DTV_list_1.csv'   # DTV transmitters csv
filtered = 'DTV_1775_full.csv'   # Two 177.5 MHz DTV transmitters csv
#filtered = 'Filtered_DTV_list_2.csv'  # DTV carnavon transmitter
bandwidth = 7.  # MHz

outputdir = 'prop_out'

'''
=======================================
# Here input the parameters for the ITU-R 452-16 model
=======================================
'''
#	Units added in function
freq = 177.5  # u.MHz
omega = 0.  # u.percent  # fraction of path over sea
temp = 290.  # u.K
press = 1013.  # u.hPa
timepercent = 0.02  # u.percent  # see P.452 for explanation
height_rg = 2.1  # u.m#height of the receiver and transmitter above gnd
G_t0, G_r0 = 0, 0  # cnv.dBi gains leave as 0. for now
zone_t, zone_r = pathprof.CLUTTER.UNKNOWN, pathprof.CLUTTER.UNKNOWN
# clutter type for transmitter/receiver
# hprof_step = 10 * u.m
hprof_step_0 = 1000  # u.m  # resolution of solution
diam = 2  # u.m # assumed diameter of transmitter

N_freqs = 3  # no. of channels
# freq_start, freq_end = 170, 180  # frequency range MHz

# central receiver position for map calc.
rx_name, rx_lon, rx_lat = 'Low_E4', 116.4525771, -26.60055525 # 116.75, -26.835

def Filter(string, substr):
    return [str for str in string if
             any(sub in str for sub in substr)]


## Start reading and using dataframes ##

# load filtered df to select licence_no #
filt_df = pd.read_csv(filtered, sep=',', low_memory=False)
filt_df = filt_df.set_index('Licence Number')
# print(filt_df.head(3))
print(len(filt_df.index))
results = []
fp = open('Transmitters/'+filtered[0:-4]+'_AzEl_list.txt','w')

for i in range(len(filt_df.index)):
	DTVant = filt_df.iloc[i]
	tx_name = str(DTVant['Callsign'])+'_'+str(DTVant['Site id'])
	print(tx_name)
	tx_freq = DTVant['Frequency(MHz)']
	tx_pol = DTVant['Polarisation']
	height_tx = DTVant['Antenna Height']
	pattern = DTVant['Antenna Pattern']
	tx_power = DTVant['Maximum ERP (W)']
	tx_lat = -1*DTVant['Latitude (deg)']
	tx_lon = DTVant['Longitude (deg)']

	freq_start = tx_freq - (bandwidth/2.)
	freq_end = tx_freq + (bandwidth/2.)

	freq = tx_freq * u.MHz
	omega = omega * u.percent  # fraction of path over sea
	temperature = temp * u.K
	pressure = press * u.hPa
	timepercent = timepercent * u.percent  # see P.452 for explanation
	# height of the receiver and transmitter above gnd
	h_tg, h_rg = height_tx * u.m, height_rg * u.m
	G_t, G_r = G_t0 * cnv.dBi, G_r0 * cnv.dBi
	# zone_t, zone_r = pathprof.CLUTTER.UNKNOWN, pathprof.CLUTTER.UNKNOWN
	# clutter type for transmitter/receiver
	# hprof_step = 10 * u.m
	# N_freqs = 100  # no. of channels
	# freq_start, freq_end = 170, 180  # frequency range MHz
	hprof_step = hprof_step_0 * u.m  # resolution of solution


	site, lon_tx, lat_tx = tx_name, tx_lon * u.deg, tx_lat * u.deg

	Site = namedtuple('sitetype', ['name', 'coord', 'pixoff', 'color'])
	sites = OrderedDict([
           # ID: tuple(Name, (lon, lat), Type, height, diameter,
           # (xoff, yoff), color)
           # ('Tx', Site('Tx', (u.Quantity(lon_tx).value,
           # u.Quantity(lat_tx).value), (20, +30), 'k')),
           ('Rec', Site(rx_name, (rx_lon, rx_lat), (60, -20), 'k')),
           ('Trans', Site(site, (tx_lon, tx_lat), (60, -20), 'k')),
       ])


	map_cen_lon = rx_lon
	map_cen_lat = rx_lat

	attenpath_calc_start = time.time()
	print('Calculating frequency dependent attenuation for '
    	'each station from path...')
	# print(freqs)
	hprof_cache = []


	hprof_cache = pathprof.height_path_data(lon_tx,
		lat_tx,
		rx_lon*u.deg,
		rx_lat*u.deg,
		hprof_step)

	pprop_fl_ras = pathprof.PathProp(freq,
    	temperature, pressure,
    	lon_tx, lat_tx, rx_lon*u.deg,
    	rx_lat*u.deg,
    	h_tg, h_rg,
    	hprof_step, timepercent,
    	zone_t=zone_t, zone_r=zone_r,)

	results.append([pprop_fl_ras.alpha_rt.value,pprop_fl_ras.eps_pr.value,pprop_fl_ras.distance.value])
	# print(pprop_fl_ras)
	# path elevation angles as seen from Tx, Rx: pprop_fl_ras.eps_pt, pprop_fl_ras.eps_pr
	# Azimuth from Tx to Rx and Rx to Tx: pprop_fl_ras.alpha_tr, pprop_fl_ras.alpha_rt
	print(pprop_fl_ras.alpha_rt, pprop_fl_ras.eps_pr)
	print(pprop_fl_ras.alpha_rt.value, pprop_fl_ras.eps_pr.value,file=fp)

results = np.array(results)
np.save('Transmitters/'+filtered[0:-4]+'_AzElDist.npy',results)
fp.close()
