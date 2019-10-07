from collections import namedtuple, OrderedDict
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from pycraf import pathprof
from pycraf import conversions as cnv
import xlrd
import time
from SKA_low_pycraf_prop import calc_prop_atten

import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
date_parser = pd.to_datetime


############
## Inputs ##
############


filtered = 'Filtered_DTV_list_1.csv'   # DTV transmitters csv
bandwidth = 7.  # MHz

outputdir = 'prop_out'

'''
=======================================
# Here input the parameters for the ITU-R 452-16 model
=======================================
'''
#	Units added in function
# freq = 180  # u.MHz
omega = 0.  # u.percent  # fraction of path over sea
temperature = 290.  # u.K
pressure = 1013.  # u.hPa
timepercent = 0.02  # u.percent  # see P.452 for explanation
height_rg = 2.1  # u.m#height of the receiver and transmitter above gnd
G_t, G_r = 0, 0  # cnv.dBi gains leave as 0. for now
zone_t, zone_r = pathprof.CLUTTER.UNKNOWN, pathprof.CLUTTER.UNKNOWN
# clutter type for transmitter/receiver
# hprof_step = 10 * u.m
hprof_step = 1000  # u.m  # resolution of solution
diam = 2  # u.m # assumed diameter of transmitter

N_freqs = 50  # no. of channels
# freq_start, freq_end = 170, 180  # frequency range MHz

# central receiver position for map calc.
rx_name, rx_lon, rx_lat = 'Low_E4', 116.75, -26.835


def Filter(string, substr):
    return [str for str in string if
             any(sub in str for sub in substr)]


## Start reading and using dataframes ##

# load filtered df to select licence_no #
filt_df = pd.read_csv(filtered, sep=',', low_memory=False)
filt_df = filt_df.set_index('Licence Number')
# print(filt_df.head(3))

for i in range(len(filt_df.index)):
	DTVant = filt_df.iloc[i]
	tx_name = str(DTVant['Callsign'])+'_'+str(DTVant['Site id'])
	print(tx_name)
	tx_freq = DTVant['Frequency(MHz)']
	tx_pol = DTVant['Polarisation']
	height_tx = DTVant['Antenna Height']
	pattern = DTVant['Antenna Pattern']
	tx_power = DTVant['Maximum ERP (W)']
	tx_lat = DTVant['Latitude (deg)']
	tx_lon = DTVant['Longitude (deg)']

	freq_start = tx_freq - (bandwidth/2.)
	freq_end = tx_freq + (bandwidth/2.)

	Atten_ant, freqs = calc_prop_atten(tx_freq, omega, temperature, pressure,
                                       timepercent, height_tx, height_rg, G_t, G_r,
                                       zone_t, zone_r, N_freqs,
                                       freq_start, freq_end, hprof_step,
                                       tx_name, tx_lon, tx_lat, rx_name,
                                       rx_lon, rx_lat, diam)
	print(Atten_ant.shape)
	# Output results to file
	np.save('Attenuation_prop'+tx_name+'.npy', Atten_ant)
	
	fig = plt.figure(figsize=(10, 10))
	# plt.semilogx(freqs,np.transpose(Atten_ant))
	plt.plot(freqs, np.transpose(Atten_ant))
	plt.xlabel('Frequency (MHz)')
	plt.ylabel('Total attenuation (dB)')
	plt.title('Station attenuation')
	plt.grid()
	plt.savefig('Station_attenuation_'+tx_name+'.png', bbox_inches='tight')




