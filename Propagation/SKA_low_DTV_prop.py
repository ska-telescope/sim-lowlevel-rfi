from collections import namedtuple, OrderedDict
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from pycraf import pathprof
from pycraf import conversions as cnv
import xlrd
import time
from SKA_low_pycraf_prop import calc_prop_atten
import os
import pprint

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
hprof_step = 10000  # u.m  # resolution of solution
diam = 2  # u.m # assumed diameter of transmitter

N_freqs = 3  # no. of channels
# freq_start, freq_end = 170, 180  # frequency range MHz

# central receiver position for map calc.
# rx_name, rx_lon, rx_lat = 'Low_E4', 116.75, -26.835
rx_name, rx_lon, rx_lat = 'Low_E4', 116.4525771, -26.60055525  # 116.75, -26.835


def Filter(string, substr):
    return [str for str in string if
            any(sub in str for sub in substr)]


if __name__ == '__main__':

    pp = pprint.PrettyPrinter()

    import argparse

    parser = argparse.ArgumentParser(
        description='Caluclate DTV RFI propagation')

    parser.add_argument('--transmitters', type=str, default='',
                        help='csv file containing transmitter properties')
    parser.add_argument('--context', type=str, default='DTV', help='DTV')
    parser.add_argument('--outdir', type=str, default='',
                        help='Directory to store results')
    parser.add_argument('--freq', type=float, default=177.5,
                        help='Central frequency (MHz)')
    parser.add_argument('--bandwidth', type=float,
                        default=7, help='Bandwidth (MHz)')
    parser.add_argument('--N_channels', type=int, default=3,
                        help='Number of frequency channels. (Must match nchannels_per_chunk for RFI simulation run)')
    parser.add_argument('--frequency_range', type=float, nargs=2, default=[170.5e6, 184.5e6],
                        help="Frequency range (Hz)")
    parser.add_argument('--hprof_step', type=float, default=1000,
                        help='Distance resolution of calculation')

    args = parser.parse_args()
    print("Starting LOW propagation calculation")

    pp.pprint(vars(args))

    # frequency = numpy.linspace(args.frequency_range[0], args.frequency_range[1], args.nchannels_per_chunk)

    filtered = args.transmitters
    bandwidth = args.bandwidth
    N_freqs = args.N_channels
    freq_start = args.frequency_range[0]
    freq_end = args.frequency_range[1]
    hprof_step = args.hprof_step
    outputdir = args.outdir

    ## Start reading and using dataframes ##

    # load filtered df to select licence_no #
    filt_df = pd.read_csv(filtered, sep=',', low_memory=False)
    filt_df = filt_df.set_index('Licence Number')
    # print(filt_df.head(3))
    # print(len(filt_df.index))
    transmitter_dict = {}

    for i in range(len(filt_df.index)):
        DTVant = filt_df.iloc[i]
        tx_name = str(DTVant['Callsign'])+'_'+str(DTVant['Site id'])
        print(tx_name)
        tx_freq = DTVant['Frequency(MHz)']
        tx_pol = DTVant['Polarisation']
        if tx_pol == 'V':
            polarization = 1
        else:  # if tx_pol == 'H'
	        polarization = 0  # 0 is default in pycraf

        height_tx = DTVant['Antenna Height']
        pattern = DTVant['Antenna Pattern']
        tx_power = DTVant['Maximum ERP (W)']
        tx_lat = -1*DTVant['Latitude (deg)']
        tx_lon = DTVant['Longitude (deg)']
        direct = DTVant['Antenna Pattern']

        freq_start = tx_freq - (bandwidth/2.)
        freq_end = tx_freq + (bandwidth/2.)
        # print(tx_freq, omega, temperature, pressure,
        #       timepercent, height_tx, height_rg, G_t, G_r,
        #       zone_t, zone_r, N_freqs,
        #       freq_start, freq_end, hprof_step,
        #       tx_name, tx_lon, tx_lat, rx_name,
        #       rx_lon, rx_lat, diam, direct, polarization)

        Atten_ant, freqs = calc_prop_atten(tx_freq, omega, temperature, pressure,
                                           timepercent, height_tx, height_rg, G_t, G_r,
                                           zone_t, zone_r, N_freqs,
                                           freq_start, freq_end, hprof_step,
                                           tx_name, tx_lon, tx_lat, rx_name,
                                           rx_lon, rx_lat, diam, direct, polarization)
        print('finished')
        # print(Atten_ant.shape)
        # Output results to file
        np.save(outputdir+'Attenuation_'+tx_name+'.npy', Atten_ant)
        transmitter_dict[tx_name] = {'location': [
            tx_lon, tx_lat], 'power': tx_power, 'height': height_tx}

        fig = plt.figure(figsize=(10, 10))
        # plt.semilogx(freqs,np.transpose(Atten_ant))
        plt.plot(freqs, np.transpose(Atten_ant))
        plt.xlabel('Frequency (MHz)')
        plt.ylabel('Total attenuation (dB)')
        plt.title('Station attenuation')
        plt.grid()
        plt.savefig(outputdir+'Station_attenuation_'+tx_name +
                    '.png', bbox_inches='tight')

        # np.save(outputdir+filtered[0:-4]+'_list.npy', transmitter_dict)
        print(filtered[0:-4].rsplit('/', 1))
        try:
        	print(outputdir+filtered[0:-4].rsplit('/', 1)[1]+'_list.npy')
	        np.save(outputdir+filtered[0:-4].rsplit('/', 1)[1]+'_list.npy', transmitter_dict)
        	# transmitter_dict = {'DTV1': {'location': [115.8605, -31.9505], 'power': 50000.0, 'height': 175}}
        except IndexError:
        	np.save(outputdir+filtered[0:-4].rsplit('/',1)[0]+'_list.npy',transmitter_dict)