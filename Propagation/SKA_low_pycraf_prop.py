#!/usr/bin/env python
# coding: utf-8


# Original by Federico Di Vruno
# Alterations by DMF


from collections import namedtuple, OrderedDict
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from pycraf import pathprof, antenna, geometry
from pycraf import conversions as cnv
import xlrd
import time

pathprof.SrtmConf.set(download='missing', server='viewpano')
# allow download of missing SRTM data


def calc_prop_atten(freq, omega, temp, press, timepercent, h_tg, h_rg,
                    G_t, G_r, zone_t, zone_r, N_freqs, freq_start,
                    freq_end, hprof_step,
                    tx_name, tx_lon, tx_lat, rx_name, rx_lon, rx_lat,
                    diam):

    freq = freq * u.MHz
    omega = omega * u.percent  # fraction of path over sea
    temperature = temp * u.K
    pressure = press * u.hPa
    timepercent = timepercent * u.percent  # see P.452 for explanation
    # height of the receiver and transmitter above gnd
    h_tg, h_rg = h_tg * u.m, h_rg * u.m
    G_t, G_r = G_t * cnv.dBi, G_r * cnv.dBi
    # zone_t, zone_r = pathprof.CLUTTER.UNKNOWN, pathprof.CLUTTER.UNKNOWN
    # clutter type for transmitter/receiver
    # hprof_step = 10 * u.m
    # N_freqs = 100  # no. of channels
    # freq_start, freq_end = 170, 180  # frequency range MHz
    hprof_step = hprof_step * u.m  # resolution of solution

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

    '''
	===========================================
	    Antennas locations

	===========================================
	'''
    # For LOW, get list of station name/locations from xlsx worksheet
    Antenna_locations = xlrd.open_workbook(
        '/local/SKA/RFI/SKA1_LOW_coordinates_REV3.xlsx')
    sheet_indx = 0
    sheet = Antenna_locations.sheet_by_index(sheet_indx)
    name_x = list()
    lat_x = np.zeros(sheet.nrows-2)
    long_x = np.zeros(sheet.nrows-2)
    for i in range(2, sheet.nrows-1):
        name_x.append(sheet.cell_value(i, 0))
        long_x[i-2] = sheet.cell_value(i, 1)
        lat_x[i-2] = sheet.cell_value(i, 2)

    # %%
    '''
    ===========================================
         Get the map height information

    ===========================================
    '''
    '''
	terrain_start = time.time()
	print('Getting terrain map information...')
    # get the data for the terrain map of region around location above i.e. transmitter
    # mostly just for getting the max/min for plotting the terrain below
	lons, lats, heightmap = pathprof.srtm_height_map(
		map_cen_lon*u.deg, map_cen_lat*u.deg,
		# lon_tx, lat_tx,
		map_size_lon, map_size_lat,
		map_resolution = map_resolution,
		hprof_step = hprof_step
		)
	terrain_end = time.time()
	print('Completed in %f sec.' %(terrain_end-terrain_start))
	'''
    '''
	===========================================
	Calculate the attenuation map

	===========================================
	'''
    '''
	attenmap_start = time.time()
	print('Calculating attenuation map...')
	# Faster approach
	# Calculates auxillary maps and height profiles for atten_map_fast
	# Height profile data independant of frequency, so can be cached
	# lon_tx,lat_tx transmitter
	# returns dictionary containing lon_t, lat_t, map_size_lon, map_size_lat, hprof_step
	# xcoords, ycoords (coords first row/col in map) also others, input directly into attenmap_fast
	hprof_cache = pathprof.height_map_data(lon_tx, lat_tx,
												atten_map_size_lon, atten_map_size_lat,
                                           		map_resolution=atten_map_resolution,
                                       			zone_t=zone_t, zone_r=zone_r,
                                       			)
	results = pathprof.atten_map_fast(freq,
	                               temperature,
	                               pressure,
	                               h_tg, h_rg,
	                               timepercent,
	                               hprof_cache,  # dict_like
	                               )
	_lons = hprof_cache['xcoords']
	_lats = hprof_cache['ycoords']
	# L_b is the total attenuation, considering all the factors.
	_total_atten = results['L_b']
	# print(_total_atten.shape,_total_atten[0])
	_fspl_atten = results['L_bfsg']  # considers only the free space loss

	attenmap_end = time.time()
	print('Completed in %f sec.' %(attenmap_end-attenmap_start))
	'''

    '''
	===========================================
	Find the attenuation at each position of SKA antennas using path info for each freq.
	===========================================
	'''
    attenpath_calc_start = time.time()
    print('Calculating frequency dependent attenuation for '
          'each station from path...')
    N_ant = len(long_x)-1
    freqs = np.logspace(np.log10(freq_start),
                        np.log10(freq_end), N_freqs)*u.MHz
    # print(freqs)
    Atten_ant = np.zeros([N_ant, N_freqs])
    results = []
    hprof_cache = []

    for k in range(N_ant):

        hprof_cache = pathprof.height_path_data(lon_tx,
                                                lat_tx,
                                                long_x[k]*u.deg,
                                                lat_x[k]*u.deg,
                                                hprof_step)
        pprop_fl_ras = pathprof.PathProp(freq,
                                         temperature, pressure,
                                         lon_tx, lat_tx, long_x[k]*u.deg,
                                         lat_x[k]*u.deg,
                                         h_tg, h_rg,
                                         hprof_step, timepercent,
                                         zone_t=zone_t, zone_r=zone_r,)

        # print('FL-FL,  Az/El: {:.1f}, {:.1f}'.format(
        #	pprop_fl_ras.alpha_tr, pprop_fl_ras.eps_pt))

        ## if had two directions could calc. angular distance #
        # ang_dist = geometry.true_angular_distance(
        #	pprop_fl.alpha_tr, pprop_fl.eps_pt,
        #	pprop_fl_ras.alpha_tr, pprop_fl_ras.eps_pt,)
        #print('Angular distance: {:.1f}'.format(ang_dist))

        ang_dist = geometry.true_angular_distance(
            pprop_fl_ras.alpha_tr, pprop_fl_ras.eps_pt,
            0.0*u.deg, 0.0*u.deg,)
        # print('Angular distance: {:.1f}'.format(ang_dist))

        for i in range(N_freqs):
            # print(k,i)

            diameter_fl = diam * u.m
            wavelen_fl = freqs[i].to(u.m, equivalencies=u.spectral())
            G_max_fl = antenna.fl_G_max_from_size(diameter_fl, wavelen_fl)
            # print('FL max. gain: {:.1f}'.format(G_max_fl))

            G_eff = antenna.fl_pattern(
                ang_dist, diameter_fl, wavelen_fl, G_max_fl)
            # print('Effective gain towards RAS station: {:.1f}'.format(G_eff))

            results = pathprof.atten_path_fast(freqs[i],
                                               temperature,
                                               pressure,
                                               h_tg, h_rg,
                                               timepercent,
                                               hprof_cache,  # dict_like
                                               )
            Atten_ant[k, i] = results['L_b'][-1].value - G_eff.value
            #        _total_atten = results['L_b']  # L_b is the total attenuation,
            # considering all the factors.
            #        _fspl_atten = results['L_bfsg']  # considers only the free
            # space loss
            print(i,k,' atten value ',Atten_ant[k,i])
    attenpath_calc_end = time.time()
    print('Completed in %f sec.' % (attenpath_calc_end-attenpath_calc_start))
    return(Atten_ant, freqs)


if __name__ == '__main__':

    '''
    =======================================
    # Here input the parameters for the ITU-R 452-16 model	
    =======================================
    '''
    #	Units added in function

    freq = 180  # u.MHz
    omega = 0.  # u.percent  # fraction of path over sea
    temperature = 290.  # u.K
    pressure = 1013.  # u.hPa
    timepercent = 0.02  # u.percent  # see P.452 for explanation
    h_tg, h_rg = 3, 2.1  # u.m#height of the receiver and transmitter above gnd
    G_t, G_r = 0, 0  # cnv.dBi
    zone_t, zone_r = pathprof.CLUTTER.UNKNOWN, pathprof.CLUTTER.UNKNOWN
    # clutter type for transmitter/receiver
    # hprof_step = 10 * u.m
    diam = 2  # u.m # assumed diameter of transmitter

    '''
	=======================================
	Path attenuation additional inputs
	=======================================
	'''

    N_freqs = 3  # no. of channels
    freq_start, freq_end = 170, 180  # frequency range MHz
    hprof_step = 10000  # u.m  # resolution of solution

    # DTV Bickley site Perth 180MHz Seven
    tx_name, tx_lon, tx_lat = 'DTVseven', 116.084166666667, -32.0083333333333
    # site, lon_tx, lat_tx = 'DTVSeven', tx_lon*u.deg, tx_lat * u.deg

    # central reciever position for map calc.
    rx_name, rx_lon, rx_lat = 'Low_E4', 116.75, -26.835

    # map_size_x = 2.0 # deg
    # map_size_y = 2.0
    # map_size_lon, map_size_lat = map_size_x * u.deg, map_size_y * u.deg

    Atten_ant, freqs = calc_prop_atten(freq, omega, temperature, pressure,
                                       timepercent, h_tg, h_rg, G_t, G_r,
                                       zone_t, zone_r, N_freqs,
                                       freq_start, freq_end, hprof_step,
                                       tx_name, tx_lon, tx_lat, rx_name,
                                       rx_lon, rx_lat, diam)

    print(Atten_ant.shape)
    # Output results to file
    np.save('Attenuation_final.npy', Atten_ant)

    fig = plt.figure(figsize=(10, 10))
    # plt.semilogx(freqs,np.transpose(Atten_ant))
    plt.plot(freqs, np.transpose(Atten_ant))
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Total attenuation (dB)')
    plt.title('Station attenuation')
    plt.grid()
    plt.savefig('Station_attenuation.png', bbox_inches='tight')
