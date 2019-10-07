#!/usr/bin/env python
# coding: utf-8


## Original by Federico Di Vruno
## Alterations by DMF


from collections import namedtuple, OrderedDict
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from pycraf import pathprof
from pycraf import conversions as cnv
import xlrd
import time

pathprof.SrtmConf.set(download='missing', server='viewpano')
# allow download of missing SRTM data:

#%%
'''
===========================================
    Coordinates
    Set the coordinates of the transmitter.
===========================================
'''
# SKA_LOW offset 1km
#tx_lon, tx_lat = 116.747085, -26.82510333
#site, lon_tx, lat_tx = 'DTV1km', tx_lon*u.deg, tx_lat * u.deg 

# SKA_LOW offset 10km
#tx_lon, tx_lat = 116.747085, -26.743875
#site, lon_tx, lat_tx = 'DTV10km', tx_lon*u.deg, tx_lat * u.deg 

# SKA_LOW offset 100km
#tx_lon, tx_lat = 116.747085, -25.931551666
#site, lon_tx, lat_tx = 'DTV100km', tx_lon*u.deg, tx_lat * u.deg 

# DTV Bickley site Perth 180MHz Seven
tx_lon, tx_lat = 116.084166666667, -32.0083333333333
site, lon_tx, lat_tx = 'DTVSeven', tx_lon*u.deg, tx_lat * u.deg

'''
===========================================
    Sites of particular interest
     
===========================================
'''

Site = namedtuple('sitetype', ['name', 'coord', 'pixoff', 'color'])
sites = OrderedDict([
    # ID: tuple(Name, (lon, lat), Type, height, diameter, (xoff, yoff), color)
    #('Tx', Site('Tx', (u.Quantity(lon_tx).value, u.Quantity(lat_tx).value), (20, +30), 'k')),
    ('Low_E4', Site('Low_E4', (116.75, -26.835), (60, -20), 'k')),
    ('Trans', Site(site,(tx_lon,tx_lat),(60,-20), 'k')),        
    ])

for site_id, site in sites.items():
    if site.name == 'Low_E4':
        map_cen_lon = site.coord[0]
        map_cen_lat = site.coord[1]


'''
===========================================
    Antennas locations
     
===========================================
'''

# For LOW, get list of station name/locations from xlsx worksheet

# Antenna_locations = xlrd.open_workbook(r'C:\Users\F.Divruno\Dropbox (SKA)\Python_codes\SKA1_Mid_coordinates.xlsx')
# Antenna_locations = xlrd.open_workbook('/local/SKA/RFI/SKA1_Mid_coordinates.xlsx')
Antenna_locations = xlrd.open_workbook('/local/SKA/RFI/SKA1_LOW_coordinates_REV3.xlsx')

sheet_indx = 0
sheet = Antenna_locations.sheet_by_index(sheet_indx)
name_x = list()
lat_x = np.zeros(sheet.nrows-2)
long_x = np.zeros(sheet.nrows-2)
for i in range(2,sheet.nrows-1):
        name_x.append(sheet.cell_value(i,0))
        long_x[i-2] = sheet.cell_value(i,1)
        lat_x[i-2] = sheet.cell_value(i,2)
        


'''
=======================================
Start of inputs

Map size inputs
=======================================
'''

do_map_solution = None   #set to variable if want map solutions to be calculated and map 
                         #plots produced.

doplotAll = None   # assign any value e.g. 1 for plots and calculated maps to be 
                   # same area. Otherwise assign as None and set independently below.


if do_map_solution:

    # map 0.5deg squared, resolution 3.6asecs,
    map_size_x = 2.0
    map_size_y = 2.0
    map_size_lon, map_size_lat = map_size_x * u.deg, map_size_y * u.deg
    map_resolution = 0.001* u.deg ## maximum resolution allowed 0.1
    hprof_step = 1000 * u.m  ## overides map_resolution when given
    map_extent = [map_cen_lon-(0.5*map_size_x),map_cen_lat-(0.5*map_size_y),
                map_cen_lon+(0.5*map_size_x), map_cen_lat+(0.5*map_size_y)]
    # lonmin,latmin,lonmax,latmax

    if doplotAll:
        atten_map_size_lon = map_size_lon
        atten_map_size_lat = map_size_lat
        atten_map_resolution = map_resolution
        atten_hprof_step = hprof_step
        map_cen_lon = tx_lon
        map_cen_lat = tx_lat 
        print('Full distance plot')
    else:
        # Need to ensure this map is large enough to cover all receivers and transmitter 
        atten_map_size_lon = 12.0 * u.deg
        atten_map_size_lat = 12.0 * u.deg
        atten_map_resolution = 0.1* u.deg
        atten_hprof_step = 15000 * u.m  ## overides map_resolution when given
        print('Restricted plotting')


'''
=======================================
Attenuation calculation inputs
=======================================
'''


# Here input the parameters for the ITU-R 452-16 model

freq = 180 * u.MHz
omega = 0. * u.percent  # fraction of path over sea
temperature = 290. * u.K
pressure = 1013. * u.hPa
timepercent = 0.02 * u.percent  # see P.452 for explanation
h_tg, h_rg = 3 * u.m, 2.1 * u.m#height of the receiver and transmitter above gnd
G_t, G_r = 0 * cnv.dBi, 0 * cnv.dBi
zone_t, zone_r = pathprof.CLUTTER.UNKNOWN, pathprof.CLUTTER.UNKNOWN 
# clutter type for transmitter/receiver
# hprof_step = 10 * u.m


'''
=======================================
Path attenuation additional inputs
=======================================
'''

N_freqs = 100  # no. of channels
freq_start, freq_end = 170, 180  # frequency range MHz
hprof_step = 1000 * u.m  # resolution of solution



'''
======================================
End of inputs
======================================
'''



if do_map_solution:

    #%%
    '''
    ===========================================
         Get the map height information
         
    ===========================================
    '''

    terrain_start = time.time()
    print('Getting terrain map information...')

    # get the data for the terrain map of region around location above i.e. transmitter
    # covers area defined above
    # mostly just for getting the max/min for plotting the terrain below
    lons, lats, heightmap = pathprof.srtm_height_map(
        map_cen_lon*u.deg, map_cen_lat*u.deg,
        #lon_tx, lat_tx,
        map_size_lon, map_size_lat,
        map_resolution=map_resolution,
        hprof_step = hprof_step
        )

    terrain_end = time.time()
    print('Completed in %f sec.' %(terrain_end-terrain_start))



    #%%
    '''
    ===========================================
        Plot terrain information
         
    ===========================================
    '''


    _lons = lons.to(u.deg).value
    _lats = lats.to(u.deg).value
    _heightmap = heightmap.to(u.m).value

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_axes((0., 0., 1.0, 1.0))

    vmin,vmax = np.min(_heightmap),np.max(_heightmap) 

    # Produces terrain colormap and norm for plotting
    terrain_cmap,terrain_norm = pathprof.terrain_cmap_factory(sealevel=vmin,vmax=vmax)

    cim = ax.imshow(
                    _heightmap,
                    origin='lower', interpolation='nearest',
                    cmap=terrain_cmap, norm=terrain_norm,
                    vmin=vmin, vmax=vmax,
                    extent=(_lons[0], _lons[-1], _lats[0], _lats[-1]),
                    )

    ax.set_xlabel('Longitude [deg]')
    ax.set_ylabel('Latitude [deg]')
    plt.title('Terrain map')

    cbar = fig.colorbar(cim,fraction=0.046, pad=0.04)
    cbar.set_label('Height (m)', rotation=270)

    # Annotate the sites of interest in the map
    for site_id, site in sites.items():
        color = site.color
        color = 'k'
        ax.annotate(
            site.name, xy=site.coord, xytext=site.pixoff, 
            textcoords='offset points', color=color,
            arrowprops=dict(arrowstyle="->", color=color)
            )
        ax.scatter(site.coord[0], site.coord[1],marker='o', c='k')

    # set aspect ratio and limits of the image
    ax.set_aspect(abs(_lons[-1] - _lons[0]) / abs(_lats[-1] - _lats[0])) 
    ax.set_ylim([_lats[0],_lats[-1]])
    ax.set_xlim([_lons[0],_lons[-1]])


    # Place a marker in each of the SKA antennas in the map
    # For LOW these are station positions
    for i in range(len(long_x)):
        ax.scatter(long_x[i], lat_x[i],marker='o', c='k')

    plt.savefig('terrain.png', bbox_inches='tight')
    print('Terrain map completed')





    '''
    ===========================================
        Calculate the attenuation map
         
    ===========================================
    '''

    attenmap_start = time.time()
    print('Calculating attenuation map...')

    '''
    # Here input the parameters for the ITU-R 452-16 model

    freq = 180 * u.MHz
    omega = 0. * u.percent  # fraction of path over sea
    temperature = 290. * u.K
    pressure = 1013. * u.hPa
    timepercent = 0.02 * u.percent  # see P.452 for explanation
    h_tg, h_rg = 3 * u.m, 2.1 * u.m#height of the receiver and transmitter above gnd
    G_t, G_r = 0 * cnv.dBi, 0 * cnv.dBi
    zone_t, zone_r = pathprof.CLUTTER.UNKNOWN, pathprof.CLUTTER.UNKNOWN 
    # clutter type for transmitter/receiver
    # hprof_step = 10 * u.m
    '''


    ## Faster approach

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
    _total_atten = results['L_b']  # L_b is the total attenuation, considering all the factors.
    #print(_total_atten.shape,_total_atten[0])
    _fspl_atten = results['L_bfsg']  # considers only the free space loss
     
    attenmap_end = time.time()
    print('Completed in %f sec.' %(attenmap_end-attenmap_start))


    '''
    ===========================================
        Plot the resulting attenuation map
        
    ===========================================
    '''

    # Plot the results selected
    vmin, vmax = 90, 200 # Max and min scale

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_axes((0., 0.1, 1.0, 0.8))
    # cbax = fig.add_axes((0.3, 0., 0.4, .02))

    if doplotAll:
        ax.set_aspect(abs(_lons[-1] - _lons[0]) / abs(_lats[-1] - _lats[0]))
        nextent = (_lons[0], _lons[-1], _lats[0], _lats[-1])
        _total_atten_fin = _total_atten
        _lons_fin = _lons
        _lats_fin = _lats
    else:
        ax.set_aspect(abs(map_extent[2]-map_extent[0]) / abs(map_extent[3]-map_extent[1]))
        #nextent = (map_extent[0], map_extent[2], map_extent[1], map_extent[3])
        lon_ind = np.where((_lons>map_extent[0]) & (_lons<map_extent[2]))[0]
        lat_ind = np.where((_lats>map_extent[1]) & (_lats<map_extent[3]))[0]
        _lons_fin = _lons[lon_ind]
        _lats_fin = _lats[lat_ind]
        nextent = (_lons_fin[0], _lons_fin[-1], _lats_fin[0], _lats_fin[-1])
        _total_atten_fin = _total_atten[np.ix_(lat_ind,lon_ind)]
        #print(_total_atten_fin) #x[np.ix_(row_indices,col_indices)]
    # map_extent(lonmin,latmin,lonmax,latmax)

    cim = ax.imshow(_total_atten_fin.to(cnv.dB).value,
                    origin='lower', interpolation='nearest', cmap='inferno_r',
                    vmin=vmin, vmax=vmax,
                    extent = nextent
                    #extent=(_lons[0], _lons[-1], _lats[0], _lats[-1]),
                    )

    # cbar = fig.colorbar(cim, cax=cbax, orientation='horizontal', )
    cbar = fig.colorbar(cim, orientation='horizontal',fraction=0.046, pad=0.08)


    cbar.set_label(r'Path propagation loss (dB)', color='black')

    '''
    cbax.xaxis.set_label_position('top')
    for t in cbax.xaxis.get_major_ticks():
        t.tick2On = True
        t.label2On = True
    '''

    ctics = np.arange(vmin, vmax, 10)
    cbar.set_ticks(ctics)

    ax.set_xlabel('Longitude [deg]')
    ax.set_ylabel('Latitude [deg]')
    ax.set_autoscale_on(False)
    #cbax.set_autoscale_on(False)


    # Annotate the coordinates of the interest sites and searches for the attenuation levels.

    lat_mesh, lon_mesh = np.meshgrid(_lats_fin,_lons_fin) # hace un mesh para buscar los puntos
    # Makes a mesh to find the points
    for site_id, site in sites.items():
        color = site.color
        color = 'b'
        aux = abs(lat_mesh-site.coord[1])+abs(lon_mesh-site.coord[0])
        i,j = np.unravel_index(aux.argmin(),aux.shape)
        ax.annotate(
            site.name + ' att: ' + str(_total_atten_fin.to(cnv.dB).value[j,i])[0:6] + ' dB', xy=site.coord, xytext=site.pixoff, 
            textcoords='offset points', color=color,
            arrowprops=dict(arrowstyle="->", color=color)
            )
        ax.scatter(site.coord[0], site.coord[1],marker='o', c='b') 

    for i in range(len(long_x)): # Puts a mark in every place there is an antenna
        ax.scatter(long_x[i], lat_x[i],marker='o', c='w')

    plt.title('Attenuation map, Freq = '+ str(freq))
        
    # ax.xaxis.tick_top()
    # ax.xaxis.set_label_position('top')
    plt.tight_layout() 
    # plt.show()
    plt.savefig('Path_attenuation.png', bbox_inches='tight')

    #%%

    '''
    ===========================================
        Plot the terrain map with attenuation contours
        Also add the contours for Free Space Path Loss. 
        
    ===========================================
    '''

    '''Shouldn't need as repeat of terrain map
    _lons = lons.to(u.deg).value
    _lats = lats.to(u.deg).value
    _heightmap = heightmap.to(u.m).value
    '''

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_axes((0., 0.1, 1.0, 0.8))
    vmin,vmax = np.min(_heightmap),np.max(_heightmap) 

    # terrain_cmap,terrain_norm = pathprof.terrain_cmap_factory(sealevel=vmin,vmax=vmax)

    cim = ax.imshow(
        _heightmap,
        origin='lower', interpolation='nearest',
        cmap=terrain_cmap, norm=terrain_norm,
        vmin=vmin, vmax=vmax,
        extent = nextent
        #extent=(_lons[0], _lons[-1], _lats[0], _lats[-1]),
        )

    cbar = fig.colorbar(cim, fraction=0.046, pad=0.04)
    cbar.set_label('Height (m)', rotation=270)

    # _fspl_atten = results['L_bfsg'] 

    ax.set_xlabel('Longitude [deg]')
    ax.set_ylabel('Latitude [deg]')
    plt.title('Terrain map with attenuation contours')

    ax.contour(_total_atten_fin.to(cnv.dB).value, levels=[95,105,115,125],
               linestyles='-',
               origin='lower',
               extent = nextent,
               #extent=(_lons[0], _lons[-1], _lats[0], _lats[-1]),
               alpha=1)

    '''
    ax.contour(_fspl_atten.to(cnv.dB).value, levels=[100],
               colors=['red'], linestyles='-',
               origin='lower',
               extent=(_lons[0], _lons[-1], _lats[0], _lats[-1]),
               alpha=1)

    ax.contourf(_total_atten.to(cnv.dB).value, levels=[0,90],
               colors='b', linestyles='-',
               origin='lower',
               extent=(_lons[0], _lons[-1], _lats[0], _lats[-1]),
               alpha=0.2)

    ax.contourf(_total_atten.to(cnv.dB).value, levels=[120,130],
               colors='r', linestyles='-',
               origin='lower',
               extent=(_lons[0], _lons[-1], _lats[0], _lats[-1]),
               alpha=0.4)

    '''

    # Position the SKA antennas in the map
    for i in range(len(long_x)):
        ax.scatter(long_x[i], lat_x[i],marker='o', c='k')      

    if doplotAll:    
        ax.set_ylim([_lats[0],_lats[-1]])
        ax.set_xlim([_lons[0],_lons[-1]])
    else:
        ax.set_ylim(map_extent[1],map_extent[3])
        ax.set_xlim(map_extent[0],map_extent[2])
    plt.tight_layout()
    # plt.show()
    plt.savefig('Pathloss_terrain.png', bbox_inches='tight')


    print('Finished plotting attenuation maps')



    '''
    ===========================================
         Find the attenuation at each position of SKA antennas using the map info
    ===========================================
    '''

    attenmapcalc_start = time.time()
    print('Calculating attenuation at each station from map...')

    lat_mesh, lon_mesh = np.meshgrid(_lats,_lons) # mesh in lats and longs

    for k in range(len(long_x)-1):
        
        aux = abs(lat_mesh-lat_x[k])+abs(lon_mesh-long_x[k])
        i,j = np.unravel_index(aux.argmin(),aux.shape)
        ## print('Antenna: %s - Att: %.2f'%(name_x[k],(_total_atten.to(cnv.dB).value[j,i])))

    #%%
    attenmapcalc_end = time.time()
    print('Completed in %f sec.' %(attenmapcalc_end-attenmapcalc_start))








'''
===========================================
    Find the attenuation at each position of SKA antennas using path info
===========================================
'''

attenpath_calc_start = time.time()
print('Calculating frequency dependent attenuation for each station from path...')

'''
N_freqs = 100  # no. of channels
freq_start, freq_end = 170, 180  # frequency range MHz
hprof_step = 1000 * u.m  # resolution of solution
'''

N_ant = len(long_x)-1
N_freqs = 3
# freqs = np.logspace(np.log10(350),np.log10(15400),N_freqs)*u.MHz
freqs = np.logspace(np.log10(freq_start),np.log10(freq_end),N_freqs)*u.MHz
print(freqs)
                                                                            

Atten_ant = np.zeros([N_ant,N_freqs])
results=[]
hprof_cache =[]

for k in range(N_ant):
    hprof_cache = pathprof.height_path_data(lon_tx,
                                            lat_tx,
                                            long_x[k]*u.deg,
                                            lat_x[k]*u.deg,
                                            hprof_step)
    for i in range(N_freqs):
        print(k,i)
        results = pathprof.atten_path_fast(freqs[i],
                                        temperature,
                                        pressure,
                                        h_tg, h_rg,
                                        timepercent,
                                        hprof_cache,  # dict_like
                                        )
        Atten_ant[k,i] = results['L_b'][-1].value #gets the last value of the attenuation path.
#        _total_atten = results['L_b']  # L_b is the total attenuation, considering all the factors.
#        _fspl_atten = results['L_bfsg']  # considers only the free space loss
        ## print(i,k,' atten value ',Atten_ant[k,i]) 
        
print(Atten_ant.shape)
# Output results to file
np.save('Attenuation_final.npy', Atten_ant)


fig = plt.figure(figsize=(10, 10))
# plt.semilogx(freqs,np.transpose(Atten_ant))
plt.plot(freqs,np.transpose(Atten_ant))
plt.xlabel('Frequency (MHz)')
plt.ylabel('Total attenuation (dB)')
plt.title('Station attenuation')
plt.grid()
plt.savefig('Station_attenuation.png',bbox_inches='tight')

attenpath_calc_end = time.time()
print('Completed in %f sec.' %(attenpath_calc_end-attenpath_calc_start))





## Check single path calculation ##

attenpath_calc_start = time.time()
print('Calculating frequency dependent attenuation for each station from single path...')

Atten_ant = np.zeros([N_ant,N_freqs])
Atten_ant_all = np.zeros([N_ant,N_freqs])
for k in range(N_ant):
    for i in range(N_freqs):

        freq_ras = freqs[i]
        pprop_fl_ras = pathprof.PathProp(
            freq_ras,temperature, pressure,
            lon_tx, lat_tx,
            long_x[k]*u.deg,lat_x[k]*u.deg,
            h_tg, h_rg,
            hprof_step,
            timepercent,
            zone_t=zone_t, zone_r=zone_r,
            )

        G_eff = -4.7*cnv.dBi
        tot_loss = pathprof.loss_complete(pprop_fl_ras, G_eff, G_r)
        (L_b0p, L_bd, L_bs, L_ba, L_b, L_b_corr, L) = tot_loss
        Atten_ant[k,i] = L_b.value
        Atten_ant_all[k,i] = L.value




print(Atten_ant.shape)
# Output results to file
#np.save('Attenuation_final.npy', Atten_ant)


fig = plt.figure(figsize=(10, 10))
# plt.semilogx(freqs,np.transpose(Atten_ant))
plt.plot(freqs,np.transpose(Atten_ant))
plt.plot(freqs,np.transpose(Atten_ant_all))
plt.xlabel('Frequency (MHz)')
plt.ylabel('Total attenuation (dB)')
plt.title('Station attenuation')
plt.grid()
plt.savefig('Station_attenuation_single.png',bbox_inches='tight')





attenpath_calc_end = time.time()
print('Completed in %f sec.' %(attenpath_calc_end-attenpath_calc_start))



'''
print('L_b0p:    {0.value:5.2f} {0.unit} - Free-space loss'.format(L_b0p))
print('L_bd:     {0.value:5.2f} {0.unit} - Basic transmission loss associated with diffraction'.format(L_bd))
print('L_bs:     {0.value:5.2f} {0.unit} - Tropospheric scatter loss'.format(L_bs))
print('L_ba:     {0.value:5.2f} {0.unit} - Ducting/layer reflection loss'.format(L_ba))
print('L_b:      {0.value:5.2f} {0.unit} - Complete path propagation loss'.format(L_b))
print('L_b_corr: {0.value:5.2f} {0.unit} - As L_b but with clutter correction'.format(L_b_corr))
print('L:        {0.value:5.2f} {0.unit} - As L_b_corr but with gain correction'.format(L))
'''



