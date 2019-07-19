# coding: utf-8

import logging

import matplotlib.pyplot as plt
import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord, EarthLocation

from data_models.polarisation import PolarisationFrame
from data_models.memory_data_models import Skycomponent, SkyModel

from processing_library.image.operations import create_empty_image_like
from wrappers.serial.visibility.base import create_blockvisibility
from wrappers.serial.image.operations import show_image, qa_image, export_image_to_fits
from wrappers.serial.simulation.configurations import create_configuration_from_MIDfile
from wrappers.serial.imaging.base import create_image_from_visibility, advise_wide_field
from processing_library.util.coordinate_support import hadec_to_azel
from wrappers.arlexecute.visibility.base import copy_visibility
from wrappers.arlexecute.visibility.coalesce import convert_blockvisibility_to_visibility, \
    convert_visibility_to_blockvisibility

from processing_components.simulation.rfi import calculate_averaged_correlation, add_noise, simulate_rfi_block
from processing_library.util.array_functions import average_chunks
from wrappers.arlexecute.simulation.configurations import create_named_configuration

if __name__ == '__main__':
    
    import matplotlib as mpl
    
#    mpl.use('Agg')
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Simulate DTV RFI')
    parser.add_argument('--context', type=str, default='DTV',
                        help='DTV')
    parser.add_argument('--rmax', type=float, default=1e3,
                        help='Maximum distance of station from centre (m)')
    parser.add_argument('--nchannels', type=int, default=1000, help='Number of channels')
    parser.add_argument('--station_skip', type=int, default=33, help='Decimate stations by this factor')
    parser.add_argument('--nthreads', type=int, default=1, help='Number of threads')
    parser.add_argument('--memory', type=int, default=8, help='Memory per worker (GB)')
    parser.add_argument('--nworkers', type=int, default=8, help='Number of workers')
    parser.add_argument('--show', type=str, default='False', help='Show images?')
    parser.add_argument('--ngroup_visibility', type=int, default=8, help='Process in visibility groups this large')
    parser.add_argument('--seed', type=int, default=18051955, help='Random number seed')
    parser.add_argument('--use_agg', type=str, default="False", help='Use Agg matplotlib backend?')
    parser.add_argument('--declination', type=float, default=-45.0, help='Declination (degrees)')
    parser.add_argument('--integration_time', type=float, default=600.0, help="Integration time (s)")
    parser.add_argument('--time_range', type=float, nargs=2, default=[-6.0, 6.0], help="Hourangle range (hours")
    parser.add_argument('--time_chunk', type=float, default=1800.0, help="Time for a chunk (s)")
    parser.add_argument('--shared_directory', type=str, default='../../shared/',
                        help='Location of pointing files')
    
    args = parser.parse_args()
    print("Starting LOW low level RFI simulation")
    
    # arlexecute.set_client(use_dask=True, threads_per_worker=1, memory_limit=32 * 1024 * 1024 * 1024, n_workers=8,
    #                       local_dir=dask_dir)
    # print(arlexecute.client)
    # arlexecute.run(init_logging)
    
    # We create a graph to make the visibility. The parameter rmax determines the distance of the
    # furthest antenna/stations used. All over parameters are determined from this number.
    
    sample_freq = 3e4
    nchannels = args.nchannels
    frequency = 170.5e6 + numpy.arange(nchannels) * sample_freq
    channel_bandwidth = numpy.ones_like(frequency) * sample_freq
    channel_average = 16
    print("%d frequency channels of width %.g MHz" % (nchannels, sample_freq * 1e-6))
    
    ntimes = 1020
    integration_time = 0.5
    times = numpy.arange(ntimes) * integration_time
    time_average = 16
    print("%d integrations of duration %g (s)" % (ntimes, integration_time))
    
    averaged_frequency = numpy.array(average_chunks(frequency, numpy.ones_like(frequency), channel_average))[0]
    averaged_channel_bandwidth = numpy.ones_like(averaged_frequency) * sample_freq
    averaged_times = numpy.array(average_chunks(times, numpy.ones_like(times), time_average))[0]
    
    rmax = args.rmax
    low = create_named_configuration('LOWR3', rmax=rmax)
    nants = len(low.names)
    print("There are %d stations" % nants)
    station_skip = args.station_skip
    low.data = low.data[::station_skip]
    nants = len(low.names)
    print("There are %d stations after decimation" % nants)
    
    declination = args.declination
    phasecentre = SkyCoord(ra=+30.0 * u.deg, dec=declination * u.deg, frame='icrs', equinox='J2000')
    
    # We generate two BlockVisibility's, one for the raw data and one for the averaged data.
    
    bvis = create_blockvisibility(low, averaged_times, averaged_frequency,
                                           averaged_channel_bandwidth=channel_bandwidth,
                                           phasecentre=phasecentre,
                                           polarisation_frame=PolarisationFrame("stokesI"),
                                           zerow=False)
    
    # Perth from Google for the moment
    emitter_location = EarthLocation(lon="115.8605", lat="-31.9505", height=0.0)
    
    correlation, uvw = simulate_rfi_block(low, times, frequency, phasecentre, emitter_location=emitter_location,
                                          emitter_power=5e4, attenuation=1.0)
    
    print("RMS interference per sample = %g Jy" % numpy.std(numpy.real(correlation)))
    noise = False
    if noise:
        correlation = add_noise(correlation, sample_freq, integration_time)
        print("RMS interference + noise per sample = %g Jy" % numpy.std(numpy.real(correlation)))
    
    for ant2 in range(nants):
        for ant1 in range(ant2, nants):
            averaged_correlation = calculate_averaged_correlation(correlation[ant2, ant1],
                                                                  channel_width=channel_average,
                                                                  time_width=time_average)
            bvis.data['vis'][:,ant2,ant1,:,0] = averaged_correlation
            bvis.data['vis'][:,ant1,ant2,:,0] = numpy.conjugate(averaged_correlation)
            
    plt.clf()
    uvdist = numpy.sqrt(bvis.uvw[...,0]**2 + bvis.uvw[...,1]**2).flatten()
    plt.plot(uvdist.flatten(),
             numpy.std(numpy.abs(bvis.vis[:,:,:,:,0]), axis=3).flatten(), '.')
    plt.title('Interference vs uv distance')
    plt.xlabel('UV distance (m)')
    plt.ylabel('Amplitude (Jy)')
    plt.show(block=False)