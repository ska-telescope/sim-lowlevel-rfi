# coding: utf-8

import logging

import matplotlib.pyplot as plt
import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord, EarthLocation

from processing_components.simulation.rfi import calculate_averaged_correlation, add_noise, simulate_rfi_block
from processing_library.util.array_functions import average_chunks
from wrappers.arlexecute.simulation.configurations import create_named_configuration


def add_ticks(plt, x, y):
    nx = x.shape[0]
    no_labels = 6  # how many labels to see on axis x
    step_x = int(nx / (no_labels - 1))  # step between consecutive labels
    x_positions = numpy.arange(0, nx, step_x)  # pixel count at label position
    x_labels = ["%.1f" % value for value in x[::step_x]]  # labels you want to see
    plt.xticks(x_positions, x_labels)
    
    ny = y.shape[0]
    no_labels = 6  # how many labels to see on axis y
    step_y = int(ny / (no_labels - 1))  # step between consecutive labels
    y_positions = numpy.arange(0, ny, step_y)  # pixel count at label position
    y_labels = ["%.1f" % value for value in y[::step_y]]  # labels you want to see
    plt.yticks(y_positions, y_labels)


if __name__ == '__main__':
    
    import matplotlib as mpl
    
    mpl.use('Agg')

    import argparse

    parser = argparse.ArgumentParser(description='Simulate DTV RFI')
    parser.add_argument('--context', type=str, default='DTV',
                        help='DTV')
    parser.add_argument('--rmax', type=float, default=1e3,
                        help='Maximum distance of station from centre (m)')
    parser.add_argument('--nchannels', type=int, default=1000, help='Number of channels')
    parser.add_argument('--station_skip', type=int, default=11, help='Decimate stations by this factor')
    parser.add_argument('--nthreads', type=int, default=1, help='Number of threads')
    parser.add_argument('--memory', type=int, default=8, help='Memory per worker (GB)')
    parser.add_argument('--nworkers', type=int, default=8, help='Number of workers')
    parser.add_argument('--show', type=str, default='False', help='Show images?')
    parser.add_argument('--ngroup_visibility', type=int, default=8, help='Process in visibility groups this large')
    parser.add_argument('--seed', type=int, default=18051955, help='Random number seed')
    parser.add_argument('--use_agg', type=str, default="True", help='Use Agg matplotlib backend?')
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
    channel_average = 16
    print("%d frequency channels of width %.g MHz" % (nchannels, sample_freq * 1e-6))
    
    ntimes = 1020
    integration_time = 0.5
    times = numpy.arange(ntimes) * integration_time
    time_average = 16
    print("%d integrations of duration %g (s)" % (ntimes, integration_time))
    
    declination = args.declination
    phasecentre = SkyCoord(ra=+30.0 * u.deg, dec=declination * u.deg, frame='icrs', equinox='J2000')
    
    # Perth from Google for the moment
    emitter_location = EarthLocation(lon="115.8605", lat="-31.9505", height=0.0)
    
    rmax = args.rmax
    low = create_named_configuration('LOWR3', rmax=rmax)
    nants = len(low.names)
    print("There are %d stations" % nants)
    station_skip = args.station_skip
    low.data = low.data[::station_skip]
    nants = len(low.names)
    print("There are %d stations after decimation" % nants)
    
    correlation, uvw = simulate_rfi_block(low, times, frequency, phasecentre, emitter_location=emitter_location,
                                          emitter_power=5e4, attenuation=1.0)
    
    print("RMS interference per sample = %g Jy" % numpy.std(numpy.real(correlation)))
    noise = False
    if noise:
        correlation = add_noise(correlation, sample_freq, integration_time)
        print("RMS interference + noise per sample = %g Jy" % numpy.std(numpy.real(correlation)))
    
    # Integrate the correlation over time and frequency
    averaged_frequency = numpy.array(average_chunks(frequency, numpy.ones_like(frequency), channel_average))[0]
    averaged_times = numpy.array(average_chunks(times, numpy.ones_like(times), time_average))[0]
    correlation_std = list()
    averaged_correlation_std = list()
    uvdistance = list()
    
    for ant2 in range(nants):
        for ant1 in range(ant2 + 1, nants):
            result = dict()
            averaged_correlation = calculate_averaged_correlation(correlation[ant2, ant1],
                                                                  channel_width=channel_average,
                                                                  time_width=time_average)
            averaged_correlation_std.append(numpy.std(numpy.real(averaged_correlation)))

            correlation_std.append(numpy.std(numpy.real(correlation[ant2, ant1])))

            offset = numpy.average(uvw[ant2,:,:] - uvw[ant1,:,:], axis=1)
            uvd = numpy.sqrt(offset[0]**2+offset[1]**2)
            uvdistance.append(uvd)
            
            if ant2 == 0:
                plt.clf()
                plt.imshow(numpy.abs(correlation[ant2, ant1,...]), origin='bottom')
                add_ticks(plt, 1e-6 * frequency, times)
                plt.title('Correlation %d %d, uv distance %.1f (m)' % (ant2, ant1, uvd))
                plt.ylabel('Time (s)')
                plt.xlabel('Frequency (MHz)')
                plt.colorbar()
                plt.savefig('correlation_amplitude_%d_%d.png' % (ant2, ant1))
                plt.show(block=False)
                plt.clf()
                plt.imshow(numpy.angle(correlation[ant2, ant1,...]), origin='bottom')
                add_ticks(plt, 1e-6 * frequency, times)
                plt.title('Correlation phase %d %d, uv distance %.1f (m)' % (ant2, ant1, uvd))
                plt.ylabel('Time (s)')
                plt.xlabel('Frequency (MHz)')
                plt.colorbar()
                plt.savefig('correlation_phase_%d_%d.png' % (ant2, ant1))
                plt.show(block=False)
                plt.clf()
                plt.imshow(numpy.abs(averaged_correlation), origin='bottom')
                add_ticks(plt, 1e-6 * averaged_frequency, averaged_times)
                plt.title('Correlation amplitude %d %d, uv distance %.1f (m)' % (ant2, ant1, uvd))
                plt.ylabel('Time (s)')
                plt.xlabel('Frequency (MHz)')
                plt.colorbar()
                plt.savefig('averaged_correlation_amplitude_%d_%d.png' % (ant2, ant1))
                plt.show(block=False)
                plt.clf()
                plt.imshow(numpy.angle(averaged_correlation), origin='bottom')
                add_ticks(plt, 1e-6 * averaged_frequency, averaged_times)
                plt.title('Correlation phase %d %d, uv distance %.1f (m)' % (ant2, ant1, uvd))
                plt.ylabel('Time (s)')
                plt.xlabel('Frequency (MHz)')
                plt.colorbar()
                plt.savefig('averaged_correlation_phase_%d_%d.png' % (ant2, ant1))
                plt.show(block=False)

    plt.clf()
    plt.semilogy(uvdistance, correlation_std, '.', color='blue', label='Non-averaged')
    plt.semilogy(uvdistance, averaged_correlation_std, '.', color='red', label='Averaged')
    plt.legend()
    plt.title('Correlation versus uv distance')
    plt.xlabel('UV distance (meters)')
    plt.ylabel('Correlation (Jy)')
    plt.savefig('correlation_vs_uv.png')
    plt.show(block=False)
