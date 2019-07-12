# coding: utf-8

import logging

import matplotlib.pyplot as plt
import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord, EarthLocation

from processing_library.util.array_functions import average_chunks
from rfi_simulation import create_propagators, calculate_averaged_correlation, calculate_rfi_at_station, \
    calculate_station_correlation_rfi, calculate_station_fringe_rotation, generate_DTV, add_noise
from wrappers.arlexecute.simulation.configurations import create_named_configuration


def add_ticks(plt, x, y):
    nx = x.shape[0]
    no_labels = 6  # how many labels to see on axis x
    step_x = int(nx / (no_labels - 1))  # step between consecutive labels
    x_positions = numpy.arange(0, nx, step_x)  # pixel count at label position
    x_labels = x[::step_x]  # labels you want to see
    plt.xticks(x_positions, x_labels)

    ny = y.shape[0]
    no_labels = 6  # how many labels to see on axis y
    step_y = int(ny / (no_labels - 1))  # step between consecutive labels
    y_positions = numpy.arange(0, ny, step_y)  # pixel count at label position
    y_labels = y[::step_y]  # labels you want to see
    plt.yticks(y_positions, y_labels)


if __name__ == '__main__':
    
    import matplotlib as mpl
    mpl.use('Agg')
    
    log = logging.getLogger()
    print("Starting LOW low level RFI simulation")
    
    # arlexecute.set_client(use_dask=True, threads_per_worker=1, memory_limit=32 * 1024 * 1024 * 1024, n_workers=8,
    #                       local_dir=dask_dir)
    # print(arlexecute.client)
    # arlexecute.run(init_logging)
    
    # We create a graph to make the visibility. The parameter rmax determines the distance of the
    # furthest antenna/stations used. All over parameters are determined from this number.
    
    sample_freq = 3e4
    nchannels = 1000
    frequency = 170.5e6 + numpy.arange(nchannels) * sample_freq
    channel_average = 4
    print("%d frequency channels of width %.g MHz" % (nchannels, sample_freq*1e-6))
    
    ntimes = 1020
    integration_time = 0.5
    times = numpy.arange(ntimes) * integration_time
    time_average = 4
    print("%d integrations of duration %g (s)" % (ntimes, integration_time))
    
    phasecentre = SkyCoord(ra=+30.0 * u.deg, dec=-45.0 * u.deg, frame='icrs', equinox='J2000')
    pole = SkyCoord(ra=+0.0 * u.deg, dec=-90.0 * u.deg, frame='icrs', equinox='J2000')
    
    site = EarthLocation(lon="116.76444824", lat="-26.824722084", height=300.0)
    
    # Perth from Google for the moment
    perth = EarthLocation(lon="115.8605", lat="-31.9505", height=0.0)
    
    rmax = 3000.0
    low = create_named_configuration('LOWR3', rmax=rmax)
    antskip = 7
    low.data = low.data[::antskip]
    nants = len(low.names)
    print("There are %d stations" % nants)
    
    # Calculate the power spectral density of the DTV station: Watts/Hz
    emitter = generate_DTV(frequency, times, power=50e3, timevariable=False)
    
    plt.clf()
    plt.imshow(numpy.abs(emitter), origin='bottom')
    add_ticks(plt, 1e-6*frequency, times)
    plt.title('DTV Power Spectral Density')
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Time (s)')
    cbar = plt.colorbar()
    cbar.set_label('Emission (Watts/Hz)', rotation=270)
    plt.tight_layout()
    plt.savefig("Emission.png")
    plt.show(block=False)
    
    # Calculate the propagators for signals from Perth to the stations in low
    # These are fixed in time but vary with frequency. The ad hoc attenuation
    # is set to produce signal roughly equal to noise at LOW
    attenuation = 3e-4
    propagators = create_propagators(low, perth, frequency=frequency, attenuation=attenuation)
    
    plt.clf()
    for i in range(0, propagators.shape[0], 33):
        plt.plot(frequency, numpy.angle(propagators[i, :]), '-', label=str(i))
    plt.title("Propagator phases, Perth to LOW, rmax = %.1f (m)" % rmax)
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Phase (rad)')
    plt.savefig("propagators.png")
    plt.show(block=False)
    
    # Now calculate the RFI at the stations, based on the emitter and the propagators
    rfi_at_station = calculate_rfi_at_station(propagators, emitter)
    
    plt.clf()
    plt.imshow(numpy.abs(rfi_at_station[0, ...]), origin='bottom')
    add_ticks(plt, times, 1e-6*frequency)
    plt.title('RFI at station, amplitude, with respect to station %s' % low.names[0])
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (MHz)')
    plt.savefig("rfi_at_station_phase.png")
    plt.show(block=False)
    
    plt.clf()
    plt.imshow(numpy.angle(rfi_at_station[0, ...]))
    add_ticks(plt, 1e-6*frequency, times)
    plt.title('RFI at station, phase, with respect to station %s' % low.names[0])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Time')
    plt.savefig("rfi_at_station_amplitude.png")
    plt.show(block=False)
    
    # Station fringe rotation: shape [nants, ntimes, nchan] complex phasor to be applied to
    # reference to the pole.
    fringe_rotation, uvw = calculate_station_fringe_rotation(low.xyz, times, frequency, phasecentre, pole)
    plt.clf()
    plt.imshow(numpy.angle(fringe_rotation[1, :, :]), origin='bottom')
    add_ticks(plt, 1e-6 * frequency, times)
    plt.title('Fringe rotation phase, %s' % low.names[1])
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Times (s)')
    plt.savefig("fringe_rotation.png")
    plt.show(block=False)
    
    # Calculate the rfi correlationm using the fringe rotation and the rfi at the station
    # [nants, nants, ntimes, nchan]
    correlation = calculate_station_correlation_rfi(fringe_rotation, rfi_at_station)
    print("RMS interference per sample = %g Jy" % numpy.std(numpy.real(correlation)))
    noise = False
    if noise:
        correlation = add_noise(correlation, sample_freq, integration_time)
        print("RMS interference + noise per sample = %g Jy" % numpy.std(numpy.real(correlation)))

    for ant in [1]:
        plt.clf()
        plt.imshow(numpy.angle(correlation[0, ant, :, :]), origin='bottom')
        add_ticks(plt, 1e-6 * frequency, times)
        plt.title('Correlation phase, %s - %s ' % (low.names[0], low.names[ant]))
        plt.xlabel('Frequency (MHz)')
        plt.ylabel('Time (s)')
        plt.savefig("correlation_phase.png")
        plt.show(block=False)
        
        plt.clf()
        plt.imshow(numpy.abs(correlation[0, ant, :, :]), origin='bottom')
        add_ticks(plt, 1e-6 * frequency, times)
        plt.title('Correlation amplitude, %s - %s ' % (low.names[0], low.names[ant]))
        plt.xlabel('Frequency (MHz)')
        plt.ylabel('Time (s)')
        plt.colorbar()
        plt.savefig("correlation_amplitude.png")
        plt.show(block=False)
    
    # Integrate the correlation over time and frequency
    if channel_average > 1 or time_average > 1:
        correlation_std = list()
        uvdistance = list()
    
        for ant2 in range(nants):
            for ant1 in range(ant2+1, nants):
                result = dict()
                averaged_correlation = calculate_averaged_correlation(correlation[ant2, ant1], channel_width=channel_average,
                                                                  time_width=time_average)
                corr = numpy.std(numpy.real(averaged_correlation))
                correlation_std.append(corr)
                
                offset = numpy.average(uvw[ant2,:,:] - uvw[ant1,:,:], axis=1)
                uvd = numpy.sqrt(offset[0]**2+offset[1]**2)
                uvdistance.append(uvd)
            
        plt.clf()
        plt.plot(uvdistance, correlation_std, '.')
        plt.title('Correlation versus uv distance')
        plt.xlabel('UV distance (meters)')
        plt.ylabel('Correlation (Jy)')
        plt.savefig('correlation_vs_uv.png')
        plt.show(block=False)
        exit()

        for ant in [1, nants // 2, nants - 1]:
            averaged_correlation = calculate_averaged_correlation(correlation[0, ant],
                                                                  channel_width=channel_average,
                                                                  time_width=time_average)

            print("RMS averaged interference per sample = %g Jy" % numpy.std(numpy.real(averaged_correlation)))

            averaged_times = numpy.array(average_chunks(times, numpy.ones_like(times), time_average))[0]
            averaged_frequency = numpy.array(average_chunks(frequency, numpy.ones_like(frequency), channel_average))[0]
            
            plt.clf()
            plt.imshow(numpy.abs(averaged_correlation), origin='bottom')
            add_ticks(plt, 1e-6 * averaged_frequency, averaged_times)
            plt.title('Averaged correlation amplitude, %s - %s ' % (low.names[0], low.names[ant]))
            plt.xlabel('Frequency (MHz)')
            plt.ylabel('Time (s)')
            plt.colorbar()
            plt.savefig("averaged_correlation_amplitude.png")
            plt.show(block=False)
        
            plt.clf()
            plt.imshow(numpy.angle(averaged_correlation), origin='bottom')
            add_ticks(plt, 1e-6 * averaged_frequency, averaged_times)
            plt.title('Averaged correlation phase, %s - %s ' % (low.names[0], low.names[ant]))
            plt.xlabel('Frequency (MHz)')
            plt.ylabel('Time (s)')
            plt.colorbar()
            plt.savefig("averaged_correlation_phase.png")
            plt.show(block=False)
            
