# coding: utf-8

import logging

import matplotlib.pyplot as plt
import numpy
from astropy import constants
from astropy import units as u
from astropy.coordinates import SkyCoord, EarthLocation

from processing_library.util.array_functions import average_chunks2
from processing_library.util.coordinate_support import xyz_to_uvw, skycoord_to_lmn
from wrappers.arlexecute.simulation.configurations import create_named_configuration


def generate_DTV(frequency, times, power=50e3, timevariable=False):
    """ Calculate sqrt(power) as a function of time and frequency
    
    :param frequency: (sample frequencies)
    :param times: sample times (s)
    :param power: DTV emitted power W
    :return:
    """
    nchan = len(frequency)
    ntimes = len(times)
    shape = [ntimes, nchan]
    sshape = [ntimes, nchan // 2]
    bchan = nchan // 4
    echan = 3 * nchan // 4
    amp = numpy.sqrt(2.0) * power / (max(frequency) - min(frequency))
    print("RMS DTV power spectral density = %g W/Hz" % amp)
    signal = numpy.zeros(shape, dtype='complex')
    if timevariable:
        sshape = [ntimes, nchan // 2]
        signal[:, bchan:echan] += numpy.random.normal(0.0, numpy.sqrt(amp), sshape) \
                                + 1j * numpy.random.normal(0.0, numpy.sqrt(amp), sshape)
    else:
        sshape = [nchan // 2]
        signal[:, bchan:echan] += (numpy.random.normal(0.0, numpy.sqrt(amp), sshape)
                                + 1j * numpy.random.normal(0.0, numpy.sqrt(amp), sshape))[numpy.newaxis, ...]

    return signal


def add_noise(visibility, bandwidth, int_time):
    """Determine noise rms per visibility
    
    :returns: visibility with noise added
    """
    # The specified sensitivity (effective area / T_sys) is roughly 610 m ^ 2 / K in the range 160 - 200MHz
    # sigma_vis = 2 k T_sys / (area * sqrt(tb)) = 2 k 512 / (610 * sqrt(tb)
    sens = 610
    k_b = 1.38064852e-23
    bt = bandwidth * int_time
    sigma = 2 * 1e26 * k_b / ((sens / 512) * (numpy.sqrt(bt)))
    print("RMS noise per sample = %g Jy" % sigma)
    sshape = visibility.shape
    visibility += numpy.random.normal(0.0, sigma, sshape) + 1j * numpy.random.normal(0.0, sigma, sshape)
    return visibility


def create_propagators(config, interferer, frequency, attenuation=1e-9):
    """ Create a set of propagators

    :return: Array
    """
    nchannels = len(frequency)
    nants = len(config.data['names'])
    interferer_xyz = [interferer.geocentric[0].value, interferer.geocentric[1].value, interferer.geocentric[2].value]
    propagators = numpy.zeros([nants, nchannels], dtype='complex')
    for iant, ant_xyz in enumerate(config.xyz):
        vec = ant_xyz - interferer_xyz
        # This ignores the Earth!
        r = numpy.sqrt(vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2)
        k = 2.0 * numpy.pi * frequency / constants.c.value
        propagators[iant, :] = numpy.exp(- 1.0j * k * r) / r
    return propagators * attenuation


def calculate_rfi_at_station(propagators, emitter=None):
    """ Calculate the rfi at each station
    
    :param propagators:
    :param emitter:
    :return:
    """
    rfi_at_station = emitter[numpy.newaxis, :, :] * propagators[:, numpy.newaxis, :]
    return rfi_at_station

def calculate_station_fringe_rotation(ants_xyz, times, frequency, phasecentre, pole):
    # Time corresponds to hour angle
    
    uvw = xyz_to_uvw(ants_xyz, times, phasecentre.dec.rad)
    nants, nuvw = uvw.shape
    ntimes = len(times)
    uvw = uvw.reshape([nants, ntimes, 3])
    lmn = skycoord_to_lmn(phasecentre, pole)
    delay = numpy.dot(uvw, lmn)
    nchan = len(frequency)
    phase = numpy.zeros([nants, ntimes, nchan])
    for ant in range(nants):
        for chan in range(nchan):
            phase[ant, :, chan] = delay[ant] * frequency[chan] / constants.c.value
    return numpy.exp(2.0 * numpy.pi * 1j * phase)


def calculate_station_correlation_rfi(fringe_rotation, rfi_at_station):
    """ Calculate the correlation of the RFI
    
    :return: Correlation(nant, nants, ntimes, nchan] in Jy
    """
    phased_rotated_rfi_at_station = fringe_rotation * rfi_at_station
    nants, ntimes, nchan = fringe_rotation.shape
    correlation = numpy.zeros([nants, nants, ntimes, nchan], dtype='complex')

    for time in range(ntimes):
        for chan in range(nchannels):
            correlation[..., time, chan] = numpy.outer(phased_rotated_rfi_at_station[..., time, chan],
                                                       numpy.conjugate(phased_rotated_rfi_at_station[..., time, chan]))
    return correlation * 1e26


def calculate_averaged_correlation(correlation, channel_width=4, time_width=4):
    wts = numpy.ones(correlation.shape, dtype='float')
    return average_chunks2(correlation, wts, (channel_width, time_width))[0]


if __name__ == '__main__':
    log = logging.getLogger()
    print("Starting LOW low level RFI simulation")
    
    # arlexecute.set_client(use_dask=True, threads_per_worker=1, memory_limit=32 * 1024 * 1024 * 1024, n_workers=8,
    #                       local_dir=dask_dir)
    # print(arlexecute.client)
    # arlexecute.run(init_logging)
    
    # We create a graph to make the visibility. The parameter rmax determines the distance of the
    # furthest antenna/stations used. All over parameters are determined from this number.
    
    start_frequency = 1.06e8
    channel_spacing = 8e3
    nchannels = 500
    frequency = start_frequency + numpy.arange(nchannels) * channel_spacing
    channel_bandwidth = numpy.repeat(channel_spacing, nchannels)
    
    ntimes = 500
    integration_time = 0.25
    times = numpy.arange(ntimes) * integration_time
    
    phasecentre = SkyCoord(ra=+30.0 * u.deg, dec=-45.0 * u.deg, frame='icrs', equinox='J2000')
    pole = SkyCoord(ra=+0.0 * u.deg, dec=-90.0 * u.deg, frame='icrs', equinox='J2000')
    
    site = EarthLocation(lon="116.76444824", lat="-26.824722084", height=300.0)
    
    # Perth from Google for the moment
    perth = EarthLocation(lon="115.8605", lat="-31.9505", height=0.0)
    
    rmax = 3000.0
    low = create_named_configuration('LOWR3', rmax=rmax)
    low.data = low.data[::33]
    nants = len(low.names)
    print("There are %d stations" % nants)

    attenuation = 1e-9
    emitter = generate_DTV(frequency, times, power=50e3)

    plt.clf()
    plt.imshow(numpy.abs(emitter), origin='bottom')
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Time (s)')

    cbar = plt.colorbar()
    plt.title(("Gain towards Perth = %.1f dB" % (10 * numpy.log10(attenuation))))
    cbar.set_label('RFI at station (sqrt(Jy))', rotation=270)
    plt.tight_layout()
    plt.savefig("RFI_at_station.png")
    plt.show()

    # Calculate the propagators for signals from Perth to the stations in low
    # These are fixed in time but vary with frequency.
    propagators = create_propagators(low, perth, frequency=frequency, attenuation=attenuation)
    
    plt.clf()
    for i in range(0, propagators.shape[0], 33):
        plt.plot(frequency, numpy.angle(propagators[i, :]), '.', label=str(i))
    plt.title("Propagator phases, Perth to LOW, rmax = %.1f (m)" % rmax)
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Phase (rad)')
    plt.savefig("propagators.png")
    plt.show(block=False)
    
    # Now calculate the RFI at the stations
    rfi_at_station = calculate_rfi_at_station(propagators, emitter)

    plt.clf()
    plt.imshow(numpy.abs(rfi_at_station[0, ...]))
    plt.title('RFI at station amplitude, with respect to station %s' % low.names[0])
    plt.xlabel('Time')
    plt.ylabel('Channel')
    plt.savefig("rfi_at_station_phase.png")
    plt.show(block=False)

    plt.clf()
    plt.imshow(numpy.angle(rfi_at_station[0, ...]))
    plt.title('RFI at station phase, with respect to station %s' % low.names[0])
    plt.xlabel('Channel')
    plt.ylabel('Time')
    plt.savefig("rfi_at_station_amplitude.png")
    plt.show(block=False)

    # Station fringe rotation: shape [nants, ntimes, nchan] complex phasor to be applied to
    # reference to the pole.
    fringe_rotation = calculate_station_fringe_rotation(low.xyz, times, frequency, phasecentre, pole)
    plt.clf()
    plt.imshow(numpy.angle(fringe_rotation[:, 0, :]))
    plt.title('Fringe rotation phase,  %s' % low.names[0])
    plt.xlabel('Integration')
    plt.ylabel('Channel')
    plt.savefig("fringe_rotation.png")
    plt.show(block=False)
    
    # Correlation: [nants, nants, ntimes, nchan]
    correlation = calculate_station_correlation_rfi(fringe_rotation, rfi_at_station)
    
    print("RMS interference per sample = %g Jy" % numpy.std(numpy.real(correlation)))
    
    plt.clf()
    plt.imshow(numpy.angle(correlation[0, 10, :, :]))
    plt.title('Correlation phase, %s - %s ' % (low.names[0], low.names[1]))
    plt.xlabel('Channel')
    plt.ylabel('Integration')
    plt.savefig("correlation_phase.png")
    plt.show(block=False)
    
    plt.clf()
    plt.imshow(numpy.abs(correlation[0, 10, :, :]))
    plt.title('Correlation amplitude, %s - %s ' % (low.names[0], low.names[1]))
    plt.xlabel('Channel')
    plt.ylabel('Integration')
    plt.colorbar()
    plt.savefig("correlation_amplitude.png")
    plt.show(block=False)
    
    noise = False
    if noise:
        correlation = add_noise(correlation, channel_spacing, integration_time)
        print("RMS interference + noise per sample = %g Jy" % numpy.std(numpy.real(correlation)))

    # Integrate over time and frequency
    smeared_correlation = calculate_averaged_correlation(correlation[0, 10], channel_width=10, time_width=10)

    print("RMS smeared interference per sample = %g Jy" % numpy.std(numpy.real(smeared_correlation)))

    plt.clf()
    plt.imshow(numpy.abs(smeared_correlation))
    plt.title('Smeared correlation amplitude, %s - %s ' % (low.names[0], low.names[1]))
    plt.xlabel('Channel')
    plt.ylabel('Integration')
    plt.colorbar()
    plt.savefig("averaged_correlation_amplitude.png")
    plt.show(block=False)

    plt.clf()
    plt.imshow(numpy.angle(smeared_correlation))
    plt.title('Smeared correlation phase, %s - %s ' % (low.names[0], low.names[1]))
    plt.xlabel('Channel')
    plt.ylabel('Integration')
    plt.colorbar()
    plt.savefig("averaged_correlation_phase.png")
    plt.show(block=False)
