import numpy
from astropy import constants

from processing_library.util.array_functions import average_chunks2
from processing_library.util.coordinate_support import xyz_to_uvw, skycoord_to_lmn

def generate_DTV(frequency, times, power=50e3, timevariable=False, frequency_variable=False):
    """ Calculate sqrt(power) as a function of time and frequency

    :param frequency: (sample frequencies)
    :param times: sample times (s)
    :param power: DTV emitted power W
    :return:
    """
    nchan = len(frequency)
    ntimes = len(times)
    shape = [ntimes, nchan]
    bchan = nchan // 4
    echan = 3 * nchan // 4
    amp = power / (max(frequency) - min(frequency))
    print("RMS DTV power spectral density = %g W/Hz" % amp)
    signal = numpy.zeros(shape, dtype='complex')
    if timevariable:
        if frequency_variable:
            sshape = [ntimes, nchan // 2]
            signal[:, bchan:echan] += numpy.random.normal(0.0, numpy.sqrt(amp/2.0), sshape) \
                                    + 1j * numpy.random.normal(0.0, numpy.sqrt(amp/2.0), sshape)
        else:
            sshape = [ntimes]
            signal[:, bchan:echan] += numpy.random.normal(0.0, numpy.sqrt(amp/2.0), sshape) \
                                    + 1j * numpy.random.normal(0.0, numpy.sqrt(amp/2.0), sshape)
    else:
        if frequency_variable:
            sshape = [nchan // 2]
            signal[:, bchan:echan] += (numpy.random.normal(0.0, numpy.sqrt(amp/2.0), sshape)
                                   + 1j * numpy.random.normal(0.0, numpy.sqrt(amp/2.0), sshape))[numpy.newaxis, ...]
        else:
            signal[:, bchan:echan] = amp
    
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
    rfi_at_station[numpy.abs(rfi_at_station)<1e-15] = 0.
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
    phase[...] -= phase[0, :, :][numpy.newaxis,...]
    return numpy.exp(2.0 * numpy.pi * 1j * phase), uvw


def calculate_station_correlation_rfi(fringe_rotation, rfi_at_station):
    """ Calculate the correlation of the RFI

    :return: Correlation(nant, nants, ntimes, nchan] in Jy
    """
    phased_rotated_rfi_at_station = fringe_rotation * rfi_at_station
    nants, ntimes, nchan = fringe_rotation.shape
    correlation = numpy.zeros([nants, nants, ntimes, nchan], dtype='complex')
    
    for time in range(ntimes):
        for chan in range(nchan):
            correlation[..., time, chan] = numpy.outer(phased_rotated_rfi_at_station[..., time, chan],
                                                       numpy.conjugate(phased_rotated_rfi_at_station[..., time, chan]))
    return correlation * 1e26


def calculate_averaged_correlation(correlation, channel_width, time_width):
    wts = numpy.ones(correlation.shape, dtype='float')
    return average_chunks2(correlation, wts, (channel_width, time_width))[0]
