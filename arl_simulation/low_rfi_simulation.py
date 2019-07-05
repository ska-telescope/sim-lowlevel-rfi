# coding: utf-8

import logging

import matplotlib.pyplot as plt
import numpy
from astropy import constants
from astropy import units as u
from astropy.coordinates import SkyCoord, EarthLocation

from processing_library.util.coordinate_support import xyz_to_uvw, skycoord_to_lmn
from wrappers.arlexecute.simulation.configurations import create_named_configuration


def create_propagators(config, interferer: EarthLocation, frequency, **kwargs):
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
    
    return propagators


def calculate_interferer_fringe(propagators):
    nants, nchannels = propagators.shape
    # Now calculate the interferer propagator fringe. This is static in time.
    interferer_fringe = numpy.zeros([nants, nants, nchannels], dtype='complex')
    for chan in range(nchannels):
        interferer_fringe[..., chan] = numpy.outer(propagators[..., chan],
                                                   numpy.conjugate(propagators[..., chan]))
    return interferer_fringe

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
            phase[ant,:,chan] = delay[ant]* frequency[chan] / constants.c.value
    return numpy.exp(2.0 * numpy.pi * 1j * phase)

def calculate_station_correlation(fringe_rotation):
    nants, ntimes, nchan = fringe_rotation.shape
    correlation = numpy.zeros([nants, nants, ntimes, nchan], dtype='complex')
    for time in range(ntimes):
        for chan in range(nchannels):
            correlation[...,time,chan] = numpy.outer(fringe_rotation[...,time,chan],
                                                     numpy.conjugate(fringe_rotation[...,time,chan]))
    return correlation


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
    nchannels = 128
    frequency = start_frequency + numpy.arange(nchannels) * channel_spacing
    channel_bandwidth = numpy.repeat(channel_spacing, nchannels)
    
    ntimes = 100
    integration_time = 0.25
    times = numpy.arange(ntimes) * integration_time
    
    phasecentre = SkyCoord(ra=+30.0 * u.deg, dec=-45.0 * u.deg, frame='icrs', equinox='J2000')
    pole = SkyCoord(ra=+0.0 * u.deg, dec=-90.0 * u.deg, frame='icrs', equinox='J2000')

    site = EarthLocation(lon="116.76444824", lat="-26.824722084", height=300.0)
    
    # Perth from Google for the moment
    perth = EarthLocation(lon="115.8605", lat="-31.9505", height=0.0)
    
    rmax = 300.0
    low = create_named_configuration('LOWR3', rmax=rmax)
    nants = len(low.names)
    
    # Calculate the propagators for signals from Perth to the stations in low
    # These are fixed in time but vary with frequency.
    propagators = create_propagators(low, perth, frequency=frequency)
    
    plt.clf()
    for i in range(0, propagators.shape[0], 33):
        plt.plot(frequency, numpy.angle(propagators[i, :]), '.', label=str(i))
    plt.title("Propagator phases, Perth to LOW, rmax = %.1f (m)" % rmax)
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Phase (rad)')
    plt.savefig("propagators.png")
    plt.show(block=False)
    
    # Now calculate the interferer propagator fringe. This is static in time.
    interferer_fringe = calculate_interferer_fringe(propagators)
    
    plt.clf()
    plt.imshow(numpy.angle(interferer_fringe[0, ...]))
    plt.title('Propagator fringe phase, with respect to station %s' % low.names[0])
    plt.xlabel('Channel')
    plt.ylabel('Station id')
    plt.savefig("interferer_fringes.png")
    plt.show(block=False)

    # Station fringe rotation: shape [nants, ntimes, nchan] complex phasor to be applied to
    # reference to the pole.
    fringe_rotation = calculate_station_fringe_rotation(low.xyz, times, frequency, phasecentre, pole)
    plt.clf()
    plt.imshow(numpy.angle(fringe_rotation[:, 0, :]))
    plt.title('Fringe rotation phase, station %s' % low.names[0])
    plt.xlabel('Integration')
    plt.ylabel('Channel')
    plt.savefig("fringe_rotation.png")
    plt.show(block=False)

    # Correlation: [nants, nants, ntimes, nchan]
    correlation = calculate_station_correlation(fringe_rotation)
    plt.clf()
    plt.imshow(numpy.angle(correlation[0, 1, :, :]))
    plt.title('Correlation phase, station %s - station %s ' % (low.names[0], low.names[1]))
    plt.xlabel('Channel')
    plt.ylabel('Integration')
    plt.savefig("correlation.png")
    plt.show(block=False)

