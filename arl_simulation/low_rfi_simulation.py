# coding: utf-8


results_dir = "."
dask_dir = "./dask-work-space"

import logging

import matplotlib.pyplot as plt
import numpy
from astropy import constants
from astropy import units as u
from astropy.coordinates import SkyCoord, EarthLocation

from wrappers.arlexecute.simulation.configurations import create_named_configuration


def init_logging():
    logging.basicConfig(filename='%s/ska-pipeline.log' % results_dir,
                        filemode='a',
                        format='%%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.INFO)


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
        r = numpy.sqrt(vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2)
        k = 2.0 * numpy.pi * frequency / constants.c.value
        propagators[iant, :] = numpy.exp(- 1.0j * k * r) / r
    
    return propagators


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
    
    site = EarthLocation(lon="116.76444824", lat="-26.824722084", height=300.0)
    
    # Perth from Google for the moment
    perth = EarthLocation(lon="115.8605", lat="-31.9505", height=0.0)
    
    rmax = 3000.0
    low = create_named_configuration('LOWR3', rmax=rmax)
    print(low)
    propagators = create_propagators(low, perth, frequency=frequency)
    
    plt.clf()
    for i in range(0,propagators.shape[0],33):
        plt.plot(frequency, numpy.angle(propagators[i,:]), '.', label=str(i))
    plt.title("Propagator phases, rmax = %.1f (m)" % rmax)
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Phase (rad)')
    plt.show()
