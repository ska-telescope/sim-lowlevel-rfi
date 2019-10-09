# coding: utf-8
import pprint

import matplotlib.pyplot as plt
import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord, EarthLocation
import astropy.constants as const

from data_models.polarisation import PolarisationFrame
from processing_components.simulation.rfi import calculate_averaged_correlation, simulate_rfi_block
from processing_library.image.operations import create_image
from processing_library.util.array_functions import average_chunks
from workflows.arlexecute.imaging.imaging_arlexecute import invert_list_arlexecute_workflow, sum_invert_results
from wrappers.arlexecute.execution_support.arlexecute import arlexecute
from wrappers.arlexecute.execution_support.dask_init import get_dask_Client
from wrappers.arlexecute.image.operations import show_image, export_image_to_fits
from wrappers.arlexecute.simulation.configurations import create_named_configuration
from wrappers.arlexecute.simulation.noise import addnoise_visibility
from wrappers.arlexecute.visibility.base import create_blockvisibility
from wrappers.arlexecute.visibility.coalesce import convert_blockvisibility_to_visibility

if __name__ == '__main__':

    phasecentre = SkyCoord(ra=+0.0 * u.deg, dec=-45.0 * u.deg, frame='icrs', equinox='J2000')

    times = numpy.arange(-21600, 21600, 1200)
    frequency = [1.74e8]
    channel_bandwidth = [10e6]
    
    for rmax in [500, 750, 1000, 1500, 2000, 2500]:
        low = create_named_configuration('LOWR3', rmax=rmax)

        s2r = numpy.pi / 43200.0
        bvis = create_blockvisibility(low, s2r * times, frequency,
                                      channel_bandwidth=channel_bandwidth,
                                      phasecentre=phasecentre,
                                      polarisation_frame=PolarisationFrame("stokesI"))
        
        plt.clf()
        plt.plot(bvis.uvw[...,0].flatten(), bvis.uvw[...,1].flatten(), '.')
        plt.title('rmax = %.2f' % rmax)
        plt.show()
