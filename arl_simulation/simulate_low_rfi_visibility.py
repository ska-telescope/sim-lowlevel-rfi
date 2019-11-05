# coding: utf-8
""" Simulate low level RFI

We are interested in the effects of RFI signals that cannot be detected in the visibility data. Therefore,
in our simulations we add attenuation selected to give SNR about 1 in the unaveraged time-frequency data.
This is about 180dB for a DTV station in Perth.

The scenario is:
* There is a TV station at a remote location (e.g. Perth), emitting a broadband signal (7MHz) of known power (50kW).
* The emission from the TV station arrives at LOW stations with phase delay and attenuation. Neither of these are
well known but they are probably static.
* The RFI enters LOW stations in a sidelobe of the station beam. Calculations by Fred Dulwich indicate that this
provides attenuation of about 55 - 60dB for a source close to the horizon.
* The RFI enters each LOW station with fixed delay and zero fringe rate (assuming no e.g. ionospheric ducting or
reflection from a plane)
* In tracking a source on the sky, the signal from one station is delayed and fringe-rotated to stop the fringes for
one direction on the sky.
* The fringe rotation stops the fringe from a source at the phase tracking centre but phase rotates the RFI, which
now becomes time-variable.
* The correlation data are time- and frequency-averaged over a timescale appropriate for the station field of view.
This averaging decorrelates the RFI signal.
* We want to study the effects of this RFI on statistics of the visibilities, and on images made on source and
at the pole.

The simulate_low_rfi_visibility.py script averages the data producing baseline-dependent decorrelation.
The effect of averaging is not more than about -20dB but it does vary with baseline giving the radial
power spectrum we see. The 55-60 dB is part of the 180dB. To give a signal to noise on 1 or less, the
terrain propagation must be about 100dB.

The simulation is implemented in some functions in ARL, and the script simulate_low_rfi_visibility is available
in the SKA Github repository sim-lowlevel-rfi. Distributed processing is implemented via Dask. The outputs are
fits file and plots of the images: on signal channels and on pure noise channels, and for the source of
interest and the Southern Celestial Pole. The unaveraged MeasurementSets are also output, one per time chunk.

"""
import os
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

from processing_components.visibility.base import export_blockvisibility_to_ms
from workflows.arlexecute.imaging.imaging_arlexecute import invert_list_arlexecute_workflow, sum_invert_results
from wrappers.arlexecute.execution_support.arlexecute import arlexecute
from wrappers.arlexecute.execution_support.dask_init import get_dask_Client
from wrappers.arlexecute.image.operations import show_image, export_image_to_fits
from wrappers.arlexecute.simulation.configurations import create_named_configuration
from wrappers.arlexecute.visibility.base import create_blockvisibility
from wrappers.arlexecute.visibility.coalesce import convert_blockvisibility_to_visibility

def add_noise(bvis):
    # The specified sensitivity (effective area / T_sys) is roughly 610 m ^ 2 / K in the range 160 - 200MHz
    # sigma_vis = 2 k T_sys / (area * sqrt(tb)) = 2 k 512 / (610 * sqrt(tb)
    sens = 610
    bt = bvis.channel_bandwidth[0] * bvis.integration_time[0]
    sigma = 2 * 1e26 * const.k_B.value / ((sens/512) * (numpy.sqrt(bt)))
    sshape = bvis.vis.shape
    bvis.data['vis'] += numpy.random.normal(0.0, sigma, sshape) + 1j * numpy.random.normal(0.0, sigma, sshape)
    return bvis


def simulate_rfi_image(config, times, frequency, channel_bandwidth, phasecentre, polarisation_frame,
                       time_average, channel_average, attenuation, noise,
                       emitter_location, emitter_power, use_pole, waterfall, write_ms):
    averaged_frequency = numpy.array(average_chunks(frequency, numpy.ones_like(frequency), channel_average))[0]
    averaged_channel_bandwidth, wts = numpy.array(
        average_chunks(channel_bandwidth, numpy.ones_like(frequency), channel_average))
    averaged_channel_bandwidth *= wts
    averaged_times = numpy.array(average_chunks(times, numpy.ones_like(times), time_average))[0]
    
    s2r = numpy.pi / 43200.0
    bvis = create_blockvisibility(config, s2r * times, frequency,
                                  channel_bandwidth=channel_bandwidth,
                                  phasecentre=phasecentre,
                                  polarisation_frame=polarisation_frame,
                                  zerow=False)
    
    bvis = simulate_rfi_block(bvis, emitter_location=emitter_location,
                              emitter_power=emitter_power, attenuation=attenuation, use_pole=use_pole)

    if noise:
        bvis = add_noise(bvis)

    if waterfall:
        plot_waterfall(bvis)

    if write_ms:
        msname = "simulate_rfi_%.1f.ms" % (times[0])
        export_blockvisibility_to_ms(msname, [bvis], "RFI")

    averaged_bvis = create_blockvisibility(config, s2r * averaged_times, averaged_frequency,
                                           channel_bandwidth=averaged_channel_bandwidth,
                                           phasecentre=phasecentre,
                                           polarisation_frame=polarisation_frame,
                                           zerow=False)
    npol = 1
    for itime, _ in enumerate(averaged_times):
        atime = itime * time_average
        for ant2 in range(nants):
            for ant1 in range(ant2, nants):
                for ichan, _ in enumerate(averaged_frequency):
                    achan = ichan * channel_average
                    for pol in range(npol):
                        averaged_bvis.data['vis'][itime, ant2, ant1, ichan, pol] = \
                            calculate_averaged_correlation(
                                bvis.data['vis'][atime:(atime+time_average), ant2, ant1, achan:(achan+channel_average), pol],
                                time_average, channel_average)[0,0]
                        averaged_bvis.data['vis'][itime, ant1, ant2, ichan, pol] = \
                            numpy.conjugate(averaged_bvis.data['vis'][itime, ant2, ant1, ichan, pol])
                    achan += 1
        atime += 1
    
    del bvis
    
    if noise:
        averaged_bvis = add_noise(averaged_bvis)
    
    return averaged_bvis

def plot_waterfall(bvis):
    print(bvis.uvw.shape)
    uvdist = numpy.hypot(bvis.uvw[0,:,:,0], bvis.uvw[0,:,:,1])
    print(uvdist.shape)
    uvdistmax = 0.0
    max_ant1=0
    max_ant2=0
    for ant2 in range(bvis.nants):
        for ant1 in range(ant2+1):
            if uvdist[ant2, ant1] > uvdistmax:
                uvdistmax = uvdist[ant2, ant1]
                max_ant1 = ant1
                max_ant2 = ant2
                
    basename = os.path.basename(os.getcwd())
    fig=plt.figure()
    fig.suptitle('%s: Baseline [%d, %d], ha %.2f' % (basename, max_ant1, max_ant2, bvis.time[0]))
    plt.subplot(121)
    plt.gca().set_title("Amplitude")
    plt.gca().imshow(numpy.abs(bvis.vis[: , max_ant1, max_ant2, :, 0]), origin='bottom')
    plt.gca().set_xlabel('Channel')
    plt.gca().set_ylabel('Time')
    plt.subplot(122)
    plt.gca().imshow(numpy.angle(bvis.vis[: , max_ant1, max_ant2, :, 0]), origin='bottom')
    plt.gca().set_title("Phase")
    plt.gca().set_xlabel('Channel')
    plt.gca().set_ylabel('Time')
    plt.savefig('waterfall_%d_%d_ha_%.2f.png' % (max_ant1, max_ant2, bvis.time[0]))
    plt.show(block=False)


if __name__ == '__main__':
    
    pp = pprint.PrettyPrinter()
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Simulate DTV RFI')
    
    parser.add_argument('--use_dask', type=str, default='True', help='Use Dask to distribute processing?')
    
    parser.add_argument('--context', type=str, default='DTV', help='DTV')
    parser.add_argument('--rmax', type=float, default=3e3, help='Maximum distance of station from centre (m)')
    parser.add_argument('--seed', type=int, default=18051955, help='Random number seed')
    parser.add_argument('--station_skip', type=int, default=33, help='Decimate stations by this factor')
    
    parser.add_argument('--show', type=str, default='False', help='Show images?')
    parser.add_argument('--attenuation', type=float, default=1.0, help='Attenuation factor')
    parser.add_argument('--noise', type=str, default='False', help='Add noise?')
    parser.add_argument('--ngroup_visibility', type=int, default=8, help='Process in visibility groups this large')
    parser.add_argument('--do_psf', type=str, default="False", help='Make the PSF?')
    parser.add_argument('--use_agg', type=str, default="False", help='Use Agg matplotlib backend?')
    parser.add_argument('--write_fits', type=str, default="True", help='Write fits files?')
    
    parser.add_argument('--declination', type=float, default=-45.0, help='Declination (degrees)')
    
    parser.add_argument('--npixel', type=int, default=1025, help='Number of pixel per axis in image')
    
    parser.add_argument('--nchannels_per_chunk', type=int, default=1024, help='Number of channels in a chunk')
    parser.add_argument('--channel_average', type=int, default=16, help="Number of channels in a chunk to average")
    parser.add_argument('--frequency_range', type=float, nargs=2, default=[170.5e6, 184.5e6],
                        help="Frequency range (Hz)")
    
    parser.add_argument('--nintegrations_per_chunk', type=int, default=64,
                        help='Number of integrations in a time chunk')
    parser.add_argument('--time_average', type=int, default=16, help="Number of integrations in a chunk to average")
    parser.add_argument('--integration_time', type=float, default=0.25, help="Integration time (s)")
    parser.add_argument('--time_range', type=float, nargs=2, default=[-6.0, 6.0], help="Hourangle range (hours)")
    
    parser.add_argument('--emitter_longitude', type=float, default=115.8605, help="Emitter longitude")
    parser.add_argument('--emitter_latitude', type=float, default=-31.9505, help="Emitter latitude")
    parser.add_argument('--emitter_power', type=float, default=5e4, help="Emitter power (W)]")

    parser.add_argument('--use_pole', type=str, default="False", help='Set RFI source at pole?')
    parser.add_argument('--waterfall', type=str, default="False", help='Plot waterfalls?')
    parser.add_argument('--write_ms', type=str, default="False", help='Write measurmentsets?')

    args = parser.parse_args()
    print("Starting LOW low level RFI simulation")

    pp.pprint(vars(args))

    write_ms = args.write_ms == "True"
    
    numpy.random.seed(args.seed)
    
    if args.use_dask == "True":
        client = get_dask_Client(threads_per_worker=1,
                                 processes=True,
                                 memory_limit=32 * 1024 * 1024 * 1024,
                                 n_workers=8)
        arlexecute.set_client(client=client)
        print(arlexecute.client)
    else:
        print("Running in serial mode")
        arlexecute.set_client(use_dask=False)
    
    emitter_location = EarthLocation(lon=args.emitter_longitude, lat=args.emitter_latitude, height=0.0)
    emitter_power = args.emitter_power
    print("Emitter is %.1f kW at location %s" % (1e-3 * emitter_power, emitter_location.geodetic))
    
    if args.waterfall == "True":
        waterfall = True
    else:
        waterfall = False
    
    if args.noise == "True":
        noise = True
        print("Adding noise to simulated data")
    else:
        noise = False
        
    if args.use_pole == "True":
        print("Placing emitter at the southern celestial pole")
        use_pole= True
    else:
        use_pole = False
        
    rmax = args.rmax
    low = create_named_configuration('LOWR3', rmax=rmax)
    nants = len(low.names)
    print("There are %d stations" % nants)
    station_skip = args.station_skip
    low.data = low.data[::station_skip]
    nants = len(low.names)
    print("There are %d stations after decimation" % nants)
    
    npixel = args.npixel
    
    declination = args.declination
    phasecentre = SkyCoord(ra=+0.0 * u.deg, dec=declination * u.deg, frame='icrs', equinox='J2000')
    pole = SkyCoord(ra=+0.0 * u.deg, dec=-90.0 * u.deg, frame='icrs', equinox='J2000')
    
    # Number of integrations in a time chunk
    nintegrations_per_chunk = args.nintegrations_per_chunk
    # Integration time within a chunk
    integration_time = args.integration_time
    # Number of integrations to average
    time_average = args.time_average
    # Integration time after averaging
    average_integration_time = time_average * integration_time
    print("Each chunk has %d integrations of duration %.2f (s)" %
          (args.nintegrations_per_chunk, integration_time))
    
    frequency = numpy.linspace(args.frequency_range[0], args.frequency_range[1], args.nchannels_per_chunk)
    channel_bandwidth = (frequency[-1] - frequency[0]) / (args.nchannels_per_chunk - 1)
    channel_average = args.channel_average
    print("Each chunk has %d frequency channels of width %.3f (MHz)" %
          (args.nchannels_per_chunk, channel_bandwidth * 1e-6))
    channel_bandwidth = numpy.ones_like(frequency) * channel_bandwidth
    
    start_times = numpy.arange(args.time_range[0] * 3600.0, args.time_range[1] * 3600.0,
                               nintegrations_per_chunk * integration_time)
    print("Start times", start_times)
    results = list()
    pole_results = list()
    
    chunk_start_times = [start_times[i:i + args.ngroup_visibility]
                         for i in range(0, len(start_times), args.ngroup_visibility)]
    print("Chunk start times", [c[0] for c in chunk_start_times])
    
    dopsf = args.do_psf == "True"
    
    # Find the average frequencies
    averaged_frequency = numpy.array(average_chunks(frequency, numpy.ones_like(frequency), channel_average))[0]
    if len(averaged_frequency) > 1:
        step = abs(averaged_frequency[-1] - averaged_frequency[0]) / (len(averaged_frequency)-1)
    else:
        step = channel_average * channel_bandwidth
    averaged_channel_bandwidth = step * numpy.ones_like(averaged_frequency)
    
    print("Each averaged chunk has %d integrations of duration %.2f (s)" %
          (nintegrations_per_chunk // time_average, time_average * integration_time))
    print("Each averaged chunk has %d channels of width %.3f (MHz)" %
          (len(averaged_frequency), 1e-6 * averaged_channel_bandwidth[0]))
    print("Processing %d time chunks in groups of %d" % (len(start_times), args.ngroup_visibility))

    cellsize = 1e-4 * (3000.0 / rmax)
    model_graph = arlexecute.execute(create_image)(cellsize=cellsize, npixel=npixel,
                                                   frequency=averaged_frequency,
                                                   channel_bandwidth=averaged_channel_bandwidth,
                                                   phasecentre=phasecentre,
                                                   polarisation_frame=PolarisationFrame(
                                                       "stokesI"))
    pole_model_graph = arlexecute.execute(create_image)(cellsize=cellsize, npixel=npixel,
                                                        frequency=averaged_frequency,
                                                        channel_bandwidth=averaged_channel_bandwidth,
                                                        phasecentre=pole,
                                                        polarisation_frame=PolarisationFrame(
                                                            "stokesI"))
    
    # We process the chunks (and accumulate the images) in two stages to avoid a large final reduction
    for chunk_start_time in chunk_start_times:
        chunk_results = list()
        chunk_pole_results = list()
        
        for start_time in chunk_start_time:
            times = start_time + numpy.arange(args.ngroup_visibility) * integration_time
            averaged_times = numpy.array(average_chunks(times, numpy.ones_like(times), time_average))[0]
            
            # Perform the simulation for this chunk
            bvis_graph = arlexecute.execute(simulate_rfi_image)(low, times, frequency, channel_bandwidth, phasecentre,
                                                                polarisation_frame=PolarisationFrame("stokesI"),
                                                                time_average=time_average,
                                                                channel_average=channel_average,
                                                                attenuation=args.attenuation,
                                                                noise=noise,
                                                                emitter_location=emitter_location,
                                                                emitter_power=emitter_power,
                                                                use_pole=use_pole,
                                                                waterfall=waterfall,
                                                                write_ms=write_ms)
            
            # Convert BlockVisibility to imaging-specific Visibility
            vis_graph = arlexecute.execute(convert_blockvisibility_to_visibility)(bvis_graph)
            
            # Define the images and make the dirty images
            result = invert_list_arlexecute_workflow([vis_graph], [model_graph],
                                                     context='2d', dopsf=dopsf)
            chunk_results.append(result[0])
            result = invert_list_arlexecute_workflow([vis_graph], [pole_model_graph],
                                                     context='2d', dopsf=dopsf)
            chunk_pole_results.append(result[0])
        
        # Sum over results over this chunk
        chunk_final_result = arlexecute.execute(sum_invert_results)(chunk_results)
        results.append(chunk_final_result)
        
        chunk_final_pole_result = arlexecute.execute(sum_invert_results)(chunk_pole_results)
        pole_results.append(chunk_final_pole_result)
    
    # Now construct and run a graph to sum the results from the summations over each chunk
    final_result = arlexecute.execute(sum_invert_results)(results)
    dirty, sumwt = arlexecute.compute(final_result, sync=True)
    
    # We are done! make plots and fits files
    type = 'dirty'
    if dopsf:
        type = 'psf'
    
    if args.write_fits == "True":
        imagename = "simulate_rfi_%.1f_%s.fits" % (declination, type)
        export_image_to_fits(dirty, imagename)
    
    plt.clf()
    show_image(dirty, chan=len(frequency) // channel_average // 2)
    plotname = "simulate_rfi_%.1f_%s.png" % (declination, type)
    plt.title('Image of the target field')
    plt.savefig(plotname)
    plt.show(block=False)

    final_pole_result = arlexecute.execute(sum_invert_results)(pole_results)
    pole_dirty, pole_sumwt = arlexecute.compute(final_pole_result, sync=True)

    if args.write_fits == "True":
        imagename = "simulate_rfi_pole_%s.fits" % (type)
        export_image_to_fits(pole_dirty, imagename)
    
    plt.clf()
    show_image(pole_dirty, chan=len(frequency) // channel_average // 2)
    plotname = "simulate_rfi_pole_%s.png" % (type)
    plt.title('Image of the southern celestial pole')
    plt.savefig(plotname)
    plt.show(block=False)
