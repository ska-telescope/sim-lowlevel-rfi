from astropy import units as u
from astropy.coordinates import SkyCoord

from rfi_simulation import create_propagators, calculate_rfi_at_station, \
    calculate_station_correlation_rfi, calculate_station_fringe_rotation, generate_DTV


def simulate_rfi_block(config, times, frequency, phasecentre, emitter_location, emitter_power=5e4,
                       attenuation=1.0):
    """ Simulate RFI block
    
    :param config: ARL telescope Configuration
    :param times: observation times (hour angles)
    :param frequency: frequencies
    :param phasecentre:
    :param emitter_location: EarthLocation of emitter
    :param emitter_power: Power of emitter
    :param attenuation: Attenuation to be applied to signal
    :return:
    """
    # Calculate the power spectral density of the DTV station: Watts/Hz
    emitter = generate_DTV(frequency, times, power=emitter_power, timevariable=False)
    
    # Calculate the propagators for signals from Perth to the stations in low
    # These are fixed in time but vary with frequency. The ad hoc attenuation
    # is set to produce signal roughly equal to noise at LOW
    propagators = create_propagators(config, emitter_location, frequency=frequency,
                                     attenuation=attenuation)
    # Now calculate the RFI at the stations, based on the emitter and the propagators
    rfi_at_station = calculate_rfi_at_station(propagators, emitter)
    
    # Station fringe rotation: shape [nants, ntimes, nchan] complex phasor to be applied to
    # reference to the pole.
    pole = SkyCoord(ra=+0.0 * u.deg, dec=-90.0 * u.deg, frame='icrs', equinox='J2000')
    fringe_rotation, uvw = calculate_station_fringe_rotation(config.xyz, times, frequency, phasecentre, pole)
    
    # Calculate the rfi correlationm using the fringe rotation and the rfi at the station
    # [nants, nants, ntimes, nchan]
    correlation = calculate_station_correlation_rfi(fringe_rotation, rfi_at_station)
    
    return correlation, uvw
