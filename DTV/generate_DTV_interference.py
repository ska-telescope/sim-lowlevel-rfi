"""
Created on Mon Jun 24 10:52:25 2019
    Reads a wav file with SDR IQ capture of FM stations located in :
    https://mega.nz/#F!3UUUnSiD!WLhWZ3ff4f4Pi7Ko_zcodQ
    
    Also: https://drive.google.com/open?id=1itb_ePcPeDRXrVBIVL-1Y3wrt8yvpW28
    
    Also generates IQ stream sampled at 2.4Msps to simulate a similar spectrum
    sinusoids, this might be useful in an early stage to use a known signal.

@author: f.divruno
"""

# !/usr/bin/env python3


import wave

import matplotlib.pyplot as plt
import numpy

# ------------  PARAMETERS

def calculate_noise_visibility(bandwidth, int_time, diameter, t_sys, eta):
    """Determine noise rms per visibility
    :returns: Sigma [nrows]
    """
    
    k_b = 1.38064852e-23
    area = numpy.pi * (diameter / 2.) ** 2
    bt = bandwidth * int_time
    sigma = (numpy.sqrt(2) * k_b * t_sys) / (area * eta * (numpy.sqrt(bt)))
    sigma *= 1e26
    return sigma

def generate_DTV(frequency, times, power=50e3, gain=1e-9):
    nchan = len(frequency)
    ntimes = len(times)
    shape = [ntimes, nchan]
    sshape = [ntimes, nchan//2]
    bchan = nchan//4
    echan = 3*nchan//4
    amp = 1e26 * gain * numpy.sqrt(2.0) * power / (max(frequency)-min(frequency))
    print("RMS signal per sample = %g Jy" % amp)
    signal = numpy.zeros(shape, dtype='complex')
    signal[:, bchan:echan] +=  numpy.random.normal(0.0, amp, sshape) + 1j * numpy.random.normal(0.0, amp, sshape)
    return signal


def add_noise(waterfall, bandwidth, int_time, diameter, t_sys, eta):
    """Determine noise rms per visibility
    :returns: Sigma [nrows]
    """
    # The specified sensitivity (effective area / T_sys) is roughly 610 m ^ 2 / K in the range 160 - 200MHz
    # sigma_vis = 2 k T_sys / (area * sqrt(tb)) = 2 k 512 / (610 * sqrt(tb)
    sens = 610
    k_b = 1.38064852e-23
    bt = bandwidth * int_time
    sigma = 2 * 1e26 * k_b / ((sens/512) * (numpy.sqrt(bt)))
    print("RMS noise per sample = %g Jy" % sigma)
    sshape = waterfall.shape
    waterfall += numpy.random.normal(0.0, sigma, sshape) + 1j * numpy.random.normal(0.0, sigma, sshape)
    return waterfall


if __name__ == "__main__":
    sample_freq = 2e5
    frequency = numpy.arange(170.5e6, 184.5e6, sample_freq)
    nchan = len(frequency)
    tscan = 0.2
    times = numpy.arange(0.0, 30.0, 0.2)
    ntimes = len(times)
    gain = 1e-18
    waterfall = generate_DTV(frequency, times, 1.0, gain)
    
    waterfall = add_noise(waterfall, bandwidth=sample_freq, int_time=tscan, diameter=35.0, t_sys=200, eta=0.9)
    
    plt.clf()
    plt.imshow(numpy.abs(waterfall), origin='bottom')
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Time (s)')
    
    fticks = numpy.arange(171e6, 185e6, 5e6)
    ifticks = numpy.round((fticks-numpy.min(frequency))/(sample_freq)).astype('int')
    plt.xticks(ifticks, fticks*1e-6)
    tticks = numpy.array([5, 10, 15, 20, 25])
    itticks = numpy.round((tticks-numpy.min(times))/tscan).astype('int')
    plt.yticks(itticks, tticks)
    
    cbar = plt.colorbar()
    plt.title(("Gain towards Perth = %.1f dB" % (10*numpy.log10(gain))))
    cbar.set_label('Power Spectral Density (Jy)', rotation=270)
    plt.tight_layout()
    plt.savefig("DTV_Waterfall.png")
    plt.show()
