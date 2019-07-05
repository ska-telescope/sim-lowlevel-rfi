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

def generate_DTV(frequency, times, power=50e3):
    nchan = len(frequency)
    ntimes = len(times)
    shape = [ntimes, nchan]
    sshape = [ntimes, nchan//2]
    bchan = nchan//4
    echan = 3*nchan//4
    amp = numpy.sqrt(2.0) * power / (max(frequency)-min(frequency))
    signal = numpy.zeros(shape, dtype='complex')
    signal[:, bchan:echan] +=  numpy.random.normal(0.0, amp, sshape) + 1j * numpy.random.normal(0.0, amp, sshape)
    return signal

if __name__ == "__main__":
    sample_freq = 2e5
    frequency = numpy.arange(170.5e6, 184.5e6, sample_freq)
    nchan = len(frequency)
    tscan = 0.2
    times = numpy.arange(0.0, 30.0, 0.2)
    ntimes = len(times)
    waterfall = generate_DTV(frequency, times, 1.0)
    
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
    cbar.set_label('Power Spectral Density (W/Hz)', rotation=270)
    plt.tight_layout()
    plt.savefig("DTV_Waterfall.png")
    plt.show()
