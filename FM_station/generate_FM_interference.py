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
import numpy as np

# ------------  PARAMETERS

N = 5000  # number of samples to read
nAverages = 10  # number of averages
# folder = "C:\\Users\\F.Divruno\\Downloads\\" # change this to your folder.
# filename = "17-22-08_89100kHz.wav"
folder = "FM_station_data/"
filename = "17-22-08_89100kHz.wav"

CenterFrequency = 89100e3  # Centre freq of the recording is the number at the end of the filename.

# Read an IQ recording of FM stations:
wav_in = wave.open(folder + filename, "r")
print("WAV summary is: ", wav_in.getparams())

sampleFreq = wav_in.getframerate()  # 2.4e6 sample freq of the SDR to acquire this signals
max_number_frames = wav_in.getnframes()
timeMax = max_number_frames / sampleFreq
print("There are %.2g (I,Q) samples, sampled at %g (MHz), over time %.1f (secs)" % (max_number_frames, sampleFreq/1e6, timeMax))
dt = 1.0 / sampleFreq
print("Interval between samples = %g (us)" % (dt*1e6))
t = np.linspace(0, timeMax, N)

## Waterfall plot
wav_in.rewind()
ntimes = 256
nchan = 512

tscan = nchan / sampleFreq
print("Interval between frequency scans = %g (us)" % (tscan*1e6))
print("Frequency resolution = %g kHz" % (1e-3 * sampleFreq/nchan))

timeInterval = nchan / sampleFreq
timeMax = ntimes / sampleFreq  # duration of the loaded signals
scantimes = tscan * np.arange(ntimes)

freq = np.fft.fftshift(np.fft.fftfreq(nchan, d=1 / sampleFreq) + CenterFrequency)

waterfall = np.zeros([ntimes, nchan])

for sample in range(ntimes):
    I = np.zeros(nchan)
    Q = np.zeros(nchan)
    # We read and FFT nchan samples, advancing the time by nchan/samplingFreq f
    for n in range(nchan):
        aux = wav_in.readframes(1)
        I[n] = aux[0]
        Q[n] = aux[1]
    I_fft = np.fft.fftshift(np.fft.fft(I))
    Q_fft = np.fft.fftshift(np.fft.fft(Q))
    waterfall[sample, :] = abs(I_fft - 1j * Q_fft)

plt.clf()
plt.imshow(10.0*np.log10(waterfall), origin='bottom')
plt.xlabel('Frequency (MHz)')
plt.ylabel('Time (ms)')
fticks = np.array([88.0e6, 88.5e6, 89.0e6, 89.5e6, 90.0e6])
ifticks = np.round(nchan*(fticks-np.min(freq))/(sampleFreq)).astype('int')
plt.xticks(ifticks, fticks*1e-6)

tticks = 4.0 * np.array([0.25e-2, 0.50e-2, 0.75e-2, 1.0e-2, 1.25e-2])
itticks = np.round((tticks-np.min(scantimes))/tscan).astype('int')
plt.yticks(itticks, tticks*1e3)

cbar = plt.colorbar()
cbar.set_label('Power (dB)', rotation=270)
plt.tight_layout()
plt.savefig("Waterfall.png")
plt.show()
