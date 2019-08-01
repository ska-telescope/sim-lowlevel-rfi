# coding: utf-8
""" Power spectrum for an image, adapted from Fred Dulwich code
"""
import pprint
import os
import astropy.constants as consts
import matplotlib.pyplot as plt
import numpy

from processing_library.image.operations import fft_image, copy_image
from wrappers.arlexecute.image.operations import import_image_from_fits, show_image


def radial_profile(image, centre=None):
    if centre is None:
        centre = (image.shape[0] // 2, image.shape[1] // 2)
    x, y = numpy.indices((image.shape[0:2]))
    r = numpy.sqrt((x - centre[0]) ** 2 + (y - centre[1]) ** 2)
    r = r.astype(numpy.int)
    return numpy.bincount(r.ravel(), image.ravel()) / numpy.bincount(r.ravel())


if __name__ == '__main__':
    
    basename = os.path.basename(os.getcwd())
    
    pp = pprint.PrettyPrinter()
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Display power spectrum of image')
    
    parser.add_argument('--image', type=str, default=None, help='Image name')
    parser.add_argument('--signal_channel', type=int, default=None, help='Channel containing signal+noise only')
    parser.add_argument('--noise_channel', type=int, default=0, help='Channel containing noise only')
    parser.add_argument('--resolution', type=float, default=5e-4,
                        help='Resolution in radians needed for conversion to K')
    
    args = parser.parse_args()
    print("Display power spectrum of an image")
    
    im = import_image_from_fits(args.image)
    
    nchan, npol, ny, nx = im.shape
    
    if args.signal_channel is None:
        signal_channel = nchan // 2
    else:
        signal_channel = args.signal_channels
        
    noise_channel = args.noise_channel
    resolution = args.resolution
    
    plt.clf()
    show_image(im, chan=signal_channel)
    plt.title('Signal image %s' % (basename))
    plt.savefig('simulation_image_channel_%d.png' % signal_channel)
    plt.show()
    plt.clf()
    show_image(im, chan=noise_channel)
    plt.title('Noise image %s' % (basename))
    plt.savefig('simulation_noise_channel_%d.png' % signal_channel)
    plt.show()

    print(im)
    imfft = fft_image(im)
    print(imfft)
    
    omega = numpy.pi * resolution ** 2 / (4 * numpy.log(2.0))
    wavelength = consts.c / numpy.average(im.frequency)
    kperjy = 1e-26 * wavelength ** 2 / (2 * consts.k_B * omega)
    
    im_spectrum = copy_image(imfft)
    im_spectrum.data = kperjy.value * numpy.abs(imfft.data)
    plt.clf()
    show_image(im_spectrum, chan=signal_channel, vmax=0.01 * numpy.max(im_spectrum.data[signal_channel,...]))
    plt.gca().set_title("Amplitude(FFT(image)) %s" % (basename))
    plt.tight_layout()
    plt.savefig('power_spectrum_image_channel_%d.png' % signal_channel)
    plt.show()
    noisy = numpy.max(im_spectrum.data[noise_channel, 0]) > 0.0
    if noisy:
        plt.clf()
        show_image(im_spectrum, chan=noise_channel, vmax=0.01 * numpy.max(im_spectrum.data[noise_channel,...]))
        plt.gca().set_title("Amplitude(FFT(noise)) %s" % (basename))
        plt.tight_layout()
        plt.savefig('power_spectrum_image_noise_channel_%d.png' % signal_channel)
        plt.show()

    profile = radial_profile(im_spectrum.data[signal_channel, 0])
    noise_profile = radial_profile(im_spectrum.data[noise_channel, 0])

    plt.clf()
    cellsize_uv = numpy.abs(imfft.wcs.wcs.cdelt[0])
    lambda_max = cellsize_uv * len(profile)
    lambda_axis = numpy.linspace(cellsize_uv, lambda_max, len(profile))
    theta_axis = 180.0 / (numpy.pi * lambda_axis)
    plt.plot(theta_axis, profile, color='blue', label='signal')
    if noisy:
        plt.plot(theta_axis, noise_profile, color='red', label='noise')
    plt.gca().set_title("Power spectrum of image %s" % (basename))
    plt.gca().legend()
    plt.gca().set_xlabel(r"$\theta$")
    plt.gca().set_ylabel(r"$K^2$")
    plt.gca().set_xscale('log')
    plt.gca().set_yscale('log')
    plt.gca().set_ylim(1e-6 * numpy.max(profile), 2.0 * numpy.max(profile))
    plt.tight_layout()
    plt.savefig('power_spectrum_profile_channel_%d.png' % signal_channel)
    plt.show()
    
    filename = 'power_spectrum_channel.csv'
    results = list()
    for row in range(len(theta_axis)):
        result = dict()
        result['inverse_theta'] = theta_axis[row]
        result['profile'] = profile[row]
        result['noise_profile'] = noise_profile[row]
        results.append(result)
        
    import csv
    with open(filename, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=results[0].keys(), delimiter=',', quotechar='|',
                                quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        for result in results:
            writer.writerow(result)
        csvfile.close()
    
