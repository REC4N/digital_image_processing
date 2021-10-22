# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Author: Edgar Alejandro Recancoj Pajarez
# W01404391
# ECE 5220 - Image procesing
# Homework 3
# Date: 09/28/2021
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Import modules
import imageio
from matplotlib import pyplot as plt
from numba import jit
import numpy as np
import time


def minus_one(f):
    """
    Multiply image by (-1)^(x + y) representing a Fourier shift operation.

    :param f: Numpy array of floating numbers representing image.
    :returns: Numpy array representing shifted image.
    """

    # Check if image is floating point. If not, convert it.
    if f.dtype != np.dtype("float64"):
        print("    NOTE: Input image is %s. Converting to float64!" % str(f.dtype))
        f.astype(np.float64)
    # Create output array.
    g = np.zeros_like(f).astype(np.float64)
    # Apply Fourier Shift to image.
    helper_fourier_shift(f, g)
    return g

@jit(nopython=True, fastmath=True)
def helper_fourier_shift(f, g):
    """
    Helper function for minus_one(). Needed for Numba implementation.

    :param f: Numpy array of floating numbers representing input image.
    :param g: Numpy array of floating numbers representing output image.
    :returns: Numpy array representing shifted image.
    """

    for x in range(f.shape[0]):
        for y in range(f.shape[1]):
            g[x, y] = f[x, y] * (-1)**(x + y)

def dft2D(f):
    """
    Compute 2D FFT on a image (using only the 1D FFT).

    :param f: Numpy array of floating numbers representing image.
    :returns: Numpy array resulting of applying the 2D FFT to image.
    """

    # Compute 1D FFT along rows.
    processed_rows = np.fft.fft(f, axis=0)
    # Compute 1D FFT along columns.
    F = np.fft.fft(processed_rows, axis=1)
    return F

def idft2D(F):
    """
    Compute 2D inverse FFT on a image (using the forward 2D FFT).

    :param f: Numpy array of floating numbers representing image in frequency domain.
    :returns: Numpy array resulting of applying the 2D inverse FFT.
    """

    # Compute complex conjugate of F.
    F_conj = np.conjugate(F)
    # Obtain conjugate of f by applying 2D forward FFT.
    f_conj = dft2D(F_conj) / (F.shape[0] * F.shape[1])
    # Obtain f by taking its complex conjugate.
    f = np.conjugate(f_conj)
    return f
    
def lp_filter_tf(type, P, Q, D_0, n=2):
    """
    Generate a P x Q low pass filter transfer function H depending on the type specified. 

    :param type: Determines the type of the low pass filter. Available options are
    "ideal", "gaussian" and "butterworth".
    :param P: Size of filter in u direction.
    :param Q: Size of filter in v direction.
    :param D_0: If type is either "ideal" or "butterworth", it represents the cutoff
    frequency. If type is "gaussian", it represents the standard deviation.
    :param n: Order of the butterworth filter.
    :returns: Numpy array representing the low pass filter transfer function H.
    """
    # Create transfer function H.
    H = np.zeros((P, Q))
    if type == "ideal":
        create_lp_ideal(H, P, Q, D_0)
    elif type == "gaussian":
        create_lp_gaussian(H, P, Q, D_0)
    elif type == "butterworth":
        create_lp_butterworth(H, P, Q, D_0, n)
    else:
        print(    "ERROR: No valid type was selected.")
    return H

@jit(nopython=True, fastmath=True)
def create_lp_ideal(H, P, Q, D_0):
    """
    Create an ideal P x Q low pass filter with cutoff frequency D_0.

    :param H: Numpy array representing the low pass filter transfer function.
    :param P: Size of filter in u direction.
    :param Q: Size of filter in v direction.
    :param D_0: Cutoff frequency of filter.
    :returns: (Void)
    """

    for u in range(P):
        for v in range(Q):
            dist = distance_from_center(u, v, P, Q)
            if dist <= D_0:
                H[u, v] = 1
            else:
                H[u, v] = 0

@jit(nopython=True, fastmath=True)
def create_lp_gaussian(H, P, Q, D_0):
    """
    Create a gaussian P x Q low pass filter with standard deviation D_0.

    :param H: Numpy array representing the low pass filter transfer function.
    :param P: Size of filter in u direction.
    :param Q: Size of filter in v direction.
    :param D_0: Standard deviation.
    :returns: (Void)
    """

    for u in range(P):
        for v in range(Q):
            H[u, v] = np.e ** (-(distance_from_center(u, v, P, Q)**2)/(2 * (D_0**2)))

@jit(nopython=True, fastmath=True)
def create_lp_butterworth(H, P, Q, D_0, n):
    """
    Create a butterworth P x Q low pass filter with cutoff frequency D_0 and n order.

    :param H: Numpy array representing the low pass filter transfer function.
    :param P: Size of filter in u direction.
    :param Q: Size of filter in v direction.
    :param D_0: Cutoff frequency.
    :param n: Order of butterworth filter.
    :returns: (Void)
    """

    for u in range(P):
        for v in range(Q):
            H[u, v] = 1 / (1 + (distance_from_center(u, v, P, Q) / D_0)**(2*n))

def hp_filter_tf(type, P, Q, D_0, n=2):
    """
    Generate a P x Q high pass filter transfer function H depending on the type specified. 

    :param type: Determines the type of the high pass filter. Available options are
    "ideal", "gaussian" and "butterworth".
    :param P: Size of filter in u direction.
    :param Q: Size of filter in v direction.
    :param D_0: If type is either "ideal" or "butterworth", it represents the cutoff
    frequency. If type is "gaussian", it represents the standard deviation.
    :param n: Order of the butterworth filter.
    :returns: Numpy array representing the high pass filter transfer function H.
    """

    # Obtain corresponding low pass transfer function filter.
    H_lp = lp_filter_tf(type, P, Q, D_0, n)
    # Obtain the high pass transfer function by substracting 1 from the low pass
    # transfer function.
    H_hp = 1 - H_lp
    return H_hp

def dft_filtering(f, H, padmode="replicate"):
    """
    Filter image f with a given filter transfer function H.

    :param f: Numpy array representing image.
    :param H: Numpy array of filter transfer function.
    :param padmode: Type of padding to use. If "replicate", use replicate padding. If "zeros", use
    zero padding.
    :returns: Numpy array representing the filtered image.
    """

    # Step 1: Calculate dimensions of image in frequency domain with padding added.
    M, N = f.shape
    # Step 2: Form padded image.
    if padmode == "replicate":
        type = "edge"
    else:
        type = "constant"
    f_padded = np.pad(f, ((0, M), (0, N)), mode=type).astype(np.float64)
    # Step 3: Center the Fourier Transform on the P x Q frequency rectangle.
    f_centered = minus_one(f_padded)
    # Step 4: Compute DFT on centered padded image.
    F = dft2D(f_centered)
    # Step 5: Multiply F with the filter transfer function H.
    G = H * F
    # Step 6: Obtain the filtered image of size P x Q.
    g_padded = minus_one(np.real(idft2D(G)))
    # Step 7: Extract the M x N region of g_padded for final result.
    g = g_padded[:M, :N]
    return g



@jit(nopython=True, fastmath=True)
def distance_from_center(u, v, P, Q):
    """
    Calculate euclidean distance between point (u, v) in the frequency domain and
    the center of the P x Q frequency rectange.

    :param u: Vertical position in frequency domain.
    :param v: Horizontal position in frequency domain.
    :param P: Size of filter in u direction.
    :param Q: Size of filter in v direction.
    :returns: Distance from center to point (u, v)
    """

    return np.sqrt((u - P/2)**2 + (v - Q/2)**2)

def blur(img, type, D_0, n=2, padmode="replicate"):
    """
    Blur image using 2D FFT.

    :param img: Numpy array of the image.
    :param D_0: Cutoff frequency for low pass filter.
    :param type: Type of blurring
    :returns: Numpy array of blurred image.
    """

    # Obtain dimensions of image in spatial and frequency domain.
    M, N = img.shape
    P = 2 * M
    Q = 2 * N
    if type == "gaussian":
        # Create a gaussian filter transfer function.
        gaussian_filter = lp_filter_tf("gaussian", P, Q, D_0)
        # Apply gaussian blur using 2D FFT.
        gaussian_blur = dft_filtering(f=img, H=gaussian_filter, padmode=padmode)
        return scale_img(gaussian_blur)
    else:
        # Create a butterworth filter transfer function.
        butterworth_filter = lp_filter_tf("butterworth", P, Q, D_0, n)
        # Apply butterworth blur using 2D FFT.
        butterworth_blur = dft_filtering(f=img, H=butterworth_filter, padmode=padmode)
        return scale_img(butterworth_blur)

def sharpen(img, type):
    """
    Apply sharpening to image using 2D FFT.

    :param img: Numpy array of the image.
    :param type: Type of sharpening. Options available are "unsharp_masking" and "highboost".
    :returns: Numpy array of sharpened image.
    """

    # Apply gaussian blur using 2D FFT.
    blurred_img = blur(img, "gaussian", 120)
    # Create the mask.
    mask = img - blurred_img
    # In unsharp maksing, weight portion of mask is equal to 1. In highboost filtering, 
    # weight portion of mask is greater than 1.
    if type == "unsharp_masking":
        k = 1
    else:
        k = 1.6
    # Add weighted portion of mask back to original image.
    sharp_img = img + k * mask
    return scale_img(sharp_img)

def scale_img(img):
    """
    Scale img array between 0 and 255
    :param img: Numpy array of the image. 
    :returns: Scaled image with values [0, 255]
    """
    # Scale values of img between 0 and 255.
    img -= np.amin(img)
    img /= np.amax(img)
    img *= 255
    return img

if __name__ == "__main__":
    # Section: Low pass filtering.
    # Create LP transfer functions.
    start = time.time() 
    lp_ideal = lp_filter_tf("ideal", P=512, Q=512, D_0=100)
    stop = time.time()
    print("Created ideal LP TF in %.2f seconds." % (stop - start))
    start = time.time() 
    lp_gaussian = lp_filter_tf("gaussian", P=512, Q=512, D_0=50)
    stop = time.time()
    print("Created gaussian LP TF in %.2f seconds." % (stop - start))
    start = time.time() 
    lp_butterworth = lp_filter_tf("butterworth", P=512, Q=512, D_0=80, n=2)
    stop = time.time()
    print("Created butterworth LP TF in %.2f seconds." % (stop - start))

    # Create images.
    fig1 = plt.figure(num=1, figsize=(14, 5))
    fig1.add_subplot(1, 3, 1)
    plt.imshow(lp_ideal, cmap="gray")
    plt.title("Ideal LP transfer function.")
    fig1.add_subplot(1, 3, 2)
    plt.imshow(lp_gaussian, cmap="gray")
    plt.title("Gaussian LP transfer function.")
    fig1.add_subplot(1, 3, 3)
    plt.imshow(lp_butterworth, cmap="gray")
    plt.title("Butterworth LP transfer function.")

    # Section: High pass filtering.
    # Create HP transfer functions.
    start = time.time() 
    hp_ideal = hp_filter_tf("ideal", P=512, Q=512, D_0=10)
    stop = time.time()
    print("Created ideal HP TF in %.2f seconds." % (stop - start))
    start = time.time() 
    hp_gaussian = hp_filter_tf("gaussian", P=512, Q=512, D_0=100)
    stop = time.time()
    print("Created gaussian HP TF in %.2f seconds." % (stop - start))
    start = time.time() 
    hp_butterworth = hp_filter_tf("butterworth", P=512, Q=512, D_0=150, n=2.5)
    stop = time.time()
    print("Created butterworth HP TF in %.2f seconds." % (stop - start))

    # Create images.
    fig2 = plt.figure(num=2, figsize=(14, 5))
    fig2.add_subplot(1, 3, 1)
    plt.imshow(hp_ideal, cmap="gray")
    plt.title("Ideal HP transfer function.")
    fig2.add_subplot(1, 3, 2)
    plt.imshow(hp_gaussian, cmap="gray")
    plt.title("Gaussian HP transfer function.")
    fig2.add_subplot(1, 3, 3)
    plt.imshow(hp_butterworth, cmap="gray")
    plt.title("Butterworth HP transfer function.")

    # Section: Image blurring.
    # Read image to blur.
    pattern = imageio.imread("testpattern1024.tif")

    # Apply blurring using Gaussian filter.
    start = time.time() 
    gaussian_blur = blur(img=pattern, type="gaussian", D_0=12)
    stop = time.time()
    print("Blurred testpattern1024.tif using Gaussian filter in %.2f seconds." % (stop - start))
    
    # Apply blurring using Butterworth filter.
    start = time.time() 
    butterworth_blur = blur(img=pattern, type="butterworth", D_0=20, n=2)
    stop = time.time()
    print("Blurred testpattern1024.tif using Butterworth filter in %.2f seconds." % (stop - start))

    # Show original and blurred images.
    fig3 = plt.figure(num=3, figsize=(14, 5))
    fig3.add_subplot(1, 3, 1)
    plt.imshow(pattern, cmap="gray")
    plt.axis("off")
    plt.title("Test Pattern - Original Image")
    fig3.add_subplot(1, 3, 2)
    plt.imshow(gaussian_blur.astype(np.int64), cmap="gray")
    plt.axis("off")
    plt.title("Test Pattern - Gaussian blur")
    fig3.add_subplot(1, 3, 3)
    plt.imshow(butterworth_blur.astype(np.int64), cmap="gray")
    plt.axis("off")
    plt.title("Test Pattern - Butterworth blur")

    # Section: Image sharpening.
    # Read image to sharpen.
    moon = imageio.imread("blurry-moon.tif")

    # Apply unsharp masking.
    start = time.time() 
    unsharp_sharpening = sharpen(moon, "unsharp_masking")
    stop = time.time()
    print("Sharpened blurry-moon.tif using unsharp masking in %.2f seconds." % (stop - start))

    # Apply highboost sharpening.
    start = time.time() 
    highboost_sharpening = sharpen(moon, "highboost")
    stop = time.time()
    print("Sharpened blurry-moon.tif using highboost filtering in %.2f seconds." % (stop - start))

    # Show original and sharpened images.
    fig4 = plt.figure(num=4, figsize=(14, 5))
    fig4.add_subplot(1, 3, 1)
    plt.imshow(moon, cmap="gray")
    plt.axis("off")
    plt.title("Blurry Moon - Original Image")
    fig4.add_subplot(1, 3, 2)
    plt.imshow(unsharp_sharpening.astype(np.int64), cmap="gray")
    plt.axis("off")
    plt.title("Blurry Moon - Unsharp masking")
    fig4.add_subplot(1, 3, 3)
    plt.imshow(highboost_sharpening.astype(np.int64), cmap="gray")
    plt.axis("off")
    plt.title("Blurry Moon - Highboost filtering")

    # Show all images.
    plt.show()
    