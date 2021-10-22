# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Author: Edgar Alejandro Recancoj Pajarez
# W01404391
# ECE 5220 - Image procesing
# Homework 4
# Date: 09/29/2021
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Import modules
import imageio
from matplotlib import pyplot as plt
from numba import jit
import numpy as np
import time

def arith_mean(g, m, n):
    """
    Filter an image using the arithmetic mean filter.

    :param g: Numpy array representing input image.
    :param m: Neighborhood size in x direction.
    :param n: Neighborhood size in y direction.
    :returns: Numpy array representing filtered image.
    """
    # Determine the number of zeros to be added to the boundary of the image.
    pad_x = int(np.floor(m/2))
    pad_y = int(np.floor(n/2))
    # Create a padded arrray with replicate padding.
    pad_g = np.pad(g, ((pad_x, pad_x), (pad_y, pad_y)), mode="edge").astype(np.float64)
    # Create array that contains the denoised image.
    f = np.zeros_like(g).astype(np.float64)
    # Process arithmetic mean filter.
    process_arith_mean(pad_g, f, m, n)
    return f.astype(np.int64)

@jit(nopython=True, fastmath=True)
def process_arith_mean(pad_g, f, m, n):
    """
    Perform arithmetic mean filtering. Separated from arith_mean() to use Numba.

    :param pad_g: Numpy array representing padded image.
    :param f: Numpy array representing denoised image.
    :param m: Neighborhood size in x direction.
    :param n: Neighborhood size in y direction.
    :returns: (Void) 
    """

    # Obtain dimensions from the padded image.
    M, N = f.shape
    # Perform arithmetic mean filter on image.
    for x in range(M):
        for y in range(N):
            f[x, y] = pad_g[x : x + m, y : y + n].sum() / (m * n)
    
def geo_mean(g, m, n):
    """
    Filter an image using the geometric mean filter.

    :param g: Numpy array representing input image.
    :param m: Neighborhood size in x direction.
    :param n: Neighborhood size in y direction.
    :returns: Numpy array representing filtered image.
    """
    # Determine the number of zeros to be added to the boundary of the image.
    pad_x = int(np.floor(m/2))
    pad_y = int(np.floor(n/2))
    # Create a padded arrray with replicate padding.
    pad_g = np.pad(g, ((pad_x, pad_x), (pad_y, pad_y)), mode="edge").astype(np.float64)
    # Create array that contains the denoised image.
    f = np.zeros_like(g).astype(np.float64)
    # Process geometric mean filter.
    process_geo_mean(pad_g, f, m, n)
    return f.astype(np.int64)

@jit(nopython=True, fastmath=True)
def process_geo_mean(pad_g, f, m, n):
    """
    Perform geometric mean filtering. Separated from geo_mean() to use Numba.

    :param pad_g: Numpy array representing padded image.
    :param f: Numpy array representing denoised image.
    :param m: Neighborhood size in x direction.
    :param n: Neighborhood size in y direction.
    :returns: (Void) 
    """

    # Obtain dimensions from the padded image.
    M, N = f.shape
    # Perform geometric mean filter on image.
    for x in range(M):
        for y in range(N):
            f[x, y] = pad_g[x : x + m, y : y + n].prod() ** (1/ (m * n))

def har_mean(g, m, n):
    """
    Filter an image using the harmonic mean filter.

    :param g: Numpy array representing input image.
    :param m: Neighborhood size in x direction.
    :param n: Neighborhood size in y direction.
    :returns: Numpy array representing filtered image.
    """
    # Determine the number of zeros to be added to the boundary of the image.
    pad_x = int(np.floor(m/2))
    pad_y = int(np.floor(n/2))
    # Create a padded arrray with replicate padding.
    pad_g = np.pad(g, ((pad_x, pad_x), (pad_y, pad_y)), mode="edge").astype(np.float64)
    # Create array that contains the denoised image.
    f = np.zeros_like(g).astype(np.float64)
    # Process harmonic mean filter.
    process_har_mean(pad_g, f, m, n)
    return f.astype(np.int64)

@jit(nopython=True, fastmath=True)
def process_har_mean(pad_g, f, m, n):
    """
    Perform harmonic mean filtering. Separated from har_mean() to use Numba.

    :param pad_g: Numpy array representing padded image.
    :param f: Numpy array representing denoised image.
    :param m: Neighborhood size in x direction.
    :param n: Neighborhood size in y direction.
    :returns: (Void) 
    """

    # Obtain dimensions from the padded image.
    M, N = f.shape
    # Perform harmonic mean filter on image.
    for x in range(M):
        for y in range(N):
            f[x, y] = (m * n) / (1/pad_g[x : x + m, y : y + n]).sum() 

def cthar_mean(g, m, n, q):
    """
    Filter an image using the contraharmonic mean filter.

    :param g: Numpy array representing input image.
    :param m: Neighborhood size in x direction.
    :param n: Neighborhood size in y direction.
    :returns: Numpy array representing filtered image.
    """
    # Determine the number of zeros to be added to the boundary of the image.
    pad_x = int(np.floor(m/2))
    pad_y = int(np.floor(n/2))
    # Create a padded arrray with replicate padding.
    pad_g = np.pad(g, ((pad_x, pad_x), (pad_y, pad_y)), mode="edge").astype(np.float64)
    # Create array that contains the denoised image.
    f = np.zeros_like(g).astype(np.float64)
    # Process contraharmonic mean filter.
    process_cthar_mean(pad_g, f, m, n, q)
    return f.astype(np.int64)

@jit(nopython=True, fastmath=True)
def process_cthar_mean(pad_g, f, m, n, q):
    """
    Perform contraharmonic mean filtering. Separated from cthar_mean() to use Numba.

    :param pad_g: Numpy array representing padded image.
    :param f: Numpy array representing denoised image.
    :param m: Neighborhood size in x direction.
    :param n: Neighborhood size in y direction.
    :returns: (Void) 
    """

    # Obtain dimensions from the padded image.
    M, N = f.shape
    # Perform contraharmonic mean filter on image.
    for x in range(M):
        for y in range(N):
            f[x, y] = (np.power(pad_g[x : x + m, y : y + n], (q+1)).sum()) / (np.power(pad_g[x : x + m, y : y + n], (q)).sum())

def min_filter(g, m, n):
    """
    Filter an image using the min filter.

    :param g: Numpy array representing input image.
    :param m: Neighborhood size in x direction.
    :param n: Neighborhood size in y direction.
    :returns: Numpy array representing filtered image.
    """
    # Determine the number of zeros to be added to the boundary of the image.
    pad_x = int(np.floor(m/2))
    pad_y = int(np.floor(n/2))
    # Create a padded arrray with replicate padding.
    pad_g = np.pad(g, ((pad_x, pad_x), (pad_y, pad_y)), mode="edge").astype(np.float64)
    # Create array that contains the denoised image.
    f = np.zeros_like(g).astype(np.float64)
    # Process min filter.
    process_min_filter(pad_g, f, m, n)
    return f.astype(np.int64)

@jit(nopython=True, fastmath=True)
def process_min_filter(pad_g, f, m, n):
    """
    Perform min filtering. Separated from min_filter() to use Numba.

    :param pad_g: Numpy array representing padded image.
    :param f: Numpy array representing denoised image.
    :param m: Neighborhood size in x direction.
    :param n: Neighborhood size in y direction.
    :returns: (Void) 
    """

    # Obtain dimensions from the padded image.
    M, N = f.shape
    # Perform min filtering on image.
    for x in range(M):
        for y in range(N):
            f[x, y] = np.amin(pad_g[x : x + m, y : y + n])

def max_filter(g, m, n):
    """
    Filter an image using the max filter.

    :param g: Numpy array representing input image.
    :param m: Neighborhood size in x direction.
    :param n: Neighborhood size in y direction.
    :returns: Numpy array representing filtered image.
    """
    # Determine the number of zeros to be added to the boundary of the image.
    pad_x = int(np.floor(m/2))
    pad_y = int(np.floor(n/2))
    # Create a padded arrray with replicate padding.
    pad_g = np.pad(g, ((pad_x, pad_x), (pad_y, pad_y)), mode="edge").astype(np.float64)
    # Create array that contains the denoised image.
    f = np.zeros_like(g).astype(np.float64)
    # Process max filter.
    process_max_filter(pad_g, f, m, n)
    return f.astype(np.int64)

@jit(nopython=True, fastmath=True)
def process_max_filter(pad_g, f, m, n):
    """
    Perform max filtering. Separated from max_filter() to use Numba.

    :param pad_g: Numpy array representing padded image.
    :param f: Numpy array representing denoised image.
    :param m: Neighborhood size in x direction.
    :param n: Neighborhood size in y direction.
    :returns: (Void) 
    """

    # Obtain dimensions from the padded image.
    M, N = f.shape
    # Perform max filtering on image.
    for x in range(M):
        for y in range(N):
            f[x, y] = np.amax(pad_g[x : x + m, y : y + n])

def median_filter(g, m, n):
    """
    Filter an image using the median filter.

    :param g: Numpy array representing input image.
    :param m: Neighborhood size in x direction.
    :param n: Neighborhood size in y direction.
    :returns: Numpy array representing filtered image.
    """
    # Determine the number of zeros to be added to the boundary of the image.
    pad_x = int(np.floor(m/2))
    pad_y = int(np.floor(n/2))
    # Create a padded arrray with replicate padding.
    pad_g = np.pad(g, ((pad_x, pad_x), (pad_y, pad_y)), mode="edge").astype(np.float64)
    # Create array that contains the denoised image.
    f = np.zeros_like(g).astype(np.float64)
    # Process median filter.
    process_median_filter(pad_g, f, m, n)
    return f.astype(np.int64)

@jit(nopython=True, fastmath=True)
def process_median_filter(pad_g, f, m, n):
    """
    Perform median filtering. Separated from median_filter() to use Numba.
    
    :param pad_g: Numpy array representing padded image.
    :param f: Numpy array representing denoised image.
    :param m: Neighborhood size in x direction.
    :param n: Neighborhood size in y direction.
    :returns: (Void) 
    """

    # Obtain dimensions from the padded image.
    M, N = f.shape
    # Perform median filtering on image.
    for x in range(M):
        for y in range(N):
            f[x, y] = np.median(pad_g[x : x + m, y : y + n])

def adapt_median_filter(g, s_xy, s_max):
    """
    Filter an image using an adaptive median filter.

    :param g: Numpy array representing input image.
    :param s_xy: Initial size of neighborhood.
    :param x_max: Maximum allowable size of neighborhood.
    :returns: Numpy array representing filtered image.
    """
    
    m_max = s_max
    n_max = s_max
    # Determine the number of zeros to be added to the boundary of the image.
    pad_x = int(np.floor(m_max/2))
    pad_y = int(np.floor(n_max/2))
    # Create a padded arrray with replicate padding.
    pad_g = np.pad(g, ((pad_x, pad_x), (pad_y, pad_y)), mode="edge").astype(np.float64)
    # Create array that contains the denoised image.
    f = np.zeros_like(g).astype(np.float64)
    # Process adaptive median filter.
    process_adapt_median_filter(pad_g, f, s_xy, s_max)
    return f.astype(np.int64)

@jit(nopython=True, fastmath=True)
def process_adapt_median_filter(pad_g, f, s_xy, s_max):
    """
    Perform adaptive median filtering. Separated from adapt_median_filter() to use Numba.

    :param pad_g: Numpy array representing padded image.
    :param f: Numpy array representing denoised image.
    :param s_xy: Initial size of neighborhood.
    :param x_max: Maximum allowable size of neighborhood.
    :returns: (Void) 
    """

    # Obtain dimensions from the padded image.
    M, N = f.shape
    # Perform adaptive median filtering on image.
    for x in range(M):
        for y in range(N):
            # Compute algorithm to obtain intensity value at point (x, y).
            f[x, y] = adative_median_algorithm(pad_g, x, y, s_xy, s_max)

@jit(nopython=True, fastmath=True)
def adative_median_algorithm(pad_g, x, y, s_xy, s_max):
    """
    Helper function for adaptive median algorithm computation.
    """

    while (True):
        # Create neighborhood array with size given by s_xy.
        current_S = pad_g[x : x + s_xy, y : y + s_xy]
        # Obtain statistics from current neighborhood.
        z_min = np.amin(current_S)
        z_max = np.amax(current_S)
        z_med = np.median(current_S)
        z_xy = current_S[int(np.floor(s_xy/2)), int(np.floor(s_xy/2))]
        # Level A:
        if z_med > z_min and z_med < z_max:
            # Level B:
            if z_xy > z_min and z_xy < z_max:
                return z_xy
            else:
                return z_med
        else:
            # Increase size of current odd neighborhood.
            s_xy += 2
        if s_xy <= s_max:
            # Continue algorithm until size of current neighborhood exceeds max size.
            continue
        else:
            # When max size of neighborhood is reached, return the median value of S.
            return z_med


if __name__ == "__main__":
    # Section: Linear spatial filtering.
    # Read image to denoise.
    circuitboard_gaussian = imageio.imread("circuitboard-gaussian.tif")
    circuitboard_pepper = imageio.imread("circuitboard-pepper.tif")
    circuitboard_salt = imageio.imread("circuitboard-salt.tif")
    circuitboard_saltandpep = imageio.imread("circuitboard-saltandpep.tif")
    hubble = imageio.imread("hubble.tif")

    # Apply arithmetic mean filtering.
    start = time.time()
    arith_mean_filtered = arith_mean(g=circuitboard_gaussian, m=3, n=3) 
    stop = time.time()
    print("Filtered circuitboard-gaussian.tif using arith mean filter in %.2f seconds" %
          (stop - start))
    # Apply geometric mean filtering.
    start = time.time()
    geo_mean_filtered = geo_mean(g=circuitboard_gaussian, m=3, n=3) 
    stop = time.time()
    print("Filtered circuitboard-gaussian.tif using geo mean filter in %.2f seconds" %
          (stop - start))
    # Apply harmonic mean filtering.
    start = time.time()
    har_mean_filtered = har_mean(g=circuitboard_salt, m=3, n=3) 
    stop = time.time()
    print("Filtered circuitboard-gaussian.tif using har mean filter in %.2f seconds" %
          (stop - start))
    # Apply contraharmonic mean filtering.
    start = time.time()
    cthar_mean_filtered = cthar_mean(g=circuitboard_salt, m=3, n=3, q=-1.5) 
    stop = time.time()
    print("Filtered circuitboard-gaussian.tif using cthar mean filter in %.2f seconds" %
          (stop - start))
    # Apply min filtering.
    start = time.time()
    min_filtered = min_filter(g=hubble, m=17, n=17) 
    stop = time.time()
    print("Filtered hubble.tif using min filter in %.2f seconds" %
          (stop - start))
    # Apply median filtering.
    start = time.time()
    median_filtered = median_filter(g=circuitboard_saltandpep, m=7, n=7) 
    stop = time.time()
    print("Filtered circuitboard-saltandpep.tif using median filter in %.2f seconds" %
          (stop - start))
    # Apply adaptive median filtering.
    start = time.time()
    adapt_median_filtered = adapt_median_filter(g=circuitboard_saltandpep, s_xy=3, s_max=7) 
    stop = time.time()
    print("Filtered circuitboard-saltandpep.tif using adapt median filter in %.2f seconds" %
          (stop - start))

    # Show original and filtered images.
    fig1 = plt.figure(num=1, figsize=(20, 6))
    fig1.add_subplot(1, 3, 1)
    plt.imshow(circuitboard_gaussian, cmap="gray")
    plt.axis("off")
    plt.title("Original Image")
    fig1.add_subplot(1, 3, 2)
    plt.imshow(arith_mean_filtered, cmap="gray")
    plt.axis("off")
    plt.title("Arithmetic Mean filtered") 
    fig1.add_subplot(1, 3, 3)
    plt.imshow(geo_mean_filtered, cmap="gray")
    plt.axis("off")
    plt.title("Geometric Mean filtered") 

    fig2 = plt.figure(num=2, figsize=(20, 6))
    fig2.add_subplot(1, 2, 1)
    plt.imshow(circuitboard_salt, cmap="gray")
    plt.axis("off")
    plt.title("Original Image")
    fig2.add_subplot(1, 2, 2)
    plt.imshow(har_mean_filtered, cmap="gray")
    plt.axis("off")
    plt.title("Harmonic Mean filtered") 

    fig3 = plt.figure(num=3, figsize=(20, 6))
    fig3.add_subplot(1, 2, 1)
    plt.imshow(circuitboard_salt, cmap="gray")
    plt.axis("off")
    plt.title("Original Image")
    fig3.add_subplot(1, 2, 2)
    plt.imshow(cthar_mean_filtered, cmap="gray", vmin=0, vmax=255)
    plt.axis("off")
    plt.title("Contraharmonic Mean filtered") 

    fig4 = plt.figure(num=4, figsize=(20, 6))
    fig4.add_subplot(1, 2, 1)
    plt.imshow(hubble, cmap="gray")
    plt.axis("off")
    plt.title("Original Image")
    fig4.add_subplot(1, 2, 2)
    plt.imshow(min_filtered, cmap="gray", vmin=0, vmax=255)
    plt.axis("off")
    plt.title("Contraharmonic Mean filtered") 

    fig5 = plt.figure(num=5, figsize=(20, 6))
    fig5.add_subplot(1, 3, 1)
    plt.imshow(circuitboard_saltandpep, cmap="gray")
    plt.axis("off")
    plt.title("Original Image")
    fig5.add_subplot(1, 3, 2)
    plt.imshow(median_filtered, cmap="gray")
    plt.axis("off")
    plt.title("Median filtered") 
    fig5.add_subplot(1, 3, 3)
    plt.imshow(adapt_median_filtered, cmap="gray")
    plt.axis("off")
    plt.title("Adaptive median filtered") 

    # Show all images.
    plt.show()