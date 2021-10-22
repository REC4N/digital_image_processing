# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Author: Edgar Alejandro Recancoj Pajarez
# W01404391
# ECE 5220 - Image procesing
# Homework 2 - Spatial filtering
# Date: 09/14/2021
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Import modules
import imageio
from matplotlib import pyplot as plt
from numba import jit
import numpy as np

def conv_2d(f, w):
    """
    Perform a 2D convolution to image using a specified kernel. Replicate padding is used.

    :param f: Numpy array of the image.
    :param w: Numpy array of the kernel.
    :returns: Convolved image with each intensity value as an 8 bit unsigned integer.
    """

    # Determine the number of zeros to be added to the boundary of the image.
    pad_num = int(np.floor(w.shape[0]/2))
    # Create a padded array with replicate padding.
    pad_img = np.pad(f, pad_num, mode="edge").astype(np.float64)
    # Flip the kernel array in both the left/right and up/down direcitons.
    flipped_kernel = np.flip(w)
    # Create array that contains the convolved image.
    convolved_img = np.zeros_like(f).astype(np.float64)
    # Perform 2D convolution and store output on convolved_img.
    process_convolution(convolved_img, flipped_kernel, pad_img)
    return convolved_img

@jit(nopython=True, fastmath=True)
def process_convolution(convolved_img, flipped_kernel, pad_img):
    """
    Perform 2D convolution calculations. Separated from conv_2d to use Numba.

    :param convolved_img: Numpy array that holds information from applying convolution to
    original image.
    :param flipped_kernel: Numpy array of the kernel that is flipped in both axis.
    :param pad_img: Numpy array containing padded original image.
    :returns: (Void) Convolved image is written on the same numpy array.
    """

    # Obtain the number of rows and columns from both the convolved image and flipped kernel.
    M, N = convolved_img.shape
    m, n = flipped_kernel.shape
    # Perform 2D convolution.
    for x in range(M):
        for y in range(N):
                convolved_img[x, y] = (flipped_kernel * pad_img[x : x + m, y: y + n]).sum()

def scale_img(img, max, min):
    """
    Scale img array between min and max.

    :param img: Numpy array of the image. 
    :returns: Scaled image with values [min, max]
    """
    # Scale values of img based on min and max values specified by caller.
    img -= np.amin(img)
    img /= np.amax(img)
    img *= 255
    return img

def blur(img, m, sig, K = 1):
    """
    Blur image using Gaussian kernel.

    :param img: Numpy array of the image. 
    :param m: Dimensions of the kernel.
    :param sig: Variance or "spread" of the Gaussian function about its mean. 
    :param K: General constant used on Gaussian function.
    :returns: Numpy array of the blurred image.
    """
    max_original = np.amax(img)
    min_original = np.amin(img)
    # Create Gaussian kernel with specified values.
    kernel = gauss_kernel(m, sig, K)
    # Process image.
    blurred_img = conv_2d(img, kernel)
    return scale_img(blurred_img, max_original, min_original)

def sharpen(img, mode):
    """
    Apply sharpening to image.

    :param img: Numpy array of the image. 
    :param mode: Type of sharpening to apply. "unsharp_masking" uses unsharp masking.
    "highboost" uses same steps as "unsharp_masking" but the weight portion of
    the mask is greater than 1. "laplacian" apply sharpening using a laplacian kernel.
    "sobel" uses a sobel kernel to apply sharpening.
    :returns: Numpy array of the sharpened image.
    """

    max_original = np.amax(img)
    min_original = np.amin(img)
    if mode == "unsharp_masking":
        # In unsharp masking, weight portion of mask is equal to 1.
        k = 1
        # Step 1: Blur image with Gaussian kernel.
        blurred_img = blur(img, m=5, sig=1, K=1)
        # Step 2: Create the mask.
        mask = img - blurred_img
        # Step 3: Add weighted portion of mask back to original image.
        sharp_img = img + k * mask 
        return scale_img(sharp_img, max_original, min_original)
    elif mode == "highboost":
        # In unsharp masking, weight portion of mask is greater than 1.
        k = 3
        # Step 1: Blur image with Gaussian kernel.
        blurred_img = blur(img, m=5, sig=1, K=1)
        # Step 2: Create the mask.
        mask = img - blurred_img
        # Step 3: Add weighted portion of mask back to original image.
        sharp_img = img + k * mask 
        return scale_img(sharp_img, max_original, min_original)
    elif mode == "laplacian":
        # Create laplacian kernel.
        laplacian_ker = np.array([[0, -1, 0],
                                  [-1, 4, -1],
                                  [0, -1, 0]])
        # Step 1: Blur image with Gaussian kernel.
        blurred_img = blur(img, m=5, sig=1, K=1)
        # Step 2: Apply edge detection filter to blurred image.
        edge_detected_img = conv_2d(blurred_img, laplacian_ker)
        # Obtain sharp image.
        sharp_img = img + edge_detected_img
        return scale_img(sharp_img, max_original, min_original)
    elif mode == "sobel":
        # Create sobel kernel.
        sobel_ker_1 = np.array([[-1, -2, -1],
                                [0, 0, 0],
                                [1, 2, 1]])
        sobel_ker_2 = np.array([[-1, 0, 1],
                                [-2, 0, 2],
                                [-2, 0, 1]])
        # Step 1: Blur image with Gaussian kernel.
        blurred_img = blur(img, m=9, sig=1, K=1)
        # Step 2: Apply sobel kernels to blurred image.
        sobel_img_1 = conv_2d(blurred_img, sobel_ker_1)
        sobel_img_2 = conv_2d(sobel_img_1, sobel_ker_2)
        return scale_img(sobel_img_2, max_original, min_original)
    else:
        print("No mode selected. No sharpening applied!")
        sharp_img = img 
        return sharp_img

def gauss_kernel(m, sig, K = 1):
    """
    Generate a normalized Gaussian lowpass kernel of size m x m.

    :param m: Dimensions of the kernel.
    :param sig: Variance or "spread" of the Gaussian function about its mean. 
    :param K: General constant used on Gaussian function.
    :returns: Numpy array representing m x m Gaussian kernel.
    """

    # Find indexing offset from the dimensions given by m.
    offset = np.int64(np.floor(m / 2))
    # Create an array that contains the custom indices [-offset, offset]
    # to process Gaussian function.
    custom_indices = np.linspace(-offset, offset, m)
    # Create two 2D arrays that represent the coordinates for both i and j directions.
    i_indices, j_indices = np.meshgrid(custom_indices, custom_indices)
    # Pass the coordinates arrays into the Gaussian function to calculate
    # the unnormalized kernel.
    kernel = K * np.exp(-(np.square(i_indices) + np.square(j_indices))/(2 * sig**2))
    # Return the normalized kernel.
    return (kernel / np.sum(kernel)).astype(np.float64)

if __name__ == "__main__":
    # Everything (in theory) works, except sobel filtering (last figure)

    # Read first image to process.
    pattern = imageio.imread("testpattern1024.tif")
    # Show original test pattern image.
    plt.figure(1)
    plt.title("Test Pattern - Original image")
    plt.imshow(pattern, cmap="gray")
    # Apply blurring to test pattern image.
    blurred_img = blur(pattern, m=120, sig=20, K=1)
    
    plt.figure(2)
    plt.title("Test Pattern - Blurred image")
    plt.imshow(blurred_img.astype(np.uint8), cmap="gray")
    # Read second image to process.
    moon = imageio.imread("blurry-moon.tif")
    # Show original test pattern image.
    plt.figure(3)
    plt.title("Blurry moon - Original image")
    plt.imshow(moon, cmap="gray")
    # Apply sharpen using different types of methods.
    unsharp_img = sharpen(moon, mode="unsharp_masking")
    highboost_img = sharpen(moon, mode="highboost")
    laplacian_img = sharpen(moon, mode="laplacian")
    sobel_img = sharpen(moon, mode="sobel")
    # Show sharpened images.
    plt.figure(4)
    plt.title("Blurry moon - Unsharp masking")
    plt.imshow(unsharp_img.astype(np.uint8), cmap="gray")
    plt.figure(5)
    plt.title("Blurry moon - Highboost filtering")
    plt.imshow(highboost_img.astype(np.uint8), cmap="gray")
    plt.figure(6)
    plt.title("Blurry moon - Laplacian")
    plt.imshow(laplacian_img.astype(np.uint8), cmap="gray")
    plt.figure(7)
    plt.title("Blurry moon - Sobel")
    plt.imshow(sobel_img.astype(np.uint8), cmap="gray")
    plt.show()
