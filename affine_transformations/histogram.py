# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Author: Edgar Alejandro Recancoj Pajarez
# W01404391
# ECE 5220 - Image procesing
# Homework 1
# Date: 09/11/2021
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Import modules
import numpy as np

def image_hist(f, mode="n"):
    """
    Compute histogram of 256-level grayscale image whose intensities are nonnegative.

    :param f: Numpy array of the image.
    :param mode: If "n", histogram is normalized. If "u", histogram is unnormalized.
    :returns: Computed histogram as a Numpy array.
    """
    # Create numpy array that stores unnormalized histogram.
    hist = np.zeros((256), dtype=np.uint64) 
    # Check the number of times that each level in the 8-bit image occurs.
    for i in range(256):
        hist[i] = np.count_nonzero(f == i)
    if mode == "u":
        # Return the unnormalized histogram if mode is "u"
        return hist
    else:
        # Create the normalized histogram based on the unnormalized 
        # histogram and the total number of pixels in the image
        # i.e. M rows x N columns.
        return hist/(f.shape[0]*f.shape[1])

def int_xform(f, mode, param = None):
    """
    Transform intensities of an input 8-bit grayscale image.

    :param f: Numpy array of the image in the range [0, 1]
    :param mode: Types of intensity transformations (negative, log or gamma).
    :param param: When mode is "gamma", "param" is the scalar value related to
    gamma in the intensity transformation.
    :returns: Processed image in the range [0, 1] 
    """

    # Choose between the different modes.
    if mode == "negative":
        # Apply the negative of the image.
        return 1 - f
    elif mode == "log":
        # Apply the log transformation to the image.
        return np.log(1 + f)
    elif mode == "gamma":
        # Apply the power-law (gamma) trasformation to the image
        # where param is equal to the gamma factor.
        return np.power(f, param)
    else:
        # When mode is neither negative, log, or gamma, function
        # defaults to negative.
        return 1 - f

def hist_equal(f):
    """
    Perform histogram equalization on 8-bit image.

    :param f: Numpy array of the input image.
    :returns: Processed image.
    """
    # Obtained normalized histogram for input image f.
    normalized_hist = image_hist(f, "n")
    # Apply the transformation function to find equalized histogram.
    # Such values are rounded to their neared integer.
    equalized_values = np.uint64(np.rint(255 * np.cumsum(normalized_hist)))
    # Return the processed image after the histogram equalization.
    return equalized_values[f]