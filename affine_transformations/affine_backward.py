# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Author: Edgar Alejandro Recancoj Pajarez
# W01404391
# ECE 5220 - Image procesing
# Homework 1
# Description: 
# Date: 09/11/2021
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Import modules
from matplotlib import pyplot as plt
from numba import jit
import numpy as np


def image_translate(f, tx, ty, mode="black"):
    """
    Translate an image tx pixels in the x direction and ty pixels in the y direction.

    :param f: Numpy array of the image.
    :param tx: Translation factor in x direction.
    :param ty: Translation factor in y direction.
    :param mode: If "black", background is black. If "white", background is white.
    :returns: Translated image with each intensity value as an 8 bit unsigned integer.
    """

    # Create affine matrix for translation operations.
    translation_matrix = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float64)
    # Create matrix for the ouput translated image.
    translated_img = np.zeros((f.shape[0], f.shape[1]), dtype=np.float64)
    # Process the image with the translation matrix.
    process_image(f, translation_matrix, translated_img)
    # Determine an intensity value for the empty space after translating image.
    # If mode is neither black or white, default to black.
    if mode == "black":
        fill_intensity = 0
    elif mode == "white":
        fill_intensity = 255
    else:
        fill_intensity = 0
    # Fill the empty spaces in the image in the x direction.
    if tx > 0:
        translated_img[:tx,:] = fill_intensity
    elif tx < 0:
        translated_img[tx:,:] = fill_intensity
    # Fill the empty spaces in the image in the y direction.
    if ty > 0:
        translated_img[:,:ty] = fill_intensity
    elif ty < 0:
        translated_img[:,ty:] = fill_intensity
    
    # Return the processed image as an numpy array of 8-bit unsigned integers.
    return translated_img.astype(np.uint8)


def image_scaling(f, cx, cy):
    """
    Scale an image cx times in the x direction and cy times in the y direction.

    :param f: Numpy array of the image.
    :param cx: Positive scaling factor in x direction.
    :param cy: Positive scaling factor in y direction.
    :returns: Scaled image with each intensity value as an 8 bit unsigned integer.
    """

    # Create affine matrix for scaling operations.
    scaling_matrix = np.array([[cx, 0, 0], [0, cy, 0], [0, 0, 1]], dtype=np.float64)
    # Create matrix for the ouput scaled image.
    scaled_img = np.zeros((cx * f.shape[0], cy * f.shape[1]), dtype=np.float64)
    # Process the image with the scaling matrix.
    process_image(f, scaling_matrix, scaled_img)
    # Return the processed image as an numpy array of 8-bit unsigned integers.
    return scaled_img.astype(np.uint8)


def image_shear(f, sv, sh):
    """
    Shear an image sv times in x direction and sh times in y direction. 

    :param f: Numpy array of the image.
    :param sv: Shear scaling factor in x direction.
    :param sh: Shear scaling factor in y direction.
    :returns: Sheared image with each intensity value as an 8 bit unsigned integer.
    """

    # Create affine matrix for shear operations.
    shear_matrix = np.array([[1, sv, 0], [sh, 1, 0], [0, 0, 1]], dtype=np.float64)
    # Create matrix for the ouput sheared image.
    sheared_img = np.zeros((f.shape[0], f.shape[1]), dtype=np.float64)
    # Process the image with the shear matrix.
    process_image(f, shear_matrix, sheared_img)
    # Return the processed image as an numpy array of 8-bit unsigned integers.
    return sheared_img.astype(np.uint8)


def image_rotate(f, theta, mode="crop"):
    """
    Rotate an image about its center, where theta is angle of rotation in degrees.
    A positive angle produces counter-clockwise rotation. 

    :param f: Numpy array of the image.
    :param theta: Angle of rotation in degrees.
    :param mode: If "crop", rotated image is cropped about its center to the same size
    as the input image. If "full", rotated image is the smallest size capable of
    containing the full rotated image for any angle.
    :returns: Rotated image with each intensity value as an 8 bit unsigned integer.
    """
    # Convert angle from degrees to radians.
    theta = np.radians(theta)
    # Calculate coordinates of image center.
    x_mid = np.floor(f.shape[0])
    y_mid = np.floor(f.shape[1])
    # Create affine matrix for rotation operations with respect to the origin.
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]], dtype=np.float64)
    forward_translation_matrix = np.array([[1, 0, x_mid], [0, 1, y_mid], [0, 0, 1]], dtype=np.float64)
    backward_translation_matrix = np.array([[1, 0, -x_mid], [0, 1, -y_mid], [0, 0, 1]], dtype=np.float64)
    final_matrix = forward_translation_matrix @ rotation_matrix @ backward_translation_matrix
    # Create matrix for the ouput sheared image.
    rotated_img = np.zeros((f.shape[0], f.shape[1]), dtype=np.float64)
    # Process the image with the shear matrix.
    process_image(f, final_matrix, rotated_img)
    # Return the processed image as an numpy array of 8-bit unsigned integers.
    return rotated_img.astype(np.uint8)

@jit(nopython=True)
def process_image(img, affine_matrix, processed_img):
    """
    Process image with the given affine matrix.

    :param img: Numpy array to be processed.
    :param affine_matrix: Affine matrix used for processing.
    :param processed_img: Write the processed array to processed_img
    :returns: (Void)
    """

    # Calculate inverse of affine matrix to use inverse mapping.
    T_inverse = np.linalg.inv(affine_matrix)
    # For each pixel in output image, use inverse mapping to find its intensity values.
    for i, row in enumerate(processed_img):
        for j in range(len(row)):
            processed_img[i, j] = apply_transformation(i, j, img, T_inverse)

@jit(nopython=True)
def apply_transformation(out_i, out_j, input_img, T_inverse):
    """
    Scan the output pixel locations and compute the corresponding location
    in the input image. Also, interpolate among the nearest input pixels
    to determine the intensity of the ouput pixel using nearest-neighbor
    interpolation.

    :param out_i: Index of the row of the output image pixel.
    :param out_j: Index of the column of the output image pixel.
    :param input_img: Numpy array containing intensity values of the image.
    :param T_inverse: Inverse of the transform matrix to use on input image.
    :returns: Intensity value of the input image that corresponds to the
    specified location in the ouput image given by out_i and out_j.
    """
    
    # Obtain max dimensions of the input image (M rows x N columns).
    M, N = input_img.shape[0] - 1, input_img.shape[1] - 1
    # Use inverse mapping to obtain input image pixel location that
    # corresponds to specified ouput image pixel. i.e. (i, j) = T_inverse(i', j')
    # Note: x and y are floating point values.
    x, y, _ = np.dot(T_inverse, np.array([out_i, out_j, 1], dtype=np.float64))
    # Interpolate using nearest neighbor to obtain intensity level of
    # valid input image pixel that corresponds to its output image pixel.
    intensity = interpolate(x, y, M, N, input_img, 0)
    # Return intensity level of the pixel found using interpolation.
    return intensity

@jit(nopython=True)
def interpolate(i, j, i_max, j_max, img, fill_intensity):
    """
    Interpolate among near pixels using nearest-neighbor interpolation.

    :param i: Decimal value to interpolate in x direction.
    :param j: Decimal value to interpolate in y direction.
    :param i_max: Index of maximum valid pixel of image in x direction.
    :param j_max: Index of maximum valid pixel of image in y direction.
    :param img: Numpy array containing intensity values of a image.
    :returns: Intensity value given by image at specified pixel.
    """

    # Check if input pixel is the desired pixel.
    if np.floor(i) == i and np.floor(j) == j:
        i, j = int(i), int(j)
    else:
        # Interpolate in x direction.
        if np.abs(i - np.floor(i)) < np.abs(i - np.ceil(i)):
            # Set pixel x location near its nearest lowest integer.
            i = int(np.floor(i))
        else:
            # Set pixel x location near its nearest highest integer.
            i = int(np.ceil(i))
        # Interpolate in y direction.
        if np.abs(j - np.floor(j)) < np.abs(j - np.ceil(j)):
            # Set pixel y location near its nearest lowest integer.
            j = int(np.floor(j))
        else:
            # Set pixel x location near its nearest highest integer.
            j = int(np.ceil(j))
        # When pixels are out of bounds, set their intensity depending
        # on the mode. For "black", intensity level is 0. For "white",
        # intensity level is 255.
        if i < 0 or i > i_max or j < 0 or j > j_max:
            return fill_intensity
    # Return the location of desired pixel in the input image to obtain the
    # intensity level (supposing that a inverse mapping was used).
    return img[i, j]