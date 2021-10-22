# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Author: Edgar Alejandro Recancoj Pajarez
# W01404391
# ECE 5220 - Image procesing
# Homework 1
# Date: 09/11/2021
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Import modules
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
    translation_matrix = np.array([[1, 0, tx], 
                                   [0, 1, ty], 
                                   [0, 0, 1]], 
                                   dtype=np.float64)
    # Create matrix for the output translated image.
    if mode == "black":
        translated_img = np.zeros((f.shape[0], f.shape[1]), dtype=np.float64)
    elif mode == "white":
        translated_img = np.full((f.shape[0], f.shape[1]), 255, dtype=np.float64)
    else:
        # If mode is neither black nor white, default to black.
        translated_img = np.zeros((f.shape[0], f.shape[1]), dtype=np.float64)
    # Process the image with the translation matrix.
    process_image(f, translation_matrix, translated_img)
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
    scaling_matrix = np.array([[cx, 0, 0], 
                               [0, cy, 0], 
                               [0, 0, 1]], 
                               dtype=np.float64)
    # Create matrix for the ouput scaled image.
    scaled_img = np.zeros((np.uint64(np.ceil(cx * f.shape[0])), 
                           np.uint64(np.ceil(cy * f.shape[1]))), 
                           dtype=np.float64)
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
    shear_matrix = np.array([[1, sv, 0], 
                             [sh, 1, 0], 
                             [0, 0, 1]], 
                             dtype=np.float64)
    # Create matrix for the ouput sheared image.
    sheared_img = np.zeros((f.shape[0], f.shape[1]), dtype=np.float64)
    # Calculate translation transformations to show image in the center.
    x_mid = np.floor(f.shape[0]/2)
    y_mid = np.floor(f.shape[1]/2)
    forward_translation_matrix = np.array([[1, 0, x_mid], 
                                           [0, 1, y_mid], 
                                           [0, 0, 1]], 
                                           dtype=np.float64)
    backward_translation_matrix = np.array([[1, 0, -x_mid], 
                                            [0, 1, -y_mid], 
                                            [0, 0, 1]], 
                                            dtype=np.float64)
    final_matrix = forward_translation_matrix @ shear_matrix @ backward_translation_matrix
    # Process the image with the transformation matrix.
    process_image(f, final_matrix, sheared_img)
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
    theta_rad = np.deg2rad(theta)
    if mode == "full":
        # Calculate dimensions of new image when in full mode.
        M = np.uint64(np.ceil(np.abs(f.shape[1] * np.sin(theta_rad)) 
                              + np.abs(f.shape[0] * np.cos(theta_rad))))
        N = np.uint64(np.ceil(np.abs(f.shape[1] * np.cos(theta_rad)) 
                              + np.abs(f.shape[0] * np.sin(theta_rad))))
        rotated_img = np.zeros((M, N), dtype=np.float64)
    else:
        # If mode is not full, then use crop mode.
        M, N = f.shape[0], f.shape[1]
    # Create matrix for the ouput rotated image.
    rotated_img = np.zeros((M, N), dtype=np.float64)
    # Calculate coordinates of image center.
    x_mid = np.floor(M/2)
    y_mid = np.floor(N/2)
    # Calculate distance to move image back after rotating in order for it to appear correctly.
    x_mid_back = np.floor(f.shape[0]/2)
    y_mid_back = np.floor(f.shape[1]/2)
    # Create affine matrix for rotation operations with respect to the center of image.
    rotation_matrix = np.array([[np.cos(theta_rad), -np.sin(theta_rad), 0], 
                                [np.sin(theta_rad), np.cos(theta_rad), 0], 
                                [0, 0, 1]], 
                                dtype=np.float64)
    # Create translation matrices used to show image in the center.
    forward_translation_matrix = np.array([[1, 0, x_mid], 
                                          [0, 1, y_mid], 
                                          [0, 0, 1]], 
                                          dtype=np.float64)
    backward_translation_matrix = np.array([[1, 0, -x_mid_back], 
                                            [0, 1, -y_mid_back], 
                                            [0, 0, 1]], 
                                            dtype=np.float64)
    final_matrix = forward_translation_matrix @ rotation_matrix @ backward_translation_matrix
    # Process the image with the transformation matrix.
    process_image(f, final_matrix, rotated_img)
    # Return the processed image as an numpy array of 8-bit unsigned integers.
    return rotated_img.astype(np.uint8)

@jit(nopython=True)
def process_image(input_img, transformation_matrix, processed_img):
    """
    Scan the input pixel locations and compute the corresponding location
    in the output image. Also, interpolate among the nearest input pixels
    to determine the intensity of the ouput pixel using nearest-neighbor
    interpolation.

    :param input_img: Numpy array to be processed.
    :param transformation_matrix: Transformation matrix used for processing.
    :param processed_img: Write the processed array to processed_img
    :returns: (Void)
    """

    # Obtain max dimensions of the input image and output image (M rows x N columns).
    M_out, N_out = processed_img.shape[0], processed_img.shape[1]
    M_input, N_input = input_img.shape[0], input_img.shape[1]
    # Calculate inverse matrix
    T_inverse = np.linalg.inv(transformation_matrix)
    # For each pixel in output image, use inverse mapping to find its intensity values.
    for i in range(M_out):
        for j in range(N_out):
            # Create output pixel (that is also the current pixel during iteration).
            output_pixel = np.array([i, j, 1], dtype=np.float64)
            # Calculate input pixel using inverse mapping.
            input_pixel = np.dot(T_inverse, output_pixel)
            # Check if input pixel is withing valid bounds.
            if input_pixel[0] > M_input - 1 or input_pixel[0] < 0 or input_pixel[1] > N_input - 1  or input_pixel[1] < 0:
                # If pixel is not within valid bounds, continue to next iteration.
                continue
            else:
                # If pixel is within valid bounds, use interpolation to obtain intensity
                # value of the corresponding pixel in the processed image.
                interpolate(output_pixel, input_pixel, input_img, processed_img) 

@jit(nopython=True)
def interpolate(output_pixel, input_pixel, input_img, output_img):
    """
    Interpolate among near pixels using nearest-neighbor interpolation.

    :param output_pixel: Numpy array containing location of output pixel. 
    :param input_pixel: Numpy array containing location of input pixel (the one to interpolate)
    :param input_img: Numpy array containing intensity values of input image.
    :param output_img: Numpy array containing intensity values of output image.
    :returns: (Void)
    """
    # Define input pixel location as (i, j).
    i, j, _ = input_pixel
    # Define output pixel location as (x, y).
    x, y, _ = output_pixel 
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
    # Map the intensity level on the output image based on the input image pixel.
    output_img[np.uint64(x)][np.uint64(y)] = input_img[i][j] 
