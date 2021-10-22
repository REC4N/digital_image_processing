# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Author: Edgar Alejandro Recancoj Pajarez
# W01404391
# ECE 5220 - Image procesing
# Homework 5
# Date: 10/09/2021
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Import modules
import imageio
from matplotlib import pyplot as plt
from numba import jit
import numpy as np
import time

def motion_blur_tf(P, Q, a, b, T):
    """
    Create an blurring transfer function H with size P x Q.

    :param P: Size of filter in u direction.
    :param Q: Size of filter in v direction.
    :param a: Parameter in eq. 5-77 for blurring effect.
    :param b: Parameter in eq. 5-77 for blurring effect.
    :param T: Parameter in eq. 5-77 for blurring effect.
    :returns: Numpy array H representing the transfer function.
    """
    # Create transfer function H.
    H = np.zeros((P, Q), dtype=np.complex128)
    # Process blurring transfer function using equation 5-77 from book.
    create_motion_blur_tf(H, a, b, T)
    return H

@jit(nopython=True, fastmath=True)
def create_motion_blur_tf(H, a, b, T):
    """
    Helper function to create motion blur transfer function. 
    Separated from main function for Numba implementation.

    :param H: Numpy array representing the transfer function filter.
    :param a: Parameter in eq. 5-77 for blurring effect.
    :param b: Parameter in eq. 5-77 for blurring effect.
    :param T: Parameter in eq. 5-77 for blurring effect.
    :returns: (Void)
    """

    P, Q = H.shape
    for u in range(P):
        for v in range(Q):
            expression = np.pi * ((u - P//2)*a + (v - Q//2)*b)
            if expression == 0:
                expression = 0.000000001 
            H[u, v] = (T / expression) * np.sin(expression) * np.e ** (-1j * expression)

def weiner_tf(H, K):
    """
    Create parametric Weiner transfer function W. 

    :param H: Numpy array representing a degradation transfer function.
    :param K: Parameter for parametric Weiner transfer function from eq. 5-85.
    :returns: Numpy array W representing the Weiner transfer function.
    """

    W = (1 / H) * ((np.conjugate(H) * H) / ((np.conjugate(H) * H) + K))
    return W

def constrained_ls_tf(H, gam):
    """
    Create constrained least squares transfer function C. 

    :param H: Numpy array representing a degradation transfer function.
    :param gam: Parameter for constrained least squares transfer function from eq. 5-89.
    :returns: Numpy array C representing the constrained least squares transfer function.
    """
    # Create a laplacian transfer function with size given by H.
    P, Q = H.shape
    L = laplacian_tf(P=P, Q=Q)
    # Calculate constrained least squares transfer function.
    C = np.conjugate(H) / ((np.conjugate(H) * H) + gam * (np.conjugate(L) * L))
    return C

def laplacian_tf(P, Q):
    """
    Create laplacian transfer function filter with size P x Q.

    :param P: Size of filter in u direction.
    :param Q: Size of filter in v direction.
    :returns: Numpy array H representing the laplacian transfer function.
    """

    # Create transfer function H.
    H = np.zeros((P, Q), dtype=np.complex128)
    # Create laplacian transfer function.
    laplacian = np.array([[0, -1, 0], 
                         [-1, 4, -1], 
                         [0, -1, 0]], dtype=np.int64)
    # Center laplacian kernel on P x Q rectangle.
    H[int(np.floor(P/2))-1: int(np.floor(P/2))+2, int(np.floor(Q/2))-1: int(np.floor(Q/2))+2] = laplacian
    # Calculate laplacian transfer function.
    H = np.fft.fftshift(np.fft.fft2(H))
    return H

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
        f_padded = np.pad(f, ((0, M), (0, N)), mode=type).astype(np.float64)
    elif padmode == "no_pad":
        f_padded = f
    else:
        type = "constant"
        f_padded = np.pad(f, ((0, M), (0, N)), mode=type).astype(np.float64)
    F = np.fft.fftshift(np.fft.fft2(f_padded))
    # Multiply F with the filter transfer function H.
    G = H * F
    # Step 6: Obtain the filtered image of size P x Q.
    g_padded = np.real(np.fft.ifft2(np.fft.fftshift(G)))
    # Step 7: Extract the M x N region of g_padded for final result.
    g = g_padded[:M, :N]
    return g

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
    # **********************************************************************************
    # Section: Motion blurring ********************************************************
    # **********************************************************************************

    # Read image to blur.
    boy = imageio.imread("boy.tif")
    M_boy, N_boy = boy.shape
    P_boy = M_boy
    Q_boy = N_boy

    # Calcule H with a = b = 0.1 and T = 1.
    start = time.time()
    motion_blur_filter = motion_blur_tf(P=P_boy, Q=Q_boy, a=0.1, b=0.1, T=1)
    stop = time.time()
    spectrum_H = scale_img(np.abs(motion_blur_filter))
    print("Created motion blur transfer function in %.2f seconds." % (stop - start))
    # Display spectrum of motion blur transfer funtion H.
    plt.figure(1)
    plt.imshow(spectrum_H.astype(np.uint8), cmap="gray")
    plt.title("Spectrum of Motion blur Transfer function")

    # Apply blurring to image.
    start = time.time() 
    blurred_image = dft_filtering(f=boy, H=motion_blur_filter, padmode="no_pad")
    stop = time.time()
    # Create images to display.
    fig2 = plt.figure(num=2, figsize=(14, 5))
    fig2.add_subplot(1, 2, 1)
    plt.imshow(boy, cmap="gray")
    plt.axis("off")
    plt.title("Original image")
    fig2.add_subplot(1, 2, 2)
    plt.imshow(blurred_image, cmap="gray")
    plt.axis("off")
    plt.title("Blurred image")

    # **********************************************************************************
    # Section: Parametric Weiner filter ************************************************
    # **********************************************************************************

    # Read image to restore using the parametric Weiner filter.
    boy_blurred = imageio.imread("boy-blurred.tif")
    M_blur, N_blur = boy_blurred.shape
    # Create motion blur degradation function without padding.
    degradation_motion_blur = motion_blur_tf(P=M_blur, Q=N_blur, a=0.1, b=-0.1, T=1)
    # Create parametric Wiener filter transfer function to restore blurred image.
    weiner_filter = weiner_tf(H=degradation_motion_blur, K=0.0001)
    # Restore blurred image.
    start = time.time() 
    restored_image_weiner = scale_img(dft_filtering(f=boy_blurred, H=weiner_filter, padmode="no_pad"))
    stop = time.time()
    print("Restored image using parametric Wiener filter transfer function in %.2f seconds." 
          % (stop - start))


    # **********************************************************************************
    # Section: Constrained least squares filter ****************************************
    # **********************************************************************************

    # Create constrained least squares filter transfer function to restore blurred image.
    constrained_ls_filter = constrained_ls_tf(H=degradation_motion_blur, gam=0.0001)
    # Restore blurred image.
    start = time.time() 
    restored_image_constrained = scale_img(dft_filtering(f=boy_blurred, H=constrained_ls_filter, padmode="no_pad"))
    stop = time.time()
    print("Restored image using constrained least squares filter transfer function in %.2f seconds." 
          % (stop - start))

    # Create images to display.
    fig3 = plt.figure(num=3, figsize=(14, 5))
    fig3.add_subplot(1, 3, 1)
    plt.imshow(boy_blurred, cmap="gray")
    plt.axis("off")
    plt.title("Original image")
    fig3.add_subplot(1, 3, 2)
    plt.imshow(restored_image_weiner.astype(np.uint8), cmap="gray")
    plt.axis("off")
    plt.title("Restored image (Wiener)")
    fig3.add_subplot(1, 3, 3)
    plt.imshow(restored_image_constrained.astype(np.uint8), cmap="gray")
    plt.axis("off")
    plt.title("Restored image (CLS)")

    # Show figures.
    plt.show()


