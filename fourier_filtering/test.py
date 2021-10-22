# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Author: Edgar Alejandro Recancoj Pajarez
# W01404391
# ECE 5220 - Image procesing
# Homework 2 - Test cases
# Date: 09/14/2021
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Import modules
from fourier_filtering import lp_filter_tf, minus_one, dft2D, idft2D
import imageio
from matplotlib import pyplot as plt
import numpy as np
import time

# Convolution tests:
def test_fourier_shift():
    print("-> test_fourier_shift()")
    f = np.arange(24).reshape(4, 6)
    print("Original img:")
    print(f)
    g = minus_one(f)
    print("Shifted img:")
    print(g)
    print("")

def test_2d_fft():
    print("-> test_2d_fft()")
    f = np.arange(16).reshape((4,4))
    print("f:")
    print(f)
    print("")
    F = dft2D(f)
    print("F")
    print(F)
    print("")
    

def test_2d_ifft():
    print("-> test_2d_ifft()")
    f = np.arange(16).reshape((4,4))
    print("f:")
    print(f)
    print("")
    F = dft2D(f)
    print("F")
    print(F)
    print("")
    test = idft2D(F)
    print("f")
    print(test)
    print("")

def test_lp_ideal():
    print("-> test_lp_ideal()")
    H = lp_filter_tf("ideal", P=512, Q=512, D_0=100)
    plt.figure(1)
    plt.title("test_lp_ideal()")
    plt.imshow(H, cmap="gray")
    plt.show()

def test_lp_gaussian():
    print("-> test_lp_gaussian()")
    H = lp_filter_tf("gaussian", P=512, Q=512, D_0=40)
    plt.figure(1)
    plt.title("test_lp_gaussian()")
    plt.imshow(H, cmap="gray")
    plt.show()

def test_lp_butterworth():
    print("-> test_lp_butterworth()")
    H = lp_filter_tf("butterworth", P=512, Q=512, D_0=80, n=2)
    plt.figure(1)
    plt.title("test_lp_butterworth()")
    plt.imshow(H, cmap="gray")
    plt.show()

# Lowpass filtering tests
def test_gaussian_filter():
    print("-> test_gaussian_filter()")
    kernel = gauss_kernel(3, 1, 1)
    print(kernel)

def test_blur():
    m = 120 
    sig = 20 
    K = 1
    img = imageio.imread("testpattern1024.tif")
    print(img)
    kernel = gauss_kernel(m, sig, K)
    start = time.time()
    blurred = conv_2d(img, kernel)
    end = time.time()
    print(np.amin(blurred))
    print(np.amax(blurred))
    print("Blurring with %i x %i kernel (sig=%i, K=%i) took %f seconds"
            % (m, m, sig, K, (end - start)))
    print()
    plt.figure(1)
    plt.imshow(img, cmap="gray")
    plt.figure(2)
    plt.imshow(blurred, cmap="gray")
    plt.show()

def test_sharpening():
    img = imageio.imread("blurry-moon.tif")
    img = imageio.imread("testpattern1024.tif")
    start = time.time()
    sharpened = sharpen(img, "unsharp_masking")
    end = time.time()
    print("Unsharp mask took %f seconds"
            % ((end - start)))
    start = time.time()
    sharpened2 = sharpen(img, "highboost")
    end = time.time()
    print("Highboost took %f seconds"
            % ((end - start)))
    start = time.time()
    sharpened3 = sharpen(img, "laplacian")
    end = time.time()
    print("Laplacian took %f seconds"
            % ((end - start)))
    start = time.time()
    sharpened4 = sharpen(img, "sobel")
    end = time.time()
    print("Sobel took %f seconds"
            % ((end - start)))
    plt.figure(1)
    plt.imshow(img, cmap="gray")
    plt.figure(2)
    plt.imshow(sharpened.astype(np.uint8), cmap="gray")
    plt.figure(3)
    plt.imshow(sharpened2.astype(np.uint8), cmap="gray")
    plt.figure(4)
    plt.imshow(sharpened3.astype(np.uint8), cmap="gray")
    plt.figure(5)
    plt.imshow(sharpened4.astype(np.uint8), cmap="gray")
    plt.show()



if __name__ == "__main__":
    print("ECE 5220 - Homework 3 - Tests\n")
    #test_fourier_shift()
    #test_2d_ifft()
    test_lp_butterworth() 