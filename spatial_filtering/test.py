# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Author: Edgar Alejandro Recancoj Pajarez
# W01404391
# ECE 5220 - Image procesing
# Homework 2 - Test cases
# Date: 09/14/2021
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Import modules
from numpy.lib.arraypad import pad
from filtering import conv_2d, gauss_kernel, sharpen
import imageio
from matplotlib import pyplot as plt
import numpy as np
import time

# Convolution tests:
def test_convolve_small_1():
    print("-> test_convolve_small_1()")
    img = np.zeros((7, 7))
    kernel = np.array([[1, 2, 3], 
                       [4, 5, 6],
                       [7, 8, 9]])
    img[3][3] = 1
    print("Original img:")
    print(img)
    print("")
    print("Kernel:")
    print(kernel)
    print("")
    convolved = conv_2d(img, kernel)
    print("Convolved img:")
    print(convolved)
    print("")

def test_convolve_small_2():
    print("-> test_convolve_small_2()")
    img = np.arange(16).reshape((4,4)) + 1
    kernel = np.array([[-1, 0, 1], 
                       [-1, 0, 1],
                       [-1, 0, 1]])
    print("Original img:")
    print(img)
    print("")
    convolved = conv_2d(img, kernel)
    print("Convolved img:")
    print(convolved)
    print("")
    

def test_convolve_big():
    print("-> test_convolve_small_1()")
    img = np.zeros((512, 512))
    kernel = np.array([[1, 2, 3], 
                       [4, 5, 6],
                       [7, 8, 9]])
    img[256][256] = 1
    print("Original img:")
    print(img)
    print("")
    print("Kernel:")
    print(kernel)
    print("")
    convolved = conv_2d(img, kernel)
    print("Convolved img:")
    print(convolved)
    print("")

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
    print("ECE 5220 - Homework 2 - Tests\n")
    #test_blur()
    test_sharpening()
    #test_convolve_small_2()