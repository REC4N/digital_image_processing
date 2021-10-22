# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Author: Edgar Alejandro Recancoj Pajarez
# W01404391
# ECE 5220 - Image procesing
# Homework 6
# Date: 10/16/2021
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Import modules
import imageio
from matplotlib import pyplot as plt
from matplotlib import gridspec
from numba import jit
import numpy as np
import time

def tmat(xform, n):
    """
    Create a transformation matrix depending on type of xform selected.

    :param xform: Type of transformation matrix to create. Valid options:
    "fourier", "hartley", "cosine", and "sine".
    :param n: Size of output transformation matrix.
    :returns: Transformation matrix A of size n x n.
    """

    # Create transformation matrix A.
    A = np.zeros((n, n))
    # Create indices matrices.
    x, u = np.meshgrid(np.arange(n), np.arange(n))
    # Calculate transformation matrix A depending on xform selected.
    if xform == "fourier":
        # Compute DFT matrix.
        A = (1 / np.sqrt(n)) * np.exp(-1j * 2 * np.pi * u * x / n)
    elif xform == "hartley":
        # Compute DFT matrix.
        A_dft = (1 / np.sqrt(n)) * np.exp(-1j * 2 * np.pi * u * x / n)
        # Obtain DHT matrix from DFT matrix.
        A = np.real(A_dft) - np.imag(A_dft)
    elif xform == "cosine":
        # Compute DCT matrix.
        A[:1] = np.sqrt(1 / n)
        A[1:] = np.sqrt(2 / n) * np.cos((2*x[1:] + 1) * u[1:] * np.pi / (2 * n))
    elif xform == "sine":
        # Compute DST matrix.
        A = np.sqrt(2 / (n + 1)) * np.sin(((x + 1) * (u + 1) * np.pi) / (n + 1))
    elif xform == "hadamard":
        # Next commented lines computes a natural ordered WH matrix using recursion.
        # Create base case for Walsh-Hadamard transformation matrix.
        # wh_base = np.array([[1, 1], [1, -1]])
        # Size of matrix.
        # N = int(np.log2(n))
        # A = (1 / np.sqrt(n)) * recursive_walsh_hadamard(N, wh_base)
        # Compute sequency ordered DWH matrix.
        A = (1 / np.sqrt(n)) * sequency_hadamard(n)
    elif xform == "slant":
        # Create base case for Slant transformation matrix.
        slant_base = np.array([[1, 1], [1, -1]])
        # Compute Slant matrix.
        A = (1 / np.sqrt(n)) * recursive_slant(n, slant_base)
    elif xform == "haar":
        # Compute Haar matrix.
        A = haar_matrix(n)
    else:
        print(    "WARNING: No valid xform selected.")
    return A

@jit(nopython=True, fastmath=True)
def sequency_hadamard(n):
    """
    Create a Walsh-Hadamard transformation matrix in sequency order.

    :param n: Size of output transformation matrix.
    :returns: Transformation matrix H of size n x n.
    """
    H = np.zeros((n, n))
    for u in range(n):
        for x in range(n):
            sum = 0
            for i in range(int(np.log2(n))):
                sum += b(i, x) * p(i, u, int(np.log2(n)))
            H[u, x] = np.power((-1), sum)
    return H

@jit(nopython=True, fastmath=True)
def b(index, z):
    """
    Helper function to compute b_i coefficient in Walsh-Hadamard matrix.

    :param index: Current index to compute.
    :param z: Binary value.
    :returns: Coefficient value.
    """
    return z >> index & 1

@jit(nopython=True, fastmath=True)
def p(index, z, n):
    """
    Helper function to compute p_i coefficient in Walsh-Hadamard matrix.

    :param index: Current index to compute.
    :param z: Binary value.
    :param n: Size of output transformation matrix.
    :returns: Coefficient value.
    """
    if index == 0:
        return b(n - 1, z)
    else:
        return b(n - index, z) + b(n - index - 1, z)


def recursive_walsh_hadamard(N, wh_matrix):
    """
    Helper function to compute natural-ordered Walsh-Hadamard tranformation matrix.
    Note: Not currently used but left here for future reference.

    :param N: Size of output transformation matrix.
    :returns: Numpy array representing natural-ordered Walsh-Hadamard matrix.
    """
    if N == 1:
        # 2**n => N
        return wh_matrix
    else:
        return np.kron(wh_matrix, recursive_walsh_hadamard(N - 1, wh_matrix))

def recursive_slant(N, slant_matrix):
    """
    Helper function to compute Slant transformation matrix.
    Important: Recursive slant matrix generation ensures correct sequency order up to N <= 7.
    For N >= 8, book suggests to use Gray Code conversion and Bit reversal to obtain
    new ordered rows. Applying such change of rows in final matrix does not produce
    correct order. Methods for such process are included in this code. Possible 
    solution is applying the row conversion with each matrix call inside the recursion.

    :param N: Size of output transformation matrix.
    :returns: Numpy array representing Slant tranformation matrix.
    """

    # Base Case:
    if N == 2:
        return slant_matrix
    else:
        # Slant recursive = S_temp1 @ S_temp2
        # Compute size of identity matrices used to create S_temp1.
        new_size = (N//2) - 2
        S_temp1 = np.zeros((N, N))
        if new_size == 0:
            I = np.array([])
        else:
            I = np.eye(new_size)
        # Compute coefficients for portions of S_temp1.
        a = np.sqrt((3 * N**2) / (4 * (N**2 - 1)))
        b = np.sqrt((N**2 - 4) / (4 * (N**2 - 1)))
        # Create S_temp1 with given specifications by book.
        # Group of columns 1
        S_temp1[0:2, 0:2] = np.array([[1, 0], [a, b]])
        S_temp1[(2 + new_size):(new_size + 4), 0:2] = np.array([[0, 1], [-b, a]])
        # Group of columns 2
        S_temp1[0:2, 0:2] = np.array([[1, 0], [a, b]])
        S_temp1[2:(2 + new_size), 2:(2 + new_size)] = I
        S_temp1[(new_size + 4):, 2:(2 + new_size)] = I
        # Group of columns 3
        S_temp1[0:2, (2 + new_size):(new_size + 4)] = np.array([[1, 0], [-a, b]])
        S_temp1[(2 + new_size):(new_size + 4), (2 + new_size):(new_size + 4)] = np.array([[0, -1], [b, a]])
        # Group of columns 4
        S_temp1[2:(2 + new_size), (new_size + 4):] = I
        S_temp1[(new_size + 4):, (new_size + 4):] = I
        # Compute dot product to create new temporal matrix.
        return S_temp1 @ (np.kron(np.array([[1, 0], [0, 1]]), recursive_slant(N//2, slant_matrix)))


def reverse_bits(n, len_bits):
    """
    Reverse bits given decimal number and length of bits used.
    Note: Method not used. Left here for future reference.
    """
    result = 0
    for i in range(len_bits):
        result <<= 1
        result |= n & 1
        n >>= 1
    return result

def convert_gray(n):
    """
    Convert number to gray code.
    Note: Method not used. Left here for future reference.
    """
    if n == 0:
        return 0
    x = 1
    while x * 2 <= n:
        x *= 2
    return x + convert_gray(2 * x - n - 1)

def sequency(row, n):
    """
    Obtain row according to sequency ordering.
    Note: Method not used. Left here for future reference.
    """
    return reverse_bits(convert_gray(row), int(np.log2(n)))

@jit(nopython=True, fastmath=True)
def haar_matrix(n):
    """
    Create Haar transformation matrix.

    :param n: Size of output transformation matrix.
    :returns: Transformation matrix of size n x n.
    """

    # Create Haar matrix.
    haar = np.zeros((n, n))
    # Value is constant for u = 0.
    haar[:1,:] = 1
    # Compute rest of matrix using transformation kernel.
    for u in range(1, n):
        for x in range(n):
            haar[u, x] = h(u, x/n)
    return (1 / np.sqrt(n)) * haar

@jit(nopython=True, fastmath=True)
def h(u, z):
    """
    Helper function to compute haar coefficients given by algorithm
    in book.
    """

    # Obtain largest power of 2 contained in u.
    p = 0 if u == 0 else int(np.floor(np.log2(u)))
    # Obtain remainder.
    q = u - 2**p
    # Output depends of value of z within range [0, 1)
    if u == 0:
        out = 1
    elif z >= q/(2**p) and z < (q + 0.5)/(2**p):
        out =  2**(p/2)
    elif z >= (q + 0.5)/(2**p) and z < (q + 1)/(2**p):
        out =  -2**(p/2)
    else:
        out = 0
    return out


def transform(f, xform):
    """
    Compute the forward transform of f depending on type of transform selected.
    (if f is 2-D, a square matrix is assumed).

    :param f: 1-D or 2-D Numpy array.
    :param xform: Type of forward transformation to compute. Valid options:
    "fourier", "hartley", "cosine", "sine", "slant", "hadamard", and "haar".
    :returns: 1-D or 2-D Numpy array t after applying transformation.
    """

    # Create transformation matrix A specified by xform.
    A = tmat(xform, f.shape[0])
    # Check if f is 1-D or 2-D.
    if f.ndim == 1:
        t = A @ f
    else:
        t = A @ f @ A.T
    return t

def inv_transform(t, xform):
    """
    Compute the inverse transform of t depending on type of transform selected.
    (if t is 2-D, a square matrix is assumed).

    :param t: 1-D or 2-D Numpy array.
    :param xform: Type of inverse transformation to compute. Valid options:
    "fourier", "hartley", "cosine", "sine", "slant", "hadamard", and "haar".
    :returns: 1-D or 2-D Numpy array f after applying transformation.
    """

    # Create transformation matrix A specified by xform.
    A = tmat(xform, t.shape[0])
    # Check if t is 1-D or 2-D.
    if t.ndim == 1:
        if xform == "fourier":
            # Special case when transformation matrix is unitary.
            f = A.conj().T @ t
        else:
            f = A.T @ t
    else:
        if xform == "fourier":
            # Special case when transformation matrix is unitary.
            f = A.conj().T @ t @ A.conj()
        else:
            f = A.T @ t @ A
    return f


@jit(nopython=True, fastmath=True)
def lp_ideal(P, Q, D_0, centered=True):
    """
    Create an ideal P x Q low pass filter with cutoff frequency D_0.

    :param P: Size of filter in u direction.
    :param Q: Size of filter in v direction.
    :param D_0: Cutoff frequency of filter.
    :returns: Numpy array H representing the low pass filter transfer function.
    """
    # Create transfer function H.
    H = np.zeros((P, Q))
    # Create low pass filter.
    for u in range(P):
        for v in range(Q):
            if centered == True:
                dist = np.sqrt((u - P/2)**2 + (v - Q/2)**2)
            else:
                dist = np.sqrt((u)**2 + (v)**2)
            if dist <= D_0:
                H[u, v] = 1
            else:
                H[u, v] = 0
    return H

def hp_ideal(P, Q, D_0, centered=True):
    """
    Create an ideal P x Q high pass filter with cutoff frequency D_0.

    :param P: Size of filter in u direction.
    :param Q: Size of filter in v direction.
    :param D_0: Cutoff frequency of filter.
    :returns: Numpy array H representing the high pass filter transfer function.
    """
    # Create low pass filter.
    H_lp = lp_ideal(P, Q, D_0, centered)
    # Create high pass filter from the low pass filter.
    H_hp = 1 - H_lp
    return H_hp

def ideal_filter(f, xform, type, r):
    """
    Filter 2-D f numpy array using ideal highpass or lowpass filtering.
    (square matrix is assumed).

    :param f: Square 2-D Numpy array.
    :param xform: Type of transformation to use. Valid options:
    "fourier", "hartley", "cosine", and "sine".
    :param type: Type of filtering. Valid options: "high" and "low".
    :param r: Cutoff frequency of filter.
    :returns: 2-D Numpy array g after applying ideal filtering.
    """

    # Pad image f in transform domain .
    M, N = f.shape
    f_padded = np.pad(f, ((0, M), (0, N)), mode="edge").astype(np.float64)
    # Compute transformation of image padded and corresponding filter.
    if xform == "fourier" or xform == "hartley":
        # Special case when transformation is DFT or DHT.
        F = np.fft.fftshift(transform(f_padded, xform))
        # Create centered ideal filter depending on type.
        if type == "low":
            H = lp_ideal(P=(2*M), Q=(2*N), D_0=r, centered=True)
        else:
            H = hp_ideal(P=(2*M), Q=(2*N), D_0=r, centered=True)
    else:
        F = transform(f_padded, xform)
        # Create ideal filter depending on type.
        if type == "low":
            H = lp_ideal(P=(2*M), Q=(2*N), D_0=r, centered=False)
        else:
            H = hp_ideal(P=(2*M), Q=(2*N), D_0=r, centered=False)
       
    # Apply filtering to image in transform domain.
    G = H * F
    # Compute inverse transformation to obtain filtered padded image.
    if xform == "fourier":
        # Special case when transformation is DFT.
        g_padded = np.real(inv_transform(np.fft.fftshift(G), xform)) 
    elif xform == "hartley":
        # Special case when transformation is DHT.
        g_padded = inv_transform(np.fft.fftshift(G), xform)
    else:
        g_padded = inv_transform(G, xform)
    # Extract the M x N region of g padded for final result.
    g = g_padded[:M, :N]
    return g

def basis_image(xform, n):
    """
    Compute basis images for a given transform.

    :param xform: Type of transformation matrix to create. Valid options:
    "fourier", "hartley", "cosine", "sine", "hadamard", "slant" and "haar".
    :param n: Size of transformation matrix.
    :returns: 3-D Numpy array i containing n x n basis images, each basis
    image of size n x n.
    """

    # Calculate basis images array i depending on xform selected.
    if xform == "fourier":
        # Create basis images 3-D array.
        i = np.zeros((n**2, n, n), dtype=np.complex128)
        basis_image = np.zeros((n, n), dtype=np.complex128)
        u_index = 0
        v_index = 0
        for count in range(n**2):
            # Compute each DFT basis image.
            compute_dft_basis_image(basis_image, u_index, v_index, n)
            # Assign the basis image into the 3-D array.
            i[count,:,:] = basis_image
            # Keep track of current basis image.
            v_index += 1
            if v_index >= n:
                u_index += 1
                v_index = 0
        # Return both real and imaginary basis images of DFT.
        return (np.real(i), np.imag(i))
    else:
        # All transforms (except DFT) do not require complex values in array
        # and return only one 3-D array.
        i = np.zeros((n**2, n, n), dtype=np.float64)
        basis_image = np.zeros((n, n), dtype=np.float64)
        u_index = 0
        v_index = 0
        if xform == "hartley":
            for count in range(n**2):
                # Compute each DHT basis image.
                compute_dht_basis_image(basis_image, u_index, v_index, n)
                # Assign the basis image into the 3-D array.
                i[count,:,:] = basis_image
                # Keep track of current basis image.
                v_index += 1
                if v_index >= n:
                    u_index += 1
                    v_index = 0
        elif xform == "cosine":
            for count in range(n**2):
                # Compute each DCT basis image.
                compute_dct_basis_image(basis_image, u_index, v_index, n)
                # Assign the basis image into the 3-D array.
                i[count,:,:] = basis_image
                # Keep track of current basis image.
                v_index += 1
                if v_index >= n:
                    u_index += 1
                    v_index = 0
        elif xform == "sine":
            for count in range(n**2):
                # Compute each DST basis image.
                compute_dst_basis_image(basis_image, u_index, v_index, n)
                # Assign the basis image into the 3-D array.
                i[count,:,:] = basis_image
                # Keep track of current basis image.
                v_index += 1
                if v_index >= n:
                    u_index += 1
                    v_index = 0
        elif xform == "hadamard":
            for count in range(n**2):
                # Compute each Walsh-Hadamard basis image.
                compute_dwht_basis_image(basis_image, u_index, v_index, n)
                # Assign the basis image into the 3-D array.
                i[count,:,:] = basis_image
                # Keep track of current basis image.
                v_index += 1
                if v_index >= n:
                    u_index += 1
                    v_index = 0
        elif xform == "haar":
            for count in range(n**2):
                # Compute each Haar basis image.
                compute_haar_basis_image(basis_image, u_index, v_index, n)
                # Assign the basis image into the 3-D array.
                i[count,:,:] = basis_image
                # Keep track of current basis image.
                v_index += 1
                if v_index >= n:
                    u_index += 1
                    v_index = 0
        else:
            print(    "WARNING: No valid xform selected.")
        return i

@jit(nopython=True, fastmath=True)
def compute_dft_basis_image(basis_img, u, v, n):
    """
    Compute basis image for DFT.

    :param basis_img: 2-D Numpy array containing current basis image.
    :param u: Current u index.
    :param v: Current v index.
    :param n: Size of transformation matrix.
    :returns: Void
    """
    for x in range(n):
        for y in range(n):
            basis_img[x, y] = (1 / n) * np.exp(-1j * 2 * np.pi * ((v * y / n) + (x * u / n)))

@jit(nopython=True, fastmath=True)
def compute_dht_basis_image(basis_img, u, v, n):
    """
    Compute basis image for DHT.

    :param basis_img: 2-D Numpy array containing current basis image.
    :param u: Current u index.
    :param v: Current v index.
    :param n: Size of transformation matrix.
    :returns: Void
    """
    for x in range(n):
        for y in range(n):
            basis_img[x, y] = (2 / n) * np.cos(((2 * np.pi * u * x) / n) - (np.pi / 4)) * np.cos(((2 * np.pi * v * y) / n) - (np.pi / 4))

@jit(nopython=True, fastmath=True)
def compute_dct_basis_image(basis_img, u, v, n):
    """
    Compute basis image for DCT.

    :param basis_img: 2-D Numpy array containing current basis image.
    :param u: Current u index.
    :param v: Current v index.
    :param n: Size of transformation matrix.
    :returns: Void
    """
    if u == 0:
        alpha_u = np.sqrt(1 / n)
    else:
        alpha_u = np.sqrt(2 / n)
    if v == 0:
        alpha_v = np.sqrt(1 / n)
    else:
        alpha_v = np.sqrt(2 / n)
    for x in range(n):
        for y in range(n):
            basis_img[x, y] = alpha_u * alpha_v * np.cos((2*x + 1) * u * np.pi / (2 * n)) * np.cos((2*y + 1) * v * np.pi / (2 * n))

@jit(nopython=True, fastmath=True)
def compute_dst_basis_image(basis_img, u, v, n):
    """
    Compute basis image for DST.

    :param basis_img: 2-D Numpy array containing current basis image.
    :param u: Current u index.
    :param v: Current v index.
    :param n: Size of transformation matrix.
    :returns: Void
    """
    for x in range(n):
        for y in range(n):
            basis_img[x, y] = (2 / (n + 1)) * np.sin(((x + 1) * (u + 1) * np.pi) / (n + 1)) * np.sin(((y + 1) * (v + 1) * np.pi) / (n + 1))

@jit(nopython=True, fastmath=True)
def compute_dwht_basis_image(basis_img, u, v, n):
    """
    Compute basis image for DWHT.

    :param basis_img: 2-D Numpy array containing current basis image.
    :param u: Current u index.
    :param v: Current v index.
    :param n: Size of transformation matrix.
    :returns: Void
    """
    for x in range(n):
        for y in range(n):
            sum = 0
            for i in range(np.log2(n)):
                sum += b(i, x) * p(i, u, int(np.log2(n))) + b(i, y) * p(i, v, int(np.log2(n)))
            basis_img[x, y] = (1 / n) * np.power((-1), sum)

@jit(nopython=True, fastmath=True)
def compute_haar_basis_image(basis_img, u, v, n):
    """
    Compute basis image for Haar transform.

    :param basis_img: 2-D Numpy array containing current basis image.
    :param u: Current u index.
    :param v: Current v index.
    :param n: Size of transformation matrix.
    :returns: Void
    """
    for x in range(n):
        for y in range(n):
            basis_img[x, y] = (1 / n) * h(u, x/n) * h(v, y/n)

def scale_img(img):
    """
    Scale img array between 0 and 255
    :param img: Numpy array of the image. 
    :returns: Scaled image with values [0, 255]
    """
    # Check if image only contains one value.
    if np.amax(img) == np.amin(img):
        img[:] = 255 
        return img
    else:
        # Scale values of img between 0 and 255.
        img -= np.amin(img)
        img /= np.amax(img)
        img *= 255
        return img

if __name__ == "__main__":
    # **********************************************************************************
    # Section: Fourier transforms ******************************************************
    # **********************************************************************************

    print("Transformation matrices ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    # Transformation matrix of DFT.
    A_fourier = tmat("fourier", 8)
    print("\nDFT:")
    with np.printoptions(precision=3, suppress=True, linewidth=np.inf):
        print(A_fourier)
    # Transformation matrix of DHT.
    A_hartley = tmat("hartley", 8)
    print("\nDHT:")
    with np.printoptions(precision=2, suppress=True, linewidth=np.inf):
        print(A_hartley)
    # Transformation matrix of DCT.
    A_cosine = tmat("cosine", 8)
    print("\nDCT:")
    with np.printoptions(precision=2, suppress=True, linewidth=np.inf):
        print(A_cosine)
    # Transformation matrix of DST.
    A_sine = tmat("sine", 8)
    print("\nDST:")
    with np.printoptions(precision=2, suppress=True, linewidth=np.inf):
        print(A_sine)
    # Transformation matrix of DWHT.
    A_hadamard = tmat("hadamard", 8)
    print("\nDWHT:")
    with np.printoptions(precision=2, suppress=True, linewidth=np.inf):
        print(A_hadamard)
    # Transformation matrix of SLT.
    A_slant = tmat("slant", 8)
    print("\nSlant transformation matrix:")
    with np.printoptions(precision=2, suppress=True, linewidth=np.inf):
        print(A_slant)
    # Transformation matrix of Haar transform.
    A_haar = tmat("haar", 8)
    print("\nHaar transformation matrix:")
    with np.printoptions(precision=2, suppress=True, linewidth=np.inf):
        print(A_haar)
    
    print("\n1-D Test cases: ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    # Create input vector.
    f = np.array([1, 3, 5, 9, 8, 2, 4, 8]).T
    print("f = ", end="")
    print(f)
    # Compute DFT.
    t_dft = transform(f, "fourier")
    print("\nDFT:")
    with np.printoptions(precision=2, suppress=True, linewidth=np.inf):
        print(t_dft)
    # Compute iDFT.
    f_dft = inv_transform(t_dft, "fourier")
    print("iDFT:")
    with np.printoptions(precision=2, suppress=True, linewidth=np.inf):
        print(np.real(f_dft))
    # Compute DHT.
    t_dht = transform(f, "hartley")
    print("\nDHT:")
    with np.printoptions(precision=2, suppress=True, linewidth=np.inf):
        print(t_dht)
    # Compute iDHT.
    f_dht = inv_transform(t_dht, "hartley")
    print("iDHT:")
    with np.printoptions(precision=2, suppress=True, linewidth=np.inf):
        print(f_dht)
    # Compute DCT.
    t_dct = transform(f, "cosine")
    print("\nDCT:")
    with np.printoptions(precision=2, suppress=True, linewidth=np.inf):
        print(t_dct)
    # Compute iDCT.
    f_dct = inv_transform(t_dct, "cosine")
    print("iDCT:")
    with np.printoptions(precision=2, suppress=True, linewidth=np.inf):
        print(f_dct)
    # Compute DST.
    t_dst = transform(f, "sine")
    print("\nDST:")
    with np.printoptions(precision=2, suppress=True, linewidth=np.inf):
        print(t_dst)
    # Compute iDST.
    f_dst = inv_transform(t_dst, "sine")
    print("iDST:")
    with np.printoptions(precision=2, suppress=True, linewidth=np.inf):
        print(f_dst)
    # Compute DWHT.
    t_dwht = transform(f, "hadamard")
    print("\nDWHT:")
    with np.printoptions(precision=2, suppress=True, linewidth=np.inf):
        print(t_dwht)
    # Compute iDWHT.
    f_dwht = inv_transform(t_dwht, "hadamard")
    print("iDWHT:")
    with np.printoptions(precision=2, suppress=True, linewidth=np.inf):
        print(f_dwht)
    # Compute Haar transform.
    t_haar = transform(f, "haar")
    print("\nHaar Transform:")
    with np.printoptions(precision=2, suppress=True, linewidth=np.inf):
        print(t_haar)
    # Compute iHaar Transform.
    f_haar = inv_transform(t_haar, "haar")
    print("iHaar Transform:")
    with np.printoptions(precision=2, suppress=True, linewidth=np.inf):
        print(f_haar)

    # **********************************************************************************
    # Section: Filtering ***************************************************************
    # **********************************************************************************

    print("\nFiltering ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    # Read image to filter.
    img = imageio.imread("characterTestPattern688.tif")

    # Filter image using DFT and LP filter.
    start = time.time()
    lp_dft = ideal_filter(f=img, xform="fourier", type="low", r=60)
    stop = time.time()
    print("LP DFT filtering: %.2f seconds." % (stop - start))

    # Filter image using DHT and LP filter.
    start = time.time()
    lp_dht = ideal_filter(f=img, xform="hartley", type="low", r=60)
    stop = time.time()
    print("LP DHT filtering: %.2f seconds." % (stop - start))
    
    # Filter image using DCT and LP filter.
    start = time.time()
    lp_dct = ideal_filter(f=img, xform="cosine", type="low", r=120)
    stop = time.time()
    print("LP DCT filtering: %.2f seconds." % (stop - start))

    # Filter image using DST and LP filter.
    start = time.time()
    lp_dst = ideal_filter(f=img, xform="sine", type="low", r=120)
    stop = time.time()
    print("LP DST filtering: %.2f seconds." % (stop - start))

    # Filter image using DFT and HP filter.
    start = time.time()
    hp_dft = ideal_filter(f=img, xform="fourier", type="high", r=60)
    stop = time.time()
    print("HP DFT filtering: %.2f seconds." % (stop - start))

    # Filter image using DHT and HP filter.
    start = time.time()
    hp_dht = ideal_filter(f=img, xform="hartley", type="high", r=60)
    stop = time.time()
    print("HP DHT filtering: %.2f seconds." % (stop - start))
    
    # Filter image using DCT and HP filter.
    start = time.time()
    hp_dct = ideal_filter(f=img, xform="cosine", type="high", r=120)
    stop = time.time()
    print("HP DCT filtering: %.2f seconds." % (stop - start))

    # Filter image using DST and HP filter.
    start = time.time()
    hp_dst = ideal_filter(f=img, xform="sine", type="high", r=120)
    stop = time.time()
    print("HP DST filtering: %.2f seconds." % (stop - start))

    # Create images to display.
    fig1 = plt.figure(num=1, figsize=(14, 10))
    fig1.add_subplot(3, 2, 1)
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.title("Original image")
    fig1.add_subplot(3, 2, 2)
    plt.imshow(lp_dft.astype(int), cmap="gray", vmin=0, vmax=255)
    plt.axis("off")
    plt.title("LP DFT")
    fig1.add_subplot(3, 2, 3)
    plt.imshow(lp_dht.astype(int), cmap="gray", vmin=0, vmax=255)
    plt.axis("off")
    plt.title("LP DHT")
    fig1.add_subplot(3, 2, 4)
    plt.imshow(lp_dct.astype(int), cmap="gray", vmin=0, vmax=255)
    plt.axis("off")
    plt.title("LP DCT")
    fig1.add_subplot(3, 2, 5)
    plt.imshow(lp_dst.astype(int), cmap="gray", vmin=0, vmax=255)
    plt.axis("off")
    plt.title("LP DST")

    fig2 = plt.figure(num=2, figsize=(14, 10))
    fig2.add_subplot(3, 2, 1)
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.title("Original image")
    fig2.add_subplot(3, 2, 2)
    plt.imshow(hp_dft.astype(int), cmap="gray", vmin=0, vmax=255)
    plt.axis("off")
    plt.title("HP DFT")
    fig2.add_subplot(3, 2, 3)
    plt.imshow(hp_dht.astype(int), cmap="gray", vmin=0, vmax=255)
    plt.axis("off")
    plt.title("HP DHT")
    fig2.add_subplot(3, 2, 4)
    plt.imshow(hp_dct.astype(int), cmap="gray", vmin=0, vmax=255)
    plt.axis("off")
    plt.title("HP DCT")
    fig2.add_subplot(3, 2, 5)
    plt.imshow(hp_dst.astype(int), cmap="gray", vmin=0, vmax=255)
    plt.axis("off")
    plt.title("HP DST")

    # **********************************************************************************
    # Section: Basis images ************************************************************
    # **********************************************************************************

    print("\nBasis images ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    # Size of basis images array.
    n = 4
    # Uncomment next line to have a greater detail of basis images.
    # n = 8
    # Compute basis images of DFT.
    start = time.time()
    dft_basis_real, dft_basis_imag = basis_image("fourier", n)
    stop = time.time()
    print("DFT Basis images: %.2f seconds." % (stop - start))
    # Compute basis images for DHT.
    start = time.time()
    dht_basis = basis_image("hartley", n)
    stop = time.time()
    print("DHT Basis images: %.2f seconds." % (stop - start))
    # Compute basis images for DCT.
    start = time.time()
    dct_basis = basis_image("cosine", n)
    stop = time.time()
    print("DCT Basis images: %.2f seconds." % (stop - start))
    # Compute basis images for DST.
    start = time.time()
    dst_basis = basis_image("sine", n)
    stop = time.time()
    print("DST Basis images: %.2f seconds." % (stop - start))
    # Compute basis images for DWHT.
    start = time.time()
    dwht_basis = basis_image("hadamard", n)
    stop = time.time()
    print("DWHT Basis images: %.2f seconds." % (stop - start))
    # Compute basis images for Haar Transform.
    start = time.time()
    haar_basis = basis_image("haar", n)
    stop = time.time()
    print("Haar transform Basis images: %.2f seconds." % (stop - start))
    print("Creating plots for basis images...")

    # Show basis images of DFT.
    nrow = n
    ncol = n
    fig3 = plt.figure(num=3, figsize=(ncol+1, nrow+1))
    plt.suptitle("DFT - Basis images - Real")
    gs = gridspec.GridSpec(nrow, ncol,
            wspace=0.0, hspace=0.0, 
            top=1.-0.5/(nrow+1), bottom=0.5/(nrow+1), 
            left=0.5/(ncol+1), right=1-0.5/(ncol+1)) 
    current = 0
    for i in range(nrow):
        for j in range(ncol):
            basis_im = scale_img(dft_basis_real[current,:,:])
            ax = plt.subplot(gs[i, j])
            ax.imshow(basis_im, cmap="gray", vmin=0, vmax=255)
            ax.set_xticklabels([])  
            ax.set_yticklabels([])  
            current += 1

    fig4 = plt.figure(num=4, figsize=(ncol+1, nrow+1))
    plt.suptitle("DFT - Basis images - Imaginary")
    gs = gridspec.GridSpec(nrow, ncol,
            wspace=0.0, hspace=0.0, 
            top=1.-0.5/(nrow+1), bottom=0.5/(nrow+1), 
            left=0.5/(ncol+1), right=1-0.5/(ncol+1)) 
    current = 0
    for i in range(nrow):
        for j in range(ncol):
            basis_im = scale_img(dft_basis_imag[current,:,:])
            ax = plt.subplot(gs[i, j])
            ax.imshow(basis_im, cmap="gray", vmin=0, vmax=255)
            ax.set_xticklabels([])  
            ax.set_yticklabels([])  
            current += 1

    # Show basis images of DHT.
    fig5 = plt.figure(num=5, figsize=(ncol+1, nrow+1))
    plt.suptitle("DHT - Basis images")
    gs = gridspec.GridSpec(nrow, ncol,
            wspace=0.0, hspace=0.0, 
            top=1.-0.5/(nrow+1), bottom=0.5/(nrow+1), 
            left=0.5/(ncol+1), right=1-0.5/(ncol+1)) 
    current = 0
    for i in range(nrow):
        for j in range(ncol):
            basis_im = scale_img(dht_basis[current,:,:])
            ax = plt.subplot(gs[i, j])
            ax.imshow(basis_im, cmap="gray", vmin=0, vmax=255)
            ax.set_xticklabels([])  
            ax.set_yticklabels([])  
            current += 1

    # Show basis images of DCT.
    fig6 = plt.figure(num=6, figsize=(ncol+1, nrow+1))
    plt.suptitle("DCT - Basis images")
    gs = gridspec.GridSpec(nrow, ncol,
            wspace=0.0, hspace=0.0, 
            top=1.-0.5/(nrow+1), bottom=0.5/(nrow+1), 
            left=0.5/(ncol+1), right=1-0.5/(ncol+1)) 
    current = 0
    for i in range(nrow):
        for j in range(ncol):
            basis_im = scale_img(dct_basis[current,:,:])
            ax = plt.subplot(gs[i, j])
            ax.imshow(basis_im, cmap="gray", vmin=0, vmax=255)
            ax.set_xticklabels([])  
            ax.set_yticklabels([])  
            current += 1

    # Show basis images of DST.
    fig7 = plt.figure(num=7, figsize=(ncol+1, nrow+1))
    plt.suptitle("DST - Basis images")
    gs = gridspec.GridSpec(nrow, ncol,
            wspace=0.0, hspace=0.0, 
            top=1.-0.5/(nrow+1), bottom=0.5/(nrow+1), 
            left=0.5/(ncol+1), right=1-0.5/(ncol+1)) 
    current = 0
    for i in range(nrow):
        for j in range(ncol):
            basis_im = scale_img(dst_basis[current,:,:])
            ax = plt.subplot(gs[i, j])
            ax.imshow(basis_im, cmap="gray", vmin=0, vmax=255)
            ax.set_xticklabels([])  
            ax.set_yticklabels([])  
            current += 1

    # Show basis images of DWHT.
    fig8 = plt.figure(num=8, figsize=(ncol+1, nrow+1))
    plt.suptitle("DWHT - Basis images")
    gs = gridspec.GridSpec(nrow, ncol,
            wspace=0.0, hspace=0.0, 
            top=1.-0.5/(nrow+1), bottom=0.5/(nrow+1), 
            left=0.5/(ncol+1), right=1-0.5/(ncol+1)) 
    current = 0
    for i in range(nrow):
        for j in range(ncol):
            basis_im = scale_img(dwht_basis[current,:,:])
            ax = plt.subplot(gs[i, j])
            ax.imshow(basis_im, cmap="gray", vmin=0, vmax=255)
            ax.set_xticklabels([])  
            ax.set_yticklabels([])  
            current += 1

    # Show basis images of DWHT.
    fig9 = plt.figure(num=9, figsize=(ncol+1, nrow+1))
    plt.suptitle("Haar transform - Basis images")
    gs = gridspec.GridSpec(nrow, ncol,
            wspace=0.0, hspace=0.0, 
            top=1.-0.5/(nrow+1), bottom=0.5/(nrow+1), 
            left=0.5/(ncol+1), right=1-0.5/(ncol+1)) 
    current = 0
    for i in range(nrow):
        for j in range(ncol):
            basis_im = scale_img(haar_basis[current,:,:])
            ax = plt.subplot(gs[i, j])
            ax.imshow(basis_im, cmap="gray", vmin=0, vmax=255)
            ax.set_xticklabels([])  
            ax.set_yticklabels([])  
            current += 1
    print("Complete.")

    # Show figures
    plt.show()
