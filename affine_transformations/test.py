# Import modules
from histogram import hist_equal, image_hist, int_xform
from affine import image_rotate, image_scaling, image_shear, image_translate, process_image
import imageio
from matplotlib import pyplot as plt
import numpy as np

import time
""" 
image = imageio.imread("girl.tif")

start = time.time()
scaled_img = image_scaling(image, 2, 2)
end = time.time()
print("Scaling took %f seconds" % (end - start))
start = time.time()
translated_img = image_translate(image, -100, -100, "black")
end = time.time()
print("Translating took %f seconds" % (end - start))
start = time.time()
sheared_img = image_shear(image, 0.5, 0.5)
end = time.time()
print("Shearing took %f seconds" % (end - start))
start = time.time()
rotated_img_full = image_rotate(image, 45, "full")
end = time.time()
print("Rotating full took %f seconds" % (end - start))
start = time.time()
rotated_img_crop = image_rotate(image, 45, "crop")
end = time.time()
print("Rotating crop took %f seconds" % (end - start))


plt.figure(1)

plt.title("Original image")
plt.imshow(image, cmap="gray")

plt.figure(2)
plt.title("2X image")
plt.imshow(scaled_img, cmap="gray")
#print(scaled_img)
imageio.imwrite("scaled_image.tif", scaled_img)


plt.figure(3)
plt.title("Translate image by 100 pixels to the right")
plt.imshow(translated_img, cmap="gray")
#print(scaled_img)
imageio.imwrite("translated_image.tif", translated_img)


plt.figure(4)
plt.title("Shear image by 0.5 in both directions")
plt.imshow(sheared_img, cmap="gray")
#print(scaled_img)
imageio.imwrite("sheared_image.tif", sheared_img)

plt.figure(5)
plt.title("Rotated image by 90 degrees from origin")
plt.imshow(rotated_img_crop, cmap="gray")
#print(scaled_img)
imageio.imwrite("rotated_image_crop.tif", rotated_img_crop)

plt.figure(6)
plt.title("Rotated image by 90 degrees from origin")
plt.imshow(rotated_img_full, cmap="gray")
plt.show()
#print(scaled_img)
imageio.imwrite("rotated_image_full.tif", rotated_img_full)
"""
"""
image = imageio.imread("spillway-dark.tif")
equalization = hist_equal(image)
 
plt.figure(1)
plt.imshow(image, cmap="gray")
plt.figure(2)
plt.imshow(equalization, cmap="gray")
plt.show()

"""
image = imageio.imread("spillway-dark.tif")
image = image / 255.0
negative = int_xform(image, "negative")
log = int_xform(image, "log")
gamma = int_xform(image, "gamma", 5)
plt.figure(1)
plt.imshow(image, cmap="gray")
plt.figure(2)
plt.imshow(negative, cmap="gray")
plt.figure(3)
plt.imshow(log, cmap="gray")
plt.figure(4)
plt.imshow(gamma, cmap="gray")
plt.show()