import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
import scipy.ndimage.filters as filter
import time


img=mpimg.imread('lena.jpg')[:,:,0]
start_time = time.time()
edges = filter.gaussian_filter(img, sigma=3, order=0, output=None, mode='reflect', cval=0.0, truncate=4.0)
end_time = time.time()
total_time = end_time - start_time
print('Time taken seconds')
print(total_time)

plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Gausian Blur'), plt.xticks([]), plt.yticks([])

plt.show()
