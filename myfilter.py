import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
import scipy.ndimage.filters as filter
import time

#image filters
kernel_edge_detect = np.array([[0.,1.,0.],
                               [1.,-4.,1.],
                               [0.,1.,0.]])

kernel_sharpen = np.array([[0.,-1.,0.],
                           [-1.,5.,-1.],
                           [0.,-1.,0.]])

kernel_blur = np.array([[1.,1.,1.],
                        [1.,1.,1.],
                        [1.,1.,1.]])

kernel_list = [kernel_edge_detect,kernel_sharpen,kernel_blur]
title_list = ['edge-detect','sharpen','blur']


#take image as input with only one channel R
image=mpimg.imread('lena.jpg')[:,:,0]

#img = mpimg.imread('lena.jpg')
#lum_img = img[:,:,2]
#plt.title('test123')
#plt.imshow(lum_img)

# shape of the image here means w,h
shape = image.shape
#Create a matrix of size shape float data type
newimage = np.ndarray(shape,dtype=np.float)
myimage = np.ndarray(shape,dtype=np.float)

#creating a matrix of all zeros of size with m+2xn+2
supershape = (shape[0] + 2,shape[1] + 2)
supermatrix = np.zeros(supershape,dtype=np.float)
# array splicing i:j:k i= starting pt, j=end point, k=step
supermatrix[1:-1,1:-1] = image

# lists to store the images after applying convolution
imagelist_std_convolve = []
imagelist_my_convolve = []

#create a plotting of images
fig = plt.figure(figsize=(14, 6.5), dpi=100)

fig.add_subplot(4,4,1)
plt.title('Original')
plt.imshow(image,cmap=plt.cm.gray)
plt.axis('off')

#aplying normalization
def normalize(matrix):
    sum = np.sum(matrix)
    if sum > 0.:
        return matrix / sum
    else:
        return matrix

#applying convolutuion
def neighbors(r,c,supermatrix):
    m = supermatrix[r:r+3,c:c+3]
    return m

def convolve(n,kernel):
    sum = 0
    for (rr,cc),value in np.ndenumerate(n):
        sum += n[rr,cc] * kernel[rr,cc]
    
    return sum % 255

def my_convolve(matrix,super,kernel,shape):
    result = np.ndarray(shape,dtype=np.float)
    
    for (r,c),value in np.ndenumerate(matrix):
        n = neighbors(r,c,super)
        result[r,c] = convolve(n,kernel)
    
    return result

#apllying the inbuilt convolve using the filters we defined
for i in range(len(kernel_list)):
    kernel_list[i] = normalize(kernel_list[i])
    newimage = filter.convolve(image,kernel_list[i],mode='constant',cval=0)
    imagelist_std_convolve.append(newimage)
    fig.add_subplot(4,4,i+2)
    plt.title(title_list[i])
    plt.imshow(newimage,cmap=plt.cm.gray)
    plt.axis('off')

#applying our own convolve using the same filters
for i in range(len(kernel_list)):
    flipped_kernel = kernel_list[i].copy()
    flipped_kernel = np.fliplr(flipped_kernel)
    flipped_kernel = np.flipud(flipped_kernel)
    flipped_kernel = normalize(flipped_kernel)
    start_time = time.time()
    myimage = my_convolve(image,supermatrix,flipped_kernel,shape)
    end_time = time.time()
    total_time = end_time - start_time
    print('Time taken seconds i:',title_list[i])
    print(total_time)
    imagelist_my_convolve.append(myimage)
    fig.add_subplot(4,4,i+2+3)
    plt.title('my ' + title_list[i])
    plt.imshow(myimage,cmap=plt.cm.gray)
    plt.axis('off')
    plt.tight_layout()

#final plot of all outcomes
plt.show()
