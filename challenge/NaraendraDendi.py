import numpy as np
import glob
## 1. It's okay to import whatever you want from the local util module if you would like:
#  You can import functions from util folder like this

from util.filters import roberts_cross

def convert_to_grayscale(im):

    return np.mean(im, axis = 2)

def filter_2d(im, kernel):
    M, N = kernel.shape
    H, W = im.shape
    filtered_image = np.zeros((H-M+1, W-N+1), dtype = 'float64')
    
    for i in range(filtered_image.shape[0]):
        for j in range(filtered_image.shape[1]):
            image_patch = im[i:i+M, j:j+N]
            filtered_image[i, j] = np.sum(np.multiply(image_patch, kernel))
            
    return filtered_image
def findMean(G):
#     return mean(G)
    return np.percentile(G,96)

def sobel(gray):
    Kx = np.array([[1, 0, -1],
                   [2, 0, -2],
                   [1, 0, -1]])
    Ky = np.array([[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]])
    Gx = filter_2d(gray, Kx)
    Gy = filter_2d(gray, Ky)
    return [Gx, Gy]
def classify(im):
    '''
    Example submission for coding challenge. 
    
    Args: im (nxmx3) unsigned 8-bit color image 
    Returns: One of three strings: 'brick', 'ball', or 'cylinder'
    
    '''
    gray = convert_to_grayscale(im/255.)
    G = sobel(gray)
    magnitude = np.sqrt(G[0]**2+G[1]**2)
    direction = np.arctan2(G[0], G[1])
    thresh = findMean(magnitude)
    edges_and_angles = np.zeros(magnitude.shape)*np.NaN
    edges_and_angles[magnitude>thresh] = direction[magnitude>thresh]
    edges_and_angles = edges_and_angles[~np.isnan(edges_and_angles)]
    counts, bin_edges = np.histogram(edges_and_angles, bins=60)
    counts.astype(int)

    labels = ['brick', 'ball', 'cylinder']
    a=np.average(counts)
    maximum=max(counts)
    delta =max(counts)*thresh/10
    val = 0
    if a<(maximum*34/100)+delta: val =0
    elif a+delta>=maximum*46/100: val =1
    else: val =2

    return labels[val]
