__author__ = 'chinna'
import theano
from theano import tensor as T
import numpy as np
import matplotlib.pyplot as plt
from theano import shared
from theano import function
import scipy as sp
from scipy import signal
from PIL import Image

def convert_to_image(image, name):
    im = Image.fromarray(image,mode='RGB')
    im.save(name)
    print "image saved to:",name

def get_kernel(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def add_gnoise_util(image):
    kernel = get_kernel(shape=(5,5),sigma=10)
    for i in range(3):
        image[:,:,i]   = signal.convolve2d(image[:,:,i], kernel, boundary='fill', fillvalue=0,mode='same')
        image[:,:,i]   = signal.convolve2d(image[:,:,i], kernel, boundary='fill', fillvalue=0,mode='same')
    return image

def add_gnoise(images):
    for i in range(images.shape[0]):
        images[i] = add_gnoise_util(images[i])
    return images

# inplace noise addition
def noisify(input_images, config):
    images=np.swapaxes(input_images,axis1=2,axis2=3)
    images=np.swapaxes(images,axis1=1,axis2=2)
    init_shape = images.shape[0]
    print "# of images to add noise :",init_shape
    convert_to_image(images[0],config['image']+'before.jpeg')

    print "adding noise type:", config['noise']

    if config['noise'] == 'guassian':
        images  = add_gnoise(images)
    elif config['noise'] == 'normal':
        noise = 255*np.random.randn(*(images.shape))
        images += noise
    elif config['noise'] == 'uniform':
        noise = 255*np.random.uniform(0,1,size=(images.shape))
        images += noise
    elif config['noise'] == 'black_out':
        images = 0*np.ones_like(images)
    else :
        print "Not added any noise"
    convert_to_image(images[0],config['image']+'after_'+config['noise']+'.jpeg')
    images=np.swapaxes(images,axis1=1,axis2=2)
    images=np.swapaxes(images,axis1=2,axis2=3)
    print images.shape
    return images



