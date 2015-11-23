import numpy as np
import scipy as sp
from scipy import signal
import os
from PIL import Image

datasets_dir = 'media/datasets/'

def one_hot(x,n):
	if type(x) == list:
		x = np.array(x)
	x = x.flatten()
	o_h = np.zeros((len(x),n))
	o_h[np.arange(len(x)),x] = 1
	return o_h

def mnist(ntrain=60000,ntest=10000,onehot=True):
	data_dir = os.path.join(datasets_dir,'mnist/')
	fd = open(os.path.join(data_dir,'train-images-idx3-ubyte'))
	loaded = np.fromfile(file=fd,dtype=np.uint8)
	trX = loaded[16:].reshape((60000,28*28)).astype(float)

	fd = open(os.path.join(data_dir,'train-labels-idx1-ubyte'))
	loaded = np.fromfile(file=fd,dtype=np.uint8)
	trY = loaded[8:].reshape((60000))

	fd = open(os.path.join(data_dir,'t10k-images-idx3-ubyte'))
	loaded = np.fromfile(file=fd,dtype=np.uint8)
	teX = loaded[16:].reshape((10000,28*28)).astype(float)

	fd = open(os.path.join(data_dir,'t10k-labels-idx1-ubyte'))
	loaded = np.fromfile(file=fd,dtype=np.uint8)
	teY = loaded[8:].reshape((10000))

	trX = (trX)/255.
	teX = (teX)/255.

	trX = trX[:ntrain]
	trY = trY[:ntrain]

	teX = teX[:ntest]
	teY = teY[:ntest]

	if onehot:
		trY = one_hot(trY, 10)
		teY = one_hot(teY, 10)
	else:
		trY = np.asarray(trY)
		teY = np.asarray(teY)

	return trX,teX,trY,teY

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
    image = image.reshape(28,28)
    kernel = get_kernel(shape=(11,11),sigma=10)
    image   = signal.convolve2d(image, kernel, boundary='fill', fillvalue=0,mode='same')
    image   = signal.convolve2d(image, kernel, boundary='fill', fillvalue=0,mode='same')
    return image.reshape((784,))

def add_guassian_noise(input_image):
    for i in range(len(input_image)):
        input_image[i] = add_gnoise_util(input_image[i])
    return input_image


def convert_to_image(image, name):
    a = (image*255).reshape((28,28)).astype('uint8')
    im = Image.fromarray(a)
    im.save(name)

# example samples_list = [ trX, trY]
def mnist_with_noise(samples, percent):
    images  = samples[0]
    classes = samples[1]
    num_noisy_samples = (percent/100.0)*len(images)
    seq = np.arange(len(images))
    #convert_to_image(images[4999],"before.jpg")
    #noise = np.random.randn(*(images[:num_noisy_samples].shape))
    #noise = np.random.uniform(0,1,size=(images[:num_noisy_samples].shape))
    noise = np.ones_like(images[:num_noisy_samples])
    images[:num_noisy_samples] = noise
    #images[:num_noisy_samples] = add_guassian_noise(images[:num_noisy_samples])
    #convert_to_image(images[4999],"after.jpg")
    images[images>1]=1.0
    images[images<0]=0.0
    bundle = zip(images,classes,seq)
    np.random.shuffle(bundle)
    nimages,nclasses,seq = zip(*bundle)
    return seq
