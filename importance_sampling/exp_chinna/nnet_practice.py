__author__ = 'chinna'
import theano
from theano import tensor as T
import numpy as np
import matplotlib.pyplot as plt
from theano import shared
from theano import function

import theano
from theano import tensor as T
import numpy as np
from load import mnist
from load import mnist_with_noise
from PIL import Image
#from foxhound.utils.vis import grayscale_grid_vis, unit_scale
from scipy.misc import imsave
import scipy as sp
import numpy as np
from PIL import Image
from scipy import signal

nhidden = 1
ntrain = 50000

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01))

def sgd(cost, params, lr=0.05):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        updates.append([p, p - g * lr])
    return updates

def model(X, w_h):
    h0 = T.nnet.sigmoid(T.dot(X, w_h[0]))
    pyx = T.nnet.softmax(T.dot(h0,w_h[1]))
    return pyx


def main_loop():
    trX, teX, trY, teY = mnist(ntrain=ntrain,ntest=2000,onehot=True)
    print "before", trX[30][0:10]
    seq = mnist_with_noise([trX,trY],10)
    print "after", trX[30][0:10]
    X = T.fmatrix()
    Y = T.fmatrix()
    #grads = T.fvector()

    w_h = [init_weights((784, 625)), init_weights((625, 10))]

    py_x = model(X, w_h)
    y_x = T.argmax(py_x, axis=1)

    cost = T.mean(T.nnet.categorical_crossentropy(py_x, Y))
    params = w_h
    updates = sgd(cost, params)
    grads = T.grad(cost=cost,wrt=params)
    grad_for_norm = T.grad(cost=cost,wrt=params)

    train = theano.function(inputs=[X, Y], outputs=[cost,grads[0],grads[1]], updates=updates, allow_input_downcast=True)
    predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)
    get_grad = theano.function(inputs=[X,Y],outputs=[cost,grad_for_norm[0],grad_for_norm[1]], allow_input_downcast=True)

    mb_size = 128
    for i in range(2):
        grad_list = []
        for start, end in zip(range(0, len(trX), mb_size), range(mb_size, len(trX), mb_size)):
            cost,grads[0],grads[1] = train(trX[start:end], trY[start:end])
        print np.mean(np.argmax(teY, axis=1) == predict(teX))

    noisy_grads = []
    normal_grads = []
    noisy_cost = []
    normal_cost = []
    mb_size = 1
    n_predicts = 0
    for i in seq:
        cost,grad0,grad1 = get_grad(trX[i:i+1], trY[i:i+1])
        norm = np.linalg.norm(grad0) 
        if i < 0.1*ntrain:
            n_predicts += (np.argmax(trY[i:i+1], axis=1)==predict(trX[i:i+1]))
            noisy_grads.append(norm)
            noisy_cost.append(cost)
        else:
            normal_grads.append(norm)
            normal_cost.append(cost)


    print "noisy grad : mean,var - " ,np.mean(noisy_grads),np.var(noisy_grads)
    print "normal grad: mean,var - " ,np.mean(normal_grads),np.var(normal_grads)

    print "noisy cost : mean,var - " ,np.mean(noisy_cost),np.var(noisy_cost)
    print "normal cost: mean,var - " ,np.mean(normal_cost),np.var(normal_cost)

    print " noisy predicts out of 5000 -", n_predicts
    plt.plot(noisy_grads)
    plt.plot(normal_grads)

    plt.savefig('grad0.jpeg')

if __name__ == "__main__":
    main_loop()


