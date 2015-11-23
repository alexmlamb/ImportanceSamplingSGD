__author__ = 'chinna'

import matplotlib.pyplot as plt
import theano
from theano import tensor as T
from load import mnist
from load import mnist_with_noise
import numpy as np


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

def get_pre_softmax_func(X,w_h):
    h0 = T.nnet.sigmoid(T.dot(X, w_h[0]))
    pyx = T.dot(h0,w_h[1])
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
    pre_softmax = get_pre_softmax_func(X, w_h)
    cost = T.mean(T.nnet.categorical_crossentropy(py_x, Y))
    params = w_h
    updates = sgd(cost, params)
    grads = T.grad(cost=cost,wrt=params)
    grad_for_norm = T.grad(cost=cost,wrt=params)

    train = theano.function(inputs=[X, Y], outputs=[cost,grads[0],grads[1]], updates=updates, allow_input_downcast=True)
    predict = theano.function(inputs=[X], outputs=[y_x,py_x], allow_input_downcast=True)
    get_grad = theano.function(inputs=[X,Y],outputs=[cost,grad_for_norm[0],grad_for_norm[1],pre_softmax], allow_input_downcast=True)
    #get_pre_softmax = theano.function([X],)
    mb_size = 128
    for i in range(1):
        grad_list = []
        for start, end in zip(range(0, len(trX), mb_size), range(mb_size, len(trX), mb_size)):
            cost,grads0,grads1 = train(trX[start:end], trY[start:end])
        y,p_y =  predict(teX)
        print np.mean(np.argmax(teY, axis=1) == y)

    noisy_grads = []
    normal_grads = []
    noisy_cost = []
    normal_cost = []
    mb_size = 1
    n_predicts = 0
    noisy_pre_softmax_norm = []
    normal_pre_softmax_norm = []
    for i in seq:
        cost,grad0,grad1,pre_soft = get_grad(trX[i:i+1], trY[i:i+1])
        norm = np.linalg.norm(grad0)
        y,py = predict(trX[i:i+1])
        if i < 0.1*ntrain:
            n_predicts += (np.argmax(trY[i:i+1], axis=1)==y)
            noisy_grads.append(norm)
            noisy_cost.append(cost)
            noisy_pre_softmax_norm.append(np.linalg.norm(pre_soft))


        else:
            normal_grads.append(norm)
            normal_cost.append(cost)
            normal_pre_softmax_norm.append(np.linalg.norm(pre_soft))



    print "noisy grad : mean,var - " ,np.mean(noisy_grads),np.var(noisy_grads)
    print "normal grad: mean,var - " ,np.mean(normal_grads),np.var(normal_grads)

    print "noisy cost : mean,var - " ,np.mean(noisy_cost),np.var(noisy_cost)
    print "normal cost: mean,var - " ,np.mean(normal_cost),np.var(normal_cost)

    print "noisy pre_softmax norm  : mean,var - " ,np.mean(noisy_pre_softmax_norm),np.var(noisy_pre_softmax_norm)
    print "normal pre softmax norm : mean,var - " ,np.mean(normal_pre_softmax_norm),np.var(normal_pre_softmax_norm)
    print " noisy predicts out of 5000 -", n_predicts
    plt.plot(noisy_grads)
    plt.plot(normal_grads)

    plt.savefig('grad0.jpeg')



if __name__ == "__main__":
    main_loop()



