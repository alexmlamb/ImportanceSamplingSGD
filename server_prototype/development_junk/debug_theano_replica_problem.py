
import theano
from theano import tensor
import numpy as np

def run():

    
    X = tensor.matrix('features')
    y = tensor.matrix('targets')
    #theta = tensor.scalar('theta')
    theta = theano.shared(name='theta', value=1.0)


    cost = theta*(X.sum() + y.sum())

    N = 10
    Xtrain = np.ones((N, 100), dtype=np.float32)
    ytrain = np.ones((N, 10), dtype=np.float32)

    f = theano.function([X,y],
                        [cost, tensor.grad(cost, theta)])

    (c, grad_theta) = f(Xtrain, ytrain)

    replicas = 1
    L_nc = []
    L_ngrad_theta = []
    for n in range(N):
        (nc, ngrad_theta) = f(np.tile(Xtrain[n,:].reshape((1,100)), (replicas,1)), np.tile(ytrain[n,:].reshape((1,10)), (replicas,1)))
        #(nc, grad_theta) = (0.0, 0.0)
        L_nc.append(nc)
        L_ngrad_theta.append(ngrad_theta)

    print "Original cost : %f." % c
    #print "Iterating through minibatch : %s" % str(L_nc)
    print "Sum iterating through minibatchwith sum %f." % sum(L_nc)

    print "Original grad : %f." % grad_theta
    #print "Iterating through minibatch : %s" % str(L_nc)
    print "Sum iterating through minibatchwith sum %f." % sum(L_ngrad_theta)



if __name__ == "__main__":
    run()