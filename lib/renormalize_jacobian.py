import theano
import theano.tensor as T
import numpy as np

'''
Given a Jacobian 3-tensor and a vector of the example's gradient norms,
renormalizes each row to have an L2 norm of 1.0.  

'''
def renormalize_jacobian(J, gradient_norm): 

    return J / T.addbroadcast(T.reshape(gradient_norm, (gradient_norm.shape[0], 1,1)))

def compute_gradient_norm(J): 

    gradient_norm = T.sqrt(T.sum(T.sqr(J), axis = (1,2)))

    return gradient_norm

def test_renorm(): 

    j = np.random.normal(size = (20, 100, 100))

    J = T.tensor3()

    grad_norm = compute_gradient_norm(J)

    rJ = renormalize_jacobian(J, grad_norm)

    new_grad_norm = compute_gradient_norm(rJ)

    f = theano.function([J], [new_grad_norm], allow_input_downcast = 'True')

    print f(j)[0]

if __name__ == "__main__": 
    test_renorm()



