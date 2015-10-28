
import numpy as np

import theano
from theano import tensor

from blocks.bricks import MLP, Rectifier, Softmax
from blocks.initialization import Constant, IsotropicGaussian
from blocks.bricks.cost import CategoricalCrossEntropy, BinaryCrossEntropy

from blocks.graph import ComputationGraph
from blocks.filter import VariableFilter
from blocks.roles import INPUT, OUTPUT, WEIGHT, BIAS


def run_experiment():

    np.random.seed(42)

    X = tensor.matrix('features')
    y = tensor.matrix('targets')

    mlp = MLP(  activations=[Rectifier(), Rectifier(), Softmax()],
                dims=[100, 50, 50, 10],
                weights_init=IsotropicGaussian(std=0.1), biases_init=IsotropicGaussian(std=0.01))
    mlp.initialize()
    y_hat = mlp.apply(X)

    # This whole thing is thrown out of whack by the fact that Blocks divises
    # the cost by N internally because it calls .mean() on the output.
    # This is a fine thing to do, but it throws off the computation
    # of the individual gradients because they find themselves divided
    # by that factor of N which has nothing to do with them.
    cost = CategoricalCrossEntropy().apply(y, y_hat) * X.shape[0]
    #cost = CategoricalCrossEntropy().apply(y_hat, y)
    #cost = BinaryCrossEntropy().apply(y.flatten(), y_hat.flatten())

    cg = ComputationGraph([y_hat])
    
    """
    print "--- INPUT ---"
    for v in VariableFilter(bricks=mlp.linear_transformations, roles=[INPUT])(cg.variables):
        print v.tag.annotations[0].name

    print "--- OUTPUT ---"
    #print(VariableFilter(bricks=mlp.linear_transformations, roles=[OUTPUT])(cg.variables))
    for v in VariableFilter(bricks=mlp.linear_transformations, roles=[OUTPUT])(cg.variables):
        print v.tag.annotations[0].name

    print "--- WEIGHT ---"
    #print(VariableFilter(bricks=mlp.linear_transformations, roles=[WEIGHT])(cg.variables))
    for v in VariableFilter(bricks=mlp.linear_transformations, roles=[WEIGHT])(cg.variables):
        print v.tag.annotations[0].name
    print "--- BIAS ---"
    #print(VariableFilter(bricks=mlp.linear_transformations, roles=[BIAS])(cg.variables))
    for v in VariableFilter(bricks=mlp.linear_transformations, roles=[BIAS])(cg.variables):
        print v.tag.annotations[0].name
    """

    # check out .tag on the variables to see which layer they belong to

    print "----------------------------"

    from expression_builder import SumGradSquareNormAndVariance

    sgsnav = SumGradSquareNormAndVariance.make_from_blocks(mlp, cg, cost)
    sgsnav_grad = sgsnav.get_sum_gradient_square_norm()
    sgsnav_var = sgsnav.get_sum_gradient_variance()


    #L_grads = [tensor.grad(cost, p) for p in cg.parameters]
    L_grads = [tensor.grad(cost, v) for v in VariableFilter(roles=[WEIGHT, BIAS])(cg.variables)]

    # works on the sum of the gradients in a mini-batch
    sum_square_norm_gradients = sum([tensor.sqr(g).sum() for g in L_grads])


    N = 10

    # Option 1.
    Xtrain = np.random.randn(N, 100).astype(np.float32)
    ytrain = np.zeros((N, 10), dtype=np.float32)
    for n in range(N):
        label = np.random.randint(low=0, high=10)
        ytrain[n, label] = 1.0

    # Option 2.
    #Xtrain = np.ones((N, 100)).astype(np.float32)
    #ytrain = np.ones((N, 10), dtype=np.float32)




    def grad_covariance(Xtrain, ytrain):
        N = Xtrain.shape[0]
        assert N == ytrain.shape[0]

        # add the BIAS here   roles=[WEIGHT, BIAS]   when you want to include it again
        fc = theano.function([X, y], [tensor.grad(cost, v) for v in VariableFilter(roles=[WEIGHT, BIAS])(cg.variables)])

        L_minibatch_grads = fc(Xtrain, ytrain)
        LL_single_grads = []
        for n in range(N):
            LL_single_grads.append(fc(Xtrain[n,:].reshape((1,100)), ytrain[n,:].reshape((1,10))))

        result = np.zeros((N,))

        for (n, L_single_grads) in enumerate(LL_single_grads):

            #print "n : %d" % n
            #print L_single_grads

            for (minibatch_grad, single_grad) in zip(L_minibatch_grads, L_single_grads):
                #print single_grad.shape
                #print minibatch_grad.shape
                B = (single_grad - minibatch_grad)**2
                #print B.shape
                #print B.sum()
                result[n] += B.sum()

        return result


    f = theano.function([X,y],
                        [cost,
                            sgsnav_grad,
                            sgsnav_var,
                            sum_square_norm_gradients])

    [c0, measured_sgsnav_grad_norm, measured_sgsnav_var, _] = f(Xtrain, ytrain)


    L_c, L_measured_grad_norm = ([], [])
    for n in range(N):
        [c, _, _, measured_grad_norm] = f(Xtrain[n,:].reshape((1,100)), ytrain[n,:].reshape((1,10)))
        L_c.append(c0)
        L_measured_grad_norm.append(measured_grad_norm)

    print "Cost for whole mini-batch in single shot : %f." % c
    print "Cost for whole mini-batch accumulated    : %f." % sum(L_c)
    print ""
    print "Square-norm of all gradients for each data point in single shot :"
    print measured_sgsnav_grad_norm.reshape((1,-1))
    print "Square-norm of all gradients for each data point iteratively :"
    print np.array(L_measured_grad_norm).reshape((1,-1))
    print ""
    print "Difference max abs : %f." % np.max(np.abs(measured_sgsnav_grad_norm - np.array(L_measured_grad_norm)))
    #print "Difference max abs : %f." % np.max(np.abs(v0 - np.array(L_gs2)))
    print ""
    #print "Ratios (from experimental method): "
    #print np.array(L_gs1).reshape((1,-1)) / v0.reshape((1,-1))
    #print "Ratios (from scan) : "
    #print np.array(L_gs1).reshape((1,-1)) / sc0.reshape((1,-1))

    print ""
    print ""
    print "measured_sgsnav_var"
    print measured_sgsnav_var
    print "grad_covariance(Xtrain, ytrain)"
    print grad_covariance(Xtrain, ytrain)


if __name__ == "__main__":
    run_experiment()
