
from collections import defaultdict
import numpy as np

import theano
from theano import tensor

from blocks.bricks import MLP, Rectifier, Softmax
from blocks.initialization import Constant, IsotropicGaussian
from blocks.bricks.cost import CategoricalCrossEntropy, BinaryCrossEntropy

from blocks.graph import ComputationGraph
from blocks.filter import VariableFilter
from blocks.roles import INPUT, OUTPUT, WEIGHT, BIAS



# Note : You'll probably have to work out through shapes and transpositions.

def get_square_norm_gradients(D_by_layer, cost, accum = 0):

    # This returns a theano variable that will be of shape (minibatch_size, ).
    # It will contain, for each training example, the associated square-norm of the total gradient.
    # If you take the element-wise square-root afterwards, you will get
    # the associated 2-norms, which is what you want for importance sampling.

    gradient_component_count = 0

    for (layer_name, D) in D_by_layer.items():

        input_square_norms = tensor.sqr(D['input']).sum(axis=1)
        backprop_output = tensor.grad(cost, D['output'])
        backprop_output_square_norms = tensor.sqr(backprop_output).sum(axis=1)

        if D.has_key('weight') and D.has_key('bias'):
            accum = accum + (input_square_norms + 1) * backprop_output_square_norms
            gradient_component_count += 2
        elif D.has_key('weight'):
            accum = accum + input_square_norms * backprop_output_square_norms
            gradient_component_count += 1
        elif D.has_key('bias'):
            print "This is strange. Only a bias and no weights."
            accum = accum + backprop_output_square_norms
            gradient_component_count += 1            
        else:
            print "No contribution at all to get_square_norm_gradients for layer %d." % layer_name

    print "There are %d gradient components found in get_square_norm_gradients." % gradient_component_count
    return accum




def get_square_norm_gradients_scan(D_by_layer, cost, accum = 0):

    # This returns a theano variable that will be of shape (minibatch_size, ).
    # It will contain, for each training example, the associated square-norm of the total gradient.
    # If you take the element-wise square-root afterwards, you will get
    # the associated 2-norms, which is what you want for importance sampling.

    for (layer_name, D) in D_by_layer.items():

        backprop_output = tensor.grad(cost, D['output'])

        if D.has_key('weight'):
            A = D['input']
            B = backprop_output
            S, _ =  theano.scan(fn=lambda A, B: tensor.sqr(tensor.outer(A,B)).sum(),
                                        sequences=[A,B])
            accum = accum + S

        if D.has_key('bias'):

            B = backprop_output
            S, _ =  theano.scan(fn=lambda B: tensor.sqr(B).sum(),
                                        sequences=[B])
            accum = accum + S
        
    return accum



def get_mean_square_norm_gradients_variance_method_00(D_by_layer, cost, accum = 0):

    # This returns a theano variable that will be of shape (minibatch_size, ).
    # It will contain, for each training example, the associated mean of the
    # variance wrt the gradient of that minibatch.

    for (layer_name, D) in D_by_layer.items():

        input = D['input']
        input_square_norms = tensor.sqr(D['input']).sum(axis=1)
        backprop_output = tensor.grad(cost, D['output'])
        # I don't think that theano recomputes this.
        # It should be just redundant nodes in the computational graph
        # that end up being computed only once anyways.
        grad_weight = tensor.grad(cost, D['weight'])
        grad_bias = tensor.grad(cost, D['bias'])
        backprop_output_square_norms = tensor.sqr(backprop_output).sum(axis=1)

        if D.has_key('weight'):
            accum = accum + input_square_norms * backprop_output_square_norms
            accum = accum + tensor.sqr(grad_weight).sum() # all the terms get this "middle" expression added to them
            accum = accum - 2 * (backprop_output.dot(grad_weight.T) * input).mean(axis=1)
        if D.has_key('bias'):
            pass
            # TODO : Implement this.

    return accum


def get_mean_square_norm_gradients_variance_method_01(D_by_layer, cost, accum = 0):

    # This returns a theano variable that will be of shape (minibatch_size, ).
    # It will contain, for each training example, the associated mean of the
    # variance wrt the gradient of that minibatch.

    for (layer_name, D) in D_by_layer.items():

        input = D['input']
        input_square_norms = tensor.sqr(D['input']).sum(axis=1)
        backprop_output = tensor.grad(cost, D['output'])
        # I don't think that theano recomputes this.
        # It should be just redundant nodes in the computational graph
        # that end up being computed only once anyways.
        grad_weight = tensor.grad(cost, D['weight'])
        grad_bias = tensor.grad(cost, D['bias'])
        backprop_output_square_norms = tensor.sqr(backprop_output).sum(axis=1)

        if D.has_key('weight'):

            gW = grad_weight
            A = input
            B = backprop_output
            #accum = accum + sum(theano.scan(fn=lambda A, B, W: tensor.sqr(tensor.outer(A,B) - gW).flatten(),
            #                            sequences=[A,B], non_sequences=[gW])) / A.shape[0]
            S, _ =  theano.scan(fn=lambda A, B, W: tensor.sqr(tensor.outer(A,B) - gW).flatten(),
                                        sequences=[A,B], non_sequences=[gW])
            accum = accum + S.mean(axis=1)
            #for e in S:
            #    accum = accum + e


        if D.has_key('bias'):
            pass
            # TODO : Implement this.

    return accum


def get_linear_transformation_roles(mlp, cg):

    D_by_layer = defaultdict(dict)

    for (role, role_str) in [(INPUT, 'input'), (OUTPUT, 'output'), (WEIGHT, 'weight'), (BIAS, 'bias')]:

        for v in VariableFilter(bricks=mlp.linear_transformations, roles=[role])(cg.variables):
            key = v.tag.annotations[0].name
            D_by_layer[key][role_str] = v
            #D_by_layer[key][role_str] = v

    return D_by_layer



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


    D_by_layer = get_linear_transformation_roles(mlp, cg)

    # returns a vector with one entry for each in the mini-batch
    individual_sum_square_norm_gradients_method_experimental = get_square_norm_gradients(D_by_layer, cost)
    individual_sum_square_norm_gradients_method_scan = get_square_norm_gradients_scan(D_by_layer, cost)



    print "There are %d entries in cg.parameters." % len(cg.parameters)
    L_grads_method_01 = [tensor.grad(cost, p) for p in cg.parameters]
    L_grads_method_02 = [tensor.grad(cost, v) for v in VariableFilter(roles=[WEIGHT, BIAS])(cg.variables)]

    # works on the sum of the gradients in a mini-batch
    sum_square_norm_gradients_method_01 = sum([tensor.sqr(g).sum() for g in L_grads_method_01])
    sum_square_norm_gradients_method_02 = sum([tensor.sqr(g).sum() for g in L_grads_method_02])


    individual_mean_square_norm_gradients_variance_method_00 = get_mean_square_norm_gradients_variance_method_00(D_by_layer, cost)
    individual_mean_square_norm_gradients_variance_method_01 = get_mean_square_norm_gradients_variance_method_01(D_by_layer, cost)



    N = 10

    # Option 1.
    Xtrain = np.random.randn(N, 100).astype(np.float32)
    ytrain = np.zeros((N, 10), dtype=np.float32)
    for n in range(N):
        label = np.random.randint(low=0, high=10)
        ytrain[n, label] = 1.0

    # Option 2.
    Xtrain = np.ones((N, 100)).astype(np.float32)
    ytrain = np.ones((N, 10), dtype=np.float32)




    def grad_covariance(Xtrain, ytrain):
        N = Xtrain.shape[0]
        assert N == ytrain.shape[0]

        # add the BIAS here   roles=[WEIGHT, BIAS]   when you want to include it again
        fc = theano.function([X, y], [tensor.grad(cost, v) for v in VariableFilter(roles=[WEIGHT])(cg.variables)])

        L_minibatch_grads = fc(Xtrain, ytrain)
        LL_single_grads = []
        for n in range(N):
            LL_single_grads.append(fc(np.tile(Xtrain[n,:].reshape((1,100)), (N,1)), np.tile(ytrain[n,:].reshape((1,10)), (N,1))))

        individual_mean_square_norm_gradients_variance_method_02 = np.zeros((N,))

        for (n, L_single_grads) in enumerate(LL_single_grads):

            for (minibatch_grad, single_grad) in zip(L_minibatch_grads, L_single_grads):
                #print single_grad.shape
                #print minibatch_grad.shape
                B = (single_grad - minibatch_grad)**2
                #print B.shape
                individual_mean_square_norm_gradients_variance_method_02[n] += B.mean()

        return individual_mean_square_norm_gradients_variance_method_02


    f = theano.function([X,y],
                        [cost,
                            individual_sum_square_norm_gradients_method_experimental,
                            individual_sum_square_norm_gradients_method_scan,
                            individual_mean_square_norm_gradients_variance_method_00,
                            individual_mean_square_norm_gradients_variance_method_01,
                            sum_square_norm_gradients_method_01,
                            sum_square_norm_gradients_method_02])

    [c, v0, sc0, var0, var1, gs1, gs2] = f(Xtrain, ytrain)

    #print "[c, v0, gs1, gs2]"
    L_c, L_v0, L_gs1, L_gs2 = ([], [], [], [])
    for n in range(N):
        #replicas = 2
        #[nc, nv0, _, _, _, ngs1, ngs2] = f(np.tile(Xtrain[n,:].reshape((1,100)), (replicas,1)), np.tile(ytrain[n,:].reshape((1,10)), (replicas,1)))
        [nc, nv0, _, _, _, ngs1, ngs2] = f(Xtrain[n,:].reshape((1,100)), ytrain[n,:].reshape((1,10)))
        
        L_c.append(nc)
        L_v0.append(nv0)
        L_gs1.append(ngs1)
        L_gs2.append(ngs2)

    print "Cost for whole mini-batch in single shot : %f." % c
    print "Cost for whole mini-batch accumulated    : %f." % sum(L_c)
    print ""
    print "Square-norm of all gradients for each data point in single shot :"
    print v0.reshape((1,-1))
    print "Square-norm of all gradients for each data point in single shot using scan :"
    print sc0.reshape((1,-1))
    print "Square-norm of all gradients for each data point iteratively :"
    print np.array(L_gs1).reshape((1,-1))
    #print "Square-norm of all gradients for each data point iteratively :"
    #print np.array(L_gs2).reshape((1,-1))
    print ""
    print "Difference max abs : %f." % np.max(np.abs(v0 - np.array(L_gs1)))
    #print "Difference max abs : %f." % np.max(np.abs(v0 - np.array(L_gs2)))
    print ""
    print "Ratios (from experimental method): "
    print np.array(L_gs1).reshape((1,-1)) / v0.reshape((1,-1))
    print "Ratios (from scan) : "
    print np.array(L_gs1).reshape((1,-1)) / sc0.reshape((1,-1))

    #print ""
    #print ""
    #print "var0"
    #print var0
    #print "var1"
    #print var1
    #print "grad_covariance(Xtrain, ytrain)"
    #print grad_covariance(Xtrain, ytrain)


if __name__ == "__main__":
    run_experiment()
