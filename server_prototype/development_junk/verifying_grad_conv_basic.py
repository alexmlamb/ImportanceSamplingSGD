

from collections import defaultdict
import numpy as np

import theano
from theano import tensor
import theano.tensor.nnet

from blocks.bricks import MLP, Rectifier, Softmax
from blocks.bricks.conv import (ConvolutionalLayer, ConvolutionalSequence,
                                ConvolutionalActivation, Flattener)
from blocks.initialization import Constant, IsotropicGaussian, Uniform
from blocks.bricks.cost import CategoricalCrossEntropy, BinaryCrossEntropy

from blocks.graph import ComputationGraph
from blocks.filter import VariableFilter
from blocks.roles import INPUT, OUTPUT, WEIGHT, BIAS, FILTER, PARAMETER



def get_sum_square_norm_gradients_conv_transformations(D_by_layer, cost, accum = 0):

    gradient_component_count = 0

    for (layer_name, D) in D_by_layer.items():

        sumc_A_square = tensor.sqr(D['input'].sum(axis=1, keepdims=True))
        backprop_output = tensor.grad(cost, D['output'])
        sumf_Bo_square = tensor.sqr(backprop_output.sum(axis=1, keepdims=True))

        square_norm_grad_wrt_filters = theano.tensor.nnet.conv.conv2d(sumc_A_square, sumf_Bo_square).flatten(2).sum(axis=1)

        assert D.has_key('weight')

        gradient_component_count += 1
        accum += square_norm_grad_wrt_filters

    print "There are %d gradient components found in get_sum_square_norm_gradients." % gradient_component_count
    return accum


def get_conv_layers_transformation_roles(cg):

    # expects a cg that encapsulates only the convolutions

    D_by_layer = defaultdict(dict)

    for (role, role_str) in [(INPUT, 'input'), (OUTPUT, 'output'), (FILTER, 'weight'), (BIAS, 'bias')]:
        for v in VariableFilter(roles=[role])(cg.variables):
            key = v.tag.annotations[0].__hash__()
            D_by_layer[key][role_str] = v

    # Return only the components that have each of 'input', 'output', 'filter' and 'bias'.
    filtered_D_by_layer = dict((k,v) for (k,v) in D_by_layer.items() if len(v.keys()) == 4)

    return filtered_D_by_layer




def run_experiment():

    np.random.seed(42)

    X = tensor.tensor4('features')
    nbr_channels = 3
    image_shape = (12, 12)

    conv_layers = [ ConvolutionalLayer( filter_size=(2,2),
                                        num_filters=10,
                                        activation=Rectifier().apply,
                                        border_mode='valid',
                                        pooling_size=(1,1),
                                        weights_init=Uniform(width=0.1),
                                        biases_init=Uniform(width=0.01),
                                        name='conv0')]
    conv_sequence = ConvolutionalSequence(  conv_layers,
                                            num_channels=nbr_channels,
                                            image_size=image_shape)
    #conv_sequence.push_allocation_config()
    conv_sequence.initialize()
    
    flattener = Flattener()
    conv_output = conv_sequence.apply(X)
    y_hat = flattener.apply(conv_output)
    # Whatever. Not important since we're not going to actually train anything.
    cost = tensor.sqr(y_hat).sum()



    L_grads_method_02 = [tensor.grad(cost, v) for v in VariableFilter(roles=[FILTER])(ComputationGraph([y_hat]).variables)]
    # works on the sum of the gradients in a mini-batch
    sum_square_norm_gradients_method_02 = sum([tensor.sqr(g).sum() for g in L_grads_method_02])




    D_by_layer = get_conv_layers_transformation_roles(ComputationGraph(conv_output))
    individual_sum_square_norm_gradients_method_00 = get_sum_square_norm_gradients_conv_transformations(D_by_layer, cost)



    N = 8
    Xtrain = np.random.randn(N, nbr_channels, image_shape[0], image_shape[1]).astype(np.float32)


    f = theano.function([X],
                        [cost,
                            individual_sum_square_norm_gradients_method_00,
                            sum_square_norm_gradients_method_02])


    [c, v0, gs2] = f(Xtrain)

    #print "[c, v0, gs2]"
    L_c, L_v0, L_gs2 = ([], [], [])
    for n in range(N):
        [nc, nv0, ngs2] = f(Xtrain[n,:].reshape((1,Xtrain.shape[1],Xtrain.shape[2], Xtrain.shape[3])))
        L_c.append(nc)
        L_v0.append(nv0)
        L_gs2.append(ngs2)

    print "Cost for whole mini-batch in single shot : %f." % c
    print "Cost for whole mini-batch accumulated    : %f." % sum(L_c)
    print ""
    print "Square-norm of all gradients for each data point in single shot :"
    print v0.reshape((1,-1))
    print "Square-norm of all gradients for each data point iteratively :"
    print np.array(L_gs2).reshape((1,-1))
    print ""
    print "Difference max abs : %f." % np.max(np.abs(v0 - np.array(L_gs2)))
    print ""
    print "Ratios : "
    print np.array(L_gs2).reshape((1,-1)) / v0.reshape((1,-1))

if __name__ == "__main__":
    run_experiment()
