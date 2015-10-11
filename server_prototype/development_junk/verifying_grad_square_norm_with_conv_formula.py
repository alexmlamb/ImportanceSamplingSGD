
from collections import defaultdict
import numpy as np

import theano
from theano import tensor

from blocks.bricks import MLP, Rectifier, Softmax
from blocks.bricks.conv import (ConvolutionalLayer, ConvolutionalSequence,
                                ConvolutionalActivation, Flattener)
from blocks.initialization import Constant, IsotropicGaussian, Uniform
from blocks.bricks.cost import CategoricalCrossEntropy, BinaryCrossEntropy

from blocks.graph import ComputationGraph
from blocks.filter import VariableFilter
from blocks.roles import INPUT, OUTPUT, WEIGHT, BIAS, FILTER, PARAMETER



# Note : You'll probably have to work out through shapes and transpositions.

def get_sum_square_norm_gradients_linear_transformations(D_by_layer, cost, accum = 0):

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
            print "No contribution at all to get_sum_square_norm_gradients for layer %d." % layer_name

    print "There are %d gradient components found in get_sum_square_norm_gradients." % gradient_component_count
    return accum




def get_linear_transformation_roles(mlp, cg):

    D_by_layer = defaultdict(dict)

    for (role, role_str) in [(INPUT, 'input'), (OUTPUT, 'output'), (WEIGHT, 'weight'), (BIAS, 'bias')]:

        for v in VariableFilter(bricks=mlp.linear_transformations, roles=[role])(cg.variables):
            key = v.tag.annotations[0].name
            D_by_layer[key][role_str] = v
            #D_by_layer[key][role_str] = v

    return D_by_layer





def get_sum_square_norm_gradients_conv_transformations(D_by_layer, cost, accum = 0):

    gradient_component_count = 0

    for (layer_name, D) in D_by_layer.items():

        input_square_norms = tensor.sqr(D['input']).flatten(2).sum(axis=1)
        backprop_output = tensor.grad(cost, D['output'])
        backprop_output_square_norms = tensor.sqr(backprop_output).flatten(2).sum(axis=1)

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
            print "No contribution at all to get_sum_square_norm_gradients for layer %d." % layer_name

    print "There are %d gradient components found in get_sum_square_norm_gradients." % gradient_component_count
    return accum


def get_conv_layers_transformation_roles(cg):

    # expects a cg that encapsulates only the convolutions

    D_by_layer = defaultdict(dict)

    for (role, role_str) in [(INPUT, 'input'), (OUTPUT, 'output'), (FILTER, 'weight'), (BIAS, 'bias')]:
        for v in VariableFilter(roles=[role])(cg.variables):
            key = v.tag.annotations[0].__hash__()
            D_by_layer[key][role_str] = v


    #for v in VariableFilter()(cg.parameters):
    #    # for some reason, these convolutional bricks end up with the same name "convolutional"
    #    # which isn't helpful at all
    #    print v.tag.annotations[0].__hash__()

    # Return only the components that have each of 'input', 'output', 'filter' and 'bias'.
    filtered_D_by_layer = dict((k,v) for (k,v) in D_by_layer.items() if len(v.keys()) == 4)

    return filtered_D_by_layer



def run_experiment():

    np.random.seed(42)

    #X = tensor.matrix('features')
    X = tensor.tensor4('features')
    y = tensor.matrix('targets')
    nbr_channels = 3
    image_shape = (30, 30)

    conv_layers = [ ConvolutionalLayer( filter_size=(4,4),
                                        num_filters=10,
                                        activation=Rectifier().apply,
                                        border_mode='full',
                                        pooling_size=(1,1),
                                        weights_init=Uniform(width=0.1),
                                        biases_init=Constant(0.0),
                                        name='conv0'),
                    ConvolutionalLayer( filter_size=(3,3),
                                        num_filters=14,
                                        activation=Rectifier().apply,
                                        border_mode='full',
                                        pooling_size=(1,1),
                                        weights_init=Uniform(width=0.1),
                                        biases_init=Constant(0.0),
                                        name='conv1')]
    conv_sequence = ConvolutionalSequence(  conv_layers,
                                            num_channels=nbr_channels,
                                            image_size=image_shape)
    #conv_sequence.push_allocation_config()
    conv_sequence.initialize()
    conv_output_dim = np.prod(conv_sequence.get_dim('output'))
    #conv_output_dim = 25*25

    flattener = Flattener()

    mlp = MLP(  activations=[Rectifier(), Rectifier(), Softmax()],
                dims=[conv_output_dim, 50, 50, 10],
                weights_init=IsotropicGaussian(std=0.1), biases_init=IsotropicGaussian(std=0.01))
    mlp.initialize()

    conv_output = conv_sequence.apply(X)
    y_hat = mlp.apply(flattener.apply(conv_output))

    cost = CategoricalCrossEntropy().apply(y, y_hat)
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
    individual_sum_square_norm_gradients_method_00 = get_sum_square_norm_gradients_linear_transformations(D_by_layer, cost)

    #import pprint
    #pp = pprint.PrettyPrinter(indent=4)
    #pp.pprint(get_conv_layers_transformation_roles(ComputationGraph(conv_output)).items())

    D_by_layer = get_conv_layers_transformation_roles(ComputationGraph(conv_output))
    individual_sum_square_norm_gradients_method_00 += get_sum_square_norm_gradients_conv_transformations(D_by_layer, cost)



    print "There are %d entries in cg.parameters." % len(cg.parameters)
    L_grads_method_01 = [tensor.grad(cost, p) for p in cg.parameters]
    L_grads_method_02 = [tensor.grad(cost, v) for v in VariableFilter(roles=[WEIGHT, BIAS])(cg.variables)]

    # works on the sum of the gradients in a mini-batch
    sum_square_norm_gradients_method_01 = sum([tensor.sqr(g).sum() for g in L_grads_method_01])
    sum_square_norm_gradients_method_02 = sum([tensor.sqr(g).sum() for g in L_grads_method_02])

    N = 8
    Xtrain = np.random.randn(N, nbr_channels, image_shape[0], image_shape[1]).astype(np.float32)

    # Option 1.
    ytrain = np.zeros((N, 10), dtype=np.float32)
    for n in range(N):
        label = np.random.randint(low=0, high=10)
        ytrain[n, label] = 1.0

    # Option 2, just to debug situations with NaN.
    #ytrain = np.random.rand(N, 10).astype(np.float32)
    #for n in range(N):
    #    ytrain[n,:] = ytrain[n,:] / ytrain[n,:].sum()


    f = theano.function([X,y],
                        [cost,
                            individual_sum_square_norm_gradients_method_00,
                            sum_square_norm_gradients_method_01,
                            sum_square_norm_gradients_method_02])

    [c, v0, gs1, gs2] = f(Xtrain, ytrain)

    #print "[c, v0, gs1, gs2]"
    L_c, L_v0, L_gs1, L_gs2 = ([], [], [], [])
    for n in range(N):
        [nc, nv0, ngs1, ngs2] = f(Xtrain[n,:].reshape((1,Xtrain.shape[1],Xtrain.shape[2], Xtrain.shape[3])), ytrain[n,:].reshape((1,10)))
        L_c.append(nc)
        L_v0.append(nv0)
        L_gs1.append(ngs1)
        L_gs2.append(ngs2)

    print "Cost for whole mini-batch in single shot : %f." % c
    print "Cost for whole mini-batch accumulated    : %f." % sum(L_c)
    print ""
    print "Square-norm of all gradients for each data point in single shot :"
    print v0.reshape((1,-1))
    print "Square-norm of all gradients for each data point iteratively :"
    print np.array(L_gs1).reshape((1,-1))
    print "Square-norm of all gradients for each data point iteratively :"
    print np.array(L_gs2).reshape((1,-1))
    print ""
    print "Difference max abs : %f." % np.max(np.abs(v0 - np.array(L_gs1)))
    print "Difference max abs : %f." % np.max(np.abs(v0 - np.array(L_gs2)))
    print ""
    print "Ratios : "
    print np.array(L_gs1).reshape((1,-1)) / v0.reshape((1,-1))

if __name__ == "__main__":
    run_experiment()
