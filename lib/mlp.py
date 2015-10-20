"""
This tutorial introduces the multilayer perceptron using Theano.

 A multilayer perceptron is a logistic regressor where
instead of feeding the input to the logistic regression you insert a
intermediate layer, called the hidden layer, that has a nonlinear
activation function (usually tanh or sigmoid) . One can use many such
hidden layers making the architecture deep. The tutorial will also tackle
the problem of MNIST digit classification.

.. math::

    f(x) = G( b^{(2)} + W^{(2)}( s( b^{(1)} + W^{(1)} x))),

References:

    - textbooks: "Pattern Recognition and Machine Learning" -
                 Christopher M. Bishop, section 5

"""
__docformat__ = 'restructedtext en'


import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T
import random
import time
import matplotlib.pyplot as plt

from logistic_sgd import LogisticRegression, load_data, load_data_svhn
from jacobian_forloop import jacobian_forloop
from ImportanceSampling import sampleInstances

theano.config.floatX = 'float32'

config = {}
#config["dataset"] = "mnist"
config["dataset"] = "svhn"

if config["dataset"] == "svhn":
    input_size = 32 * 32 * 3
elif config["dataset"] == "mnist":
    input_size = 28 * 28

# start-snippet-1
class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input
        # end-snippet-1

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]


# start-snippet-2
class MLP(object):
    """Multi-Layer Perceptron Class

    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function tanh or the
    sigmoid function (defined here by a ``HiddenLayer`` class)  while the
    top layer is a softmax layer (defined here by a ``LogisticRegression``
    class).
    """

    def __init__(self, rng, input, n_in, n_hidden, n_out):
        """Initialize the parameters for the multilayer perceptron

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie

        :type n_hidden: int
        :param n_hidden: number of hidden units

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie

        """

        self.hiddenLayer1 = HiddenLayer(
            rng=rng,
            input=input,
            n_in=n_in,
            n_out=n_hidden,
            activation=(lambda layer_input: T.maximum(0.0, layer_input))
        )

        self.hiddenLayer2 = HiddenLayer(rng=rng, input=self.hiddenLayer1.output,n_in=n_hidden,n_out=n_hidden,activation=(lambda layer_input: T.maximum(0.0, layer_input)))

        # The logistic regression layer gets as input the hidden units
        # of the hidden layer
        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenLayer2.output,
            n_in=n_hidden,
            n_out=n_out
        )
        # end-snippet-2 start-snippet-3
        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.L1 = (
            abs(self.hiddenLayer1.W).sum() + abs(self.hiddenLayer2.W).sum()
            + abs(self.logRegressionLayer.W).sum()
        )

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = (
            (self.hiddenLayer1.W ** 2).sum() + (self.hiddenLayer2.W ** 2).sum()
            + (self.logRegressionLayer.W ** 2).sum()
        )

        # negative log likelihood of the MLP is given by the negative
        # log likelihood of the output of the model, computed in the
        # logistic regression layer
        self.negative_log_likelihood = (
            self.logRegressionLayer.negative_log_likelihood
        )
        # same holds for the function computing the number of errors
        self.errors = self.logRegressionLayer.errors

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = self.hiddenLayer1.params + self.hiddenLayer2.params + self.logRegressionLayer.params
        # end-snippet-3

        self.layers = [input, self.hiddenLayer1.output, self.hiddenLayer2.output, self.logRegressionLayer.output]

        # keep track of model input
        self.input = input

#was using 500
def test_mlp(base_learning_rate=0.1, L1_reg=0.00, L2_reg=0.001, n_epochs=9999000,
             dataset='mnist.pkl.gz', batch_size=128, n_hidden=1024):
    """
    Demonstrate stochastic gradient descent optimization for a multilayer
    perceptron

    This is demonstrated on MNIST.

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
    gradient

    :type L1_reg: float
    :param L1_reg: L1-norm's weight when added to the cost (see
    regularization)

    :type L2_reg: float
    :param L2_reg: L2-norm's weight when added to the cost (see
    regularization)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: the path of the MNIST dataset file from
                 http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz


   """

    upsample_zero = True

    if config["dataset"] == "mnist": 
        datasets = load_data(dataset, upsample_zero)
    elif config["dataset"] == "svhn":
        datasets = load_data_svhn()

    print "Datasets loaded into memory"

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    print "printing all shapes"
    print train_set_x.shape
    print valid_set_x.shape
    print test_set_x.shape

    # compute number of minibatches for training, validation and testing
    n_train_batches = len(train_set_x) / batch_size
    n_valid_batches = len(valid_set_x) / batch_size
    n_test_batches = len(test_set_x) / batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # allocate symbolic variables for the data
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    learning_rate = T.scalar('learning_rate')

    imp_weights = T.vector('importance weights')

    rng = numpy.random.RandomState(1234)

    # construct the MLP class
    classifier = MLP(
        rng=rng,
        input=x,
        n_in=input_size,
        n_hidden=n_hidden,
        n_out=10
    )

    # start-snippet-4
    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    cost = classifier.negative_log_likelihood(y) * imp_weights + L2_reg * classifier.L2_sqr
    # end-snippet-4

    get_cost = theano.function(inputs = [x,y], outputs = [classifier.negative_log_likelihood(y), classifier.errors(y)])

    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    print "compiling test model function"
    test_model = theano.function(
        inputs=[x,y],
        outputs=classifier.errors(y),
        allow_input_downcast = True
    )

    validate_model = theano.function(
        inputs=[x,y],
        outputs=classifier.errors(y),
        allow_input_downcast = True
    )

    # start-snippet-5
    # compute the gradient of cost with respect to theta (sotred in params)
    # the resulting gradients will be stored in a list gparams
    #gparams = [T.grad(cost, param) for param in classifier.params]

    gparams = []

    for param in classifier.params:

        grad_param = T.grad(T.mean(cost), param)

        gparams += [grad_param]

    gparams_mom = []

    for param in classifier.params:

        gparam_mom = theano.shared(numpy.zeros(param.get_value(borrow=True).shape, dtype = 'float32'))

        gparams_mom += [gparam_mom]

    #Function that computes gradient norm

    def compute_gradient_norm(loss, layerLst):
        gradient_norm_f = 0.0

        for index in range(1, len(layerLst)):
            output_layer = layerLst[index]
            input_layer = layerLst[index - 1]
            gradient_norm_f += (input_layer**2).sum(axis = 1) * (T.grad(loss, output_layer)**2).sum(axis = 1)

        gradient_norm_f = T.sqrt(gradient_norm_f)

        return gradient_norm_f


    get_gradient = theano.function(inputs = [x,y], outputs = [compute_gradient_norm(T.sum(classifier.negative_log_likelihood(y)), classifier.layers)], allow_input_downcast = True)

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs

    # given two lists of the same length, A = [a1, a2, a3, a4] and
    # B = [b1, b2, b3, b4], zip generates a list C of same size, where each
    # element is a pair formed from the two lists :
    #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
    updates = [
        (param, T.cast(param - learning_rate * gparam_mom, theano.config.floatX))
        for param, gparam_mom in zip(classifier.params, gparams_mom)
    ]

    momentum_rate = 0.0
    print "running with momentum rate", momentum_rate

    for i in range(0, len(gparams_mom)):
        gparam_mom = gparams_mom[i]
        gparam = gparams[i]
        new_gparam_mom = gparam_mom * theano.shared(numpy.asarray(momentum_rate, dtype = 'float32')) + theano.shared(numpy.asarray(1.0 - momentum_rate, dtype = 'float32')) * gparam
        updates += [(gparam_mom, T.cast(new_gparam_mom, theano.config.floatX))]

    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    print "compiling training model"
    t2 = time.time()
    train_model = theano.function(
        inputs=[x, y, learning_rate, imp_weights],
        outputs=[T.mean(classifier.negative_log_likelihood(y)), compute_gradient_norm(T.sum(classifier.negative_log_likelihood(y)), classifier.layers)],
        updates=updates,
        allow_input_downcast = True
    )
    print "compilation finished in", time.time() - t2
    # end-snippet-5

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'

    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    validation_frequency = 100

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    trainErrorRate = 0.0
    averageTrainCost = 0.0
    trainCostMap = {}

    logFile = open('logs/mnist_isgd_saved.txt', 'w')
    doValidate = False

    epoch = 0
    done_looping = False

    x_mb_train = []
    y_mb_train = []
    x_mb_valid = []
    y_mb_valid = []
    x_mb_test = []
    y_mb_test = []
    mb_indices = []
    importance_weights_mb = []

    minibatch_index = 0
 

    t0 = time.time()

    maxGradientNorm = 1.0
    costMap = {}
    freshnessMap = {}
    gradientMap = {}

    for k in range(0,len(train_set_x)):
        freshnessMap[k] = -1
        gradientMap[k] = 1.0

    compute_gradient_all = True
    compute_cost_all = False

    print "Number training examples", len(train_set_x)
    print "Learning Rate", base_learning_rate

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1

        indexLst = range(0, len(train_set_x))

        random.shuffle(indexLst)


        x_mb_train = []
        y_mb_train = []
        mb_indices = []


        

        t1 = time.time()

        cost_mb_size = 2000
        grad_mb_size = 2000


        x_mb_grad = []
        y_mb_grad = []
        g_indices = []

        imp_indices = range(0, len(train_set_x))
        random.shuffle(imp_indices)

        #Taking 20000
        #Compute cost over all instances.  
        for i in imp_indices[:20000]:

            if not compute_cost_all and not compute_gradient_all:
                break

            all_error_lst = []

            if random.uniform(0,1) < 1.0: 
                x_mb_train += [train_set_x[i]]
                y_mb_train += [train_set_y[i]]
                mb_indices += [i]
                x_mb_grad += [train_set_x[i]]
                y_mb_grad += [train_set_y[i]]
                g_indices += [i]

            if len(x_mb_train) == cost_mb_size:

                if compute_cost_all:
                    cost_lst, error_lst = get_cost(numpy.asarray(x_mb_train, dtype = 'float32'), numpy.asarray(y_mb_train, dtype = 'int32'))
                    all_error_lst += [error_lst]

                    for j in range(0, cost_mb_size):
                        costMap[mb_indices[j]] = cost_lst[j].tolist()

                x_mb_train = []
                y_mb_train = []
                mb_indices = []

            if len(x_mb_grad) == grad_mb_size:
                if compute_gradient_all:
                    gradient_lst = get_gradient(numpy.asarray(x_mb_grad, dtype = 'float32'), numpy.asarray(y_mb_grad, dtype = 'int32'))


                    for j in range(0, grad_mb_size): 
                        gradientMap[g_indices[j]] = gradient_lst[0][j].tolist()
                        freshnessMap[g_indices[j]] = minibatch_index

                g_indices = []
                x_mb_grad = []
                y_mb_grad = []


        #print "computed all costs for minibatch in", time.time() - t1

        if compute_cost_all:
            sumCost = sum(costMap.values())
            numCost = len(costMap)
            trainErrorRate = numpy.mean(all_error_lst)
            averageTrainCost = sumCost / numCost

        #print "Average cost", averageCost
        #print "Number costs", numCost
        #print "Variance cost", numpy.asarray(costMap.values()).var()
        #print "max cost", max(costMap.values())


        #Sample from indices where mbIndex - fMap[index] < k.  


        #indexLst,importanceWeights = sampleInstances(indexLst, gradientMap, batch_size, freshnessMap, minibatch_index)

        indexLst,importanceWeights = indexLst[:batch_size], [1.0] * batch_size

        #indexLstImp, importanceWeightsFirstHalf = sampleInstances(indexLst, gradientMap, batch_size / 4, freshnessMap, minibatch_index)
        #indexLst = indexLst[:batch_size * 3 / 4] + indexLstImp
        #importanceWeights = [1.0] * (batch_size * 3 / 4) + importanceWeightsFirstHalf

        assert len(importanceWeights) == 128

        #if random.uniform(0,1) < 0.1:
        #    print importanceWeights, max(importanceWeights), min(importanceWeights)

        #averageCostTrain = sum(costMap) / len(costMap)

        t0_e = time.time()

        for i in indexLst:

            t0_e = time.time()

            #doTrain could be based on uniform sampling
            doTrain = True

            if doTrain: 
                x_mb_train += [train_set_x[i]]
                y_mb_train += [train_set_y[i]]
                mb_indices += [i]


            if len(x_mb_train) == batch_size: 

                
                train_model_outputs = train_model(numpy.asarray(x_mb_train, dtype = 'float32'), numpy.asarray(y_mb_train, dtype = 'int32'), numpy.asarray(base_learning_rate, dtype = 'float32'),numpy.asarray(importanceWeights, dtype = 'float32'))

                minibatch_avg_cost, gradient_norm_lst = train_model_outputs

                #Update gradientMap


                for k in range(0, len(mb_indices)):
                    #freshnessMap[mb_indices[k]] = minibatch_index
                    #gradientMap[mb_indices[k]] = gradient_norm_lst[k]
                    trainCostMap[mb_indices[k]] = minibatch_avg_cost

                if doValidate:
                    print "Gradient Norm", gradient_norm_lst.sum()

                minibatch_index += 1
                x_mb_train = []
                y_mb_train = []
                mb_indices = []
                importance_weights_mb = []

                iter = (epoch - 1) * n_train_batches + minibatch_index


                if minibatch_index % validation_frequency == 0:
                    doValidate = True
                    # compute zero-one loss on validation set
                    validation_losses = []
                
                    for j in range(0, len(valid_set_x)): 
                        x_mb_valid += [valid_set_x[j]]
                        y_mb_valid += [valid_set_y[j]]

                        if len(x_mb_valid) == batch_size: 
                            validation_losses += [validate_model(x_mb_valid, y_mb_valid)]

                            x_mb_valid = []
                            y_mb_valid = []

                    this_validation_loss = numpy.mean(numpy.asarray(validation_losses))

                    print "time", time.time() - t0
                    t0 = time.time()

                    trainErrorRate = 0.0

                    averageTrainCost = sum(trainCostMap.values()) / len(trainCostMap)

                    print(
                        'MB Processed %i, validation error %f %%, training error %f %%, training log-likelihood %f' %
                        (
                            epoch,
                            this_validation_loss * 100.,
                            trainErrorRate * 100,
                            averageTrainCost
                        )
                    )

                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:
                        #improve patience if loss improvement is good enough
                        if (
                            this_validation_loss < best_validation_loss *
                            improvement_threshold
                        ):
                            patience = max(patience, iter * patience_increase)
    
                        best_validation_loss = this_validation_loss
                        best_iter = iter

                        # test it on the test set
                        test_losses = []
                    
                        for j in range(0, len(test_set_x)): 
                            x_mb_test += [test_set_x[j]]
                            y_mb_test += [test_set_y[j]]

                            if len(x_mb_test) == batch_size: 
                                test_losses += [test_model(x_mb_test, y_mb_test)]

                                x_mb_test = []
                                y_mb_test = []
                    
                    
                        test_score = numpy.mean(test_losses)
    
                        print(('     epoch %i, minibatch %i/%i, test error of '
                               'best model %f %%') %
                              (epoch, minibatch_index + 1, n_train_batches,
                               test_score * 100.))

                else:
                    doValidate = False

                if patience <= iter:
                    done_looping = False
                    #break


    end_time = timeit.default_timer()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))


if __name__ == '__main__':
    test_mlp()
