
import theano
from theano import tensor as T
import numpy as np
from theano import shared
from theano import function

from theano import tensor as T
from load import mnist
from load import mnist_with_noise
from scipy.misc import imsave
import scipy as sp

from scipy import signal
from load_data import load_data, normalizeMatrix
import cPickle as pickle

# Tool to get the special formulas for gradient norms and variance.
from fast_individual_gradient_norms.expression_builder import SumGradSquareNormAndVariance

class NeuralNetwork:


    def __init__(self, model_config):

        nhidden_layers = len(model_config["hidden_sizes"])
        nhidden = model_config["hidden_sizes"][0]
        print "num_hidden_layers      :",nhidden_layers
        print "hidden_units_per_layer :",nhidden
        X = T.fmatrix()
        Y = T.ivector()
        scaling_factors = T.fvector()
        num_input = model_config["num_input"]
        num_output = 10

        self.num_minibatches_processed_master = 0
        self.num_minibatches_processed_worker = 0

        L_W, L_b = NeuralNetwork.build_parameters(num_input, num_output, model_config["hidden_sizes"], scale=0.01)
        L_W_momentum, L_b_momentum, = NeuralNetwork.build_parameters(num_input, num_output, model_config["hidden_sizes"], scale=0.0, name_suffix="_momentum")
        self.parameters = L_W + L_b
        self.momentum   = L_W_momentum + L_b_momentum

        print self.parameters

        (L_layer_inputs, L_layer_desc) = NeuralNetwork.build_layers(X, L_W, L_b)
        py_x = L_layer_inputs[-1]

        y_x = T.argmax(py_x, axis=1)

        individual_cost = -1.0 * (T.log(py_x)[T.arange(Y.shape[0]), Y])
        # It's acceptable to use the `mean` here instead of the `sum`,
        # but to get the gradient_square_norm we will really have
        # to multiply them by N**2. Otherwise, they get "unfairly"
        # divided by this N factor, which then is getting squared.
        # We do this to preserve the property that the cost is not
        # getting scaled by the minibatch size, which is generally
        # a desirable property that we want in order to be able to
        # compare different hyper-parameters.
        # Note that multiplying the cost by N before passing it
        # to the `sgsnav` would result in two computational graphs,
        # which is a bad idea. Instead we'll have to scale the values
        # returned by the `sgsnav`.
        cost = T.mean(individual_cost)

        scaled_individual_cost = scaling_factors * individual_cost
        # Here this is `mean`, because the scaling factors
        # do not sum to 1.0. Intuitively, something that was twice
        # as likely to occur than unde the Uniform Sampling regime
        # will have a scaling factor of 0.5.
        scaled_cost = T.mean(scaled_individual_cost)

        updates = NeuralNetwork.sgd(scaled_cost, self.parameters, self.momentum, model_config["learning_rate"], model_config["momentum_rate"])

        # To be clear, this is the square norm of the (mean gradient in the minibatch).
        # Not to be confused with the mean of the (square norm of the gradients of each element of the minibatch).
        minibatch_gradient_mean_square_norm = sum([T.sqr(T.grad(cost, p)).sum() for p in self.parameters])


        sgsnav = SumGradSquareNormAndVariance()
        for layer_desc in L_layer_desc:
            sgsnav.add_layer_for_gradient_square_norm(input=layer_desc['input'], weight=layer_desc['weight'],
                                                      bias=layer_desc['bias'], output=layer_desc['output'], cost=cost)

            #sgsnav.add_layer_for_gradient_variance( input=layer_desc['input'], weight=layer_desc['weight'],
            #                                        bias=layer_desc['bias'], output=layer_desc['output'], cost=cost)

        individual_gradient_square_norm = sgsnav.get_sum_gradient_square_norm() * (X.shape[0]**2)
        #individual_gradient_variance = sgsnav.get_sum_gradient_variance()

        mean_gradient_square_norm = T.mean(individual_gradient_square_norm)
        #mean_gradient_variance = T.mean(individual_gradient_variance)

        individual_accuracy = T.eq(T.argmax(py_x, axis = 1), Y)
        accuracy = T.mean(individual_accuracy)

        # At this point we compile two auxiliary theano functions
        # that are going to do all the heavy-lifting for the corresponding
        # methods `worker_process_minibatch` and `master_process_minibatch`.

        # Note the absence of updates in this function.
        self.func_process_worker_minibatch = theano.function(inputs=[X, Y],
                                                            outputs=[individual_cost,
                                                                     individual_accuracy,
                                                                     individual_gradient_square_norm,
                                                                     minibatch_gradient_mean_square_norm],
                                                            allow_input_downcast=True)

        self.func_master_process_minibatch = theano.function(inputs=[X, Y, scaling_factors],
                                     outputs=[cost, accuracy, mean_gradient_square_norm, individual_gradient_square_norm],
                                     updates=updates, allow_input_downcast=True)

        # This is just never getting used.
        #self.predict = theano.function(inputs=[X], outputs=[y_x, py_x], allow_input_downcast=True)


        print "Model compilation complete"
        self.data = load_data(model_config)
        self.mean = self.data["mean"]
        self.std = self.data["std"]
        print "%s data loaded..." % model_config["dataset"]


    @staticmethod
    def build_layers(X, L_W, L_b):
        L_layer_inputs = [X]
        L_layer_desc = []
        for i in range(0, len(L_W)):

            print "accessing layer", i

            # the next inputs are always the last ones in the list
            inputs = L_layer_inputs[-1]
            weights = L_W[i]
            biases = L_b[i]
            activations = biases + T.dot(inputs, weights)

            if i < len(L_W) - 1:
                # all other layers except the last
                layer_outputs = T.maximum(0.0, activations)
            else:
                # last layer
                layer_outputs = T.nnet.softmax(activations)

            L_layer_inputs.append(layer_outputs)

            # This is formatted to be fed to the "fast_individual_gradient_norms".
            # The naming changes a bit because of that, because we're realling referring
            # to the linear component itself and not the stuff that happens after.
            layer_desc = {'weight' : weights, 'bias' : biases, 'output':activations, 'input':inputs}
            L_layer_desc.append(layer_desc)

        return L_layer_inputs, L_layer_desc


    @staticmethod
    def floatX(X):
        return np.asarray(X, dtype=theano.config.floatX)

    @staticmethod
    def init_weights(shape, name, scale = 0.01):
        return theano.shared(NeuralNetwork.floatX(np.random.randn(*shape) * scale), name=name)

    @staticmethod
    def sgd(cost, params, momemtum, lr, mr):
        grads = T.grad(cost=cost, wrt=params)
        updates = []
        for p, g, v in zip(params, grads, momemtum):
            v_prev = v
            updates.append([v, mr * v - g * lr])
            v = mr*v - g*lr
            updates.append([p, p  - mr*v_prev + (1 + mr)*v ])

        return updates


    @staticmethod
    def build_parameters(num_input, num_output, hidden_sizes, scale, name_suffix=""):

        # The name suffix is to be used in the case of the momentum.

        L_sizes = [num_input] + hidden_sizes + [num_output]
        L_W = []
        L_b = []

        for (layer_number, (dim_in, dim_out)) in enumerate(zip(L_sizes, L_sizes[1:])):
            W = NeuralNetwork.init_weights((dim_in, dim_out), scale=scale, name=("%0.3d_weight%s"%(layer_number, name_suffix)))
            b = NeuralNetwork.init_weights((dim_out,), scale=0.0, name=("%0.3d_bias%s"%(layer_number, name_suffix)))
            L_W.append(W)
            L_b.append(b)

        return L_W, L_b



    def worker_process_minibatch(self, A_indices, segment, L_measurements):
        assert segment in ["train", "valid", "test"]
        for key in L_measurements:
            assert key in [ "individual_importance_weight", "individual_gradient_square_norm",
                            "individual_loss", "individual_accuracy",
                            "minibatch_gradient_mean_square_norm",
                            # old measurement names
                            "importance_weight", "gradient_square_norm",
                            "loss", "accuracy"]

        X_minibatch = normalizeMatrix(self.data[segment][0][A_indices], self.mean, self.std)
        Y_minibatch = self.data[segment][1][A_indices]
        assert np.all(np.isfinite(X_minibatch))
        assert np.all(np.isfinite(Y_minibatch))

        # These numpy arrays here have the same names as theano variables
        # elsewhere in this class. Don't get confused.
        (individual_cost, individual_accuracy, individual_gradient_square_norm, minibatch_gradient_mean_square_norm) = self.func_process_worker_minibatch(X_minibatch, Y_minibatch)

        individual_importance_weight = np.sqrt(individual_gradient_square_norm)

        # DEBUG : Set all those quantities to something random, and see what happens.
        #         This is not the same thing as actually influencing the gradients.
        #         We are just messing with the values in the database.
        # Tip :   Use values that don't have expectation 0. The trouble with
        #         things that have expectation 0 is that you get a mu2 term
        #         that is very small and difficult to measure accurately.
        #         With random values, and expectation 0, you are easily
        #         misled to conclude that you cannot do anything useful
        #         with the measurements available because the differences
        #         to estimate Trace(Covariance) lead to negative values !

        #print "Before override."
        #print "individual_gradient_square_norm.shape : %s" % str(individual_gradient_square_norm.shape,)
        #print "minibatch_gradient_mean_square_norm.shape : %s" % str(minibatch_gradient_mean_square_norm.shape,)
        #(N, d) = (individual_gradient_square_norm.shape[0], 10)
        #G = (np.random.randn(N, d) + np.tile(np.arange(d), (N,1))).astype(np.float32)
        #G = np.tile(np.arange(d), (N,1)).astype(np.float32)
        #individual_gradient_square_norm = (G**2).sum(axis=1)
        #minibatch_gradient_mean_square_norm = ((G.mean(axis=0))**2).sum()
        #print "After override."
        #print "individual_gradient_square_norm.shape : %s" % str(individual_gradient_square_norm.shape,)
        #print "minibatch_gradient_mean_square_norm.shape : %s" % str(minibatch_gradient_mean_square_norm.shape,)

        # The individual_gradient_square_norm.mean() and individual_gradient_variance.mean()
        # are not written to the database, but they would be really nice to log to have a
        # good idea of what is happening on the worker.

        self.num_minibatches_processed_worker += 1

        # We can change the quantity that corresponds to 'importance_weight'
        # by changing the entry in the `mapping` dictionary below.
        mapping = { 'individual_importance_weight' : individual_importance_weight,
                    'individual_cost' : individual_cost,
                    'individual_loss' : individual_cost,
                    'individual_accuracy' : individual_accuracy.astype(dtype=np.float32),
                    'individual_gradient_square_norm' : individual_gradient_square_norm,
                    'minibatch_gradient_mean_square_norm' : np.array(minibatch_gradient_mean_square_norm),
                    # old measurement names
                    'importance_weight' : individual_importance_weight,
                    'cost' : individual_cost,
                    'loss' : individual_cost,
                    'accuracy' : individual_accuracy.astype(dtype=np.float32),
                    'gradient_square_norm' : individual_gradient_square_norm
                    }

        # Returns a full array for every data point in the minibatch.
        res = dict((measurement, mapping[measurement]) for measurement in L_measurements)
        return res


    def master_process_minibatch(self, A_indices, A_scaling_factors, segment):
        assert A_indices.shape == A_scaling_factors.shape, "Failed to assertion that %s == %s." % (A_indices.shape, A_scaling_factors.shape)
        assert segment in ["train"]

        X_minibatch = normalizeMatrix(self.data[segment][0][A_indices], self.mean, self.std)
        Y_minibatch = self.data[segment][1][A_indices]
        assert np.all(np.isfinite(X_minibatch))
        assert np.all(np.isfinite(Y_minibatch))

        # These numpy arrays here have the same names as theano variables
        # elsewhere in this class. Don't get confused.
        (cost, accuracy, mean_gradient_square_norm, individual_gradient_square_norm) = self.func_master_process_minibatch(X_minibatch, Y_minibatch, A_scaling_factors)

        self.num_minibatches_processed_master += 1

        # The mean_gradient_square_norm and mean_gradient_variance
        # are not written to the database, but they would be really nice to log to have a
        # good idea of what is happening on the master.

        # Returns nothing. The master should have used this call to
        # update its internal parameters.
        # return

        # For use in debugging, we return here the `individual_gradient_square_norm`.
        # This can change to suit our debugging needs.
        return individual_gradient_square_norm


