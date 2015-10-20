
from collections import defaultdict
import numpy as np

import theano
from theano import tensor

class SumGradSquareNormAndVariance():

    """
    This whole class processes linear application layers
    that has the following feedforward function :
    
        X.dot(weights) + bias = Y
    
    where X.shape is (nbr_samples, input_size)
          Y.shape is (nbr_samples, output_size)
    
    In every method from this class, the parameters
    (ex : weight, bias) are always theano shared variables
    coming from a model implementation. The other arguments
    are regular theano symbolic variables (ex : input, output).

    In the case of the variance, we need to be supplied
    with the minibatch gradients themselves.
    On alternative is to supply the loss and let the
    method get the gradient that it wants.

    WARNING : If you have a model that divides the cost
              by the minibatch size, then you will get
              results that end up being divided by the
              minibatch size too.
              Conceptually, this is problematic because
              it's not really what the "individual"
              gradient square norms represent.
    """

    def __init__(self):

        # There is a "sum" in those names because we sum
        # with respect to all the coefficients.
        # It is not a sum in the sense that it keeps an array
        # where each entry corresponds to one minibatch point.
        self.accumulated_sum_gradient_square_norm = 0.0
        self.accumulated_sum_gradient_variance = 0.0

    def get_sum_gradient_square_norm(self):
        return self.accumulated_sum_gradient_square_norm

    def get_sum_gradient_variance(self):
        return self.accumulated_sum_gradient_variance



    def add_layer_for_gradient_square_norm( self, input, weight, bias,
                                            output=None, cost=None,
                                            backprop_output=None):

        # `input` is the input to the layer; a theano symbolic variable
        #
        # `weight` : parameter; a theano shared variable
        # `bias`: parameter; a theano shared variable (can be `None`)
        #
        # `output` is the output of that layer; a theano symbolic variable
        # `cost` is the cost at the end of the model; a theano symbolic variable
        #
        # `backprop_output` is tensor.grad(cost, output); a theano symbolic variable
        #
        # You need to supply either `output` and `cost`, or just the `backprop_output`.

        if backprop_output is None:
            assert output is not None
            assert cost is not None
            backprop_output = tensor.grad(cost, output)

        input_square_norms = tensor.sqr(input).sum(axis=1)
        backprop_output_square_norms = tensor.sqr(backprop_output).sum(axis=1)

        if bias is not None:
            self.accumulated_sum_gradient_square_norm += (input_square_norms + 1) * backprop_output_square_norms
        else:
            self.accumulated_sum_gradient_square_norm += input_square_norms * backprop_output_square_norms


    def add_layer_for_gradient_variance(self, input, weight, bias,
                                        output=None, cost=None,
                                        backprop_output=None, backprop_weight=None, backprop_bias=None):

        # Supply all values from either {output, cost} or {backprop_output, backprop_weight, backprop_bias}.

        if backprop_output is None:
            assert output is not None
            assert cost is not None
            backprop_output = tensor.grad(cost, output)

        if backprop_weight is None:
            assert cost is not None
            backprop_weight = tensor.grad(cost, weight)

        if bias is None:
            backprop_bias = None
        elif bias is not None and backprop_bias is None:
            assert cost is not None
            backprop_bias = tensor.grad(cost, bias)


        input_square_norms = tensor.sqr(input).sum(axis=1)
        backprop_output_square_norms = tensor.sqr(backprop_output).sum(axis=1)

        A = input_square_norms * backprop_output_square_norms
        C = tensor.sqr(backprop_weight).sum() # all the terms get this "middle" expression added to them
        B = (backprop_output.dot(backprop_weight.T) * input).sum(axis=1)

        self.accumulated_sum_gradient_variance += (A - 2*B + C)

        if backprop_bias is not None:
            # this last `sum` could be a component-wise `max` if we wanted
            # to carry the maximum of the variances instead of the sum of squares
            self.accumulated_sum_gradient_variance += tensor.sqr(backprop_output - backprop_bias.reshape((1,-1))).sum(axis=1)


    # If you're using a blocks model, you can use this alternative constructor.

    @classmethod
    def make_from_blocks(cls, mlp, cg, cost):

        instance = cls()
        
        from blocks.graph import ComputationGraph
        from blocks.filter import VariableFilter
        from blocks.roles import INPUT, OUTPUT, WEIGHT, BIAS


        D_by_layer = defaultdict(dict)
        for (role, role_str) in [(INPUT, 'input'), (OUTPUT, 'output'), (WEIGHT, 'weight'), (BIAS, 'bias')]:
            for v in VariableFilter(bricks=mlp.linear_transformations, roles=[role])(cg.variables):
                key = v.tag.annotations[0].name
                D_by_layer[key][role_str] = v
                #D_by_layer[key][role_str] = v


        for (layer_name, D) in D_by_layer.items():

            weight = D['weight']
            if D.has_key('bias'):
                bias = D['bias']
            else:
                bias = None
            input = D['input']
            output = D['output']

            instance.add_layer_for_gradient_square_norm(input=input, weight=weight, bias=bias, output=output, cost=cost)
            instance.add_layer_for_gradient_variance(input=input, weight=weight, bias=bias, output=output, cost=cost)

        return instance





