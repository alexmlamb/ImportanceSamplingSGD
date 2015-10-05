'''
Main code for running importance sampled sgd code.  
'''

import theano
import theano.tensor as T
from blocks.bricks import Linear, Rectifier, Softmax
from blocks.bricks import MLP
import fuel
import fuel.datasets
from fuel.datasets.mnist import MNIST
from fuel.datasets.svhn import SVHN
from fuel.datasets.cifar100 import CIFAR100
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme
from blocks.algorithms import GradientDescent
from blocks.bricks.cost import CategoricalCrossEntropy
from blocks.algorithms import Scale, Adam, AdaGrad, AdaDelta
from blocks.main_loop import MainLoop
from blocks.extensions import FinishAfter, Printing
from blocks.model import Model

config = {}

config["dataset"] = MNIST
config["minibatch_size"] = 256


training_set = config["dataset"](["train"])
testing_set = config["dataset"](["test"])

#x = T.tensor3("features") #(NumExamples, X, Y)
x = T.tensor4('features') #(NumExamples, X*Y)
x_flat = x.flatten(2)
y = T.lmatrix('targets')


model = MLP(activations = [Rectifier(), Rectifier(), Softmax()], dims = [784,400,400,10])

y_hat = model.apply(x_flat)

cost = CategoricalCrossEntropy().apply(y.flatten(), y_hat)

data_stream = DataStream(training_set, iteration_scheme = SequentialScheme(training_set.num_examples, batch_size = config["minibatch_size"]))

gradients = {}

for lt in model.linear_transformations:
    for param in lt.parameters:
        gradients[param] = T.grad(cost, param)

algorithm = GradientDescent(cost = cost, step_rule = Scale(0.01), gradients = gradients)


main_loop = MainLoop(model=Model(cost), data_stream=data_stream, algorithm=algorithm, extensions=[FinishAfter(after_n_epochs=200), Printing()])

main_loop.run()







