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
from fuel.schemes import SequentialScheme, ShuffledScheme
from blocks.algorithms import GradientDescent
from blocks.bricks.cost import CategoricalCrossEntropy
from blocks.algorithms import Scale, Adam, AdaGrad, AdaDelta
from blocks.main_loop import MainLoop
from blocks.extensions import FinishAfter, Printing
from blocks.model import Model
from blocks.extensions import FinishAfter, Timing, Printing, ProgressBar
from blocks.extensions.monitoring import DataStreamMonitoring
from blocks.initialization import IsotropicGaussian, Constant, Orthogonal

config = {}

config["dataset"] = MNIST
config["minibatch_size"] = 2048

if config["dataset"] == MNIST:
    training_set = config["dataset"](["train"])
    testing_set = config["dataset"](["test"])
    input_size = 28 * 28
elif config["dataset"] == SVHN:
    training_set = config["dataset"](2, ["train", "extra"])
    testing_set = config["dataset"](2, ["test"])
    input_size = 32 * 32 * 3

print "Number examples", training_set.num_examples

#x = T.tensor3("features") #(NumExamples, X, Y)
x = T.tensor4('features') #(NumExamples, X*Y)
x_flat = x.flatten(2)
y = T.lmatrix('targets')

#Use x_flat

hidden_size = 4000
input_to_hidden = Linear(name='input_to_hidden', input_dim=input_size, output_dim=hidden_size)
h = Rectifier("H1").apply(input_to_hidden.apply(x_flat))
hidden_to_hidden = Linear(name='hidden_to_hidden', input_dim = hidden_size, output_dim = hidden_size)
h2 = Rectifier("H2").apply(hidden_to_hidden.apply(h))
hidden_to_output = Linear(name='hidden_to_output', input_dim=hidden_size, output_dim=10)
y_hat = Softmax().apply(hidden_to_output.apply(h2))

input_to_hidden.weights_init = hidden_to_output.weights_init = hidden_to_hidden.weights_init = Orthogonal(0.01)
input_to_hidden.biases_init = hidden_to_output.biases_init = hidden_to_hidden.biases_init = Constant(0)
input_to_hidden.initialize()
hidden_to_hidden.initialize()
hidden_to_output.initialize()

layer_list = [input_to_hidden, hidden_to_output]

cost = CategoricalCrossEntropy().apply(y.flatten(), y_hat)

data_stream = DataStream(training_set, iteration_scheme = ShuffledScheme(training_set.num_examples, batch_size = config["minibatch_size"]))

gradients = {}

for lt in layer_list:
    for param in lt.parameters:
        gradients[param] = T.grad(cost, param)

algorithm = GradientDescent(cost = cost, step_rule = AdaDelta(), gradients = gradients)

train_monitoring = DataStreamMonitoring(variables = [cost], data_stream = data_stream, prefix = "train")

print "param value", param.get_value()

main_loop = MainLoop(model=Model(cost), data_stream=data_stream, algorithm=algorithm, extensions=[Timing(), train_monitoring, FinishAfter(after_n_epochs=200), Printing()])

main_loop.run()

print "param value", param.get_value()





