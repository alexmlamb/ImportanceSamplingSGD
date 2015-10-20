
import os

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.objectives import categorical_crossentropy

import theano


class SVHN2DatasetOnGPU(object):

    def __init__(self, hdf5path=None):
        # TODO : infer the location automatically based on what Fuel downloaded

        if hdf5path is None and os.environ.has_key("FUEL_DATA_PATH"):
            hdf5path = os.path.join(os.environ["FUEL_DATA_PATH"], "svhn_format_2.hdf5")
        else:
            assert os.path.exists(hdf5path) and os.path.isfile(hdf5path)

        import h5py
        f = h5py.File(hdf5path, mode='r')
        
        # https://github.com/mila-udem/fuel/blob/fuel/datasets/svhn.py
        # Range [0, 73257) along with [99289, 630420).
        Ntrain = 73257-0 + 630420-99289

        # Range [73257, 99289).
        Ntest = 99289 - 73257


        # Ouch. It's 1.9GB when each value is an uint8.
        # This means that it gets to 7.6GB as float32.
        # You can't expect to store this on the graphics card.
        # Maybe we should, and that's the whole point of
        # having Titan X cards.
        # Maybe Theano would run fine with this anyways,
        # and it would do the conversion automatically ?
        # Yeah, it seems like this is totally feasible.
        # Theano will do the conversion. Just divide the values
        # by 255.0 right after loading them.
        # Is that even necessary ?

class ImportanceSamplingModel(object):

    def __init__(self):

        optimizer = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        self._build_model(optimizer)

    def _build_model(self, optimizer):

        model = Sequential()
        model.add(Convolution2D(32, 3, 3, 3, border_mode='full')) 
        model.add(Activation('relu'))
        model.add(Convolution2D(32, 32, 3, 3))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(poolsize=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Convolution2D(64, 32, 3, 3, border_mode='full')) 
        model.add(Activation('relu'))
        model.add(Convolution2D(64, 64, 3, 3)) 
        model.add(Activation('relu'))
        model.add(MaxPooling2D(poolsize=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(64*8*8, 256))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Dense(256, 10))
        model.add(Activation('softmax'))

        # Get the categorical_crossentropy from the Keras source
        #     https://github.com/fchollet/keras/blob/master/keras/objectives.py
        # and multiply it by a shared variable, which will be used
        # the incorporate the effects of importance sampling weights.
        #
        # Actually, there is a way to implement this with sample_weight in Keras directly.
        #self.loss_scaling_factor = theano.shared(1.0, name='loss_scaling_factor')
        #def special_categorical_crossentropy(y_true, y_pred):
        #    return self.loss_scaling_factor * categorical_crossentropy(y_true, y_pred)
        #model.compile(loss=special_categorical_crossentropy, optimizer=optimizer)

        model.compile(loss=special_categorical_crossentropy, optimizer=optimizer)
        
        self.model = model



    #
    # The API for the distributed training part.
    #
    # The parameters are passed in a serialized form
    # (and must be in a consistent scheme used by all master/workers).

    def get_parameters(self):
        # Used by the master only.
        return np.random.rand(4,5).astype(np.float32)

    def set_parameters(self, parameters):
        # Used by the workers only.
        pass

    def train(self, batch_name, lower_index, upper_index, suffix, weight, total_weights, nbr_of_weights):
        # Perform one contribution with that particular batch.
        # Assumes that the model knows everything about how to get the data
        # and to fetch the right subset specified by the [lower_index, upper_index).
        #
        # Can ignore `batch_name` and `suffix`. They are useless arguments.
        # But they might be used to switch between some alternatives and return
        # other quantities.
        #

        # see sample_weight in
        #    train_on_batch(self, X, y, accuracy=False, class_weight=None, sample_weight=None)
        #    https://github.com/fchollet/keras/blob/master/keras/models.py
        self.model.train_on_batch(X, y, sample_weight=weight / (total_weights * nbr_of_weights))

        print "Model was called to train on %s." % batch_name



model.fit(X_train, Y_train, batch_size=32, nb_epoch=1)



        model.fit(X_train, y_train, nb_epoch=20, batch_size=16)
        score = model.evaluate(X_test, y_test, batch_size=16)
