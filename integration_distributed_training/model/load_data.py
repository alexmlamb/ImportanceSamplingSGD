import numpy
import gzip
import cPickle

#Returns list of tuples containing training, validation, and test instances.
def load_data_svhn(config):

    numpy.random.seed(config["seed"])

    import scipy.io as sio
    train_file = config["svhn_file_train"]
    extra_file = config["svhn_file_extra"]
    test_file = config["svhn_file_test"]

    train_object = sio.loadmat(train_file)
    extra_object = sio.loadmat(extra_file)
    test_object = sio.loadmat(test_file)

    print "objects loaded"

    # Note from Guillaume : This is NOT what we agreed on.
    # This is loading the data and converting it into an 8 GB array.
    # We had agreed to store this as uint8 temporarily and do the
    # conversion on a mini-batch basis.
    train_X = numpy.asarray(train_object["X"], dtype = 'float32')
    extra_X = numpy.asarray(extra_object["X"], dtype = 'float32')
    test_X = numpy.asarray(test_object["X"], dtype = 'float32')

    train_Y = numpy.asarray(train_object["y"], dtype = 'int8')
    extra_Y = numpy.asarray(extra_object["y"], dtype = 'int8')
    test_Y = numpy.asarray(test_object["y"], dtype = 'int8')


    del train_object
    del extra_object
    del test_object

    #By default SVHN labels are from 1 to 10.
    #This shifts them to be between 0 and 9.
    train_Y -= 1
    extra_Y -= 1
    test_Y -= 1

    assert train_Y.min() == 0
    assert train_Y.max() == 9

    train_X = numpy.swapaxes(train_X, 0,3)

    extra_X = numpy.swapaxes(extra_X, 0,3)

    test_X = numpy.swapaxes(test_X, 0,3)

    train_X = numpy.vstack((train_X, extra_X))
    train_Y = numpy.vstack((train_Y, extra_Y))

    train_indices = numpy.random.choice(train_X.shape[0], int(train_X.shape[0] * (1.0 - config["fraction_validation"])), replace = False)
    valid_indices = numpy.setdiff1d(range(0,train_X.shape[0]), train_indices)

    valid_X = train_X[valid_indices]
    valid_Y = train_Y[valid_indices]

    train_X = train_X[train_indices]
    train_Y = train_Y[train_indices]

    #get mean and std for each filter and each pixel.
    x_mean = train_X.mean(axis = (0))
    x_std = train_X.std(axis = (0))


    train_X = (train_X - x_mean) / x_std
    valid_X = (valid_X - x_mean) / x_std
    test_X = (test_X - x_mean) / x_std

    train_X = numpy.reshape(train_X, (train_X.shape[0], -1))
    valid_X = numpy.reshape(valid_X, (valid_X.shape[0], -1))
    test_X  = numpy.reshape(test_X,  (test_X.shape[0], -1))

    print "Training Set", train_X.shape, train_Y.shape
    print "Validation Set", valid_X.shape, valid_Y.shape
    print "Test Set", test_X.shape, test_Y.shape

    # Note from Guillaume : You need to keep around the mean/std here to be
    # able to divide them when on a minibatch.
    return {"train": (train_X, train_Y.flatten()), "valid" : (valid_X, valid_Y.flatten()), "test" : (test_X, test_Y.flatten()), "mean" : x_mean, "std" : x_std}


def load_data_mnist(config):
    dataset = config["mnist_file"]

    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    test_set_x, test_set_y = test_set
    valid_set_x, valid_set_y = valid_set
    train_set_x, train_set_y = train_set

    rval = {"train" : (train_set_x, train_set_y), "valid" : (valid_set_x, valid_set_y), "test" : (test_set_x, test_set_y)}

    #[(train_set_x, train_set_y), (valid_set_x, valid_set_y),
           # (test_set_x, test_set_y)]

    return rval

def normalizeMatrix(X, mean, std):
    #new_X = (X - mean) / std
    #new_X = numpy.reshape(X, (X.shape[0], -1)).astype('float32')

    new_X = X

    return new_X

def load_data(config):
    if config["dataset"] == "svhn":
        return load_data_svhn(config)
    elif config["dataset"] == "mnist":
        return load_data_mnist(config)
    else:
        raise Exception("Dataset must be either svhn or mnist")



