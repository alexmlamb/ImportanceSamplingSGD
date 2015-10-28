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
    train_X = numpy.asarray(train_object["X"], dtype = 'uint8')
    extra_X = numpy.asarray(extra_object["X"], dtype = 'uint8')
    test_X = numpy.asarray(test_object["X"], dtype = 'uint8')

    train_Y = numpy.asarray(train_object["y"], dtype = 'uint8')
    extra_Y = numpy.asarray(extra_object["y"], dtype = 'uint8')
    test_Y = numpy.asarray(test_object["y"], dtype = 'uint8')

    print "converted to numpy arrays"

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

    print "axes swapped"

    train_X = numpy.vstack((train_X, extra_X))
    train_Y = numpy.vstack((train_Y, extra_Y))

    print "vstacked"

    # BUG : YOU CAN'T SHUFFLE THE DATASETS AND EXPECT EVERYONE TO AGREE ON THE INDICES.
    #       This would require setting the seed in advance.

    print "train_X.shape"
    print train_X.shape

    train_indices = numpy.random.choice(train_X.shape[0], int(train_X.shape[0] * (1.0 - config["fraction_validation"])), replace = False)
    valid_indices = numpy.setdiff1d(range(0,train_X.shape[0]), train_indices)

    print "train_X.shape"
    print train_X.shape

    print "indices collected"

    valid_X = train_X[valid_indices]
    valid_Y = train_Y[valid_indices]

    train_X = train_X[train_indices]
    train_Y = train_Y[train_indices]

    print "train_X.shape"
    print train_X.shape

    print "indices indexed"

    assert not (config["load_svhn_normalization_from_file"] and config["save_svhn_normalization_to_file"])

    #get mean and std for each filter and each pixel.
    if not config["load_svhn_normalization_from_file"]:

        import safe_mean_std_var

        x_mean, x_std, _ = safe_mean_std_var.mean_std_var(train_X, axis=0)
        #x_mean = train_X.mean(axis = (0))
        #x_std = train_X.std(axis = (0))

        if config["save_svhn_normalization_to_file"]:
            cPickle.dump({"mean" : x_mean, "std" : x_std}, open(config["svhn_normalization_value_file"], "w"), protocol = cPickle.HIGHEST_PROTOCOL)
    else:
        svhn_normalization_values = cPickle.load(open(config["svhn_normalization_value_file"]))
        x_mean = svhn_normalization_values["mean"]
        x_std = svhn_normalization_values["std"]

    print "computed mean and var"

    print "Training Set", train_X.shape, train_Y.shape
    print "Validation Set", valid_X.shape, valid_Y.shape
    print "Test Set", test_X.shape, test_Y.shape

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
    new_X = (X - mean) / std
    new_X = numpy.reshape(new_X, (new_X.shape[0], -1)).astype('float32')

    return new_X

def load_data(config):
    if config["dataset"] == "svhn":
        return load_data_svhn(config)
    elif config["dataset"] == "mnist":
        return load_data_mnist(config)
    else:
        raise Exception("Dataset must be either svhn or mnist")
