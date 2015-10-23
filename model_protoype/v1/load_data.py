import numpy

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

    train_X = numpy.reshape(train_X, (train_X.shape[0],train_X.shape[1] * train_X.shape[2] * train_X.shape[3]))
    valid_X = numpy.reshape(valid_X, (valid_X.shape[0],valid_X.shape[1] * valid_X.shape[2] * valid_X.shape[3]))
    test_X = numpy.reshape(test_X, (test_X.shape[0],test_X.shape[1] * test_X.shape[2] * test_X.shape[3]))

    print "Training Set", train_X.shape, train_Y.shape
    print "Validation Set", valid_X.shape, valid_Y.shape
    print "Test Set", test_X.shape, test_Y.shape

    return [train_X, train_Y.flatten().tolist(), valid_X, valid_Y.flatten().tolist(), test_X, test_Y.flatten().tolist()]
    #[(train_X, train_Y.flatten().tolist()), (valid_X, valid_Y.flatten().tolist()), (test_X, test_Y.flatten().tolist())]

    

def load_data_mnist(config):
    dataset = config["mnist_file"]

    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    test_set_x, test_set_y = test_set
    valid_set_x, valid_set_y = valid_set
    train_set_x, train_set_y = train_set

    rval = [train_set_x, train_set_y, valid_set_x, valid_set_y, test_set_x, test_set_y]
    #[(train_set_x, train_set_y), (valid_set_x, valid_set_y),
           # (test_set_x, test_set_y)]

    return rval



def load_data(config):
    if config["dataset"] == "svhn":
        return load_data_svhn(config)
    elif config["dataset"] == "mnist":
        return load_data_mnist(config)
    else:
        raise Exception("Dataset must be either svhn or mnist")






if __name__ == "__main__":

    from config import get_config

    config = get_config()

    print "config loaded"

    load_data(config)






