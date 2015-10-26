
def get_model_config():

    config = {}

    #Importance sampling or vanilla sgd.
    config["importance_algorithm"] = "isgd"
    #config["importance_algorithm"] = "sgd"

    #Momentum rate, where 0.0 corresponds to not using momentum
    config["momentum_rate"] = 0.9

    #The learning rate to use on the gradient averaged over a minibatch
    config["learning_rate"] = 0.1

    #config["dataset"] = "mnist"
    config["dataset"] = "svhn"

    if config["dataset"] == "mnist":
        config["num_input"] = 784
    elif config["dataset"] == "svhn":
        config["num_input"] = 3072

    config["mnist_file"] = "/data/lisatmp4/lambalex/mnist/mnist.pkl.gz"
    config["svhn_file_train"] = "/data/lisatmp4/lambalex/svhn/train_32x32.mat"
    config["svhn_file_extra"] = "/data/lisatmp4/lambalex/svhn/extra_32x32.mat"
    config["svhn_file_test"] = "/data/lisatmp4/lambalex/svhn/test_32x32.mat"

    config["hidden_sizes"] = [2048, 2048,2048,2048]

    config["seed"] = 9999494

    config["learning_rate"] = 0.01

    #Weights are initialized to N(0,1) * initial_weight_size
    config["initial_weight_size"] = 0.01

    #Hold this fraction of the instances in the validation dataset
    config["fraction_validation"] = 0.05

    return config


def get_database_config():

    # Some of those values are placeholder.
    # Need to update the (Ntrain, Nvalid, Ntest) to the actual values for SVHN.

    (Ntrain, Nvalid, Ntest) = (10000, 0, 1000)

    serialized_parameters_format ="opaque_string"

    # These two values don't have to be the same.
    # It might be possible that the master runs on a GPU
    # and the workers run on CPUs just to try stuff out.
    workers_minibatch_size = 128
    master_minibatch_size = 128

    # This is not really being used anywhere.
    # We should consider deleting it after making sure that it
    # indeed is not being used, but then we could argue that it
    # would be a good idea to use that name to automatically determine
    # the values of (Ntrain, Nvalid, Ntest).
    dataset_name='svhn'

    L_measurements=["importance_weight", "gradient_square_norm", "loss", "accuracy"]


    # Optional field : 'server_scratch_path'

    #
    # The rest of this code is just checks and quantities generated automatically.
    #

    L_segments = []
    assert 0 < Ntrain
    if Ntrain != 0:
        L_segments.append("train")
    if Nvalid != 0:
        L_segments.append("valid")
    if Ntest != 0:
        L_segments.append("test")


    assert workers_minibatch_size is not None and 0 < workers_minibatch_size
    assert master_minibatch_size is not None and 0 < master_minibatch_size
    assert dataset_name is not None
    assert Ntrain is not None and 0 < Ntrain
    assert Nvalid is not None
    assert Ntest is not None

    assert serialized_parameters_format in ["opaque_string", "ndarray_float32_tostring"]

    return dict(workers_minibatch_size=workers_minibatch_size,
                master_minibatch_size=master_minibatch_size,
                dataset_name=dataset_name,
                Ntrain=Ntrain,
                Nvalid=Nvalid,
                Ntest=Ntest,
                L_measurements=L_measurements,
                L_segments=L_segments,
                want_only_indices_for_master=True,
                want_exclude_partial_minibatch=True,
                serialized_parameters_format=serialized_parameters_format)

def get_helios_config():
    # Optional.
    return {}
