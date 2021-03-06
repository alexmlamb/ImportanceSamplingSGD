

def get_model_config():

    config = {}

    #Importance sampling or vanilla sgd.
    config["turn_off_importance_sampling"] = False

    #Momentum rate, where 0.0 corresponds to not using momentum
    config["momentum_rate"] = 0.9

    #The learning rate to use on the gradient averaged over a minibatch

    config["learning_rate"] = 0.01

    #config["dataset"] = "mnist"
    config["dataset"] = "svhn"

    if config["dataset"] == "mnist":
        config["num_input"] = 784
    elif config["dataset"] == "svhn":
        config["num_input"] = 3072
    data_root = "/Users/chinna/Downloads/"#"/u/lambalex/data/"
    config["mnist_file"] = data_root + "mnist/mnist.pkl.gz"
    config["svhn_file_train"] = data_root + "svhn/train_32x32.mat"
    config["svhn_file_extra"] = data_root + "svhn/extra_32x32.mat"
    config["svhn_file_test"] =  data_root + "svhn/test_32x32.mat"

    config["save_svhn_normalization_to_file"] = False
    config["load_svhn_normalization_from_file"] = True

    config["svhn_normalization_value_file"] = "/u/lambalex/data/svhn/svhn_normalization_values.pkl"

    config["hidden_sizes"] = [2048, 2048, 2048, 2048]

    config["seed"] = 9999494


    #Weights are initialized to N(0,1) * initial_weight_size
    config["initial_weight_size"] = 0.01

    #Hold this fraction of the instances in the validation dataset
    config["fraction_validation"] = 0.05

    config["importance_weight_additive_constant"] = 10.0

    return config


def get_database_config():

    #Log files will be put into this folder.  If "logging_folder" is set to none, then nothing will be logged to the file.

    logging_folder = "/Users/chinna/ImportanceSamplingSGD_Alex/model_protoype/exp_chinna/logging"#"/data/lisatmp3/chinna/logging/"

    # Some of those values are placeholder.
    # Need to update the (Ntrain, Nvalid, Ntest) to the actual values for SVHN.

    (Ntrain, Nvalid, Ntest) = (574168, 30220, 26032)
    # Training Set (574168, 32, 3, 32) (574168, 1)
    # Validation Set (30220, 32, 3, 32) (30220, 1)
    # Test Set (26032, 32, 3, 32) (26032, 1)
    # svhn data loaded...

    master_usable_importance_weights_threshold_to_ISGD = 0.1

    serialized_parameters_format ="opaque_string"

    # These two values don't have to be the same.
    # It might be possible that the master runs on a GPU
    # and the workers run on CPUs just to try stuff out.
    workers_minibatch_size = 128
    master_minibatch_size = 128

    #The master will only consider importance weights which were updated this number of seconds ago.
    staleness_threshold = 20.0

    # This is not really being used anywhere.
    # We should consider deleting it after making sure that it
    # indeed is not being used, but then we could argue that it
    # would be a good idea to use that name to automatically determine
    # the values of (Ntrain, Nvalid, Ntest).
    dataset_name='svhn'

    L_measurements=["individual_importance_weight", "gradient_square_norm", "loss", "accuracy", "minibatch_gradient_mean_square_norm", "individual_gradient_square_norm"]

    minimum_number_of_minibatch_processed_before_parameter_update = 10
    nbr_batch_processed_per_public_parameter_update = 10

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
                serialized_parameters_format=serialized_parameters_format,
                staleness_threshold=staleness_threshold,
                minimum_number_of_minibatch_processed_before_parameter_update=minimum_number_of_minibatch_processed_before_parameter_update,
                nbr_batch_processed_per_public_parameter_update=nbr_batch_processed_per_public_parameter_update,
                master_usable_importance_weights_threshold_to_ISGD=master_usable_importance_weights_threshold_to_ISGD,
                logging_folder = logging_folder)

def get_helios_config():
    # Optional.
    return {}
