

def get_model_config():

    config = {}

    #Importance sampling or vanilla sgd.
    config["turn_off_importance_sampling"] = False

    #Momentum rate, where 0.0 corresponds to not using momentum
    config["momentum_rate"] = 0.95

    #The learning rate to use on the gradient averaged over a minibatch

    config["learning_rate"] = 0.01

    #config["dataset"] = "mnist"
    #config["dataset"] = "svhn"
    config["dataset"] = "kaldi-i84"

    if config["dataset"] == "mnist":
        config["num_input"] = 784
    elif config["dataset"] == "svhn":
        (Ntrain, Nvalid, Ntest) = (574168, 30220, 26032)
        config["num_input"] = 3072
        config["num_output"] = 10
        config["normalize_data"] = True
    elif config["dataset"] == "kaldi-i84":
        (Ntrain, Nvalid, Ntest) = (5436921, 389077, 253204)
        config["num_input"] = 861
        config["num_output"] = 3472
        config["normalize_data"] = False

    config["mnist_file"] = "/u/lambalex/data/mnist/mnist.pkl.gz"
    config["svhn_file_train"] = "/u/lambalex/data/svhn/train_32x32.mat"
    config["svhn_file_extra"] = "/u/lambalex/data/svhn/extra_32x32.mat"
    config["svhn_file_test"] = "/u/lambalex/data/svhn/test_32x32.mat"

    config["save_svhn_normalization_to_file"] = False
    config["load_svhn_normalization_from_file"] = True

    config["kaldi-i84_file_train"] = "/u/lambalex/data/kaldi/i84_train.gz"
    config["kaldi-i84_file_valid"] = "/u/lambalex/data/kaldi/i84_valid.gz"
    config["kaldi-i84_file_test"] = "/u/lambalex/data/kaldi/i84_test.gz"

    config["svhn_normalization_value_file"] = "/u/lambalex/data/svhn/svhn_normalization_values.pkl"

    config["hidden_sizes"] = [2048, 2048, 2048, 2048]

    config["seed"] = 9999494

    #Weights are initialized to N(0,1) * initial_weight_size
    config["initial_weight_size"] = 0.01

    #Hold this fraction of the instances in the validation dataset
    config["fraction_validation"] = 0.05

    #I actually think that this should be another config
    config["master_routine"] = ["sync_params"] + ["refresh_importance_weights"] + (["process_minibatch"] * 10)
    config["worker_routine"] = ["sync_params"] + (["process_minibatch"] * 10)

    config['Ntrain'] = Ntrain
    config['Nvalid'] = Nvalid
    config['Ntest'] = Ntest

    assert config['Ntrain'] is not None and 0 < config['Ntrain']
    assert config['Nvalid'] is not None
    assert config['Ntest'] is not None

    return config


def get_database_config():

    #Log files will be put into this folder.  If "logging_folder" is set to none, then nothing will be logged to the file.  

    logging_folder = "/u/lambalex/DeepLearning/ImportanceSampling/logs/"

    importance_weight_additive_constant = 1.0

    master_usable_importance_weights_threshold_to_ISGD = 0.02

    serialized_parameters_format ="opaque_string"

    connection_setup_timeout = 10*60

    do_background_save = False

    # These two values don't have to be the same.
    # It might be possible that the master runs on a GPU
    # and the workers run on CPUs just to try stuff out.
    workers_minibatch_size = 4096
    master_minibatch_size = 256

    #The master will only consider importance weights which were updated this number of seconds ago.  
    staleness_threshold_seconds = 30.0

    # This is not really being used anywhere.
    # We should consider deleting it after making sure that it
    # indeed is not being used, but then we could argue that it
    # would be a good idea to use that name to automatically determine
    # the values of (Ntrain, Nvalid, Ntest).
    dataset_name='svhn'

    #L_measurements=["individual_importance_weight", "gradient_square_norm", "loss", "accuracy", "minibatch_gradient_mean_square_norm", "individual_gradient_square_norm"]

    L_measurements=["individual_importance_weight", "individual_gradient_square_norm", "individual_loss", "individual_accuracy", "minibatch_gradient_mean_square_norm"]

    minimum_number_of_minibatch_processed_before_parameter_update = 10
    nbr_batch_processed_per_public_parameter_update = 10

    # Optional field : 'server_scratch_path'

    #
    # The rest of this code is just checks and quantities generated automatically.

    doTrain = True
    doValidation = True
    doTest = True

    L_segments = []

    if doTrain:
        L_segments.append("train")
    if doValidation:
        L_segments.append("valid")
    if doTest:
        L_segments.append("test")


    staleness_threshold_num_minibatches_master_processed = 60

    assert workers_minibatch_size is not None and 0 < workers_minibatch_size
    assert master_minibatch_size is not None and 0 < master_minibatch_size
    assert dataset_name is not None

    assert serialized_parameters_format in ["opaque_string", "ndarray_float32_tostring"]

    return dict(workers_minibatch_size=workers_minibatch_size,
                master_minibatch_size=master_minibatch_size,
                dataset_name=dataset_name,
                L_measurements=L_measurements,
                L_segments=L_segments,
                want_only_indices_for_master=True,
                want_exclude_partial_minibatch=True,
                serialized_parameters_format=serialized_parameters_format,
                staleness_threshold_seconds=staleness_threshold_seconds,
                minimum_number_of_minibatch_processed_before_parameter_update=minimum_number_of_minibatch_processed_before_parameter_update,
                nbr_batch_processed_per_public_parameter_update=nbr_batch_processed_per_public_parameter_update,
                master_usable_importance_weights_threshold_to_ISGD=master_usable_importance_weights_threshold_to_ISGD,
                logging_folder = logging_folder,
                importance_weight_additive_constant=importance_weight_additive_constant,
                connection_setup_timeout=connection_setup_timeout,
                staleness_threshold_num_minibatches_master_processed=staleness_threshold_num_minibatches_master_processed,
                do_background_save=do_background_save)

def get_helios_config():
    # Optional.
    return {}
