import numpy as np
import os

def get_model_config():

    model_config = {}

    #Importance sampling or vanilla sgd.
    model_config["importance_algorithm"] = "isgd"
    #model_config["importance_algorithm"] = "sgd"

    #Momentum rate, where 0.0 corresponds to not using momentum
    model_config["momentum_rate"] = 0.9

    #The learning rate to use on the gradient averaged over a minibatch
    model_config["learning_rate"] = 0.1

    #model_config["dataset"] = "mnist"
    model_config["dataset"] = "svhn"

    if model_config["dataset"] == "mnist":
        model_config["num_input"] = 784
    elif model_config["dataset"] == "svhn":
        model_config["num_input"] = 3072

    # Pick one, depending where you run this.
    # This could be done differently too by looking at fuelrc
    # or at the hostname.
    #data_root = "/data/lisatmp4/lambalex"
    #data_root = "/Users/gyomalin/Documents/fuel_data"
    data_root = "/rap/jvb-000-aa/data/alaingui"

    model_config["mnist_file"] = os.path.join(data_root, "mnist/mnist.pkl.gz")
    model_config["svhn_file_train"] = os.path.join(data_root, "svhn/train_32x32.mat")
    model_config["svhn_file_extra"] = os.path.join(data_root, "svhn/extra_32x32.mat")
    model_config["svhn_file_test"] = os.path.join(data_root, "svhn/test_32x32.mat")

    model_config["load_svhn_normalization_from_file"] = True
    model_config["save_svhn_normalization_to_file"] = False
    model_config["svhn_normalization_value_file"] = os.path.join(data_root, "svhn/svhn_normalization_values.pkl")

    model_config["hidden_sizes"] = [2048, 2048,2048,2048]

    # Note from Guillaume : I'm not fond at all of using seeds in a context
    # where we don't need them.
    model_config["seed"] = 9999494

    model_config["learning_rate"] = 0.01

    #Weights are initialized to N(0,1) * initial_weight_size
    model_config["initial_weight_size"] = 0.01

    #Hold this fraction of the instances in the validation dataset
    model_config["fraction_validation"] = 0.05

    return model_config


def get_database_config():

    # Some of those values are placeholder.
    # Need to update the (Ntrain, Nvalid, Ntest) to the actual values for SVHN.

    (Ntrain, Nvalid, Ntest) = (574168, 30220, 26032)
    # Training Set (574168, 32, 3, 32) (574168, 1)
    # Validation Set (30220, 32, 3, 32) (30220, 1)
    # Test Set (26032, 32, 3, 32) (26032, 1)
    # svhn data loaded...

    # This was first set to 0.0.
    # Then we discussed how using a small epsilon instead
    # would make all the data potentially a candidate.
    # Now we've set to this be NaN because that's
    # a way to indicate that they've never been sampled,
    # and the master can then wait for all of them
    # to be updated at least once before starting the training.
    default_importance_weight = np.NaN
    want_master_to_wait_for_all_importance_weights_to_be_present = True

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
                serialized_parameters_format=serialized_parameters_format,
                default_importance_weight=default_importance_weight,
                want_master_to_wait_for_all_importance_weights_to_be_present=want_master_to_wait_for_all_importance_weights_to_be_present)

def get_helios_config():
    # Optional.

    # Let's say we force quit after 6 hours.
    # We're going to get stopped by the walltime of the helios launch script in any case.
    force_quit_after_total_duration = 6*3600

    experiment_id = "001"

    experiment_output_root = "/rap/jvb-000-aa/data/alaingui/experiments_ISGD/"
    experiment_output_dir = os.path.join(experiment_output_root, experiment_id)

    # Read ${MOAB_JOBARRAYINDEX} from environment.
    jobid = 0

    return {'force_quit_after_total_duration' : force_quit_after_total_duration,
            'experiment_output_root' : experiment_output_root,
            'jobid' : jobid}
