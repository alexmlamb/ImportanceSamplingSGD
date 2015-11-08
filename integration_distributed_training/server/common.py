
import time
import redis
import numpy as np

def get_rsconn_with_timeout(server_ip, server_port, server_password=None,
                            timeout=60, wait_for_parameters_to_be_present=True):

    assert server_ip
    assert server_port
    #assert server_password

    initial_conn_timestamp = time.time()
    success = False
    while time.time() - initial_conn_timestamp < timeout:

        try:
            rsconn = redis.StrictRedis(host=server_ip, port=server_port, password=server_password)
            print "Connected to local server."
            success = True
            break
        except:
            time.sleep(5)
            print "Failed to connect to local server. Will retry in 5s."

    if not success:
        print "Quitting."
        quit()

    print "Pinging local server : %s" % (rsconn.ping(),)


    initial_conn_timestamp = time.time()
    success = False
    while time.time() - initial_conn_timestamp < timeout:
        if rsconn.get("initialization_is_done") == "True":
            print "Experiment is properly initialized. We start now."
            success = True
            break
        else:
            print "Experiment has not been initialized completely yet. Will retry in 5s."
            time.sleep(5)

    if not success:
        print "Quitting."
        quit()

    if wait_for_parameters_to_be_present:

        initial_conn_timestamp = time.time()
        success = False
        while time.time() - initial_conn_timestamp < timeout:
            if 0 < len(rsconn.get("parameters:current")):
                print "The current parameters are found on the server. We start now."
                success = True
                break
            else:
                print "The current parameters are not yet on the server. Will retry in 5s."
                time.sleep(5)

        if not success:
            print "Quitting."
            quit()

    return rsconn


# Note that this is NOT the "variance" measurement that is relevant in SGD.
# It is merely the variance on the measurement throughout the dataset.
# Be careful about not confusing those two.

def get_mean_variance_measurement_on_database(rsconn, segment, measurement):
    # The elements come out in any order,
    # so there wasn't much reason to return the array
    # instead of just the mean and variance.

    L = []
    for value_str in rsconn.hvals("H_%s_minibatch_%s" % (segment, measurement)):
        value = np.fromstring(value_str, dtype=np.float32)
        L.append(value)

    if len(L) == 0:
        return (np.nan, np.nan, 0, 0.0)

    A = np.hstack(L)
    assert len(A.shape) == 1

    # weed out anything np.nan
    I = np.isfinite(A)
    if I.sum() == 0:
        return (np.nan, np.nan, 0, 0.0)

    usable_A = A[I]
    ratio_of_usable_values = usable_A.shape[0] * 1.0 / A.shape[0]
    return usable_A.mean(), usable_A.var(), usable_A.shape[0], ratio_of_usable_values




def wait_until_all_measurements_are_updated_by_workers(rsconn, segment, measurement):

    #segment = "train"
    #measurement = "importance_weight"

    timestamp_start = time.time()

    db_list_name = "L_workers_%s_minibatch_indices_ALL" % segment
    db_hash_name = "H_%s_minibatch_%s_measurement_last_update_timestamp" % (segment, measurement)
    nbr_minibatches = rsconn.llen(db_list_name)
    assert 0 < nbr_minibatches

    while True:
        successfully_updated = np.zeros((nbr_minibatches,), dtype=np.bool)
        for i in range(nbr_minibatches):
            minibatch_indices_str = rsconn.lindex(db_list_name, i)
            timestamp_str = rsconn.hget(db_hash_name, minibatch_indices_str)
            assert timestamp_str is not None and 0 < len(timestamp_str)
            timestamp = float(timestamp_str)
            if timestamp_start < timestamp:
                successfully_updated[i] = True
            else:
                # yeah, this isn't necessary, but it's just explicit
                successfully_updated[i] = False

        if np.all(successfully_updated):
            print "Successfully waited for all the workers to update the measurement %s for segment %s." % (measurement, segment)
            return
        else:
            print "Only %0.3f updated for the measurement %s for segment %s." % (successfully_updated.mean(), measurement, segment)
            time.sleep(5)




def get_trace_covariance_information(rsconn, segment, minimum_ratio_present=0.5):
    #segment = "train"
    # Does not return values unless there are at least `minimum_ratio_present`
    # of the importance weights that are np.isfinite.

    db_list_name_L_minibatch_indices_str = "L_workers_%s_minibatch_indices_ALL" % segment
    nbr_minibatches = rsconn.llen(db_list_name_L_minibatch_indices_str)

    L_previous_individual_importance_weight = []
    L_individual_importance_weight = []
    L_individual_gradient_square_norm = []
    L_minibatch_gradient_mean_square_norm = []
    for i in range(nbr_minibatches):
        minibatch_indices_str = rsconn.lindex(db_list_name_L_minibatch_indices_str, i)

        previous_individual_importance_weight_str = rsconn.hget("H_%s_minibatch_%s" % (segment, "previous_individual_importance_weight"), minibatch_indices_str)
        individual_importance_weight_str = rsconn.hget("H_%s_minibatch_%s" % (segment, "individual_importance_weight"), minibatch_indices_str)
        individual_gradient_square_norm_str = rsconn.hget("H_%s_minibatch_%s" % (segment, "individual_gradient_square_norm"), minibatch_indices_str)
        minibatch_gradient_mean_square_norm_str = rsconn.hget("H_%s_minibatch_%s" % (segment, "minibatch_gradient_mean_square_norm"), minibatch_indices_str)

        # Some un-necessary controls required to avoid gotos,
        # and because the language doesn't have the break(2).
        want_skip_this_minibatch = False
        for e in [previous_individual_importance_weight_str, individual_importance_weight_str, individual_gradient_square_norm_str, minibatch_gradient_mean_square_norm_str]:
            if e is None or len(e) == 0:
                want_skip_this_minibatch = True
        if want_skip_this_minibatch:
            continue

        previous_individual_importance_weight = np.fromstring(previous_individual_importance_weight_str, dtype=np.float32).astype(np.float64)
        individual_importance_weight          = np.fromstring(individual_importance_weight_str, dtype=np.float32).astype(np.float64)
        individual_gradient_square_norm       = np.fromstring(individual_gradient_square_norm_str, dtype=np.float32).astype(np.float64)
        minibatch_gradient_mean_square_norm   = np.fromstring(minibatch_gradient_mean_square_norm_str, dtype=np.float32).astype(np.float64)

        if not (np.all(np.isfinite(previous_individual_importance_weight)) and
                np.all(np.isfinite(individual_importance_weight)) and
                np.all(np.isfinite(individual_gradient_square_norm)) and
                np.all(np.isfinite(minibatch_gradient_mean_square_norm))):
            continue


        L_previous_individual_importance_weight.append( previous_individual_importance_weight )
        L_individual_importance_weight.append( individual_importance_weight )
        L_individual_gradient_square_norm.append( individual_gradient_square_norm )
        L_minibatch_gradient_mean_square_norm.append( minibatch_gradient_mean_square_norm )

    nbr_minibatches_used = len(L_previous_individual_importance_weight)
    r = nbr_minibatches_used * 1.0 / nbr_minibatches
    if  r <= minimum_ratio_present - 1e-8:
        print "Called get_trace_covariance for segment %s, but we simply could not retrieve more than ratio %f from the database." % (segment, r)
        #import pdb; pdb.set_trace()
        return (None, None, None, None, nbr_minibatches_used, nbr_minibatches)

    A_previous_individual_importance_weight = np.concatenate(L_previous_individual_importance_weight, axis=0)
    A_individual_importance_weight = np.concatenate(L_individual_importance_weight, axis=0)
    A_individual_gradient_square_norm = np.concatenate(L_individual_gradient_square_norm, axis=0)
    # this one is different because it's made out of one element per minibatch
    A_minibatch_gradient_mean_square_norm = np.concatenate(L_minibatch_gradient_mean_square_norm, axis=0)

    print A_previous_individual_importance_weight.shape
    print A_individual_importance_weight.shape

    if np.any(A_minibatch_gradient_mean_square_norm < 0.0) or np.any(np.logical_not(np.isfinite(A_minibatch_gradient_mean_square_norm))):
        import pdb; pdb.set_trace()

    approximated_mu_norm_square = np.sqrt(A_minibatch_gradient_mean_square_norm).mean()**2
    if approximated_mu_norm_square is None:
        import pdb; pdb.set_trace()

    USGD_main_term = A_individual_gradient_square_norm.mean()
    ISGD_main_term = np.sqrt(A_individual_gradient_square_norm).mean()**2

    staleISGD_main_term_1 = np.sqrt(A_previous_individual_importance_weight).mean()
    staleISGD_main_term_2 = (A_individual_gradient_square_norm / np.sqrt(A_previous_individual_importance_weight)).mean()
    staleISGD_main_term = staleISGD_main_term_1 * staleISGD_main_term_2

    return (USGD_main_term, staleISGD_main_term, ISGD_main_term, approximated_mu_norm_square, nbr_minibatches_used, nbr_minibatches)
