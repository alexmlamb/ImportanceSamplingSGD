
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




def get_trace_covariance_information(rsconn, segment):
    #segment = "train"

    # In this method, there care be race conditions that lead
    # to discrepancies in measurements present.
    # For example, we should expect that `individual_importance_weight`
    # be present if and only if `individual_gradient_square_norm` is present,
    # but that can fail to be true because of the multiprocess environment.
    # In those cases, we'll try to be as conservative as possible.

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

        for e in [previous_individual_importance_weight_str, individual_importance_weight_str, individual_gradient_square_norm_str, minibatch_gradient_mean_square_norm_str]:
            if e is None or len(e) == 0:
                print "Error. The database should contain default values for all the measurements. Use np.nan to indicate a yet-undefined value."


        L_minibatch_gradient_mean_square_norm.append(   np.fromstring(minibatch_gradient_mean_square_norm_str, dtype=np.float32).astype(np.float64) )
        L_previous_individual_importance_weight.append( np.fromstring(previous_individual_importance_weight_str, dtype=np.float32).astype(np.float64) )
        L_individual_importance_weight.append(          np.fromstring(individual_importance_weight_str, dtype=np.float32).astype(np.float64) )
        L_individual_gradient_square_norm.append(       np.fromstring(individual_gradient_square_norm_str, dtype=np.float32).astype(np.float64) )

    previous_individual_importance_weight = np.concatenate(L_previous_individual_importance_weight, axis=0)
    individual_importance_weight = np.concatenate(L_individual_importance_weight, axis=0)
    individual_gradient_square_norm = np.concatenate(L_individual_gradient_square_norm, axis=0)
    # this one is different because it's made out of one element per minibatch
    minibatch_gradient_mean_square_norm = np.concatenate(L_minibatch_gradient_mean_square_norm, axis=0)

    # Note that `minibatch_gradient_mean_square_norm` is an array
    # with a completely different shape than the other ones because it
    # has one value per worker minibatch.
    H = np.isfinite(minibatch_gradient_mean_square_norm)
    I = np.isfinite(individual_importance_weight) * np.isfinite(individual_gradient_square_norm)
    J = I * np.isfinite(previous_individual_importance_weight)

    # Conceptually, we would expect that I.mean() == H.mean()
    # but this can fail to hold due to race conditions.
    ratio_of_usable_indices_for_USGD_and_ISGD = np.min([I.mean(), H.mean()])
    ratio_of_usable_indices_for_ISGDstale = J.mean()

    assert ratio_of_usable_indices_for_ISGDstale <= ratio_of_usable_indices_for_USGD_and_ISGD

    if 0.0 < ratio_of_usable_indices_for_USGD_and_ISGD:
        approximated_mu_norm_square = np.sqrt(minibatch_gradient_mean_square_norm[H]).mean()**2
        USGD_main_term = individual_gradient_square_norm[I].mean()
        ISGD_main_term = np.sqrt(individual_gradient_square_norm[I]).mean()**2
    else:
        approximated_mu_norm_square = None
        USGD_main_term = None
        ISGD_main_term = None

    if 0.0 < ratio_of_usable_indices_for_ISGDstale:
        J = J * (1e-16 < individual_gradient_square_norm)
        staleISGD_main_term_1 = np.sqrt(previous_individual_importance_weight[J]).mean()
        staleISGD_main_term_2 = (individual_gradient_square_norm[J] / np.sqrt(previous_individual_importance_weight[J])).mean()
        staleISGD_main_term = staleISGD_main_term_1 * staleISGD_main_term_2
    else:
        staleISGD_main_term = None

    return (USGD_main_term, staleISGD_main_term, ISGD_main_term, approximated_mu_norm_square, ratio_of_usable_indices_for_USGD_and_ISGD, ratio_of_usable_indices_for_ISGDstale, nbr_minibatches)
