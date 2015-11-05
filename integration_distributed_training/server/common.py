
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
