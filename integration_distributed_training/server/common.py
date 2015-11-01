
import time
import redis

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



def get_mean_variance_measurement_on_database(rsconn, segment, measurement):
    # The elements come out in any order,
    # so there wasn't much reason to return the array
    # instead of just the mean and variance.

    L = []
    for value_str in rsconn.hvals("H_%s_minibatch_%s" % (segment, measurement)):
        value = np.fromstring(value_str, dtype=np.float32)
        L.append(value)

    A = np.vstack(L)
    # weed out anything np.nan
    usable_A = A[np.isfinite(A)]
    ratio_of_usable_values = (usable_A.shape[] / A.shape[0]) if A.shape[0] > 0 else 0.0
    return usable_A.mean(), usable_A.var(), usable_A.shape[0], ratio_of_usable_values
