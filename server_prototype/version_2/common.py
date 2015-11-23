
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


def read_config(rsconn):

    dataset_name = rsconn.get("config:dataset_name")
    L_measurements = rsconn.lrange("config:L_measurements", 0, rsconn.llen("config:L_measurements"))
    L_segments = rsconn.lrange("config:L_segments", 0, rsconn.llen("config:L_segments"))

    workers_minibatch_size = int(rsconn.get("config:workers_minibatch_size"))
    master_minibatch_size = int(rsconn.get("config:master_minibatch_size"))
    want_only_indices_for_master = rsconn.get("config:want_only_indices_for_master") in ["1", "true", "True"]
    want_exclude_partial_minibatch = rsconn.get("config:want_exclude_partial_minibatch") in ["1", "true", "True"]
    serialized_parameters_format = rsconn.get("config:serialized_parameters_format")

    #return dict(dataset_name=dataset_name,
    #L_measurements=L_measurements, workers_minibatch_size=workers_minibatch_size)

    return locals()
