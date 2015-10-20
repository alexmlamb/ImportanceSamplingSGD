


def get_rsconn_with_timeout(server_ip, server_port, server_password, timeout=60):

    assert server_ip
    assert server_port
    assert server_password

    initial_conn_timestamp = time.time()
    success = False
    while time.time() - initial_conn_timestamp < timeout:

        try:
            rsconn = redis.StrictRedis(host=server_ip, port=server_port, password=server_password)
            print "Service Worker connected to local server."
            success = True
            break
        except:
            time.sleep(5)
            print "Service Worker failed to connect to local server. Will retry in 5s."

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
        else:
            print "Experiment has not been initialized completely yet. Will retry in 5s."
            time.sleep(5)

    if not success:
        print "Quitting."
        quit()

    return rsconn


def read_config(rsconn):

    dataset_name = rsconn.get("config:dataset_name")
    L_measurements = rsconn.lrange("config:L_measurements", 0, rsconn.llen("config:L_measurements"))
    L_segments = rsconn.lrange("config:L_segments", 0, rsconn.llen("config:L_segments"))

    workers_minibatch_size = rsconn.get("config:workers_minibatch_size")
    master_minibatch_size = rsconn.get("config:master_minibatch_size")
    want_only_indices_for_master = rsconn.get("config:want_only_indices_for_master")
    want_exclude_partial_minibatch = rsconn.get("config:want_exclude_partial_minibatch")

    #return dict(dataset_name=dataset_name,
    #L_measurements=L_measurements, workers_minibatch_size=workers_minibatch_size)

    return locals()
