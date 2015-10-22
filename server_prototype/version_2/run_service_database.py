

import redis
import numpy as np
import json
import time

import sys, os
import getopt

import redis_server_wrapper

def configure(  rsconn,
                workers_minibatch_size, master_minibatch_size,
                dataset_name,
                Ntrain, Nvalid, Ntest,
                L_measurements = ["importance_weight"],
                want_only_indices_for_master=True,
                want_exclude_partial_minibatch=True):

    # `workers_minibatch_size` is an int describing how large are the minibatches for the workers.
    # `master_minibatch_size` is an int describing how large are the minibatches for the master.
    # `dataset_name` is a string that is somewhat useless, but is still a good idea to include in the config.
    # `Ntrain` is the total number of training examples (to be split into minibatches).
    # `Nvalid` is the total number of validation examples (to be split into minibatches).
    # `Ntest`  is the total number of test examples (to be split into minibatches).
    # `L_measurements` is a list of quantities computed for each example.
    # `want_only_indices_for_master` determines whether the master will be given arrays of indices or minibatch data directly.
    # `want_exclude_partial_minibatch` indicates if we want to forget about the data that doesn't fit in a complete minibatch.


    rsconn.delete("initialization_is_done")


    # "parameters:current" will contain a numpy float32 array
    # represented efficiently as a string (max 128MB, potential scaling problems)
    rsconn.set("parameters:current", "")
    rsconn.set("parameters:current_timestamp", time.time())
    # potentially not used
    rsconn.set("parameters:current_datestamp", time.strftime("%Y-%m-%d %H:%M:%S"))

    # This could have been the argument to this function,
    # but we don't have to decide on this right now.
    L_dataset_desc = []
    L_segments = []

    assert 0 < Ntrain
    if Ntrain != 0:
        L_dataset_desc.append({'segment' : "train", 'N' : Ntrain})
        L_segments.append("train")
    if Nvalid != 0:
        L_dataset_desc.append({'segment' : "valid", 'N' : Nvalid})
        L_segments.append("valid")
    if Ntest != 0:
        L_dataset_desc.append({'segment' : "test", 'N' : Ntest})
        L_segments.append("test")


    for dataset_desc in L_dataset_desc:

        segment = dataset_desc['segment']
        N = dataset_desc['N']

        rsconn.delete("L_workers_%s_minibatch_indices_QUEUE" % segment)
        rsconn.delete("L_workers_%s_minibatch_indices_ALL" % segment)

        for lower_index in range(0, N+1, workers_minibatch_size):

            if N <= lower_index:
                continue

            # The data points corresponding to `upper_index` are NOT to be included.
            upper_index = np.min([lower_index + workers_minibatch_size, N])

            if want_exclude_partial_minibatch and (upper_index - lower_index < workers_minibatch_size):
                continue

            assert upper_index - lower_index <= workers_minibatch_size

            A_indices = np.arange(lower_index, upper_index, dtype=np.int32)
            A_indices_str = A_indices.tostring()

            rsconn.rpush("L_workers_%s_minibatch_indices_QUEUE" % segment, A_indices_str)
            rsconn.rpush("L_workers_%s_minibatch_indices_ALL" % segment, A_indices_str)

            for measurement in L_measurements:
                # Write 0.0 as default value in all the measurements.
                rsconn.hset("H_%s_minibatch_%s" % (segment, measurement), A_indices_str, np.zeros(A_indices.shape, dtype=np.float32).tostring(order='C'))
                rsconn.hset("H_%s_minibatch_%s_last_update_timestamp" % (segment, measurement), A_indices_str, time.time())

                #print "H_%s_minibatch_%s" % (segment, measurement)

    # The master does not really differentiate between the various
    # segments of the dataset. It just takes whatever it is fed
    # because nothing other than the training data should go in there.
    # We decided to put "_train_" in there nonetheless. Seems more consistent.

    # used when `want_only_indices_for_master` is True
    rsconn.delete("L_master_train_minibatch_indices_and_info_QUEUE")
    # used when `want_only_indices_for_master` is False
    rsconn.delete("L_master_train_minibatch_data_and_info_QUEUE")


    rsconn.set("config:dataset_name", dataset_name)
    rsconn.delete("config:L_measurements")
    for measurement in L_measurements:
        rsconn.rpush("config:L_measurements", measurement)
    rsconn.delete("config:L_segments")
    for segment in L_segments:
        rsconn.rpush("config:L_segments", segment)
    rsconn.set("config:workers_minibatch_size", workers_minibatch_size)
    rsconn.set("config:master_minibatch_size", master_minibatch_size)
    rsconn.set("config:want_only_indices_for_master", want_only_indices_for_master)
    rsconn.set("config:want_exclude_partial_minibatch", want_exclude_partial_minibatch)

    rsconn.set("initialization_is_done", True)



def run(server_scratch_path, server_port, server_password,
        workers_minibatch_size, master_minibatch_size,
        dataset_name,
        Ntrain, Nvalid, Ntest):

    if server_scratch_path is None:
        server_scratch_path = "."

    if server_port is None:
        server_port = np.random.randint(low=1025, high=65535)

    # password can be None.
    # Consider maybe generating one at random, shared with workers later.

    assert workers_minibatch_size is not None and 0 < workers_minibatch_size
    assert master_minibatch_size is not None and 0 < master_minibatch_size
    assert dataset_name is not None
    assert Ntrain is not None and 0 < Ntrain
    assert Nvalid is not None
    assert Ntest is not None

    # The ip of the server will have to be communicated in some way
    # to the other workers on the helios cluster.
    # We'll write to a file. Later. Not important for now.

    rserv = redis_server_wrapper.EphemeralRedisServer(  scratch_path=server_scratch_path,
                                                        port=server_port, password=server_password)
    rserv.start()
    time.sleep(5)
    rsconn = rserv.get_client()
    print "pinging master server : %s" % (rsconn.ping(),)

    # Hardcode those two measurements for now.
    # Moreover, we're using the gradient_square_norm as importance_weight
    # so it's going to be the same value.
    L_measurements = ["importance_weight", "gradient_square_norm", "loss"]

    configure(  rsconn,
                workers_minibatch_size=workers_minibatch_size, master_minibatch_size=master_minibatch_size,
                dataset_name=dataset_name,
                Ntrain=Ntrain, Nvalid=Nvalid, Ntest=Ntest,
                L_measurements=L_measurements)


    # You might want to do something to shutdown the redis server more gracefully upon hitting ctrl-c
    # or have some other way of closing it. Not important for now.
    while True:
        print "..running server.."
        time.sleep(5)



def usage():
    print ""

def main(argv):
    """
    """

    try:
        opts, args = getopt.getopt(sys.argv[1:], "hv", ["server_scratch_path=", "server_port=", "server_password=",
                                                        "workers_minibatch_size=", "master_minibatch_size=",
                                                        "dataset_name=",
                                                        "Ntrain=", "Nvalid=", "Ntest="])
    except getopt.GetoptError as err:
        # print help information and exit:
        print str(err) # will print something like "option -a not recognized"
        usage()
        sys.exit(2)

    server_scratch_path = None
    server_port = None
    server_password = None

    workers_minibatch_size = 256
    master_minibatch_size = 256
    dataset_name = None # "SVHN2"
    Ntrain = 0
    Nvalid = 0
    Ntest = 0



    verbose = False
    for o, a in opts:
        if o == "-v":
            verbose = True
        elif o in ("-h", "--help"):
            usage()
            sys.exit()
        elif o in ("--server_scratch_path"):
            server_scratch_path = a
        elif o in ("--server_port"):
            server_port = int(a)
        elif o in ("--server_password"):
            server_password = a
        elif o in ("--workers_minibatch_size"):
            workers_minibatch_size = int(a)
        elif o in ("--master_minibatch_size"):
            master_minibatch_size = int(a)
        elif o in ("--dataset_name"):
            dataset_name = a
        elif o in ("--Ntrain"):
            Ntrain = int(a)
        elif o in ("--Nvalid"):
            Nvalid = int(a)
        elif o in ("--Ntest"):
            Ntest = int(a)
        else:
            assert False, "unhandled option"

    # The validity of the arguments is verified in the `run` function.
    run(server_scratch_path, server_port, server_password,
        workers_minibatch_size, master_minibatch_size,
        dataset_name,
        Ntrain, Nvalid, Ntest)


if __name__ == "__main__":
    main(sys.argv)


"""
    python run_service_database.py --server_port=5982 --server_password="patate" --workers_minibatch_size=128 --master_minibatch_size=512 --Ntrain=2000 --dataset_name=SVHN2

    python run_service_database.py --Ntrain=512 --dataset_name=SVHN2

"""
