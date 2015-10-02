

import redis
import numpy as np
import json
import time

import sys, os
import getopt

import redis_server_wrapper

def configure(rsconn, batch_size, Ntrain, batch_desc_suffix, nbr_indices_sampled=32, want_exclude_partial_batch=True):

    # `batch_size` is an int,
    # `Ntrain` is the total number of training examples (to be split into batches).
    # `batch_desc_suffix` is something like "grad_norm2" describing the name of the quantity used as importance weights.
    # `nbr_indices_sampled` is the number of indices that we'll sample at a time to feed to the server.

    # "parameters:current" will contain a numpy float32 array
    # represented efficiently as a string (max 128MB, potential scaling problems)
    rsconn.set("parameters:current", "")
    rsconn.set("parameters:current_timestamp", time.time())
    # potentially not used
    rsconn.set("parameters:current_datestamp", time.strftime("%Y-%m-%d %H:%M:%S"))

    # No tolerance at all for not resampling as agressively as possible.
    resampling_threshold = nbr_indices_sampled

    L_batch_name = []
    for lower_index in range(0, Ntrain+1, batch_size):

        if Ntrain <= lower_index:
            continue

        # The data points corresponding to `upper_index` are NOT to be included,
        # but that upper_index is part of the name of batches.
        # ex : "batch_0000:0025_grad_norm2"
        upper_index = np.min([lower_index + batch_size, Ntrain])

        if want_exclude_partial_batch and (upper_index - lower_index < batch_size):
            continue
        
        assert upper_index - lower_index <= batch_size

        batch_name = "batch:%0.9d-%0.9d_%s" % (lower_index, upper_index, batch_desc_suffix)
        L_batch_name.append(batch_name)

    rsconn.delete("batch:L_names_all")
    rsconn.delete("batch:L_names_todo")
    
    for batch_name in L_batch_name:
        rsconn.rpush("batch:L_names_all", batch_name)
        # This will be a numpy float64 value represented efficiently as string.
        rsconn.set(batch_name, np.ones((1,), dtype=np.float64).tostring(order='C'))
        # print "%s set to 1.0 as float64" % batch_name
    print "Added %d batches to the server keys." % len(L_batch_name)

    # This will be a list of strings of the form "(batch_0000:0025_grad_norm2, 1.8743222e-8, 8.09980e-2)". "
    # Where the values contained are float64 in an inefficient format to be converted to python float objects.
    # The entries will be parsed using regexp and don't have to be very efficient
    # (unlike every other float64 value in here).
    rsconn.delete("importance_samples:L_(batch_name, weight, total_weights)")

    rsconn.set("config:resampling_threshold", resampling_threshold)
    rsconn.set("config:nbr_indices_sampled", nbr_indices_sampled)
    rsconn.set("config:want_exclude_partial_batch", want_exclude_partial_batch)



def run(local_server_scratch_path, local_server_port, local_server_password,
        batch_size, Ntrain, batch_desc_suffix,
        nbr_indices_sampled=1):

    if local_server_scratch_path is None:
        local_server_scratch_path = "."

    if local_server_port is None:
        local_server_port = np.random.randint(low=1025, high=65535)

    # password can be None.
    # Consider maybe generating one at random, shared with workers later.
    
    assert batch_size is not None
    assert Ntrain is not None
    assert batch_desc_suffix is not None

    # The ip of the server will have to be communicated in some way
    # to the other workers on the helios cluster.
    # We'll write to a file. Later. Not important for now.

    rserv = redis_server_wrapper.EphemeralRedisServer(  scratch_path=local_server_scratch_path,
                                                        port=local_server_port, password=local_server_password)
    rserv.start()
    time.sleep(5)
    rsconn = rserv.get_client()
    print "pinging master server : %s" % (rsconn.ping(),)


    configure(rsconn, batch_size, Ntrain, batch_desc_suffix, nbr_indices_sampled=nbr_indices_sampled)

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
        opts, args = getopt.getopt(sys.argv[1:], "hv", ["local_server_scratch_path=", "local_server_port=", "local_server_password=",
                                                        "batch_size=", "Ntrain=", "batch_desc_suffix=", "nbr_indices_sampled="])
    except getopt.GetoptError as err:
        # print help information and exit:
        print str(err) # will print something like "option -a not recognized"
        usage()
        sys.exit(2)

    local_server_scratch_path = None
    local_server_port = None
    local_server_password = None

    batch_size = None
    Ntrain = None
    batch_desc_suffix = None
    nbr_indices_sampled = 1

    verbose = False
    for o, a in opts:
        if o == "-v":
            verbose = True
        elif o in ("-h", "--help"):
            usage()
            sys.exit()
        elif o in ("--local_server_scratch_path"):
            local_server_scratch_path = a
        elif o in ("--local_server_port"):
            local_server_port = int(a)
        elif o in ("--local_server_password"):
            local_server_password = a
        elif o in ("--batch_size"):
            batch_size = int(a)
        elif o in ("--Ntrain"):
            Ntrain = int(a)
        elif o in ("--batch_desc_suffix"):
            batch_desc_suffix = a
        elif o in ("--nbr_indices_sampled"):
            nbr_indices_sampled = int(a)

        else:
            assert False, "unhandled option"
 
    # The validity of the arguments is verified in the `run` function.
    run(local_server_scratch_path, local_server_port, local_server_password,
        batch_size, Ntrain, batch_desc_suffix,
        nbr_indices_sampled=nbr_indices_sampled)


if __name__ == "__main__":
    main(sys.argv)


"""
    python run_service_database.py --local_server_port=5982 --local_server_password="patate" --batch_size=32 --Ntrain=562 --batch_desc_suffix="grad_norm2"

    python run_service_database.py --batch_size=32 --Ntrain=562 --batch_desc_suffix="grad_norm2"


"""
