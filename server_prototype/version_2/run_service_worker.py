
import redis
import numpy as np
import json
import time
import re

import sys, os
import getopt

SIMULATED_BATCH_UPDATE_TIME = 1.0

from mocked_model import ModelAPI

from common import get_rsconn_with_timeout, read_config

def run(server_ip, server_port, server_password):

    rsconn = get_rsconn_with_timeout(server_ip, server_port, server_password, timeout=60)
    config = read_config(rsconn)

    model_api = ModelAPI()

    #D_segment_priorities = {'train' : 50, 'valid' : 1, 'test' : 1}
    segment_priorities_p = np.array([50, 1, 1])
    segment_priorities_p /= segment_priorities_p.sum()
    segment_priorities_v = ['train', 'valid', 'test']
    def sample_segment():
        r = np.random.rand()
        for (i, e) in enumerate(segment_priorities_p):
            if r <= e:
                return segment_priorities_v[i]

    # The worker has to watch two things.
    #
    # (1) Have the parameters been updated on the server ?
    #     Check out the timestamp to determine if they have been updated.
    #     (Because of the assumption that the master updates the parameters
    #     and *then* the timestamp.)
    #     If they have been updated, we want to fetch a copy a convert it
    #     to a numpy array.
    #
    # (2) Process left-most entries from L_workers_%s_minibatch_indices_QUEUE
    #     according to their priorities given by `segment_priorities_p`.
    #
    # Because the check for parameter updates should be ultra fast compared to
    # the cost of processing one minibatch, we chose to do it every iteration.

    current_parameters = None
    parameters_current_timestamp = ""
    while True:

        # Task (1)

        new_parameters_current_timestamp = rsconn.get("parameters:current_timestamp")
        if parameters_current_timestamp != new_parameters_current_timestamp:

            current_parameters_str = rsconn.get("parameters:current")
            if len(current_parameters_str) == 0:
                print "No parameters found in the server."
                print "Might as well sleep, but this is NEVER supposed to happen."
                time.sleep(0.2)
                continue
            else:
                current_parameters = np.fromstring(current_parameters_str, dtype=np.float32)
                parameters_current_timestamp = new_parameters_current_timestamp
                model_api.set_serialized_parameters(current_parameters)
                continue


        # Task (2)

        segment = sample_segment()
        queue_name = "L_workers_%s_minibatch_indices_QUEUE" % segment
        if rsconn.llen(queue_name) == 0:
            print "The worker has nothing to do."
            print "The queue %s is empty." % queue_name
            # TODO : Adjust the duration of the sleep.
            time.sleep(0.2)
            continue

        nbr_batch_name_todo = rsconn.llen("batch:L_names_todo")  # for debugging, potentially oudated value
        if batch_name is None or len(batch_name) == 0:
            # Note that the "batch:L_names_todo" might be temporarily gone
            # from the server because that's how the assistant is updating it
            # to put fresh values in there. The worker just needs to stay calm
            # when that happens and not quit() in despair.
            print "The worker has nothing to do. Might as well sleep."
            # TODO : Adjust the duration of the sleep.
            time.sleep(0.2)
            continue
        else:
            (lower_index, upper_index, suffix) = decode_batch_name(batch_name)

            print "The worker is processing %s. Estimated %d batches left in the todo list." % (batch_name, nbr_batch_name_todo)
            print "(lower_index, upper_index, suffix)"
            print (lower_index, upper_index, suffix)

            # TODO : Compute the actual gradient norm here.
            rsconn.set(batch_name, np.ones((1,), dtype=np.float64).tostring(order='C'))

            # Sleep to simulate work time.
            time.sleep(SIMULATED_BATCH_UPDATE_TIME)

            print ""
            continue


def usage():
    print ""

def main(argv):
    """
    """

    try:
        opts, args = getopt.getopt(sys.argv[1:], "hv", ["server_ip=", "server_port=", "server_password="])
    except getopt.GetoptError as err:
        # print help information and exit:
        print str(err) # will print something like "option -a not recognized"
        usage()
        sys.exit(2)

    server_ip = "localhost"
    server_port = None
    server_password = None

    verbose = False
    for o, a in opts:
        if o == "-v":
            verbose = True
        elif o in ("-h", "--help"):
            usage()
            sys.exit()
        elif o in ("--server_ip"):
            server_ip = a
        elif o in ("--server_port"):
            server_port = int(a)
        elif o in ("--server_password"):
            server_password = a
        else:
            assert False, "unhandled option"


    run(server_ip, server_port, server_password)


if __name__ == "__main__":
    main(sys.argv)


"""
    python run_service_worker.py --server_port=5982 --server_password="patate"

"""
