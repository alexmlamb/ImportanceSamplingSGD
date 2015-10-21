
import redis
import numpy as np
import json
import time
import re

import sys, os
import getopt

from mocked_model import ModelAPI

from common import get_rsconn_with_timeout, read_config

def run(server_ip, server_port, server_password):

    rsconn = get_rsconn_with_timeout(server_ip, server_port, server_password,
                                     timeout=60, wait_for_parameters_to_be_present=True)
    config = read_config(rsconn)
    L_measurements = config['L_measurements']

    model_api = ModelAPI()

    #D_segment_priorities = {'train' : 50, 'valid' : 1, 'test' : 1}
    segment_priorities_p = np.array([50, 1, 1], dtype=np.float32)
    segment_priorities_p /= segment_priorities_p.sum()
    segment_priorities_p = segment_priorities_p.cumsum()
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
    # the cost of processing one minibatch, we can chose to do it every iteration.
    # However, this does not mean that it's such a good idea to update the parameters
    # so aggressively because it could lead to a situation where many workers just
    # overrun the server with requests for updates.
    # TODO : Maybe have a timer similar to the one for Legion extensions where we
    #        throttle based on the time that it takes to sync the parameters on average.

    # This could be a constant from the configuration.
    minimum_number_of_minibatch_processed_before_parameter_update = 5
    M = minimum_number_of_minibatch_processed_before_parameter_update
    m = M



    current_parameters = None
    parameters_current_timestamp = ""
    while True:

        currently_want_to_update_parameters = (M <= m)

        # Task (1)

        if currently_want_to_update_parameters:

            new_parameters_current_timestamp = rsconn.get("parameters:current_timestamp")
            if parameters_current_timestamp != new_parameters_current_timestamp:

                current_parameters_str = rsconn.get("parameters:current")
                if len(current_parameters_str) == 0:
                    print "No parameters found in the server."
                    print "Might as well sleep, but this is NEVER supposed to happen."
                    time.sleep(0.2)
                    m = 0
                    continue
                else:
                    current_parameters = np.fromstring(current_parameters_str, dtype=np.float32)
                    parameters_current_timestamp = new_parameters_current_timestamp
                    model_api.set_serialized_parameters(current_parameters)
                    m = 0
                    continue

            else:
                # Go on to Task (2).
                pass
                # Note that, if it's time to update the parameters, but new ones
                # just have not reached the database (ex : maybe the master has not
                # pushed anything), then we don't really want to reset m=0 because we'll
                # simply get the parameter update as soon as the master sends it to the
                # database.



        # Task (2)

        segment = sample_segment()
        queue_name = "L_workers_%s_minibatch_indices_QUEUE" % segment
        if rsconn.llen(queue_name) == 0:
            print "The worker has nothing to do."
            print "The queue %s is empty." % queue_name
            # TODO : Adjust the duration of the sleep.
            time.sleep(0.2)
            continue

        current_minibatch_indices_str = rsconn.lpop(queue_name)
        if current_minibatch_indices_str is None or len(current_minibatch_indices_str) == 0:
            # This is very unexpected, because it implies that we have a queue
            # that is shorted than the number of workers. It's not illegal, but
            # just generally not recommended for a setup.
            print "The worker has nothing to do."
            print "It is as though queue %s was empty when we tried to pop an element from the left." % queue_name
            # TODO : Adjust the duration of the sleep.
            time.sleep(0.2)
            m += 1
            continue
        else:
            current_minibatch_indices = np.fromstring(current_minibatch_indices_str, dtype=np.int32)
            # This returns a dictionary of numpy arrays.
            DA_measurements = model_api.worker_process_minibatch(current_minibatch_indices, segment, L_measurements)

            # Update the measurements. Update the timestamps.
            # None of the measurements should be missing.
            for measurement in L_measurements:
                A_values = DA_measurements[measurement]
                assert type(A_values) == np.ndarray, "Your `worker_process_minibatch` function is supposed to return an array of np.float32 as measurements, but now those values are not even numpy arrays. They are %s instead." % type(A_values)
                assert A_values.dtype == np.float32, "Your `worker_process_minibatch` function is supposed to return an array of np.float32 as measurements, but now that array has dtype %s instead." % A_values.dtype
                # Write 0.0 as default value in all the measurements.
                rsconn.hset("H_%s_minibatch_%s" % (segment, measurement), current_minibatch_indices_str, A_values)

                previous_update_timestamp = float(rsconn.hget("H_%s_minibatch_%s_last_update_timestamp" % (segment, measurement), current_minibatch_indices_str))
                current_update_timestamp = time.time()
                rsconn.hset("H_%s_minibatch_%s_last_update_timestamp" % (segment, measurement), current_minibatch_indices_str, current_update_timestamp)

                # TODO : Not sure yet where we would want to log this. Figure it out.
                #        It would need to be in a place where the segment/measurement
                #        are also recorded, because we can't expect the "valid" and "test"
                #        measurements to be averaged with the "train". That would not
                #        be an interesting thing to record.
                delay_between_measurement_refresh = previous_update_timestamp - current_update_timestamp

                # Push back that minibatch to the right of the queue.
                # It will eventually find its way back to some worker,
                # but we will cover all the other ones before that happens.
                rsconn.rpush(queue_name, current_minibatch_indices_str)
                print "Processed one minibatch from (%s, %s). Pushed back to back of the line." % (segment, measurement)

                m += 1
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
