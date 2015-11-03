
import redis
import numpy as np
import json
import time
import re

import sys, os
import getopt

# Change this to the real model once you want to plug it in.

# Mocked ModelAPI. Used for debugging certain things.
#from integration_distributed_training.model.mocked_model import ModelAPI
#
# The actual model that runs on SVHN.
from integration_distributed_training.model.model import ModelAPI


from common import get_rsconn_with_timeout

def run(DD_config, D_server_desc):

    # TODO : Get rid of this cheap hack to circumvent my OSX's inability to see itself.
    if D_server_desc['hostname'] in ["szkmbp"]:
        D_server_desc['hostname'] = "localhost"

    rsconn = get_rsconn_with_timeout(D_server_desc['hostname'], D_server_desc['port'], D_server_desc['password'],
                                     timeout=60, wait_for_parameters_to_be_present=True)

    L_measurements = DD_config['database']['L_measurements']
    serialized_parameters_format = DD_config['database']['serialized_parameters_format']

    model_api = ModelAPI(DD_config['model'])


    #D_segment_priorities = {'train' : 50, 'valid' : 1, 'test' : 1}
    segment_priorities_p = np.array([20, 1, 1], dtype=np.float32)
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
    minimum_number_of_minibatch_processed_before_parameter_update = 4
    M = minimum_number_of_minibatch_processed_before_parameter_update
    m = M

    current_parameters = None
    parameters_current_timestamp = ""
    while True:

        currently_want_to_update_parameters = (M <= m)

        # Task (1)

        if currently_want_to_update_parameters:

            tic = time.time()
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

                    if serialized_parameters_format == "opaque_string":
                        parameters_current_timestamp = new_parameters_current_timestamp
                        model_api.set_serialized_parameters(current_parameters_str)
                        m = 0
                        toc = time.time()
                        print "The worker has received new parameters. This took %f seconds." % (toc - tic,)
                        continue
                    elif serialized_parameters_format == "ndarray_float32_tostring":
                        current_parameters = np.fromstring(current_parameters_str, dtype=np.float32)
                        parameters_current_timestamp = new_parameters_current_timestamp
                        model_api.set_serialized_parameters(current_parameters)
                        m = 0
                        toc = time.time()
                        print "The worker has received new parameters. This took %f seconds." % (toc - tic,)
                        continue
                    else:
                        print "Fatal error : invalid serialized_parameters_format : %s." % serialized_parameters_format
                        quit()

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
            tic = time.time()
            current_minibatch_indices = np.fromstring(current_minibatch_indices_str, dtype=np.int32)
            # This returns a dictionary of numpy arrays.
            DA_measurements = model_api.worker_process_minibatch(current_minibatch_indices, segment, L_measurements)

            # Update the measurements. Update the timestamps.
            # None of the measurements should be missing.
            for measurement in L_measurements:
                A_values = DA_measurements[measurement]
                assert type(A_values) == np.ndarray, "Your `worker_process_minibatch` function is supposed to return an array of np.float32 as measurements (%s), but now those values are not even numpy arrays. They are %s instead." % (measurement, type(A_values))
                if A_values.dtype == np.float64:
                    # This conversion is acceptable.
                    A_values = A_values.astype(np.float32)
                assert A_values.dtype == np.float32, "Your `worker_process_minibatch` function is supposed to return an array of np.float32 as measurements (%s), but now that array has dtype %s instead." % (measurement, A_values.dtype)

                number_of_invalid_values = np.logical_not(np.isfinite(A_values)).sum()
                if 0 < number_of_invalid_values:
                    print "FATAL ERROR. You have %d invalid values returned for %s." % (number_of_invalid_values, measurement)
                    print A_values
                    quit()
                    #print "Starting debugger."
                    #import pdb; pdb.set_trace()

                # Write 0.0 as default value in all the measurements.
                rsconn.hset("H_%s_minibatch_%s" % (segment, measurement), current_minibatch_indices_str, A_values.tostring(order='C'))

                previous_update_timestamp_str = rsconn.hget("H_%s_minibatch_%s_measurement_last_update_timestamp" % (segment, measurement), current_minibatch_indices_str)
                if previous_update_timestamp_str is None or len(previous_update_timestamp_str) == 0:
                    # This is a garbage value, but it's going to be used the first around, and only then.
                    previous_update_timestamp = 0.0
                else:
                    previous_update_timestamp = float(previous_update_timestamp_str)

                print "timestamp delta between updates to that measurement : %f" % (time.time() - previous_update_timestamp, )

                current_update_timestamp = time.time()
                rsconn.hset("H_%s_minibatch_%s_measurement_last_update_timestamp" % (segment, measurement), current_minibatch_indices_str, current_update_timestamp)

                delay_between_measurement_update = float(current_update_timestamp) - float(previous_update_timestamp)
                delay_between_measurement_update_and_parameter_update = float(current_update_timestamp) - float(parameters_current_timestamp)

                rsconn.hset("H_%s_minibatch_%s_delay_between_measurement_update" % (segment, measurement), current_minibatch_indices_str, delay_between_measurement_update)
                rsconn.hset("H_%s_minibatch_%s_delay_between_measurement_update_and_parameter_update" % (segment, measurement), current_minibatch_indices_str, delay_between_measurement_update_and_parameter_update)

                #print "delay_between_measurement_update : %f seconds" % delay_between_measurement_update
                #print "delay_between_measurement_update_and_parameter_update : %f seconds" % delay_between_measurement_update_and_parameter_update

                # Be careful. If you re-indent the next block deeper,
                # you'll mess up everything with the re-queuing of the minibatches.

            # Push back that minibatch to the right of the queue.
            # It will eventually find its way back to some worker,
            # but we will cover all the other ones before that happens.
            rsconn.rpush(queue_name, current_minibatch_indices_str)
            toc = time.time()
            print "Processed one minibatch from %s. Pushed back to back of the line. Total time taken is %f seconds." % (segment, toc - tic)

            m += 1
            continue
