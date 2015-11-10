
import redis
import numpy as np
import json
import time
import re

import signal

import sys, os
import getopt

# Change this to the real model once you want to plug it in.

# Mocked ModelAPI. Used for debugging certain things.
#from integration_distributed_training.model.mocked_model import ModelAPI
#
# The actual model that runs on SVHN.
from integration_distributed_training.model.model import ModelAPI
from common import get_rsconn_with_timeout
import integration_distributed_training.server.logger

def run(DD_config, D_server_desc):

    # TODO : Get rid of this cheap hack to circumvent my OSX's inability to see itself.
    if D_server_desc['hostname'] in ["szkmbp"]:
        D_server_desc['hostname'] = "localhost"

    rsconn = get_rsconn_with_timeout(D_server_desc['hostname'], D_server_desc['port'], D_server_desc['password'],
                                     timeout=60, wait_for_parameters_to_be_present=True)

    L_measurements = DD_config['database']['L_measurements']
    serialized_parameters_format = DD_config['database']['serialized_parameters_format']
    worker_routine = DD_config['model']['worker_routine']
    if worker_routine[0] != "sync_params":
        print "Error. Your worker_routine should always start with 'sync_params'."
        print worker_routine
        quit()

    logger = integration_distributed_training.server.logger.RedisLogger(rsconn, queue_prefix_identifier="service_worker")

    def signal_handler(signal, frame):
        print "You pressed CTRL+C."
        print "Closing the logger."
        logger.log('event', "Received SIGTERM.")
        logger.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

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


    # This will be useful for the workers when they want to stamp their importance weights
    # with a timestamp that reflects the last time that they got a fresh set of parameters.
    current_parameters = None
    parameters_current_timestamp_str = ""

    logger.log('event', "Before entering service_worker main loop.")
    while True:

        for next_action in worker_routine:
            assert next_action in [ "sync_params", "process_minibatch"]

            if next_action == "sync_params":

                logger.log('event', "sync_params")

                new_parameters_current_timestamp_str = rsconn.get("parameters:current_timestamp")
                if parameters_current_timestamp_str != new_parameters_current_timestamp_str:
                    tic = time.time()
                    current_parameters_str = rsconn.get("parameters:current")
                    toc = time.time()
                    logger.log('timing_profiler', {'sync_params_from_database' : (toc-tic)})

                    if len(current_parameters_str) == 0:
                        print "Error. No parameters found in the server."
                        print "We could recover from this error by just ignoring it, and getting the parameters the next time around."
                        quit()

                    if serialized_parameters_format == "opaque_string":
                        parameters_current_timestamp_str = new_parameters_current_timestamp_str
                        tic = time.time()
                        model_api.set_serialized_parameters(current_parameters_str)
                        toc = time.time()
                        logger.log('timing_profiler', {'model_api.set_serialized_parameters' : (toc-tic)})
                        print "The worker has received new parameters. This took %f seconds." % (toc - tic,)
                        continue
                    elif serialized_parameters_format == "ndarray_float32_tostring":
                        current_parameters = np.fromstring(current_parameters_str, dtype=np.float32)
                        parameters_current_timestamp_str = new_parameters_current_timestamp_str
                        tic = time.time()
                        model_api.set_serialized_parameters(current_parameters)
                        toc = time.time()
                        logger.log('timing_profiler', {'model_api.set_serialized_parameters' : (toc-tic)})
                        print "The worker has received new parameters. This took %f seconds." % (toc - tic,)
                        continue
                    else:
                        print "Fatal error : invalid serialized_parameters_format : %s." % serialized_parameters_format
                        quit()

            elif next_action == "process_minibatch":

                logger.log('event', "process_minibatch")
                tic_process_minibatch = time.time()

                segment = sample_segment()
                queue_name = "L_workers_%s_minibatch_indices_QUEUE" % segment
                if rsconn.llen(queue_name) == 0:
                    msg = "The worker has nothing to do.\nThe queue %s is empty." % queue_name
                    print msg
                    logger.log('event', msg)
                    # TODO : Adjust the duration of the sleep.
                    time.sleep(0.2)
                    continue

                current_minibatch_indices_str = rsconn.lpop(queue_name)
                if current_minibatch_indices_str is None or len(current_minibatch_indices_str) == 0:
                    # This is very unexpected, because it implies that we have a queue
                    # that is shorter than the number of workers. It's not illegal, but
                    # just generally not recommended for a setup.
                    msg = "The worker has nothing to do.\nIt is as though queue %s was empty when we tried to pop an element from the left." % queue_name
                    print msg
                    logger.log('event', msg)
                    # TODO : Adjust the duration of the sleep.
                    time.sleep(0.2)
                    continue


                current_minibatch_indices = np.fromstring(current_minibatch_indices_str, dtype=np.int32)

                # There is a special thing to do with the individual_importance_weight.
                # We want to keep around their previous values.

                tic = time.time()
                tmp_str = rsconn.hget("H_%s_minibatch_%s" % (segment, "individual_importance_weight"), current_minibatch_indices_str)
                rsconn.hset("H_%s_minibatch_%s" % (segment, "previous_individual_importance_weight"), current_minibatch_indices_str, tmp_str)
                toc = time.time()
                logger.log('timing_profiler', {'copied individual_importance_weight to previous_individual_importance_weight' : (toc-tic)})

                tic = time.time()
                # This returns a dictionary of numpy arrays.
                DA_measurements = model_api.worker_process_minibatch(current_minibatch_indices, segment, L_measurements)
                toc = time.time()
                logger.log('timing_profiler', {'worker_process_minibatch' : (toc-tic)})

                tic_send_measurements_to_database = time.time()
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
                        msg = "FATAL ERROR. You have %d invalid values returned for %s." % (number_of_invalid_values, measurement)
                        print msg
                        print A_values
                        logger.log('event', msg)
                        quit()
                        #print "Starting debugger."
                        #import pdb; pdb.set_trace()

                    rsconn.hset("H_%s_minibatch_%s" % (segment, measurement), current_minibatch_indices_str, A_values.tostring(order='C'))

                    previous_update_timestamp_str = rsconn.hget("H_%s_minibatch_%s_measurement_last_update_timestamp" % (segment, measurement), current_minibatch_indices_str)
                    if previous_update_timestamp_str is None or len(previous_update_timestamp_str) == 0:
                        print "The measurements are supposed to be initialized when starting the database."
                        print "They are supposed to have a timestamp set at that time."
                        print "This is not a serious error from which we could not recover, but it signals that there is a bug, so let's quit() here."
                        quit()
                        #previous_update_timestamp = 0.0
                    else:
                        previous_update_timestamp = float(previous_update_timestamp_str)

                    print "(%s, %s) timestamp delta between updates to that measurement : %f" % (segment, measurement, time.time() - previous_update_timestamp, )

                    current_update_timestamp = time.time()
                    rsconn.hset("H_%s_minibatch_%s_measurement_last_update_timestamp" % (segment, measurement), current_minibatch_indices_str, current_update_timestamp)

                    delay_between_measurement_update = current_update_timestamp - previous_update_timestamp
                    delay_between_measurement_update_and_parameter_update = current_update_timestamp - float(parameters_current_timestamp_str)

                    rsconn.hset("H_%s_minibatch_%s_delay_between_measurement_update" % (segment, measurement), current_minibatch_indices_str, delay_between_measurement_update)
                    rsconn.hset("H_%s_minibatch_%s_delay_between_measurement_update_and_parameter_update" % (segment, measurement), current_minibatch_indices_str, delay_between_measurement_update_and_parameter_update)


                    #print "delay_between_measurement_update : %f seconds" % delay_between_measurement_update
                    #print "delay_between_measurement_update_and_parameter_update : %f seconds" % delay_between_measurement_update_and_parameter_update

                    # Be careful. If you re-indent the next block deeper,
                    # you'll mess up everything with the re-queuing of the minibatches.

                toc_send_measurements_to_database = time.time()
                logger.log('timing_profiler', {'send_measurements_to_database' : (toc_send_measurements_to_database-tic_send_measurements_to_database)})
                # We could log this for every measurement, but we'll just log it for one of them.
                # Otherwise, this is multiplying the messaging without real need.
                # We'll use the values for the last measurement, which outlasts the for loop above
                # due to shitty python scoping.
                logger.log('delay', {'delay_between_measurement_update' : delay_between_measurement_update, 'delay_between_measurement_update_and_parameter_update':delay_between_measurement_update_and_parameter_update})


                # Push back that minibatch to the right of the queue.
                # It will eventually find its way back to some worker,
                # but we will cover all the other ones before that happens.
                rsconn.rpush(queue_name, current_minibatch_indices_str)
                toc_process_minibatch = time.time()
                logger.log('timing_profiler', {'process_minibatch' : (toc_process_minibatch-tic_process_minibatch)})
                msg = "Processed one minibatch from %s. Pushed back to back of the line. Total time taken is %f seconds." % (segment, toc_process_minibatch-tic_process_minibatch)
                print msg
                logger.log('event', msg)
