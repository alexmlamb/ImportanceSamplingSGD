
import redis
import numpy as np
import json
import time
import re

import signal

import sys, os
import getopt
import random

# Change this to the real model once you want to plug it in.

# Mocked ModelAPI. Used for debugging certain things.
#from integration_distributed_training.model.mocked_model import ModelAPI
#
# The actual model that runs on SVHN.
from integration_distributed_training.model.model import ModelAPI
from sampling_for_master import sample_indices_and_scaling_factors, get_raw_importance_weights, filter_raw_importance_weights, record_importance_weights_statistics

from startup import get_rsconn_with_timeout, check_if_parameters_are_present
from common import setup_python_logger, wait_until_all_measurements_are_updated_by_workers

# python's logging module
import logging
import pprint
# our own logger that sends stuff over to the database
import integration_distributed_training.server.logger

def run(DD_config, D_server_desc):

    if D_server_desc['hostname'] in ["szkmbp"]:
        D_server_desc['hostname'] = "localhost"

    rsconn = get_rsconn_with_timeout(D_server_desc,
                                     timeout=DD_config['database']['connection_setup_timeout'], wait_for_parameters_to_be_present=False)

    L_measurements = DD_config['database']['L_measurements']

    master_minibatch_size = DD_config['database']['master_minibatch_size']
    serialized_parameters_format = DD_config['database']['serialized_parameters_format']
    Ntrain = DD_config['model']['Ntrain']
    # Default behavior is to have no staleness, and perform ISGD from the moment that we
    # get values for all the importance weights. Until then, we do USGD.
    staleness_threshold_seconds = DD_config['database']['staleness_threshold_seconds']
    staleness_threshold_num_minibatches_master_processed = DD_config['database']['staleness_threshold_num_minibatches_master_processed']
    importance_weight_additive_constant = DD_config['database']['importance_weight_additive_constant']

    want_master_to_do_USGD_when_ISGD_is_not_possible = DD_config['database'].get('want_master_to_do_USGD_when_ISGD_is_not_possible', True)
    master_usable_importance_weights_threshold_to_ISGD = DD_config['database'].get('master_usable_importance_weights_threshold_to_ISGD', 1.0)
    master_routine = DD_config['model']['master_routine']
    if master_routine[0] != "sync_params":
        print "Error. Your master_routine should always start with 'sync_params'."
        print master_routine
        quit()
    turn_off_importance_sampling = DD_config["model"].get("turn_off_importance_sampling", False)



    # set up python logging system for logging_folder
    setup_python_logger(folder=DD_config["database"]["logging_folder"])
    logging.info(pprint.pformat(DD_config))

    remote_redis_logger = integration_distributed_training.server.logger.RedisLogger(rsconn, queue_prefix_identifier="service_master")



    model_api = ModelAPI(DD_config['model'])
    # This `record_machine_info` has to be called after the component that
    # makes use of theano if we hope to properly record the theano.config.
    integration_distributed_training.server.logger.record_machine_info(remote_redis_logger)

    # It's very important to determine if we're resuming from a previous run,
    # in which case we really want to load the paramters to resume training.
    if not check_if_parameters_are_present(rsconn):
        ### resuming_from_previous_run = False ###
        msg = "Starting a new run."
        remote_redis_logger.log('event', msg)
        logging.info(msg)
    else:
        ### resuming_from_previous_run = True ###

        msg = "Resuming from previous run."
        remote_redis_logger.log('event', msg)
        logging.info(msg)

        # This whole section is taken almost exactly from the service_worker.
        tic = time.time()
        current_parameters_str = rsconn.get("parameters:current")
        toc = time.time()
        remote_redis_logger.log('timing_profiler', {'sync_params_from_database' : (toc-tic)})

        if len(current_parameters_str) == 0:
            print "Error. No parameters found in the server."
            quit()

        if serialized_parameters_format == "opaque_string":
            tic = time.time()
            model_api.set_serialized_parameters(current_parameters_str)
            toc = time.time()
            remote_redis_logger.log('timing_profiler', {'model_api.set_serialized_parameters' : (toc-tic)})
            logging.info("The master has received initial parameters. This took %f seconds." % (toc - tic,))

        elif serialized_parameters_format == "ndarray_float32_tostring":
            parameters_current_timestamp_str = new_parameters_current_timestamp_str
            tic = time.time()
            model_api.set_serialized_parameters(current_parameters)
            toc = time.time()
            remote_redis_logger.log('timing_profiler', {'model_api.set_serialized_parameters' : (toc-tic)})
            logging.info("The master has received initial parameters. This took %f seconds." % (toc - tic,))

        else:
            logging.info("Fatal error : invalid serialized_parameters_format : %s." % serialized_parameters_format)
            quit()


    # Run just a simple test to make sure that the importance weights have been
    # set to something. In theory, there should always be valid values in there,
    # so this is just a sanity check.
    segment = "train"
    measurement = "individual_importance_weight"
    #nbr_of_present_importance_weights = rsconn.hlen("H_%s_minibatch_%s" % (segment, measurement))
    #assert 0 < nbr_of_present_importance_weights, "Error. The database should have been set up to have dummy importance weights at least."

    #print "Master found %d importance weights in the database." % nbr_of_present_importance_weights


    # The master splits its time between two tasks.
    #
    # (1) Publish the parameters back to the server,
    #     which triggers a cascade of re-evaluation of
    #     importance weights for every mi    sys.exit(0)

    # (2) Get samples representing training examples
    #     on which you perform training steps, taking into
    #     consideration all the things about the importance weights.
    #
    # Ultimately, the parameters must be shared, but it is
    # wasteful to do it at every training step. We have to find
    # the right balance.
    #
    # Task (1) should also be triggered on the first iteration
    # to initialize the parameters on the server before anything
    # else (that being said, the initial weights for all the batches
    # are 1.0, so things could start with Task (2) since the assistant
    # would start by resampling the indices.

    queue_name = "L_master_train_minibatch_indices_and_info_QUEUE"

    num_minibatches_master_processed_str = rsconn.get("parameters:num_minibatches_master_processed")
    if num_minibatches_master_processed_str is None or len(num_minibatches_master_processed_str) == 0:
        num_minibatches_master_processed = 0.0
        rsconn.set("parameters:num_minibatches_master_processed", num_minibatches_master_processed)
    else:
        num_minibatches_master_processed = float(num_minibatches_master_processed_str)

    print "num_minibatches_master_processed is %f" % num_minibatches_master_processed

    # The main loop runs until the user hits CTLR+C or until
    # the Helios cluster sends the SIGTERM to that process
    # five minutes before the end of training.
    def signal_handler(signal, frame):
        print "SIGTERM received for the first time."
        print "Will break from master main loop."
        print "Will make logger sync to database before terminating."
        print ""
        signal_handler.remote_redis_logger.log('event', "Received SIGTERM.")
        signal_handler.remote_redis_logger.close()
        sys.exit(0)
    #
    signal.signal(signal.SIGINT, signal_handler)
    # I'm forced to use weird function properties because python
    # has stupid scoping rules.
    signal_handler.remote_redis_logger = remote_redis_logger

    # cache those values to use them for more than one computation
    D_importance_weights_and_more = None
    extra_statistics = None

    remote_redis_logger.log('event', "Before entering service_master main loop.")
    while True:

        for next_action in master_routine:
            assert next_action in [ "sync_params", "refresh_importance_weights", "process_minibatch", # the normal ones
                                    "wait_for_workers_to_update_all_the_importance_weights"]          # the special ones

            if next_action == "sync_params":

                remote_redis_logger.log('event', "sync_params")
                tic = time.time()
                if serialized_parameters_format == "opaque_string":
                    current_parameters_str = model_api.get_serialized_parameters()
                elif serialized_parameters_format == "ndarray_float32_tostring":
                    current_parameters_str = model_api.get_serialized_parameters().tostring(order='C')
                else:
                    logging.info("Fatal error : invalid serialized_parameters_format : %s." % serialized_parameters_format)
                    quit()
                toc = time.time()
                remote_redis_logger.log('timing_profiler', {'read_parameters_from_model' : (toc-tic)})

                tic = time.time()
                rsconn.set("parameters:current", current_parameters_str)
                rsconn.set("parameters:current_timestamp", time.time())
                rsconn.set("parameters:num_minibatches_master_processed", num_minibatches_master_processed)
                # potentially not used
                rsconn.set("parameters:current_datestamp", time.strftime("%Y-%m-%d %H:%M:%S"))
                toc = time.time()
                remote_redis_logger.log('timing_profiler', {'sync_params_to_database' : (toc-tic)})
                print "The master has updated the parameters."

            elif next_action == "wait_for_workers_to_update_all_the_importance_weights":

                # This next line is to ask for the master to wait until everything has been
                # updated. This can take a minute or so, and it's not a very good approach.
                # However, it's the way to see what would happen if we implemented ISGD exactly
                # without using any stale importance weights.
                #    wait_until_all_measurements_are_updated_by_workers(rsconn, "train", "importance_weight")
                logging.info("Error. Take the time to test wait_for_workers_to_update_all_the_importance_weights if you want to use it. I expect it to work, though.")

            elif next_action == "refresh_importance_weights":

                remote_redis_logger.log('event', "refresh_importance_weights")
                tic = time.time()
                _, D_importance_weights_and_more = get_raw_importance_weights(rsconn)

                (_, D_importance_weights_and_more, extra_statistics) = filter_raw_importance_weights(
                                                                            D_importance_weights_and_more,
                                                                            staleness_threshold_seconds=staleness_threshold_seconds,
                                                                            staleness_threshold_num_minibatches_master_processed=staleness_threshold_num_minibatches_master_processed,
                                                                            importance_weight_additive_constant=importance_weight_additive_constant,
                                                                            num_minibatches_master_processed=num_minibatches_master_processed)

                record_importance_weights_statistics(   D_importance_weights_and_more, extra_statistics,
                                            remote_redis_logger=remote_redis_logger, logging=logging,
                                            want_compute_entropy=True)

                toc = time.time()
                remote_redis_logger.log('timing_profiler', {'refresh_importance_weights' : (toc-tic)})
                #print "The master has obtained fresh importance weights."

            elif next_action == "process_minibatch":

                remote_redis_logger.log('event', "process_minibatch")
                #if A_importance_weights is None or nbr_of_usable_importance_weights is None:
                #    # nothing can be done here
                #    remote_redis_logger.log('event', "process_minibatch skipped")
                #    continue
                #else:
                #    remote_redis_logger.log('event', "process_minibatch")

                # Cause importance sampling to be done randomly if the value of
                # `turn_off_importance_sampling` is a floating-point value.
                # Note that another good approach would have been to alternate between
                # one mode and the other.
                if type(turn_off_importance_sampling) == float:
                    assert 0.0 <= turn_off_importance_sampling
                    assert turn_off_importance_sampling <= 1.0
                    if np.random.rand() <= turn_off_importance_sampling:
                        decision_to_turn_off_importance_sampling_this_iteration = True
                    else:
                        decision_to_turn_off_importance_sampling_this_iteration = False
                else:
                    assert type(turn_off_importance_sampling) == bool
                    decision_to_turn_off_importance_sampling_this_iteration = turn_off_importance_sampling

                tic = time.time()
                (intent, mode, A_sampled_indices, A_scaling_factors) = sample_indices_and_scaling_factors(
                        D_importance_weights_and_more=D_importance_weights_and_more,
                        extra_statistics=extra_statistics,
                        nbr_samples=master_minibatch_size,
                        master_usable_importance_weights_threshold_to_ISGD=master_usable_importance_weights_threshold_to_ISGD,
                        want_master_to_do_USGD_when_ISGD_is_not_possible=want_master_to_do_USGD_when_ISGD_is_not_possible,
                        turn_off_importance_sampling=decision_to_turn_off_importance_sampling_this_iteration)
                toc = time.time()
                remote_redis_logger.log('timing_profiler', {'sample_indices_and_scaling_factors' : (toc-tic)})

                if intent == 'wait_and_retry':
                    remote_redis_logger.log(['event', "Master does not have enough importance weights to do ISGD, and doesn't want to default to USGD. Sleeping for 2 seconds."])
                    time.sleep(2.0)
                    continue

                if intent == 'proceed':
                    remote_redis_logger.log('event', "Master proceeding with round of %s." % (mode,))
                    tic = time.time()

                    #if not np.all(np.isfinite(A_scaling_factors)):
                    #    import pdb; pdb.set_trace()

                    model_api.master_process_minibatch(A_sampled_indices, A_scaling_factors, "train")
                    toc = time.time()
                    remote_redis_logger.log('timing_profiler', {'master_process_minibatch' : (toc-tic), 'mode':mode})
                    logging.info("The master has processed a minibatch using %s." % mode)
                    num_minibatches_master_processed += 1


    remote_redis_logger.log('event', "Master exited from main loop")
    remote_redis_logger.close()
    quit()


# Stuff comment out that we might want to put back if we have more diagnostics to do.
    # new_gradient_norm_from_worker_after_update = model_api.worker_process_minibatch(A_sampled_indices, "train", ["importance_weight"])["importance_weight"]
    #
    # old_gradient_norm = get_importance_weights(rsconn, staleness_threshold=float('inf'), N=DD_config['database']['Ntrain'])
    #
    # if random.uniform(0,1) < 0.01:
    #     print "Master proceeding with round of %s at timestamp %f." % (mode, time.time())
    #     print "old,new pairs", zip(old_gradient_norm[0][A_sampled_indices].round(8).tolist(), new_gradient_norm_from_worker.round(8).tolist())
    #     print "old,new pairs after update", zip(old_gradient_norm[0][A_sampled_indices].round(8).tolist(), new_gradient_norm_from_worker_after_update.round(8).tolist())
    #     print "change in gradnorm after update", zip(new_gradient_norm_from_worker.round(8).tolist(), new_gradient_norm_from_worker_after_update.round(8).tolist())
    #     print "Average gradient norm in sampled minibatch", new_gradient_norm_from_worker.mean()
    #     print "Correlation between gradient norm from database and gradient norm computed on master", np.corrcoef(old_gradient_norm[0][A_sampled_indices], new_gradient_norm_from_worker)
    # break



# Extra debugging information for `sample_indices_and_scaling_factors`
# as it was first successfully written. This is documentation.
"""
nbr_samples = 10
A_importance_weights = np.array([0, 0, 5, 10, 0], np.float64)
nbr_of_present_importance_weights = 5

A_importance_weights = A_importance_weights.astype(np.float64)
p = A_importance_weights / A_importance_weights.sum()
A_sampled_indices_counts = np.random.multinomial(nbr_samples, p)
I = np.where(0 < A_sampled_indices_counts)[0]
A_sampled_indices = np.array(reduce(lambda x,y : x + y, [[i] * A_sampled_indices_counts[i] for i in I]))
A_unnormalized_scaling_factors = np.array([np.float64(1.0)/A_importance_weights[i] for i in A_sampled_indices])
Z = ( nbr_of_present_importance_weights / A_importance_weights.sum())
A_scaling_factors = (A_unnormalized_scaling_factors / Z).astype(np.float64)

>>> A_importance_weights
array([  0.,   0.,   5.,  10.,   0.])
>>> p
array([ 0.        ,  0.        ,  0.33333333,  0.66666667,  0.        ])
>>> A_sampled_indices_counts
array([0, 0, 3, 7, 0])
>>> I
array([2, 3])
>>> A_sampled_indices
array([2, 2, 2, 3, 3, 3, 3, 3, 3, 3])
>>> A_unnormalized_scaling_factors
array([ 0.2,  0.2,  0.2,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1])
>>> Z
0.33333333333333331
>>> A_scaling_factors
array([ 0.6,  0.6,  0.6,  0.3,  0.3,  0.3,  0.3,  0.3,  0.3,  0.3])
"""
