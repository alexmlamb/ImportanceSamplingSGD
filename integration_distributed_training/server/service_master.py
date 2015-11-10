
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
from sampling_for_master import sample_indices_and_scaling_factors, get_importance_weights

from common import get_rsconn_with_timeout, wait_until_all_measurements_are_updated_by_workers

import integration_distributed_training.server.logger

def run(DD_config, D_server_desc):

    if D_server_desc['hostname'] in ["szkmbp"]:
        D_server_desc['hostname'] = "localhost"

    rsconn = get_rsconn_with_timeout(D_server_desc['hostname'], D_server_desc['port'], D_server_desc['password'],
                                     timeout=60, wait_for_parameters_to_be_present=False)

    L_measurements = DD_config['database']['L_measurements']
    want_only_indices_for_master = DD_config['database']['want_only_indices_for_master']
    master_minibatch_size = DD_config['database']['master_minibatch_size']
    serialized_parameters_format = DD_config['database']['serialized_parameters_format']
    Ntrain = DD_config['database']['Ntrain']
    # Default behavior is to have no staleness, and perform ISGD from the moment that we
    # get values for all the importance weights. Until then, we do USGD.
    staleness_threshold = DD_config['database'].get('staleness_threshold', None)
    want_master_to_do_USGD_when_ISGD_is_not_possible = DD_config['database'].get('want_master_to_do_USGD_when_ISGD_is_not_possible', True)
    master_usable_importance_weights_threshold_to_ISGD = DD_config['database'].get('master_usable_importance_weights_threshold_to_ISGD', 1.0)
    master_routine = DD_config['model']['master_routine']
    if master_routine[0] != "sync_params":
        print "Error. Your master_routine should always start with 'sync_params'."
        print master_routine
        quit()

    importance_weight_additive_constant = DD_config['database'].get('importance_weight_additive_constant', None)

    logger = integration_distributed_training.server.logger.RedisLogger(rsconn, queue_prefix_identifier="service_master")

    model_api = ModelAPI(DD_config['model'])

    if not want_only_indices_for_master:
        print "Error. At the current time we support only the of feeding data to the master through indices (instead of actual data)."
        quit()

    # Run just a simple test to make sure that the importance weights have been
    # set to something. In theory, there should always be valid values in there,
    # so this is just a sanity check.
    segment = "train"
    measurement = "individual_importance_weight"
    nbr_of_present_importance_weights = rsconn.hlen("H_%s_minibatch_%s" % (segment, measurement))
    assert 0 < nbr_of_present_importance_weights, "Error. The database should have been set up to have dummy importance weights at least."
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


    num_minibatches_master_processed = 0

    # The main loop runs until the user hits CTLR+C or until
    # the Helios cluster sends the SIGTERM to that process
    # five minutes before the end of training.
    def signal_handler(signal, frame):
        print "SIGTERM received for the first time."
        print "Will break from master main loop."
        print "Will make logger sync to database before terminating."
        print ""
        signal_handler.logger.log('event', "Received SIGTERM.")
        signal_handler.logger.close()
        sys.exit(0)
    #
    signal.signal(signal.SIGINT, signal_handler)
    # I'm forced to use weird function properties because python
    # has stupid scoping rules.
    signal_handler.logger = logger

    # cache those values to use them for more than one computation
    A_importance_weights = None
    nbr_of_usable_importance_weights = 0

    logger.log('event', "Before entering service_master main loop.")
    while True:

        for next_action in master_routine:
            assert next_action in [ "sync_params", "refresh_importance_weights", "process_minibatch", # the normal ones
                                    "wait_for_workers_to_update_all_the_importance_weights"]          # the special ones

            # TODO : Have a clause that governs resuming from the database in which
            #        we would not sync_params, but rather read from the database.
            if next_action == "sync_params":

                logger.log('event', "sync_params")
                tic = time.time()
                if serialized_parameters_format == "opaque_string":
                    current_parameters_str = model_api.get_serialized_parameters()
                elif serialized_parameters_format == "ndarray_float32_tostring":
                    current_parameters_str = model_api.get_serialized_parameters().tostring(order='C')
                else:
                    print "Fatal error : invalid serialized_parameters_format : %s." % serialized_parameters_format
                    quit()
                toc = time.time()
                logger.log('timing_profiler', {'read_parameters_from_model' : (toc-tic)})

                tic = time.time()
                rsconn.set("parameters:current", current_parameters_str)
                rsconn.set("parameters:current_timestamp", time.time())
                # potentially not used
                rsconn.set("parameters:current_datestamp", time.strftime("%Y-%m-%d %H:%M:%S"))
                toc = time.time()
                logger.log('timing_profiler', {'sync_params_to_database' : (toc-tic)})
                print "The master has updated the parameters."

            elif next_action == "wait_for_workers_to_update_all_the_importance_weights":

                # This next line is to ask for the master to wait until everything has been
                # updated. This can take a minute or so, and it's not a very good approach.
                # However, it's the way to see what would happen if we implemented ISGD exactly
                # without using any stale importance weights.
                #    wait_until_all_measurements_are_updated_by_workers(rsconn, "train", "importance_weight")
                print "Error. Take the time to test wait_for_workers_to_update_all_the_importance_weights if you want to use it. I expect it to work, though."

            elif next_action == "refresh_importance_weights":

                logger.log('event', "refresh_importance_weights")
                tic = time.time()
                A_importance_weights, nbr_of_usable_importance_weights = get_importance_weights(rsconn, staleness_threshold, importance_weight_additive_constant)
                toc = time.time()
                logger.log('timing_profiler', {'refresh_importance_weights' : (toc-tic)})
                #print "The master has obtained fresh importance weights."

            elif next_action == "process_minibatch":

                logger.log('event', "process_minibatch")
                #if A_importance_weights is None or nbr_of_usable_importance_weights is None:
                #    # nothing can be done here
                #    logger.log('event', "process_minibatch skipped")
                #    continue
                #else:
                #    logger.log('event', "process_minibatch")

                tic = time.time()
                (intent, mode, A_sampled_indices, A_scaling_factors) = sample_indices_and_scaling_factors(A_importance_weights=A_importance_weights,
                        nbr_of_usable_importance_weights=nbr_of_usable_importance_weights,
                        nbr_samples=master_minibatch_size,
                        master_usable_importance_weights_threshold_to_ISGD=master_usable_importance_weights_threshold_to_ISGD,
                        want_master_to_do_USGD_when_ISGD_is_not_possible=want_master_to_do_USGD_when_ISGD_is_not_possible,
                        Ntrain=Ntrain)
                toc = time.time()
                logger.log('timing_profiler', {'sample_indices_and_scaling_factors' : (toc-tic)})

                num_minibatches_master_processed += 1

                if intent == 'wait_and_retry':
                    logger.log(['event', "Master does not have enough importance weights to do ISGD, and doesn't want to default to USGD. Sleeping for 2 seconds."])
                    time.sleep(2.0)
                    continue

                if intent == 'proceed':
                    logger.log('event', "Master proceeding with round of %s." % (mode,))
                    tic = time.time()
                    model_api.master_process_minibatch(A_sampled_indices, A_scaling_factors, "train")
                    toc = time.time()
                    logger.log('timing_profiler', {'master_process_minibatch' : (toc-tic), 'mode':mode})
                    print "The master has processed a minibatch."

    logger.log('event', "Master exited from main loop")
    logger.close()
    quit()



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
