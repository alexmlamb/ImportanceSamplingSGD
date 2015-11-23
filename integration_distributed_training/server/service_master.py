
import redis
import numpy as np
import json
import time
import re

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
    staleness_threshold = DD_config['database']['staleness_threshold_seconds']
    want_master_to_do_USGD_when_ISGD_is_not_possible = DD_config['database'].get('want_master_to_do_USGD_when_ISGD_is_not_possible', True)
    master_usable_importance_weights_threshold_to_ISGD = DD_config['database']['master_usable_importance_weights_threshold_to_ISGD']

    model_api = ModelAPI(DD_config['model'])

    if not want_only_indices_for_master:
        print "Error. At the current time we support only the of feeding data to the master through indices (instead of actual data)."
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
    #     importance weights for every minibatch on the workers.
    #
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

    nbr_batch_processed_per_public_parameter_update = DD_config['database']['nbr_batch_processed_per_public_parameter_update']
    # TODO : Might make this stochastic, but right now it's just
    #        a bunch of iterations.

    if DD_config['database']['use_yoshuas_trick'] and DD_config['model']['turn_off_importance_sampling']:
        raise Exception("Importance sampling turned off, but using Yoshuas trick for importance sampling.  This is a contradiction")

    queue_name = "L_master_train_minibatch_indices_and_info_QUEUE"

    num_minibatches_master_processed = 0

    while True:

        # Task (1)

        tic = time.time()

        if serialized_parameters_format == "opaque_string":
            current_parameters_str = model_api.get_serialized_parameters()
        elif serialized_parameters_format == "ndarray_float32_tostring":
            current_parameters_str = model_api.get_serialized_parameters().tostring(order='C')
        else:
            print "Fatal error : invalid serialized_parameters_format : %s." % serialized_parameters_format
            quit()

        rsconn.set("num_minibatches_master_processed", num_minibatches_master_processed)
        rsconn.set("parameters:current", current_parameters_str)
        rsconn.set("parameters:current_timestamp", time.time())
        # potentially not used
        rsconn.set("parameters:current_datestamp", time.strftime("%Y-%m-%d %H:%M:%S"))
        toc = time.time()
        print "The master has updated the parameters. It took %f seconds to send to the database." % (toc - tic,)

        # This next line is to ask for the master to wait until everything has been
        # updated. This can take a minute or so, and it's not a very good approach.
        # However, it's the way to see what would happen if we implemented ISGD exactly
        # without using any stale importance weights.
        
        #wait_until_all_measurements_are_updated_by_workers(rsconn, "train", "importance_weight")

        # Task (2)

        for _ in range(nbr_batch_processed_per_public_parameter_update):

            tic = time.time()

            while True:
                t0 = time.time()
                (intent, mode, A_sampled_indices, A_scaling_factors) = sample_indices_and_scaling_factors(rsconn,
                    master_minibatch_size,
                    staleness_threshold=staleness_threshold,
                    master_usable_importance_weights_threshold_to_ISGD=master_usable_importance_weights_threshold_to_ISGD,
                    want_master_to_do_USGD_when_ISGD_is_not_possible=want_master_to_do_USGD_when_ISGD_is_not_possible,
                    Ntrain=Ntrain,
                    importance_weight_additive_constant=DD_config["model"]["importance_weight_additive_constant"],
                    turn_off_importance_sampling=DD_config["model"]["turn_off_importance_sampling"])

                if random.uniform(0,1) < 0.01:
                    print "Time to do sampling", time.time() - t0

                # Note from Guillaume : This should probably instead be a call to
                # get_mean_variance_measurement_on_database(rsconn, "train", "accuracy")
                # get_mean_variance_measurement_on_database(rsconn, "test", "accuracy")
                #

                if random.uniform(0,1) < 0.01:
                    print "scaling factors", A_scaling_factors



                num_minibatches_master_processed += 1

                if intent == 'wait_and_retry':
                    print "Master does not have enough importance weights to do ISGD, and doesn't want to default to USGD."
                    print "Sleeping for 2s."
                    time.sleep(2.0)
                    # this "continue" call will retry fetching the importance weights
                    # and we'll be stuck here forever if the importance weights never
                    # become all non-Nan.
                    continue

                if intent == 'proceed':

                    new_gradient_norm_from_worker = model_api.worker_process_minibatch(A_sampled_indices, "train", ["importance_weight"])["importance_weight"]

                    debug_this_section = True
                    if not debug_this_section:

                        print "Master proceeding with round of %s at timestamp %f." % (mode, time.time())
                        t0 = time.time()
                        model_api.master_process_minibatch(A_sampled_indices, A_scaling_factors, "train")
                        if random.uniform(0,1) < 0.01:
                            print time.time() - t0, "time to call master process minibatch"
                        # breaking will continue to the main looping section
                        break
                    else:
                        pass
                        # This is a debugging section that will be removed eventually.
                        # This section has a problem. This has been discussed.
                        # Until something useful is done here, Guillaume commented it out.

                        if DD_config['database']['use_yoshuas_trick']:
                            #Recompute gradients for selected instances.  
                            #Making scaling factors 1 / this.  

                            new_gradient_norm = model_api.worker_process_minibatch(A_sampled_indices, "train", ["importance_weight"])["importance_weight"]

                            old_gradient_norm = get_importance_weights(rsconn, staleness_threshold=float('inf'), N=DD_config['database']['Ntrain'])[0][A_sampled_indices]


                            A_scaling_factors = A_scaling_factors * (DD_config['model']['importance_weight_additive_constant'] + old_gradient_norm) / (new_gradient_norm + 0.0001)

                        model_api.master_process_minibatch(A_sampled_indices, A_scaling_factors, "train")
                        # breaking will continue to the main looping section


                        if random.uniform(0,1) < 0.005:

                            new_gradient_norm_from_worker_after_update = model_api.worker_process_minibatch(A_sampled_indices, "train", ["importance_weight"])["importance_weight"]

                            old_gradient_norm = get_importance_weights(rsconn, staleness_threshold=float('inf'), N=DD_config['database']['Ntrain'])

                            print "Master proceeding with round of %s at timestamp %f." % (mode, time.time())
                            print "old,new pairs", zip(old_gradient_norm[0][A_sampled_indices].round(8).tolist(), new_gradient_norm_from_worker.round(8).tolist())
                            print "old,new pairs after update", zip(old_gradient_norm[0][A_sampled_indices].round(8).tolist(), new_gradient_norm_from_worker_after_update.round(8).tolist())
                            print "change in gradnorm after update", zip(new_gradient_norm_from_worker.round(8).tolist(), new_gradient_norm_from_worker_after_update.round(8).tolist())
                            print "Average gradient norm in sampled minibatch", new_gradient_norm_from_worker.mean()
                            print "Correlation between gradient norm from database and gradient norm computed on master", np.corrcoef(old_gradient_norm[0][A_sampled_indices], new_gradient_norm_from_worker)
                        break


            

            toc = time.time()
            if random.uniform(0,1) < 0.01:
                print "The master has processed one minibatch. It took %f seconds." % (toc - tic,)

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
