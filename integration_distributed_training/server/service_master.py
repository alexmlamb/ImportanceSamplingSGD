
import redis
import numpy as np
import json
import time
import re

import sys, os
import getopt

# Change this to the real model once you want to plug it in.
from integration_distributed_training.model.mocked_model import ModelAPI


from common import get_rsconn_with_timeout



def get_importance_weights(rsconn):

    # Helper function for `sample_indices_and_scaling_factors`.

    segment = "train"
    measurement = "importance_weight"

    L_indices = []
    L_importance_weights = []
    for (key, value) in rsconn.hgetall("H_%s_minibatch_%s" % (segment, measurement)).items():
        A_some_indices = np.fromstring(key, dtype=np.int32)
        A_some_importance_weights = np.fromstring(value, dtype=np.float32)
        #print "key"
        #print
        print "A_some_indices"
        print A_some_indices
        print "A_some_importance_weights"
        print A_some_importance_weights
        assert A_some_indices.shape == A_some_importance_weights.shape, "Failed assertion that %s == %s." % (A_some_indices.shape, A_some_importance_weights.shape)
        L_indices.append(A_some_indices)
        L_importance_weights.append(A_some_importance_weights)

    A_unsorted_indices = np.hstack(L_indices)
    A_unsorted_importance_weights = np.hstack(L_importance_weights)
    assert A_unsorted_indices.shape == A_unsorted_importance_weights.shape

    # Find out how large you have to make your array.
    # You need to add +1 because the largest index has to represent a valid index.
    # There is no harm in using a larger value of N than necessary.
    N = A_unsorted_indices.max() + 1
    nbr_of_present_importance_weights = A_unsorted_indices.shape[0]
    A_importance_weights = np.zeros((N,), dtype=np.float32)
    A_importance_weights[A_unsorted_indices] = A_unsorted_importance_weights
    #return (A_indices, A_importance_weights)
    return A_importance_weights, nbr_of_present_importance_weights

def sample_indices_and_scaling_factors(rsconn, nbr_samples):
    # Note that these are sampled with repetitions, which
    # is what we want but which also goes against the traditional
    # traversal of the training set through a minibatch iteration scheme.

    # Note : If you plan on changing anything in this function,
    #        then you will need to update the documentation at the
    #        botton of this document.

    A_importance_weights, nbr_of_present_importance_weights = get_importance_weights(rsconn)

    if A_importance_weights.sum() < 1e-16:
        print "All the importance_weight are zero. There is nothing to be done with this."
        print "The only possibility is to report them to be as though they were all 1.0."
        #import pdb; pdb.set_trace()
        A_sampled_indices = np.random.randint(low=0, high=A_importance_weights.shape[0], size=nbr_samples).astype(np.int32)
        return A_sampled_indices, np.ones(A_sampled_indices.shape, dtype=np.float64)

    # You can get complaints from np.random.multinomial if you are in float32
    # because rounding errors can bring your sum() to a little above 1.0.
    A_importance_weights = A_importance_weights.astype(np.float64)
    p = A_importance_weights / A_importance_weights.sum()

    A_sampled_indices_counts = np.random.multinomial(nbr_samples, p)
    # Find out where the non-zero values are.
    I = np.where(0 < A_sampled_indices_counts)[0]


    # For each such non-zero value, we need to repeat that index
    # a corresponding number of times (and then concatenate everything).
    A_sampled_indices = np.array(reduce(lambda x,y : x + y, [[i] * A_sampled_indices_counts[i] for i in I]))

    A_unnormalized_scaling_factors = np.array([np.float64(1.0)/A_importance_weights[i] for i in A_sampled_indices])

    # You could argue that we want to divide this by `nbr_samples`,
    # but it depends on how you negociate the role of the minibatch size
    # in the loss function.
    #
    # Since the scaling factors will end up being used in training and each
    # attributed to one point from the minibatch, then we probably don't want
    # to include `nbr_samples` in any way.
    #
    # Basically, if we had uniform importance weights, here we would be
    # multiplying by Ntrain and dividing by Ntrain. We are doing the equivalent
    # of that for importance sampling.
    #
    # Another thing worth noting is that we could basically return
    #     A_importance_weights.mean() / A_importance_weights
    # but then we would not be taking into consideration the situation in which
    # only some of the importance weights were specified and many were missing.
    # Maybe this is not necessary, though.
    Z = ( nbr_of_present_importance_weights / A_importance_weights.sum())
    A_scaling_factors = (A_unnormalized_scaling_factors / Z).astype(np.float64)

    return A_sampled_indices, A_scaling_factors


def run(DD_config, D_server_desc):

    # TODO : Get rid of this cheap hack to circumvent my OSX's inability to see itself.
    D_server_desc['hostname'] = "localhost"

    rsconn = get_rsconn_with_timeout(D_server_desc['hostname'], D_server_desc['port'], D_server_desc['password'],
                                     timeout=60, wait_for_parameters_to_be_present=False)

    L_measurements = DD_config['database']['L_measurements']
    want_only_indices_for_master = DD_config['database']['want_only_indices_for_master']
    master_minibatch_size = DD_config['database']['master_minibatch_size']
    serialized_parameters_format = DD_config['database']['serialized_parameters_format']

    model_api = ModelAPI(DD_config['model'])

    if not want_only_indices_for_master:
        print "Error. At the current time we support only the of feeding data to the master through indices (instead of actual data)."
        quit()

    # Run just a simple test to make sure that the importance weights have been
    # set to something. In theory, there should always be valid values in there,
    # so this is just a sanity check.
    segment = "train"
    measurement = "importance_weight"
    nbr_of_present_importance_weights = rsconn.hlen("H_%s_minibatch_%s" % (segment, measurement))
    assert 0 < nbr_of_present_importance_weights, "Error. The database should have been set up to have dummy importance weights at least."
    print "Master found %d importance weights in the database." % nbr_of_present_importance_weights


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

    nbr_batch_processed_per_public_parameter_update = 32
    # TODO : Might make this stochastic, but right now it's just
    #        a bunch of iterations.

    # Pop values from the left, because the assistant
    # is pushing fresh values to the right.
    # TODO : Ponder whether we should nevertheless pop and push
    #        from the same side, and the assistant would remove outdated
    #        values from the other side. This is about tradeoffs.
    queue_name = "L_master_train_minibatch_indices_and_info_QUEUE"

    while True:

        # Task (1)

        if serialized_parameters_format == "opaque_string":
            current_parameters_str = model_api.get_serialized_parameters()
        elif serialized_parameters_format == "ndarray_float32_tostring":
            current_parameters_str = model_api.get_serialized_parameters().tostring(order='C')
        else:
            print "Fatal error : invalid serialized_parameters_format : %s." % serialized_parameters_format
            quit()

        rsconn.set("parameters:current", current_parameters_str)
        rsconn.set("parameters:current_timestamp", time.time())
        # potentially not used
        rsconn.set("parameters:current_datestamp", time.strftime("%Y-%m-%d %H:%M:%S"))
        print "The master has updated the parameters."


        # Task (2)

        for _ in range(nbr_batch_processed_per_public_parameter_update):
            (A_sampled_indices, A_scaling_factors) = sample_indices_and_scaling_factors(rsconn, master_minibatch_size)
            model_api.master_process_minibatch(A_sampled_indices, A_scaling_factors, "train")
            print "The master has processed a minibatch."




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
