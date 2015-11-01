
import numpy as np
import time
import random

def get_importance_weights(rsconn, staleness_threshold):
    # Helper function for `sample_indices_and_scaling_factors`.

    # Note : There is a bit too much code here because this was written
    #        with the idea that not all the importance weights would be present
    #        in the database. In practice, we ended up filling up all the
    #        importance weights with a default value, so much of this section
    #        could be simplified.

    segment = "train"
    measurement = "importance_weight"

    L_indices = []
    L_importance_weights = []
    counter = 0

    numAccept = 0

    for (key, value) in rsconn.hgetall("H_%s_minibatch_%s" % (segment, measurement)).items():
        counter += 1

        #Here let's pull up the staleness records!  
        current_minibatch_indices_str = key
        timestamp = rsconn.hget("H_%s_minibatch_%s_measurement_last_update_timestamp" % (segment, measurement), current_minibatch_indices_str)

        #Master could always store map of timestamp -> minibatch index of master.  
        try:
            staleness = time.time() - float(timestamp)
        except:
            staleness = -1.0



        #Key refers to a list of indices.  
        #value refers to the associated importance weights.  
        #Now how do I get the timestamps!!!

        A_some_indices = np.fromstring(key, dtype=np.int32)
        A_some_importance_weights = np.fromstring(value, dtype=np.float32)
        #print "A_some_indices"
        #print A_some_indices
        #print "A_some_importance_weights"
        #print A_some_importance_weights
        assert A_some_indices.shape == A_some_importance_weights.shape, "Failed assertion that %s == %s." % (A_some_indices.shape, A_some_importance_weights.shape)

        if staleness < staleness_threshold:
            numAccept += 1
            L_indices.append(A_some_indices)
            L_importance_weights.append(A_some_importance_weights)
        else:
            if random.uniform(0,1) < 0.00001:
                print "REJECTING WITH STALENESS", staleness

    if random.uniform(0,1) < 0.01:
        print "% ACCEPT", numAccept * 1.0 / counter
        print "Num Accept", numAccept
        print "Counter", counter

    #print 'DEBUG rsconn.hgetall("H_%s_minibatch_%s") returned %d entries' % (segment, measurement, counter)

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

def sample_indices_and_scaling_factors(rsconn, nbr_samples, want_master_to_wait_for_all_importance_weights_to_be_present, staleness_threshold):

    A_importance_weights, nbr_of_present_importance_weights = get_importance_weights(rsconn, staleness_threshold)

    ratio_of_finite_importance_weights = np.isfinite(A_importance_weights).mean()

    if want_master_to_wait_for_all_importance_weights_to_be_present and ratio_of_finite_importance_weights < 1.0:
        print "Master has only %f of the importance weights. Waiting for all of them to be present." % ratio_of_finite_importance_weights
        return ('wait_and_retry', None, None)
    else:
        A_sampled_indices, A_scaling_factors = recipe1(A_importance_weights, nbr_of_present_importance_weights, nbr_samples)
        return ('proceed', A_sampled_indices, A_scaling_factors)



def recipe1(A_importance_weights, nbr_of_present_importance_weights, nbr_samples):

    # A_importance_weights, nbr_of_present_importance_weights = get_importance_weights(rsconn)

    # Note that these are sampled with replacements, which
    # is what we want but which also goes against the traditional
    # traversal of the training set through a minibatch iteration scheme.

    # Note : If you plan on changing anything in this function,
    #        then you will need to update the documentation at the
    #        botton of this document.

    if A_importance_weights.sum() < 1e-16:
        #print "All the importance_weight are zero. There is nothing to be done with this."
        #print "The only possibility is to report them to be as though they were all 1.0."
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


def recipe2(A_importance_weights, nbr_of_present_importance_weights, nbr_samples):
    # If you want to add another method, you can do it here.
    pass
