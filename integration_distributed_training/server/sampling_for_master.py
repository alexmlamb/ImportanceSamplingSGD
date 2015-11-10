
import numpy as np
import time
import random

def get_importance_weights(rsconn, staleness_threshold=None, importance_weight_additive_constant=None, N=None):
    # Helper function for `sample_indices_and_scaling_factors`.

    # Note : There is a bit too much code here because this was written
    #        with the idea that not all the importance weights would be present
    #        in the database.
    #        In practice, we ended up filling up all the importance weights
    #        with a default value, so this might a little too complicated.
    #        However, we also omit those that are "stale", so it's useful
    #        to have code that handles this.
    #
    #  When `staleness_threshold` is None, it's ignored.
    #  With `importance_weight_additive_constant` we allow a constant to be
    #  added to all the present and non-stale importance weights.
    #
    # The `N` argument is optional. When specified, it tells this method ahead of time
    # what shape (N,) should the resulting arrays be.
    #
    # Note that this method filters out importance weights that are NaN.

    segment = "train"
    measurement = "individual_importance_weight"

    L_indices = []
    L_importance_weights = []

    nbr_accepted = 0

    db_list_name = "L_workers_%s_minibatch_indices_ALL" % segment
    db_hash_name = "H_%s_minibatch_%s" % (segment, measurement)
    nbr_minibatches = rsconn.llen(db_list_name)
    assert 0 < nbr_minibatches

    for i in range(nbr_minibatches):
        current_minibatch_indices_str = rsconn.lindex(db_list_name, i)

        #Here let's pull up the staleness records!
        timestamp_str = rsconn.hget("H_%s_minibatch_%s_measurement_last_update_timestamp" % (segment, measurement), current_minibatch_indices_str)

        if timestamp_str is None or len(timestamp_str) == 0:
            print "ERROR. There is a bug somewhere because there is never a situation where a measurement can be present without a timestamp."
            print "We can easily recover from this error by just accepting the importance weight anyway, but we would be sweeping a bug under the rug by doing so."
            quit()

        #Master could always store map of timestamp -> minibatch index of master.
        staleness = time.time() - float(timestamp_str)

        #Key refers to a list of indices.
        #value refers to the associated importance weights.
        #Now how do I get the timestamps!!!

        value_str = rsconn.hget(db_hash_name, current_minibatch_indices_str)
        A_some_indices = np.fromstring(current_minibatch_indices_str, dtype=np.int32)
        A_some_importance_weights = np.fromstring(value_str, dtype=np.float32)

        # This is a relaxation of the importance sampling optimal sampling proposal.
        if (importance_weight_additive_constant is not None and
            np.isfinite(importance_weight_additive_constant) and
            0.0 <= importance_weight_additive_constant):
             A_some_importance_weights = A_some_importance_weights + importance_weight_additive_constant

        assert A_some_indices.shape == A_some_importance_weights.shape, "Failed assertion that %s == %s." % (A_some_indices.shape, A_some_importance_weights.shape)

        if staleness_threshold is None or staleness <= staleness_threshold:
            nbr_accepted += 1
            L_indices.append(A_some_indices)
            L_importance_weights.append(A_some_importance_weights)
        else:
            # This is okay during development, but it's not a nice way to
            # monitor a quantity.
            #if random.uniform(0,1) < 0.00001:
            #    print "REJECTING WITH STALENESS", staleness
            pass

    #if random.uniform(0,1) < 0.01:
    #    print "Accepted %d / %d = %f of importance weights minibatches. " % (nbr_accepted, nbr_minibatches, nbr_accepted * 1.0 / nbr_minibatches)

    if len(L_indices) == 0:
        # All the importance weights are stale.
        return (None, 0)

    #print 'DEBUG rsconn.hgetall("H_%s_minibatch_%s") returned %d entries' % (segment, measurement, counter)

    A_unsorted_indices = np.hstack(L_indices)
    A_unsorted_importance_weights = np.hstack(L_importance_weights)
    assert A_unsorted_indices.shape == A_unsorted_importance_weights.shape

    # Weed out the values that are np.nan or other np.inf.
    I = np.isfinite(A_unsorted_importance_weights)
    if I.sum() == 0:
        # `I` contains boolean values, so summing them and comparing to 0.0
        # is equivalent to making sure that they're all False.
        #
        # There are no valid importance weights.
        return (None, 0)
    A_unsorted_indices = A_unsorted_indices[I]
    A_unsorted_importance_weights = A_unsorted_importance_weights[I]

    # Find out how large you have to make your array.
    # You need to add +1 because the largest index has to represent a valid index.
    if N is not None:
        assert A_unsorted_indices.max() + 1 <= N
    else:
        N = A_unsorted_indices.max() + 1
    nbr_of_present_importance_weights = A_unsorted_indices.shape[0]
    A_importance_weights = np.zeros((N,), dtype=np.float32)
    A_importance_weights[A_unsorted_indices] = A_unsorted_importance_weights
    #return (A_indices, A_importance_weights)
    return A_importance_weights, nbr_of_present_importance_weights


def sample_indices_and_scaling_factors( A_importance_weights,
                                        nbr_of_usable_importance_weights,
                                        nbr_samples,
                                        master_usable_importance_weights_threshold_to_ISGD=None,
                                        want_master_to_do_USGD_when_ISGD_is_not_possible=True,
                                        Ntrain=None):

    # The reason why we want to pass `Ntrain` to this function is because we might
    # want to make decisions based on the number of present importance weights.

    if master_usable_importance_weights_threshold_to_ISGD is not None or want_master_to_do_USGD_when_ISGD_is_not_possible:
        assert Ntrain is not None, "Ntrain has to be specified to sample_indices_and_scaling_factors when we use master_usable_importance_weights_threshold_to_ISGD or want_master_to_do_USGD_when_ISGD_is_not_possible."

    # Try to do ISGD before trying anything else..
    if master_usable_importance_weights_threshold_to_ISGD is not None:
        ratio_of_usable_importance_weights = nbr_of_usable_importance_weights * 1.0 / Ntrain
        if master_usable_importance_weights_threshold_to_ISGD <= ratio_of_usable_importance_weights:
            print "Master has a ratio of usable importance weights %d / %d = %f which meets the required threshold of %f." % (nbr_of_usable_importance_weights, Ntrain, ratio_of_usable_importance_weights, master_usable_importance_weights_threshold_to_ISGD)
            A_sampled_indices, A_scaling_factors = recipe1(A_importance_weights, nbr_of_usable_importance_weights, nbr_samples)
            return ('proceed', 'ISGD', A_sampled_indices, A_scaling_factors)
        else:
            print "Master has a ratio of usable importance weights %f which falls shorts of the required threshold of %f." % (ratio_of_usable_importance_weights, master_usable_importance_weights_threshold_to_ISGD)


    # So, we're not going to do ISGD, but maybe we still want to do USGD.
    # In that particular case, we don't care at all about how stale the values could be,
    # or not even about whether the importance weights are present or not.
    if want_master_to_do_USGD_when_ISGD_is_not_possible:
        A_sampled_indices = np.random.permutation(Ntrain)[0:nbr_samples]
        A_scaling_factors = np.ones(A_sampled_indices.shape)
        return ('proceed', 'USGD', A_sampled_indices, A_scaling_factors)
    else:
        return ('wait_and_retry', None, None, None)


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

    #A_unnormalized_scaling_factors = np.array([np.float64(1.0)/A_importance_weights[i] for i in A_sampled_indices])
    A_unnormalized_scaling_factors = A_unnormalized_scaling_factors = (np.float64(1.0) / A_importance_weights[A_sampled_indices]).astype(np.float64)

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
