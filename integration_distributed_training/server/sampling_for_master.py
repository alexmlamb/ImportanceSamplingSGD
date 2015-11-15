
import numpy as np
import time
import random

def get_raw_importance_weights(rsconn):

    # Often used as helper function to get what is
    # required for `sample_indices_and_scaling_factors`
    # and for `filter_importance_weights`.

    segment = "train"
    measurement = "individual_importance_weight"

    L_indices = []
    L_importance_weights = []
    L_abs_update_timestamp = []
    L_delay_update_master_parameter = []
    L_abs_update_master_minibatch = []

    db_list_name = "L_workers_%s_minibatch_indices_ALL" % segment
    db_hash_name = "H_%s_minibatch_%s" % (segment, measurement)
    nbr_minibatches = rsconn.llen(db_list_name)
    assert 0 < nbr_minibatches

    def f(database_name_hash_pattern, current_minibatch_indices_str, shape):
        # ex :   database_name_hash_pattern = "H_%s_minibatch_%s_measurement_last_update_timestamp"
        # abs_update_timestamp_str = rsconn.hget("H_%s_minibatch_%s_measurement_last_update_timestamp" % (segment, measurement), current_minibatch_indices_str)
        A_str = rsconn.hget(database_name_hash_pattern % (segment, measurement), current_minibatch_indices_str)
        if A_str is not None and len(A_str) != 0:
            A = (float(A_str) * np.ones(shape)).astype(np.float64)
        else:
            A = (np.nan * np.ones(shape)).astype(np.float64)
        return A


    for i in range(nbr_minibatches):
        current_minibatch_indices_str = rsconn.lindex(db_list_name, i)

        # This part of the code assumes that the database will always contain
        # an entry for each of the importance weights, since they should be
        # initialized to a default value in all cases.
        value_str = rsconn.hget(db_hash_name, current_minibatch_indices_str)
        A_some_indices = np.fromstring(current_minibatch_indices_str, dtype=np.int32)
        A_some_importance_weights = np.fromstring(value_str, dtype=np.float32).astype(np.float64)
        assert A_some_indices.shape == A_some_importance_weights.shape, "Failed assertion that %s == %s." % (A_some_indices.shape, A_some_importance_weights.shape)

        # Pull up the staleness records and other timing measurements.
        A_some_abs_update_timestamp              = f("H_%s_minibatch_%s_measurement_last_update_timestamp",                     current_minibatch_indices_str, A_some_indices.shape)
        A_some_delay_update_master_parameter = f("H_%s_minibatch_%s_delay_between_measurement_update_and_parameter_update", current_minibatch_indices_str, A_some_indices.shape)
        A_some_abs_update_master_minibatch       = f("H_%s_minibatch_%s_measurement_num_minibatches_master_processed",          current_minibatch_indices_str, A_some_indices.shape)

        L_indices.append(A_some_indices)
        L_importance_weights.append(A_some_importance_weights)
        L_abs_update_timestamp.append(A_some_abs_update_timestamp)
        L_delay_update_master_parameter.append(A_some_delay_update_master_parameter)
        L_abs_update_master_minibatch.append(A_some_abs_update_master_minibatch)

    A_unsorted_indices = np.hstack(L_indices)
    A_unsorted_importance_weights = np.hstack(L_importance_weights)
    A_unsorted_abs_update_timestamp = np.hstack(L_abs_update_timestamp)
    A_unsorted_delay_update_master_parameter = np.hstack(L_delay_update_master_parameter)
    A_unsorted_abs_update_master_minibatch = np.hstack(L_abs_update_master_minibatch)

    assert A_unsorted_indices.shape == A_unsorted_importance_weights.shape
    assert A_unsorted_indices.shape == A_unsorted_importance_weights.shape
    assert A_unsorted_indices.shape == A_unsorted_abs_update_timestamp.shape
    assert A_unsorted_indices.shape == A_unsorted_delay_update_master_parameter.shape
    assert A_unsorted_indices.shape == A_unsorted_abs_update_master_minibatch.shape

    # You can trust this value of `N` because we operate under the assumption
    # that all the weights are going to be present.
    N = A_unsorted_indices.max() + 1

    D_results = {'N':N}
    # That array `I` is the key to sort everything back into the original order.
    I = A_unsorted_indices
    for (k, A_unsorted) in [('importance_weight', A_unsorted_importance_weights),
                            ('abs_update_timestamp', A_unsorted_abs_update_timestamp),
                            ('delay_update_master_parameter', A_unsorted_delay_update_master_parameter),
                            ('abs_update_master_minibatch', A_unsorted_abs_update_master_minibatch)]:
        A_sorted = np.zeros((N,), dtype=np.float64)
        A_sorted[I] = A_unsorted
        D_results[k] = A_sorted

    # Keep a simple interface and return the importance weights in the first argout.
    # The dictionary goes in the second argout for users who want to get more info.
    return D_results['importance_weight'], D_results



def filter_raw_importance_weights(  D_importance_weight_and_more,
                                    staleness_threshold_seconds=None,
                                    staleness_threshold_num_minibatches_master_processed=None,
                                    importance_weight_additive_constant=None,
                                    num_minibatches_master_processed=None):

    # Helper function for `sample_indices_and_scaling_factors`.
    # Takes the output from `get_raw_importance_weights`
    # as the `D_importance_weight_and_more` argument.
    # It mutates that dictionary and adds other useful fields,
    # so it's practically how this methods produces an output.
    #
    # When `staleness_threshold_seconds` is None, it's ignored.
    # When `staleness_threshold_num_minibatches_master_processed` is None, it's ignored.
    # With both present, we apply the strictest conditions.
    #
    # With `importance_weight_additive_constant` we allow a constant to be
    # added to all the present and non-stale importance weights.
    # Since all the invalid/ususable importance weights will end up being np.nan or ignored
    # it doesn't matter whether this gets added to those weights too.
    #
    # The `num_minibatches_master_processed` is required when we want
    # to impose a threshold that depends on `staleness_threshold_num_minibatches_master_processed`.
    # Otherwise, we cannot tell what is the current reference in order
    # to apply the threshold.


    extra_statistics = {}

    # Start with a full array `I` of boolean values indicating that we want the
    # corresponding elements, and progressively reduce that by applying each
    # constraint sequentially.
    I = np.isfinite(D_importance_weight_and_more['importance_weight'])
    extra_statistics['importance_weights:ratio_satisfying:finite'] = I.mean()

    if staleness_threshold_seconds is not None:
        now = time.time()
        I2 = (now <= D_importance_weight_and_more['abs_update_timestamp'] + staleness_threshold_seconds )
        extra_statistics['importance_weights:ratio_satisfying:staleness_threshold_seconds'] = I2.mean()
        I = I * I2

    if staleness_threshold_num_minibatches_master_processed is not None:
        assert num_minibatches_master_processed is not None, "Error. To use `staleness_threshold_num_minibatches_master_processed`, you need to have a value of `num_minibatches_master_processed` that is not None."
        I3 = (num_minibatches_master_processed <= D_importance_weight_and_more['abs_update_master_minibatch'] + staleness_threshold_num_minibatches_master_processed )
        extra_statistics['importance_weights:ratio_satisfying:staleness_threshold_num_minibatches_master_processed'] = I3.mean()
        I = I * I3


    D_importance_weight_and_more['importance_weights:all'] = D_importance_weight_and_more['importance_weight']
    D_importance_weight_and_more['importance_weights:usable'] = np.copy(D_importance_weight_and_more['importance_weight'])
    # Write np.nan in the spots where the importance weights are declared not usable.
    D_importance_weight_and_more['importance_weights:usable'][np.logical_not(I)] = np.nan

    if (importance_weight_additive_constant is not None and
        np.isfinite(importance_weight_additive_constant) and
        0.0 <= importance_weight_additive_constant):
        D_importance_weight_and_more['importance_weights:all:plus_additive_constant'] = D_importance_weight_and_more['importance_weights:all'] + importance_weight_additive_constant
        D_importance_weight_and_more['importance_weights:usable:plus_additive_constant'] = D_importance_weight_and_more['importance_weights:usable'] + importance_weight_additive_constant
    else:
        D_importance_weight_and_more['importance_weights:all:plus_additive_constant'] = D_importance_weight_and_more['importance_weights:all']
        D_importance_weight_and_more['importance_weights:usable:plus_additive_constant'] = D_importance_weight_and_more['importance_weights:usable']

    nbr_of_usable_importance_weights = I.sum()
    extra_statistics['nbr_of_usable_importance_weights'] = nbr_of_usable_importance_weights
    extra_statistics['ratio_of_usable_importance_weights'] = I.mean()
    extra_statistics['N'] = I.shape[0]

    return (D_importance_weight_and_more['importance_weights:usable:plus_additive_constant'],
            D_importance_weight_and_more,
            extra_statistics)


def record_importance_weights_statistics(   D_importance_weight_and_more, extra_statistics,
                                            remote_redis_logger=None, logging=None,
                                            want_compute_entropy=True):
    # The arguments to this function are the things returned
    # by `filter_raw_importance_weights`.

    def compute_entropy(p):
        epsilon = 1e-32
        p = p[np.isfinite(p)]
        p = p / p.sum()
        return -(p * np.log(p)).sum()

    if want_compute_entropy:
        extra_statistics['importance_weights:usable:entropy'] = compute_entropy(D_importance_weight_and_more['importance_weights:usable'])
        extra_statistics['importance_weights:usable:plus_additive_constant:entropy'] = compute_entropy(D_importance_weight_and_more['importance_weights:usable:plus_additive_constant'])

        extra_statistics['importance_weights:all:entropy'] = compute_entropy(D_importance_weight_and_more['importance_weights:all'])
        extra_statistics['importance_weights:all:plus_additive_constant:entropy'] = compute_entropy(D_importance_weight_and_more['importance_weights:all:plus_additive_constant'])

    if remote_redis_logger is not None:
        remote_redis_logger.log('importance_weights_statistics', extra_statistics)

    if logging is not None:
        for (k, v) in extra_statistics.items():
            logging.info("-- importance weights extra statistics --")
            logging.info("    %s    %f" %(k, v))






def sample_indices_and_scaling_factors( D_importance_weights_and_more,
                                        extra_statistics,
                                        nbr_samples,
                                        master_usable_importance_weights_threshold_to_ISGD=None,
                                        want_master_to_do_USGD_when_ISGD_is_not_possible=True,
                                        turn_off_importance_sampling=False):

    if turn_off_importance_sampling:
        N = extra_statistics['N']
        #A_sampled_indices = np.random.permutation(N)[0:nbr_samples]
        A_sampled_indices = np.random.choice(N, nbr_samples)
        A_scaling_factors = np.ones(A_sampled_indices.shape)
        return ('proceed', 'USGD', A_sampled_indices, A_scaling_factors)

    # Try to do ISGD before trying anything else..
    if master_usable_importance_weights_threshold_to_ISGD is not None:

        if master_usable_importance_weights_threshold_to_ISGD <= extra_statistics['ratio_of_usable_importance_weights']:
            if random.uniform(0,1) < 0.01:
                print "Master has a ratio of usable importance weights %d / %d = %f which meets the required threshold of %f." % (extra_statistics['nbr_of_usable_importance_weights'], extra_statistics['N'], extra_statistics['ratio_of_usable_importance_weights'], master_usable_importance_weights_threshold_to_ISGD)

            importance_weights_with_filled_zeros = np.copy(D_importance_weights_and_more['importance_weights:usable:plus_additive_constant'])
            I_to_fill = np.logical_not(np.isfinite(importance_weights_with_filled_zeros))
            importance_weights_with_filled_zeros[I_to_fill] = 0.0
            A_sampled_indices, A_scaling_factors = recipe2( importance_weights_with_filled_zeros,
                                                            extra_statistics['nbr_of_usable_importance_weights'], nbr_samples)
            return ('proceed', 'ISGD', A_sampled_indices, A_scaling_factors)
        else:
            if random.uniform(0,1) < 0.01:
                print "Master has a ratio of usable importance weights %f which falls shorts of the required threshold of %f." % (extra_statistics['ratio_of_usable_importance_weights'], master_usable_importance_weights_threshold_to_ISGD)


    # So, we're not going to do ISGD, but maybe we still want to do USGD.
    # In that particular case, we don't care at all about how stale the values could be,
    # or not even about whether the importance weights are present or not.
    if want_master_to_do_USGD_when_ISGD_is_not_possible:
        N = extra_statistics['N']
        #A_sampled_indices = np.random.permutation(N)[0:nbr_samples]
        A_sampled_indices = np.random.choice(N, nbr_samples)
        A_scaling_factors = np.ones(A_sampled_indices.shape)
        return ('proceed', 'USGD', A_sampled_indices, A_scaling_factors)
    else:
        return ('wait_and_retry', None, None, None)

#
# This `recipe1` is kept around in case we find a bug with `recipe2`,
# which is just strictly better.
#
def recipe1(A_importance_weights, nbr_of_present_importance_weights, nbr_samples):

    # A_importance_weights, nbr_of_present_importance_weights = get_importance_weights(rsconn)

    # Note that these are sampled with replacements, which
    # is what we want but which also goes against the traditional
    # traversal of the training set through a minibatch iteration scheme.

    # Note : If you plan on changing anything in this function,
    #        then you will need to update the documentation at the
    #        botton of this document.

    if A_importance_weights is None or A_importance_weights.sum() < 1e-16:
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

    # A_importance_weights, nbr_of_present_importance_weights = get_importance_weights(rsconn)

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

    A_sampled_indices = np.random.choice(p.shape[0], size=nbr_samples, p=p)

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
