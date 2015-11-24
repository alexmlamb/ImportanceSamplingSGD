
import redis
import numpy as np
import json
import time
import numpy as np

import sys, os
import getopt

import signal
import sys

from startup import delete_bootstrap_file, check_if_parameters_are_present, check_if_any_initialization_has_even_been_done, set_initialization_as_done
from common import setup_python_logger, get_mean_variance_measurement_on_database, get_trace_covariance_information

# TODO : Make use of this.
from sampling_for_master import get_raw_importance_weights, filter_raw_importance_weights, record_importance_weights_statistics

# python's logging module
import logging
import pprint
# our own logger that sends stuff over to the database
import integration_distributed_training.server.logger

def configure(  rsconn,
                workers_minibatch_size, master_minibatch_size,
                dataset_name,
                Ntrain, Nvalid, Ntest,
                L_measurements,
                L_segments,
                want_only_indices_for_master=True,
                want_exclude_partial_minibatch=True,
                default_importance_weight=0.0,
                **kwargs):

    # `workers_minibatch_size` is an int describing how large are the minibatches for the workers.
    # `master_minibatch_size` is an int describing how large are the minibatches for the master.
    # `dataset_name` is a string that is somewhat useless, but is still a good idea to include in the config.
    # `Ntrain` is the total number of training examples (to be split into minibatches).
    # `Nvalid` is the total number of validation examples (to be split into minibatches).
    # `Ntest`  is the total number of test examples (to be split into minibatches).
    # `L_measurements` is a list of quantities computed for each example.
    # `L_segments` is a list of ['train', 'valid', 'test'].
    # `want_only_indices_for_master` determines whether the master will be given arrays of indices or minibatch data directly.
    # `want_exclude_partial_minibatch` indicates if we want to forget about the data that doesn't fit in a complete minibatch.

    # We will use **dataset_config to specify all the arguments of this function.
    # For that reason, we need to have the extra **kwargs to eat up whatever is left.

    # "parameters:current" will contain a numpy float32 array
    # represented efficiently as a string (max 128MB, potential scaling problems)
    rsconn.set("parameters:current", "")
    rsconn.set("parameters:current_timestamp", time.time())
    # potentially not used
    rsconn.set("parameters:current_datestamp", time.strftime("%Y-%m-%d %H:%M:%S"))

    for segment in L_segments:

        # That's a bit of a hacky way to specify Ntrain/Nvalid/Ntest.
        N = {'train':Ntrain, 'valid':Nvalid, 'test':Ntest}[segment]

        rsconn.delete("L_workers_%s_minibatch_indices_QUEUE" % segment)
        rsconn.delete("L_workers_%s_minibatch_indices_ALL" % segment)

        for lower_index in range(0, N+1, workers_minibatch_size):

            if N <= lower_index:
                continue

            # The data points corresponding to `upper_index` are NOT to be included.
            upper_index = np.min([lower_index + workers_minibatch_size, N])

            if want_exclude_partial_minibatch and (upper_index - lower_index < workers_minibatch_size):
                continue

            assert upper_index - lower_index <= workers_minibatch_size

            A_indices = np.arange(lower_index, upper_index, dtype=np.int32)
            A_indices_str = A_indices.tostring()

            rsconn.rpush("L_workers_%s_minibatch_indices_QUEUE" % segment, A_indices_str)
            rsconn.rpush("L_workers_%s_minibatch_indices_ALL" % segment, A_indices_str)

            for measurement in L_measurements:
                if measurement in ['minibatch_gradient_mean_square_norm']:
                    # one float32 per minibatch
                    shape = (1,)
                else:
                    # one array per minibatch
                    shape = A_indices.shape

                rsconn.hset("H_%s_minibatch_%s" % (segment, measurement), A_indices_str, (np.float32(default_importance_weight) * np.ones(shape, dtype=np.float32)).tostring(order='C'))
                rsconn.hset("H_%s_minibatch_%s_measurement_last_update_timestamp" % (segment, measurement), A_indices_str, time.time())

            for measurement in ['previous_individual_importance_weight']:
                rsconn.hset("H_%s_minibatch_%s" % (segment, measurement), A_indices_str, (np.float32(default_importance_weight) * np.ones(A_indices.shape, dtype=np.float32)).tostring(order='C'))


    # The master does not really differentiate between the various
    # segments of the dataset. It just takes whatever it is fed
    # because nothing other than the training data should go in there.
    # We decided to put "_train_" in there nonetheless. Seems more consistent.

    # used when `want_only_indices_for_master` is True
    rsconn.delete("L_master_train_minibatch_indices_and_info_QUEUE")
    # used when `want_only_indices_for_master` is False
    rsconn.delete("L_master_train_minibatch_data_and_info_QUEUE")


def refresh_QUEUE_from_ALL(rsconn, L_segments, remote_redis_logger=None, logging=None):

    # This function is meant to be used when resuming training. Its goal is to
    # repair one potential thing that can break over training sessions.
    # The workers can be killed in-between the time when they popped a minibatch
    # to process, and the time where they add it back. This would cause a loss
    # of minibatches over time, and we have to clean it up.

    for segment in L_segments:

        name_ALL = "L_workers_%s_minibatch_indices_ALL" % segment
        name_QUEUE = "L_workers_%s_minibatch_indices_QUEUE" % segment

        if rsconn.llen(name_ALL) == rsconn.llen(name_QUEUE):
            continue
        else:
            msg = "We have %d elements in rsconn.llen(name_ALL), and %d elements in rsconn.llen(name_QUEUE). We need to add back the minibatches that slipped through the cracks." % (rsconn.llen(name_ALL), rsconn.llen(name_QUEUE))
            rsconn.delete(name_QUEUE)
            for i in range(rsconn.llen(name_ALL)):
                contents_str = rsconn.lindex(name_ALL, i)
                rsconn.rpush(name_QUEUE, contents_str)

            if remote_redis_logger is not None:
                remote_redis_logger.log('event', msg)
            if logging is not None:
                logging.info(msg)




def run(DD_config, rserv, rsconn, bootstrap_file, D_server_desc):

    importance_weight_additive_constant = DD_config['database']['importance_weight_additive_constant']

    # set up logging system for logging_folder
    setup_python_logger(folder=DD_config["database"]["logging_folder"])
    logging.info(pprint.pformat(DD_config))

    # set up logging system to the redis server
    remote_redis_logger = integration_distributed_training.server.logger.RedisLogger(rsconn, queue_prefix_identifier="service_database")
    integration_distributed_training.server.logger.record_machine_info(remote_redis_logger)

    if check_if_any_initialization_has_even_been_done(rsconn):
        #rsconn.set("resuming_from_previous_run", True)
        msg = "Resuming from previous run."
        remote_redis_logger.log('event', msg)
        logging.info(msg)

        # So we're about to resume training. Just as a sanity check, though, we should
        # really 1) make sure that the parameters are all there
        #        2) the contents of "L_workers_%s_minibatch_indices_ALL" is used to populate "L_workers_%s_minibatch_indices_QUEUE"
        #

        if not check_if_parameters_are_present(rsconn):
            msg = "Error. We are supposed to be resuming from a previous training session, but the parameters are not found in the database."
            remote_redis_logger.log('event', msg)
            logging.info(msg)
            quit()

        refresh_QUEUE_from_ALL(rsconn, DD_config['database']['L_segments'], remote_redis_logger, logging)
        set_initialization_as_done(rsconn, D_server_desc)

    else:

        configure(  rsconn, Ntrain = DD_config['model']['Ntrain'], Nvalid = DD_config['model']['Nvalid'], Ntest = DD_config['model']['Ntest'],
                    **DD_config['database'])
        #rsconn.set("resuming_from_previous_run", False)
        msg = "Starting a new run."
        remote_redis_logger.log('event', msg)
        logging.info(msg)

        set_initialization_as_done(rsconn, D_server_desc)


    # Use `rserv` to be able to shut down the
    # redis-server when the user hits CTRL+C.
    # Otherwise, the server is left in the background
    # and this can cause problems due to scripts
    # getting tangled together.

    def signal_handler(signal, frame):
        logging.info("You pressed CTRL+C.")
        logging.info("Closing the remote_redis_logger.")
        remote_redis_logger.log('event', "Received SIGTERM.")
        remote_redis_logger.close()
        logging.info("Sending save and shutdown commands to the redis-server.")
        rserv.stop(want_save=True)
        delete_bootstrap_file(bootstrap_file)
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)

    maximum_validation_accuracy = -1.0

    def sansgton(x):
        # sanitize singleton so it can be written to json.
        # Basically, it could be None, but you can't call float(None).
        # It can't be a np.float32 or np.float64 either, you can you have to call float(x)
        # on those values.
        if x is None:
            return float(np.nan)
        else:
            return float(x)

    remote_redis_logger.log('event', "Before entering service_database main loop.")

    while True:
        logging.info("Running server. Press CTLR+C to stop. Timestamp %f." % time.time())
        logging.info("Number minibatches processed by master    " + str(rsconn.get("parameters:num_minibatches_master_processed")))

        for segment in ["train", "valid", "test"]:
            logging.info("-- %s " % segment)
            for measurement in ["individual_loss", "individual_accuracy", "individual_gradient_square_norm"]:
                (mean, variance, N, r) = get_mean_variance_measurement_on_database(rsconn, segment, measurement)
                std = np.sqrt(variance)
                if segment == "valid" and measurement == "individual_accuracy" and mean > maximum_validation_accuracy:
                    maximum_validation_accuracy = mean
                    logging.info("                                                                      ---Highest Validation Accuracy so Far---")
                logging.info("---- %s : mean %f, std %f    with %0.4f of values used." % (measurement, mean, std, r))
                remote_redis_logger.log('measurement', {'name':measurement, 'segment':segment, 'mean':sansgton(mean), 'std':sansgton(std), 'ratio_used':r})

        logging.info("Highest Validation Accuracy seen so far " + str(maximum_validation_accuracy))

        time.sleep(10.0)

        # TODO : Figure a way to factor the staleness into this whole thing making
        #        making a horrible mess of spaghetti.

        # This is just extra. We always do the computation with 0.0 in all cases.
        L_importance_weight_additive_constant = [importance_weight_additive_constant]

        (usgd2, staleisgd2, isgd2, mu2, ratio_of_usable_indices_for_USGD_and_ISGD, ratio_of_usable_indices_for_ISGDstale, nbr_minibatches, D_other_staleISGD_main_term) = get_trace_covariance_information(rsconn, "train", L_importance_weight_additive_constant=L_importance_weight_additive_constant)
        # Make sure that you have a reasonable number of readings before
        # reporting those statistics.
        if 0.1 <= ratio_of_usable_indices_for_USGD_and_ISGD:
            assert usgd2 is not None and isgd2 is not None and mu2 is not None
            logging.info("Approximative norm squares of the mean gradient over whole dataset : %0.12f." % (mu2, ))
            logging.info("Trace(Cov USGD) without mu2 : %0.12f." % (usgd2 ,))
            logging.info("Trace(Cov ISGD) without mu2: %0.12f." % (isgd2 ,))
        else:
            logging.info("ratio_of_usable_indices_for_USGD_and_ISGD %f not high enough to report those numbers" % ratio_of_usable_indices_for_USGD_and_ISGD)

        if 0.1 <= ratio_of_usable_indices_for_ISGDstale:
            logging.info("Trace(Cov Stale ISGD) without mu2 : %0.12f." % (staleisgd2 ,))
            for (k, v) in D_other_staleISGD_main_term.items():
                logging.info("Trace(Cov Stale ISGD) without mu2 : %0.12f.  Using importance_weight_additive_constant %f." % (v, k))
        else:
            logging.info("ratio_of_usable_indices_for_ISGDstale %f not high enough to report those numbers" % ratio_of_usable_indices_for_ISGDstale)
        logging.info("")

        remote_redis_logger.log( 'SGD_trace_variance',
                    {'approx_mu2':sansgton(mu2), 'usgd2':sansgton(usgd2), 'isgd2':sansgton(isgd2), 'staleisgd2':sansgton(staleisgd2),
                     'extra_staleisgd2' : D_other_staleISGD_main_term,
                     'ratio_of_usable_indices_for_USGD_and_ISGD':ratio_of_usable_indices_for_USGD_and_ISGD,
                     'ratio_of_usable_indices_for_ISGDstale':ratio_of_usable_indices_for_ISGDstale})

        time.sleep(10.0)
        logging.info("")

        # have the database save itself to the file at every iteration through the loop
        if DD_config['database']['do_background_save']:
            rsconn.bgsave()


