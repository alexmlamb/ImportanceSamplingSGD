
import redis
import numpy as np
import json
import time

import sys, os
import getopt


def sample_batch_name_triplets(batch_L_names_all, L_weights, total_weights, nbr_indices_sampled):

    # TODO : Implement this for real.
    #        Right now we're just returning a valid value,
    #        but we're absolutely not sampling correctly.

    A_weights = np.array(L_weights)

    accum = []
    for i in range(nbr_indices_sampled):
        accum.append((batch_L_names_all[i], A_weights[i], total_weights))

    return accum


def encode_batch_name_triplets(batch_name, weight, total_weights):
    return "(%s, %f, %f)" % (batch_name, weight, total_weights)




def run(server_ip, server_port, server_password):

    assert server_ip
    assert server_port
    assert server_password

    timeout = 60

    initial_conn_timestamp = time.time()
    while time.time() - initial_conn_timestamp < timeout:
    
        try:
            rsconn = redis.StrictRedis(host=server_ip, port=server_port, password=server_password)
            print "Service Assistant connected to local server."
            break
        except:
            time.sleep(5)
            print "Service Assistant failed to connect to local server. Will retry in 5s."

    print "Pinging local server : %s" % (rsconn.ping(),)


    success = False
    initial_config_timestamp = time.time()
    while time.time() - initial_config_timestamp < timeout:
    
        parameters_current_timestamp = rsconn.get("parameters:current_timestamp")
        if len(parameters_current_timestamp) == 0:
            print "Database is not configured yet. Waiting for 5 seconds before retrying."
            time.sleep(5)
            continue

        nbr_of_batches = rsconn.llen("batch:L_names_all")
        if nbr_of_batches == 0:
            print "Database is not configured yet. Waiting for 5 seconds before retrying."
            time.sleep(5)
            continue

        batch_L_names_all = rsconn.lrange("batch:L_names_all", 0, nbr_of_batches)
        assert rsconn.llen("batch:L_names_todo") == 0, "Was this database initialized by someone else before ?"

        nbr_indices_sampled = int(rsconn.get("config:nbr_indices_sampled"))
        assert 0 < nbr_indices_sampled

        resampling_threshold = int(rsconn.get("config:resampling_threshold"))
        assert 0 < resampling_threshold
        assert resampling_threshold <= nbr_indices_sampled

        success = True
        break

    if success is False:
        print "The database was not configured in time and the assistant just timed out (in %d seconds)." % timeout
        quit()



    # At this point we just assume that everything is configured properly
    # and the assistant can do its job.

    while True:

        maintenance_task_1_was_required = False
        maintenance_task_2_was_required = False

        # There are two maintenance tasks here.
        #
        # 1) When the parameters have been updated, you need to copy over
        #    the contents from "batch:L_names_all" to "batch:L_names_todo"
        #    in a shuffled order. Clear "batch:L_names_todo" beforehand.
        #
        # 2) When the number of drawn samples in
        #    "importance_samples:L_(batch_name, weight, total_weights)"
        #    drops below `nbr_indices_sampled`, then you have to resample
        #    more values. Maybe you want to push out the stale values out
        #    by inserting on one side and removing on the other side ?

        new_parameters_current_timestamp = rsconn.get("parameters:current_timestamp")
        if parameters_current_timestamp != new_parameters_current_timestamp:
            # Maintenance task (1) should be done.

            parameters_current_timestamp = new_parameters_current_timestamp

            # ...

            maintenance_task_1_was_required = True



        nbr_samples_left = rsconn.llen("importance_samples:L_(batch_name, weight, total_weights)")
        if nbr_samples_left < resampling_threshold:
            # Maintenance task (2) should be done.

            L_weights = []
            total_weights = np.float64(0.0)
            for batch_name in batch_L_names_all:
                weight_str = rsconn.get(batch_name)
                assert 0 < len(weight_str), batch_name
                weight = np.fromstring(weight_str, dtype=np.float64)
                L_weights.append(weight)
                total_weights += weight

            L_encoded_samples = [encode_batch_name_triplets(*v3) for v3 in sample_batch_names(batch_L_names_all, L_weights, total_weights, nbr_indices_sampled)]

            # Warning. We'll get a little bit of a race condition if the master is fetching
            #          indices from "importance_samples:L_(batch_name, weight, total_weights)"
            #          while we clear out that list to repopulate it.
            #          The master will have to be careful and retry when it finds
            #          that list to be empty (in-between interventions from the assistant).

            for encoded_sample in L_encoded_samples:
                rsconn.lpush("importance_samples:L_(batch_name, weight, total_weights)", encoded_sample)

            while nbr_indices_sampled <= rsconn.llen("importance_samples:L_(batch_name, weight, total_weights)"):
                # throw away one element, starting from the oldest,
                # until we reach the target `nbr_indices_sampled`
                rsconn.rpop("importance_samples:L_(batch_name, weight, total_weights)")


            maintenance_task_2_was_required = True



        # There should be some kind of sleep to avoid hammering the database too agressively
        # if neither maintenance task was required. Maybe set a kind of sleeping schedule that
        # backs down to 10s if it doesn't get updates, but it would start at something low like 0.5s.
        if maintenance_task_1_was_required is False and maintenance_task_2_was_required is False:
            time.sleep(1.0)

        # end While



def usage():
    print ""

def main(argv):
    """
    """

    try:
        opts, args = getopt.getopt(sys.argv[1:], "hv", ["server_ip=", "server_port=", "server_password="])
    except getopt.GetoptError as err:
        # print help information and exit:
        print str(err) # will print something like "option -a not recognized"
        usage()
        sys.exit(2)

    server_ip = "localhost"
    server_port = None
    server_password = None

    verbose = False
    for o, a in opts:
        if o == "-v":
            verbose = True
        elif o in ("-h", "--help"):
            usage()
            sys.exit()
        elif o in ("--server_ip"):
            server_ip = a
        elif o in ("--server_port"):
            server_port = int(a)
        elif o in ("--server_password"):
            server_password = a
        else:
            assert False, "unhandled option"
 

    run(server_ip, server_port, server_password)


if __name__ == "__main__":
    #main(sys.argv)

    for s in list_all_heap_keys(range(0,10), 2)['all']:
        print s