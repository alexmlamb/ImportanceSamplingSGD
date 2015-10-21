
import redis
import numpy as np
import json
import time
import copy

import sys, os
import getopt

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

        nbr_indices_sampled_minimum = int(rsconn.get("config:nbr_indices_sampled_minimum"))
        assert 0 < nbr_indices_sampled_minimum

        nbr_indices_sampled_maximum = int(rsconn.get("config:nbr_indices_sampled_maximum"))
        assert 0 < nbr_indices_sampled_maximum
        assert nbr_indices_sampled_minimum <= nbr_indices_sampled_maximum

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
        # (1) When the parameters have been updated, you need to copy over
        #     the contents from "batch:L_names_all" to "batch:L_names_todo"
        #     in a shuffled order. Clear "batch:L_names_todo" beforehand.
        #
        # (2) When the number of drawn samples in
        #     "importance_samples:L_(batch_name, weight, total_weights)"
        #     drops below `nbr_indices_sampled_minimum`, then you have to resample
        #     more values. We get enough to bring the number up to
        #     `nbr_indices_sampled_maximum` to be good for a while.
        #     Insert on one side and removing on the other side (older indices).
        #
        # The structure of the while True loop for the assistant is slightly different
        # than for the other roles because it has to restrain itself to avoid
        # looping too quickly. In the other roles, there is always one of the tasks
        # which is expected to be scheduled as aggressively as possible.

        new_parameters_current_timestamp = rsconn.get("parameters:current_timestamp")
        if parameters_current_timestamp != new_parameters_current_timestamp:
            # Maintenance task (1) should be done.

            parameters_current_timestamp = new_parameters_current_timestamp

            # shallow copy. Don't modify the strings themselves,
            # but make a new list that can be shuffled.
            # That being said, it would probably be fine
            # to shuffle the original list `batch_L_names_all`,
            # but it's probably wiser to keep it intact since it's
            # meant to cache the value found on the server, which
            # we are fine with reading only once and using many times.
            batch_L_names_todo = copy.copy(batch_L_names_all)
            np.random.shuffle(batch_L_names_todo)

            # We just delete the list in order to clear it out.
            # This means that workers reading from the list have
            # to be careful to avoid freaking out if they don't
            # find it for a split-second. They should just retry.
            rsconn.delete("batch:L_names_todo")
            for e in batch_L_names_todo:
                rsconn.rpush("batch:L_names_todo", e)

            maintenance_task_1_was_required = True


        nbr_indices_sampled = rsconn.llen("importance_samples:L_(batch_name, weight, total_weights)")

        if nbr_indices_sampled < nbr_indices_sampled_minimum:
            # Maintenance task (2) should be done.

            L_weights = []
            total_weights = np.float64(0.0)
            for batch_name in batch_L_names_all:
                weight_str = rsconn.get(batch_name)
                assert 0 < len(weight_str), batch_name
                weight = np.fromstring(weight_str, dtype=np.float64)
                L_weights.append(weight)
                total_weights += weight

            L_encoded_samples = [encode_batch_name_triplets(*v3) for v3 in sample_batch_name_triplets(batch_L_names_all, L_weights, total_weights, nbr_indices_sampled_maximum)]

            # Warning. We'll get a little bit of a race condition if the master is fetching
            #          indices from "importance_samples:L_(batch_name, weight, total_weights)"
            #          while we clear out that list to repopulate it.
            #          The master will have to be careful and retry when it finds
            #          that list to be empty (in-between interventions from the assistant).

            for encoded_sample in L_encoded_samples:
                rsconn.lpush("importance_samples:L_(batch_name, weight, total_weights)", encoded_sample)

            while nbr_indices_sampled_maximum < rsconn.llen("importance_samples:L_(batch_name, weight, total_weights)"):
                # throw away one element, starting from the oldest,
                # until we reach the target `nbr_indices_sampled_maximum`
                rsconn.rpop("importance_samples:L_(batch_name, weight, total_weights)")

            print "Maintenance task (2) completed. We now have %d sampled indices available (up from the %d that we had)." % (rsconn.llen("importance_samples:L_(batch_name, weight, total_weights)"), nbr_indices_sampled)
            maintenance_task_2_was_required = True



        # There should be some kind of sleep to avoid hammering the database too agressively
        # if neither maintenance task was required. Maybe set a kind of sleeping schedule that
        # backs down to 10s if it doesn't get updates, but it would start at something low like 0.5s.
        if maintenance_task_1_was_required is False and maintenance_task_2_was_required is False:
            print "No maintenance task currently required. Sleeping."
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
    main(sys.argv)


"""
    python run_service_assistant.py --server_port=5982 --server_password="patate"

"""
