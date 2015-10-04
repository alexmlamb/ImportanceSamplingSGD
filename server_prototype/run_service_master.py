
import redis
import numpy as np
import json
import time
import re

import sys, os
import getopt


SIMULATED_BATCH_UPDATE_TIME = 1.0

# (batch_name, weight, total_weights)
# with batch_name being of the form
#     "batch:%0.9d-%0.9d_%s" % (lower_index, upper_index, batch_desc_suffix)

prog_batch_name_triplet = re.compile(r"\((batch:(\d+)-(\d+)_(.*?)),\s([\-\d\.]+),\s([\-\d\.]+)\)")

def decode_batch_name_triplet(batch_name_triplet):

    m = prog_batch_name_triplet.match(batch_name_triplet)

    if m:
        batch_name = m.group(1)
        lower_index = int(m.group(2))
        upper_index = int(m.group(3))
        suffix = m.group(4)
        weight = np.float64(m.group(5))
        total_weights = np.float64(m.group(6))
    else:
        print "Failed to decode the batch_name_triplet : %s." % batch_name_triplet
        print "This should never happen, and you have a bug somewhere."
        quit()

    return (batch_name, lower_index, upper_index, suffix, weight, total_weights)



class MockModel(object):
    def get_parameters(self):
        return np.random.rand(4,5).astype(np.float32)
    #def set_parameters(self, parameters):
    #    pass
    def train(self, batch_name, lower_index, upper_index, suffix, weight, total_weights):
        print "Model was called to train on %s." % batch_name


def run(server_ip, server_port, server_password):

    assert server_ip
    assert server_port
    assert server_password

    timeout = 60

    initial_conn_timestamp = time.time()
    while time.time() - initial_conn_timestamp < timeout:
    
        try:
            rsconn = redis.StrictRedis(host=server_ip, port=server_port, password=server_password)
            print "Service Master connected to local server."
            break
        except:
            time.sleep(5)
            print "Service Master failed to connect to local server. Will retry in 5s."

    print "Pinging local server : %s" % (rsconn.ping(),)


    model = MockModel()

    # The master splits its time between two tasks.
    #
    # (1) Publish the parameters back to the server,
    #     which triggers a cascade of re-evaluation of
    #     importance weights for every batch on the workers.
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

    while True:

        # Task (1)
    
        current_parameters_str = model.get_parameters().tostring(order='C')
        rsconn.set("parameters:current", current_parameters_str)
        rsconn.set("parameters:current_timestamp", time.time())
        # potentially not used
        rsconn.set("parameters:current_datestamp", time.strftime("%Y-%m-%d %H:%M:%S"))
        print "The master has updated the parameters."


        # Task (2)

        for _ in range(nbr_batch_processed_per_public_parameter_update):

            batch_name_triplet = rsconn.lpop("importance_samples:L_(batch_name, weight, total_weights)")
            nbr_batch_name_triplet_remaining = rsconn.llen("importance_samples:L_(batch_name, weight, total_weights)") # for debugging, potentially oudated value

            if batch_name_triplet is None or len(batch_name_triplet) == 0:
                # Note that the "batch:L_names_todo" might be temporarily gone
                # from the server because that's how the assistant is updating it
                # to put fresh values in there. The worker just needs to stay calm
                # when that happens and not quit() in despair.
                print "The master has nothing to do. Might as well sleep."
                # TODO : Adjust the duration of the sleep.
                time.sleep(0.2)
                continue

            (batch_name, lower_index, upper_index, suffix, weight, total_weights) = decode_batch_name_triplet(batch_name_triplet)

            print "The master is processing %s. There are %d left." % (batch_name, nbr_batch_name_triplet_remaining)
            #print "(batch_name, lower_index, upper_index, suffix, weight, total_weights)"
            #print (batch_name, lower_index, upper_index, suffix, weight, total_weights)

            # TODO : Actually use a real model here.
            model.train(batch_name, lower_index, upper_index, suffix, weight, total_weights)

            # Sleep to simulate work time.
            time.sleep(SIMULATED_BATCH_UPDATE_TIME)

            print ""






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
    python run_service_master.py --server_port=5982 --server_password="patate"

"""
