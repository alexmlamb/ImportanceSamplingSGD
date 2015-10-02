
import redis
import numpy as np
import json
import time
import re

import sys, os
import getopt

# batch_name being of the form
#     "batch:%0.9d-%0.9d_%s" % (lower_index, upper_index, batch_desc_suffix)

prog_batch_name = re.compile(r"batch:(\a+)-(\a+)_(.*?)")

def decode_batch_name(batch_name):

    m = prog_batch_name.match(batch_name)

    if m:
        lower_index = int(m.group(1))
        upper_index = int(m.group(2))
        suffix = m.group(3)
    else:
        print "Failed to decode the batch_name : %s." % batch_name
        print "This should never happen, and you have a bug somewhere."
        quit()

    return (lower_index, upper_index, suffix)




def run(server_ip, server_port, server_password):

    assert server_ip
    assert server_port
    assert server_password

    timeout = 60

    initial_conn_timestamp = time.time()
    while time.time() - initial_conn_timestamp < timeout:
    
        try:
            rsconn = redis.StrictRedis(host=server_ip, port=server_port, password=server_password)
            print "Service Worker connected to local server."
            break
        except:
            time.sleep(5)
            print "Service Worker failed to connect to local server. Will retry in 5s."

    print "Pinging local server : %s" % (rsconn.ping(),)


    # Now we just feed from anything inside of "batch:L_names_todo"
    # and we send the computed norms to the corresponding key on the server.

    # The worker has to watch two things.
    #
    # (1) Have the parameters been updated on the server ?
    #     Check out the timestamp to determine if they have been updated.
    #     (Because of the assumption that the master updates the paramters
    #     and *then* the timestamp.)
    #     If they have been updated, we want to fetch a copy a convert it
    #     to a numpy array.
    #
    # (2) Is there something to process in "batch:L_names_todo" ?
    #     If there is, then we'll process it.

    current_parameters = None
    parameters_current_timestamp = ""
    while True:

        # Task (1)

        new_parameters_current_timestamp = rsconn.get("parameters:current_timestamp")
        if parameters_current_timestamp != new_parameters_current_timestamp:

            current_parameters_str = rsconn.get("parameters:current")
            if len(current_parameters_str) == 0:
                print "No parameters found in the server. Might as well sleep."
                time.sleep(0.2)
                continue
            current_parameters = np.fromstring(current_parameters_str, dtype=np.float32)
            parameters_current_timestamp = new_parameters_current_timestamp
            continue


        # Task (2)

        batch_name = rsconn.lpop("batch:L_names_todo")
        if batch_name is None or len(batch_name) == 0:
            # Note that the "batch:L_names_todo" might be temporarily gone
            # from the server because that's how the assistant is updating it
            # to put fresh values in there. The worker just needs to stay calm
            # when that happens and not quit() in despair.
            print "The worker has nothing to do. Might as well sleep."
            # TODO : Adjust the duration of the sleep.
            time.sleep(0.2)
            continue

        (lower_index, upper_index, suffix) = decode_batch_name(batch_name)

        print "The worker is processing %s." % batch_name
        print "(lower_index, upper_index, suffix)"
        print (lower_index, upper_index, suffix)

        # TODO : Compute the actual gradient norm here.
        rsconn.set(batch_name, 1.0)
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
    python run_service_worker.py --server_port=5982 --server_password="patate"

"""
