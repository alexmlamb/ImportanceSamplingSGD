
import sys, os
import getopt
import time

import numpy as np

import json

from integration_distributed_training.server.redis_server_wrapper import EphemeralRedisServer

def main_entry_point_for_all_executable_scripts(argv, want_start_redis_server_and_create_bootstrap_file):

    # There is a bit of a strange conceptual mismatch here when we decide to
    # start the server (which is required when creating the bootstrap file).
    # This is partly explained by the fact that we don't really know the port number
    # ahead of time because any port selected could end up being used by another
    # program (and we don't even retry another port in this case).

    try:
        opts, args = getopt.getopt(sys.argv[1:], "hv", ["config_file=", "bootstrap_file="])
    except getopt.GetoptError as err:
        # print help information and exit:
        print str(err) # will print something like "option -a not recognized"
        usage()
        sys.exit(2)

    config_file = None
    bootstrap_file = None

    verbose = False
    for o, a in opts:
        if o == "-v":
            verbose = True
        elif o in ("-h", "--help"):
            usage()
            sys.exit()
        elif o in ("--config_file"):
            config_file = a
        elif o in ("--bootstrap_file"):
            bootstrap_file = a
        else:
            assert False, "unhandled option"

    assert config_file is not None
    assert bootstrap_file is not None


    # load the config file
    # get some config for database
    # start the database if you're supposed to start it
    # write to the bootstrap file (if applicable)
    # wait for bootstrap file to be available (if you're not the one writing to it)
    # read stuff from the boostrap file

    print "Will load the config_file : %s." % config_file

    DD_config = load_config_file(config_file)
    print "config_file loaded."

    if want_start_redis_server_and_create_bootstrap_file:
        (rserv, rsconn, D_server_desc) = start_redis_server(DD_config['database'])
        write_bootstrap_file(bootstrap_file, D_server_desc)
    else:
        D_server_desc = load_bootstrap_file(bootstrap_file)
        # No need for those here, but let's just put None values in them
        # to facilitate the call.
        rserv, rsconn = (None, None)

    return DD_config, D_server_desc, rserv, rsconn

def load_config_file(config_file):

    import imp

    # Seems like that first argument does not matter at all.
    mod = imp.load_source('config_file', config_file)
    #print config_file
    #mod = imp.load_source(config_file, config_file)

    return {'model' : mod.get_model_config(),
            'database' : mod.get_database_config(),
            'helios' : mod.get_helios_config()}


def start_redis_server(database_config):

    # Right now we don't even use what is in the `database_config` argument.

    if database_config.has_key('server_scratch_path'):
        server_scratch_path = database_config['server_scratch_path']
    else:
        server_scratch_path = "."

    server_port = np.random.randint(low=1025, high=65535)

    server_password = "".join(["%d" % np.random.randint(low=0, high=10) for _ in range(10)])

    rserv = EphemeralRedisServer(   scratch_path=server_scratch_path,
                                    port=server_port, password=server_password)

    rserv.start()
    time.sleep(5)
    rsconn = rserv.get_client()
    print "pinging master server : %s" % (rsconn.ping(),)

    import socket
    hostname = socket.gethostname()

    D_server_desc = {'hostname' : hostname, 'port' : server_port, 'password' : server_password}
    return (rserv, rsconn, D_server_desc)


def load_bootstrap_file(bootstrap_file, timeout=5*60):

    initial_timestamp = time.time()
    success = False
    while time.time() - initial_timestamp < timeout:

        if os.path.exists(bootstrap_file):
            print "The filesystem tells us that the bootstrap file exists."
            # Sleeping might still be a good idea in case the changes have
            # not propagated completely, but it's a bit overkill to start
            # with this when we don't know if it's necessary.
            # time.sleep(30)
            D_server_desc = json.load(open(bootstrap_file, 'r'))
            return D_server_desc
        else:
            time.sleep(5)
            print "Failed to read bootstrap file %s. Will retry in 5s." % bootstrap_file

        print "Fatal error. Failed to read bootstrap file %s after %d seconds timeout. Quitting." % (bootstrap_file, timeout)
        quit()


def write_bootstrap_file(bootstrap_file, D_server_desc):
    # D_server_desc contains keys ['hostname', 'port', 'password']
    json.dump(D_server_desc, open(bootstrap_file, "w"), indent=4, separators=(',', ': '))
    print "Wrote %s." % bootstrap_file
