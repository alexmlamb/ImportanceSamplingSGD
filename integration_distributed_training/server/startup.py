
import sys, os
import getopt
import time
import hashlib

import numpy as np

import redis
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
        delete_bootstrap_file(bootstrap_file)
        write_bootstrap_file(bootstrap_file, D_server_desc)
    else:
        D_server_desc = load_bootstrap_file(bootstrap_file)
        # No need for those here, but let's just put None values in them
        # to facilitate the call.
        rserv, rsconn = (None, None)

    return DD_config, D_server_desc, rserv, rsconn, bootstrap_file

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

    if database_config.has_key('redis_rdb_path_plus_filename'):

        rdb_dir = os.path.dirname(database_config['redis_rdb_path_plus_filename'])
        rdb_filename = os.path.basename(database_config['redis_rdb_path_plus_filename'])

        if len(rdb_dir) == 0:
            rdb_dir = "."

        print "rdb dir: %s" % rdb_dir
        print "rdb filename : %s" % rdb_filename

    else:
        rdb_dir = "."
        rdb_filename = None

    server_port = np.random.randint(low=1025, high=65535)

    server_password = "".join(["%d" % np.random.randint(low=0, high=10) for _ in range(10)])

    rserv = EphemeralRedisServer(   scratch_path=rdb_dir,
                                    port=server_port, password=server_password,
                                    dbfilename=rdb_filename)

    rserv.start()
    time.sleep(5)
    rsconn = rserv.get_client()
    print "pinging master server : %s" % (rsconn.ping(),)

    import socket
    hostname = socket.gethostname()

    D_server_desc = {'hostname' : hostname, 'port' : server_port, 'password' : server_password}
    return (rserv, rsconn, D_server_desc)


def load_bootstrap_file(bootstrap_file, timeout=30*60):

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

def delete_bootstrap_file(bootstrap_file):
    if os.path.exists(bootstrap_file):
        print "We will attempt to delete the old version of the bootstrap file."

        try:
            os.remove(bootstrap_file)
            print "Removed boostrap file %s." % bootstrap_file
        except OSError:
            print "Failed to remove the bootstrap file %s." % bootstrap_file
            print "Not sure why this happened, but this will create problems later because some permissions are wrong."
            pass
    else:
        print "No need to delete the bootstrap_file %s beause it does not exist." % bootstrap_file

# This is a way to provide a minimum of encapsulation to avoid
# having the string "initialization_is_done" sprinkled everywhere.

def get_session_identifier(D_server_desc):
    # This returns a unique identifier that allows us to resume training.
    # There are lots of traps, otherwise, and things that prevent us from
    # properly communicating the readiness of the database to resume training
    # if we don't have a unique string that refers to this current "session",
    # so to speak.

    s = "initialization_is_done %s %d" % (D_server_desc['password'], D_server_desc['port'])
    return hashlib.sha256(s).hexdigest()

def get_initialized_key(session_identifier):
    return "initialization_is_done:%s" % session_identifier

def get_parameters_key():
    return "parameters:current"

def check_if_parameters_are_present(rsconn):
    if 0 < len(rsconn.get(get_parameters_key())):
        return True
    else:
        return False

def set_initialization_as_done(rsconn, D_server_desc):
    initialized_key = get_initialized_key(get_session_identifier(D_server_desc))
    rsconn.set(initialized_key, True)
    print "%s was set to True" % initialized_key

def check_if_any_initialization_has_even_been_done(rsconn):
    return (0 < len(rsconn.keys(pattern="initialization_is_done:*")))


def get_rsconn_with_timeout(D_server_desc,
                            timeout=60, wait_for_parameters_to_be_present=True):

    server_hostname = D_server_desc['hostname']
    server_port = D_server_desc['port']
    server_password = D_server_desc['password']
    session_identifier = get_session_identifier(D_server_desc)
    initialized_key = get_initialized_key(session_identifier)
    parameters_key = get_parameters_key()

    initial_conn_timestamp = time.time()
    success = False
    while time.time() - initial_conn_timestamp < timeout:

        try:
            rsconn = redis.StrictRedis(host=server_hostname, port=server_port, password=server_password)
            print "Connected to local server."
            success = True
            break
        except:
            print "Unexpected error:", sys.exc_info()[0]
            print  sys.exc_info()
            print "Failed to connect to local server. Will retry in 5s."
            time.sleep(5)

    if not success:
        print "Quitting."
        quit()

    print "Pinging local server : %s" % (rsconn.ping(),)


    initial_conn_timestamp = time.time()
    success = False
    while time.time() - initial_conn_timestamp < timeout:
        if rsconn.get(initialized_key) in ["true", "True", "1"]:
            print "Experiment is properly initialized. We start now."
            success = True
            break
        else:
            print "Experiment is not ready to start. Waiting for key %s to be True. Will retry in 5s." % initialized_key
            time.sleep(5)

    if not success:
        print "Quitting."
        quit()

    if wait_for_parameters_to_be_present:
        initial_conn_timestamp = time.time()
        success = False
        while time.time() - initial_conn_timestamp < timeout:
            if 0 < len(rsconn.get(parameters_key)):
                print "The current parameters are found on the server. We start now."
                success = True
                break
            else:
                print "The current parameters are not yet on the server. Will retry in 5s."
                time.sleep(5)

        if not success:
            print "Quitting."
            quit()

    return rsconn
