
import redis
import numpy as np
import json
import time

import sys, os
import getopt

from integration_distributed_training.server.startup import main_entry_point_for_all_executable_scripts
import integration_distributed_training.server.service_database

def usage():
    print ""

def main(argv):

    want_start_redis_server_and_create_bootstrap_file = True
    (DD_config, D_server_desc, rserv, rsconn) = main_entry_point_for_all_executable_scripts(argv, want_start_redis_server_and_create_bootstrap_file)

    integration_distributed_training.server.service_database.run(DD_config, rsconn)


if __name__ == "__main__":
    main(sys.argv)


"""
    export PYTHONPATH=$PYTHONPATH:/u/lambalex/DeepLearning/ImportanceSampling/
    python run_database.py --config_file="../config_files/config_lamb_01.py" --bootstrap_file="bootstrap_09232"
"""
