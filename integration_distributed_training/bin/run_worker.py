
import redis
import numpy as np
import json
import time

import sys, os
import getopt

from integration_distributed_training.server.startup import main_entry_point_for_all_executable_scripts
import integration_distributed_training.server.service_worker

def usage():
    print ""

def main(argv):

    want_start_redis_server_and_create_bootstrap_file = False
    (DD_config, D_server_desc, _, _, _) = main_entry_point_for_all_executable_scripts(argv, want_start_redis_server_and_create_bootstrap_file)

    integration_distributed_training.server.service_worker.run(DD_config, D_server_desc)


if __name__ == "__main__":
    main(sys.argv)


"""
    export PYTHONPATH=$PYTHONPATH:${HOME}/Documents/ImportanceSamplingSGD
    cd ${HOME}/Documents/ImportanceSamplingSGD/integration_distributed_training/bin
    python run_worker.py --config_file="../config_files/config_09232.py" --bootstrap_file="bootstrap_09232"
"""
