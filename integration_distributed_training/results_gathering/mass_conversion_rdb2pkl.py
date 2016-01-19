


# Pick one, depending where you run this.
# This could be done differently too by looking at fuelrc
# or at the hostname.
import socket
importance_sampling_sgd_root = {    "serendib":"/home/dpln/Documents/ImportanceSamplingSGD",
                                    "lambda":"/home/gyomalin/Documents/ImportanceSamplingSGD",
                                    "szkmbp":"/Users/gyomalin/Documents/ImportanceSamplingSGD"}[socket.gethostname().lower()]

experiment_dir = {  "serendib":None,
                    "lambda":"/mnt/raid5vault6/tmp/ML/ICLR2016_ISGD/helios_experiments",
                    "szkmbp":"/Users/gyomalin/Documents/helios_experiments"}[socket.gethostname().lower()]

import os
import subprocess

# set this to True if you want to avoid re-doing jobs.
# set this to False if you might have transferred updated rdb files since the last time.
want_skip_over_already_done = False

# TODO : Maybe add a `renice` to the command.

L_cmd = []
L_missing = []
#for i in range(70, 270):
for i in range(90, 120) + range(200, 220) + range(250, 270):
    rdb_path = "%s/%0.5d/%0.5d.rdb" % (experiment_dir, i, i)
    pkl_path = "%s/%0.5d/%0.5d.pkl" % (experiment_dir, i, i)
    if want_skip_over_already_done and os.path.exists(pkl_path):
        print "Skipping over %s because it already exists." % pkl_path
        continue

    if not os.path.exists(rdb_path):
        print "ERROR. Skipping over %s because it's missing !" % rdb_path
        L_missing.append(rdb_path)
        continue

    cmd = "PYTHONPATH=${PYTHONPATH}:%s %s/integration_distributed_training/bin/rdb2pkl %s" % (importance_sampling_sgd_root, importance_sampling_sgd_root, rdb_path)
    #cmd = "PYTHONPATH=${PYTHONPATH}:%s nice -n 10 %s/integration_distributed_training/bin/rdb2pkl %s" % (importance_sampling_sgd_root, importance_sampling_sgd_root, rdb_path)
    #print cmd
    L_cmd.append(cmd)

if len(L_cmd) == 0:
    print "Nothing to do. All the files are there."
    quit()
else:
    print "We have %d files to convert." % len(L_cmd)

from multiprocessing import Pool

def f(cmd):
    print cmd
    s = subprocess.check_output(cmd, shell=True)
    print s
    return s


#if True:
if len(L_cmd) <= 6:
    print "Doing everything serially."
    for cmd in L_cmd:
        f(cmd)
else:
    print "Doing everything with a multiprocessing Pool."
    pool = Pool(processes=4)
    pool.map(f, L_cmd)


print "Missing : %s" % str(L_missing)
