#!/bin/bash

#PBS -l nodes=1:gpus=2
#PBS -l walltime=4:00:00
#PBS -A jvb-000-ag
#PBS -m bea
#PBS -l feature=k20

# Use msub on helios1 to submit this.
#

# msub ~/Documents/ImportanceSamplingSGD/integration_distributed_training/config_files/helios/12_repeat20x030/launch_00076.sh
# msub -l depend=51777 ~/Documents/ImportanceSamplingSGD/integration_distributed_training/config_files/helios/12_repeat20x030/launch_00076.sh


#OUTPUT=`msub ~/Documents/ImportanceSamplingSGD/integration_distributed_training/config_files/helios/12_repeat20x030/launch_00076.sh`
#OUTPUT=`echo $OUTPUT | tr -d " "`
#echo $OUTPUT

#for i in `seq 2 8`;
#do
#    OUTPUT=`msub -l depend=${OUTPUT} ~/Documents/ImportanceSamplingSGD/integration_distributed_training/config_files/helios/12_repeat20x030/launch_00076.sh`
#    OUTPUT=`echo $OUTPUT | tr -d " "`
#    echo $OUTPUT
#done


# Note that the range above includes both the first and last number.
# The %3 sign after is to instruct the scheduler that we want to run
# as much as %3 simultaneous jobs. For our uses, there isn't much
# of a reason to launch anything less since we're not dealing
# with sequential and independent jobs.


# Before running this, check out a copy of ImportanceSamplingSGD locally.
# Not sure why again, but it works for the HTTP address and not the SSH address.
# git clone https://github.com/alexmlamb/ImportanceSamplingSGD.git ImportanceSamplingSGD

export IMPORTANCE_SAMPLING_SGD_ROOT=${HOME}/Documents/ImportanceSamplingSGD
export PYTHONPATH=${PYTHONPATH}:${IMPORTANCE_SAMPLING_SGD_ROOT}
export IMPORTANCE_SAMPLING_SGD_BIN=${IMPORTANCE_SAMPLING_SGD_ROOT}/integration_distributed_training/bin

export CONFIG_FILE=${IMPORTANCE_SAMPLING_SGD_ROOT}/integration_distributed_training/config_files/helios/12_repeat20x030/config_00076.py

# The config file will contain other information such as the directory in
# which we want to output logs.

# Put garbage in there. It's important that this be a unique file that can
# be reached by all the tasks launched since it's going to be how they
# communicate between themselves initially to share where the database is running,
# what port it's on and what's the password.
export BOOTSTRAP_FILE=${HOME}/bootstrap_experiment_00076

# The whole stdbuf is not necessary, but I left it there because it fixes
# some of the strange behavior when we try to redirect the output to a file.


stdbuf -i0 -o0 -e0 python ${IMPORTANCE_SAMPLING_SGD_BIN}/run_database.py --config_file=${CONFIG_FILE} --bootstrap_file=${BOOTSTRAP_FILE} &

# Sleep just a little bit to make sure that, if there was some problem last time a bootstrap_file that failed to delete,
# then the run_database.py script would get the opportunity to overwrite that bootstrap file before the master/workers
# start sniffing around to find the bootstrap_file.
sleep 10

THEANO_FLAGS=device=gpu0,floatX=float32 stdbuf -i0 -o0 -e0 python ${IMPORTANCE_SAMPLING_SGD_BIN}/run_master.py --config_file=${CONFIG_FILE} --bootstrap_file=${BOOTSTRAP_FILE} &
THEANO_FLAGS=device=gpu1,floatX=float32 stdbuf -i0 -o0 -e0 python ${IMPORTANCE_SAMPLING_SGD_BIN}/run_worker.py --config_file=${CONFIG_FILE} --bootstrap_file=${BOOTSTRAP_FILE} &
#THEANO_FLAGS=device=gpu2,floatX=float32 stdbuf -i0 -o0 -e0 python ${IMPORTANCE_SAMPLING_SGD_BIN}/run_worker.py --config_file=${CONFIG_FILE} --bootstrap_file=${BOOTSTRAP_FILE} &
#THEANO_FLAGS=device=gpu3,floatX=float32 stdbuf -i0 -o0 -e0 python ${IMPORTANCE_SAMPLING_SGD_BIN}/run_worker.py --config_file=${CONFIG_FILE} --bootstrap_file=${BOOTSTRAP_FILE} &

sleep 3600
sleep 3600
sleep 3600
sleep 3600
sleep 3600
sleep 3600
