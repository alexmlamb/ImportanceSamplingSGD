#!/bin/bash

#PBS -l nodes=1:gpus=1
#PBS -l walltime=00:05:00
#PBS -A jvb-000-ag
#PBS -m bea
#PBS -M chinnadhurai@gmail.com
#PBS -t [1]


# Use msub on helios1 to submit this.


# Before running this, check out a copy of ImportanceSamplingSGD locally.
# Not sure why again, but it works for the HTTP address and not the SSH address.
# git clone https://github.com/alexmlamb/ImportanceSamplingSGD.git ImportanceSamplingSGD

export IMPORTANCE_SAMPLING_SGD_ROOT=${HOME}/Documents/ImportanceSamplingSGD
export PYTHONPATH=${PYTHONPATH}:${IMPORTANCE_SAMPLING_SGD_ROOT}
export EXP_CHINNA=${IMPORTANCE_SAMPLING_SGD_ROOT}/model_protoype/exp_chinna


# Note that, from the perspective of a script, the assigned GPU is always gpu0,
# regardless of which one it actually is on the machine (when we have only one assigned).

if [ ${MOAB_JOBARRAYINDEX} = "1" ]
then
    # The whole stdbuf is not necessary, but I left it there because it fixes
    # some of the strange behavior when we try to redirect the output to a file.
    THEANO_FLAGS=device=gpu0,floatX=float32 stdbuf -i0 -o0 -e0 python ${EXP_CHINNA}/nnet_practice.py
