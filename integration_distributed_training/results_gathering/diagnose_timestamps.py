
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


import numpy as np
import os
import pickle

def diagnose_timestamps(results, pkl_path):

    DL_individual_accuracy = {'train' :[], 'valid' :[], 'test' :[]}
    DL_individual_loss = {'train' :[], 'valid' :[], 'test' :[]}

    t0 = results['logging']['service_database'].values()[0]['machine_info'][0][0]

    #assert len(results['logging']['service_worker'].values()) == 1
    assert len(results['logging']['service_master'].values()) == 1
    assert len(results['logging']['service_database'].values()) == 1

    for sd in results['logging']['service_database'].values():
        for (timestamp, e) in sd['measurement']:
            if e['name'] == 'individual_loss':
                DL_individual_loss[e['segment']].append((timestamp, e['mean']))
            elif e['name'] == 'individual_accuracy':
                DL_individual_accuracy[e['segment']].append((timestamp, e['mean']))

    print ""
    print pkl_path
    for segment in ['train', 'valid', 'test']:
        for (measurement, DL) in [('individual_loss', DL_individual_loss), ('individual_accuracy', DL_individual_accuracy)]:
            start = DL[segment][0][0]
            end = DL[segment][-1][0]
            for (t, v) in DL[segment]:
                if v is not None:
                    (first_available_t, first_available_v) = (t, v)
                    break
            print "    Results from %s %s: (%f, %f). Warm-up %0.3f hours. Duration %0.3f hours. Not-None data available only starting after %0.3f hours." % (measurement, segment,
                                                                                                                                        start, end,
                                                                                                                                        (start - t0) / 3600,
                                                                                                                                        (end - start) / 3600, (first_available_t - start) / 3600)

def run():

    #(start_experiment_index, end_experiment_index) = (70, 120)
    (start_experiment_index, end_experiment_index) = (70, 150)
    #(start_experiment_index, end_experiment_index) = (120, 150)

    L_parsed_results = []

    for i in range(start_experiment_index, end_experiment_index):
        pkl_path = "%s/%0.5d/%0.5d.pkl" % (experiment_dir, i, i)
        if not os.path.exists(pkl_path):
            print "Skipping over %s because it's missing." % pkl_path
            continue

        results = pickle.load(open(pkl_path, "r"))
        diagnose_timestamps(results, pkl_path)


if __name__ == "__main__":
    run()
