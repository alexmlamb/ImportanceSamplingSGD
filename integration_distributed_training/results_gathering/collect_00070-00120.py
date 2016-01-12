

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


def parse_results(results):

    sd = results['logging']['service_database'].values()[0]

    # We don't want to have the initial time be set by other measurements
    # that we're not even interested in. But this is a debatable decision.
    t0 = None

    DL_individual_accuracy = {'train' :[], 'valid' :[], 'test' :[]}
    DL_individual_loss = {'train' :[], 'valid' :[], 'test' :[]}

    for (timestamp, e) in sd['measurement']:
        if e['name'] == 'individual_loss':
            if t0 is None:
                t0 = timestamp
            DL_individual_loss[e['segment']].append((timestamp - t0, e['mean']))
        elif e['name'] == 'individual_accuracy':
            if t0 is None:
                t0 = timestamp
            DL_individual_accuracy[e['segment']].append((timestamp - t0, e['mean']))

    return (DL_individual_accuracy, DL_individual_loss)


import numpy as np
import re
import os
import pickle

import matplotlib
# This has already been specified in .scitools.cfg
# so we don't need to explicitly pick 'Agg'.
matplotlib.use('Agg')
import pylab
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages


def run():

    #(start_experiment_index, end_experiment_index) = (70, 120)
    #(start_experiment_index, end_experiment_index) = (70, 90)
    #(start_experiment_index, end_experiment_index) = (120, 140)
    (start_experiment_index, end_experiment_index) = (120, 170)

    L_parsed_results = []

    for i in range(start_experiment_index, end_experiment_index):
        pkl_path = "%s/%0.5d/%0.5d.pkl" % (experiment_dir, i, i)
        if not os.path.exists(pkl_path):
            print "Skipping over %s because it's missing." % pkl_path
            continue

        results = pickle.load(open(pkl_path, "r"))
        L_parsed_results.append(parse_results(results))
        print "Parsed %s." % pkl_path

    # TODO : iterate over measurements, iterate over segments, produce png and pdf

    #measurement = 'individual_accuracy'
    #segment = 'train'

    for measurement in ['individual_accuracy', 'individual_loss']:
        for segment in ['train', 'valid', 'test']:
            for plot_suffix in ['pdf', 'png']:

                output_path = '%0.5d-%0.5d_%s_%s.%s' % (start_experiment_index, end_experiment_index, measurement, segment, plot_suffix)
                pylab.hold(True)
                max_timestamp = 0.0
                for E in L_parsed_results:

                    R = {'individual_accuracy':E[0], 'individual_loss':E[1]}[measurement]

                    domain = np.array([r[0] for r in R[segment]]) / 3600
                    values = np.array([r[1] for r in R[segment]])
                    if max_timestamp < domain[-1]:
                        max_timestamp = domain[-1]

                    handle = pylab.plot( domain,
                                         values,
                                         label=segment, linewidth=1 )

                    #pylab.plot( [L_domain[0]/3600, L_domain[-1]/3600], [1.00, 1.00], '--', c="#FF7F00", linewidth=0.5)

                plt.xlabel("time in hours")
                if measurement == "individual_accuracy":
                    plt.ylim([0.50, 1.05])
                    plt.title("Prediction accuracy over whole dataset")
                elif measurement == "individual_loss":
                    plt.title("Loss over whole dataset")

                # http://stackoverflow.com/questions/14442099/matplotlib-how-to-show-all-digits-on-ticks
                xx, locs = plt.xticks()
                ll = ['%.2f' % a for a in xx]
                plt.xticks(xx, ll)

                #plt.legend(loc=7)

                if re.match(r".*\.pdf", output_path):
                    with PdfPages(output_path) as pdf:
                        pdf.savefig()
                else:
                    pylab.savefig(output_path, dpi=250)

                pylab.close()
                print "Wrote %s." % output_path


if __name__ == "__main__":
    run()
