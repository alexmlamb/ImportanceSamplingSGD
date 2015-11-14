
import numpy as np
import pickle

import matplotlib
# This has already been specified in .scitools.cfg
# so we don't need to explicitly pick 'Agg'.
# matplotlib.use('Agg')
import pylab
import matplotlib.pyplot as plt


def filter_cuts(L_domain, L_values, threshold=60):

    # Remove the breaks in time sequence between sessions.
    # We remove cuts that are wider than `threshold` in seconds.
    #
    # L_domain is a list of timestamps (float values representing seconds)
    # L_values is the corresponding list of values for the times

    assert len(L_domain) == len(L_values)
    if len(L_values) == 0:
        return ([], [])

    accum_time_subtraction = L_domain[0]

    L_filtered_domain = [0.0]
    L_filtered_values = [L_values[0]]
    for (t0, t1, v0, v1) in zip(L_domain, L_domain[1:], L_values, L_values[1:]):

        if t1 - t0 <= threshold:
            # normal situation
            L_filtered_domain.append(t1 - accum_time_subtraction)
            L_filtered_values.append(v1)
        else:
            # Not normal. We will throw away one point because it's hard to
            # find where to put it when we decide to glue the two sides together.
            accum_time_subtraction += t1 - t0

    return (L_filtered_domain, L_filtered_values)

def run():

    results_pickle_file = "/home/gyomalin/tmp/backup_000_iter03.pkl"
    E = pickle.load(open(results_pickle_file, "r"))

    L_service_worker = E['logging']['service_worker'].values()
    L_service_master = E['logging']['service_master'].values()
    L_service_database = E['logging']['service_database'].values()

    DL_individual_accuracy = {'train' :[], 'valid' :[], 'test' :[]}
    DL_individual_loss = {'train' :[], 'valid' :[], 'test' :[]}

    for sd in L_service_database:
        for (timestamp, e) in sd['measurement']:
            if e['name'] == 'individual_loss':
                DL_individual_loss[e['segment']].append((timestamp, e['mean']))
            elif e['name'] == 'individual_accuracy':
                DL_individual_accuracy[e['segment']].append((timestamp, e['mean']))


    print "Generating plots."

    for (measurement, DL_stv) in [('individual_accuracy', DL_individual_accuracy), ('individual_loss', DL_individual_loss)]:

        output_path = "/home/gyomalin/tmp/iter03_%s.png" % measurement
        pylab.hold(True)

        L_handles = []
        for (segment, tv) in DL_stv.items():

            # we want those values sorted by increasing timestamp
            tv = sorted(tv, key=lambda e: e[0])

            L_domain = [e[0] for e in tv]
            L_values = [e[1] for e in tv]

            (L_domain, L_values) = filter_cuts(L_domain, L_values)

            handle = pylab.plot( np.array(L_domain), np.array(L_values), label=segment, linewidth=2 )
            L_handles.append( handle )

        plt.legend(loc=7)
        pylab.savefig(output_path, dpi=250)
        pylab.close()
        print "Wrote %s." % output_path


if __name__ == "__main__":
    run()
