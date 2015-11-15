
import numpy as np
import pickle
import re

import matplotlib
# This has already been specified in .scitools.cfg
# so we don't need to explicitly pick 'Agg'.
# matplotlib.use('Agg')
import pylab
import matplotlib.pyplot as plt


class TimegapCutter(object):

    # Remove the breaks in time sequence between sessions.
    # We remove cuts that are wider than `threshold` in seconds.
    #

    def __init__(self, threshold):
        assert 0 < threshold
        self.LL_timestamps = []
        self.threshold = threshold
        self.closed = False

    def add_timestamps(self, L_timestamps):
        self.LL_timestamps.append(L_timestamps)

    def finalize(self, ):
        self.closed = False
        assert 0 < len(self.LL_timestamps)

        # flatten the list
        L_timestamps = sorted(sum(LL_timestamps, []))

        # This will be a list of pairs (t, dt) that basically says that
        # if you come after this value `t`, you should take away `dt` to
        # get your value in the final mapping.
        # Constructing `self.L_break_points` is the whole point of this function.
        self.L_break_points = []

        #accum_time_subtraction = L_timestamps[0]
        for (t0, t1) in zip(L_timestamps, L_timestamps[1:]):

            if t1 - t0 <= threshold:
                # normal situation
                pass
            else:
                # Not normal. We will throw away one point because it's hard to
                # find where to put it when we decide to glue the two sides together.
                #accum_time_subtraction += t1 - t0
                self.L_break_points.append( t0, t1 - t0 )

    def cut(self, L_timestamps):
        assert self.closed == False
        assert len(L_timestamps) == len(L_values)
        if len(L_values) == 0:
            return (np.array([]), np.array([]))

        # This thing can fail in a horrible way if the
        # timestamps are not sorted. Might as well sort
        # everything over all the time to be sure.
        tv = zip(L_timestamps, L_values)
        tv = sorted(tv, key=lambda e: e[0])
        L_timestamps = [e[0] for e in tv]
        L_values = [e[1] for e in tv]

        A_timestamps = np.array(L_timestamps)
        A_values = np.array(L_values)

        for (t, dt) in self.breakpoints:
            I = t < A_timestamps
            A_values[I] -= dt

        return (A_timestamps, A_values)

    # TODO : Wait ... this thing needs the timestamps to be ordered, but it does not need the values to be given as argument.
    #        We can sort them elsewhere !
    #        Maybe have them as optional argument ?



def test_timegap_cutter():

    tgc = TimegapCutter(10)
    tgc.add_timestamps(range(0,10))
    tgc.add_timestamps(range(15,20))
    tgc.add_timestamps(range(35,40))
    tgc.finalize()

    (A_timestamps, A_values) = tgc.cut(range(0, 8), range(15, 20))

    A_timestamps_ref = np.array(range(0,10) + range(10,15) + range()


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

    #results_pickle_file = "/home/gyomalin/tmp/backup_000_iter24.pkl"
    results_pickle_file = "/mnt/dodrio/recent/ICLR2016_ISGD/helios_experiments/003/003_iter10_run2.pkl"
    E = pickle.load(open(results_pickle_file, "r"))

    m = re.match(r"(.*).pkl", results_pickle_file)
    assert m
    plot_output_pattern = m.group(1) + "_%s.png"
    process_loss_accuracy(E, plot_output_pattern)
    process_action_ISGD_vs_USGD(E, plot_output_pattern)

def process_loss_accuracy(E, plot_output_pattern):

    #L_service_worker = E['logging']['service_worker'].values()
    #L_service_master = E['logging']['service_master'].values()
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

        output_path = plot_output_pattern % measurement
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

def process_action_ISGD_vs_USGD(E, plot_output_pattern):

    #L_service_worker = E['logging']['service_worker'].values()
    L_service_master = E['logging']['service_master'].values()
    #L_service_database = E['logging']['service_database'].values()

    counts = {'USGD':0, 'ISGD':0}
    for m in L_service_master:
        for (timestamp, e) in m['event']:
            #[1447495834.662256, u'Master proceeding with round of USGD.']
            if e == "Master proceeding with round of USGD.":
                counts['USGD'] += 1
            elif e == "Master proceeding with round of ISGD.":
                counts['ISGD'] += 1

    print counts






def process_trcov(E, plot_output_pattern):

    #L_service_worker = E['logging']['service_worker'].values()
    #L_service_master = E['logging']['service_master'].values()
    L_service_database = E['logging']['service_database'].values()

    LT_approx_mu2 = []
    LT_extra_staleisgd2 = []
    LT_usgd2 = []
    LT_staleisgd2 = []
    LT_isgd2 = []

    additive_importance_const_str = None
    for sd in L_service_database:
        for (timestamp, e) in sd['SGD_trace_variance']:
            if 'approx_mu2' in e:
                LT_approx_mu2.append((timestamp, e['approx_mu2']))
            if 'usgd2' in e:
                LT_usgd2.append((timestamp, e['usgd2']))
            if 'staleisgd2' in e:
                LT_staleisgd2.append((timestamp, e['staleisgd2']))
            if 'isgd2' in e:
                LT_isgd2.append((timestamp, e['isgd2']))
            # This one is a bit different because we're deadling with a dictionary
            if 'extra_staleisgd2' in e:
                (additive_importance_const_str, v) = e['extra_staleisgd2'].items()[0]
                LT_extra_staleisgd2.append((timestamp, v))






    print "Generating plots."

    for (measurement, DL_stv) in [('individual_accuracy', DL_individual_accuracy), ('individual_loss', DL_individual_loss)]:

        output_path = plot_output_pattern % measurement
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
