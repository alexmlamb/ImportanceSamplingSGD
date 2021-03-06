
import numpy as np
import pickle
import re
import os

from collections import defaultdict

import matplotlib
# This has already been specified in .scitools.cfg
# so we don't need to explicitly pick 'Agg'.
matplotlib.use('Agg')
import pylab
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages


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
        assert self.closed == False
        self.LL_timestamps.append(L_timestamps)

    def finalize(self, ):
        self.closed = True
        assert 0 < len(self.LL_timestamps)

        # flatten the list
        L_timestamps = sorted(sum(self.LL_timestamps, []))

        # This will be a list of pairs (t, dt) that basically says that
        # if you come after this value `t`, you should take away `dt` to
        # get your value in the final mapping.
        # Constructing `self.L_break_points` is the whole point of this function.
        self.L_break_points = []

        # This will make everything start at 0.
        t0 = L_timestamps[0]
        self.L_break_points = [(t0 - 1.0, t0)]

        #accum_time_subtraction = L_timestamps[0]
        for (t0, t1) in zip(L_timestamps, L_timestamps[1:]):

            if t1 - t0 <= self.threshold:
                # normal situation
                pass
            else:
                # Not normal. We will throw away one point because it's hard to
                # find where to put it when we decide to glue the two sides together.
                #accum_time_subtraction += t1 - t0
                self.L_break_points.append( ( 0.5*(t0+t1), t1 - t0) )

    def cut(self, L_timestamps, L_values=None):

        # This thing can work with only L_timestamps and not use L_values.
        # This can lead to a bug, though, when the user supplies L_timestamps
        # that really should be ordered (for this function to work properly)
        # and this would require L_values to also be reordered in the same way.
        #
        # Basically, you can call this without thinking too much about things
        # if you specify L_values and you let this method take care of those concerns.

        assert self.closed == True
        if L_values is not None:
            assert len(L_timestamps) == len(L_values)
            if len(L_timestamps) == 0:
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
        else:
            if len(L_timestamps) == 0:
                return np.array([])
            assert L_timestamps == sorted(L_timestamps)
            A_timestamps = np.array(L_timestamps)
            A_values = None

        A_timestamps_copy = np.copy(A_timestamps)
        for (t, dt) in self.L_break_points:
            # You really cannot be using the same `A_timestamps` to mutate
            # and to find for the locations that should be shifted.
            # You need to make a copy to do that.
            I = (t <= A_timestamps_copy)
            A_timestamps[I] -= dt

        if A_values is None:
            return A_timestamps
        else:
            return (A_timestamps, A_values)


def test_timegap_cutter():

    tgc = TimegapCutter(10)
    tgc.add_timestamps(range(1,10))
    tgc.add_timestamps(range(15,20))
    tgc.add_timestamps(range(35,40))
    tgc.finalize()

    A_timestamps = tgc.cut(range(2, 8) + range(15, 20) + range(36, 38))
    A_timestamps_ref = np.array(range(2,8) + range(15,20) + range(20, 22)) - 1
    print "tgc.L_break_points"
    print tgc.L_break_points
    print ""
    print A_timestamps
    print A_timestamps_ref
    assert np.all(A_timestamps == A_timestamps_ref)

    # Same thing again but with a gap of 5.

    tgc = TimegapCutter(5)
    tgc.add_timestamps(range(0,10))
    tgc.add_timestamps(range(15,20))
    tgc.add_timestamps(range(35,40))
    tgc.finalize()

    # [0,1,2,3,4,5,6,7,8,9,   9,10,11,12,13,   13,14,15,16,17]

    A_timestamps = tgc.cut([2,3,4,5,6,7,8] + [17,18,19] + [35,36,38])
    A_timestamps_ref = np.array([2,3,4,5,6,7,8] + [11,12,13] + [13,14,16])
    print "tgc.L_break_points"
    print tgc.L_break_points
    print ""
    print A_timestamps
    print A_timestamps_ref
    assert np.all(A_timestamps == A_timestamps_ref)

# This function is a bit outdated by TimegapCutter.
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
    #results_pickle_file = "/mnt/dodrio/recent/ICLR2016_ISGD/helios_experiments/002/backup_002.pkl"
    #results_pickle_file = "/Users/gyomalin/Documents/helios_experiments/000/000_iter8_run2.pkl"

    #for results_pickle_file in ["/mnt/dodrio/recent/ICLR2016_ISGD/helios_experiments/000/000_iter8_run2.pkl",
    #                            "/mnt/dodrio/recent/ICLR2016_ISGD/helios_experiments/001/001_iter8_run2.pkl",
    #                            "/mnt/dodrio/recent/ICLR2016_ISGD/helios_experiments/002/002_iter10_run2.pkl",
    #                            "/mnt/dodrio/recent/ICLR2016_ISGD/helios_experiments/003/003_iter10_run2.pkl",
    #                            "/mnt/dodrio/recent/ICLR2016_ISGD/helios_experiments/002/backup_002.pkl"]:

    import socket
    helios_experiments_dir = {   #"serendib":"/home/dpln/data/data_lisa_data",
                    "lambda":"/mnt/dodrio/recent/ICLR2016_ISGD/helios_experiments",
                    "szkmbp":"/Users/gyomalin/Documents/helios_experiments"}[socket.gethostname().lower()]

    # lambda
    helios_experiments_dir = "/mnt/dodrio/recent/ICLR2016_ISGD/helios_experiments"
    # szkmbp
    # helios_experiments_dir = "/Users/gyomalin/Documents/helios_experiments"

    #for results_pickle_file in [os.path.join(helios_experiments_dir, "000/000_24h.pkl"),
    #                             os.path.join(helios_experiments_dir, "001/001_24h.pkl"),
    #                             os.path.join(helios_experiments_dir, "002/002_24h.pkl"),
    #                             os.path.join(helios_experiments_dir, "003/003_24h.pkl"),
    #                             os.path.join(helios_experiments_dir, "004/004_24h.pkl"),
    #                             os.path.join(helios_experiments_dir, "005/005_24h.pkl"),
    #                             os.path.join(helios_experiments_dir, "006/006_24h.pkl"),
    #                             os.path.join(helios_experiments_dir, "007/007_24h.pkl"),
    #                             os.path.join(helios_experiments_dir, "008/008_24h.pkl")]:

    # for results_pickle_file in [os.path.join(helios_experiments_dir, "010/010.pkl"),
    #                             os.path.join(helios_experiments_dir, "011/011.pkl"),
    #                             os.path.join(helios_experiments_dir, "012/012.pkl"),
    #                             os.path.join(helios_experiments_dir, "013/013.pkl"),
    #                             os.path.join(helios_experiments_dir, "014/014.pkl"),
    #                             os.path.join(helios_experiments_dir, "015/015.pkl"),
    #                             os.path.join(helios_experiments_dir, "016/016.pkl"),
    #                             os.path.join(helios_experiments_dir, "017/017.pkl")]:

    for results_pickle_file in [os.path.join(helios_experiments_dir, "%0.3d/%0.3d.pkl" % (d, d)) for d in range(61, 70)]:

    #for results_pickle_file in [os.path.join(helios_experiments_dir, "034/034.pkl"),
    #                            os.path.join(helios_experiments_dir, "034b/034b.pkl")]:

        if not os.path.exists(results_pickle_file):
            print "Error. File %s does not exist." % results_pickle_file
            continue

        print "Processing %s." % results_pickle_file
        E = pickle.load(open(results_pickle_file, "r"))

        m = re.match(r"(.*).pkl", results_pickle_file)
        assert m

        #recorded_results = process_loss_accuracy(E, m.group(1) + "_%s.png")
        recorded_results = process_loss_accuracy(E, m.group(1) + "_%s.pdf")
        plot_output_path = m.group(1) + "_raw_accuracy_loss.pkl"
        pickle.dump(recorded_results, open(plot_output_path, "w"), protocol=pickle.HIGHEST_PROTOCOL)

        process_action_ISGD_vs_USGD(E, None)
        #process_trcov(E, m.group(1) + "_sqrttrcov.png")
        process_trcov(E, m.group(1) + "_sqrttrcov.pdf")
        #process_ratio_of_usable_importance_weights(E, m.group(1) + "_ratio_usable_importance_weights.png")
        process_ratio_of_usable_importance_weights(E, m.group(1) + "_ratio_usable_importance_weights.pdf")

        #process_profiling_time_spent_workers(E, m.group(1) + "_worker_activity_pie.png")
        process_profiling_time_spent_workers(E, m.group(1) + "_worker_activity_prop_pie.pdf", report_proportional_values=True)
        process_profiling_time_spent_workers(E, m.group(1) + "_worker_activity_separate_pie.pdf", report_proportional_values=False)

        #process_profiling_time_spent_master(E, m.group(1) + "_master_activity_pie.png")
        process_profiling_time_spent_master(E, m.group(1) + "_master_activity_prop_pie.pdf", report_proportional_values=True)
        process_profiling_time_spent_master(E, m.group(1) + "_master_activity_separate_pie.pdf", report_proportional_values=False)

    # TODO : Plot the ratios of used importance weights, just to be sure how much we're using.


def process_loss_accuracy(E, plot_output_pattern):

    #L_service_worker = E['logging']['service_worker'].values()
    #L_service_master = E['logging']['service_master'].values()
    L_service_database = E['logging']['service_database'].values()

    DL_individual_accuracy = {'train' :[], 'valid' :[], 'test' :[]}
    DL_individual_loss = {'train' :[], 'valid' :[], 'test' :[]}

    L_service_database = sorted(L_service_database, key=lambda e: e['measurement'][0][0])

    for sd in L_service_database:
        for (timestamp, e) in sd['measurement']:
            if e['name'] == 'individual_loss':
                DL_individual_loss[e['segment']].append((timestamp, e['mean']))
            elif e['name'] == 'individual_accuracy':
                DL_individual_accuracy[e['segment']].append((timestamp, e['mean']))

    print "Generating plots."

    recorded_results = {}
    for (measurement, DL_stv) in [('individual_accuracy', DL_individual_accuracy), ('individual_loss', DL_individual_loss)]:

        # setup the timecutter
        tgc = TimegapCutter(60)
        for (segment, tv) in DL_stv.items():
            L_domain = [e[0] for e in tv]
            tgc.add_timestamps( L_domain )
        tgc.finalize()

        output_path = plot_output_pattern % measurement
        pylab.hold(True)

        L_handles = []
        for (segment, tv) in DL_stv.items():

            # we want those values sorted by increasing timestamp
            # tv = sorted(tv, key=lambda e: e[0])

            L_domain = [e[0] for e in tv]
            L_values = [e[1] for e in tv]

            (L_domain, L_values) = tgc.cut(L_domain, L_values)
            #
            #(L_domain, L_values) = filter_cuts(L_domain, L_values)

            handle = pylab.plot( np.array(L_domain) / 3600, np.array(L_values), label=segment, linewidth=2 )
            L_handles.append( handle )

            N = len(L_values)
            print "Average of last values for %s %s : %f" % (measurement, segment, np.array(L_values)[(N*9/10):].mean()   )
            recorded_results[(segment, measurement)] = (np.array(L_domain), np.array(L_values))

        pylab.plot( [L_domain[0]/3600, L_domain[-1]/3600], [1.00, 1.00], '--', c="#FF7F00", linewidth=0.5)
        plt.xlabel("time in hours")
        if measurement == "individual_accuracy":
            plt.ylim([0.70, 1.05])
            plt.title("Prediction accuracy over whole dataset")
        elif measurement == "individual_loss":
            plt.title("Loss over whole dataset")

        # http://stackoverflow.com/questions/14442099/matplotlib-how-to-show-all-digits-on-ticks
        xx, locs = plt.xticks()
        ll = ['%.2f' % a for a in xx]
        plt.xticks(xx, ll)

        plt.legend(loc=7)

        if re.match(r".*\.pdf", output_path):
            with PdfPages(output_path) as pdf:
                pdf.savefig()
        else:
            pylab.savefig(output_path, dpi=250)

        pylab.close()
        print "Wrote %s." % output_path

    return recorded_results



# This method is only there to debug some problems when the plotting
# just isn't doing what it's supposed to do.
def DEBUG_process_loss_accuracy(E, plot_output_pattern):

    #L_service_worker = E['logging']['service_worker'].values()
    #L_service_master = E['logging']['service_master'].values()
    L_service_database = E['logging']['service_database'].values()

    DL_individual_accuracy = {'train' :[], 'valid' :[], 'test' :[]}
    DL_individual_loss = {'train' :[], 'valid' :[], 'test' :[]}

    L_service_database = sorted(L_service_database, key=lambda e: e['measurement'][0][0])

    print "Generating plots."

    for measurement in ['individual_accuracy', 'individual_loss']:

        output_path = plot_output_pattern % measurement
        pylab.hold(True)

        L_handles = []

        for segment in ['train', 'valid', 'test']:
            for sd in L_service_database:

                L_domain = []
                L_values = []

                for (timestamp, e) in sd['measurement']:
                    if e['name'] == measurement and e['segment'] == segment:
                        L_domain.append(timestamp)
                        L_values.append(e['mean'])

                handle = pylab.plot( np.array(L_domain), np.array(L_values), label=segment, linewidth=2 )
                L_handles.append( handle )

        plt.title("%s over whole dataset" % measurement)
        plt.xlabel("time in seconds")
        #plt.legend(loc=7)
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






def process_trcov(E, output_path):

    #L_service_worker = E['logging']['service_worker'].values()
    #L_service_master = E['logging']['service_master'].values()
    L_service_database = E['logging']['service_database'].values()

    # Disclaimer : This function is getting ugly.
    #              However, there are little irregularities which make this
    #              completely unreadable if we try to factor it. Maybe.

    LP_approx_mu2 = []
    LP_extra_staleisgd2 = []
    LP_usgd2 = []
    LP_staleisgd2 = []
    LP_isgd2 = []

    LP_usgd2_minusmu2 = []
    LP_staleisgd2_minusmu2 = []
    LP_isgd2_minusmu2 = []
    LP_extra_staleisgd2_minusmu2 = []

    additive_importance_const_str = None
    for sd in L_service_database:
        for (timestamp, e) in sd['SGD_trace_variance']:
            if 'approx_mu2' in e:
                LP_approx_mu2.append((timestamp, e['approx_mu2']))
            if 'usgd2' in e:
                LP_usgd2.append((timestamp, e['usgd2']))
                LP_usgd2_minusmu2.append((timestamp, e['usgd2'] - e['approx_mu2']))
            if 'staleisgd2' in e:
                LP_staleisgd2.append((timestamp, e['staleisgd2']))
                LP_staleisgd2_minusmu2.append((timestamp, e['staleisgd2'] - e['approx_mu2']))
            if 'isgd2' in e:
                LP_isgd2.append((timestamp, e['isgd2']))
                LP_isgd2_minusmu2.append((timestamp, e['isgd2'] - e['approx_mu2']))
            # This one is a bit different because we're deadling with a dictionary
            if 'extra_staleisgd2' in e:
                B = e['extra_staleisgd2'].items()
                if 0 < len(B):
                    (additive_importance_const_str, v) = e['extra_staleisgd2'].items()[0]
                    LP_extra_staleisgd2.append((timestamp, v))
                    LP_extra_staleisgd2_minusmu2.append((timestamp, v - e['approx_mu2']))

    tgc = TimegapCutter(60)
    tgc.add_timestamps( [e[0] for e in LP_approx_mu2] )
    tgc.add_timestamps( [e[0] for e in LP_usgd2] )
    tgc.add_timestamps( [e[0] for e in LP_staleisgd2] )
    tgc.add_timestamps( [e[0] for e in LP_isgd2] )
    tgc.add_timestamps( [e[0] for e in LP_extra_staleisgd2] )
    tgc.finalize()

    pylab.hold(True)
    L_handles = []

    L_domain = [e[0] for e in LP_usgd2_minusmu2]
    L_values = [e[1] for e in LP_usgd2_minusmu2]
    (L_domain, L_values) = tgc.cut(L_domain, L_values)

    handle = pylab.plot( np.array(L_domain) / 3600, np.sqrt(np.array(L_values)), label='SGD', linewidth=2, c='#0000FF' )
    L_handles.append( handle )

    L_domain = [e[0] for e in LP_staleisgd2_minusmu2]
    L_values = [e[1] for e in LP_staleisgd2_minusmu2]
    (L_domain, L_values) = tgc.cut(L_domain, L_values)
    handle = pylab.plot( np.array(L_domain) / 3600, np.sqrt(np.array(L_values)), label='ISSGD stale', linewidth=2, c='#00a65a' )
    L_handles.append( handle )

    L_domain = [e[0] for e in LP_isgd2_minusmu2]
    L_values = [e[1] for e in LP_isgd2_minusmu2]
    (L_domain, L_values) = tgc.cut(L_domain, L_values)
    handle = pylab.plot( np.array(L_domain) / 3600, np.sqrt(np.array(L_values)), label='ISSGD ideal', linewidth=2, c='#bc0024' )
    L_handles.append( handle )

    #params = {  'legend.fontsize': 20, 'font.size':16   }
    #pylab.rcParams.update(params)

    # http://stackoverflow.com/questions/14442099/matplotlib-how-to-show-all-digits-on-ticks
    xx, locs = plt.xticks()
    ll = ['%.2f' % a for a in xx]
    plt.xticks(xx, ll)

    # remove this to get the actual curve
    plt.ylim([0.0, 100.0])

    #plt.title("Square root of Trace(Cov) computed over whole dataset")
    plt.ylabel("sqrt(tr(cov))")
    plt.xlabel("time in hours")
    plt.legend(loc=2)
    if re.match(r".*\.pdf", output_path):
        with PdfPages(output_path) as pdf:
            pdf.savefig()
    else:
        pylab.savefig(output_path, dpi=250)
    pylab.close()
    print "Wrote %s." % output_path





def process_ratio_of_usable_importance_weights(E, output_path):

    if not E.has_key('logging'):
        print "Skipping %s because we're missing E['logging']." % output_path
        return

    if not E['logging'].has_key('service_master'):
        print "Skipping %s because we're missing E['logging']['service_master']." % output_path
        return

    L_service_master = E['logging']['service_master'].values()

    LPL_results = []
    for sm in L_service_master:
        L_domain = []
        L_values_ratio = []
        #L_values_entropy = []
        if not sm.has_key('importance_weights_statistics'):
            print "Skipping %s because we're missing importance_weights_statistics in one master." % output_path
            return

        for (timestamp, e) in sm['importance_weights_statistics']:
            L_domain.append(timestamp)
            L_values_ratio.append(e['ratio_of_usable_importance_weights'])
            #L_values_entropy.append(e['importance_weights:ratio_satisfying:staleness_threshold_seconds'])

        LPL_results.append( (L_domain, L_values_ratio) )



    print "Generating plots."

    pylab.hold(True)
    #L_handles = []

    L_domain_flattened = sum([e[0] for e in LPL_results], [])
    L_values_flattened = sum([e[1] for e in LPL_results], [])

    # setup the timecutter
    tgc = TimegapCutter(60)
    tgc.add_timestamps( L_domain_flattened )
    tgc.finalize()


    (L_domain_flattened, L_values_flattened) = tgc.cut(L_domain_flattened, L_values_flattened)

    # smoothing taken from
    #     http://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.convolve.html#scipy.signal.convolve
    from scipy import signal
    win = signal.hann(20)
    L_values_flattened_filtered = signal.convolve(L_values_flattened, win, mode='same') / sum(win)

    handle = pylab.plot( np.array(L_domain_flattened), np.array(L_values_flattened_filtered), linewidth=2 )
    #handle = pylab.plot( np.array(L_domain_flattened), np.array(L_values_flattened), linewidth=2 )
    #L_handles.append( handle )

    plt.title("ratio of usable importance weights")
    plt.xlabel("time in seconds")
    #plt.legend(loc=7)
    pylab.savefig(output_path, dpi=250)
    pylab.close()
    print "Wrote %s." % output_path




def process_profiling_time_spent_workers(E, output_path, report_proportional_values=True):

    if not E.has_key('logging'):
        print "Skipping %s because we're missing E['logging']." % output_path
        return

    if not E['logging'].has_key('service_worker'):
        print "Skipping %s because we're missing E['logging']['service_worker']." % output_path
        return

    L_service_worker = E['logging']['service_worker'].values()

    D_worker_total_time_spent = defaultdict(float)
    D_worker_total_time_counts = defaultdict(int)

    for sw in L_service_worker:
        if  not sw.has_key('timing_profiler'):
            print "Error. Failed to find the timing_profiler so we'll skip this whole experiment."
            return
        for (timestamp, e) in sw['timing_profiler']:
            for (k, v) in e.items():
                if k in ['mode']:
                    # omit that one since it's a string that says "USGD" or "ISGD"
                    continue
                D_worker_total_time_spent[k] += v
                D_worker_total_time_counts[k] += 1

    L_things_to_compare = [ ('read_params_from_database', 'db params', D_worker_total_time_spent['sync_params_from_database'], D_worker_total_time_counts['sync_params_from_database']),
                            ('move_params_to_GPU', 'GPU params', D_worker_total_time_spent['model_api.set_serialized_parameters'], D_worker_total_time_counts['model_api.set_serialized_parameters']),
                            ('send_prob_weights_to_database', 'db weights',  D_worker_total_time_spent['send_measurements_to_database'], D_worker_total_time_counts['send_measurements_to_database']),
                            ('process_minibatch', 'fwd-back', D_worker_total_time_spent['worker_process_minibatch'], D_worker_total_time_counts['worker_process_minibatch']) ]

    total_time = sum([e[2] for e in L_things_to_compare])
    for (lk, sk, v, c) in L_things_to_compare:
        print "Worker activity %s took %0.3f of %f seconds." % (lk, v / total_time, total_time)

    long_labels, short_labels, sizes, counts = zip(*L_things_to_compare)
    if report_proportional_values:
        pass
    else:
        sizes = [e / z for (e, z) in zip(sizes, counts)]

    z = sum(e for e in sizes)
    sizes = [e / z for e in sizes]



    # http://matplotlib.org/examples/pie_and_polar_charts/pie_demo_features.html
    colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral']
    explode = (0, 0, 0, 0)
    patches, texts, autotexts = plt.pie(sizes, explode=explode, labels=short_labels, colors=colors,
                    autopct='%1.1f%%', shadow=False, startangle=90)

    plt.legend(patches, long_labels, loc="best")

    # Set aspect ratio to be equal so that pie is drawn as a circle.
    plt.axis('equal')

    if re.match(r".*\.pdf", output_path):
        with PdfPages(output_path) as pdf:
            pdf.savefig()
    else:
        pylab.savefig(output_path, dpi=250)
    pylab.close()
    print "Wrote %s." % output_path




def process_profiling_time_spent_master(E, output_path, report_proportional_values=True):

    if not E.has_key('logging'):
        print "Skipping %s because we're missing E['logging']." % output_path
        return

    if not E['logging'].has_key('service_master'):
        print "Skipping %s because we're missing E['logging']['service_master']." % output_path
        return

    L_service_master = E['logging']['service_master'].values()

    D_master_total_time_spent = defaultdict(float)
    D_master_total_counts = defaultdict(int)

    for sm in L_service_master:
        if  not sm.has_key('timing_profiler'):
            print "Error. Failed to find the timing_profiler so we'll skip this whole experiment."
            return

        for (timestamp, e) in sm['timing_profiler']:
            for (k, v) in e.items():
                if k in ['mode']:
                    # omit that one since it's a string that says "USGD" or "ISGD"
                    continue
                D_master_total_time_spent[k] += v
                D_master_total_counts[k] += 1

    L_things_to_compare = [ ('read prob weights from database', 'db weights', D_master_total_time_spent['refresh_importance_weights'], D_master_total_counts['refresh_importance_weights']),
                            ('sample indices', 'indices', D_master_total_time_spent['sample_indices_and_scaling_factors'], D_master_total_counts['sample_indices_and_scaling_factors']),
                            ('process minibatch', 'fwd-back', D_master_total_time_spent['master_process_minibatch'], D_master_total_counts['master_process_minibatch']),
                            ('read params from GPU', 'GPU params', D_master_total_time_spent['read_parameters_from_model'], D_master_total_counts['read_parameters_from_model']),
                            ('send params to database', 'db params', D_master_total_time_spent['sync_params_to_database'], D_master_total_counts['sync_params_to_database'])                             ]

    total_time = sum([e[2] for e in L_things_to_compare])
    for (lk, sk, v, c) in L_things_to_compare:
        print "Master activity %s took %0.3f of %f seconds." % (lk, v / total_time, total_time)

    long_labels, short_labels, sizes, counts = zip(*L_things_to_compare)
    if report_proportional_values:
        pass
    else:
        sizes = [e / z for (e, z) in zip(sizes, counts)]

    z = sum(e for e in sizes)
    sizes = [e / z for e in sizes]


    # http://matplotlib.org/examples/pie_and_polar_charts/pie_demo_features.html
    #sizes = [e / z for e in sizes]
    colors = ['yellowgreen', 'gold', '#ff0031', 'lightskyblue', '#ffc4fc']
    explode = (0, 0, 0, 0, 0)
    patches, texts, autotexts = plt.pie(sizes, explode=explode, labels=short_labels, colors=colors,
                    autopct='%1.1f%%', shadow=False, startangle=90)

    plt.legend(patches, long_labels, loc="best")

    # Set aspect ratio to be equal so that pie is drawn as a circle.
    plt.axis('equal')

    if re.match(r".*\.pdf", output_path):
        with PdfPages(output_path) as pdf:
            pdf.savefig()
    else:
        pylab.savefig(output_path, dpi=250)
    pylab.close()
    print "Wrote %s." % output_path




if __name__ == "__main__":
    run()
    #test_timegap_cutter()
