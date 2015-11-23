
import os
import re
import numpy as np

def run():

    root_dir = "/Users/gyomalin/Documents/importance_sampling_experiments/tmp"

    database_and_master_output_path = os.path.join(root_dir, "50426.0.out")
    L_worker_output_filenames = [os.path.join(root_dir, e) for e in ["50426.2.out", "50426.3.out", "50426.4.out", "50426.5.out", "50426.6.out", "50426.7.out"]]

    with open(database_and_master_output_path) as fp:
        LD_database_results = analyze_database_output(fp.readlines())



    #for e in LD_results[-20:]:
    #    print e
    outputfile = os.path.join(root_dir, "50426.png")
    plot(outputfile, LD_database_results)



import matplotlib
# This has already been specified in .scitools.cfg
# so we don't need to explicitly pick 'Agg'.
# matplotlib.use('Agg')
import pylab
import matplotlib.pyplot as plt

def plot(outputfile, LD_database_results, dpi=150):

    t0 = LD_database_results[0]['timestamp']

    L_domain = []
    L_values = []

    for segment in ['train']:
        for measurement in ['accuracy']:

            for D_database_results in LD_database_results:
                if D_database_results['segment'] == segment and D_database_results['measurement'] == measurement:
                    mean = D_database_results['mean']
                    timestamp = D_database_results['timestamp']
                    if np.isfinite(mean):
                        L_domain.append(timestamp - t0)
                        L_values.append(mean)
                    else:
                        pass
                        #print "Invalid value for mean : ", mean

    domain = np.array(L_domain)
    values = np.array(L_values)

    print "Generating plot."

    pylab.hold(True)
    pylab.plot(domain, values, c='#0055cc')

    pylab.draw()

    pylab.savefig(outputfile, dpi=dpi)
    pylab.close()
    print "Wrote %s." % outputfile



def analyze_database_output(L_lines):

    debug_L_lines =  """
Running server. Press CTLR+C to stop. Timestamp 1446553999.079344.
-- train
---- loss : mean nan, std nan    with 0.0000 of values used.
---- accuracy : mean nan, std nan    with 0.0000 of values used.
---- gradient_variance : mean nan, std nan    with 0.0000 of values used.
-- valid
---- loss : mean nan, std nan    with 0.0000 of values used.
---- accuracy : mean nan, std nan    with 0.0000 of values used.
---- gradient_variance : mean nan, std nan    with 0.0000 of values used.
-- test
---- loss : mean 0.823988, std 1.533044    with 1.0000 of values used.
---- accuracy : mean 0.774292, std 0.418048    with 1.0000 of values used.
---- gradient_variance : mean nan, std nan    with 0.0000 of values used.
    """.split('\n')

    timestamp_prog = re.compile(r".*?Timestamp\s(\w+\.\w+).*?")
    segment_prog = re.compile(r"--\s(\w+).*?")
    measurement_prog = re.compile(r"----\s(\w+)\s:\smean\s(.*)\s*,\sstd\s(.*?)\s+.*?")

    LD_results = []

    current_timestamp = None
    current_segment = None
    current_measurement= None
    for line in L_lines:

        if len(line) == 0:
            continue

        m = timestamp_prog.match(line)
        if m:
            current_timestamp = np.float32(m.group(1))
            #print "current_timestamp found"
            continue

        m = segment_prog.match(line)
        if m:
            current_segment = m.group(1)
            #print "current_segment found"
            continue

        m = measurement_prog.match(line)
        if m:
            (current_measurement, mean, std) = (m.group(1), np.float32(m.group(2)), np.float32(m.group(3)))
            #print "current_measurement found"

            results = { 'timestamp':current_timestamp,
                        'segment':current_segment,
                        'measurement':current_measurement,
                        'mean':mean,
                        'std':std}
            #print results
            LD_results.append(results)
            continue

        #print "No match for line : %s" % line

    return LD_results


def analyze_master_output(L_lines):
    pass

def analyze_worker_output(L_lines):
    pass


if __name__ == "__main__":
    run()
