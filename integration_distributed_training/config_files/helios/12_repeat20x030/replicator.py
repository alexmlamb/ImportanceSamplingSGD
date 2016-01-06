
import re

LP_patterns = [ (r"(.*)07_paper_again/config_030.py(.*)", "12_repeat20x030/config_%0.5d.py"),
                (r"(.*)bootstrap_experiment_030(.*)", "bootstrap_experiment_%0.5d"),
                (r"(.*)030.rdb(.*)", "%0.5d.rdb"),
                (r"(.*)/rap/jvb-000-aa/data/alaingui/experiments_ISGD/030(.*)", "/rap/jvb-000-aa/data/alaingui/experiments_ISGD/%0.5d"),
                (r"(.*)07_paper_again/launch_030.sh(.*)", "12_repeat20x030/launch_%0.5d.sh")
                ]


LP_files = [ ("config_030.py", "config_%0.5d.py"), ("launch_030.sh", "launch_%0.5d.sh")]


experiment_start_index = 70
nreps = 50

L_extra_cmds_1 = []
L_extra_cmds_2 = []

for experiment_index in range(experiment_start_index, experiment_start_index + nreps):

    for src_filename, dest_filename_pattern in LP_files:
        dest_filename = dest_filename_pattern % experiment_index

        with open(src_filename, 'r') as f_in:
            with open(dest_filename, 'w') as f_out:

                for line in f_in.readlines():
                    for s,p in LP_patterns:
                        m = re.match(s, line)
                        if m:
                            line = m.group(1) + (p % experiment_index) + m.group(2) + "\n"
                            print "Performed replacement to get line : %s" % line
                            # Don't even attempt to match more than once.
                            break

                    # Write to destination regardless of whether you matches something or not.
                    f_out.write(line)

    L_extra_cmds_1.append("mkdir /rap/jvb-000-aa/data/alaingui/experiments_ISGD/%0.5d" % experiment_index)

    L_extra_cmds_2.append("cd /rap/jvb-000-aa/data/alaingui/experiments_ISGD/%0.5d" % experiment_index)
    L_extra_cmds_2.append("msub ~/Documents/ImportanceSamplingSGD/integration_distributed_training/config_files/helios/12_repeat20x030/launch_%0.5d.sh" % experiment_index)
    L_extra_cmds_2.append("")

with open("extra_cmd_1.sh", 'w') as f:
    for cmd in L_extra_cmds_1:
        f.write(cmd + "\n")

with open("extra_cmd_2.sh", 'w') as f:
    for cmd in L_extra_cmds_2:
        f.write(cmd + "\n")
