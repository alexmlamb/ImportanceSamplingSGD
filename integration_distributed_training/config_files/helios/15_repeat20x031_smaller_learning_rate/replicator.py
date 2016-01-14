
import re

LP_patterns = [ (r"(.*)07_paper_again/config_031.py(.*)", "15_repeat20x031_smaller_learning_rate/config_%0.5d.py"),
                (r"(.*)bootstrap_experiment_031(.*)", "bootstrap_experiment_%0.5d"),
                (r"(.*)031.rdb(.*)", "%0.5d.rdb"),
                (r"(.*)/rap/jvb-000-aa/data/alaingui/experiments_ISGD/031(.*)", "/rap/jvb-000-aa/data/alaingui/experiments_ISGD/%0.5d"),
                (r"(.*)07_paper_again/launch_031.sh(.*)", "15_repeat20x031_smaller_learning_rate/launch_%0.5d.sh")
                ]


LP_files = [ ("config_031.py", "config_%0.5d.py"), ("launch_031.sh", "launch_%0.5d.sh")]


experiment_start_index = 220
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
    L_extra_cmds_2.append("msub ~/Documents/ImportanceSamplingSGD/integration_distributed_training/config_files/helios/15_repeat20x031_smaller_learning_rate/launch_%0.5d.sh" % experiment_index)
    L_extra_cmds_2.append("")

with open("extra_cmd_1.sh", 'w') as f:
    for cmd in L_extra_cmds_1:
        f.write(cmd + "\n")

with open("extra_cmd_2.sh", 'w') as f:
    for cmd in L_extra_cmds_2:
        f.write(cmd + "\n")
