import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import math

#Two arguments are files for two log files.  

exp1 = sys.argv[1]
exp2 = sys.argv[2]

def log2legend(key):
    keyMap = {}
    keyMap[''] = 

def extract_error_map(fh):

    m_tr = {}
    m_va = {}
    m_te = {}

    m = {}

    m["tr"] = m_tr
    m["va"] = m_va
    m["te"] = m_te

    mb = -1
    lastSeen = "none"

    for line in fh:
        if "train" in line:
            lastSeen = "tr"
        if "valid" in line:
            lastSeen = "va"
        if "test" in line:
            lastSeen = "te"

        if "Number minibatches processed by master" in line:
            mb = int(line[:-1].split(" ")[-1].replace("None","0"))


        if "accuracy" in line and lastSeen != "none":
            acc = float(line.split(" ")[4][:-1])

            m[lastSeen][mb] = acc

    return m

logFile1 = open(exp1, "r")
logFile2 = open(exp2, "r")

m1 = extract_error_map(logFile1)
m2 = extract_error_map(logFile2)

#Plotting test accuracy.  

mb_shared = sorted(list(set(m1['te'].keys()) & set(m2['te'].keys())))

mb_arr = np.asarray(mb_shared)

ind = range(0, len(mb_shared), len(mb_shared) / 10)

ticks = mb_arr[ind]

test_1 = []
test_2 = []

for mb in mb_shared:
    test_1.append(m1['te'][mb])
    test_2.append(m2['te'][mb])

plt.plot(test_1)
plt.plot(test_2)

plt.ylim([0.7,1.0])

plt.xticks(ind, ticks)
plt.legend([log2legend(exp1), log2legend(exp2)], loc = 'lower right')
plt.xlabel("Minibatches trained")
plt.ylabel("Accuracy")
plt.title("Test Accuracy Curves")

plt.savefig("plots/test_error_comp.pdf")

plt.clf()

test_1 = []
test_2 = []

for mb in mb_shared:
    test_1.append(m1['va'][mb])
    test_2.append(m2['va'][mb])

plt.plot(test_1)
plt.plot(test_2)

plt.ylim([0.7,1.0])

plt.xticks(ind, ticks)
plt.legend([log2legend(exp1), log2legend(exp2)], loc = 'lower right')
plt.xlabel("Minibatches trained")
plt.ylabel("Accuracy")
plt.title("Validation Accuracy Curves")


plt.savefig("plots/valid_error_comp.pdf")

plt.clf()

test_1 = []
test_2 = []

for mb in mb_shared:
    test_1.append(m1['tr'][mb])
    test_2.append(m2['tr'][mb])

plt.plot(test_1)
plt.plot(test_2)

plt.ylim([0.7,1.0])

plt.xticks(ind, ticks)
plt.legend([log2legend(exp1), log2legend(exp2)], loc = 'lower right')
plt.xlabel("Minibatches trained")
plt.ylabel("Accuracy")
plt.title("Train Accuracy Curves")

plt.savefig("plots/train_error_comp.pdf")

plt.clf()

#Test error using early stopping

test_1 = []
test_2 = []

best_validation_1 = -1.0
best_validation_2 = -1.0

for mb in mb_shared:

    if m1['va'][mb] > best_validation_1:
        best_validation_1 = m1['va'][mb]
        test_1.append(m1['te'][mb])
    else:
        test_1.append(test_1[-1])

    if m2['va'][mb] > best_validation_2:
        best_validation_2 = m2['va'][mb]
        test_2.append(m2['te'][mb])
    else:
        test_2.append(test_2[-1])


plt.plot(test_1)
plt.plot(test_2)

plt.ylim([0.7,1.0])

plt.xticks(ind, ticks)
plt.legend([log2legend(exp1), log2legend(exp2)], loc = 'lower right')
plt.xlabel("Minibatches trained")
plt.ylabel("Accuracy")
plt.title("Test Accuracy Curves (only using points with highest validation accuracy)")

plt.savefig("plots/test_error_comp_early_stopping.pdf")

plt.clf()








