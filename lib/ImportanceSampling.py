import numpy
import random


def sampleInstances(indexLst, cMap_unsmoothed, batch_size, fMap, mbIndex):
    #freshness_threshold = 1

    freshness_threshold = 1

    epsilon = 0.0


    cMap = {}
    for key in cMap_unsmoothed:
        cMap[key] = (cMap_unsmoothed[key] + epsilon)
    weightMap = {}
    usedKeys = []
    costLst = []
    sumCost = 0.0
    sumCost_unsmoothed = 0.0
    for key in cMap:
        if abs(mbIndex - fMap[key]) <= freshness_threshold:
            sumCost_unsmoothed += cMap_unsmoothed[key]
            sumCost += cMap[key]
            usedKeys += [key]
            costLst += [cMap[key]]
    for key in cMap:
        if abs(mbIndex - fMap[key]) <= freshness_threshold:
            weightMap[key] = cMap[key] * 1.0 / sumCost


    avgCost = sumCost_unsmoothed / len(weightMap)
    medianCost = sorted(costLst)[len(costLst) / 2]
    weightLst = []
    for key in sorted(weightMap.keys()):
        weightLst += [weightMap[key]]


    selectedIndicesRaw = numpy.random.choice(len(weightMap),batch_size,p=weightLst, replace = False).tolist()
    selectedIndices = [usedKeys[j] for j in selectedIndicesRaw]
    cmKeys = cMap.keys()
    newIndexLst = []
    impWeightLst = []

    if random.uniform(0,1) < 0.001:
        print "Top gradients", "\t", sorted(cMap.items(), key=(lambda x: x[1]))[-100:]

    sumSelected = 0.0
    for index in selectedIndices:
        newIndexLst += [cmKeys[index]]
        impWeightLst += [1.0]#[avgCost * 1.0 / cMap[cmKeys[index]]]
        sumSelected += cMap[cmKeys[index]]

    return newIndexLst, impWeightLst


if __name__ == "__main__": 

    indexLst = range(0,6)
    cMap = {}
    cMap[0] = 10000.0
    cMap[1] = 20000.0
    cMap[2] = 0.0
    cMap[3] = 0.0
    batch_size = 2
    fMap = {}
    fMap[0] = 1
    fMap[1] = 1
    fMap[2] = 1
    fMap[3] = 1
    mbIndex = 1

    indSampled = sampleInstances(indexLst, cMap, batch_size, fMap, mbIndex)

    print indSampled

#Test with synthetic data.  


