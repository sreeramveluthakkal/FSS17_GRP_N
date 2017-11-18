import csv
import random
from random import randint
import math
import operator

global dataSet

syntheticData = open("Synthetic.txt", "w")
def loadDataset(filename, numattrs):
    csvfile = open(filename, 'r')
    lines = csv.reader(csvfile)
    dataset = list(lines)
    for x in range(len(dataset)):
        for y in range(numattrs):
            dataset[x][y] = float(dataset[x][y])
    return dataset

def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)

def getNeighbors(trainingSet, eachMinorsample, k):
    distances = []
    length = len(eachMinorsample) - 1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(eachMinorsample, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
        distances.sort(key=operator.itemgetter(1))
        neighbors = []
    for x in range(k):
        neighbors.append(distances[x + 1][0])
    return neighbors



def seperateMinority(dataSet, MinorClassName, classColumnNumber):
    minorSamples = []
    for eachSample in dataSet:
        if (eachSample[classColumnNumber] == MinorClassName):
            minorSamples.append(eachSample)
    return minorSamples




def SMOTE(T, N, minorSamples, numattrs, dataSet, k):
    if (N <= 100):
        print "Number of sample to be generated should be more than 100%"
        raise ValueError
    N = int(N / 100) * T
    nnarray = []
    for eachMinor in minorSamples:
        nnarray = (getNeighbors(dataSet, eachMinor, k))
    populate(N, minorSamples, nnarray, numattrs)



def populate(N, minorSample, nnarray, numattrs):
    while (N > 0):
        nn = randint(0, len(nnarray) - 2)
        eachUnit = []
        for attr in range(0, numattrs+1):
            diff = float(nnarray[nn][attr]) - (minorSample[nn][attr])
            gap = random.uniform(0, 1)
            eachUnit.append(minorSample[nn][attr] + gap * diff)
        for each in eachUnit:
            syntheticData.write(str(each)+",")
        syntheticData.write("\n")
        N = N - 1

numattrs = 23
dataSet = loadDataset('velocity.csv', numattrs)
MinorClassName = "0"
minorSamples = seperateMinority(dataSet, MinorClassName, classColumnNumber=24)
NumberOfMinorSamples = len(minorSamples)
print "Number Of Minor Samples Present In Dataset : ", NumberOfMinorSamples
SMOTE(NumberOfMinorSamples, 500, minorSamples, numattrs - 1, dataSet, 3)