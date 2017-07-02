"""
    This function loads data from txt file
    File contains 452 rows of data with 280 column values(attributes) for each row
    Function removes columns and then rows with missing values or all 0 values
"""

import os, sys, time
import numpy as nmp
import neurolab as nl
import matplotlib.pyplot as plot
import os.path

"""
    Correlation of every two columns that don't have missing values is counted.
    Columns represent values of one attribute of EKG
"""
def seeCorrelation(lines):
    columns = []

    #Adds every column in columns which don't have missing values
    for j in range(0, len(lines[0])):
        column = []
        for i in range(0, len(lines)):
            if not (lines[i][j] == "?"):
                column.append(lines[i][j])
        if(len(column) == len(lines)):
            columns.append(column)

    #From string to float attribute values
    for i in range(0, len(columns)):
        for j in range(0, len(columns[0])):
            columns[i][j] = float(columns[i][j])

    #For every two columns counts correlation
    for i in range(0, len(columns)):
        for j in range(i+1, len(columns)):
            nmp.corrcoef(columns[i], columns[j])


"""
    Loads EKG attributes from file, 452 EKG rows with 279 attributes.
    Missing columns are eliminated, rows with more than 9 zeros are eliminated.
    Returns lines-list of EKG values for 18 attributes and results-list of results of EKG
    every item in list has 16 values, one value is 1 and others are 0.
    Item with value 1 is category in witch arrhythmia is classified.
"""
def loadData():
    print("\nLoading only important attributes...")
    indexes = [0, 4, 48, 75, 90, 102, 111, 162, 166, 168, 172, 198, 206, 210, 260, 266, 270, 276, 279]
    file = open("arrhythmia.data.txt", 'r')
    lines = file.readlines()
    results = []
    selected = []

    #make every line in list of attributes
    for i in range (0, len(lines)):
        lines[i] = lines[i].split(',')

    #seeCorrelation(lines)
    #select only attributes that will be used for training neural network
    for i in range(0, len(lines)):
        selected.append([])
        for j in indexes:
            selected[i].append(lines[i][j])
    lines = selected

    #remove lines that have missing values or more than 9 zero values
    forDeleting = []
    for line in lines:
        if(line.__contains__("?")):
            forDeleting.append(line)
        else:
            zeros = 0
            for attr in line:
                if(attr == "0" or attr =="0.0"):
                    zeros += 1
            if(zeros > 9):
                forDeleting.append(line)
    for line in forDeleting:
        lines.remove(line)

    #make list of outputs
    for line in lines:
        r = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        r[int(line[len(line)-1])-1] = 1
        results.append(r)
        line.pop()

    for i in range(0, len(lines)):
        for j in range(0, len(lines[0])):
            lines[i][j] = float(lines[i][j])
    return lines, results


def loadAllData():
    print("\nLoading all attributes...")
    file = open("arrhythmia.data.txt", 'r')
    lines = file.readlines()
    results = []

    # make every line in list of attributes
    for i in range(0, len(lines)):
        lines[i] = lines[i].split(',')

    #remove columns with all ?
    forRemove = []
    for j in range(0, len(lines[0])):
        missing = 0
        for i in range(0, len(lines)):
            if(lines[i][j] == "?" or lines[i][j] == '?'):
                missing += 1
        if(missing > (len(lines)-200)):
            forRemove.append(j)
    for i in range(0, len(lines)):
        newLine = []
        for j in range(0, len(lines[i])):
            if not forRemove.__contains__(j):
                newLine.append(lines[i][j])
        lines[i] = newLine

    # remove lines that have missing values or more than half zero values
    forDeleting = []
    for line in lines:
        for i in range(0, len(line)):
            if line[i] == '?':
                forDeleting.append(line)
                i = len(line)
        if not forDeleting.__contains__(line):
            zeros = 0
            for attr in line:
                if (attr == "0" or attr == "0.0"):
                    zeros += 1
            if (zeros > len(line)/2):
                forDeleting.append(line)

    for line in forDeleting:
        lines.remove(line)

    # make list of outputs
    for line in lines:
        r = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        r[int(line[len(line) - 1]) - 1] = 1
        results.append(r)
        line.pop()

    for i in range(0, len(lines)):
        for j in range(0, len(lines[0])):
            lines[i][j] = float(lines[i][j])
    return lines, results

"""
    Tests number of successful classified arrhythmia.
    Output is percentage.
"""
def test(net, XTest, YTest):
    netOutput = net.sim(XTest)
    true = 0
    for i in range(0, len(netOutput)):
        for j in range(0, len(YTest[0])):
            if (netOutput[i][j] == max(netOutput[i])):
                if (YTest[i][j] == 1.0):
                    true += 1
    print("\n\n\n\tPercentage of correctly classified arrhythmia using NN is: {0}" \
        .format(true/len(XTest)))

"""
    Testing successfully recognized arrhythmia
"""
def testOneOutput(net, XTest, YTest):
    netOutput = net.sim(XTest)
    true = 0
    for i in range(0, len(netOutput)):
        if (abs(netOutput[i][0]) > 0.6 and YTest[i][0] == 1):
            true += 1
        elif (abs(netOutput[i][0]) < 0.6 and YTest[i][0] == 0):
            true += 1
    print("\n\n\n\tPercentage of correctly discovered arrhythmia using NN is: {0}" \
          .format(true / len(XTest)))


"""
    Splits data in two arrays, first is 80% of data and it will be used for training NN.
    20% is for testing NN.
"""
def splitData(X, Y):
    XTest, YTest = [], []
    XInput, YOutput = [], []
    validationStart = (int)(len(X) - len(X) * 0.1 * 2)
    for i in range(0, validationStart):
        XInput.append(X[i])
        YOutput.append(Y[i])
    for i in range(validationStart, len(X)):
        XTest.append(X[i])
        YTest.append(Y[i])
    return XInput, YOutput, XTest, YTest


"""
    Making NN and training it if it doesn't exist in training.net file.
"""
def setNN(input, output):
    print("\n\tResilient backpropagation network, important attributes only...")
    XInput, YOutput, XTest, YTest = splitData(input, output)
    if os.path.isfile("training.net"):
        return nl.load("training.net")
    net = nl.net.newff([[0, 400]] * len(XInput[0]), [32, 16])
    net.trainf = nl.net.train.train_rprop
    net.out_minmax = [0, 1]
    net.init()
    result = net.train(XInput, YOutput, epochs=10000, show=1, goal=0.0001)
    net.save("training.net")
    res = net.sim(XInput)
    plot.plot(XInput, YOutput)

    #how many good classifications
    true = 0
    for i in range(0, len(res)):
        print(max(res[i]), res[i])
        for j in range(0, 16):
            if (res[i][j] == max(res[i])):
                if (YOutput[i][j] == 1.0):
                    true += 1
    print(true)
    return net

def setAllNN(input, output):
    print("\n\tResilient backpropagation network, all attributes...")
    XInput, YOutput, XTest, YTest = splitData(input, output)
    if os.path.isfile("trainingAll.net"):
        return nl.load("trainingAll.net")
    net = nl.net.newff([[0, 1000]] * len(XInput[0]), [30, 16])
    net.trainf = nl.net.train.train_rprop
    net.init()
    result = net.train(XInput, YOutput, epochs=50, show=1, goal=0.00001)
    net.save("trainingAll.net")
    res = net.sim(XInput)
    plot.plot(XInput, YOutput)

    #how many good classifications
    true = 0
    for i in range(0, len(res)):
        print(max(res[i]), res[i])
        for j in range(0, 16):
            if (res[i][j] == max(res[i])):
                if (YOutput[i][j] == 1.0):
                    true += 1
    print(true/len(XInput))
    print(true)
    return net

def setNNOneOutput(input, output, path, neurons):
    print("\n\tResilient backpropagation network, recognizing existence of arrhythmia...")
    XInput, YOutput, XTest, YTest = splitData(input, output)
    if os.path.isfile(path):
        return nl.load(path)
    net = nl.net.newff([[0, 400]] * len(XInput[0]), [neurons, 1])
    net.trainf = nl.net.train.train_bfgs
    net.init()
    result = net.train(XInput, YOutput, epochs=200, show=1, goal=0.0001)
    net.save(path)
    res = net.sim(XInput)
    plot.plot(XInput, YOutput)

    #how many good classifications
    true = 0
    for i in range(0, len(res)):
        if (abs(res[i][0]) > 0.6 and YOutput[i][0] == 1):
            true += 1
        elif (abs(res[i][0]) < 0.6 and YOutput[i][0] == 0):
            true += 1
    print(true)
    return net


def decreaseOutput(output):
    result = []
    for line in output:
        if line[0] == 1:
            result.append([1])
        else:
            result.append([0])
    return result

if __name__ == '__main__':

    #all columns
    input, output = loadAllData()
    net = setAllNN(input, output)
    XInput, YOutput, XTest, YTest = splitData(input, output)
    test(net, XTest, YTest)

    #research columns
    input, output = loadData()
    net = setNN(input, output)
    XInput, YOutput, XTest, YTest = splitData(input, output)
    test(net, XTest, YTest)

    #training one output, only if arrhythmia exists with 18 attributes
    input, output = loadData()
    output = decreaseOutput(output)
    net = setNNOneOutput(input, output, "trainingOneOutput.net", 2)
    XInput, YOutput, XTest, YTest = splitData(input, output)
    testOneOutput(net, XTest, YTest)
