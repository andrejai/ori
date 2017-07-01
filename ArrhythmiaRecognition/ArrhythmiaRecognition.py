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

def seeCorrelation(lines):
    columns = []
    for j in range(0, len(lines[0])):
        column = []
        for i in range(0, len(lines)):
            if not (lines[i].__contains__("?")):
                column.append(lines[i][j])
        columns.append(column)
    for i in range(0, len(columns)):
        for j in range(0, len(columns[0])):
            columns[i][j] = float(columns[i][j])
    output = nmp.correlate(columns[0], columns[1], mode='full')
    print(output)

"""
    Loads EKG attributes from file, 452 EKG rows with 279 attributes.
    Missing columns are eliminated, rows with more than 9 zeros are eliminated.
    Returns lines-list of EKG values for 18 attributes and results-list of results of EKG
    every item in list has 16 values, one value is 1 and others are 0.
    Item with value 1 is category in witch arrhythmia is classified.
"""
def loadData():
    indexes = [0, 4, 48, 75, 90, 102, 111, 162, 166, 168, 172, 198, 206, 210, 260, 266, 270, 276, 279]
    file = open("arrhythmia.data.txt", 'r')
    lines = file.readlines()
    results = []
    selected = []

    #make every line in list of attributes
    for i in range (0, len(lines)):
        lines[i] = lines[i].split(',')

    #select only attributes that will be used for training neural network
    for i in range(0, len(lines)):
        selected.append([])
        for j in indexes:
            selected[i].append(lines[i][j])
    lines = selected

    #remove lines that have missing values or more than 9 zero values
    for line in lines:
        if(line.__contains__("?")):
            lines.remove(line)
        else:
            zeros = 0
            for attr in line:
                if(attr == "0" or attr =="0.0"):
                    zeros += 1
            if(zeros > 9):
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

def splitData(X, Y):
    XValidation, YValidation = [], []
    XTest, YTest = [], []
    XInput, YOutput = [], []
    validationStart = (int)(len(X) - len(X) * 0.1 * 2)
    testStart = (int)(len(X) - len(X) * 0.1)
    for i in range(0, validationStart):
        XInput.append(X[i])
        YOutput.append(Y[i])
    for i in range(validationStart, testStart):
        XValidation.append(X[i])
        YValidation.append(Y[i])
    for i in range(testStart, len(X)):
        XTest.append(X[i])
        YTest.append(Y[i])
    return XInput, YOutput, XValidation, YValidation, XTest, YTest

def setNN(input, output):
    XInput, YOutput, XValidation, YValidation, XTest, YTest = splitData(input, output)
    if os.path.isfile("training.net"):
        return nl.load("training.net")
    net = nl.net.newff([[0, 1000]] * len(XInput[0]), [18, 16])
    net.trainf = nl.net.train.train_rprop
    net.out_minmax = [0, 1]
    net.init()
    result = net.train(XInput, YOutput, epochs=5000, show=1, goal=0.000001)
    net.save("training.net")
    res = net.sim(XInput)
    true = 0
    for i in range(0, len(res)):
        print(max(res[i]), res[i])
        for j in range(0, 16):
            if (res[i][j] == max(res[i])):
                if (YOutput[i][j] == 1.0):
                    true += 1
    print(true)

    return net

if __name__ == '__main__':

    #reasearch columns
    input, output = loadData()
    net = setNN(input, output)
    XInput, YOutput, XValidation, YValidation, XTest, YTest = splitData(input, output)
    print(YValidation)
    res = net.sim(XValidation)
    true = 0
    for i in range(0, len(res)):
        print(max(res[i]), res[i])
        for j in range(0, 16):
            if (res[i][j] == max(res[i])):
                if (YValidation[i][j] == 1.0):
                    true += 1
    print(true)
    res = net.sim(XTest)
    true = 0
    for i in range(0, len(res)):
        print(max(res[i]), res[i])
        for j in range(0, 16):
            if (res[i][j] == max(res[i])):
                if (YTest[i][j] == 1.0):
                    true += 1
    print(len(XTest))
    print(true)

    #y = net.sim(result.net, XInput)
    #plot.plot(XInput, net.sim(XInput), col="blue", pch="+")
    #plot.plot(XInput, YOutput, col="red", pch="+")
    #net.points(XInput, YOutput, col="red", pch="x")
