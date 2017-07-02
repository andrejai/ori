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
            print("Correlation for "+str(i)+" "+str(j))
            print(nmp.corrcoef(columns[i], columns[j]))


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

    seeCorrelation(lines)
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


"""
    Tests number of successful classified arrhythmia.
    Output is percentage.
"""
def test(net, XTest, YTest):
    netOutput = net.sim(XTest)
    true = 0
    for i in range(0, len(netOutput)):
        for j in range(0, 16):
            if (netOutput[i][j] == max(netOutput[i])):
                if (YTest[i][j] == 1.0):
                    true += 1
    print(true/len(XTest))


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
    XInput, YOutput, XTest, YTest = splitData(input, output)
    if os.path.isfile("training.net"):
        return nl.load("training.net")
    net = nl.net.newff([[0, 400]] * len(XInput[0]), [32, 16])
    net.trainf = nl.net.train.train_rprop
    net.out_minmax = [0, 1]
    net.init()
    result = net.train(XInput, YOutput, epochs=5000, show=1, goal=0.00001)
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

if __name__ == '__main__':

    #reasearch columns
    input, output = loadData()
    net = setNN(input, output)
    XInput, YOutput, XTest, YTest = splitData(input, output)
    test(net, XTest, YTest)

