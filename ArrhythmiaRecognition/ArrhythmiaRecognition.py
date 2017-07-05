import os, sys, time
import numpy as nmp
import neurolab as nl
import matplotlib.pyplot as plot
import os.path
import pylab
import math
import sys
nmp.seterr(divide='ignore', invalid='ignore') #ignore dividing with 0 problems

#classes of arrhythmia
CLASS = ["Normal", "Ischemic changes (Coronary Artery Disease)", "Old Anterior Myocardial Infarction",
         "Old Inferior Myocardial Infarction", "Sinus tachycardy", "Sinus bradycardy",
         "Ventricular Premature Contraction (PVC)", "Supraventricular Premature Contraction",
         "Left bundle branch block", "Right bundle branch block", "1. degree AtrioVentricular block",
         "2. degree AV block", "3. degree AV block", "Left ventricule hypertrophy", "Atrial Fibrillation or Flutter",
         "Others"]

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

    choosenCols = [] #indexes of columns with minimum correlations
    choosenCols.append(0)

    correlations = nmp.corrcoef(columns)

    plot.matshow(correlations, cmap=plot.cm.Greens) #show graph of correlations
    pylab.show()

    #adds only columns that dont have correlation bigger than 0,6
    for i in range(0, len(correlations)):
        for j in range(i+1, len(correlations[0])):
            if i in choosenCols:
                if (abs(correlations[i][j]) < 0.6 and (not choosenCols.__contains__(j))):
                    choosenCols.append(j)
                elif abs(correlations[i][j]) > 0.6 and choosenCols.__contains__(j):
                    choosenCols.remove(j)

    result = []
    for i in choosenCols:
        result.append(columns[i])
    return result

"""
    Loads EKG attributes from file, 452 EKG rows with 279 attributes.
    Missing columns are eliminated, rows with more than 9 zeros are eliminated.
    Returns lines-list of EKG values for 18 attributes and results-list of results of EKG
    every item in list has 16 values, one value is 1 and others are 0.
    Item with value 1 is category in witch arrhythmia is classified.
"""
def loadData():
    print("\nLoading only important 18 attributes from research...")
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
    for i in range(0, len(lines)):
        r = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        r[int(lines[i][len(lines[i])-1])-1] = 1
        results.append(r)
        lines[i].pop()
    maxVal = 0
    for i in range(0, len(lines)):
        for j in range(0, len(lines[0])):
            lines[i][j] = float(lines[i][j])
        if(maxVal < max(lines[i])):
            maxVal = max(lines[i])
    for i in range(0, len(lines)):
        for j in range(0, len(lines[0])):
            lines[i][j] = lines[i][j]/maxVal
    return lines, results

"""
    Load EKG attributes and eliminate missing values and invalid EKG results.
    Returns list of EKG results.
"""
def loadAllData():
    print("\n\nLoading all attributes from file...")
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
    for i in range(0, len(lines)):
        r = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        r[int(lines[i][len(lines[i])-1])-1] = 1
        results.append(r)
        lines[i].pop()

    maxVal = 0
    for i in range(0, len(lines)):
        for j in range(0, len(lines[0])):
            lines[i][j] = float(lines[i][j])
        if (maxVal < max(lines[i])):
            maxVal = max(lines[i])
    for i in range(0, len(lines)):
        for j in range(0, len(lines[0])):
            lines[i][j] = lines[i][j] / maxVal
    return lines, results

"""
    Loads only columns with correlations smaller than 0.6
    This function uses same data set as two functions before.
"""
def loadCorrelationCols():
    print("\n\nLoading only attributes that have correlation smaller than 0.6 for NN...")
    file = open("arrhythmia.data.txt", 'r')
    lines = file.readlines()
    results = []
    selected = []

    # make every line in list of attributes
    for i in range(0, len(lines)):
        lines[i] = lines[i].split(',')
        result = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        result[int(lines[i][len(lines[i])-1])-1] = 1
        results.append(result)
        lines[i].pop()
    cols = seeCorrelation(lines)
    for j in range(0, len(cols[0])):
        ekg = []
        for i in range(0, len(cols)):
            ekg.append(cols[i][j])
        selected.append(ekg)

    maxVal = 0
    for i in range(0, len(selected)):
        if (maxVal < max(selected[i])):
            maxVal = max(selected[i])
    for i in range(0, len(selected)):
        for j in range(0, len(selected[0])):
            selected[i][j] = selected[i][j] / maxVal
    return selected, results

"""
    Tests number of successful classified arrhythmia.
    Output is percentage.
"""
def test(net, XTest, YTest):
    netOutput = net.sim(XTest)
    true = 0
    existence = 0
    for i in range(0, len(netOutput)):
        for j in range(0, len(YTest[0])):
            if (netOutput[i][j] == max(netOutput[i])):
                if (YTest[i][j] == 1.0):
                    true += 1
                if(j == 0 and YTest[i][j] == 1.0):
                    existence += 1
                elif YTest[i][0] == 0 and j != 0:
                    existence += 1
    print("\t\tPercentage of correctly classified arrhythmia using NN is: {0}%" \
        .format((true/len(XTest))*100))
    print("\t\tPercentage of correctly discovering presence of arrhythmia using classification NN is: {0}%" \
          .format((existence/len(XTest))*100))

"""
    Testing successfully recognized arrhythmia
"""
def testOneOutput(net, XTest, YTest):
    netOutput = net.sim(XTest)
    true = 0
    for i in range(0, len(netOutput)):
        if (netOutput[i][0] > 0.6 and YTest[i][0] == 1):
            true += 1
        elif (netOutput[i][0] < 0.6 and YTest[i][0] == 0):
            true += 1
    print("\t\tPercentage of correctly discovered arrhythmia using NN is: {0}%" \
          .format((true/len(XTest))*100))


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
    Making NN and training it if it doesn't exist in path file.
"""
def setNN(input, output, path, neurons, epochs):
    print("\tResilient backpropagation network with {0} neurons in hidden layer and {1} epochs"\
          .format(neurons, epochs))
    XInput, YOutput, XTest, YTest = splitData(input, output)
    if os.path.isfile(path):
        print("\n\tLoading network from "+path)
        return nl.load(path)

    net = nl.net.newff([[-1, 1]] * len(XInput[0]), [neurons, 16])
    net.trainf = nl.net.train.train_rprop
    net.init()
    result = net.train(XInput, YOutput, epochs=epochs, show=10, goal=0.0001)
    print("\tSaving network in "+path)
    net.save(path)
    res = net.sim(XInput)
    plot.plot(XInput, YOutput)

    #how many good classifications
    true = 0
    for i in range(0, len(res)):
        for j in range(0, 16):
            if (res[i][j] == max(res[i])):
                if (YOutput[i][j] == 1.0):
                    true += 1
    return net

def setAllNN(input, output, path, neurons):
    print("\tResilient backpropagation network with {0} neurons in hidden layer and {1} epochs"\
          .format(neurons, 50))
    XInput, YOutput, XTest, YTest = splitData(input, output)
    if os.path.isfile(path):
        print("\n\tLoading network from " + path)
        return nl.load(path)
    net = nl.net.newff([[-1, 1]] * len(XInput[0]), [neurons, 16])
    net.trainf = nl.net.train.train_rprop
    net.init()
    result = net.train(XInput, YOutput, epochs=50, show=10, goal=0.00001)
    print("\tSaving network in "+path)
    net.save(path)
    res = net.sim(XInput)
    plot.plot(XInput, YOutput)

    #how many good classifications
    true = 0
    for i in range(0, len(res)):
        for j in range(0, 16):
            if (res[i][j] == max(res[i])):
                if (YOutput[i][j] == 1.0):
                    true += 1
    return net

def setNNOneOutput(input, output, path, neurons, epochs):
    print("\tResilient backpropagation network with {0} neurons in hidden layer and {1} epochs"\
          .format(neurons, epochs))
    XInput, YOutput, XTest, YTest = splitData(input, output)
    if os.path.isfile(path):
        print("\n\tLoading network from " + path)
        return nl.load(path)
    net = nl.net.newff([[-1, 1]] * len(XInput[0]), [neurons, 1])
    net.trainf = nl.net.train.train_rprop
    net.init()
    result = net.train(XInput, YOutput, epochs=epochs, show=1, goal=0.0001)
    print("\tSaving network in "+path)
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
    return net

"""
    Return [[1], [0]] when arrhythmia doesn't exist or [[0], [1]] when it does.
"""
def decreaseOutput(output):
    result = []
    for line in output:
        if line[0] == 1:
            result.append([1])
        else:
            result.append([0])
    return result

def writeTestsToFile(XTest, YTest):
    f = open('test.txt', 'w')
    testData = ""
    for i in range(0, len(XTest)):
        testData += str(XTest[i]) + " " + str(YTest[i].index(1) + 1) + "\n"
    f.write(testData)
    f.close()

def writeResult(line):
    f = open('result.txt', 'a')
    f.write(line)
    f.close()

if __name__ == '__main__':

    #all columns with 30 neurons in hidden layer
    inp, output = loadAllData()
    net_all = setAllNN(inp, output, "training_all_30_16.net", 30)
    XInput, YOutput, XTest, YTest = splitData(inp, output)
    test(net_all, XTest, YTest)

    #research columns with 32 neurons in hidden layer
    inp, output = loadData()
    net_18_100 = setNN(inp, output, "training_18_32_16_100epochs.net", 32, 100)
    XInput, YOutput, XTest, YTest = splitData(inp, output)
    writeTestsToFile(XTest, YTest)
    test(net_18_100, XTest, YTest)

    #training one output, only if arrhythmia exists with 18 attributes
    inp, output = loadData()
    output = decreaseOutput(output)
    net = setNNOneOutput(inp, output, "trainingOneOutputProba.net", 6, 500)
    XInput, YOutput, XTest, YTest = splitData(inp, output)
    testOneOutput(net, XTest, YTest)

    # Columns based on minimal correlation with 30 neurons in hidden layer
    inp, output = loadCorrelationCols()
    print("Using {0} columns"\
            .format(len(inp[0])))
    net_corr = setNN(inp, output, "training_correlation_cols.net", 30, 200)
    print("\n\tAttributes that are used were picked based on correlations between attributes.")
    XInput, YOutput, XTest, YTest = splitData(inp, output)
    test(net_corr, XTest, YTest)

    user_input = input("Enter data, press enter to finish: ")
    while(user_input != ""):

        line = []
        line.append(user_input.split(','))
        for i in range(0, len(line[0])):
           line[0][i] = float(line[0][i])
        res = net_18_100.sim(line)
        res = list(res)
        res[0] = list(res[0])
        print("Arrhythmia class: {0} ({1})" \
              .format(res[0].index(max(res[0]))+1, CLASS[res[0].index(max(res[0]))]))
        if(res[0].index(max(res[0])) == 0):
            print("Arrhythmia is not recognized! You don't have arrhythmia!")
        user_input = input("Enter data, press enter to exit: ")


