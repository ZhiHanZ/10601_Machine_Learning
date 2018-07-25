ddimport inspect
import math
import csv
import sys

class treeNode:
    def __init__(self, leftBranch = None, rightBranch = None,
                 featureCol = None, featureValue = None, voteResults = None):
        self.leftBranch = leftBranch
        self.rightBranch = rightBranch
        self.featureCol = featureCol
        self.featureValue = featureValue
        self.voteResults = voteResults

class leafNode:
    def __init__(self, voteResults = None):
        self.voteResults = voteResults

################################################################################
#new path file and write content into
def writeFile(filePath, trainError, testError):
    with open(filePath, "wt") as file:
        file.write("error(train): %f"%trainError + "\n" +
                   "error(test): %f"%testError)

def outputLabels(filePath, labels):
    with open(filePath, "wt") as file:
        file.write(labels)

#load data
#store rows and return allRows list
def loadData(inputFilePath):
    inputFile = open(inputFilePath)
    data = csv.reader(inputFile)
    allRows = list()
    for row in data:
        allRows.append(row)
    return allRows[1:]

#split the data into two groups
#@param allRows input data
#@columnIndex colunm index for some row
#@featureValue yes or no value
#@return two groups
def splitOn(allRows, featureIndex, featureValue):
    firstG, secondG = list(), list()
    for row in allRows:
        if row[featureIndex] == featureValue:
            firstG.append(row)
        else:
            secondG.append(row)
    groups = (firstG, secondG)
    return groups

#count the votes on input data
#@allRows input data
#@return voteCounts dict counts (yes/no : count) pairs
def countVotes(allRows):
    #lable result is at the last column
    labelIndex = len(allRows[0]) - 1
    voteResults = dict()
    for row in allRows:
        label = row[labelIndex]
        voteResults[label] = voteResults.get(label, 0) + 1
    return voteResults

#compute entropy on input data
#@allRows input data
#@return entropy on input data
def computeEntropy(allRows):
    voteResults = countVotes(allRows)
    #The first row contains feature and label names
    total = len(allRows)
    entropy = 0
    for key in voteResults.keys():
        p = float(voteResults[key]) / total
        entropy += - p * math.log(p, 2)
    return entropy

#@return feature values set and label column index
def getFeatureValues(allRows):
    labelCol = len(allRows[0]) - 1

    featureValues = list()
    #The first row contains feature and label names
    for row in allRows:
        for col in range(len(allRows[0]) - 1):
            featureValues.append(row[col])
    return (labelCol, set(featureValues))

#Learn a decision tree recursively
#@allRows current data
#@return a decision tree node
def treeLearner(allRows, maxDepth, currDepth = 1, maxInfoGain = 0):
    #global currDepth
    currNodeEntropy = computeEntropy(allRows)
    maxGainGroups = None
    featureCol = None
    splitValue = None

    labelCol, featureValues = getFeatureValues(allRows)

    for col in range(labelCol):
        for featureValue in featureValues:
            firstG, secondG = splitOn(allRows, col, featureValue)
            #compute the first group conditional entropy
            if len(firstG) > 0:
                p1 = float(len(firstG)) / len(allRows)
                conditionalEntropy1 = p1 * computeEntropy(firstG)
            else: conditionalEntropy1 = 0
            #conpute the second group conditional entropy
            if len(secondG) > 0:
                p2 = float(len(secondG)) / len(allRows)
                conditionalEntropy2 = p2 * computeEntropy(secondG)
            else: conditionalEntropy2 = 0
            #compute information gain to the current node entropy
            infoGain = currNodeEntropy - (conditionalEntropy1 + conditionalEntropy2)

            if infoGain > maxInfoGain:
                maxInfoGain = infoGain
                maxGainGroups = (firstG, secondG)
                featureCol, splitValue = col, featureValue

    if maxInfoGain > 0 and len(maxGainGroups[0]) > 0 and len(maxGainGroups[1]) > 0 and currDepth <= maxDepth:
        leftBranch = treeLearner(maxGainGroups[0], maxDepth, currDepth + 1)
        rightBranch = treeLearner(maxGainGroups[1], maxDepth, currDepth + 1)
        return treeNode(leftBranch = leftBranch, rightBranch = rightBranch,
                        featureCol = featureCol, featureValue = splitValue,
                        voteResults = countVotes(allRows))
    else:
        return leafNode(voteResults = countVotes(allRows))
        

def printTree(tree, indent = ''):
    if isinstance(tree, leafNode):
        print(indent + 'Vote Results'+ str(tree.voteResults))
    # Is this a leaf node?
    elif tree != None:
        print(str(tree.voteResults))
        # Print the criteria
        print(' Feature Column Index -> %s : %s?'%(str(tree.featureCol), str(tree.featureValue)))

        # Print the branches
        print(indent + 'True->',)
        printTree(tree.leftBranch, indent + "\t")
        print(indent + 'False->',)
        printTree(tree.rightBranch, indent + "\t")

# make predictions on input data row based on learned tree
def predict(node, row):
    while not isinstance(node, leafNode):
        col = node.featureCol
        value = node.featureValue
        if row[col] == node.featureValue:
            node = node.leftBranch
        else:
            node = node.rightBranch
    maxVote = -1
    for key in node.voteResults:
        voteNum = node.voteResults[key]
        if voteNum > maxVote:
            prediction, maxVote = key, voteNum
    return prediction

#@return prediction labels for input data allRows
def getPredictions(node, allRows):
    predictedLabels, realLabels = list(), list()
    for row in allRows:
        prediction = predict(node, row)
        realLabels.append(row[-1])
        predictedLabels.append(prediction)
    errNum, total = 0, len(allRows)
    for i in range(len(realLabels)):
        if predictedLabels[i] != realLabels[i]:
            errNum += 1
    error = float(errNum) / total

    return (error, predictedLabels)

#@trainData trainng data file path
#@testData test data file path
#@maxPath dictate the maximum lerned tree depth
#@trainOut traning results output file path
#@testOut test results output file path
#@matricsOut write in training error and test error
def decisionTree(trainData, testData, maxDepth, trainOut, testOut, metricsOut):
    trainRows = loadData(trainData)
    learnedTree = treeLearner(trainRows, maxDepth)
    printTree(learnedTree)

    trainError, trainPredictions = getPredictions(learnedTree, trainRows)

    trainLabels = ""
    for label in trainPredictions:
        trainLabels += label + "\n"
    outputLabels(trainOut, trainLabels)

    testRows = loadData(testData)
    testError, testPredictions = getPredictions(learnedTree, testRows)
    testLabels = ""
    for label in testPredictions:
        testLabels += label + "\n"
    outputLabels(testOut, testLabels)
    writeFile(metricsOut, trainError, testError)


def main():
    decisionTree(sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5], sys.argv[6])

if __name__ == '__main__':
    main()