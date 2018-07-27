import math
import copy
import numpy as np
import csv
import sys

# load data and store all the rows from input file
def loadData(filePath):
    file = open(filePath)
    data = csv.reader(file)
    allRows = list()
    for row in data:
        if len(row) == 0:
            allRows.append([])
        elif len(row) != 0:
            newRow = row[0].split('\t')
            allRows.append(newRow)
    return allRows

# process all the rows to get
# distinctWords -> a dic contains unique word mapping to position index
# classes -> a dic contains classes mapping to position index
# inverseClasses -> a dic contains position index mapping to classes
def preprocess_model1(allRows):
    distinctWords = dict()
    classes, inverseClasses = dict(), dict()
    for row in allRows:
        if len(row) != 0:
            if row[0] not in distinctWords:
                distinctWords[row[0]] = len(distinctWords) + 1
            if row[1] not in classes:
                classes[row[1]] = len(classes) + 1
                inverseClasses[len(classes)] = row[1]
    return (distinctWords, classes, inverseClasses)

def preprocess_model2(allRows):
    distinctWords = dict()
    classes, inverseClasses = dict(), dict()
    for row in allRows:
        if len(row) != 0:
            if row[0] not in distinctWords:
                distinctWords[row[0]] = len(distinctWords) + 1
            if row[1] not in classes:
                classes[row[1]] = len(classes) + 1
                inverseClasses[len(classes)] = row[1]
    distinctWords['BOS'] = len(distinctWords) + 1
    distinctWords['EOS'] = len(distinctWords) + 1
    return (distinctWords, classes, inverseClasses)

# construct input data feature matrix
# number of input data is n
# each data -> column vector m x 1
# each data stacks -> featureMatrix n x m
def getFeatureMatrix_model1(allRows, distinctWords):
    featureMatrix = list()
    for row in allRows:
        if len(row) != 0:
            featureVector = [0] * len(distinctWords)
            featureVector[distinctWords[row[0]] - 1] = 1
            featureMatrix.append(featureVector)
    return np.array(featureMatrix)
# for model2
def getFeatureMatrix_model2(allRows, distinctWords):
    featureMatrix = list()
    prev = 'BOS'
    for i in range(len(allRows)):
        if len(allRows[i]) != 0:
            prefeatureVector, currfeatureVector, aftfeatureVector =\
            [0] * len(distinctWords), [0] * len(distinctWords), [0] * len(distinctWords)
            if i > 0 and len(allRows[i - 1]) != 0: 
                prev = allRows[i - 1][0]
            if i + 1 == len(allRows) or len(allRows[i + 1]) == 0:
                aft = 'EOS'
            else: aft = allRows[i + 1][0]
            curr = allRows[i][0]

            prefeatureVector[distinctWords[prev] - 1] = 1
            currfeatureVector[distinctWords[curr] - 1] = 1
            aftfeatureVector[distinctWords[aft] - 1] = 1
            prev = 'BOS'
            featureVector = prefeatureVector + currfeatureVector + aftfeatureVector
            featureMatrix.append(featureVector)
    return np.array(featureMatrix)

# construc input data label matrix
# number of input data is n
# number of classes is k
# each label -> column vector k x 1
# each label stacks -> labelMatrix n x k
def getTruelabelMatrix(allRows, classes):
    truelabelMatrix = list()
    for row in allRows:
        if len(row) != 0:
            truelabelVector = [0] * len(classes)
            truelabelVector[classes[row[1]] - 1] = 1
            truelabelMatrix.append(truelabelVector)
    return np.array(truelabelMatrix)

def initializeParams(weightMtrixShape, biasVectorShape):
    weightMatrix = np.zeros(weightMtrixShape)
    biasVector = np.zeros(biasVectorShape)
    return weightMatrix, biasVector

# compute input matrix W.T * X + b
def getInputMatrix(featureMatrix, weightMatrix, biasVector):
    return featureMatrix.dot(weightMatrix) + biasVector

# compute the softmax output
# return matrix -> n x k
# n -> number of data
# k -> nunmber of classes
def softmax(inputMatrix):
    return (np.exp(inputMatrix.T) / np.sum(np.exp(inputMatrix), axis = 1)).T

# CITE CODE from https://stackoverflow.com/questions/46510929/
# mle-log-likelihood-for-logistic-regression-gives-divide-by-zero-error
# compute conditional log likelihood
def computeLikelihood(activations, truelabels):
    crossEntropy = - np.sum(np.log(activations) * truelabels, axis = 1)
    likelihood = np.mean(crossEntropy)
    return likelihood

# convert softmax output into predicted labels of classes
# n number of data
# return label prediction vector -> n x 1
def getPredictedlabel(activations, classes):
    predictionsHotCode = activations.argmax(axis = 1)
    print(predictionsHotCode)
    key = predictionsHotCode[0] + 1
    predictedlabel = classes[key]
    return predictedlabel

def predict(data, distinctWords, inverseClasses, trainedWeights, trainedBias, model = 1):
    testAllRows = data
    if model == 1:
        featureMatrix = getFeatureMatrix_model1(testAllRows, distinctWords)
    elif model == 2:
        featureMatrix = getFeatureMatrix_model2(testAllRows, distinctWords)

    numData, numFeatures = featureMatrix.shape
    predictions = list()
    for i in range(numData):
        linInput = getInputMatrix(featureMatrix[i], trainedWeights, trainedBias)
        activations = softmax(linInput)

        predictedlabel = getPredictedlabel(activations, inverseClasses)
        predictions.append(predictedlabel)
    return predictions

# data -> train/test/tune raw data
# extract labels from data
# return truelabels -> list containes labels for data
# messages seperated by ''
def extractTruelabels(data):
    truelabels = list()
    for row in data:
        if len(row) != 0: truelabels.append(row[1])
        elif len(row) == 0: truelabels.append("")
    predicdlabels = 42
    return truelabels

# compare true labels and predicted labels
# compute error rate
# from predicted labels(contains '') and true labels(no '') to
# construct output preictied lables -> messages seperated by ''
def getError(trainedWeights, trainedBias, data, distinctWords, inverseClasses, model):
    truelabels = extractTruelabels(data)
    predictedlabels = predict(data, distinctWords, inverseClasses, trainedWeights, trainedBias, model)
    outputPredictedlabels = copy.deepcopy(predictedlabels)
    errNum = 0
    totalNum = len(predictedlabels)
    for i in range(len(truelabels)):
        if truelabels[i] == '': outputPredictedlabels.insert(i, '')
    for i in range(len(truelabels)):
        if truelabels[i] != outputPredictedlabels[i]: errNum += 1

    errorRate = float(errNum) / float(totalNum)
    return errorRate, outputPredictedlabels

# to apply multinomial regression
# adapt SGD to learn the model
# allRows -> train data
# tuneAllRows -> validation data
def multiNomialRegression(numEpochs, allRows, tuneAllRows, distinctWords, classes, inverseClasses, metricOut, model = 1, learningRate = 0.5):
    # featureMatrix as input data -> n x m
    # m -> number of features; n -> number of data
    if model == 1:
        trainFeatureMatrix = getFeatureMatrix_model1(allRows, distinctWords)
        tuneFeatureMatrix = getFeatureMatrix_model1(tuneAllRows, distinctWords)
    elif model == 2:
        trainFeatureMatrix = getFeatureMatrix_model2(allRows, distinctWords)
        tuneFeatureMatrix = getFeatureMatrix_model2(tuneAllRows, distinctWords)

    # truelabelMatrix for train data as true labels -> n x k
    # k -> number of classes
    # n -> number of data
    truelabelMatrix = getTruelabelMatrix(allRows, classes)
    print(truelabelMatrix)
    # tuneTruelabel for tune data as true labels -> n x k
    tuneTruelabel = getTruelabelMatrix(tuneAllRows, classes)
    # numData and numFeatures derived from train data
    # numClasses derived from train data true labels
    numData, numFeatures = trainFeatureMatrix.shape
    numClasses = truelabelMatrix.shape[1]

    # initialize weights and bias terms all to zeros
    # weightMatrix -> m x k
    # biasVector -> 1 x k 
    weightMatrixShape = (numFeatures, numClasses)
    biasVectorShape = (1, numClasses)
    weightMatrix, biasVector = initializeParams(weightMatrixShape, biasVectorShape)

    # list to store train likelihood and validations likelihood
    trainLikelihoods, tuneLikelihoods = list(), list()
    # preset number of epochs to iterate the data
    for i in range(numEpochs):
        # SGD to update params
        for j in range(numData):
            linInput = getInputMatrix(trainFeatureMatrix[j], weightMatrix, biasVector)
            activations = softmax(linInput)
            residuals = activations - truelabelMatrix[j]
            # compute gadient
            grad = np.dot(np.reshape(trainFeatureMatrix[j], (numFeatures, 1)), residuals)
            # update weight Matrix and bias terms
            weightMatrix -= learningRate * grad
            biasVector -= learningRate * np.sum(residuals, axis = 0)

        #compute train set likelihood
        trainLinInput = getInputMatrix(trainFeatureMatrix, weightMatrix, biasVector)
        trainActivations = softmax(trainLinInput)
        trainLikelihood = computeLikelihood(trainActivations, truelabelMatrix)
        trainLikelihoods.append(trainLikelihood)
        writeMetricFile(trainLikelihood, i + 1, metricOut, "train")
        #compute tune set likelihood
        tuneLinInput = getInputMatrix(tuneFeatureMatrix, weightMatrix, biasVector)
        tuneActivations = softmax(tuneLinInput)
        tuneLikelihood = computeLikelihood(tuneActivations, tuneTruelabel)
        tuneLikelihoods.append(tuneLikelihood)
        writeMetricFile(tuneLikelihood, i + 1, metricOut, "validation")
    return (weightMatrix, biasVector, trainLikelihoods, tuneLikelihoods)

################################################################################
# model1 and model2 program
def model1(trainAllRows, tuneAllRows, testAllRows, metricOut, numEpochs):
    distinctWords, classes, inverseClasses = preprocess_model1(trainAllRows)
    trainedWeights, trainedBias, trainLikelihoods, tuneLikelihoods = multiNomialRegression(numEpochs, trainAllRows, tuneAllRows, 
                                                        distinctWords, classes, inverseClasses, metricOut)

    trainError, trainPredictions = getError(trainedWeights, trainedBias, trainAllRows, distinctWords, inverseClasses, 1)
    testError, testPredictions = getError(trainedWeights, trainedBias, testAllRows, distinctWords, inverseClasses, 1)
    return (trainPredictions, trainLikelihoods, trainError, 
            testPredictions, tuneLikelihoods, testError)

def model2(trainAllRows, tuneAllRows, testAllRows, metricOut, numEpochs):
    model = 2
    distinctWords, classes, inverseClasses = preprocess_model2(trainAllRows)
    trainedWeights, trainedBias, trainLikelihoods, tuneLikelihoods = multiNomialRegression(numEpochs, trainAllRows, tuneAllRows, 
                                                        distinctWords, classes, inverseClasses, metricOut, model)

    trainError, trainPredictions = getError(trainedWeights, trainedBias, trainAllRows, distinctWords, inverseClasses, 2)
    testError, testPredictions = getError(trainedWeights, trainedBias, testAllRows, distinctWords, inverseClasses, 2)
    return (trainPredictions, trainLikelihoods, trainError, 
            testPredictions, tuneLikelihoods, testError)

################################################################################
# write output functions
# write output labels in out.label
def writeOutLabelsFile(contents, filePath):
    contents = "\n".join(contents)
    with open(filePath, 'wt') as file:
        file.write(contents)

# write metrics data in metricsOut.txt
def writeErrorFile(trainError, testError, metricsOut):
    contents = list()
    trainError = "error(train): %0.6f"%trainError
    testError = "error(test): %0.6f"%testError
    contents.append(trainError)
    contents.append(testError)
    contents = "\n".join(contents)
    with open(metricsOut, 'a') as file:
        file.write(contents)

def writeMetricFile(likelihood, epochNum, metricsOut, type):
    if type == "train":
        content = "epoch=%d likelihood(train): %0.6f\n"%(epochNum, likelihood)
        with open(metricsOut, 'a') as file:
            file.write(content)
    elif type == "validation":
        content = "epoch=%d likelihood(validation): %0.6f\n"%(epochNum, likelihood)
        with open(metricsOut, 'a') as file:
            file.write(content)

################################################################################
# miniSiri program
def miniSiri(trainData, tuneData, testData, trainOut, testOut, metricsOut, numEpochs, model):
    trainAllRows = loadData(trainData)
    tuneAllRows = loadData(tuneData)
    testAllRows = loadData(testData)
    if model == 1:
        (trainPredictions, trainLikelihoods, trainError,
         testPredictions, tuneLikelihoods, testError) = model1(trainAllRows, tuneAllRows, testAllRows, metricsOut, numEpochs)
    elif model == 2:
        (trainPredictions, trainLikelihoods, trainError,
         testPredictions, tuneLikelihoods, testError) = model2(trainAllRows, tuneAllRows, testAllRows, metricsOut, numEpochs)
    writeOutLabelsFile(trainPredictions, trainOut)
    writeOutLabelsFile(testPredictions, testOut)
    writeErrorFile(trainError, testError, metricsOut)

def main():
    miniSiri(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], 
             sys.argv[5], sys.argv[6], int(sys.argv[7]), int(sys.argv[8]))

if __name__ == '__main__':
    main()