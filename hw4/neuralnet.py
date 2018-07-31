# neuralnet program for handwritten letters classifier
# @author Xuliang Sun
################################################################################
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import csv
import sys
################################################################################
# labels -> true labels for input 
# Y -> one hot encoding for labels n x k (n number, k classes)
def getLabelMatrix(labels):
    classes = set(labels)
    Y = list()
    for num in labels:
        num = int(num)
        label = [0] * len(classes)
        label[num] = 1
        Y.append(label)
    return Y

# load data and store allRows and labels
# allRows -> input data n x m (n number, m features)
# labels -> true labels
def loadData(filePath):
    file = open(filePath)
    data = csv.reader(file)
    X, labels = list(), list()
    for row in data:
        X.append(row[1:])
        labels.append(row[0])
    Y = getLabelMatrix(labels)
    return np.array(X, dtype = "float64"), np.array(Y, dtype = "float64")
################################################################################
# get layer size from X, Y and hidden_units
# input data X -> (n x m), true labels coding matrix Y -> (n x k)
# inputLayer -> number of features -> m
# hidderLayer -> predifined hidden_units
# outputLayer -> number of classes -> k
def getLayerSize(X, Y, hidden_units):
    inputLayer = X.shape[1]
    hiddenLayer = hidden_units
    outputLayer = Y.shape[1]
    return (inputLayer, hiddenLayer, outputLayer)

# initialize weights randomly from a uniform distribution over [-0.1,0.1]
def randomInit(inputLayer, hiddenLayer, outputLayer):
    alpha = np.ones((inputLayer, hiddenLayer)) * 0.1
    b1 = np.ones((1, hiddenLayer)) * 0.1
    beta = np.ones((hiddenLayer, outputLayer)) * 0.1
    b2 = np.ones((1, outputLayer)) * 0.1
    return (alpha, b1, beta, b2)

# initialize weights to zeros
def zerosInit(inputLayer, hiddenLayer, outputLayer):
    alpha = np.zeros((inputLayer, hiddenLayer))
    b1 = np.zeros((1, hiddenLayer))
    beta = np.zeros((hiddenLayer, outputLayer))
    b2 = np.zeros((1, outputLayer))
    return (alpha, b1, beta, b2)

# initialize each layer parameters based on init_flag
# flag == 1 -> random initialization from uniform distribution [-0.1, 0.1]
# flag == 2 -> zeros initializarion to all zeros
def initParams(inputLayer, hiddenLayer, outputLayer, init_flag):
    if (init_flag == 1):
        parameters = randomInit(inputLayer, hiddenLayer, outputLayer)
    elif (init_flag == 2):
        parameters = zerosInit(inputLayer, hiddenLayer, outputLayer)
    return parameters

# sigmoid function for input layer
def sigmoid(z):
    return float(1) / (1 + np.exp(-z))

# softmax funtion for output layer
def softmax(z):
    return np.exp(z) / np.sum(np.exp(z), axis = 1, keepdims = True)

# zi -> pre-activations; ai -> post-activations(activation)
# forward propagation to calculate zi, ai at each layer
# cache -> store zi, ai to back-propagation
def forwardProp(X, parameters):
    alpha, b1, beta, b2 = parameters
    z1 = np.dot(X, alpha) + b1
    #print("z1: ", z1)
    a1 = sigmoid(z1)
    #print("a1: ",a1)
    z2 = np.dot(a1, beta) + b2
    #print("z2: ", z2)
    a2 = softmax(z2)
    #print("a2: ", a2)

    cache = (z1, a1, z2, a2)
    output = a2
    return output, cache

# back propagation to calulate gradients at each layer
# to update parameters
def backProp(X, Y, cache, parameters):
    alpha, b1, beta, b2 = parameters
    z1, a1, z2, a2 = cache

    dz2 = a2 - Y
    dBeta = np.dot(a1.T, dz2)
    db2 = np.sum(dz2, axis = 0, keepdims = True)

    dz1 = np.dot(dz2, beta.T) * a1 * (1 - a1)
    dAlpha = np.dot(X.T, dz1)
    db1 = np.sum(dz1, axis = 0, keepdims = True)

    gradients = (dAlpha, db1, dBeta, db2)
    return gradients

################################################################################
X = np.array([1, 1, 0, 0, 1, 1]).reshape(1, 6)
Y = np.array([0, 1, 0]).reshape(1, 3)
alpha = np.array([[ 1, 3, 2, 1],
                  [ 2, 1, 2, 0],
                  [-3, 2, 2, 2],
                  [ 0, 1, 2, 1],
                  [ 1, 0, 2,-2],
                  [-3, 2, 1, 2]])
beta = np.array([[ 1, 1, 3],
                 [ 2,-1, 1],
                 [-2, 1,-1],
                 [ 1, 2, 1]])
b1 = np.array([[1, 1, 1, 1]])
b2 = np.array([[1, 1, 1]])
parameters = (alpha, b1, beta, b2)
results = forwardProp(X, parameters)
gradients = backProp(X, Y, results[1], parameters)

################################################################################
# gradient descent to update parameters
def updateParams(parameters, gradients, learning_rate):
    alpha, b1, beta, b2 = parameters
    dAlpha, db1, dBeta, db2 = gradients

    alpha = alpha - dAlpha * learning_rate
    #print("update alpha: ", alpha.T)
    b1 = b1 - db1 * learning_rate
    #print("update b1: ", b1)
    beta = beta - dBeta * learning_rate
    #print("update beta: ", beta.T)
    b2 = b2 - db2 * learning_rate
    #print("update b2: ", b2)
    parameters = alpha, b1, beta, b2

    return parameters
parameters = updateParams(parameters, gradients, 1)
# predict based on nn output
# predictions -> maximum probs index list
def predict(output):
    predictions = output.argmax(axis = 1)
    return predictions

# calculate error rate based on
# predictions and true labels input
def calcError(predictions, trueLabels):
    total, error = len(predictions), 0
    for i in range(len(predictions)):
        index = predictions[i]
        if trueLabels[i][index] != 1:
            error += 1
    return float(error) / float(total)

# compute negative log likelihood
def computeCost(output, Y):
    crossEntropy = - np.sum(np.log(output) * Y, axis = 1)
    likelihood = np.mean(crossEntropy)
    return likelihood

def plotLikelihood(trainLikelihoods, tuneLikelihoods):
    epoch = [(i+1) for i in range(len(trainLikelihoods))]
    fig, ax = plt.subplots()
    ax.plot(epoch, trainLikelihoods, label = 'Train likelihood')
    ax.plot(epoch, tuneLikelihoods, label = 'Validation likelihood')
    legend = ax.legend(loc='upper center', shadow=True)
    plt.show()

def plot():
    y1 = [1.442991, 1.428561, 1.427407, 1.426400, 1.425855]
    y2 = [1.446801,  1.438041, 1.434919, 1.432039, 1.429836]
    x = [5, 20, 50, 100, 200]
    fig, ax = plt.subplots()
    ax.plot(x, y1, label = 'Train')
    ax.plot(x, y2, label = 'Validation')
    legend = ax.legend(loc='upper center', shadow=True)
    plt.show()



#print("cost", computeCost(results[0], Y))
# main loop to train 2 layer NN model
def layer2NN(X, Y, hidden_units, tuneX, tuneY, init_flag, num_epoch, learning_rate, metrics_out):
    inputLayer, hiddenLayer, outputLayer = getLayerSize(X, Y, hidden_units)
    parameters = initParams(inputLayer, hiddenLayer, outputLayer, init_flag)
    trainCrossEntropies, tuneCrossEntropies = list(), list()

    for i in range(num_epoch):
        for j in range(len(X)):
            a2, cache = forwardProp(np.reshape(X[j], (1, X.shape[1])), parameters)
            gradients = backProp(np.reshape(X[j], (1, X.shape[1])), 
                                 np.reshape(Y[j], (1, Y.shape[1])), 
                                 cache, parameters)
            parameters = updateParams(parameters, gradients, learning_rate)

        trainA2 = forwardProp(X, parameters)[0]
        trainCrossEntropy = computeCost(trainA2, Y)
        trainCrossEntropies.append(trainCrossEntropy)
        # write in cost after each epoch
        writeMetricFile(trainCrossEntropy, i + 1, metrics_out, "train")

        tuneA2 = forwardProp(tuneX, parameters)[0]
        tuneCrossEntropy = computeCost(tuneA2, tuneY)
        tuneCrossEntropies.append(tuneCrossEntropy)
        # write in cost after each epoch
        writeMetricFile(tuneCrossEntropy, i + 1, metrics_out, "validation")

    trainA2 = forwardProp(X, parameters)[0]
    trainPredictions = predict(trainA2)
    trainError = calcError(trainPredictions, Y)

    tuneA2 = forwardProp(tuneX, parameters)[0]
    tunePredictions = predict(tuneA2)
    tuneError = calcError(tunePredictions, tuneY)
    plotLikelihood(trainCrossEntropies, tuneCrossEntropies)
    return (trainPredictions, trainCrossEntropies, 
            tunePredictions, tuneCrossEntropies,
            trainError, tuneError, trainCrossEntropies, tuneCrossEntropies) 


################################################################################
# write output functions
# write output labels in out.label
def writeOutLabels(predictions, metrics_out):
    #contents = "\n".join(predictions)
    with open(metrics_out, 'a') as file:
        for label in predictions:
            file.write(str(label))
            file.write("\n")
        #file.write(contents)

# write metrics data in metrics_out
def writeError(trainError, tuneError, metrics_out):
    contents = list()
    trainError = "error(train): %0.6f"%trainError
    tuneError = "error(validation): %0.6f"%tuneError
    contents.append(trainError)
    contents.append(tuneError)
    contents = ("\n").join(contents)
    with open(metrics_out, 'a') as file:
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
# for pre set-up and handle output files
def neuralnet(train_input, validation_input, train_out, validation_out, metrics_out, 
              num_epoch, hidden_units, init_flag, learning_rate):
    trainX, trainY = loadData(train_input)
    tuneX, tuneY = loadData(validation_input)

    (trainPredictions,trainCrossEntropy, 
    tunePredictions, tuneCrossEntropy,
    trainError, tuneError, train, tune) = layer2NN(trainX, trainY, hidden_units, tuneX, tuneY, 
                                 init_flag, num_epoch, learning_rate, metrics_out)
    
    writeOutLabels(trainPredictions, train_out)
    writeOutLabels(tunePredictions, validation_out)
    writeError(trainError, tuneError, metrics_out)

    return (train, tune)

################################################################################
def main():
    neuralnet(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], 
             sys.argv[5], int(sys.argv[6]), int(sys.argv[7]), int(sys.argv[8]),
             float(sys.argv[9]))

if __name__ == '__main__':
    main()