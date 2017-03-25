#!/usr/bin/python
# -*- coding: UTF-8 -*-

# author: wangyao_bupt@hotmail.com
import numpy as np

# FullConnectedNetwork2 is a fully connected neural network, using sigmoid as activation function
# The difference between FullConnectedNetwork2 and FullConnectedNetwork is:
# In FullConnectedNetwork2, the weights and bias is stored in matrix format.The NeuralNode class is NOT used in FullConnectedNetwork2
class FullConnectedNetwork2:
  __debugMode__ = True

  # @param iStructure is a list containing L enlements, representing L layers of a fully connected netowrk
  # The 1st element (aka iStructure[0]) define number of input nodes
  # the i-th element (aka iStructure[i-1]) define number of neural nodes in i-th layer
  # @param batchSize define the number of data whthin a batch
  def __init__(self, iStructure, ibatchSize):
    self.networkStructure = iStructure
    self.numberOfLayers = len(self.networkStructure)
    self.batchSize = ibatchSize
    #weightMatrixList is a container of Weight Matrixes of each layer
    self.weightMatrixList = np.ndarray((self.numberOfLayers - 1),np.object)
    for layerIdx in range(0, self.numberOfLayers - 1):
      ## each element in weightMatrixList is a matrix, the content is as below
      # ROW_0 = [weight(^l)(_00), weight(^l)(_01),weight(^l)(_02),...weight(^l)(_0S_l), b(^l)(_1)]
      # .....
      # ROW_(s_(l+1) - 1) = [weight(^l)(_(s_(l+1) - 1)0), weight(^l)(_(s_(l+1) - 1)1),weight(^l)(_(s_(l+1) - 1)2),...weight(^l)(_(s_(l+1) - 1)(S_l)), b(^l)(_(s_(l+1) - 1))]
      # l: layer index
      # weight(^l)(_ji): the weight from i-th node in (l) layer to j-th node in (l+1) layer
      # b(^l)(_j): the bias of j-th node in (l+1) layer
      # s_l: number of nodes in (l) layer
      if (not FullConnectedNetwork2.__debugMode__):
        weightMatrixBetweenCurAndNextLayer = np.random.randn(
          iStructure[layerIdx+1],iStructure[layerIdx]+1)
      else:
        weightMatrixBetweenCurAndNextLayer = np.zeros((iStructure[layerIdx+1],iStructure[layerIdx]+1))
        weightMatrixBetweenCurAndNextLayer[0] = [1,-1,1]
        weightMatrixBetweenCurAndNextLayer[1] = [2, -2, 1]
      self.weightMatrixList[layerIdx] = weightMatrixBetweenCurAndNextLayer
    # activationValueMatrix is activation values of each layer in each sample in a batch,
    # activationValueMatrix's shape is (batchSize, numberOfLayers), each element is an 1-D vector,
    # For example, given batchIdx = b, LayerIndex = l, the activationValueMatrix[b][l] represents
    #  the value (1-D vector with self.networkStructure[l] dims) calculated from l-th layer
    # For the input layer, activationValueMatrix[b][0] = inputVector
    self.activationValueMatrix = np.ndarray((self.batchSize, len(self.networkStructure)), np.object)
    # zMatrix is the value before activation of each layer in each sample in a batch
    # the shape of zMatrix is similiar to activationValueMatrix
    # since input layer does not contain any weight, self.zMatrix[b][0] is meaningless
    self.zMatrix = np.ndarray((self.batchSize, len(self.networkStructure)), np.object)

  # Calculate forward value of a batch of input data
  # the input data shape should be (batchSize, numberOfInputNode)
  # batchSize == self.batchSize
  # numberOfInputNode == self.networkStructure[0]
  # return value is a matrix in shape (batchSize, numberOfOutputNode)
  def forward(self, inputData):
    batchSize = inputData.shape[0]
    if batchSize != self.batchSize:
      print "Invalid Batch Size:",batchSize
    result =np.ndarray((batchSize, self.networkStructure[-1]))
    for sampleIdx in range(0, batchSize):
      #the a[sampleIdx][0] always equal to inputVector
      self.activationValueMatrix[sampleIdx][0] = inputData[sampleIdx]
      for layerIdx in range(1, self.numberOfLayers):
        activation_and_one = np.ones(self.activationValueMatrix[sampleIdx][layerIdx - 1].shape[0]+1)
        activation_and_one[:-1] = self.activationValueMatrix[sampleIdx][layerIdx - 1]
        self.zMatrix[sampleIdx][layerIdx] \
          = np.matmul(self.weightMatrixList[layerIdx - 1], activation_and_one)
        self.activationValueMatrix[sampleIdx][layerIdx] = self.sigmoid(self.zMatrix[sampleIdx][layerIdx])
      result[sampleIdx] = self.activationValueMatrix[sampleIdx][self.numberOfLayers-1]
    return result

  # Evaluate error between predicted result and label
  # both predictedResult and label 's shapes are (batchSize, numberOfOutputNode)
  # the loss function is defined as http://ufldl.stanford.edu/wiki/index.php/Backpropagation_Algorithm
  # ignoring regularization term
  def lossEvaluation(self, predictedResult, label):
    loss = 0.0;
    for sampleIdx in range(0, self.batchSize):
      loss += 0.5* np.sum((predictedResult[sampleIdx] - label[sampleIdx])**2)
    return loss / self.batchSize

  # Train the network using label
  # the label should be in the same batch size as data, i.e. in shape of (batchSize, numberOfOutputNode
  def train(self, data, label, learningRate):
    predictedResult = self.forward(data)
    if FullConnectedNetwork2.__debugMode__:
      print 'loss before training: ', self.lossEvaluation(predictedResult, label)
    # for each node i in layer l, we would like to compute an "error term"  that measures
    # how much that node was "responsible" for any errors in our output
    # delta represent such error
    delta = np.ndarray((self.batchSize, self.numberOfLayers-1), np.object)

    ## initialize delta_weightAndBias to all zeros
    delta_weightAndBias = np.ndarray((self.numberOfLayers-1), np.object)
    for layerIdx in range(0, self.numberOfLayers-1):
      delta_weightAndBias[layerIdx] = np.zeros(self.weightMatrixList[layerIdx].shape)

    ## calculate deritives of each weight/bias at each node in eachlayer in each sample
    for sampleIdx in range(0, self.batchSize):
      delta[sampleIdx][-1] = \
        -(label[sampleIdx] - self.activationValueMatrix[sampleIdx][self.numberOfLayers-1])\
        *self.dsigmoiddx_usingActivationValue(self.activationValueMatrix[sampleIdx][self.numberOfLayers-1])
      for layerIdx in range(self.numberOfLayers-3, -1, -1):
        #build vector for f'(z_i)
        dsigmoidVector = np.zeros(self.networkStructure[layerIdx+1])
        for nodeIdx in range(0, self.networkStructure[layerIdx+1]):
          dsigmoidVector[nodeIdx] = self.dsigmoiddx_usingActivationValue(self.activationValueMatrix[sampleIdx][layerIdx+1][nodeIdx])
        # build weight matrix, remove bias column
        weightMatrix = self.weightMatrixList[layerIdx];
        weightMatrix = np.zeros((self.weightMatrixList[layerIdx].shape[0],self.weightMatrixList[layerIdx].shape[1]-1) )
        for rowIdx in range(0, weightMatrix.shape[0]):
          for colIdx in range(0, weightMatrix.shape[1]):
            weightMatrix[rowIdx][colIdx] = self.weightMatrixList[layerIdx][rowIdx][colIdx]
        weightMatrix = np.transpose(weightMatrix)
        #calculate delta
        delta[sampleIdx][layerIdx] = np.matmul(weightMatrix, delta[sampleIdx][layerIdx+1]) * dsigmoidVector

      for layerIdx in range(0, self.numberOfLayers-1):
        ## dWeightAndBias is value of desired partial derivatives
        dWeightAndBias = np.zeros(self.weightMatrixList[layerIdx].shape)
        for i in range(0, self.weightMatrixList[layerIdx].shape[0]):
          for j in range(0, self.weightMatrixList[layerIdx].shape[1]):
            if j >= self.networkStructure[layerIdx]:
              #dbias
              dWeightAndBias[i][j] = delta[sampleIdx][layerIdx][i]
            else:
              dWeightAndBias[i][j]  = delta[sampleIdx][layerIdx][i] \
                                             * self.activationValueMatrix[sampleIdx][layerIdx][j];
        if self.__debugMode__:
          print 'Sample Idex = ', sampleIdx, " LayerIdx = ", layerIdx, " dWeightAndBias = ", dWeightAndBias
        ##For each layer in each sample, update  delta_weightAndBias
        delta_weightAndBias[layerIdx] += dWeightAndBias

    if self.__debugMode__:
      for sampleIdx in range(0,self.batchSize):
        for layerIdx in range(0, self.numberOfLayers-1):
          print 'Sample Idex = ', sampleIdx, " LayerIdx = ", layerIdx, " Delta = ",delta[sampleIdx][layerIdx]

    ##Update weights and bias
    for layerIdx in range(0, self.numberOfLayers-1):
      self.weightMatrixList[layerIdx] = self.weightMatrixList[layerIdx] - \
                                        learningRate*delta_weightAndBias[layerIdx]/self.batchSize

  def sigmoid(self, x):
    result = 1 / (1 + np.exp(-1*x))
    return result

  def dsigmoiddx(self, x):
    return (1-self.sigmoid(x))*self.sigmoid(x)

  def dsigmoiddx_usingActivationValue(self, activationValue):
    return (1-activationValue)*activationValue

  def debugPrintWeights(self):
    for layerIdx in range(0, self.numberOfLayers-1):
      for nodeIdx in range(0, self.networkStructure[layerIdx+1]):
        print 'Layer ', (layerIdx+1), ' Node ', nodeIdx, " Weight = ", self.weightMatrixList[layerIdx][nodeIdx][:-1], " bias = ", self.weightMatrixList[layerIdx][nodeIdx][-1]

def generateTestDataAndLabel(batchSize):
  testData = np.random.randn(batchSize, 2)
  label = np.zeros([batchSize, 2])
  for batchIdx in range(0, batchSize):
    if testData[batchIdx][0] >= testData[batchIdx][1]:
      label[batchIdx][0] = 1
    else:
      label[batchIdx][1] = 1
  return [testData, label]

if __name__ == "__main__":
  batchSize = 1
  learningRate = 0.03

  FullConnectedNetwork2.__debugMode__ = True
  network = FullConnectedNetwork2([2,2,2],batchSize)
  network.debugPrintWeights()

  testData = np.ones((1, 2))
  testData[0][1] = 2
  label = np.zeros((1, 2))
  label[0][1] = 1
  #[testData, label] = generateTestDataAndLabel(batchSize)

  predictedResult = network.forward(testData)

  for dataIdx in range(0, batchSize):
    print 'inputData = ', testData[dataIdx], "Layer 1 actValue=", network.activationValueMatrix[dataIdx][1], " PredictedResult = ", network.activationValueMatrix[dataIdx][2], " Label = ", label[dataIdx]

  for dataBatchIdx in range(0, 1):
    #[testData, label] = generateTestDataAndLabel(batchSize)
    for trainLoopIdx in range(0, 1):
      network.train(testData, label, learningRate)

  network.debugPrintWeights()
  network.forward(testData)
  for dataIdx in range(0, batchSize):
    print 'inputData = ', testData[dataIdx], "Layer 1 actValue=", network.activationValueMatrix[dataIdx][1], " PredictedResult = ", network.activationValueMatrix[dataIdx][2], " Label = ", label[dataIdx]