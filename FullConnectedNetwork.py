#!/usr/bin/python
# -*- coding: UTF-8 -*-
import numpy as np
from NeuralNode import NeuralNode

#全连接前向神经网络
class FullConnectedNetwork:
  def __init__(self, iStructure):
    #网络结构，以list表达，例如[N0, N1, N2, N3]表示4层网络，
    #第一层为输入层，包含N0个节点，最后一层为输出层包含N3个节点，网络包含2个隐层，分别有N1和N2个节点
    self.networkStructure = iStructure
    # NodeList中保存每一层的神经元，每一层为一个单独的List，对于输入层，保存“”.
    # 每一层对应的List（不含输入层）保存神经元对象
    self.nodeList = genNodeList(self.networkStructure)

  #根据一条输入数据预测结果
  #输入数据向量形状为1*numberOfInputNodes)
  #返回值形状为为1*numberOfOutputNodes)
  def predict(self, inputData):
    if inputData.shape[0] != self.networkStructure[0]:
      print "Invalid input data shape: ", inputData.shape
      return
    # 为了避免重复计算，用List保存每个节点的前向计算值，
    # 这个List包含N_Layer个ndarray，分别代表每一层的计算结果
    forwardValueList = [inputData]
    for layerCount in range(1, len(self.networkStructure)):
      forwardValueVec = np.zeros(self.networkStructure[layerCount])
      for indexInCurLayer in range(0, forwardValueVec.shape[0]):
        forwardValueVec[indexInCurLayer] = self.nodeList[layerCount][indexInCurLayer].forward(forwardValueList[layerCount - 1])
      forwardValueList.append(forwardValueVec)
    result = forwardValueList[-1]
    return result

  #根据批量输入数据预测结果
  #输入数据向量形状为batchSize*numberOfInputNodes)
  #返回值形状为为batchSize*numberOfOutputNodes)
  def predict_batch(self,iBatchData):
    batchSize = iBatchData.shape[0]
    result = np.zeros([batchSize, self.networkStructure[-1]])
    for batchIdx in range(0, batchSize):
      result[batchIdx] = self.predict(iBatchData[batchIdx])
    return result

  #根据训练数据和标签训练模型
  #输入数据data向量形状为(batch_size, numberOfInputNodes)
  #标签数据label向量形状为为(batch_size, numberOfOutputNodes), 这里只考虑分类问题，label采用one-hot vector的形式
  def train(self, data, label, learningRate):
    #使用梯度下降法计算
    totalLoss = 0
    for batchIdx in range(0, data.shape[0]):
      # 得到前向预测结果向量
      predict_result = self.predict(data[batchIdx])

      # 使用交叉熵计算Loss
      loss = cross_entropy(softmax(predict_result), label[batchIdx])
      totalLoss += loss;
      # dpredict = dLoss /dpredict
      dpredict = (label[batchIdx] - predict_result)
      #用于每一层训练的梯度向量，其形状为1*numOfNodeInThisLayer，处理每一层时形状在变化
      gradient = dpredict
      #DebugOutput
      #print 'batchIdx', batchIdx, 'data = ', data[batchIdx], 'predictedResult=',predict_result,  'Softmax(predictedResult)=', softmax(predict_result), ' label = ', label[batchIdx], 'loss = ', loss, ' gradient = ', gradient
      #从输出层向前逐层做训练, 第0层不用训练
      for layerIdx in range(len(self.networkStructure) -1, 0, -1):
        for nodeIdxInOneLayer in range(0, self.networkStructure[layerIdx]):
          #对每个节点计算梯度
          grad = self.nodeList[layerIdx][nodeIdxInOneLayer].backward(gradient[nodeIdxInOneLayer])
          #训练每个节点
          self.nodeList[layerIdx][nodeIdxInOneLayer].adjustWeightAndBias(learningRate, grad[0], grad[1])
          #print 'layer',layerIdx, 'node', nodeIdxInOneLayer, 'Weight', self.nodeList[layerIdx][nodeIdxInOneLayer].weight
        #计算gradient，供前一层训练使用,当layerIndex=1时，前一层不用训练
        if (layerIdx > 1):
          #梯度向量形状为1*(layerIdx-1)层节点数目s
          weightMatrix = np.zeros([self.networkStructure[layerIdx-1], self.networkStructure[layerIdx]])
          deltaVec = np.zeros(self.networkStructure[layerIdx])
          for nodeIdx in range(0, self.networkStructure[layerIdx]):
            weightMatrix[:, nodeIdx] = self.nodeList[layerIdx][nodeIdx].weight
            deltaVec[nodeIdx] = self.nodeList[layerIdx][nodeIdx].delta
          gradient = np.matmul(weightMatrix, deltaVec)
          #print 'Gradient for layer ', layerIdx-1, ' =',gradient

      #Debug: 训练完一条数据之后再算一遍，看Loss是否有降低
      predict_result = self.predict(data[batchIdx])
      loss = cross_entropy(softmax(predict_result), label[batchIdx])
      #print 'batchIdx', batchIdx, 'After Train Loss = ', loss
      #fcNetwork.debug_print()
    print "Total Loss=", totalLoss

  def debug_print(self):
    for layerIdx in range(1, len(self.networkStructure)):
      for nodeIdx in range(0, self.networkStructure[layerIdx]):
        print 'LayerIdx = ', layerIdx,' NodeIdx = ', nodeIdx ," ",  self.nodeList[layerIdx][nodeIdx].getParam()

#根据网络结构List生成神经元List
def genNodeList(networkStructure):
  nodeList = []
  for layerIdx in range(0, len(networkStructure)):
    if layerIdx ==  0:
      nodeList.append("")
    else:
      nodesInThisLayer =  []
      for nodeIdx in range(0, networkStructure[layerIdx]):
        nodesInThisLayer.append(NeuralNode(networkStructure[layerIdx-1]))
      nodeList.append(nodesInThisLayer)

  return nodeList

#计算两个向量的交叉熵
#两个输入向量具有同样的形状，都是1*N的向量
def cross_entropy(data,label):
  return -1*np.sum(np.nan_to_num(np.dot(label, np.log(data))))

#计算向量的softmax值
def softmax(data):
  s = np.sum(np.exp(data))
  return np.exp(data) / s

#处理经过softmat得到的向量，将最大值的维度设置为1，其余维度设置为0
def argmax(data):
  maxIdx = np.argmax(data)
  result = np.zeros(data.shape)
  result[maxIdx] = 1
  return result

def generateTestDataAndLabel(batchSize):
  testData = np.random.randn(batchSize, 2)
  label = np.zeros([batchSize, 2])
  for batchIdx in range(0, batchSize):
    if testData[batchIdx][0] >= testData[batchIdx][1]:
      label[batchIdx][0] = 1
    else:
      label[batchIdx][1] = 1
  return [testData, label]

def evaluate(fullConnectedNetwork, testData, label):
  if testData.shape[0] != label.shape[0]:
    print '数据和标签的Batch数目不匹配，无法评估'
    return
  tp = 0
  tn = 0
  fp = 0
  fn = 0
  predictedValues = fullConnectedNetwork.predict_batch(testData);
  for batchIdx in range(0, testData.shape[0]):
    if (argmax(softmax(predictedValues[batchIdx])) == label[batchIdx]).all():
      if label[batchIdx][0] == 1:
        tp = tp + 1
      else:
        tn = tn + 1
    else:
      if label[batchIdx][0] == 1:
        fn = fn + 1
      else:
        fp = fp + 1
  if (tp+fp == 0):
    precision = 0
  else:
    precision = tp * 1.0 / (tp + fp)
  if (tp+fn == 0):
    recall = 0
  else:
    recall = tp*1.0 / (tp+fn)
  accuracy =  (tp+tn)*1.0 / (tp+tn+fp+fn)
  print 'tp = ', tp, ", tn =",tn, " fp = ", fp,' fn=', fn
  print "precision = ", precision, " recall = ", recall, " acc =  ", accuracy
  return [precision, recall, accuracy]

if __name__ == '__main__':
  fcNetwork = FullConnectedNetwork([2, 2, 2])
  fcNetwork.debug_print()

  batchSize = 10
  learningRate = 0.03

  [testData, label] = generateTestDataAndLabel(batchSize)
  for index in range(0, label.shape[0]):
    print "testData[",index, ']=', testData[index], "  label[",index, ']=',label[index]

  print "before train: "
  evaluate(fcNetwork,testData,label)

  for trainIndex in range(0, 10):
    fcNetwork.train(testData, label, learningRate)
    print "result after train loop ", trainIndex
    evaluate(fcNetwork, testData, label)
    fcNetwork.debug_print()