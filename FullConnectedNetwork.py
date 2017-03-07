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

  #根据输入数据预测结果
  #输入数据向量形状为(batch_size, numberOfInputNodes)
  #返回值形状为为(batch_size, numberOfOutputNodes)
  def predict(self, inputData):
    if inputData.shape[1] != self.networkStructure[0]:
      print "Invalid input data shape: ", inputData.shape
      return
    result = np.zeros((inputData.shape[0], self.networkStructure[-1]))
    for batchIdx in range(0, inputData.shape[0]):
      inputItem = inputData[batchIdx]
      #为了避免重复计算，用List保存每个节点的前向计算值，
      #这个List包含N_Layer个ndarray，分别代表每一层的计算结果
      forwardValueList = [inputItem]
      for layerCount in range(1, len(self.networkStructure)):
        forwardValueVec = np.zeros(self.networkStructure[layerCount])
        for indexInCurLayer in range(0, forwardValueVec.shape[0]):
          forwardValueVec[indexInCurLayer] = self.nodeList[layerCount][indexInCurLayer].forward(forwardValueList[layerCount-1])
        forwardValueList.append(forwardValueVec)
      result[batchIdx] =  forwardValueList[-1]
    return result

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


if __name__ == '__main__':
  fcn = FullConnectedNetwork([2,2,2])
  batchSize = 10
  inputData = np.random.randn(batchSize, 2)
  print "input:",inputData
  result =  fcn.predict(inputData)
  print "output:",result
