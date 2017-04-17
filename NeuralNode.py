#!/usr/bin/python
# -*- coding: UTF-8 -*-

# author: wangyao_bupt@hotmail.com
import numpy as np

class NeuralNode:
  
  def __init__(self, inputDim):
    #当前结点与前一级的连接数目
    self.iDims = inputDim
    #权重向量，Shape = (iDims, )
    self.weight = np.random.rand(self.iDims)
    #self.weight = np.ones(self.iDims)
    #偏置 
    self.bias = np.random.rand(1)
    #激活函数的输入
    self.z = 1
    #当前层的残差
    self.delta = 1
    #前级节点的输入向量，必须与iDims匹配，Shape = (iDims, )的向量
    self.x = []
    #ada算法新增状态变量
    self.m_weight = np.zeros(self.iDims)
    self.v_weight = np.zeros(self.iDims)
    self.m_bias = 0
    self.v_bias = 0
    self.t = 0

  #forward: 输入1*iDims向量，计算前向结果
  def forward(self, ix):
    if (ix.shape <> (self.iDims,)):
      print ("Wrong input shape: x.shape = " + str(ix.shape))
      return
    self.x = ix
    self.z = np.dot(self.x, self.weight) + self.bias
    #为了避免计算溢出，对z做最大值和最小值限制
    if self.z >1000:
      self.z = 1000
    elif self.z < -1000:
      self.z = -1000
    return sigmoid(self.z)
  
  #backward: 输入前一级计算出的梯度，输出为两个数组
  #第一个数组: dx，iDims*1向量，即当前节点对于前一级每个输入的梯度
  #第二个数组：dw，iDims*1向量，当前节点对于每个权重的梯度
  #第三个数组：dbias, 1*1向量，当前节点对于偏置量的梯度
  def backward(self, gradient):
    try:
      #print 'self.z = ', self.z
      dz = (1 - self.z) * self.z #Sigmoid函数的求导
    except RuntimeWarning:
      print 'self.z = ', self.z
      if np.isnan(dz):
        dz = np.nan_to_num(dz)
      print 'dz=', dz

    self.delta = dz*gradient
    dw = self.x * self.delta  # 回传到w
    if np.isnan(dw).any():
      dz = np.nan_to_num(dw)
    dbias = self.delta  # 回传到bias
    if np.isnan(dbias).any():
      dz = np.nan_to_num(dbias)
    return [dw, dbias]
  
  #根据AdamOptimization算法调整
  def adaOptimization(self, learnRate, dw, dbias):
    #AdamOptimization算法需要的常量
    beta1=0.9
    beta2=0.999
    eps=1e-8
    self.t = self.t+1
    self.m_weight = self.m_weight*beta1 + (1-beta1)*dw
    self.v_weight = self.v_weight*beta2 + (1-beta2)*(dw*dw)
    m_w = self.m_weight / (1-beta1**self.t)
    v_w = self.v_weight / (1-beta2**self.t)
    self.weight = self.weight - learnRate*m_w/(np.sqrt(v_w)+eps)
    
    self.m_bias = self.m_bias*beta1 + (1-beta1)*dbias
    self.v_bias = self.v_bias*beta2 + (1-beta2)*(dbias*dbias)
    m_b = self.m_bias / (1-beta1**self.t)
    v_b = self.v_bias / (1-beta2**self.t)
    self.bias = self.bias - learnRate*m_b/(np.sqrt(v_b)+eps)
  
  #根据学习率和梯度调整weight和bias参数
  def adjustWeightAndBias(self, learnRate, dw, dbias):
    self.weight = self.weight - learnRate*dw
    self.bias = self.bias - learnRate*dbias
  
  #打印节点内部参数
  def printParam(self):
    print "Weight = ", self.weight , " Bias = ", self.bias

  def getParam(self):
    return [self.weight, self.bias]

def sigmoid(x):
  result = 1 / (1 + np.exp(-1*x))
  return result

#Sigmoid函数对X的导数
def dsigmoiddx(x):
  return (1-sigmoid(x))*sigmoid(x)

#测试神经元训练，使用梯度下降法训练参数
def unitTest_naiveTrain():
  print "In unitTest_naiveTrain"
  n1 = NeuralNode(2)
  n1.printParam();
  prevWeight = n1.weight
  
  x = np.ones(2)
  x[0] = 2
  x[1] = 2
  
  target = 1/(1+np.exp(1))
  counter = 0
  for i in range(1000000):
    counter=i
    print "Round",i
    fowardResult = n1.forward(x)
    #print "Forward Result:",fowardResult
    loss = (fowardResult-target)*(fowardResult-target)
    print "Loss=",loss
    dLossdvalue = 2*(target-fowardResult)
    grad = n1.backward(dLossdvalue)
    #print "grad=",grad
    n1.adjustWeightAndBias(0.001, grad[0], grad[1])
    if np.sum(np.abs(prevWeight - n1.weight)) < 1e-7:
      break
    prevWeight = n1.weight
    n1.printParam()
    print ""
  n1.printParam()
  return [counter, loss, n1.weight, n1.bias]
  
#测试神经元训练,使用Adam训练算法
def unitTest_AdamOptimize():
  print "In unitTest_AdamOptimize"
  n1 = NeuralNode(2)
  n1.printParam();
  prevWeight = n1.weight
  
  x = np.ones(2)
  x[0] = 2
  x[1] = 2
  
  target = 1/(1+np.exp(1))
  
  counter = 0
  for i in range(1000000):
    #print "Round",i
    counter = i
    fowardResult = n1.forward(x)
    #print "Forward Result:",fowardResult
    loss = (fowardResult-target)*(fowardResult-target)
    #print "Loss=",loss
    dLossdvalue = 2*(target-fowardResult)
    grad = n1.backward(dLossdvalue)
    #print "grad=",grad
    n1.adaOptimization(0.001, grad[0], grad[1])
    if np.sum(np.abs(prevWeight - n1.weight)) < 1e-7:
      break
    prevWeight = n1.weight
    #n1.printParam()
    #print ""
  
  n1.printParam()
  return [counter, loss, n1.weight, n1.bias]

def trainWithLargerDataSet(sizeOfDataSet):
  iDims = 2
  iterationNumber = 1000;
  n1 = NeuralNode(iDims)
  trainDataSet = np.random.randn(sizeOfDataSet,iDims)
  np.savetxt('trainData.csv', trainDataSet, delimiter=',')
  prevWeight = n1.weight
  prevBias = n1.bias

  for iterIdx in range(0, iterationNumber):
    loss = 0
    for sampleIdx in range(0, sizeOfDataSet):
      # print "sampleIdx",sampleIdx
      fowardResult = n1.forward(trainDataSet[sampleIdx])
      # print "Forward Result:",fowardResult
      if (trainDataSet[sampleIdx][0] >= trainDataSet[sampleIdx][1] ):
        target = 0
      else:
        target = 1
      loss = loss + (fowardResult - target) * (fowardResult - target)
      # print "Loss=",loss
      dLossdvalue = 2 * (target - fowardResult)
      grad = n1.backward(dLossdvalue)
      # print "grad=",grad
      n1.adjustWeightAndBias(0.03, grad[0], grad[1])
    if iterIdx % 10 == 0:
      print "Loss=", loss, " iterIdx=", iterIdx
      print n1.getParam()," iterIdx=", iterIdx
    if (iterIdx > 0) and (np.sum(np.abs(prevWeight - n1.weight)) + np.abs(prevBias - n1.bias) < 1e-7):
      break
    prevWeight = n1.weight
    prevBias = n1.bias

  return [iterIdx, loss, n1.weight, n1.bias]


if __name__ == '__main__':
  naiveResultStr = ""
  adamResultStr = ""
  for i in range(1):
    # naiveResult =  unitTest_naiveTrain()
    # naiveResultStr = naiveResultStr + str(naiveResult) + "\n"
    # adamResult =  unitTest_AdamOptimize()
    # adamResultStr = adamResultStr + str(adamResult) + "\n"
    result = trainWithLargerDataSet(1000)
  print result