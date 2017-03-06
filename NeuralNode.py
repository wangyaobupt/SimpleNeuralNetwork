#!/usr/bin/python
# -*- coding: UTF-8 -*-
import numpy as np

class NeuralNode:
  
  def __init__(self, inputDim):
    #当前结点与前一级的连接数目
    self.iDims = inputDim
    #权重向量，Shape = (iDims, )
    self.weight = 10*np.random.rand(self.iDims)
    #偏置 
    self.bias = 1
    #激活函数之前的计算结果
    self.dotValue = 1
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
    self.dotValue = np.dot(self.x,self.weight) + self.bias
    return sigmoid(self.dotValue)
  
  #backward: 输入前一级计算出的梯度，输出为两个数组
  #第一个数组: dx，iDims*1向量，即当前节点对于前一级每个输入的梯度
  #第二个数组：dw，iDims*1向量，当前节点对于每个权重的梯度
  #第三个数组：dbias, 1*1向量，当前节点对于偏置量的梯度
  def backward(self, gradient):
    ddot =  (1-self.dotValue) * self.dotValue #Sigmoid函数的求导
    dx = self.weight*ddot*gradient # 回传到x
    dw = self.x*ddot*gradient # 回传到w
    dbias = ddot*gradient # 回传到bias
    return [dx, dw, dbias]
  
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
    print "Weight = " + str(self.weight)
    print "Bias = " + str(self.bias)

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
    #print "Round",i
    fowardResult = n1.forward(x)
    #print "Forward Result:",fowardResult
    loss = (fowardResult-target)*(fowardResult-target)
    #print "Loss=",loss
    dLossdvalue = 2*(target-fowardResult)
    grad = n1.backward(dLossdvalue)
    #print "grad=",grad
    n1.adjustWeightAndBias(0.001, grad[1], grad[2])
    if np.sum(np.abs(prevWeight - n1.weight)) < 1e-7:
      break
    prevWeight = n1.weight
    #n1.printParam()
    #print ""
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
    n1.adaOptimization(0.001, grad[1], grad[2])
    if np.sum(np.abs(prevWeight - n1.weight)) < 1e-7:
      break
    prevWeight = n1.weight
    #n1.printParam()
    #print ""
  
  n1.printParam()
  return [counter, loss, n1.weight, n1.bias]

if __name__ == '__main__':
  naiveResultStr = ""
  adamResultStr = ""
  for i in range(100):
    naiveResult =  unitTest_naiveTrain()
    naiveResultStr = naiveResultStr + str(naiveResult) + "\n"
    adamResult =  unitTest_AdamOptimize()
    adamResultStr = adamResultStr + str(adamResult) + "\n"
  print naiveResultStr
  print ""
  print adamResultStr