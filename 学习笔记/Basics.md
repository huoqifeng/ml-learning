本系列关于AI/DeepLearning的大部分资料来源于Andrew NG在Coursera的讲课本内容，  
希望系统学习的同学请自行在Coursera注册学习或旁听： https://www.coursera.org  


## 为什么需要深度神经网络

在AI盛行的今天，DeepLearning和Neural Network几乎可以说是Machine Learning的代名词，  
尽管机器学习涵盖很多的领域，比如KNN, SVM,为什么NN占据了统治地位呢？下面Andrew NG的这张图可以说的很明白：  


![img](https://huoqifeng.github.io/img/deeplearning/scale-drive-nn.png)


 - 当数据集比较小的时候，传统机器学习的结果和NN的结果是差不多的
 - 当数据集逐渐增加，传统机器学习的结果达到一个平台，而NN还有提升的空间
 - 当数据集再增加，小规模的NN逐渐达到一个平台，而中等规模和大规模的NN还有提升空间
 - 数据集还在增加，中等规模的NN的结果也会达到一个平台，而大规模的NN还能有比较好的提高
 - 随着数据的增加，超大规模的NN可以带来巨大的performance的提升，当然，这个是以现代计算机的计算能力为基础的

总结起来，DeepLearning 和 Neural Network近几年大规模的流行和应用依赖两个条件：
 - 大规模的数据
 - 飞快提升的计算能力 


 ## Binary Classification
 机器学习的问题可以归结为识别问题，识别其实就是归结为分类问题，分类又分为是否的二分类和多对象的分类。
我们从二分类Binary Classification来开始我们的学习，多分类的问题只是在输出用Softmax替代Sigmoid,会在以后的文章提到。
Binary Classification是一个True/False问题，比如对于一个图片，图片中的是不是一只猫，对于一个邮件，是不是对你很重要等等。
以下图来说：

![img](https://huoqifeng.github.io/img/deeplearning/binary-classification.png)


 - 每一个图片都是一个64x64像素的图
 - 每一个像素都可以分解成R,G,B三基色，三个Channel。
 - 那么特征向量X就是64x64x3 = 12288
 - 对应的Y，可能是Cat(1),不是Cat(0)
 
在Binary Classification中用的Notation如下图：
![img](https://huoqifeng.github.io/img/deeplearning/notation-binary-classification.png) 
 - 对于每一个数据样本（x,y）,x是一个n维向量，n是特征数，y是0或1
 - m是样本数量
 - 总的样本的特征X 是m列的矩阵，每一列是一个样本的特征向量
 - 总的样本的Y也是一个m列的矩阵，每一列是一个1维向量，[0] 或 [1]
 

那么机器学习就是要通过对m个样例的分析和学习，得到一个模型（Model）,对于新的图片，应用这个模型来判断，新的图片是不是一只猫（1或者0）。
Logistic Regression就是这样的一种算法，基本的算法就是给每一个特征分配一个权重w，这样W就是一个n维的向量，n是特征的数量，
算法通过学习（迭代）得到W,那么对新的图片应用W就可以得到预测值，下面是Logistic Regression用到的Notation:
![img](https://huoqifeng.github.io/img/deeplearning/notation-logistic-regression.png)  
 - 需要注意的是常数b，是一个正则化因子，后面的文章会提到
 - W是一个n维向量，由样例通过迭代得到。
 - W + b 就是Model
 - 因为通过W,b计算的结果并不是1,0， 所以要应用一个Sigmoid 函数，使得计算结果在0,1之间
 - 后面会有讲到，W,b计算的直接结果是Z，应用Sigmoid函数之后的结果记为A，Sigmiod函数叫做Activation函数

## Lost, Cost 和梯度下降

 要计算W,b就需要不停的迭代，直到满意为止，这个就是个优化问题。
什么是满意呢？就是误差最小的时候，Lost, Cost就是定义误差的，经常采用的方法就是Gradient Descent
今天我们就来看Lost, Cost和Gradient Descent。

## 什么是Lost 和 Cost
![img](https://huoqifeng.github.io/img/deeplearning/lost-cost.png)
 - Lost 是一个样例上的误差（损失）
 - Cost 是所有m个样例的误差

## 什么是Gradient Descent
![img](https://huoqifeng.github.io/img/deeplearning/gradient-descent.png)
 - Gradient Descent 梯度下降， 是要找到下降最快的方向，也就是导数或导函数
 - 应用Gradient Descent 不停迭代计算dw, db得到不同的W, b,用不同时间的W,b重新计算Lost, Cost
 - 数学上可以证明对于凸函数一定能找到最优点或者局部最优点
 - 后续的课程会讲到如何如何找到全局最优点和避免过拟合

## 一次函数在某个点的导数
![img](https://huoqifeng.github.io/img/deeplearning/derivator.png)

## 高次函数在某个点的导数
![img](https://huoqifeng.github.io/img/deeplearning/derivator1.png)
![img](https://huoqifeng.github.io/img/deeplearning/derivator2.png)

## 如何实现Gradient Descent
![img](https://huoqifeng.github.io/img/deeplearning/implement-gradient-descent.png)
 - 在实现Gradient Descent的时候要用到导数。
 - 计算dw, db,后面讲完 Computatiom Graph 会详细描述

 接下来看看如何计算导数dw, db

## 导数
![img](https://huoqifeng.github.io/img/deeplearning/derivator.png)
 - 导数是导函数在某一点的值
 - 计算如上面的例子

## 高次函数的导数
![img](https://huoqifeng.github.io/img/deeplearning/derivator1.png)
![img](https://huoqifeng.github.io/img/deeplearning/derivator2.png)

## Computation Graph
要计算导数的计算先介绍computation Graph
![img](https://huoqifeng.github.io/img/deeplearning/computation-graph.png)
 - 蓝线是Forward Computation Graph
 - 红线代表Backward Computation Graph

## 用Computation Graph 计算导数
![img](https://huoqifeng.github.io/img/deeplearning/computation-graph-derivator.png)
 - 首先计算 dJ/dv = 3
 - 在计算 dv/da = 1
 - dJ/da = (dJ/dv)*(dv/da) = 1*3 = 3
