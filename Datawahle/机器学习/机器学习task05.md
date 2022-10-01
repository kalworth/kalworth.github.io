# 李宏毅机器学习第四章任务

## 深度学习的发展趋势

回顾一下deep learning的历史：

* 1958: Perceptron (linear model)
* 1969: Perceptron has limitation
* 1980s: Multi-layer perceptron
  * Do not have significant difference from DNN today
* 1986: Backpropagation
  * Usually more than 3 hidden layers is not helpful
* 1989: 1 hidden layer is “good enough”, why deep?
* 2006: RBM initialization (breakthrough)
* 2009: GPU
* 2011: Start to be popular in speech recognition
* 2012: win ILSVRC image competition
  感知机（Perceptron）非常像我们的逻辑回归（Logistics Regression）只不过是没有 `sigmoid`激活函数。09年的GPU的发展是很关键的，使用GPU矩阵运算节省了很多的时间。

### *激活函数*

**激活函数是向神经网络中引入非线性因素，通过激活函数神经网络就可以拟合各种曲线。激活函数主要分为饱和激活函数（Saturated Neurons）和非饱和函数（One-sided Saturations）** 。Sigmoid和Tanh是饱和激活函数，而ReLU以及其变种为非饱和激活函数。非饱和激活函数主要有如下优势：

1.非饱和激活函数可以解决梯度消失问题。

2.非饱和激活函数可以加速收敛。

![](https://pic2.zhimg.com/80/v2-9bede9b4cd7bba5cce65d6d83404681d_720w.jpg)

#### 梯度消失(Vanishing Gradients)

Sigmoid的函数图像和Sigmoid的梯度函数图像分别为(a)、(e)，从图像可以看出，函数两个边缘的梯度约为0，梯度的取值范围为(0,0.25)。求解方程为：

![](https://www.zhihu.com/equation?tex=y%3D1%2F%281%2Be%5E%7B-x%7D%29)

![](https://www.zhihu.com/equation?tex=y%5E%7B%27%7D%3Dy%281-y%29)

1. Sigmoid极容易导致梯度消失问题。饱和神经元会使得梯度消失问题雪上加霜，**假设神经元输入Sigmoid的值特别大或特别小，对应的梯度约等于0，即使从上一步传导来的梯度较大，该神经元权重(w)和偏置(bias)的梯度也会趋近于0，导致参数无法得到有效更新。**
2. 计算费时。 在神经网络训练中，常常要计算Sigmid的值进行幂计算会导致耗时增加。
3. Sigmoid函数不是关于原点中心对称的（zero-centered)。

Tanh激活函数解决了原点中心对称问题。

## 深度学习的三个步骤

* Step1：神经网络（Neural network）
* Step2：模型评估（Goodness of function）
* Step3：选择最优函数（Pick best function）

![](https://oss.linklearner.com/leeml/chapter13/res/chapter13-1.png)

### Step1：神经网络

神经网络（Neural network）里面的节点，类似我们的神经元。

![](https://oss.linklearner.com/leeml/chapter13/res/chapter13-2.png)

神经网络也可以有很多不同的连接方式，这样就会产生不同的结构（structure）在这个神经网络里面，我们有很多逻辑回归函数，其中每个逻辑回归都有自己的权重和自己的偏差，这些权重和偏差就是参数。
那这些神经元都是通过什么方式连接的呢？其实连接方式都是你手动去设计的。

#### 完全连接前馈神经网络

概念：前馈（feedforward）也可以称为前向，从信号流向来理解就是输入信号进入网络后，信号流动是单向的，即信号从前一层流向后一层，一直到输出层，其中任意两层之间的连接并没有反馈（feedback），亦即信号没有从后一层又返回到前一层。

![img](https://oss.linklearner.com/leeml/chapter13/res/chapter13-3.png)

##### 全链接和前馈的理解

* 输入层（Input Layer）：1层
* 隐藏层（Hidden Layer）：N层
* 输出层（Output Layer）：1层

![](https://oss.linklearner.com/leeml/chapter13/res/chapter13-6.png)

* 为什么叫全链接呢？
  * 因为layer1与layer2之间两两都有连接，所以叫做Fully Connect；
* 为什么叫前馈呢？
  * 因为现在传递的方向是由后往前传，所以叫做Feedforward。

##### 深度的理解

那什么叫做Deep呢？Deep = Many hidden layer。那到底可以有几层呢？这个就很难说了，以下是老师举出的一些比较深的神经网络的例子

![](https://oss.linklearner.com/leeml/chapter13/res/chapter13-7.png)

![](https://oss.linklearner.com/leeml/chapter13/res/chapter13-8.png)


* 2012 AlexNet：8层
* 2014 VGG：19层
* 2014 GoogleNet：22层
* 2015 Residual Net：152层
* 101 Taipei：101层

随着层数变多，错误率降低，随之运算量增大，通常都是超过亿万级的计算。对于这样复杂的结构，我们一定不会一个一个的计算，对于亿万级的计算，使用loop循环效率很低。


#### 本质：通过隐藏层进行特征转换

把隐藏层通过特征提取来替代原来的特征工程，这样在最后一个隐藏层输出的就是一组新的特征（相当于黑箱操作）而对于输出层，其实是把前面的隐藏层的输出当做输入（经过特征提取得到的一组最好的特征）然后通过一个多分类器（可以是softmax函数）得到最后的输出y。

#### 矩阵计算

计算方法就是：sigmoid（权重w【黄色】 * 输入【蓝色】+ 偏移量b【绿色】）= 输出

我们引入矩阵计算来加速计算过程，神经网络在推算特征的过程本质上是空间向量的一系列变化

![](https://oss.linklearner.com/leeml/chapter13/res/chapter13-10.png)

**为了防止卷积网络出现无效层，所以我们引进激活函数。**

因为在通过一系列的变化层的过程中，如果每次变化是线性的，那么多个相邻的神经层就有可以合并隐藏的可能，这会使得整个卷积网络设计显得低效而且愚蠢，更重要的是，非线性变化有助于改变向量本身的结构，使得函数输出约束在一定的范围内，防止出现爆内存等情况。


#### 神经网络的一些设计问题：

* 多少层？ 每层有多少神经元？
  这个问我们需要用尝试加上直觉的方法来进行调试。对于有些机器学习相关的问题，我们一般用特征工程来提取特征，但是对于深度学习，我们只需要设计神经网络模型来进行就可以了。对于语音识别和影像识别，深度学习是个好的方法，因为特征工程提取特征并不容易。
* 结构可以自动确定吗？
  有很多设计方法可以让机器自动找到神经网络的结构的，比如进化人工神经网络（Evolutionary Artificial Neural Networks）但是这些方法并不是很普及 。
* 我们可以设计网络结构吗？
  可以的，比如 CNN卷积神经网络（Convolutional Neural Network ）

### Step2: 模型评估

对于一个训练模型，我们往往使用Loss函数来评估其训练的好坏，对于一个神经网络而言我们采用**交叉熵（cross entropy）**来评判一个网络是否准确。

#### 什么是交叉熵？

该长度（把来自一个分布q的消息使用另一个分布p的最佳代码传达的平均消息长度）称为交叉熵。 形式上，我们可以将交叉熵定义为：

![](https://www.zhihu.com/equation?tex=H_p%28q%29%3D%5Csum_%7Bx%7D%7Bq%28x%29%5Clog_2%5Cleft%28+%5Cfrac%7B1%7D%7Bp%28x%29%7D+%5Cright%29%7D++++++++++++%3D-%5Csum_%7Bx%7D%7Bq%28x%29%5Clog_2+p%28x%29+%7D)

![](https://pic2.zhimg.com/80/v2-c0a21dc73338358f4a9e5f9f73ce5e01_720w.jpg)


注意，交叉熵 **不是对称的** 。

那么，为什么要关心交叉熵呢？ 这是因为，交叉熵为我们提供了一种表达两种概率分布的差异的方法。 ![[公式]](https://www.zhihu.com/equation?tex=p) 和 ![[公式]](https://www.zhihu.com/equation?tex=q) 的分布越不相同， ![[公式]](https://www.zhihu.com/equation?tex=p) 相对于 ![[公式]](https://www.zhihu.com/equation?tex=q) 的交叉熵将越大于 ![[公式]](https://www.zhihu.com/equation?tex=p) 的熵。


#### 用交叉熵作为损失函数


交叉熵常用来作为分类器的损失函数。不过，其它类型的模型也可能用它做损失函数，比如生成式模型。

我们有数据集 ![[公式]](https://www.zhihu.com/equation?tex=D%3D%28x_1%2C+y_1%29%2C+%28x_2%2C+y_2%29%2C+...%2C+%28x_N%2C+y_N%29) ，其中， ![[公式]](https://www.zhihu.com/equation?tex=x%3D%5Cleft%5C%7B+x_i+%5Cright%5C%7D_%7Bi%3D1%7D%5E%7BN%7D%5Cin%7BX%7D) 是特征值或输入变量； ![[公式]](https://www.zhihu.com/equation?tex=y%3D%5Cleft%5C%7B+y_i+%5Cright%5C%7D_%7Bi%3D1%7D%5E%7BN%7D%5Cin%7BY%7D) 是观察值，也是我们期待的模型的输出，最简单的情况是它只有两个离散值的取值，比如 ![[公式]](https://www.zhihu.com/equation?tex=y_i%5Cin%5Cleft%5C%7B+%22yes%22%2C+%22no%22+%5Cright%5C%7D) ，或者 ![[公式]](https://www.zhihu.com/equation?tex=y_i%5Cin%5Cleft%5C%7B+%22positive%22%2C+%22negative%22+%5Cright%5C%7D) ，或者 ![[公式]](https://www.zhihu.com/equation?tex=y_i%5Cin%5Cleft%5C%7B+-1%2C%2B1+%5Cright%5C%7D) 。能根据新的 ![[公式]](https://www.zhihu.com/equation?tex=x) 对 ![[公式]](https://www.zhihu.com/equation?tex=y) 做出预测的模型就是我们常用的二元分类器（Binary Classifier）。

我刻意没有选用 ![[公式]](https://www.zhihu.com/equation?tex=y_i%5Cin%5Cleft%5C%7B0%2C1+%5Cright%5C%7D) 作为例子，是为了避免有人认为我们观察到的数据 ![[公式]](https://www.zhihu.com/equation?tex=y) 就是概率向量，因而可以直接套用 ![[公式]](https://www.zhihu.com/equation?tex=y) 和模型的输出 ![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7By%7D) 之间的交叉熵作为损失函数。实际上，我们可以使用交叉熵作为分类器的损失函数的根本原因是我们使用了最大似然法，即我们通过在数据集上施用**最大似然法则**从而得到了与交叉熵一致的目标函数（或者损失函数）。我们观察原始数据时是看不到概率的，即使 ![[公式]](https://www.zhihu.com/equation?tex=y%5Cin%5Cleft%5C%7B0%2C1+%5Cright%5C%7D) ，它的取值0或1只是客观上的观察值而已，其概率意义是我们后来人为地加给它的。


#### 总体损失

![](https://oss.linklearner.com/leeml/chapter13/res/chapter13-18.png)

对于损失，我们不单单要计算一笔数据的，而是要计算整体所有训练数据的损失，然后把所有的训练数据的损失都加起来，得到一个总体损失L。接下来就是在function set里面找到一组函数能最小化这个总体损失L，或者是找一组神经网络的参数*θ*，来最小化总体损失L


### Step3：选择最优函数

![](https://oss.linklearner.com/leeml/chapter13/res/chapter13-20.png)

具体流程：θ是一组包含权重和偏差的参数集合，随机找一个初试值，接下来计算一下每个参数对应偏微分，得到的一个偏微分的集合∇L就是梯度,有了这些偏微分，我们就可以不断更新梯度得到新的参数，这样不断反复进行，就能得到一组最好的参数使得损失函数的值最小

## 为什么我们需要深度学习？

一个神经网络如果权重和偏差都知道的话就可以看成一个函数，他的输入是一个向量，对应的输出也是一个向量。不论是做回归模型（linear model）还是逻辑回归（logistics regression）都是定义了一个函数集（function set）。我们可以给上面的结构的参数设置为不同的数，就是不同的函数（function）。这些可能的函数（function）结合起来就是一个函数集（function set）。这个时候你的函数集（function set）是比较大的，是以前的回归模型（linear model）等没有办法包含的函数（function），所以说深度学习（Deep Learning）能表达出以前所不能表达的情况。

(转载自知乎[阿力阿哩哩](https://www.zhihu.com/people/bie-ying-xiang-zhi-li))

目前业界有句话被广为流传：

“ **数据和特征决定了机器学习的上限，而模型与算法则是逼近这个上限而已。** ”

因此，特征工程做得好，我们得到的预期结果也就好。

那特征工程到底是什么呢？在此之前，我们得了解特征的类型：文本特征、图像特征、数值特征和类别特征等。我们知道计算机并不能直接处理非数值型数据，那么在我们要将数据灌入机器学习算法之前，就必须将数据处理成算法能理解的格式，有时甚至需要对数据进行一些组合处理如分桶、缺失值处理和异常值处理等。

这也就是特征工程做的事：提取和归纳特征，让算法最大程度地利用数据，从而得到更好的结果。

不过，相较于传统的机器学习，深度学习的特征工程会简单许多，我们一般将数据处理成算法能够理解的格式即可，后期对神经网络的训练，就是提取和归纳特征的过程。

这也是深度学习能被广泛使用的原因之一：**特征工程能够自动化。**

## 反向传播

关于反向传播，其背后的思想与动态规划问题有类似之处，以最简单的01背包为例

我们已知在容量为5，可取为1的最优解，那么可以递推容量为5，可取为2的最优解，层层递进而且过程可以反向转化。

反向传播的过程本质上也是一种特征传递的过程。
