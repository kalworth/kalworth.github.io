# 李宏毅机器学习第六章任务

## 卷积网络（CNN）

### 什么是卷积网络

卷积神经网络属于**前馈网络**的一种，是一种专门处理**类似网格数据**的神经网络，其特点就是每一层神经元只响应前一层的**局部范围**内的神经元。

卷积网络一般由： **卷积运算+非线性操作（RELU）+池化 +若干全连接层** 。

卷积网络之所以叫做卷积网络，是因为这种前馈网络其中采用了**卷积**的数学操作。在卷积网络之前，一般的网络采用的是**矩阵乘法**的方式，前一层的每一个单元都对下一层每一个单元有影响。

![img](https://img-blog.csdnimg.cn/20191030092949628.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDE3NzU2OA==,size_16,color_FFFFFF,t_70)

**卷积网络：**

![img](https://img-blog.csdnimg.cn/20191030093506990.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDE3NzU2OA==,size_16,color_FFFFFF,t_70)

响应层中的每一个神经元只受到输入层的**局部元素**的影响。以s2为例，它仅仅受到x1、x2、x3的影响，是一种稀疏的连接方式。

### 为什么使用卷积网络（卷积网络的优势）

**传统神经网络**都是采用**矩阵乘法**来建立 **输入和输出之间的关系** ，假如我们有 M 个输入和 N个输出，那么在训练过程中，我们需要 M×N 个参数去刻画输入和输出的关系 。当 M 和 N都很大，并且再加几层的卷积网络，这个参数量将会大的离谱。

卷积网络是如何处理这个问题的呢？

#### 稀疏连接

![](https://img-blog.csdnimg.cn/20191030102212595.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDE3NzU2OA==,size_16,color_FFFFFF,t_70)

卷积操作通过核函数将信息提取在了一个部分。

通过filter的不断移动，最终输入的矩阵被简化。

![](https://img-blog.csdnimg.cn/20191030102334470.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDE3NzU2OA==,size_16,color_FFFFFF,t_70)

通过CNN网络简化，流入下一级的变量将不再需要把所有的原始数据传输进去，后面的响应层只需要处理部分信息。

![](https://img-blog.csdnimg.cn/20191030093506990.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDE3NzU2OA==,size_16,color_FFFFFF,t_70)

#### 参数共享

由于filter的不变性，在卷积化的过程中我们只需要指定一个卷积核，通过卷积的数据所需要的参数大大减少，而且由于步长和核函数的性质，我们可以视作传入的数据在进入下一层神经网络时一部分将会共用同一函数，降低了模型复杂度。

#### 平移不变性

平移不变性：如果一个函数的输入做了一些改变，那么输出也跟着做出同样的改变，这就时平移不变性。

平移不变性是由参数共享的物理意义所得。在计算机视觉中，假如要识别一个图片中是否有一只猫，那么无论这只猫在图片的什么位置，我们都应该识别出来，即就是神经网络的输出对于平移不变性来说是等变的。

一下图为例：s2只和x1、x2、x3有关

![](https://img-blog.csdnimg.cn/20191030093506990.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDE3NzU2OA==,size_16,color_FFFFFF,t_70)

输入层数据向左移动一位：其中响应层的s2也只和x1、x2、x3有关，只是位置发生了稍微的变化

![](https://img-blog.csdnimg.cn/20191030141734823.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDE3NzU2OA==,size_16,color_FFFFFF,t_70)

### 池化（pooling）

#### 最大池化（Max pooling）

这个过程可以视为filter变成了一个Max函数，筛选出区域内的最大值

#### 平均池化（Average pooling）

这个过程可以视为filter变成了一个求均值的函数，求出区域内矩阵元素的平均值

#### 池化的意义

最大池化和平均池化有利于筛选出需要的主要元素，增强训练模型的鲁棒性，进行特征选择，另一方面可以解决过拟合问题。

文中转载链接：https://blog.csdn.net/weixin_44177568/article/details/102812050

### Flatten

![](https://oss.linklearner.com/leeml/chapter21/res/chapter21-22.png)

flatten就是feature map拉直，拉直之后就可以丢到fully connected feedforward netwwork，然后就结束了。

### CNN的应用

对于一个图像而言，我们可以使用多个卷积核来处理图像，从而得到不同灰度，不同特征的图像，在辨识物品图像中有重要作用。

![](https://oss.linklearner.com/leeml/chapter21/res/chapter21-27.png)

同时CNN在处理出现相同模式，或者一个模式多次出现的应用场景有较强优势。

![](https://oss.linklearner.com/leeml/chapter21/res/chapter21-38.png)

## 为什么要进行深度学习？

### 模组化

![](https://oss.linklearner.com/leeml/chapter22/res/chapter22-4.png)

当我们在做deep learning的时候，很多function被不断的重复使用，从结构上看，深度学习的结构更清晰，而且避免了设计的重复。

### 端到端的学习

![](https://oss.linklearner.com/leeml/chapter22/res/chapter22-22.png)

当我们用deep learning的时候，另外的一个好处是我们可以做End-to-end learning。

所谓的End-to-end learning的意思是这样，有时候我们要处理的问题是非常的复杂，比如说语音辨识就是一个非常复杂的问题。那么说我们要解一个machine problem我们要做的事情就是，先把一个Hypothesis funuctions(也就是找一个model)，当你要处理1的问题是很复杂的时候，你这个model里面它会是需要是一个生产线(由许多简单的function串接在一起)。比如说，你要做语音辨识，你要把语音送进来再到通过一层一层的转化，最后变成文字。当你多End-to-end learning的时候，意思就是说你只给你的model input跟output，你不告诉它说中间每一个function要咋样分工(只给input跟output，让它自己去学)，让它自己去学中间每一个function(生产线的每一个点)应该要做什么事情。

那在deep learning里面要做这件事的时候，你就是叠一个很深的neural network，每一层就是生产线的每一个点(每一层就会学到说自己要做什么样的事情)

*其余的内容在前面已经提及，在这里不再赘述。*

## 半监督学习（Semi-supervised Learnings）

### 半监督学习的生成模型

前面部分与监督学习的操作一样，先使用有监督的数据估计出 P(Ci)、μi 和 Σ，接下来使用未标记的数据 xu 来对这些参数重新估计，以二分类问题为例，估计过程主要分为如下两个步骤：

**初始化** θ={P(C1),P(C2),μ1,μ2,Σ}，（可以随机初始化，也可以根据已有的标记数据估计出来）。

**step1** ：根据初始化的参数计算无标记数据的后验概率Pθ(C1|xu) 。

**step2** ：更新模型参数：

![](https://pic3.zhimg.com/80/v2-266690c4d5bba8ee804c89df69cd10d6_720w.jpg)

解释如下：

利用已训练好的模型对unlabelled data 进行评估，得到更新后的P（C1），与μ1等参数，再带入更新θ的式子内

* 接着再返回 `step1`，直到参数收敛为止。
* 其实上面这个过程，我们用到了再机器学习领域一个超级NB的算法的思想，它就是EM(Expectation-maximization),**step1**就是 E，**step2**就是 M. 这样反复下去，在最终一定会收敛

### 半监督学习之低密度分离假设（Low-density Separation）


* 在用这个假设的时候，需要假设有一个很明显的区域(Low-density),能够把数据分开。Self-training
* 先对有标记数据训练出一个模型f*,这个可以模型可以用任何方法训练。
* 用这个 f∗ 来预测无标记的数据，预测出的就叫 pseudo label.
* 接下来，就用无标记数据中拿出一部分数据，放到有标记数据中，怎么选出这部分是自己定的，也可以对每一个数据提供一个权重。新加入了这些数据之后，就可以再重新训练一个 f∗，往复进行。
* 这招用在 regression 中，是没有用的，因为用预测出来的数字重新用来做训练，并不会影响模型的参数。
* 在做 self-training 时，其实就是把某个未标记数据指定一个分类，而在 generative model 中，其实就是把未标记数据对应于各个分类的概率计算出来。

![](https://oss.linklearner.com/leeml/chapter23/res/chapter23-9.png)

### 基于熵的正则化(Entropy-based Regularization)

假如未标记数据数据 xu 经过某一组参数估计后属于某一类的概率如下:

![](https://oss.linklearner.com/leeml/chapter23/res/chapter23-12.png)

又边红圈中的公式为熵的计算公式。由上图可知 xu 属于某一类的概率越大，熵的值E就越小，因此重新定义损失函数

![](https://pic3.zhimg.com/80/v2-76d43a2f4952788cfbf77cd74f40fbbe_720w.jpg)

，其中E(yu)可以微分，我们可以直接用梯度下降法来求解。

### Semi-supervised SVM

> 参考博客链接 [Semi-Supervised Support Vector Machines(S3VMs)_extremebingo的博客-CSDN博客](https://blog.csdn.net/extremebingo/article/details/79020907)

### 平滑假设(smoothness assumption)

![](https://oss.linklearner.com/leeml/chapter23/res/chapter23-14.png)

位于稠密数据区域的两个距离很近的样例的类标签相似，也就是说，当两个样例被稠密数据区域中的边连接时，它们在很大的概率下有相同的类标签；相反地，当两个样例被稀疏数据区域分开时，它们的类标签趋于不同。

### 基于图的方法

我们用Graph-based approach来表达这个通过高密度路径连接这件事情。就说我们现在把所有的data points都建成一个graph，每一笔data points都是这个graph上一个点，要想把他们之间的range建出来。有了这个graph以后，你就可以说：high density path的意思就是说，如果今天有两个点，他们在这个graph上面是相的(走的到)，那么他们这就是同一个class，如果没有相连，就算实际的距离也不是很远，那也不是同一个class。

![](https://oss.linklearner.com/leeml/chapter23/res/chapter23-19.png)

关于如何确立树和建立节点，实际上这个过程就是对原有数据的再挖掘与分析，将其特征分成一个又一个可处理的节点，处理的结果就是节点的通过率，也可以视为相似度。
