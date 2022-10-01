# 李宏毅机器学习第二章任务

## 回归定义和应用例子

Regression 就是找到一个函数function，通过输入特征 x，输出一个数值 Scalar。

## 常见应用

* 股市预测（Stock market forecast）

  * 输入：过去10年股票的变动、新闻咨询、公司并购咨询等
  * 输出：预测股市明天的平均值
* 自动驾驶（Self-driving Car）

  * 输入：无人车上的各个sensor的数据，例如路况、测出的车距等
  * 输出：方向盘的角度

## 创建一个模型的具体步骤（线性模型）

* step1：模型假设，选择模型框架（线性模型）
* step2：模型评估，如何判断众多模型的好坏（损失函数）
* step3：模型优化，如何筛选最优的模型（梯度下降）

### 一元线性模型（y=b+wx）

一元线性模型往往在生活中的推测中适用性较差，应用场景有限，因为其本身无法很好的处理多个异常点

### 多元线性模型

在实际应用中，function内传入的参数往往是多样的，这时一元的线性模型已经无法满足需要

所以我们假设  **线性模型 Linear model** ：*y*=**b**+∑**w**i*xi

* **x**i：就是各种特征(fetrure)
* **w**i：各个特征的权重
* **b**：偏移量

![demo1](https://oss.linklearner.com/leeml/chapter3/res/chapter3-1.png)

以宝可梦为例，在处理推测宝可梦战斗进化问题时，传入的function往往需要多个参数，不同参数有不同的权重

## 模型评估（损失函数）

![img](https://oss.linklearner.com/leeml/chapter3/res/chapter3-3.png)

将已知的10只宝可梦的进化数据放在图内观察，我们此时要做的就是描绘出预测出的function图像，观察其对于途中已知点的拟合度，如何判别这个标准呢？我们就需要引入一个函数（**损失函数）**。

![img](https://oss.linklearner.com/leeml/chapter3/res/chapter3-4.png)

![1657724016766](image/机器学习task02/1657724016766.png)

上式即为**均方误差损失函数（MSE）**

在回归问题中，均方误差损失函数用于度量样本点到回归曲线的距离，通过最小化平方损失使样本点可以更好地拟合回归曲线。均方误差损失函数（MSE）的值越小，表示预测模型描述的样本数据具有越好的精确度。由于无参数、计算成本低和具有明确物理意义等优点，MSE已成为一种优秀的距离度量方法。尽管MSE在图像和语音处理方面表现较弱，但它仍是评价信号质量的标准，在回归问题中，MSE常被作为模型的经验损失或算法的性能指标。

![img](https://www.zhihu.com/equation?tex=L%28Y%7Cf%28x%29%29%3D%5Csqrt%7B%5Cfrac%7B1%7D%7Bn%7D%5Csum_%7Bi%3D1%7D%5E%7BN%7D%7B%28Y_%7Bi%7D-f%28x_%7Bi%7D%29%29%5E%7B2%7D%7D%7D)

**L2损失又被称为欧氏距离，**是一种常用的距离度量方法，通常用于度量数据点之间的相似度。由于L2损失具有凸性和可微性，且在独立、同分布的高斯噪声情况下，它能提供最大似然估计，使得它成为回归问题、模式识别、图像处理中最常使用的损失函数。

![img](https://www.zhihu.com/equation?tex=L%28Y%7Cf%28x%29%29%3D%5Csum_%7Bi%3D1%7D%5E%7BN%7D%7B%7CY_%7Bi%7D-f%28x_%7Bi%7D%29%7C%7D)

**L1损失又称为曼哈顿距离**，表示残差的绝对值之和。L1损失函数对离群点有很好的鲁棒性，但它在残差为零处却不可导。另一个缺点是更新的梯度始终相同，也就是说，即使很小的损失值，梯度也很大，这样不利于模型的收敛。针对它的收敛问题，一般的解决办法是在优化算法中使用变化的学习率，在损失接近最小值时降低学习率。

在使用MSE的情况下，我们将 **w**, **b** 在二维坐标图中展示，如图所示：

![img](https://oss.linklearner.com/leeml/chapter3/res/chapter3-5.png)

## 最佳模型（梯度下降）

如何筛选最优的模型（参数w，b）

已知损失函数MSE，我们需要找到最符合实际需要的参数w，b

![img](https://oss.linklearner.com/leeml/chapter3/res/chapter3-6.png)

如何找寻呢？

我们可以以一个参数为例：

![img](https://oss.linklearner.com/leeml/chapter3/res/chapter3-7.png)

首先在这里引入一个概念 学习率 ：移动的步长，如图7中 \**η**


* 步骤1：随机选取一个 w
* 步骤2：计算微分，也就是当前的斜率，根据斜率来判定移动的方向
  * 大于0向右移动（增加w）
  * 小于0向左移动（减少w）
* 步骤3：根据学习率移动
* 重复步骤2和步骤3，直到找到最低点


通过过程的不断迭代，最终能够找出问题的局部最优解

**但是同时问题也来了：为什么此时的局部最优解是我们想要的全局最优解呢？**

在MSE算法内，对该函数求微分，发现其为一个凹函数

而凹函数在除去**等于0（Stuck at saddle point）和 趋近于0（Very slow at the plateau）**时都能够满足我们的需要。


## 如何验证训练好的模型的好坏

常见的验证方法即为画图观察法

我们可以将训练好的数据与真实数据对比，观察其拟合曲线与数据的贴合度

拟合情况往往有三种情况

![img](https://pic3.zhimg.com/80/v2-66592ae5d1d0fd4c8bb8b1e40e6cc022_720w.jpg)

当我们在做数据的拟合时，如果采用低次的模型拟合，往往会出现拟合不够，无法完成指定任务的情况，而当使用过高次的数据拟合时，数据往往在训练集中取得较好成绩，而在测试集中表现过差，原因就在于发生了过拟合现象，模型过于敏感而收录了太多噪声。

![img](https://oss.linklearner.com/leeml/chapter3/res/chapter3-20.png)


## 步骤优化

输入更多Pokemons数据，相同的起始CP值，但进化后的CP差距竟然是2倍。如图21，其实将Pokemons种类通过颜色区分，就会发现Pokemons种类是隐藏得比较深得特征，不同Pokemons种类影响了进化后的CP值的结果。

![](https://oss.linklearner.com/leeml/chapter3/res/chapter3-23.png)

而在数据处理的部分中，往往我们要对这些隐藏特征进行提前处理，来确保在训练中不会错过这些信息

### 常见的优化思路

#### 2个input的四个线性模型合并到一个模型中

![](https://oss.linklearner.com/leeml/chapter3/res/chapter3-25.png)

根据另一个参数xs来决定一部分系数

#### 如果希望模型更强大更好（添加多个参数，更多input）

但需要注意的是，添加参数时要对参数进行筛选，防止加入噪声

![img](https://oss.linklearner.com/leeml/chapter3/res/chapter3-28.png)

#### 加入正则化

更多特征，但是权重 w**w** 可能会使某些特征权值过高，仍旧导致overfitting，所以加入正则化

![1657728033345](image/机器学习task02/1657728033345.png)

![](https://oss.linklearner.com/leeml/chapter3/res/chapter3-29.png)

![](https://oss.linklearner.com/leeml/chapter3/res/chapter3-30.png)


## 总结

![](https://oss.linklearner.com/leeml/chapter3/res/chapter3-31.png)


* **Pokemon** ：原始的CP值极大程度的决定了进化后的CP值，但可能还有其他的一些因素。
* **Gradient descent** ：梯度下降的做法；后面会讲到它的理论依据和要点。
* **Overfitting和Regularization** ：过拟合和正则化，主要介绍了表象；后面会讲到更多这方面的理论
