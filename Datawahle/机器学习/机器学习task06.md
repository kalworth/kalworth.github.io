# 李宏毅机器学习第五章任务

## 如何解决机器学习内出现的训练问题

### 局部最小值与鞍点（saddle point）

在[梯度下降](https://so.csdn.net/so/search?q=%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D&spm=1001.2101.3001.7020)法中，通常当梯度为零时终止运算，并默认此点为全局最优点(local minima)。但这种方法有缺陷，即可能到达鞍点(saddle points)而难以前进。鞍点处函数梯度等于0，但函数值并非局部最小，如图所示：

![](https://img-blog.csdnimg.cn/20210720170104215.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQzMzg5MTM5,size_16,color_FFFFFF,t_70)

那么如何避免最终梯度下降在鞍点处呢？

这里我们可以引入二次导数来解决这个问题

![](https://img-blog.csdnimg.cn/20210720170338868.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQzMzg5MTM5,size_16,color_FFFFFF,t_70)

通过观察鞍点以泰勒展开的结过后发现如果在鞍点处，其二次导数往往向我们指出了下一步的前进方向，也就是说，二次导数可以指导我们梯度下降的方向。

但在实际问题中，我们往往要处理的是多维度，多变量的回归问题，在多维空间内很难再产生马鞍点现象。

### 批次（batch）与动量（Momentum）

#### 什么是batch 和 Momentum？

**batch**

batch字面上是批量的意思，在[深度](https://so.csdn.net/so/search?q=%E6%B7%B1%E5%BA%A6&spm=1001.2101.3001.7020)学习中指的是计算一次cost需要的输入数据个数。

**Momentum：(动量，冲量)**

结合当前梯度与上一次更新信息，用于当前更新,可以防止在更新落入驻点时数据不再更新前进。

![](https://img-blog.csdnimg.cn/5a1f3941680242308632539eed03d3cc.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAY3Bvcmlu,size_20,color_FFFFFF,t_70,g_se,x_16)

#### 详解batch

##### 如何利用batch进行优化？

在实际的最优化过程中，我们将很多的数据分成好多份，每一份算出一个L i
 出来，然后使用它迭代计算θ ∗

![](https://img-blog.csdnimg.cn/8ac050f2238a4cba9ff76454391d88a1.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAY3Bvcmlu,size_20,color_FFFFFF,t_70,g_se,x_16)

update：是使用一个batch中的数据训练出的L 迭代一次。
epoch：使用每一个batch迭代过一次θ ，每一个epoch之后会重新分配batch进行训练（Shuffle）。

##### Small batch

将一批数据分割成多个batch，这样每一个epoch会更新多次，更新次数增多，但每次更新的方向不一致，容易受到干扰。

但是对于Small batch 而言，可以避免在梯度下降的过程中落在鲁棒性较差的sharp minima内。

![](https://img-blog.csdnimg.cn/4fbf2faa6c32487e91292ce84502451b.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAY3Bvcmlu,size_20,color_FFFFFF,t_70,g_se,x_16)

因为在一些需要中我们需要鲁棒性更高的模型来处理问题，对于训练数据更好的拟合模型往往存在一定的适配问题，较小的batch就能很好的解决这个问题。

##### Large batch

将一批数据仅仅进行简单分割，每组batch内含有大量数据，一个epoch迭代的次数少，每次更新时间长，方向更加稳定。

但是在GPU越来越作为计算主力的今天，很多运算往往能够并发进行，Large batch甚至在一定范围内的性能优于Small batch。

##### Small batch VS Large batch

![](https://img-blog.csdnimg.cn/36d0a3ad4ac4447e96133d099e40110d.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAY3Bvcmlu,size_20,color_FFFFFF,t_70,g_se,x_16)

### 调整自适应的学习率

前面我们在讨论如何进行梯度下降时就有讨论这一问题，当学习率一定时，容易卡在梯度较小而并非minima 的位置，这时就需要我们对学习率进行调整。

#### Root mean square

![](https://img-blog.csdnimg.cn/6894a339802d485dba3596bfe39317c9.png)

当坡度比较大时，gradient就比较大，σ就比较大，所以learning rate就比较小。当坡度比较小时，gradient就比较小，σ就比较小，所以learning rate就比较大。因此，有了σ后，我们就可以随着参数的不同，自动地调整learning rate的大小，而当参数过大或者过小时，之前的参数变化能够对这一变化产生修正。

##### 可能存在的问题

![](https://img-blog.csdnimg.cn/1f31881839084ba9a4ee62cbbe3275f4.png)

因为之前我们把所有的gradient拿来做平均，所以在纵轴的方向，初始的位置gradient很大，过了一段时间过后，gradient就会很小，于是纵轴就累积到了很小的σ，步伐就会变大，于是就爆发时喷射。当走到gradient很大的时候，σ就会变大，于是步伐就会变小，因此就会出现图中的情况。
之前的η一直是固定值，但应该把它跟时间联系在一起。这里要用到最常用的策略是Learning Rate Decay，这样我们就可以很平稳地走到终点。当越靠近终点时，η就会越小，当它想乱喷的时候，乘上一个很小的η，就可以平稳到达终点。

![](https://img-blog.csdnimg.cn/91e819ad8b104caabeedee1651df4c02.png)

#### RMSProp

RMSProp第一步和Root Mean Square是一样的，在下面的步骤中不一样的是在计算Root Mean Square时，每一个gradient都有同等的重要性，在RMSProp中，你可以调整它的重要性。

![](https://img-blog.csdnimg.cn/1a182b051f6b44baa05317f6c20d9749.png)

α如果设很小趋近于0时，就对与我们计算出来的gradient比较重要；α如果设很大趋近于1时，就对与我们计算出来的gradient比较不重要。我们现在动态调整σ。图中的黑线是error surface。在陡峭的地方gradient会变大，我们就把α设小，就可以很快地把σ的值变大，也就可以把步伐变小；在平坦的地方gradient会变小，我们就把α设大，就可以很快地把σ的值变小，也就可以把步伐变大。

![](https://img-blog.csdnimg.cn/2981958786d14bbb914a318216752999.png)

#### Adam: RMSProp + Momentum

![](https://img-blog.csdnimg.cn/8adcb20e729c49b698399e8c87558ecd.png)

#### Warm Up

Warm Up它是让Learning Rate先变大后变小，具体变大到多少，这个需要自己手动调整。为什么要用Warm Up？因为在使用Adam RMS Prop和Adagrad的时候，我们要计算σ，σ告诉我们某一个方向是陡峭还是平滑，但这需要多笔数据才能让统计更加精准。一开始我们的收集不够精准，所以要先收集有关σ的统计数据，等到足够精准后，再让Learning Rate上升。
![](https://img-blog.csdnimg.cn/d8f3ca2834d24d2589faf7517241f913.png)

#### Summary of Optimization

1． 这里我们添加了Momentum，现在不是顺着gradient的方向来更新参数，而是把之前全部的gradient的方向，做一个总和来当做更新的方向。
2． 如下图所示，这里需要除以gradient的Root Mean Square。Momentum直接把所有的gradient加起来，它会考虑方向；而Root Mean Square只考虑大小，不考虑方向。
![](https://img-blog.csdnimg.cn/a0035bc593f548bebcc7133798ce356a.png)

### 损失函数(Loss)也可能有影响

如果要求处理多分类问题，例如分为 1，2，3 三类问题，不对输出结果做处理很容易引发误差，例如相邻之间的元素更容易判断。

这时我们就需要对类别做处理，使用one-hot vector来解决整个情况。就是将类别划分为向量

1：【1，0，0】

2：【0，1，0】

3：【0，0，1】

然后对所计算的结果经过softmax后，得到y ′ ，然后才去计算y ′到y ^的距离。

#### Softmax

Softmax从字面上来说，可以分成soft和max两个部分。max故名思议就是最大值的意思。Softmax的核心在于soft，而soft有软的含义，与之相对的是hard硬。很多场景中需要我们找出数组所有元素中值最大的元素，实质上都是求的hardmax。

下面给出Softmax函数的定义（以第i个节点输出为例）：

![img](https://www.zhihu.com/equation?tex=Softmax%28z_%7Bi%7D%29%3D%5Cfrac%7Be%5E%7Bz_%7Bi%7D%7D%7D%7B%5Csum_%7Bc+%3D+1%7D%5E%7BC%7D%7Be%5E%7Bz_%7Bc%7D%7D%7D%7D)，其中 ![[公式]](https://www.zhihu.com/equation?tex=z_%7Bi%7D) 为第i个节点的输出值，C为输出节点的个数，即分类的类别个数。通过Softmax函数就可以将多分类的输出值转换为范围在[0, 1]和为1的概率分布。

Softmax将输出层转化为对应分类的概率，并且由于指数函数的特性，它能够很快地将大数据与小数据之间产生差别。

#### [交叉熵](https://so.csdn.net/so/search?q=%E4%BA%A4%E5%8F%89%E7%86%B5&spm=1001.2101.3001.7020)（Cross-Entropy）

这个暂时不太懂....等待日后补充。

### 批次标准化（Batch Normalization）

在训练损失下降的过程中，Loss的大小本身依然受制于数据

假如数据过大，参数减小时，可能使Loss变化过大，就会对回归（regression）产生影响。

我们要做的就是尽量减小这种行为带来的影响。


#### 批标准化好处


我们知道数据预处理做标准化可以加速收敛，同理，在神经网络使用标准化也可以加速收敛，而且还有更多好处。

具有正则化的效果（抑制过拟合）
提高模型的泛化能力
允许更高的学习速率从而加速收敛。

批标准化有助于梯度传播，因此允许更深的网络。对于有些特别深的网络，只有包含多个BatchNormalization层时才能进行训练。



#### 批标准化的实现过程

1.求每一个训练批次数据的均值
2.求每一个训练批次数据的方差
3.数据进行标准化
4.训练参数 γ，β
5.输出γ通过γ与β的线性变换得到原来的数值
在训练的正向传播中，不会改变当前输出，只记录下γ与β。
在反向传播的时候，根据求得的γ与β通过链式求导方式，求出学习速率以至改变权值。
