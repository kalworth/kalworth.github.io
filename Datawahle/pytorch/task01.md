# pytorch基础知识

## 张量

如何理解张量呢，作为初学者，我很难理解张量出现的原因，也就是为什么我们选择张量这种数据结构，我们有必要在这里说明讨论清楚

在机器学习的落地实践中，我们往往要处理的是多特征的数据，例如常见的rgb图像，视频，

对于这些数据而言，对于我而言第一时间想象到的可用的数据结构就是数组，以图像为例，我可以设置一个三维数组来存放不同结构的特征。

但这时，数组固有的一些特征就会使问题复杂化，比如说数组往往只是一堆数的结合，没有办法在空间与方向上产生映射与联系。

**那么为什么就一定是张量呢？**

从代数角度讲， 它是向量的推广。我们知道，向量可以看成一维的“表格”（即分量按照顺序排成一排），矩阵是二维的“表格”（分量按照纵横位置排列）， 那么n阶张量就是所谓的n维的“表格”。**张量的严格定义是利用线性映射来描述的**。与矢量相类似，定义由若干坐标系改变时满足一定坐标转化关系的有序数组成的集合为张量。

这种特性可以保证我们在对数据进行一定的处理时不会破坏原有数据的关系，同时对于加速Gpu的计算过程也有一定的意义

### 张量的阶

张量的阶数可以理解为数据的复杂程度，对于一张图片而言，我们可以把它想象为三阶的张量，分别存储其高度宽度，以及对应的颜色（height, width, color_depth）

![](http://img.mp.sohu.com/upload/20170530/00c9c2c55abb4406876ecb385a89d4d8_th.png)

同时如果要表示由多组数据组成的数据集，我们可以引入另一个维度

(sample_size, height, width, color_depth)

![](http://img.mp.sohu.com/upload/20170530/0415535e411d47ecaddd0bc53058b6e0_th.png)

对于视频而言，我们则可以再引入一个维度，也就是时间维度

（sample_size, frames, width, height, color_depth)

这里不再举例

## 自动求导

要进行autograd必需先将tensor数据包成Variable。Varibale和tensor基本一致，所区别在于多了下面几个属性:

![img](https://upload-images.jianshu.io/upload_images/68960-7084a4be66464e40.png?imageMogr2/auto-orient/strip|imageView2/2/w/184/format/webp)

在张量的不断传播中，假如我们使用了自动求导的相关功能，那么对应数据的对应属性就会被动态更新在其中，并且每次的更新都是累加（所以在传播前要将其清零）。

如图，假设我们有一个输入变量input（数据类型为Variable）input是用户输入的，所以其创造者creator为null值，input经过第一个数据操作operation1（比如加减乘除运算）得到output1变量（数据类型仍为Variable），这个过程中会自动生成一个function1的变量（数据类型为Function的一个实例），而output1的创造者就是这个function1。随后，output1再经过一个数据操作生成output2，这个过程也会生成另外一个实例function2，output2的创造者creator为function2。

在这个向前传播的过程中，function1和function2记录了数据input的所有操作历史，当output2运行其backward函数时，会使得function2和function1自动反向计算input的导数值并存储在grad属性中。

creator为null的变量才能被返回导数，比如input，若把整个操作流看成是一张图（Graph）,那么像input这种creator为null的被称之为图的叶子（graph leaf）。而creator非null的变量比如output1和output2，是不能被返回导数的，它们的grad均为0。**所以只有叶子节点才能被autograd。**

### 计算图与反向传播


Autograd 是反向传播（Back propagation）中运用的自动微分系统。 从概念上来说，autograd 会记录一张有向无环图（Directed acyclic graph），这张图在所有创建新数据的操作被执行时记录下这些操作。图的叶子（leaves）是输入张量，根（root）是输出张量。 通过从根到叶追踪这张图，可以使用链式法则自动计算梯度。

在内部，这张有向无环图的结点都是 `Function`对象（实际上是表达式），可以通过 `apply()`来计算对图求值的结果。在正向传播中，Autograd 在计算正向传播结果的同时，构建了这张由 `Function`组成的有向无环图来以便进行反向传播（`torch.Tensor`的 `.grad_fn`属性就是这张图的入口点）。在正向传播计算完成以后，我们就可以追踪这张图来执行反向传播，计算出梯度。


![](https://tse3-mm.cn.bing.net/th/id/OIP-C.Fd62VTRnYeuKdI9B6k9SXAAAAA?w=227&h=180&c=7&r=0&o=5&dpr=1.25&pid=1.7)

## 并行计算简介

这部分内容Datawhale教程已经足够仔细，我不再赘述

---



> 参考文献：

> 百度：张量的定义与历史

> 简书：Zen_君

> 知乎：[可乐不加冰](https://www.zhihu.com/people/carrotry)
