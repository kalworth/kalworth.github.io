# pytorch基础模块与应用

## 深度学习的基本思路与方向

回顾我们在完成一项机器学习任务时的步骤，首先 **需要对数据进行预处理** ，其中重要的步骤包括数据格式的统一和必要的数据变换，同时 **划分训练集和测试集** 。接下来 **选择模型** ，并设定 **损失函数和优化方法** ，以及对应的 **超参数** （当然可以使用sklearn这样的机器学习库中模型自带的损失函数和优化器）。最后用模型去拟合训练集数据，并在 **验证集/测试集上计算模型表现** 。

深度学习和机器学习在流程上类似，但在代码实现上有较大的差异。首先， **由于深度学习所需的样本量很大，一次加载全部数据运行可能会超出内存容量而无法实现；同时还有批（batch）训练等提高模型表现的策略，需要每次训练读取固定数量的样本送入模型中训练** ，因此深度学习在数据加载上需要有专门的设计。

在模型实现上，深度学习和机器学习也有很大差异。由于深度神经网络层数往往较多，同时会有一些用于实现特定功能的层（如卷积层、池化层、批正则化层、LSTM层等），因此 **深度神经网络往往需要“逐层”搭建，或者预先定义好可以实现特定功能的模块，再把这些模块组装起来** 。这种“定制化”的模型构建方式能够充分保证模型的灵活性，也对代码实现提出了新的要求。

接下来是损失函数和优化器的设定。这部分和经典机器学习的实现是类似的。但由于模型设定的灵活性， **因此损失函数和优化器要能够保证反向传播能够在用户自行定义的模型结构上实现** 。

上述步骤完成后就可以开始训练了。我们前面介绍了GPU的概念和GPU用于并行计算加速的功能，不过 **程序默认是在CPU上运行的** ，因此在代码实现中，需要把模型和数据“放到”GPU上去做运算，同时还需要保证损失函数和优化器能够在GPU上工作。如果使用多张GPU进行训练，还需要考虑模型和数据分配、整合的问题。此外，后续计算一些指标还需要把数据“放回”CPU。这里涉及到了一系列 **有关于GPU的配置和操作** 。

**深度学习中训练和验证过程最大的特点在于读入数据是按批的，每次读入一个批次的数据，放入GPU中训练，然后将损失函数反向传播回网络最前面的层，同时使用优化器调整网络参数。这里会涉及到各个模块配合的问题。训练/验证后还需要根据设定好的指标计算模型表现。**

经过以上步骤，一个深度学习任务就完成了。

## 准备工作

提前导入我们需要的包数据，配置超参数（超参数往往还有其他非固定的数值形势，可以在训练中进一步调整）

常见超参数：

* batch size
* 初始学习率（初始）
* 训练次数（max_epochs）
* GPU配置

## 引入我们的数据

PyTorch数据读入是通过Dataset+DataLoader的方式完成的，Dataset定义好数据的格式和数据变换形式，DataLoader用iterative的方式不断读入批次数据。


如果弄明白了pytorch中dataset类，你可以创建适应任意模型的数据集接口。

所谓数据集，无非就是一组{x:y}的集合吗，你只需要在这个类里说明“有一组{x:y}的集合”就可以了。

对于图像分类任务，图像+分类

对于目标检测任务，图像+bbox、分类

对于超分辨率任务，低分辨率图像+超分辨率图像

对于文本分类任务，文本+分类


下面以一个样例为例来解析如何引入我们的Dateset


```python
class Dataset(object):
    """An abstract class representing a Dataset.
    All other datasets should subclass it. All subclasses should override
    __len__, that provides the size of the dataset, and __getitem__,
    supporting integer indexing in range from 0 to len(self) exclusive.
    """  
    def getitem(self, index):
        raise NotImplementedError  
    def len(self):
        raise NotImplementedError  
    def__add__(self, other):
        return ConcatDataset([self, other]
```


上面的代码是pytorch给出的官方代码，其中__getitem__和__len__是子类必须继承的。

很好解释，pytorch给出的官方代码限制了标准，你要按照它的标准进行数据集建立。首先，__getitem__就是获取样本对，模型直接通过这一函数获得一对样本对{x:y}。__len__是指数据集长度。


以Datawhale给出的实例代码做细致解说


```python
class MyDataset(Dataset):
    def __init__(self, data_dir, info_csv, image_list, transform=None):
        """
        Args:
            data_dir: path to image directory.
            info_csv: path to the csv file containing image indexes
                with corresponding labels.
            image_list: path to the txt file contains image names to training/validation set
            transform: optional transform to be applied on a sample.
        """
        label_info = pd.read_csv(info_csv)
        image_file = open(image_list).readlines()
        self.data_dir = data_dir
        self.image_file = image_file
        self.label_info = label_info
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image and its labels
        """
        image_name = self.image_file[index].strip('\n')
        raw_label = self.label_info.loc[self.label_info['Image_index'] == image_name]
        label = raw_label.iloc[:,0]
        image_name = os.path.join(self.data_dir, image_name)
        image = Image.open(image_name).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.image_file)
```


***首先_init_函数来构造这个数据类的基本组成部分***，本函数内传入了四个参数（data_dir，info_csv，image_list，transform）

* data_dir：图像目录的路径。
* info_csv：包含图像索引与对应标签的 csv 文件的路径
* image_list：包含训练/验证集的图像名称的txt 文件的路径
* transform：要应用于样本的可选变换。一般会在图像识别内应用，形如反转图像，图像颠倒操作

这部分操作用来引入我们要处理的相关数据，将其转化为对应的Dataset类

剩下的两个函数是pytorch官方规定的必须继承的函数

___getitem 函数要求返回一个对应数据以及其标签或者分类信息___

所以在该函数内，我们对传入的文件内的所有图片名称和路径做处理，同时要获得其对应的标签信息，返回单个图像与其对应的标签

需要注意的是在这个函数内，我们会引入当前写入的变化

```python
if self.transform is not None:
            image = self.transform(image)
```

***len函数要求返回所有数据的总长***

这一步在设计的时候需要返回数据数量，方便之后进行的迭代操作

## 批次读入数据

以上我们成功完成了将数据转化为可读入的形式，下来我们需要使用pytorch官方的读入方式


```python
from torch.utils.data import DataLoader

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=4, shuffle=True, drop_last=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, num_workers=4, shuffle=False)
```


* batch_size：样本是按“批”读入的，batch_size就是每次读入的样本数
* num_workers：有多少个进程用于读取数据
* shuffle：是否将读入的数据打乱
* drop_last：对于样本最后一部分没有达到批次数的样本，使其不再参与训练

**dataloader本质上是一个可迭代对象，可以使用iter()进行访问，采用iter(dataloader)返回的是一个迭代器，然后可以使用next()访问。**
**也可以使用enumerate(dataloader)的形式访问。**

示例代码


```python
import matplotlib.pyplot as plt
images, labels = next(iter(val_loader))
print(images.shape)
plt.imshow(images[0].transpose(1,2,0))
plt.show()
```


## 构建模型

### 神经网络构建


```python
import torch
from torch import nn

class MLP(nn.Module):
  # 声明带有模型参数的层，这里声明了两个全连接层
  def __init__(self, **kwargs):
    # 调用MLP父类Block的构造函数来进行必要的初始化。这样在构造实例时还可以指定其他函数
    super(MLP, self).__init__(**kwargs)
    self.hidden = nn.Linear(784, 256)
    self.act = nn.ReLU()
    self.output = nn.Linear(256,10)
  
   # 定义模型的前向计算，即如何根据输入x计算返回所需要的模型输出
  def forward(self, x):
    o = self.act(self.hidden(x))
    return self.output(o)   
```


同样以这一段代码为核心解析

首先我们继承[nn.module](https://zhuanlan.zhihu.com/p/100000785#:~:text=%E7%AE%80%E5%8D%95%E7%9A%84%E8%AF%B4%EF%BC%8CMod,%E7%94%A8%E7%9A%84%E5%AD%90%E7%BD%91%E7%BB%9C%E5%B5%8C%E5%A5%97%E3%80%82)，同时调用父类的构造函数来完成初始化

**同时在下面声明了两个全连接层与一个激活函数**

由于笔者对于Linear函数的具体细节还不是很熟悉，所以在下方贴出源码声明

![](https://upload-images.jianshu.io/upload_images/7437869-92097b2cc629c072.png?imageMogr2/auto-orient/strip|imageView2/2/w/665/format/webp)


in_features指的是输入的二维张量的大小，即输入的[batch_size, size]中的size。

batch_size指的是每次训练（batch)的时候样本的大小。比如CNN train的样张图片是60张，设置batch_size=15，那么iteration=4。如果想多训练几次（因为可以每次的batch不是相同的数据），那么就是epoch。

所以nn.Linear()中的输入包括有输入的图片数量，同时还有每张图片的维度。

out_features指的是输出的二维张量的大小，即输出[batch_size，size]中的size是输出的张量维度，而batch_size与输入中的一致。


**同时代码内定义了前向计算的方式**

即先通过激活函数后进入hidden层，之后进入output层


我们可以实例化 MLP 类得到模型变? net 。其中， net(X) 会调用 MLP 继承?自 Module 类的 **call** 函数，这个函数将调?用 MLP 类定义的forward 函数来完成前向计算。


**pytorch主要也是按照 `__call__`, `__init__`,`forward`三个函数实现网络层之间的架构的**

**首先创建类对象m，然后通过 `m(input)`实际上调用 `__call__(input)`，然后 `__call__(input)`调用
`forward()`函数**


### 自定义我们需要的层

先贴出参考文章

[pytorch源码阅读系列之Parameter类 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/101052508) [AI课堂第18讲：DL深度学习——PyTorch自定义网络层 (baidu.com)](https://baijiahao.baidu.com/s?id=1737784552681280788)

之前一直不理解为什么在继承module类后不需要指定相关的权重，看来是初始化时作为参数被初始化了

官方的线性层实现如下：

![](https://pics4.baidu.com/feed/f603918fa0ec08fa88353e53faa1a16754fbdae3.png?token=8c1ce681a03194da78ca0740f791b2cb)


参数含义：

输入参数：input_features是输入向量长度，output_features是输出向量的长度，input是调用该类时的输入数据；

内部参数：weight是层的权重，bias是层的偏置；

内部函数：__init__是构造函数，forward是前向传播函数，reset_parameters是参数初始化函数。

其中nn.Parameter表示当前参数需要求导。

**根据上述官方案例层总结得出，要 **实现一个自定义层需要以下三点** ：**

A.自定义一个类，该类继承自nn.Module类，并且一定要实现两个基本的函数：构造函数__init__()、层的逻辑运算函数forward()；

B.在构造函数__init__()中实现层的参数定义；

C.在前向传播forward函数中实现批数据的前向传播逻辑，只要在nn.Module的子类中定义了forward()函数，backward()函数就会被自动实现。

注意：一般情况下我们定义的参数是可导的，但是如果自定义操作不可导，就需要我们手动实现backward()函数。

因此，自定义层可以分为两种，一种是带参数的，一种是不带参数的。


在自定义含模型参数的层时，参数将定义成 Parameter，**这表示参数需要求导。**

也可以使用ParameterList 和 ParameterDict 分别定义参数的表和字典。

*ParameterList 接收一个 Parameter 实例的列表作为输入然后得到一个参数表，使用的时候可以用索引来访问某个参数，另外也可以使用 append 和 extend 在表后面新增参数。*

*ParameterDict 接收一个 Parameter 实的字典作为输入然后得到一个参数字典，然后可以按照字典的规则使用。如使用 update() 新增参数，使用 keys() 返回所有键值，使用 items() 返回所有键值对。*


### 解析一个模型示例

![](https://datawhalechina.github.io/thorough-pytorch/_images/3.4.1.png)


```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 输入图像channel：1；输出channel：6；5x5卷积核
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 2x2 Max pooling
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # 如果是方阵,则可以只使用一个数字进行定义
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # 除去批处理维度的其他所有维度
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

```


## 模型初始化

对于这部分的应用总结来说，我们在调用init方法的时候需要针对不同类型的神经层进行分类设计

```python
def initialize_weights(self):
	for m in self.modules():
		# 判断是否属于Conv2d
		if isinstance(m, nn.Conv2d):
			torch.nn.init.xavier_normal_(m.weight.data)
			# 判断是否有偏置
			if m.bias is not None:
				torch.nn.init.constant_(m.bias.data,0.3)
		elif isinstance(m, nn.Linear):
			torch.nn.init.normal_(m.weight.data, 0.1)
			if m.bias is not None:
				torch.nn.init.zeros_(m.bias.data)
		elif isinstance(m, nn.BatchNorm2d):
			m.weight.data.fill_(1) 		 
			m.bias.data.zeros_()
```

我们往往会遍历此模型中的每个神经层，来判断该神经层的类型，以此为神经层做相应的初始化，赋给其相应的参数

## 损失函数的调用

关于不同种类的损失函数与实现细节，教程中已经讲述的很全面，笔者在下面讨论如何自定义损失函数

直接利用torch.Tensor提供的接口：

![](https://img-blog.csdnimg.cn/20200117232455510.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzI0NDA3NjU3,size_16,color_FFFFFF,t_70)

## 训练与评估

一个完整的图像分类的训练过程如下所示：


```python
def train(epoch):
    model.train()
    train_loss = 0
    for data, label in train_loader:
        data, label = data.cuda(), label.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(label, output)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()*data.size(0)
    train_loss = train_loss/len(train_loader.dataset)
		print('Epoch: {}\tTraining Loss: {:.6f}'.format(epoch, train_loss))
```


开始用当前批次数据做训练时，应当先将优化器的梯度置零：


```python
optimizer.zero_grad()
```


之后将data送入模型中训练：


```python
output = model(data)
```


根据预先定义的criterion计算损失函数：


```python
loss = criterion(output, label)
```


将loss反向传播回网络：


```python
loss.backward()
```


使用优化器更新模型参数：


```python
optimizer.step()
```

对应的，一个完整图像分类的验证过程如下所示：


```python
def val(epoch):   
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data, label in val_loader:
            data, label = data.cuda(), label.cuda()
            output = model(data)
            preds = torch.argmax(output, 1)
            loss = criterion(output, label)
            val_loss += loss.item()*data.size(0)
            running_accu += torch.sum(preds == label.data)
    val_loss = val_loss/len(val_loader.dataset)
    print('Epoch: {}\tTraining Loss: {:.6f}'.format(epoch, val_loss))
```


## 可视化

可视化中我们往往会将训练的每个过程以画图的方式描绘出来，方便我们直观的感受到训练的整体走向，以及思考未来的优化方向

常用的库有

* matplotlib
* Seaborn

剩下的笔者不熟（雾

## pytorch优化器

pytorch本身提供了许多优秀的优化器，这里讨论优化器的基类Optimizer

定义如下

```python
class Optimizer(object):
    def __init__(self, params, defaults):    
        self.defaults = defaults
        self.state = defaultdict(dict)
        self.param_groups = []
```

defaults：存储的是优化器的超参数

state：参数的缓存

params_groups：管理的参数组，是一个list，其中每个元素是一个字典，顺序是params，lr，momentum，dampening，weight_decay，nesterov

基本操作如下

```python
import os
import torch

# 设置权重，服从正态分布  --> 2 x 2
weight = torch.randn((2, 2), requires_grad=True)
# 设置梯度为全1矩阵  --> 2 x 2
weight.grad = torch.ones((2, 2))
# 输出现有的weight和data
print("The data of weight before step:\n{}".format(weight.data))
print("The grad of weight before step:\n{}".format(weight.grad))
# 实例化优化器
optimizer = torch.optim.SGD([weight], lr=0.1, momentum=0.9)
# 进行一步操作
optimizer.step()
# 查看进行一步后的值，梯度
print("The data of weight after step:\n{}".format(weight.data))
print("The grad of weight after step:\n{}".format(weight.grad))
# 权重清零
optimizer.zero_grad()
# 检验权重是否为0
print("The grad of weight after optimizer.zero_grad():\n{}".format(weight.grad))
# 输出参数
print("optimizer.params_group is \n{}".format(optimizer.param_groups))
# 查看参数位置，optimizer和weight的位置一样，我觉得这里可以参考Python是基于值管理
print("weight in optimizer:{}\nweight in weight:{}\n".format(id(optimizer.param_groups[0]['params'][0]), id(weight)))
# 添加参数：weight2
weight2 = torch.randn((3, 3), requires_grad=True)
optimizer.add_param_group({"params": weight2, 'lr': 0.0001, 'nesterov': True})
# 查看现有的参数
print("optimizer.param_groups is\n{}".format(optimizer.param_groups))
# 查看当前状态信息
opt_state_dict = optimizer.state_dict()
print("state_dict before step:\n", opt_state_dict)
# 进行5次step操作
for _ in range(50):
    optimizer.step()
# 输出现有状态信息
print("state_dict after step:\n", optimizer.state_dict())
# 保存参数信息
torch.save(optimizer.state_dict(),os.path.join(r"D:\pythonProject\Attention_Unet", "optimizer_state_dict.pkl"))
print("----------done-----------")
# 加载参数信息
state_dict = torch.load(r"D:\pythonProject\Attention_Unet\optimizer_state_dict.pkl") # 需要修改为你自己的路径
optimizer.load_state_dict(state_dict)
print("load state_dict successfully\n{}".format(state_dict))
# 输出最后属性信息
print("\n{}".format(optimizer.defaults))
print("\n{}".format(optimizer.state))
print("\n{}".format(optimizer.param_groups))
```
