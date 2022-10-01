# pytorch模型定义

## pytorch模型定义的方式

### 重要的知识回顾

* Module 类是torch.nn 模块?提供的一个模型构造类nn.Module，是所有神经?网络模块的基类，我们可以继承它来定义我们想要的模型；
* PyTorch模型定义应包括两个主要部分：各个部分的初始化 __init__；数据流向定义 forward

基于 `nn.Module`，我们可以通过 `Sequential`，`ModuleList`和 `ModuleDic`三种方式定义PyTorch模型。

### Sequential

`Sequential` 类可以通过?加简单的方式定义模型。它可以接收一个子模块的有序字典(OrderedDict) 或者一系列子模块作为参数来逐一添加 `Module`的实?，?模型的前向计算就是将这些实?按添加的顺序逐?计算。

演示代码如下


```python
class MySequential(nn.Module):
    from collections import OrderedDict
    def __init__(self, *args):
        super(MySequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict): # 如果传入的是一个OrderedDict
            for key, module in args[0].items():
                self.add_module(key, module)  
                # add_module方法会将module添加进self._modules(一个OrderedDict)
        else:  # 传入的是一些Module
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
    def forward(self, input):
        # self._modules返回一个 OrderedDict，保证会按照成员添加时的顺序遍历成
        for module in self._modules.values():
            input = module(input)
        return input
```

使用实例

直接排列

```python
import torch.nn as nn
net = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 10), 
        )
print(net)
```

使用OrderedDic

```python
import collections
import torch.nn as nn
net2 = nn.Sequential(collections.OrderedDict([
          ('fc1', nn.Linear(784, 256)),
          ('relu1', nn.ReLU()),
          ('fc2', nn.Linear(256, 10))
          ]))
print(net2)
```


### ModuleList

`ModuleList` 接收一个子模块（或层，需属于 `nn.Module`类）的?表作为输入，然后也可以类似 `List`那样进行append和extend操作。同时，子模块或层的权重也会自动添加到网络中来。


```python
net = nn.ModuleList([nn.Linear(784, 256), nn.ReLU()])
net.append(nn.Linear(256, 10)) # # 类似List的append操作
print(net[-1])  # 类似List的索引访问
print(net)
```

要特别注意的是，`nn.ModuleList` 并没有定义一个网络，它只是将不同的模块储存在一起。`ModuleList`中元素的先后顺序并不代表其在网络中的真实位置顺序，需要经过forward函数指定各个层的先后顺序后才算完成了模型的定义。具体实现时用for循环即可完成：


```python
class model(nn.Module):
  def __init__(self, ...):
    super().__init__()
    self.modulelist = ...
    ...
  
  def forward(self, x):
    for layer in self.modulelist:
      x = layer(x)
    return x
```



```python
Linear(in_features=256, out_features=10, bias=True)
ModuleList(
  (0): Linear(in_features=784, out_features=256, bias=True)
  (1): ReLU()
  (2): Linear(in_features=256, out_features=10, bias=True)
)
```


### ModuleDict

`ModuleDict`和 `ModuleList`的作用类似，只是 `ModuleDict`能够更方便地为神经网络的层添加名称。


```python
net = nn.ModuleDict({
    'linear': nn.Linear(784, 256),
    'act': nn.ReLU(),
})
net['output'] = nn.Linear(256, 10) # 添加
print(net['linear']) # 访问
print(net.output)
print(net)
```


```python
Linear(in_features=784, out_features=256, bias=True)
Linear(in_features=256, out_features=10, bias=True)
ModuleDict(
  (act): ReLU()
  (linear): Linear(in_features=784, out_features=256, bias=True)
  (output): Linear(in_features=256, out_features=10, bias=True)
)
```


### 三种方法的比较与适用场景


`Sequential`适用于快速验证结果，因为已经明确了要用哪些层，直接写一下就好了，不需要同时写 `__init__`和 `forward`；

ModuleList和ModuleDict在某个完全相同的层需要重复出现多次时，非常方便实现，可以”一行顶多行“；

当我们需要之前层的信息的时候，比如 ResNets 中的残差计算，当前层的结果需要和之前层中的结果进行融合，一般使用 ModuleList/ModuleDict 比较方便。


## 利用模型块快速搭建复杂网络

下面以U-Net为例，我们来解释如何使用模型块

![](https://datawhalechina.github.io/thorough-pytorch/_images/5.2.1unet.png)


组成U-Net的模型块主要有如下几个部分：

1）每个子块内部的两次卷积（Double Convolution）

2）左侧模型块之间的下采样连接，即最大池化（Max pooling）

3）右侧模型块之间的上采样连接（Up sampling）

4）输出层的处理


### U-Net模型块实现

在使用PyTorch实现U-Net模型时，我们不必把每一层按序排列显式写出，这样太麻烦且不宜读，一种比较好的方法是先定义好模型块，再定义模型块之间的连接顺序和计算方式。就好比装配零件一样，我们先装配好一些基础的部件，之后再用这些可以复用的部件得到整个装配体。

这里的基础部件对应上一节分析的四个模型块，根据功能我们将其命名为：DoubleConv, Down, Up, OutConv。下面给出U-Net中模型块的PyTorch 实现：


```python
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
```


```python
class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)
```


```python
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
```


```python
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
```


### 利用模型块组装U-Net

使用写好的模型块，可以非常方便地组装U-Net模型。可以看到，通过模型块的方式实现了代码复用，整个模型结构定义所需的代码总行数明显减少，代码可读性也得到了提升。


```python
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
```


## PyTorch修改模型

除了自己构建PyTorch模型外，还有另一种应用场景：我们已经有一个现成的模型，但该模型中的部分结构不符合我们的要求，为了使用模型，我们需要对模型结构进行必要的修改。随着深度学习的发展和PyTorch越来越广泛的使用，有越来越多的开源模型可以供我们使用，很多时候我们也不必从头开始构建模型。因此，掌握如何修改PyTorch模型就显得尤为重要。



### 修改模型层

修改范例如下


```python
from collections import OrderedDict
classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(2048, 128)),
                          ('relu1', nn.ReLU()), 
                          ('dropout1',nn.Dropout(0.5)),
                          ('fc2', nn.Linear(128, 10)),
                          ('output', nn.Softmax(dim=1))
                          ]))
  
net.fc = classifier
```

这里的操作相当于将模型（net）最后名称为“fc”的层替换成了名称为“classifier”的结构，该结构是我们自己定义的。这里使用了第一节介绍的Sequential+OrderedDict的模型定义方式。至此，我们就完成了模型的修改，现在的模型就可以去做10分类任务了。


### 添加外部输入

实现范例如下


```python
class Model(nn.Module):
    def __init__(self, net):
        super(Model, self).__init__()
        self.net = net
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc_add = nn.Linear(1001, 10, bias=True)
        self.output = nn.Softmax(dim=1)
    
    def forward(self, x, add_variable):
        x = self.net(x)
        x = torch.cat((self.dropout(self.relu(x)), add_variable.unsqueeze(1)),1)
        x = self.fc_add(x)
        x = self.output(x)
        return x
```


### 添加额外输出

实现范例如下


```python
class Model(nn.Module):
    def __init__(self, net):
        super(Model, self).__init__()
        self.net = net
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(1000, 10, bias=True)
        self.output = nn.Softmax(dim=1)
    
    def forward(self, x, add_variable):
        x1000 = self.net(x)
        x10 = self.dropout(self.relu(x1000))
        x10 = self.fc1(x10)
        x10 = self.output(x10)
        return x10, x1000
```


## PyTorch模型保存与读取


### 模型存储格式

PyTorch存储模型主要采用pkl，pt，pth三种格式。


### 模型存储内容

一个PyTorch模型主要包含两个部分：模型结构和权重。其中模型是继承nn.Module的类，权重的数据结构是一个字典（key是层名，value是权重向量）。存储也由此分为两种形式：存储整个模型（包括结构和权重），和只存储模型权重。


```python
from torchvision import models
model = models.resnet152(pretrained=True)

# 保存整个模型
torch.save(model, save_dir)
# 保存模型权重
torch.save(model.state_dict, save_dir)
```

对于PyTorch而言，pt, pth和pkl **三种数据格式均支持模型权重和整个模型的存储** ，因此使用上没有差别。
