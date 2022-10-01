# pytorch进阶训练技巧

## 自定义损失函数

PyTorch在torch.nn模块为我们提供了许多常用的损失函数，比如：MSELoss，L1Loss，BCELoss...... 但是随着深度学习的发展，出现了越来越多的非官方提供的Loss，比如DiceLoss，HuberLoss，SobolevLoss...... 这些Loss Function专门针对一些非通用的模型，PyTorch不能将他们全部添加到库中去，因此这些损失函数的实现则需要我们通过自定义损失函数来实现。


### 以函数方式定义

代码演示如下

```python
def my_loss(output, target):
    loss = torch.mean((output - target)**2)
    return loss
```


### 以类方式定义

虽然以函数定义的方式很简单，但是以类方式定义更加常用，在以类方式定义损失函数时，我们如果看每一个损失函数的继承关系我们就可以发现 `Loss`函数部分继承自 `_loss` 部分继承自 `_WeightedLoss`, 而 `_WeightedLoss`继承自 `_loss`，`_loss`继承自  **nn.Module** 。我们可以将其当作神经网络的一层来对待，同样地，我们的损失函数类就需要继承自**nn.Module**类

Dice实现代码如下


```python
class DiceLoss(nn.Module):
    def __init__(self,weight=None,size_average=True):
        super(DiceLoss,self).__init__()
    
    def forward(self,inputs,targets,smooth=1):
        inputs = F.sigmoid(inputs)   
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()               
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        return 1 - dice

# 使用方法  
criterion = DiceLoss()
loss = criterion(input,targets)
```

*注：我们在上面的代码中继承了nn.module类，当我们需要使用一些框架之外的功能时则需要一些利用其他操作实现，多数实现方式基于numpy/scipy*

## 动态调整学习率

在实际学习应用中我们可能会遇到类似如下的情况

![](https://img-blog.csdnimg.cn/1f31881839084ba9a4ee62cbbe3275f4.png)

学习速率设置过小，会极大降低收敛速度，增加训练时间；学习率太大，可能导致参数在最优解两侧来回振荡，那么我们就需要学习率能够动态进行调整，减少我们的训练成本。


### 使用官方scheduler

在训练神经网络的过程中，学习率是最重要的超参数之一，作为当前较为流行的深度学习框架，PyTorch已经在 `torch.optim.lr_scheduler`为我们封装好了一些动态调整学习率的方法供我们使用，如下面列出的这些scheduler。


* [`lr_scheduler.LambdaLR`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LambdaLR.html#torch.optim.lr_scheduler.LambdaLR)
* [`lr_scheduler.MultiplicativeLR`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.MultiplicativeLR.html#torch.optim.lr_scheduler.MultiplicativeLR)
* [`lr_scheduler.StepLR`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html#torch.optim.lr_scheduler.StepLR)
* [`lr_scheduler.MultiStepLR`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.MultiStepLR.html#torch.optim.lr_scheduler.MultiStepLR)
* [`lr_scheduler.ExponentialLR`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ExponentialLR.html#torch.optim.lr_scheduler.ExponentialLR)
* [`lr_scheduler.CosineAnnealingLR`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html#torch.optim.lr_scheduler.CosineAnnealingLR)
* [`lr_scheduler.ReduceLROnPlateau`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html#torch.optim.lr_scheduler.ReduceLROnPlateau)
* [`lr_scheduler.CyclicLR`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CyclicLR.html#torch.optim.lr_scheduler.CyclicLR)
* [`lr_scheduler.OneCycleLR`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html#torch.optim.lr_scheduler.OneCycleLR)
* [`lr_scheduler.CosineAnnealingWarmRestarts`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.html#torch.optim.lr_scheduler.CosineAnnealingWarmRestarts)

其作用在于对optimizer中的学习率进行更新、调整，更新的方法是scheduler.step()。

通常而言，在一个batch_size内先进行optimizer.step()完成权重参数的更新过程，然后再进行scheduler.step()完成对学习率参数的更新过程。

实现代码如下


```python
# 选择一种优化器
optimizer = torch.optim.Adam(...) 
# 选择上面提到的一种或多种动态调整学习率的方法
scheduler1 = torch.optim.lr_scheduler.... 
scheduler2 = torch.optim.lr_scheduler....
...
schedulern = torch.optim.lr_scheduler....
# 进行训练
for epoch in range(100):
    train(...)
    validate(...)
    optimizer.step()
    # 需要在优化器参数更新之后再动态调整学习率
	scheduler1.step() 
	...
    schedulern.step()
```


 **注** ：

我们在使用官方给出的 `torch.optim.lr_scheduler`时，需要将 `scheduler.step()`放在 `optimizer.step()`后面进行使用。


### 自定义scheduler

虽然PyTorch官方给我们提供了许多的API，但是在实验中也有可能碰到需要我们自己定义学习率调整策略的情况，而我们的方法是自定义函数 `adjust_learning_rate`来改变 `param_group`中 `lr`的值

实现如下


```python
def adjust_learning_rate(optimizer, epoch):
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
```


```python
def adjust_learning_rate(optimizer,...):
    ...
optimizer = torch.optim.SGD(model.parameters(),lr = args.lr,momentum = 0.9)
for epoch in range(10):
    train(...)
    validate(...)
    adjust_learning_rate(optimizer,epoch)
```


## 模型微调-torchvision

简单来说，就是我们先找到一个同类的别人训练好的模型，把别人现成的训练好了的模型拿过来，换成自己的数据，通过训练调整一下参数。 在PyTorch中提供了许多预训练好的网络模型（VGG，ResNet系列，mobilenet系列......），这些模型都是PyTorch官方在相应的大型数据集训练好的。学习如何进行模型微调，可以方便我们快速使用预训练模型完成自己的任务。


### 模型微调的流程


1. 在源数据集(如ImageNet数据集)上预训练一个神经网络模型，即源模型。
2. 创建一个新的神经网络模型，即目标模型。它复制?源模型上除?输出层外的所有模型设计及其参数。我们假设这些模型参数包含?源数据集上学习到的知识，且这些知识同样适用于目标数据集。我们还假设源模型的输出层跟源数据集的标签紧密相关，因此在目标模型中不予采用。
3. 为目标模型添加一个输出?小为?标数据集类别个数的输出层，并随机初始化该层的模型参数。
4. 在目标数据集上训练目标模型。我们将从头训练输出层，而其余层的参数都是基于源模型的参数微调得到的。


### 使用已有模型结构

通过 `True`或者 `False`来决定是否使用预训练好的权重，在默认状态下 `pretrained=False`，意味着我们不使用预训练得到的权重，当 `pretrained=True`，意味着我们将使用在一些数据集上预训练得到的权重。


```python
import torchvision.models as models
resnet18 = models.resnet18(pretrained=True)
alexnet = models.alexnet(pretrained=True)
squeezenet = models.squeezenet1_0(pretrained=True)
vgg16 = models.vgg16(pretrained=True)
densenet = models.densenet161(pretrained=True)
inception = models.inception_v3(pretrained=True)
googlenet = models.googlenet(pretrained=True)
shufflenet = models.shufflenet_v2_x1_0(pretrained=True)
mobilenet_v2 = models.mobilenet_v2(pretrained=True)
mobilenet_v3_large = models.mobilenet_v3_large(pretrained=True)
mobilenet_v3_small = models.mobilenet_v3_small(pretrained=True)
resnext50_32x4d = models.resnext50_32x4d(pretrained=True)
wide_resnet50_2 = models.wide_resnet50_2(pretrained=True)
mnasnet = models.mnasnet1_0(pretrained=True)
```


### 训练特定层

在默认情况下，参数的属性 `<span class="pre">.requires_grad</span><span>?</span><span class="pre">=</span><span>?</span><span class="pre">True</span>`，如果我们从头开始训练或微调不需要注意这里。但如果我们正在提取特征并且只想为新初始化的层计算梯度，其他参数不进行改变。那我们就需要通过设置 `<span class="pre">requires_grad</span><span>?</span><span class="pre">=</span><span>?</span><span class="pre">False</span>`来冻结部分层。在PyTorch官方中提供了这样一个例程。


```python
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
```


## 模型微调 - timm


Pytorch Image Models (timm) 整合了常用的models, layers, utilities, optimizers, schedulers, data-loaders / augmentations, and reference training / validation scripts，它的目的是将各种SOTA模型整合在一起，并具有再现ImageNet训练结果的能力。

作者：Ross Wightman，来自加拿大温哥华。


### 使用和修改预训练模型

在得到我们想要使用的预训练模型后，我们可以通过 `<span class="pre">timm.create_model()</span>`的方法来进行模型的创建，我们可以通过传入参数 `<span class="pre">pretrained=True</span>`，来使用预训练模型。同样的，我们也可以使用跟torchvision里面的模型一样的方法查看模型的参数，类型


```python
import timm
import torch

model = timm.create_model('resnet34',pretrained=True)
x = torch.randn(1,3,224,224)
output = model(x)
output.shape
```


## 半精度训练


我们提到PyTorch时候，总会想到要用硬件设备GPU的支持，也就是“卡”。GPU的性能主要分为两部分：算力和显存，前者决定了显卡计算的速度，后者则决定了显卡可以同时放入多少数据用于计算。在可以使用的显存数量一定的情况下，每次训练能够加载的数据更多（也就是batch size更大），则也可以提高训练效率。另外，有时候数据本身也比较大（比如3D图像、视频等），显存较小的情况下可能甚至batch size为1的情况都无法实现。因此，合理使用显存也就显得十分重要。

我们观察PyTorch默认的浮点数存储方式用的是torch.float32，小数点后位数更多固然能保证数据的精确性，但绝大多数场景其实并不需要这么精确，只保留一半的信息也不会影响结果，也就是使用torch.float16格式。由于数位减了一半，因此被称为“半精度"


### 半精度训练的设置

在PyTorch中使用autocast配置半精度训练，同时需要在下面三处加以设置：


```
from torch.cuda.amp import autocast
```

在模型定义中，使用python的装饰器方法，用autocast装饰模型中的forward函数

```
@autocast()   
def forward(self, x):
    ...
    return x
```

在训练过程中，只需在将数据输入模型及其之后的部分放入“with autocast():“即可：

```
for x in train_loader:
	x = x.cuda()
	with autocast():
        output = model(x)
        ...
```


**注意：**

半精度训练主要适用于数据本身的size比较大（比如说3D图像、视频等）。当数据本身的size并不大时（比如手写数字MNIST数据集的图片尺寸只有28*28），使用半精度训练则可能不会带来显著的提升。


## 数据增强-imgaug

深度学习最重要的是数据。我们需要大量数据才能避免模型的过度拟合。但是我们在许多场景无法获得大量数据，例如医学图像分析。数据增强技术的存在是为了解决这个问题，这是针对有限数据问题的解决方案。数据增强一套技术，可提高训练数据集的大小和质量，以便我们可以使用它们来构建更好的深度学习模型。 在计算视觉领域，生成增强图像相对容易。即使引入噪声或裁剪图像的一部分，模型仍可以对图像进行分类，数据增强有一系列简单有效的方法可供选择，有一些机器学习库来进行计算视觉领域的数据增强



### imgaug简介

`imgaug`是计算机视觉任务中常用的一个数据增强的包，相比于 `torchvision.transforms`，它提供了更多的数据增强方法，因此在各种竞赛中，人们广泛使用 `imgaug`来对数据进行增强操作。


### imgaug的使用

imgaug仅仅提供了图像增强的一些方法，但是并未提供图像的IO操作，因此我们需要使用一些库来对图像进行导入，建议使用imageio进行读入，如果使用的是opencv进行文件读取的时候，需要进行手动改变通道，将读取的BGR图像转换为RGB图像。除此以外，当我们用PIL.Image进行读取时，因为读取的图片没有shape的属性，所以我们需要将读取到的img转换为np.array()的形式再进行处理。因此官方的例程中也是使用imageio进行图片读取。

使用范例如下


```python 
from imgaug import augmenters as iaa

# 设置随机数种子
ia.seed(4)

# 实例化方法
rotate = iaa.Affine(rotate=(-4,45))
img_aug = rotate(image=img)
ia.imshow(img_aug)
```

![img](https://datawhalechina.github.io/thorough-pytorch/_images/Lenna_original.png) to ![](https://datawhalechina.github.io/thorough-pytorch/_images/rotate.png)

这是对一张图片进行一种操作方式，但实际情况下，我们可能对一张图片做多种数据增强处理。这种情况下，我们就需要利用 `imgaug.augmenters.Sequential()`来构造我们数据增强的pipline，该方法与 `torchvison.transforms.Compose()`相类似。


```python 
iaa.Sequential(children=None, # Augmenter集合
               random_order=False, # 是否对每个batch使用不同顺序的Augmenter list
               name=None,
               deterministic=False,
               random_state=None)
# 构建处理序列
aug_seq = iaa.Sequential([
    iaa.Affine(rotate=(-25,25)),
    iaa.AdditiveGaussianNoise(scale=(10,60)),
    iaa.Crop(percent=(0,0.2))
])
# 对图片进行处理，image不可以省略，也不能写成images
image_aug = aug_seq(image=img)
ia.imshow(image_aug)
```

![](https://datawhalechina.github.io/thorough-pytorch/_images/aug_seq.png)

对一批次的图片进行处理时，我们只需要将待处理的图片放在一个 `list`中，并将image改为image即可进行数据增强操作，具体实际操作如下：


```python
images = [img,img,img,img,]
images_aug = rotate(images=images)
ia.imshow(np.hstack(images_aug))
```

imgaug相较于其他的数据增强的库，有一个很有意思的特性，即就是我们可以通过 `imgaug.augmenters.Sometimes()`对batch中的一部分图片应用一部分Augmenters,剩下的图片应用另外的Augmenters。


```python
iaa.Sometimes(p=0.5,  # 代表划分比例
              then_list=None,  # Augmenter集合。p概率的图片进行变换的Augmenters。
              else_list=None,  #1-p概率的图片会被进行变换的Augmenters。注意变换的图片应用的Augmenter只能是then_list或者else_list中的一个。
              name=None,
              deterministic=False,
              random_state=None)
```


## 使用argparse进行调参

在深度学习中时，超参数的修改和保存是非常重要的一步，尤其是当我们在服务器上跑我们的模型时，如何更方便的修改超参数是我们需要考虑的一个问题。这时候，要是有一个库或者函数可以解析我们输入的命令行参数再传入模型的超参数中该多好。到底有没有这样的一种方法呢？答案是肯定的，这个就是 Python 标准库的一部分：Argparse。


总的来说，我们可以将argparse的使用归纳为以下三个步骤。

* 创建 `<span class="pre">ArgumentParser()</span>`对象
* 调用 `<span class="pre">add_argument()</span>`方法添加参数
* 使用 `<span class="pre">parse_args()</span>`解析参数


```python
# demo.py
import argparse

# 创建ArgumentParser()对象
parser = argparse.ArgumentParser()

# 添加参数
parser.add_argument('-o', '--output', action='store_true', 
    help="shows output")
# action = `store_true` 会将output参数记录为True
# type 规定了参数的格式
# default 规定了默认值
parser.add_argument('--lr', type=float, default=3e-5, help='select the learning rate, default=1e-3') 

parser.add_argument('--batch_size', type=int, required=True, help='input batch size')  
# 使用parse_args()解析函数
args = parser.parse_args()

if args.output:
    print("This is some output")
    print(f"learning rate:{args.lr} ")
```

Datawhale分享了一种高效的参数调整方法，也就是将配置文件与执行代码分离开，需要时引入配置文件


```python
import argparse  
  
def get_options(parser=argparse.ArgumentParser()):  
  
    parser.add_argument('--workers', type=int, default=0,  
                        help='number of data loading workers, you had better put it '  
                              '4 times of your gpu')  
  
    parser.add_argument('--batch_size', type=int, default=4, help='input batch size, default=64')  
  
    parser.add_argument('--niter', type=int, default=10, help='number of epochs to train for, default=10')  
  
    parser.add_argument('--lr', type=float, default=3e-5, help='select the learning rate, default=1e-3')  
  
    parser.add_argument('--seed', type=int, default=118, help="random seed")  
  
    parser.add_argument('--cuda', action='store_true', default=True, help='enables cuda')  
    parser.add_argument('--checkpoint_path',type=str,default='',  
                        help='Path to load a previous trained model if not empty (default empty)')  
    parser.add_argument('--output',action='store_true',default=True,help="shows output")  
  
    opt = parser.parse_args()  
  
    if opt.output:  
        print(f'num_workers: {opt.workers}')  
        print(f'batch_size: {opt.batch_size}')  
        print(f'epochs (niters) : {opt.niter}')  
        print(f'learning rate : {opt.lr}')  
        print(f'manual_seed: {opt.seed}')  
        print(f'cuda enable: {opt.cuda}')  
        print(f'checkpoint_path: {opt.checkpoint_path}')  
  
    return opt  
  
if __name__ == '__main__':  
    opt = get_options()
```
