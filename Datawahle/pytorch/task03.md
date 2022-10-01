# pytorchģ�Ͷ���

## pytorchģ�Ͷ���ķ�ʽ

### ��Ҫ��֪ʶ�ع�

* Module ����torch.nn ģ��?�ṩ��һ��ģ�͹�����nn.Module����������?����ģ��Ļ��࣬���ǿ��Լ̳���������������Ҫ��ģ�ͣ�
* PyTorchģ�Ͷ���Ӧ����������Ҫ���֣��������ֵĳ�ʼ�� __init__������������ forward

���� `nn.Module`�����ǿ���ͨ�� `Sequential`��`ModuleList`�� `ModuleDic`���ַ�ʽ����PyTorchģ�͡�

### Sequential

`Sequential` �����ͨ��?�Ӽ򵥵ķ�ʽ����ģ�͡������Խ���һ����ģ��������ֵ�(OrderedDict) ����һϵ����ģ����Ϊ��������һ��� `Module`��ʵ?��?ģ�͵�ǰ�������ǽ���Щʵ?����ӵ�˳����?���㡣

��ʾ��������


```python
class MySequential(nn.Module):
    from collections import OrderedDict
    def __init__(self, *args):
        super(MySequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict): # ����������һ��OrderedDict
            for key, module in args[0].items():
                self.add_module(key, module)  
                # add_module�����Ὣmodule��ӽ�self._modules(һ��OrderedDict)
        else:  # �������һЩModule
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
    def forward(self, input):
        # self._modules����һ�� OrderedDict����֤�ᰴ�ճ�Ա���ʱ��˳�������
        for module in self._modules.values():
            input = module(input)
        return input
```

ʹ��ʵ��

ֱ������

```python
import torch.nn as nn
net = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 10), 
        )
print(net)
```

ʹ��OrderedDic

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

`ModuleList` ����һ����ģ�飨��㣬������ `nn.Module`�ࣩ��?����Ϊ���룬Ȼ��Ҳ�������� `List`��������append��extend������ͬʱ����ģ�����Ȩ��Ҳ���Զ���ӵ�����������


```python
net = nn.ModuleList([nn.Linear(784, 256), nn.ReLU()])
net.append(nn.Linear(256, 10)) # # ����List��append����
print(net[-1])  # ����List����������
print(net)
```

Ҫ�ر�ע����ǣ�`nn.ModuleList` ��û�ж���һ�����磬��ֻ�ǽ���ͬ��ģ�鴢����һ��`ModuleList`��Ԫ�ص��Ⱥ�˳�򲢲��������������е���ʵλ��˳����Ҫ����forward����ָ����������Ⱥ�˳�����������ģ�͵Ķ��塣����ʵ��ʱ��forѭ��������ɣ�


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

`ModuleDict`�� `ModuleList`���������ƣ�ֻ�� `ModuleDict`�ܹ��������Ϊ������Ĳ�������ơ�


```python
net = nn.ModuleDict({
    'linear': nn.Linear(784, 256),
    'act': nn.ReLU(),
})
net['output'] = nn.Linear(256, 10) # ���
print(net['linear']) # ����
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


### ���ַ����ıȽ������ó���


`Sequential`�����ڿ�����֤�������Ϊ�Ѿ���ȷ��Ҫ����Щ�㣬ֱ��дһ�¾ͺ��ˣ�����Ҫͬʱд `__init__`�� `forward`��

ModuleList��ModuleDict��ĳ����ȫ��ͬ�Ĳ���Ҫ�ظ����ֶ��ʱ���ǳ�����ʵ�֣����ԡ�һ�ж����С���

��������Ҫ֮ǰ�����Ϣ��ʱ�򣬱��� ResNets �еĲв���㣬��ǰ��Ľ����Ҫ��֮ǰ���еĽ�������ںϣ�һ��ʹ�� ModuleList/ModuleDict �ȽϷ��㡣


## ����ģ�Ϳ���ٴ��������

������U-NetΪ�����������������ʹ��ģ�Ϳ�

![](https://datawhalechina.github.io/thorough-pytorch/_images/5.2.1unet.png)


���U-Net��ģ�Ϳ���Ҫ�����¼������֣�

1��ÿ���ӿ��ڲ������ξ����Double Convolution��

2�����ģ�Ϳ�֮����²������ӣ������ػ���Max pooling��

3���Ҳ�ģ�Ϳ�֮����ϲ������ӣ�Up sampling��

4�������Ĵ���


### U-Netģ�Ϳ�ʵ��

��ʹ��PyTorchʵ��U-Netģ��ʱ�����ǲ��ذ�ÿһ�㰴��������ʽд��������̫�鷳�Ҳ��˶���һ�ֱȽϺõķ������ȶ����ģ�Ϳ飬�ٶ���ģ�Ϳ�֮�������˳��ͼ��㷽ʽ���ͺñ�װ�����һ����������װ���һЩ�����Ĳ�����֮��������Щ���Ը��õĲ����õ�����װ���塣

����Ļ���������Ӧ��һ�ڷ������ĸ�ģ�Ϳ飬���ݹ������ǽ�������Ϊ��DoubleConv, Down, Up, OutConv���������U-Net��ģ�Ϳ��PyTorch ʵ�֣�


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


### ����ģ�Ϳ���װU-Net

ʹ��д�õ�ģ�Ϳ飬���Էǳ��������װU-Netģ�͡����Կ�����ͨ��ģ�Ϳ�ķ�ʽʵ���˴��븴�ã�����ģ�ͽṹ��������Ĵ������������Լ��٣�����ɶ���Ҳ�õ���������


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


## PyTorch�޸�ģ��

�����Լ�����PyTorchģ���⣬������һ��Ӧ�ó����������Ѿ���һ���ֳɵ�ģ�ͣ�����ģ���еĲ��ֽṹ���������ǵ�Ҫ��Ϊ��ʹ��ģ�ͣ�������Ҫ��ģ�ͽṹ���б�Ҫ���޸ġ��������ѧϰ�ķ�չ��PyTorchԽ��Խ�㷺��ʹ�ã���Խ��Խ��Ŀ�Դģ�Ϳ��Թ�����ʹ�ã��ܶ�ʱ������Ҳ���ش�ͷ��ʼ����ģ�͡���ˣ���������޸�PyTorchģ�;��Ե���Ϊ��Ҫ��



### �޸�ģ�Ͳ�

�޸ķ�������


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

����Ĳ����൱�ڽ�ģ�ͣ�net���������Ϊ��fc���Ĳ��滻��������Ϊ��classifier���Ľṹ���ýṹ�������Լ�����ġ�����ʹ���˵�һ�ڽ��ܵ�Sequential+OrderedDict��ģ�Ͷ��巽ʽ�����ˣ����Ǿ������ģ�͵��޸ģ����ڵ�ģ�;Ϳ���ȥ��10���������ˡ�


### ����ⲿ����

ʵ�ַ�������


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


### ��Ӷ������

ʵ�ַ�������


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


## PyTorchģ�ͱ������ȡ


### ģ�ʹ洢��ʽ

PyTorch�洢ģ����Ҫ����pkl��pt��pth���ָ�ʽ��


### ģ�ʹ洢����

һ��PyTorchģ����Ҫ�����������֣�ģ�ͽṹ��Ȩ�ء�����ģ���Ǽ̳�nn.Module���࣬Ȩ�ص����ݽṹ��һ���ֵ䣨key�ǲ�����value��Ȩ�����������洢Ҳ�ɴ˷�Ϊ������ʽ���洢����ģ�ͣ������ṹ��Ȩ�أ�����ֻ�洢ģ��Ȩ�ء�


```python
from torchvision import models
model = models.resnet152(pretrained=True)

# ��������ģ��
torch.save(model, save_dir)
# ����ģ��Ȩ��
torch.save(model.state_dict, save_dir)
```

����PyTorch���ԣ�pt, pth��pkl **�������ݸ�ʽ��֧��ģ��Ȩ�غ�����ģ�͵Ĵ洢** �����ʹ����û�в��
