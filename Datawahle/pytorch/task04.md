# pytorch����ѵ������

## �Զ�����ʧ����

PyTorch��torch.nnģ��Ϊ�����ṩ����ೣ�õ���ʧ���������磺MSELoss��L1Loss��BCELoss...... �����������ѧϰ�ķ�չ��������Խ��Խ��ķǹٷ��ṩ��Loss������DiceLoss��HuberLoss��SobolevLoss...... ��ЩLoss Functionר�����һЩ��ͨ�õ�ģ�ͣ�PyTorch���ܽ�����ȫ����ӵ�����ȥ�������Щ��ʧ������ʵ������Ҫ����ͨ���Զ�����ʧ������ʵ�֡�


### �Ժ�����ʽ����

������ʾ����

```python
def my_loss(output, target):
    loss = torch.mean((output - target)**2)
    return loss
```


### ���෽ʽ����

��Ȼ�Ժ�������ķ�ʽ�ܼ򵥣��������෽ʽ������ӳ��ã������෽ʽ������ʧ����ʱ�����������ÿһ����ʧ�����ļ̳й�ϵ���ǾͿ��Է��� `Loss`�������ּ̳��� `_loss` ���ּ̳��� `_WeightedLoss`, �� `_WeightedLoss`�̳��� `_loss`��`_loss`�̳���  **nn.Module** �����ǿ��Խ��䵱���������һ�����Դ���ͬ���أ����ǵ���ʧ���������Ҫ�̳���**nn.Module**��

Diceʵ�ִ�������


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

# ʹ�÷���  
criterion = DiceLoss()
loss = criterion(input,targets)
```

*ע������������Ĵ����м̳���nn.module�࣬��������Ҫʹ��һЩ���֮��Ĺ���ʱ����ҪһЩ������������ʵ�֣�����ʵ�ַ�ʽ����numpy/scipy*

## ��̬����ѧϰ��

��ʵ��ѧϰӦ�������ǿ��ܻ������������µ����

![](https://img-blog.csdnimg.cn/1f31881839084ba9a4ee62cbbe3275f4.png)

ѧϰ�������ù�С���Ἣ�󽵵������ٶȣ�����ѵ��ʱ�䣻ѧϰ��̫�󣬿��ܵ��²��������Ž����������񵴣���ô���Ǿ���Ҫѧϰ���ܹ���̬���е������������ǵ�ѵ���ɱ���


### ʹ�ùٷ�scheduler

��ѵ��������Ĺ����У�ѧϰ��������Ҫ�ĳ�����֮һ����Ϊ��ǰ��Ϊ���е����ѧϰ��ܣ�PyTorch�Ѿ��� `torch.optim.lr_scheduler`Ϊ���Ƿ�װ����һЩ��̬����ѧϰ�ʵķ���������ʹ�ã��������г�����Щscheduler��


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

���������ڶ�optimizer�е�ѧϰ�ʽ��и��¡����������µķ�����scheduler.step()��

ͨ�����ԣ���һ��batch_size���Ƚ���optimizer.step()���Ȩ�ز����ĸ��¹��̣�Ȼ���ٽ���scheduler.step()��ɶ�ѧϰ�ʲ����ĸ��¹��̡�

ʵ�ִ�������


```python
# ѡ��һ���Ż���
optimizer = torch.optim.Adam(...) 
# ѡ�������ᵽ��һ�ֻ���ֶ�̬����ѧϰ�ʵķ���
scheduler1 = torch.optim.lr_scheduler.... 
scheduler2 = torch.optim.lr_scheduler....
...
schedulern = torch.optim.lr_scheduler....
# ����ѵ��
for epoch in range(100):
    train(...)
    validate(...)
    optimizer.step()
    # ��Ҫ���Ż�����������֮���ٶ�̬����ѧϰ��
	scheduler1.step() 
	...
    schedulern.step()
```


 **ע** ��

������ʹ�ùٷ������� `torch.optim.lr_scheduler`ʱ����Ҫ�� `scheduler.step()`���� `optimizer.step()`�������ʹ�á�


### �Զ���scheduler

��ȻPyTorch�ٷ��������ṩ������API��������ʵ����Ҳ�п���������Ҫ�����Լ�����ѧϰ�ʵ������Ե�����������ǵķ������Զ��庯�� `adjust_learning_rate`���ı� `param_group`�� `lr`��ֵ

ʵ������


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


## ģ��΢��-torchvision

����˵�������������ҵ�һ��ͬ��ı���ѵ���õ�ģ�ͣ��ѱ����ֳɵ�ѵ�����˵�ģ���ù����������Լ������ݣ�ͨ��ѵ������һ�²����� ��PyTorch���ṩ�����Ԥѵ���õ�����ģ�ͣ�VGG��ResNetϵ�У�mobilenetϵ��......������Щģ�Ͷ���PyTorch�ٷ�����Ӧ�Ĵ������ݼ�ѵ���õġ�ѧϰ��ν���ģ��΢�������Է������ǿ���ʹ��Ԥѵ��ģ������Լ�������


### ģ��΢��������


1. ��Դ���ݼ�(��ImageNet���ݼ�)��Ԥѵ��һ��������ģ�ͣ���Դģ�͡�
2. ����һ���µ�������ģ�ͣ���Ŀ��ģ�͡�������?Դģ���ϳ�?������������ģ����Ƽ�����������Ǽ�����Щģ�Ͳ�������?Դ���ݼ���ѧϰ����֪ʶ������Щ֪ʶͬ��������Ŀ�����ݼ������ǻ�����Դģ�͵�������Դ���ݼ��ı�ǩ������أ������Ŀ��ģ���в�����á�
3. ΪĿ��ģ�����һ�����?СΪ?�����ݼ�������������㣬�������ʼ���ò��ģ�Ͳ�����
4. ��Ŀ�����ݼ���ѵ��Ŀ��ģ�͡����ǽ���ͷѵ������㣬�������Ĳ������ǻ���Դģ�͵Ĳ���΢���õ��ġ�


### ʹ������ģ�ͽṹ

ͨ�� `True`���� `False`�������Ƿ�ʹ��Ԥѵ���õ�Ȩ�أ���Ĭ��״̬�� `pretrained=False`����ζ�����ǲ�ʹ��Ԥѵ���õ���Ȩ�أ��� `pretrained=True`����ζ�����ǽ�ʹ����һЩ���ݼ���Ԥѵ���õ���Ȩ�ء�


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


### ѵ���ض���

��Ĭ������£����������� `<span class="pre">.requires_grad</span><span>?</span><span class="pre">=</span><span>?</span><span class="pre">True</span>`��������Ǵ�ͷ��ʼѵ����΢������Ҫע��������������������ȡ��������ֻ��Ϊ�³�ʼ���Ĳ�����ݶȣ��������������иı䡣�����Ǿ���Ҫͨ������ `<span class="pre">requires_grad</span><span>?</span><span class="pre">=</span><span>?</span><span class="pre">False</span>`�����Ჿ�ֲ㡣��PyTorch�ٷ����ṩ������һ�����̡�


```python
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
```


## ģ��΢�� - timm


Pytorch Image Models (timm) �����˳��õ�models, layers, utilities, optimizers, schedulers, data-loaders / augmentations, and reference training / validation scripts������Ŀ���ǽ�����SOTAģ��������һ�𣬲���������ImageNetѵ�������������

���ߣ�Ross Wightman�����Լ��ô��¸绪��


### ʹ�ú��޸�Ԥѵ��ģ��

�ڵõ�������Ҫʹ�õ�Ԥѵ��ģ�ͺ����ǿ���ͨ�� `<span class="pre">timm.create_model()</span>`�ķ���������ģ�͵Ĵ��������ǿ���ͨ��������� `<span class="pre">pretrained=True</span>`����ʹ��Ԥѵ��ģ�͡�ͬ���ģ�����Ҳ����ʹ�ø�torchvision�����ģ��һ���ķ����鿴ģ�͵Ĳ���������


```python
import timm
import torch

model = timm.create_model('resnet34',pretrained=True)
x = torch.randn(1,3,224,224)
output = model(x)
output.shape
```


## �뾫��ѵ��


�����ᵽPyTorchʱ���ܻ��뵽Ҫ��Ӳ���豸GPU��֧�֣�Ҳ���ǡ�������GPU��������Ҫ��Ϊ�����֣��������Դ棬ǰ�߾������Կ�������ٶȣ�������������Կ�����ͬʱ��������������ڼ��㡣�ڿ���ʹ�õ��Դ�����һ��������£�ÿ��ѵ���ܹ����ص����ݸ��ࣨҲ����batch size���󣩣���Ҳ�������ѵ��Ч�ʡ����⣬��ʱ�����ݱ���Ҳ�Ƚϴ󣨱���3Dͼ����Ƶ�ȣ����Դ��С������¿�������batch sizeΪ1��������޷�ʵ�֡���ˣ�����ʹ���Դ�Ҳ���Ե�ʮ����Ҫ��

���ǹ۲�PyTorchĬ�ϵĸ������洢��ʽ�õ���torch.float32��С�����λ�������Ȼ�ܱ�֤���ݵľ�ȷ�ԣ����������������ʵ������Ҫ��ô��ȷ��ֻ����һ�����ϢҲ����Ӱ������Ҳ����ʹ��torch.float16��ʽ��������λ����һ�룬��˱���Ϊ���뾫��"


### �뾫��ѵ��������

��PyTorch��ʹ��autocast���ð뾫��ѵ����ͬʱ��Ҫ�����������������ã�


```
from torch.cuda.amp import autocast
```

��ģ�Ͷ����У�ʹ��python��װ������������autocastװ��ģ���е�forward����

```
@autocast()   
def forward(self, x):
    ...
    return x
```

��ѵ�������У�ֻ���ڽ���������ģ�ͼ���֮��Ĳ��ַ��롰with autocast():�����ɣ�

```
for x in train_loader:
	x = x.cuda()
	with autocast():
        output = model(x)
        ...
```


**ע�⣺**

�뾫��ѵ����Ҫ���������ݱ����size�Ƚϴ󣨱���˵3Dͼ����Ƶ�ȣ��������ݱ����size������ʱ��������д����MNIST���ݼ���ͼƬ�ߴ�ֻ��28*28����ʹ�ð뾫��ѵ������ܲ������������������


## ������ǿ-imgaug

���ѧϰ����Ҫ�������ݡ�������Ҫ�������ݲ��ܱ���ģ�͵Ĺ�����ϡ�������������ೡ���޷���ô������ݣ�����ҽѧͼ�������������ǿ�����Ĵ�����Ϊ�˽��������⣬�������������������Ľ��������������ǿһ�׼����������ѵ�����ݼ��Ĵ�С���������Ա����ǿ���ʹ���������������õ����ѧϰģ�͡� �ڼ����Ӿ�����������ǿͼ��������ס���ʹ����������ü�ͼ���һ���֣�ģ���Կ��Զ�ͼ����з��࣬������ǿ��һϵ�м���Ч�ķ����ɹ�ѡ����һЩ����ѧϰ�������м����Ӿ������������ǿ



### imgaug���

`imgaug`�Ǽ�����Ӿ������г��õ�һ��������ǿ�İ�������� `torchvision.transforms`�����ṩ�˸����������ǿ����������ڸ��־����У����ǹ㷺ʹ�� `imgaug`�������ݽ�����ǿ������


### imgaug��ʹ��

imgaug�����ṩ��ͼ����ǿ��һЩ���������ǲ�δ�ṩͼ���IO���������������Ҫʹ��һЩ������ͼ����е��룬����ʹ��imageio���ж��룬���ʹ�õ���opencv�����ļ���ȡ��ʱ����Ҫ�����ֶ��ı�ͨ��������ȡ��BGRͼ��ת��ΪRGBͼ�񡣳������⣬��������PIL.Image���ж�ȡʱ����Ϊ��ȡ��ͼƬû��shape�����ԣ�����������Ҫ����ȡ����imgת��Ϊnp.array()����ʽ�ٽ��д�����˹ٷ���������Ҳ��ʹ��imageio����ͼƬ��ȡ��

ʹ�÷�������


```python 
from imgaug import augmenters as iaa

# �������������
ia.seed(4)

# ʵ��������
rotate = iaa.Affine(rotate=(-4,45))
img_aug = rotate(image=img)
ia.imshow(img_aug)
```

![img](https://datawhalechina.github.io/thorough-pytorch/_images/Lenna_original.png) to ![](https://datawhalechina.github.io/thorough-pytorch/_images/rotate.png)

���Ƕ�һ��ͼƬ����һ�ֲ�����ʽ����ʵ������£����ǿ��ܶ�һ��ͼƬ������������ǿ������������£����Ǿ���Ҫ���� `imgaug.augmenters.Sequential()`����������������ǿ��pipline���÷����� `torchvison.transforms.Compose()`�����ơ�


```python 
iaa.Sequential(children=None, # Augmenter����
               random_order=False, # �Ƿ��ÿ��batchʹ�ò�ͬ˳���Augmenter list
               name=None,
               deterministic=False,
               random_state=None)
# ������������
aug_seq = iaa.Sequential([
    iaa.Affine(rotate=(-25,25)),
    iaa.AdditiveGaussianNoise(scale=(10,60)),
    iaa.Crop(percent=(0,0.2))
])
# ��ͼƬ���д���image������ʡ�ԣ�Ҳ����д��images
image_aug = aug_seq(image=img)
ia.imshow(image_aug)
```

![](https://datawhalechina.github.io/thorough-pytorch/_images/aug_seq.png)

��һ���ε�ͼƬ���д���ʱ������ֻ��Ҫ���������ͼƬ����һ�� `list`�У�����image��Ϊimage���ɽ���������ǿ����������ʵ�ʲ������£�


```python
images = [img,img,img,img,]
images_aug = rotate(images=images)
ia.imshow(np.hstack(images_aug))
```

imgaug�����������������ǿ�Ŀ⣬��һ��������˼�����ԣ����������ǿ���ͨ�� `imgaug.augmenters.Sometimes()`��batch�е�һ����ͼƬӦ��һ����Augmenters,ʣ�µ�ͼƬӦ�������Augmenters��


```python
iaa.Sometimes(p=0.5,  # �����ֱ���
              then_list=None,  # Augmenter���ϡ�p���ʵ�ͼƬ���б任��Augmenters��
              else_list=None,  #1-p���ʵ�ͼƬ�ᱻ���б任��Augmenters��ע��任��ͼƬӦ�õ�Augmenterֻ����then_list����else_list�е�һ����
              name=None,
              deterministic=False,
              random_state=None)
```


## ʹ��argparse���е���

�����ѧϰ��ʱ�����������޸ĺͱ����Ƿǳ���Ҫ��һ���������ǵ������ڷ������������ǵ�ģ��ʱ����θ�������޸ĳ�������������Ҫ���ǵ�һ�����⡣��ʱ��Ҫ����һ������ߺ������Խ�����������������в����ٴ���ģ�͵ĳ������иö�á�������û��������һ�ַ����أ����ǿ϶��ģ�������� Python ��׼���һ���֣�Argparse��


�ܵ���˵�����ǿ��Խ�argparse��ʹ�ù���Ϊ�����������衣

* ���� `<span class="pre">ArgumentParser()</span>`����
* ���� `<span class="pre">add_argument()</span>`������Ӳ���
* ʹ�� `<span class="pre">parse_args()</span>`��������


```python
# demo.py
import argparse

# ����ArgumentParser()����
parser = argparse.ArgumentParser()

# ��Ӳ���
parser.add_argument('-o', '--output', action='store_true', 
    help="shows output")
# action = `store_true` �Ὣoutput������¼ΪTrue
# type �涨�˲����ĸ�ʽ
# default �涨��Ĭ��ֵ
parser.add_argument('--lr', type=float, default=3e-5, help='select the learning rate, default=1e-3') 

parser.add_argument('--batch_size', type=int, required=True, help='input batch size')  
# ʹ��parse_args()��������
args = parser.parse_args()

if args.output:
    print("This is some output")
    print(f"learning rate:{args.lr} ")
```

Datawhale������һ�ָ�Ч�Ĳ�������������Ҳ���ǽ������ļ���ִ�д�����뿪����Ҫʱ���������ļ�


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
