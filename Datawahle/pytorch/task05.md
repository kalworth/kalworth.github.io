# Pytorch���ӻ�

## ���ӻ�����ṹ

�������ѧϰ�Ĳ��Ϸ�չ

����������������ĵķ�չ������ĽṹԽ��Խ���ӣ�����Ҳ����ȷ��ÿһ�������ṹ������ṹ�Լ���������Ϣ�������������Ǻ����ڶ�ʱ�������debug���������һ�������������ӻ�����ṹ�Ĺ�����ʮ���б�Ҫ�ġ����ƵĹ�������һ�����ѧϰ��Keras�п��Ե���һ������ `model.summary()`��API���ܷ����ʵ�֣����ú�ͻ���ʾ���ǵ�ģ�Ͳ����������С�������С��ģ�͵���������ȣ�������PyTorch��û������һ�ֱ����Ĺ��߰������ǿ��ӻ����ǵ�ģ�ͽṹ��

Ϊ�˽��������⣬���ǿ�����torchinfo����

### ʹ��print��������ӡ����Ļ�����Ϣ

```python
import torchvision.models as models
model = models.resnet18()
```

ͨ����������������Ǿ͵õ�resnet18��ģ�ͽṹ����ѧϰtorchinfo֮ǰ���������ȿ���ֱ��print(model)�Ľ����

```python
ResNet(
  (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
  (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (relu): ReLU(inplace=True)
  (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
  (layer1): Sequential(
    (0): Bottleneck(
      (conv1): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (downsample): Sequential(
        (0): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
   ... ...
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
  (fc): Linear(in_features=2048, out_features=1000, bias=True)
)
```

���ǿ��Է��ֵ�����print(model)��ֻ�ܵó�������������Ϣ���Ȳ�����ʾ��ÿһ���shape��Ҳ������ʾ��Ӧ�������Ĵ�С��Ϊ�˽����Щ���⣬���Ǿ���Ҫ���ܳ����ǽ�������˹� `torchinfo`��

### ʹ��torchinfo���ӻ�����ṹ

trochinfo��ʹ��Ҳ��ʮ�ּ򵥣�����ֻ��Ҫʹ�� `torchinfo.summary()`�����ˣ�����Ĳ����ֱ���model��input_size[batch_size,channel,h,w]������������Բο�[documentation](https://github.com/TylerYep/torchinfo#documentation)������������һ��ͨ��һ��ʵ������ѧϰ��

```python
import torchvision.models as models
from torchinfo import summary
resnet18 = models.resnet18() # ʵ����ģ��
summary(resnet18, (1, 3, 224, 224)) # 1��batch_size 3:ͼƬ��ͨ���� 224: ͼƬ�ĸ߿�
```

torchinfo���ո�Ϊ�ṹ������ʽ����˸�ģ��

```shell
=========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
=========================================================================================
ResNet                                   --                        --
����Conv2d: 1-1                            [1, 64, 112, 112]         9,408
����BatchNorm2d: 1-2                       [1, 64, 112, 112]         128
����ReLU: 1-3                              [1, 64, 112, 112]         --
����MaxPool2d: 1-4                         [1, 64, 56, 56]           --
����Sequential: 1-5                        [1, 64, 56, 56]           --
��    ����BasicBlock: 2-1                   [1, 64, 56, 56]           --
��    ��    ����Conv2d: 3-1                  [1, 64, 56, 56]           36,864
��    ��    ����BatchNorm2d: 3-2             [1, 64, 56, 56]           128
��    ��    ����ReLU: 3-3                    [1, 64, 56, 56]           --
��    ��    ����Conv2d: 3-4                  [1, 64, 56, 56]           36,864
��    ��    ����BatchNorm2d: 3-5             [1, 64, 56, 56]           128
��    ��    ����ReLU: 3-6                    [1, 64, 56, 56]           --
��    ����BasicBlock: 2-2                   [1, 64, 56, 56]           --
��    ��    ����Conv2d: 3-7                  [1, 64, 56, 56]           36,864
��    ��    ����BatchNorm2d: 3-8             [1, 64, 56, 56]           128
��    ��    ����ReLU: 3-9                    [1, 64, 56, 56]           --
��    ��    ����Conv2d: 3-10                 [1, 64, 56, 56]           36,864
��    ��    ����BatchNorm2d: 3-11            [1, 64, 56, 56]           128
��    ��    ����ReLU: 3-12                   [1, 64, 56, 56]           --
����Sequential: 1-6                        [1, 128, 28, 28]          --
��    ����BasicBlock: 2-3                   [1, 128, 28, 28]          --
��    ��    ����Conv2d: 3-13                 [1, 128, 28, 28]          73,728
��    ��    ����BatchNorm2d: 3-14            [1, 128, 28, 28]          256
��    ��    ����ReLU: 3-15                   [1, 128, 28, 28]          --
��    ��    ����Conv2d: 3-16                 [1, 128, 28, 28]          147,456
��    ��    ����BatchNorm2d: 3-17            [1, 128, 28, 28]          256
��    ��    ����Sequential: 3-18             [1, 128, 28, 28]          8,448
��    ��    ����ReLU: 3-19                   [1, 128, 28, 28]          --
��    ����BasicBlock: 2-4                   [1, 128, 28, 28]          --
��    ��    ����Conv2d: 3-20                 [1, 128, 28, 28]          147,456
��    ��    ����BatchNorm2d: 3-21            [1, 128, 28, 28]          256
��    ��    ����ReLU: 3-22                   [1, 128, 28, 28]          --
��    ��    ����Conv2d: 3-23                 [1, 128, 28, 28]          147,456
��    ��    ����BatchNorm2d: 3-24            [1, 128, 28, 28]          256
��    ��    ����ReLU: 3-25                   [1, 128, 28, 28]          --
����Sequential: 1-7                        [1, 256, 14, 14]          --
��    ����BasicBlock: 2-5                   [1, 256, 14, 14]          --
��    ��    ����Conv2d: 3-26                 [1, 256, 14, 14]          294,912
��    ��    ����BatchNorm2d: 3-27            [1, 256, 14, 14]          512
��    ��    ����ReLU: 3-28                   [1, 256, 14, 14]          --
��    ��    ����Conv2d: 3-29                 [1, 256, 14, 14]          589,824
��    ��    ����BatchNorm2d: 3-30            [1, 256, 14, 14]          512
��    ��    ����Sequential: 3-31             [1, 256, 14, 14]          33,280
��    ��    ����ReLU: 3-32                   [1, 256, 14, 14]          --
��    ����BasicBlock: 2-6                   [1, 256, 14, 14]          --
��    ��    ����Conv2d: 3-33                 [1, 256, 14, 14]          589,824
��    ��    ����BatchNorm2d: 3-34            [1, 256, 14, 14]          512
��    ��    ����ReLU: 3-35                   [1, 256, 14, 14]          --
��    ��    ����Conv2d: 3-36                 [1, 256, 14, 14]          589,824
��    ��    ����BatchNorm2d: 3-37            [1, 256, 14, 14]          512
��    ��    ����ReLU: 3-38                   [1, 256, 14, 14]          --
����Sequential: 1-8                        [1, 512, 7, 7]            --
��    ����BasicBlock: 2-7                   [1, 512, 7, 7]            --
��    ��    ����Conv2d: 3-39                 [1, 512, 7, 7]            1,179,648
��    ��    ����BatchNorm2d: 3-40            [1, 512, 7, 7]            1,024
��    ��    ����ReLU: 3-41                   [1, 512, 7, 7]            --
��    ��    ����Conv2d: 3-42                 [1, 512, 7, 7]            2,359,296
��    ��    ����BatchNorm2d: 3-43            [1, 512, 7, 7]            1,024
��    ��    ����Sequential: 3-44             [1, 512, 7, 7]            132,096
��    ��    ����ReLU: 3-45                   [1, 512, 7, 7]            --
��    ����BasicBlock: 2-8                   [1, 512, 7, 7]            --
��    ��    ����Conv2d: 3-46                 [1, 512, 7, 7]            2,359,296
��    ��    ����BatchNorm2d: 3-47            [1, 512, 7, 7]            1,024
��    ��    ����ReLU: 3-48                   [1, 512, 7, 7]            --
��    ��    ����Conv2d: 3-49                 [1, 512, 7, 7]            2,359,296
��    ��    ����BatchNorm2d: 3-50            [1, 512, 7, 7]            1,024
��    ��    ����ReLU: 3-51                   [1, 512, 7, 7]            --
����AdaptiveAvgPool2d: 1-9                 [1, 512, 1, 1]            --
����Linear: 1-10                           [1, 1000]                 513,000
=========================================================================================
Total params: 11,689,512
Trainable params: 11,689,512
Non-trainable params: 0
Total mult-adds (G): 1.81
=========================================================================================
Input size (MB): 0.60
Forward/backward pass size (MB): 39.75
Params size (MB): 46.76
Estimated Total Size (MB): 87.11
=========================================================================================
```

torchinfo�ṩ�˸�����ϸ����Ϣ������ģ����Ϣ��ÿһ������͡����shape�Ͳ���������ģ������Ĳ�������ģ�ʹ�С��һ��ǰ����߷��򴫲���Ҫ���ڴ��С��

 **ע��** ��

> ����ʹ�õ���colab����jupyter notebookʱ����Ҫʵ�ָ÷�����`summary()`һ���Ǹõ�Ԫ����notebook�е�cell���ķ���ֵ���������Ǿ���Ҫʹ�� `print(summary(...))`�����ӻ���

## CNN���ӻ�

��������磨CNN�������ѧϰ�зǳ���Ҫ��ģ�ͽṹ�����㷺������ͼ���������������ģ�ͱ��֣��ƶ��˼�����Ӿ��ķ�չ�ͽ�������CNN��һ�����ں�ģ�͡������ǲ���֪��CNN����λ�ýϺñ��ֵģ��ɴ˴��������ѧϰ�Ŀɽ��������⡣��������CNN�����ķ�ʽ�����ǲ����ܹ���������õĽ��������ģ�͵�³���ԣ����һ���������ԵظĽ�CNN�Ľṹ�Ի�ý�һ����Ч��������

### CNN����˿��ӻ�

��PyTorch�п��ӻ������Ҳ�ǳ����㣬���������ض���ľ���˼��ض����ģ��Ȩ�أ����ӻ�����˾͵ȼ��ڿ��ӻ���Ӧ��Ȩ�ؾ������������PyTorch�п��ӻ�����˵�ʵ�ַ�������torchvision�Դ���VGG11ģ��Ϊ����

```python
{'0': Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
 '1': ReLU(inplace=True),
 '2': MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
 '3': Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
 '4': ReLU(inplace=True),
 '5': MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
 '6': Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
 '7': ReLU(inplace=True),
 '8': Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
 '9': ReLU(inplace=True),
 '10': MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
 '11': Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
 '12': ReLU(inplace=True),
 '13': Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
 '14': ReLU(inplace=True),
 '15': MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
 '16': Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
 '17': ReLU(inplace=True),
 '18': Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
 '19': ReLU(inplace=True),
 '20': MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)}
```

����˶�Ӧ��ӦΪ����㣨Conv2d���������Եڡ�3����Ϊ�������ӻ���Ӧ�Ĳ�����

```python
conv1 = dict(model.features.named_children())['3']
kernel_set = conv1.weight.detach()
num = len(conv1.weight.detach())
print(kernel_set.shape)
for i in range(0,num):
    i_kernel = kernel_set[i]
    plt.figure(figsize=(20, 17))
    if (len(i_kernel)) > 1:
        for idx, filer in enumerate(i_kernel):
            plt.subplot(9, 9, idx+1) 
            plt.axis('off')
            plt.imshow(filer[ :, :].detach(),cmap='bwr')
```

���ڵڡ�3���������ͼ��64ά��Ϊ128ά����˹���128*64������ˣ����в��־���˿��ӻ�Ч������ͼ��ʾ��

![](https://datawhalechina.github.io/thorough-pytorch/_images/kernel_vis.png)

### CNN����ͼ���ӻ�����

���������Ӧ�������ԭʼͼ�񾭹�ÿ�ξ����õ������ݳ�Ϊ����ͼ�����ӻ��������Ϊ�˿�ģ����ȡ��Щ���������ӻ�����ͼ����Ϊ�˿�ģ����ȡ����������ʲô���ӵġ�

��ȡ����ͼ�ķ����кܶ��֣����Դ����뿪ʼ�������ǰ�򴫲���ֱ����Ҫ������ͼ�����䷵�ء��������ַ������У�������Щ�鷳�ˡ���PyTorch�У��ṩ��һ��ר�õĽӿ�ʹ��������ǰ�򴫲��������ܹ���ȡ������ͼ������ӿڵ����Ʒǳ����󣬽���hook���������������ĳ���������ͨ��������ǰ����������ĳһ������Ԥ��������һ�����ӣ����ݴ����������ϻ�������������һ������ӣ���ȡ���ӵ���Ϣ������һ�������ͼ������ʵ�����£�

```python
class Hook(object):
    def __init__(self):
        self.module_name = []
        self.features_in_hook = []
        self.features_out_hook = []

    def __call__(self,module, fea_in, fea_out):
        print("hooker working", self)
        self.module_name.append(module.__class__)
        self.features_in_hook.append(fea_in)
        self.features_out_hook.append(fea_out)
        return None
  

def plot_feature(model, idx, inputs):
    hh = Hook()
    model.features[idx].register_forward_hook(hh)
  
    # forward_model(model,False)
    model.eval()
    _ = model(inputs)
    print(hh.module_name)
    print((hh.features_in_hook[0][0].shape))
    print((hh.features_out_hook[0].shape))
  
    out1 = hh.features_out_hook[0]

    total_ft  = out1.shape[1]
    first_item = out1[0].cpu().clone()  

    plt.figure(figsize=(20, 17))
  

    for ftidx in range(total_ft):
        if ftidx > 99:
            break
        ft = first_item[ftidx]
        plt.subplot(10, 10, ftidx+1) 
  
        plt.axis('off')
        #plt.imshow(ft[ :, :].detach(),cmap='gray')
        plt.imshow(ft[ :, :].detach())
```

������������ʵ����һ��hook�֮࣬����plot_feature�����У�����hook��Ķ���ע�ᵽҪ���п��ӻ��������ĳ���С�model�ڽ���ǰ�򴫲���ʱ������hook��__call__����������Ҳ����������洢�˵�ǰ������������������features_out_hook ��һ��list��ÿ��ǰ�򴫲�һ�Σ����ǵ���һ�Σ�Ҳ����features_out_hook ���Ȼ�����1��

### CNN class activation map���ӻ�����

class activation map ��CAM�����������ж���Щ������ģ����˵����Ҫ�ģ���CNN���ӻ��ĳ����£����ж�ͼ������Щ���ص��Ԥ��������Ҫ�ġ�����ȷ����Ҫ�����ص㣬����Ҳ�����Ҫ������ݶȸ���Ȥ�������CAM�Ļ�����Ҳ��һ���Ľ��õ���Grad-CAM���Լ������֣���CAM��Grad-CAM��ʾ������ͼ��ʾ��CNN class activation map���ӻ�����

![](https://datawhalechina.github.io/thorough-pytorch/_images/cam.png)

��ȿ��ӻ����������ӻ�����ͼ��CAMϵ�п��ӻ���Ϊֱ�ۣ��ܹ�һĿ��Ȼ��ȷ����Ҫ���򣬽������пɽ����Է�����ģ���Ż��Ľ���CAMϵ�в�����ʵ�ֿ���ͨ����Դ���߰�pytorch-grad-cam��ʵ�֡�

```python
import torch
from torchvision.models import vgg11,resnet18,resnet101,resnext101_32x8d
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

model = vgg11(pretrained=True)
img_path = './dog.png'
# resize������Ϊ�˺ʹ���������ѵ��ͼƬ��Сһ��
img = Image.open(img_path).resize((224,224))
# ��Ҫ��ԭʼͼƬתΪnp.float32��ʽ������0-1֮�� 
rgb_img = np.float32(img)/255
plt.imshow(img)
```

```python
from pytorch_grad_cam import GradCAM,ScoreCAM,GradCAMPlusPlus,AblationCAM,XGradCAM,EigenCAM,FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

target_layers = [model.features[-1]]
# ѡȡ���ʵ��༤��ͼ������ScoreCAM��AblationCAM��Ҫbatch_size
cam = GradCAM(model=model,target_layers=target_layers)
targets = [ClassifierOutputTarget(preds)]   
# �Ϸ�preds��Ҫ�趨������ImageNet��1000�࣬���������Ϊ200
grayscale_cam = cam(input_tensor=img_tensor, targets=targets)
grayscale_cam = grayscale_cam[0, :]
cam_img = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
print(type(cam_img))
Image.fromarray(cam_img)
```

### ʹ��FlashTorch����ʵ��CNN���ӻ�

* ���ӻ��ݶ�

```python
# Download example images
# !mkdir -p images
# !wget -nv \
#    https://github.com/MisaOgura/flashtorch/raw/master/examples/images/great_grey_owl.jpg \
#    https://github.com/MisaOgura/flashtorch/raw/master/examples/images/peacock.jpg   \
#    https://github.com/MisaOgura/flashtorch/raw/master/examples/images/toucan.jpg    \
#    -P /content/images

import matplotlib.pyplot as plt
import torchvision.models as models
from flashtorch.utils import apply_transforms, load_image
from flashtorch.saliency import Backprop

model = models.alexnet(pretrained=True)
backprop = Backprop(model)

image = load_image('/content/images/great_grey_owl.jpg')
owl = apply_transforms(image)

target_class = 24
backprop.visualize(owl, target_class, guided=True, use_gpu=True)
```

![](https://datawhalechina.github.io/thorough-pytorch/_images/ft_gradient.png)

* ���ӻ������

```python
import torchvision.models as models
from flashtorch.activmax import GradientAscent

model = models.vgg16(pretrained=True)
g_ascent = GradientAscent(model.features)

# specify layer and filter info
conv5_1 = model.features[24]
conv5_1_filters = [45, 271, 363, 489]

g_ascent.visualize(conv5_1, conv5_1_filters, title="VGG16: conv5_1")
```

![](https://datawhalechina.github.io/thorough-pytorch/_images/ft_activate.png)

## ʹ��TensorBoard���ӻ�ѵ������

ѵ�����̵Ŀ��ӻ������ѧϰģ��ѵ���а�������Ҫ�Ľ�ɫ��ѧϰ�Ĺ�����һ���Ż��Ĺ��̣�������Ҫ�ҵ����ŵĵ���Ϊѵ�����̵�������һ����˵�����ǻ���ѵ��������ʧ��������֤������ʧ����������������ʧ������������ȷ��ѵ�����յ㣬�ҵ���Ӧ��ģ�����ڲ��ԡ���ô���˼�¼ѵ����ÿ��epoch��lossֵ���ܷ�ʵʱ�۲���ʧ�������ߵı仯����ʱ��׽ģ�͵ı仯�أ�

���⣬����Ҳϣ�����ӻ��������ݣ����������ݣ�������ͼƬ����ģ�ͽṹ�������ֲ��ȣ���Щ����������debug�в���������Դ�ǳ���Ҫ�������������ݺ�����������Ƿ�һ�£���

TensorBoard��Ϊһ����ӻ������ܹ����������ᵽ�ĸ�������TensorBoard��TensorFlow�Ŷӿ����������TensorFlow���ʹ�ã������㷺Ӧ���ڸ������ѧϰ��ܵĿ��ӻ�������

### TensorBoard���ӻ��Ļ����߼�

���ǿ��Խ�TensorBoard����һ����¼Ա�������Լ�¼����ָ�������ݣ�����ģ��ÿһ���feature map��Ȩ�أ��Լ�ѵ��loss�ȵȡ�TensorBoard����¼���������ݱ�����һ���û�ָ�����ļ�������򲻶�������TensorBoard�᲻�ϼ�¼����¼�µ����ݿ���ͨ����ҳ����ʽ���Կ��ӻ���

Pytorch������TensorBoard���ӻ��Ĵ�Ź������£�

������Pytorch��ָ��һ��Ŀ¼����һ��torch.utils.tensorboard.SummaryWriter��־д������

Ȼ�������Ҫ���ӻ�����Ϣ��������־д��������Ӧ��Ϣ��־д������ָ����Ŀ¼��

���Ϳ��Դ�����־Ŀ¼��Ϊ��������TensorBoard

### TensorBoard������������

��ʹ��TensorBoardǰ��������Ҫ��ָ��һ���ļ��й�TensorBoard�����¼���������ݡ�Ȼ�����tensorboard�е�SummaryWriter��Ϊ��������¼Ա��

����tensorboardҲ�ܼ򵥣���������������

```
tensorboard --logdir=/path/to/logs/ --port=xxxx #�������д��ڵ��� tensorboard--help���ɲ鿴
```

### TensorBoardģ�ͽṹ���ӻ�


���ӻ�ģ�͵�˼·��7.1�н��ܵķ���һ�������Ǹ���һ���������ݣ�ǰ�򴫲���õ�ģ�͵Ľṹ����ͨ��TensorBoard���п��ӻ���ʹ��add_graph��

```python
writer.add_graph(model, input_to_model = torch.rand(1, 3, 224, 224))
writer.close()
```

д������ϼ�Ŀ¼������tensorboard�Ϳ��Գɹ��鿴��ǰģ�ͽṹ


### TensorBoardͼ����ӻ�


��������ͼ����ص�����ʱ�����Է���ؽ��������ͼƬ��tensorboard�н��п��ӻ�չʾ��

* ���ڵ���ͼƬ����ʾʹ��add_image
* ���ڶ���ͼƬ����ʾʹ��add_images
* ��ʱ��Ҫʹ��torchvision.utils.make_grid������ͼƬƴ��һ��ͼƬ����writer.add_image��ʾ


### TensorBoard�����������ӻ�

TensorBoard�����������ӻ�������������ʱ��������ı仯���̣�ͨ��add_scalarʵ�֣�


```python
writer = SummaryWriter('./pytorch_tb')
for i in range(500):
    x = i
    y = x**2
    writer.add_scalar("x", x, i) #��־�м�¼x�ڵ�step i ��ֵ
    writer.add_scalar("y", y, i) #��־�м�¼y�ڵ�step i ��ֵ
writer.close()
```

�������ͬһ��ͼ����ʾ������ߣ�����Ҫ�ֱ��������·����ʹ��SummaryWriterָ��·�������Զ�����������Ҫ��tensorboard����Ŀ¼�£���ͬʱ��add_scalar���޸����ߵı�ǩʹ��һ�¼��ɣ�


```python
writer1 = SummaryWriter('./pytorch_tb/x')
writer2 = SummaryWriter('./pytorch_tb/y')
for i in range(500):
    x = i
    y = x*2
    writer1.add_scalar("same", x, i) #��־�м�¼x�ڵ�step i ��ֵ
    writer2.add_scalar("same", y, i) #��־�м�¼y�ڵ�step i ��ֵ
writer1.close()
writer2.close()
```


����Ҳ������һ��writer����forѭ���в��ϴ���SummaryWriter����һ����ѡ���ʱ���½ǵ�Runs���ֳ����˹�ѡ����ǿ���ѡ��������Ҫ���ӻ������ߡ��������ƶ�Ӧ�����·�������ƣ�������x��y����

�ⲿ�ֹ��ܷǳ��ʺ���ʧ�����Ŀ��ӻ������԰������Ǹ���ֱ�۵��˽�ģ�͵�ѵ��������Ӷ�ȷ����ѵ�checkpoint������Smoothing������ť���Ե������ߵ�ƽ���ȣ�����ʧ�����𵴽ϴ�ʱ����Smoothing���������ڹ۲�loss������仯���ơ�


### TensorBoard�����ֲ����ӻ�

��������Ҫ�Բ��������������ı仯�����߶���ֲ������о�ʱ�����Է������TensorBoard�����п��ӻ���ͨ��add_histogramʵ�֡��������һ�����ӣ�


```python
import torch
import numpy as np

# ������̬�ֲ�������ģ���������
def norm(mean, std):
    t = std * torch.randn((100, 20)) + mean
    return t
 
writer = SummaryWriter('./pytorch_tb/')
for step, mean in enumerate(range(-10, 10, 1)):
    w = norm(mean, 1)
    writer.add_histogram("w", w, step)
    writer.flush()
writer.close()
```


### ��������ʹ��TensorBoard

���Ǻ��죬�������ȱ���ʵ���˽��ֺ��ٲ���
