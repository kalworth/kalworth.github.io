# pytorch����ģ����Ӧ��

## ���ѧϰ�Ļ���˼·�뷽��

�ع����������һ�����ѧϰ����ʱ�Ĳ��裬���� **��Ҫ�����ݽ���Ԥ����** ��������Ҫ�Ĳ���������ݸ�ʽ��ͳһ�ͱ�Ҫ�����ݱ任��ͬʱ **����ѵ�����Ͳ��Լ�** �������� **ѡ��ģ��** �����趨 **��ʧ�������Ż�����** ���Լ���Ӧ�� **������** ����Ȼ����ʹ��sklearn�����Ļ���ѧϰ����ģ���Դ�����ʧ�������Ż������������ģ��ȥ���ѵ�������ݣ����� **��֤��/���Լ��ϼ���ģ�ͱ���** ��

���ѧϰ�ͻ���ѧϰ�����������ƣ����ڴ���ʵ�����нϴ�Ĳ��졣���ȣ� **�������ѧϰ������������ܴ�һ�μ���ȫ���������п��ܻᳬ���ڴ��������޷�ʵ�֣�ͬʱ��������batch��ѵ�������ģ�ͱ��ֵĲ��ԣ���Ҫÿ��ѵ����ȡ�̶���������������ģ����ѵ��** ��������ѧϰ�����ݼ�������Ҫ��ר�ŵ���ơ�

��ģ��ʵ���ϣ����ѧϰ�ͻ���ѧϰҲ�кܴ���졣���������������������϶࣬ͬʱ����һЩ����ʵ���ض����ܵĲ㣨�����㡢�ػ��㡢�����򻯲㡢LSTM��ȣ������ **���������������Ҫ����㡱�������Ԥ�ȶ���ÿ���ʵ���ض����ܵ�ģ�飬�ٰ���Щģ����װ����** �����֡����ƻ�����ģ�͹�����ʽ�ܹ���ֱ�֤ģ�͵�����ԣ�Ҳ�Դ���ʵ��������µ�Ҫ��

����������ʧ�������Ż������趨���ⲿ�ֺ;������ѧϰ��ʵ�������Ƶġ�������ģ���趨������ԣ� **�����ʧ�������Ż���Ҫ�ܹ���֤���򴫲��ܹ����û����ж����ģ�ͽṹ��ʵ��** ��

����������ɺ�Ϳ��Կ�ʼѵ���ˡ�����ǰ�������GPU�ĸ����GPU���ڲ��м�����ٵĹ��ܣ����� **����Ĭ������CPU�����е�** ������ڴ���ʵ���У���Ҫ��ģ�ͺ����ݡ��ŵ���GPU��ȥ�����㣬ͬʱ����Ҫ��֤��ʧ�������Ż����ܹ���GPU�Ϲ��������ʹ�ö���GPU����ѵ��������Ҫ����ģ�ͺ����ݷ��䡢���ϵ����⡣���⣬��������һЩָ�껹��Ҫ�����ݡ��Żء�CPU�������漰����һϵ�� **�й���GPU�����úͲ���** ��

**���ѧϰ��ѵ������֤���������ص����ڶ��������ǰ����ģ�ÿ�ζ���һ�����ε����ݣ�����GPU��ѵ����Ȼ����ʧ�������򴫲���������ǰ��Ĳ㣬ͬʱʹ���Ż����������������������漰������ģ����ϵ����⡣ѵ��/��֤����Ҫ�����趨�õ�ָ�����ģ�ͱ��֡�**

�������ϲ��裬һ�����ѧϰ���������ˡ�

## ׼������

��ǰ����������Ҫ�İ����ݣ����ó��������������������������ǹ̶�����ֵ���ƣ�������ѵ���н�һ��������

������������

* batch size
* ��ʼѧϰ�ʣ���ʼ��
* ѵ��������max_epochs��
* GPU����

## �������ǵ�����

PyTorch���ݶ�����ͨ��Dataset+DataLoader�ķ�ʽ��ɵģ�Dataset��������ݵĸ�ʽ�����ݱ任��ʽ��DataLoader��iterative�ķ�ʽ���϶����������ݡ�


���Ū������pytorch��dataset�࣬����Դ�����Ӧ����ģ�͵����ݼ��ӿڡ�

��ν���ݼ����޷Ǿ���һ��{x:y}�ļ�������ֻ��Ҫ���������˵������һ��{x:y}�ļ��ϡ��Ϳ����ˡ�

����ͼ���������ͼ��+����

����Ŀ��������ͼ��+bbox������

���ڳ��ֱ������񣬵ͷֱ���ͼ��+���ֱ���ͼ��

�����ı����������ı�+����


������һ������Ϊ������������������ǵ�Dateset


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


����Ĵ�����pytorch�����Ĺٷ����룬����__getitem__��__len__���������̳еġ�

�ܺý��ͣ�pytorch�����Ĺٷ����������˱�׼����Ҫ�������ı�׼�������ݼ����������ȣ�__getitem__���ǻ�ȡ�����ԣ�ģ��ֱ��ͨ����һ�������һ��������{x:y}��__len__��ָ���ݼ����ȡ�


��Datawhale������ʵ��������ϸ�½�˵


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


***����_init_�������������������Ļ�����ɲ���***���������ڴ������ĸ�������data_dir��info_csv��image_list��transform��

* data_dir��ͼ��Ŀ¼��·����
* info_csv������ͼ���������Ӧ��ǩ�� csv �ļ���·��
* image_list������ѵ��/��֤����ͼ�����Ƶ�txt �ļ���·��
* transform��ҪӦ���������Ŀ�ѡ�任��һ�����ͼ��ʶ����Ӧ�ã����練תͼ��ͼ��ߵ�����

�ⲿ�ֲ���������������Ҫ�����������ݣ�����ת��Ϊ��Ӧ��Dataset��

ʣ�µ�����������pytorch�ٷ��涨�ı���̳еĺ���

___getitem ����Ҫ�󷵻�һ����Ӧ�����Լ����ǩ���߷�����Ϣ___

�����ڸú����ڣ����ǶԴ�����ļ��ڵ�����ͼƬ���ƺ�·��������ͬʱҪ������Ӧ�ı�ǩ��Ϣ�����ص���ͼ�������Ӧ�ı�ǩ

��Ҫע���������������ڣ����ǻ����뵱ǰд��ı仯

```python
if self.transform is not None:
            image = self.transform(image)
```

***len����Ҫ�󷵻��������ݵ��ܳ�***

��һ������Ƶ�ʱ����Ҫ������������������֮����еĵ�������

## ���ζ�������

�������ǳɹ�����˽�����ת��Ϊ�ɶ������ʽ������������Ҫʹ��pytorch�ٷ��Ķ��뷽ʽ


```python
from torch.utils.data import DataLoader

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=4, shuffle=True, drop_last=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, num_workers=4, shuffle=False)
```


* batch_size�������ǰ�����������ģ�batch_size����ÿ�ζ����������
* num_workers���ж��ٸ��������ڶ�ȡ����
* shuffle���Ƿ񽫶�������ݴ���
* drop_last�������������һ����û�дﵽ��������������ʹ�䲻�ٲ���ѵ��

**dataloader��������һ���ɵ������󣬿���ʹ��iter()���з��ʣ�����iter(dataloader)���ص���һ����������Ȼ�����ʹ��next()���ʡ�**
**Ҳ����ʹ��enumerate(dataloader)����ʽ���ʡ�**

ʾ������


```python
import matplotlib.pyplot as plt
images, labels = next(iter(val_loader))
print(images.shape)
plt.imshow(images[0].transpose(1,2,0))
plt.show()
```


## ����ģ��

### �����繹��


```python
import torch
from torch import nn

class MLP(nn.Module):
  # ��������ģ�Ͳ����Ĳ㣬��������������ȫ���Ӳ�
  def __init__(self, **kwargs):
    # ����MLP����Block�Ĺ��캯�������б�Ҫ�ĳ�ʼ���������ڹ���ʵ��ʱ������ָ����������
    super(MLP, self).__init__(**kwargs)
    self.hidden = nn.Linear(784, 256)
    self.act = nn.ReLU()
    self.output = nn.Linear(256,10)
  
   # ����ģ�͵�ǰ����㣬����θ�������x���㷵������Ҫ��ģ�����
  def forward(self, x):
    o = self.act(self.hidden(x))
    return self.output(o)   
```


ͬ������һ�δ���Ϊ���Ľ���

�������Ǽ̳�[nn.module](https://zhuanlan.zhihu.com/p/100000785#:~:text=%E7%AE%80%E5%8D%95%E7%9A%84%E8%AF%B4%EF%BC%8CMod,%E7%94%A8%E7%9A%84%E5%AD%90%E7%BD%91%E7%BB%9C%E5%B5%8C%E5%A5%97%E3%80%82)��ͬʱ���ø���Ĺ��캯������ɳ�ʼ��

**ͬʱ����������������ȫ���Ӳ���һ�������**

���ڱ��߶���Linear�����ľ���ϸ�ڻ����Ǻ���Ϥ���������·�����Դ������

![](https://upload-images.jianshu.io/upload_images/7437869-92097b2cc629c072.png?imageMogr2/auto-orient/strip|imageView2/2/w/665/format/webp)


in_featuresָ��������Ķ�ά�����Ĵ�С���������[batch_size, size]�е�size��

batch_sizeָ����ÿ��ѵ����batch)��ʱ�������Ĵ�С������CNN train������ͼƬ��60�ţ�����batch_size=15����ôiteration=4��������ѵ�����Σ���Ϊ����ÿ�ε�batch������ͬ�����ݣ�����ô����epoch��

����nn.Linear()�е���������������ͼƬ������ͬʱ����ÿ��ͼƬ��ά�ȡ�

out_featuresָ��������Ķ�ά�����Ĵ�С�������[batch_size��size]�е�size�����������ά�ȣ���batch_size�������е�һ�¡�


**ͬʱ�����ڶ�����ǰ�����ķ�ʽ**

����ͨ������������hidden�㣬֮�����output��


���ǿ���ʵ���� MLP ��õ�ģ�ͱ�? net �����У� net(X) ����� MLP �̳�?�� Module ��� **call** �����������������?�� MLP �ඨ���forward ���������ǰ����㡣


**pytorch��ҪҲ�ǰ��� `__call__`, `__init__`,`forward`��������ʵ�������֮��ļܹ���**

**���ȴ��������m��Ȼ��ͨ�� `m(input)`ʵ���ϵ��� `__call__(input)`��Ȼ�� `__call__(input)`����
`forward()`����**


### �Զ���������Ҫ�Ĳ�

�������ο�����

[pytorchԴ���Ķ�ϵ��֮Parameter�� - ֪�� (zhihu.com)](https://zhuanlan.zhihu.com/p/101052508) [AI���õ�18����DL���ѧϰ����PyTorch�Զ�������� (baidu.com)](https://baijiahao.baidu.com/s?id=1737784552681280788)

֮ǰһֱ�����Ϊʲô�ڼ̳�module�����Ҫָ����ص�Ȩ�أ������ǳ�ʼ��ʱ��Ϊ��������ʼ����

�ٷ������Բ�ʵ�����£�

![](https://pics4.baidu.com/feed/f603918fa0ec08fa88353e53faa1a16754fbdae3.png?token=8c1ce681a03194da78ca0740f791b2cb)


�������壺

���������input_features�������������ȣ�output_features����������ĳ��ȣ�input�ǵ��ø���ʱ���������ݣ�

�ڲ�������weight�ǲ��Ȩ�أ�bias�ǲ��ƫ�ã�

�ڲ�������__init__�ǹ��캯����forward��ǰ�򴫲�������reset_parameters�ǲ�����ʼ��������

����nn.Parameter��ʾ��ǰ������Ҫ�󵼡�

**���������ٷ��������ܽ�ó���Ҫ **ʵ��һ���Զ������Ҫ��������** ��**

A.�Զ���һ���࣬����̳���nn.Module�࣬����һ��Ҫʵ�����������ĺ��������캯��__init__()������߼����㺯��forward()��

B.�ڹ��캯��__init__()��ʵ�ֲ�Ĳ������壻

C.��ǰ�򴫲�forward������ʵ�������ݵ�ǰ�򴫲��߼���ֻҪ��nn.Module�������ж�����forward()������backward()�����ͻᱻ�Զ�ʵ�֡�

ע�⣺һ����������Ƕ���Ĳ����ǿɵ��ģ���������Զ���������ɵ�������Ҫ�����ֶ�ʵ��backward()������

��ˣ��Զ������Է�Ϊ���֣�һ���Ǵ������ģ�һ���ǲ��������ġ�


���Զ��庬ģ�Ͳ����Ĳ�ʱ������������� Parameter��**���ʾ������Ҫ�󵼡�**

Ҳ����ʹ��ParameterList �� ParameterDict �ֱ�������ı���ֵ䡣

*ParameterList ����һ�� Parameter ʵ�����б���Ϊ����Ȼ��õ�һ��������ʹ�õ�ʱ�����������������ĳ������������Ҳ����ʹ�� append �� extend �ڱ��������������*

*ParameterDict ����һ�� Parameter ʵ���ֵ���Ϊ����Ȼ��õ�һ�������ֵ䣬Ȼ����԰����ֵ�Ĺ���ʹ�á���ʹ�� update() ����������ʹ�� keys() �������м�ֵ��ʹ�� items() �������м�ֵ�ԡ�*


### ����һ��ģ��ʾ��

![](https://datawhalechina.github.io/thorough-pytorch/_images/3.4.1.png)


```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # ����ͼ��channel��1�����channel��6��5x5�����
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 2x2 Max pooling
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # ����Ƿ���,�����ֻʹ��һ�����ֽ��ж���
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # ��ȥ������ά�ȵ���������ά��
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

```


## ģ�ͳ�ʼ��

�����ⲿ�ֵ�Ӧ���ܽ���˵�������ڵ���init������ʱ����Ҫ��Բ�ͬ���͵��񾭲���з������

```python
def initialize_weights(self):
	for m in self.modules():
		# �ж��Ƿ�����Conv2d
		if isinstance(m, nn.Conv2d):
			torch.nn.init.xavier_normal_(m.weight.data)
			# �ж��Ƿ���ƫ��
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

���������������ģ���е�ÿ���񾭲㣬���жϸ��񾭲�����ͣ��Դ�Ϊ�񾭲�����Ӧ�ĳ�ʼ������������Ӧ�Ĳ���

## ��ʧ�����ĵ���

���ڲ�ͬ�������ʧ������ʵ��ϸ�ڣ��̳����Ѿ������ĺ�ȫ�棬������������������Զ�����ʧ����

ֱ������torch.Tensor�ṩ�Ľӿڣ�

![](https://img-blog.csdnimg.cn/20200117232455510.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzI0NDA3NjU3,size_16,color_FFFFFF,t_70)

## ѵ��������

һ��������ͼ������ѵ������������ʾ��


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


��ʼ�õ�ǰ����������ѵ��ʱ��Ӧ���Ƚ��Ż������ݶ����㣺


```python
optimizer.zero_grad()
```


֮��data����ģ����ѵ����


```python
output = model(data)
```


����Ԥ�ȶ����criterion������ʧ������


```python
loss = criterion(output, label)
```


��loss���򴫲������磺


```python
loss.backward()
```


ʹ���Ż�������ģ�Ͳ�����


```python
optimizer.step()
```

��Ӧ�ģ�һ������ͼ��������֤����������ʾ��


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


## ���ӻ�

���ӻ������������Ὣѵ����ÿ�������Ի�ͼ�ķ�ʽ����������������ֱ�۵ĸ��ܵ�ѵ�������������Լ�˼��δ�����Ż�����

���õĿ���

* matplotlib
* Seaborn

ʣ�µı��߲��죨��

## pytorch�Ż���

pytorch�����ṩ�����������Ż��������������Ż����Ļ���Optimizer

��������

```python
class Optimizer(object):
    def __init__(self, params, defaults):    
        self.defaults = defaults
        self.state = defaultdict(dict)
        self.param_groups = []
```

defaults���洢�����Ż����ĳ�����

state�������Ļ���

params_groups������Ĳ����飬��һ��list������ÿ��Ԫ����һ���ֵ䣬˳����params��lr��momentum��dampening��weight_decay��nesterov

������������

```python
import os
import torch

# ����Ȩ�أ�������̬�ֲ�  --> 2 x 2
weight = torch.randn((2, 2), requires_grad=True)
# �����ݶ�Ϊȫ1����  --> 2 x 2
weight.grad = torch.ones((2, 2))
# ������е�weight��data
print("The data of weight before step:\n{}".format(weight.data))
print("The grad of weight before step:\n{}".format(weight.grad))
# ʵ�����Ż���
optimizer = torch.optim.SGD([weight], lr=0.1, momentum=0.9)
# ����һ������
optimizer.step()
# �鿴����һ�����ֵ���ݶ�
print("The data of weight after step:\n{}".format(weight.data))
print("The grad of weight after step:\n{}".format(weight.grad))
# Ȩ������
optimizer.zero_grad()
# ����Ȩ���Ƿ�Ϊ0
print("The grad of weight after optimizer.zero_grad():\n{}".format(weight.grad))
# �������
print("optimizer.params_group is \n{}".format(optimizer.param_groups))
# �鿴����λ�ã�optimizer��weight��λ��һ�����Ҿ���������Բο�Python�ǻ���ֵ����
print("weight in optimizer:{}\nweight in weight:{}\n".format(id(optimizer.param_groups[0]['params'][0]), id(weight)))
# ��Ӳ�����weight2
weight2 = torch.randn((3, 3), requires_grad=True)
optimizer.add_param_group({"params": weight2, 'lr': 0.0001, 'nesterov': True})
# �鿴���еĲ���
print("optimizer.param_groups is\n{}".format(optimizer.param_groups))
# �鿴��ǰ״̬��Ϣ
opt_state_dict = optimizer.state_dict()
print("state_dict before step:\n", opt_state_dict)
# ����5��step����
for _ in range(50):
    optimizer.step()
# �������״̬��Ϣ
print("state_dict after step:\n", optimizer.state_dict())
# ���������Ϣ
torch.save(optimizer.state_dict(),os.path.join(r"D:\pythonProject\Attention_Unet", "optimizer_state_dict.pkl"))
print("----------done-----------")
# ���ز�����Ϣ
state_dict = torch.load(r"D:\pythonProject\Attention_Unet\optimizer_state_dict.pkl") # ��Ҫ�޸�Ϊ���Լ���·��
optimizer.load_state_dict(state_dict)
print("load state_dict successfully\n{}".format(state_dict))
# ������������Ϣ
print("\n{}".format(optimizer.defaults))
print("\n{}".format(optimizer.state))
print("\n{}".format(optimizer.param_groups))
```
