## K折交叉验证模型调优

### CV车道渲染(baseline调优)

- 导入第三方库


```python
import os
import glob
from PIL import Image
import csv
import numpy as np

import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.utils.data as data
import matplotlib.pyplot as plt
```

    C:\Users\王佳乐\AppData\Roaming\Python\Python38\site-packages\tqdm\auto.py:22: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
      from .autonotebook import tqdm as notebook_tqdm
    

- 自定义dataset


```python
class ImageSet(data.Dataset):
    def __init__(
            self,
            images,
            labels,
            transform):
        self.transform = transform
        self.images = images
        self.labels = labels

    def __getitem__(self, item):
        imagename = self.images[item]
        try:
            image = Image.open(imagename)
            image = image.convert('RGB')
        except:
            image = Image.fromarray(np.zeros((256, 256), dtype=np.int8))
            image = image.convert('RGB')

        image = self.transform(image)
        return image, torch.tensor(self.labels[item])

    def __len__(self):
        return len(self.images)
```

- 配置GPU环境以及设置部分超参


```python
# GPU判断和配置
print(torch.cuda.is_available())
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# 设置部分超参
batch_size = 32
num_worker = 0  # windows设置为0
fold_epochs = 1    # 迭代次数
```

    True
    

- 加载训练集


```python
import pandas as pd
import codecs

lines = codecs.open('C:/Users/王佳乐/Datawhale/test/datawhale相关学习赛事/华为CV车道渲染/data/digix-2022-cv-sample-0829/train_label.csv').readlines()
train_label = pd.DataFrame({
    'image': ['C:/Users/王佳乐/Datawhale/test/datawhale相关学习赛事/华为CV车道渲染/data/digix-2022-cv-sample-0829/train_image/' + x.strip().split(',')[0] for x in lines],
    'label': [x.strip().split(',')[1:] for x in lines],
})
train_label['new_label'] = train_label['label'].apply(lambda x: int('0' in x))
```

- 标签二值化


```python
import cv2, os
def check_image(path):
    try:
        if os.path.exists(path):
            return True
        else:
            return False
    except:
        return False

train_is_valid = train_label['image'].apply(lambda x: check_image(x) )
train_label = train_label[train_is_valid]
len(train_label)
```




    1969



- 数据预处理


```python
trfs = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

- 交叉验证数据集初始化


```python
#设置划分数目
k_value = 7
#每份含有的样本数目
each_num = int(len(train_label)/k_value)   #取整
train_label.index = list(np.arange(1,len(train_label)+1))   #顺序化索引
```

- 模型定义


```python
model = torchvision.models.resnet18(pretrained=True)
model.fc = torch.nn.Linear(512, 2)
model = model.cuda()
```

    c:\Anaconda3\envs\pytorch_gpu\lib\site-packages\torchvision\models\_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and will be removed in 0.15, please use 'weights' instead.
      warnings.warn(
    c:\Anaconda3\envs\pytorch_gpu\lib\site-packages\torchvision\models\_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and will be removed in 0.15. The current behavior is equivalent to passing `weights=ResNet18_Weights.IMAGENET1K_V1`. You can also use `weights=ResNet18_Weights.DEFAULT` to get the most up-to-date weights.
      warnings.warn(msg)
    

- Loss函数和优化器设置


```python
optimizer = optim.SGD(model.parameters(), lr=0.005)
loss = nn.CrossEntropyLoss()
```

- 定义交叉验证中训练和验证函数


```python
import time
def train(t_loder):
    model.train()
    start_t = time.time()
    epoch_l = 0
    epoch_t = 0
    for batch_idx, batch in enumerate(t_loder):
        optimizer.zero_grad()
        image, label = batch
        image, label = image.to('cuda'), label.to('cuda')
        output = model(image)

        l = loss(output, label)
        l.backward()
        optimizer.step()

        batch_l = l.item()
        epoch_l += batch_l
        batch_t = time.time() - start_t
        epoch_t += batch_t
        start_t = time.time()


        
        #if batch_idx % 10  == 0:
           #print(l.item(), batch_idx, len(train_loader))

    # epoch_t = epoch_t / len(train_loader)
    # epoch_l = epoch_l / len(train_loader)
    return epoch_l,epoch_t
    #print('Training times: {}/{}\tTraining loss: {:.4f}\tAverage time: {:.2f}.'.format(t_times+1,k_value-1,epoch_l,epoch_t))

def val(v_loder):
    model.eval()
    val_loss = 0
    gt_labels = []
    pred_labels = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(v_loder):
            image,label = batch
            image,label = image.cuda(), label.cuda()
            output = model(image)
            preds = torch.argmax(output, 1)
            gt_labels.append(label.cpu().data.numpy())
            pred_labels.append(preds.cpu().data.numpy())
            loss_val = loss(output, label)
            val_loss += loss_val.item()*image.size(0)
    val_loss = val_loss/len(v_loder.dataset) 
    gt_labels, pred_labels = np.concatenate(gt_labels), np.concatenate(pred_labels)
    acc = np.sum(gt_labels==pred_labels)/len(pred_labels)
    return val_loss,acc
    #print('Validation Loss: {:.6f}, Accuracy: {:6f}'.format(val_loss, acc))
```

- 创建位移列表


```python
#循环右移
def list_move_right(A,a):
    for i in range(a):
        A.insert(0,A.pop())
    return A

```

- 开始验证


```python
for epoch in range(fold_epochs):
    #模型权重保存和读取
    save_dir = 'C:/Users/王佳乐/Datawhale/test/pytorch/model/Cv_dict.pkl'
    loaded_dict = torch.load(save_dir)
    model.state_dict = loaded_dict
    ##
    epoch_tloss = 0
    val_acc = 0
    epoch_t = 0
    train_label = train_label.sample(frac=1).reset_index(drop=True)   #打乱原数据集

    for list_num in range(k_value):      # 遍历完之后为一次完整的K折交叉
        move_list = list(range(0,k_value,1))
        use_list = list_move_right(move_list,list_num)

        for run_number in range(k_value-1):
            index_head = use_list.index(run_number)*each_num
            index_end = index_head + each_num
    
            ##    划分训练集
            train_dataset = ImageSet(train_label['image'].values[index_head:index_end],train_label['new_label'].values[index_head:index_end],trfs)
            train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=False,num_workers=0,pin_memory=True,drop_last=True)
            #      训练
            train_loss,train_time = train(train_loader)
            epoch_tloss += train_loss
            epoch_t +=train_time

        
        ##    划分验证
        val_index_head = (use_list.index(max(use_list)))*each_num
        val_index_end = val_index_head+each_num
        val_dataset = ImageSet(train_label['image'].values[val_index_head:val_index_end],train_label['new_label'].values[val_index_head:val_index_end],trfs)
        val_loder = DataLoader(train_dataset,batch_size=batch_size,shuffle=False,num_workers=0,pin_memory=True)

        #验证
        val_loss,acc = val(val_loder)
        val_acc += acc
        #print('Fold: {}/{}'.format(list_num+1,k_value))
    train_number = k_value*(k_value-1)
    val_number = k_value
    print('Fold_Epoch: {}/{}\ttrain Loss: {:.6f}\tval Acc: {:.4f}\tAverage time: {:.2f}.'.format(epoch+1,fold_epochs,epoch_tloss/train_number,val_acc/val_number,epoch_t/k_value))
    torch.save(model.state_dict, save_dir)
        
```

    fold: 1/7
    fold: 2/7
    fold: 3/7
    fold: 4/7
    fold: 5/7
    fold: 6/7
    fold: 7/7
    Fold_Epoch: 1/1	train Loss: 0.174783	val Acc: 0.9466	Average time: 8.67.
    

- Test和生成csv结果文件


```python
test_images = glob.glob('C:/Users/王佳乐/Datawhale/test/datawhale相关学习赛事/华为CV车道渲染/data/digix-2022-cv-sample-0829/test_images/*')
test_dataset = ImageSet(test_images, [0] * len(test_images), trfs)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
)
len(test_loader)
```




    313




```python
#测试
for data in test_loader:
    break
```


```python
model.eval()
to_prob = nn.Softmax(dim=1)
with torch.no_grad():
    imagenames, probs = list(), list()
    for batch_idx, batch in enumerate(test_loader):
        image, _ = batch
        image = image.to('cuda')
        pred = model(image)
        prob = to_prob(pred)
        prob = list(prob.data.cpu().numpy())
        probs += prob
```


```python
import csv
with open('submission.csv', 'w',newline = '', encoding='utf8') as fp:
    writer = csv.writer(fp)
    writer.writerow(['imagename', 'defect_prob'])
    for imagename, prob in zip(test_images, probs):
        imagename = os.path.basename(imagename)
        writer.writerow([imagename, str(prob[0])])
```
