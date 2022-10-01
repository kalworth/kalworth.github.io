# 华为CV赛事经验总结

最近和哥们一起参加了Datawahle一起组织的CV方向实践赛事，也算是取得了不小的进步和收获，在这里总结一下这次的经验和优化策略。

## 数据观察和处理

赛事开始后我们拿到了Datawhale的学习数据，大约3个G左右，这个赛道的主要命题就是识别出问题赛道，问题赛道在路线标识上往往会出现越界或者和其他标识模糊的特征，关键就在于提取这些特征。
一开始采用的是Datawhale官方提供的baseline数据处理，也就是随机旋转图像，但是在观察后，我们引入了更为标准的变换进一步扩充数据量，因为车道往往排布是比较规律的我们不希望过于随机的旋转来影响模型的准确度，同时在epoch较大时也能够保证不出现过度学习，最后取得了0.02的AUC进步


```python
trfs = transforms.Compose([
    transforms.Resize((224, 224)),
    # transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(90),
    transforms.RandomRotation((90)),
    transforms.RandomRotation(30, center=(0, 0)),
    transforms.RandomRotation(30, center=(0, 0)),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```


## 训练集以及验证集划分

在学习版数据内，数据数量有限，而在官方提供的全量数据集内，训练集的数量大大增加，但是存在一部分未被标识的数据，在🐮老师的指点下，我们最终采取了伪标签生成的策略，生成了一部分新的数据集，经过处理后我们最终拿到了非常庞大的数据。

### 伪标签生成

我们采用了先训练，后生成的办法，在保证训练模型足够可靠的情况下，卡了0.9的置信度开始生成训练标签，基本就是简单地改了一下源代码内的验证部分，
同时在训练过程中我们也加入了这部分生成的标签。

```python
model = torch.load('D:/2022_2_data/network_large.pth')
model.eval()
number=1000
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
        number-=number
        if(number<=0):
            break

import csv
with open('submission.csv', 'w',newline = '', encoding='utf8') as fp:
    writer = csv.writer(fp)
    writer.writerow(['imagename', 'defect_prob'])
    for imagename, prob in zip(test_images, probs):
        imagename = os.path.basename(imagename)
        if(prob[0]>0.9 or prob[1]>0.9):
            if(prob[0]>prob[1]):
                pred_label = '0'
            else:
                pred_label = '1'
        else:
            continue
        writer.writerow([imagename, pred_label])
        writer.writerow([imagename, str(prob[0])])
```

同时对于训练时载入数据集也要做一定的调整

```python
lines = codecs.open('D:/2022_2_data/train_label/train_label/train_label.csv').readlines()
fake_lines = codecs.open('D:/2022_2_data/train_label/train_label/fake_label.csv').readlines()
train_label = pd.DataFrame({
    'image': ['train_image/labeled_data/' + x.strip().split(',')[0] for x in lines],
    'label': [x.strip().split(',')[1:] for x in lines],
})
train_label = train_label.append(pd.DataFrame({
    'image': ['train_image/unlabeled_data/' + x.strip().split(',')[0] for x in fake_lines],
    'label': [x.strip().split(',')[1:] for x in fake_lines],
}),ignore_index=True)
```

### 验证集划分

为了直观地评价我们正在训练的模型，我们对原数据进行了一定程度的划分，实现策略为K折划分


- 交叉验证数据集初始化

```python
#设置划分数目
k_value = 7
#每份含有的样本数目
each_num = int(len(train_label)/k_value)   #取整
train_label.index = list(np.arange(1,len(train_label)+1))   #顺序化索引
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

## 经验总结

多向前辈请教（自己一个人捣鼓还是学习太慢）
最简单的优化策略-》提高算力，实际比赛的过程中经常算力吃紧
后面会写写autodl部署和最近在体验的驱动云（白嫖最香😂）
