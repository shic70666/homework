#%%
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torchvision import transforms
from pathlib import Path
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import os
from torchsummary import summary # 输出模型信息，计算模型整体参数量
from torchviz import make_dot, make_dot_from_trace # 可视化模型前馈的计算图
from visdom import Visdom # 可视化学习曲线
import time
torch.manual_seed(1)

#%% 
#! Hyper Parameters
num_epochs = 2
batch_size = 8
learning_rate = 0.01

#%% 
#! Pre-processing of Data
class SHM_Dataset(Dataset):
    """ Prepare dataset for pytorch
        Ref: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
    """

    def __init__(self, case, data_file, transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])): 
        self.case = case
        self.data_file = Path(data_file)
        self.data_df = pd.read_json(self.data_file, dtype=np.array)
        # self.data = self.data_df.cat()
        self.data = self.data_df.stack()
        self.labels = pd.DataFrame([self.case,]*self.data_df.shape[0]*self.data_df.shape[1])
        self.transform = transform
    
    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        label = np.array(self.labels.iloc[index])
        feature = np.array(self.data.iloc[index])
        feature = self.transform(feature)
        # print(f"feature: {feature.shape}, label: {label.shape}")
        return feature, label

shmDS_1 = SHM_Dataset(1, "~/Codes/homework/data/SHM/shm01s.json")
shmDS_2 = SHM_Dataset(2, "~/Codes/homework/data/SHM/shm02s.json")
shmDS_3 = SHM_Dataset(3, "~/Codes/homework/data/SHM/shm03s.json")
shmDS_4 = SHM_Dataset(4, "~/Codes/homework/data/SHM/shm04s.json")
shmDS_5 = SHM_Dataset(5, "~/Codes/homework/data/SHM/shm05s.json")
shmDS_6 = SHM_Dataset(6, "~/Codes/homework/data/SHM/shm06s.json")
shmDS_7 = SHM_Dataset(7, "~/Codes/homework/data/SHM/shm07s.json")
shmDS_8 = SHM_Dataset(8, "~/Codes/homework/data/SHM/shm08s.json")
shmDS_9 = SHM_Dataset(9, "~/Codes/homework/data/SHM/shm09s.json")
shmDS = shmDS_1 + shmDS_2 + shmDS_3 + shmDS_4 + shmDS_5 + shmDS_6 + shmDS_7 + shmDS_8 + shmDS_9

print("There is", len(shmDS), "samples in the given dataset")
# print(shmDS.__getitem__(11))

#%% 
#! Have a look on a single sample
# split samples into mini-batches and reshuffle the data to reduce overfitting
train_loader = DataLoader(shmDS, batch_size=batch_size, shuffle=True) 
test_loader = DataLoader(shmDS, batch_size=batch_size, shuffle=False)

train_features, train_labels = next(iter(train_loader))
# train_features = train_features.unsqueeze(dim=1)
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")

img = train_features[0].squeeze()
label = train_labels[0]
fig, axis = plt.subplots()
axis.imshow(img, cmap="gray")
axis.set(title=f"Label: {label}", xticks=range(16), yticks=range(16))
plt.show()

#%% 
#! Build the neural network:CNN Model
# Four-layer convolution
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Quick build with sequence tools
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.conv4 = nn.Sequential(
            nn.Conv2d(16, 81, kernel_size=3, padding=1),
            nn.BatchNorm2d(81),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(3 * 3 * 9, 9)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = out.view(out.size(0), -1) # 重构tensor维度，已知行向量，自动补齐列向量
        out = self.fc(out)
        return out

cnn = CNN()
summary(cnn, (1, 16, 16))
# make_dot(cnn, params=dict(cnn.named_parameters()))

#%% 
#! Loss and Optimizer
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

#%% 
#! Train the model: Visualization of Loss
viz = Visdom()
viz.images(train_features[:10],nrow=100)  # visdom可视化部分数据
time.sleep(0.5) # 为防止可视化视窗重叠现象，停顿0.5秒
test_data = Variable(train_features, volatile=True).float()
line = viz.line(np.arange(10)) # 创建线图可视化窗口
start_time = time.time() # 起始时间设置
time_p,  loss_p = [], [] # 可视化所需数据点
text = viz.text("<h1>convolution Nueral Network</h1>") # 创建可视化数据视窗
for epoch in range(num_epochs):
    # 由于分批次学习，输出loss为一批平均，需要累积or平均每个batch的loss
    sum_loss, sum_step = 0., 0.

    for i, (tx, ty) in enumerate(train_loader, 1):
        tx = Variable(tx)
        tx = tx.float()
        ty = Variable(ty)
        
        out = cnn(tx)
        loss = loss_func(out, ty) #TODO:modify
        sum_loss += loss.item()*len(ty)

        pred_tr = torch.max(out,1)[1]
        sum_step += ty.size(0)

        optimizer.zero_grad()  # 学习反馈
        loss.backward()
        optimizer.step()
        
        if i % 2 == 0: # 每2个batch可视化一下数据
            # 可视化部分
            time_p.append(time.time()-start_time)
            loss_p.append(sum_loss/sum_step)
            viz.line(X=np.array(time_p),
                     Y=np.array(loss_p),
                     win=line,
                     opts=dict(legend=["Loss"]))

            # visdom text 支持html语句
            viz.text("<p style='color:red'>epoch:{}</p><br><p style='color:blue'>Loss:{:.4f}</p><br>"
                     "<p style='color:green'>Time:{:.2f}</p>".format(epoch, sum_loss/sum_step,
                                                                       time.time()-start_time),
                     win=text)
            sum_loss = 0.

#%% 
#! Train the model
# for epoch in range(num_epochs):
#     for i, (train_features, train_labels) in enumerate(train_loader):
#         train_features = Variable(train_features)
#         # images = images.unsqueeze(dim=1)
#         train_features = train_features.float()
#         train_labels = Variable(train_labels)
#         # print(type(images), images)
#         # print(type(labels), labels)
#         # print("[ OK ] at this step") 

#         # Forward + Backward + Optimize
#         optimizer.zero_grad()
#         outputs = cnn(train_features)
#         # loss = loss_func(outputs, labels)
#         loss = loss_func(outputs, torch.max(train_labels, 1)[1])
#         loss.backward()
#         optimizer.step()

#         if (i + 1) % 100 == 0:
#             print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
#                   % (epoch + 1, num_epochs, i + 1, len(shmDS) // batch_size, loss.item()))

#%% 
#! Test the model
cnn.eval()  # Change to test form, application scenarios such as: dropout
correct = 0
total = 0
for test_features, test_labels in test_loader:
    test_features = Variable(test_features)
    # images = images.unsqueeze(dim=1)
    test_features = test_features.float()
    test_labels = Variable(test_labels)
    # print(test_features.shape)  Output:torch.Size([8, 1, 16, 16])

    outputs = cnn(test_features)
    # print(outputs)
    # print(outputs.shape)   Output:
    _, predicted = torch.max(outputs.data, 1)
    # print(outputs.data)
    # predicted = predicted +1 竟然是加了一个维度

    # print(predicted)
    # print(test_labels.data)
    
    total += test_labels.size(0)
    # print('ok for this step', total)
    correct += (predicted == test_labels.data).sum()
    # print('good for this step', correct)

print(' Test Accuracy: %d %%' % (100 * correct / total))

# Save the Trained Model
# torch.save(cnn.state_dict(), 'cnn.pkl')

#%% 
#! Debug Accuracy of Prediction
# outputs
print(predicted)
print(test_labels)
correct += (predicted == test_labels.data).sum()
cor_hb = 0
for i,j in zip(predicted, test_labels):
    if i == j:
        print("same lable", i, j)
        cor_hb = 1
    else:
        print(i, "≠", j)   
print("accuracy", cor_hb/len(predicted))

#%% 
#! Debugging
print("The last batch size should be", len(shmDS)%batch_size)
print("outputs has a shape of:", outputs.shape)
print("\tthe 1st item is:", outputs[1].shape, outputs[1])

print("predicted has a shape of:", predicted.shape)
print("\tthe 1st item is:", predicted[1].shape, predicted[1])

print("test_labels has a shape of:", test_labels.shape)
print("\tthe 1st item is:", test_labels[1].shape, test_labels[1])


# %%
