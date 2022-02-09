#%%
import torch
import torch.nn as nn
import torchvision.datasets as normal_datasets
import torchvision.transforms as transforms
from torch.autograd import Variable


# %% Download Dataset
num_epochs = 1
batch_size = 100
learning_rate = 0.001

# Process the data into Variable, or cuda if there is a GPU
def get_variable(x):
    x = Variable(x)
    return x.cuda() if torch.cuda.is_available() else x

# Loading some common datasets from torchvision.datasets
train_dataset = normal_datasets.MNIST(
                            root='./mnist/',                 # Data set save path
                            train=True,                      # Whether to use as a training set
                            transform=transforms.ToTensor(), # How the data is handled, can be customised by user
                            download=True)                   # If there is no data under the path, excute download

# See data loaders and batches
test_dataset = normal_datasets.MNIST(root='./mnist/',
                           train=False,
                           transform=transforms.ToTensor())

# %% Processing data, using DataLoader for batch training
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


# %% Modeling of computational diagrams : Two-layer convolution
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Quick build with sequence tools
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(7 * 7 * 32, 10)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out.view(out.size(0), -1)  # reshape
        out = self.fc(out)
        return out


cnn = CNN()
if torch.cuda.is_available():
    cnn = cnn.cuda()

# %% Defining optimisers and losses
loss_func = nn.CrossEntropyLoss() # Selection of loss functions and optimisation methods
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

# %% Performing batch training
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = get_variable(images)
        labels = get_variable(labels)

        outputs = cnn(images)
        loss = loss_func(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
                  % (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, loss.item()))


# %% Test models
cnn.eval()  # Change to test form, application scenarios such as: dropout
correct = 0
total = 0
for images, labels in test_loader:
    images = get_variable(images)
    labels = get_variable(labels)

    outputs = cnn(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels.data).sum()

print(' Test Accuracy: %d %%' % (100 * correct / total))

# Save the Trained Model
torch.save(cnn.state_dict(), 'cnn.pkl')
