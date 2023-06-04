import torch
import torchvision
import torchvision.transforms as transforms
'''import numpy as np
import matplotlib.pyplot as plt
import random
import os
from PIL import Image'''
from torch.autograd import Variable
import glob
from torch import nn
import pathlib
from torch.utils.data import DataLoader

'''os.chdir("vehicle_classification")
root_path = os.getcwd()
for i in os.listdir():
    for element in os.listdir(i):
        file_path = os.path.join(root_path, i, element)'''

transformer = transforms.Compose([
    transforms.Resize((150,150))
    ,transforms.ToTensor()
    ,transforms.Normalize([.5,.5,.5],[.5,.5,.5])])

train_path = "/Users/pranay/PycharmProjects/PyTorch/vehicle_classification"
test_path = "/Users/pranay/PycharmProjects/PyTorch/vehicle_classification_test"

train_loader = DataLoader(
    torchvision.datasets.ImageFolder(train_path, transform=transformer),
    batch_size=256, shuffle=True
)
test_loader = DataLoader(
    torchvision.datasets.ImageFolder(test_path, transform=transformer),
    batch_size=256, shuffle=True
)

root = pathlib.Path(train_path)
classes=sorted([j.name.split('/')[-1] for j in root.iterdir()])

class NNfunction(nn.Module):
    def __init__(self, num_classes=8):
        super(NNfunction,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=12,kernel_size=3,stride=1,padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=12)
        self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=12,out_channels=20,kernel_size=3,stride=1,padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels=20,out_channels=32,kernel_size=3,stride=1,padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=32)
        self.relu3 = nn.ReLU()
        self.fc = nn.Linear(in_features=32*75*75, out_features=num_classes)
        ##nn.flatten()

    def forward(self, x):
        output = self.conv1(x)
        output = self.bn1(output)
        output = self.relu1(output)
        output = self.pool(output)
        output = self.conv2(output)
        output = self.relu2(output)
        output = self.conv3(output)
        output = self.bn3(output)
        output = self.relu3(output)

        output = output.view(-1,180000)

        output = self.fc(output)

        return output

model = NNfunction(num_classes=8).to("cpu")

optimizer = torch.optim.Adam(model.parameters(), lr=1/1000, weight_decay=1/1000)
loss_function=nn.CrossEntropyLoss()

epochs = 10

temp_accuracy = 0
train_count=len(glob.glob(train_path+'/**/*.jpg'))
test_count=len(glob.glob(test_path+'/**/*.jpg'))

for epoch in range(epochs):
    model.train()
    train_accuracy=0
    train_loss=0

    for i, (images, labels) in enumerate(train_loader):
        if torch.cuda.is_available():
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())
        print(labels)
        optimizer.zero_grad()
        outputs = model(images)
        loss=loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.cpu().data*images.size(0)
        _,prediction = torch.max(outputs.data,1)

        train_accuracy+=int(torch.sum(prediction==labels.data))
    train_accuracy = train_accuracy/train_count
    train_loss = train_loss/train_count

    model.eval()

    test_accuracy = 0.0
    for i, (images, labels) in enumerate(test_loader):
        if torch.cuda.is_available():
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())

        outputs = model(images)
        _, prediction = torch.max(outputs.data, 1)
        test_accuracy+=int(torch.sum(prediction==labels.data))

    test_accuracy = test_accuracy/test_count

    print('Epoch: ' + str(epoch) + ' Train Loss: ' + str(train_loss) + ' Train Accuracy: ' + str(train_accuracy) + ' Test Accuracy: ' + str(test_accuracy))

    if test_accuracy>temp_accuracy:
        torch.save(model.state_dict(),'best_checkpoint.model')
        temp_accuracy = test_accuracy
