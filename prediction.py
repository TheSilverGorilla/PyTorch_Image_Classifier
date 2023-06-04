import torch
import torchvision
import numpy
import PIL
import os

root_path = os.getcwd()
classes = []


def get_class(dir_name):
    global classes
    classes = os.listdir(os.path.join(root_path, dir_name))

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

def load_model_and_predict(model_file):
    checkpoint = torch.load(model_file)
    pred_model = NNfunction(num_classes=8)
    pred_model.load_state_dict(checkpoint)
