import torch
import torch.nn as nn
import torchvision.models as models

# Resnet18 model with random weight initialization.
class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        self.resnet = models.resnet18()  
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes) 

    def forward(self, x):
        return self.resnet(x)

# Multilayer perceptron
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        
#Resnet50 model with random weight initialization
class ResNet50(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet50, self).__init__()
        self.resnet = models.resnet50()  
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes) 

    def forward(self, x):
        return self.resnet(x)

#Densenet121 model with random weight initialization
class DenseNet(nn.Module):
    def __init__(self, num_classes=10):
        super(DenseNet, self).__init__()
        self.densenet = models.densenet121()  
        self.densenet.classifier = nn.Linear(self.densenet.classifier.in_features, num_classes) 

    def forward(self, x):
        return self.densenet(x)