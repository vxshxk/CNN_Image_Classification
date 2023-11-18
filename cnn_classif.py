import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import torchvision
import torchvision.transforms as transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#HyperParameters
epochs=9
batch_size = 8
learning_rate = 0.01

#transform
transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
)

#initialization of training data
train_data = torchvision.datasets.CIFAR10(root='./data', train= True, transform=transform, download=True)
test_data = torchvision.datasets.CIFAR10(root='./data', train= False, transform=transform, download=True)
train_load = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_load = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

#classes
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# CNN class
class ConvNN(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv1 = nn.Conv2d(3, 6, 3)  # Change to Conv2d
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3)  # Change to Conv2d
        self.l1 = nn.Linear(16 * 6 * 6, 100)
        self.l2 = nn.Linear(100, 50)
        self.l3 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 6 * 6)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


#fetch images   
dataiter = iter(train_load)
images, labels = next(dataiter)    

model = ConvNN().to(device)

criterion= nn.CrossEntropyLoss()
optimizer= torch.optim.SGD(model.parameters(), lr=learning_rate)
for epoch in range(epochs):
    for i, (images, labels) in enumerate(train_load):
        images=images.to(device)
        labels=labels.to(device)
        
        
        output=model(images)
        loss=criterion(output, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 2000 == 0:
            print (f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_load)}], Loss: {loss.item():.4f}')
