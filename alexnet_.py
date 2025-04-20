# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class AlexNet(nn.Module):
    def __init__(self, num_classes):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, 11, stride=4)
        self.conv2 = nn.Conv2d(96, 256, 5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(256, 384, 3, padding=1)
        self.conv4 = nn.Conv2d(384, 384, 3, padding=1)
        self.conv5 = nn.Conv2d(384, 256, 3, padding=1)
        self.pool = nn.MaxPool2d(3, 2)
        self.dropout = nn.Dropout(p=0.5)

        self.fc1 = nn.Linear(6400, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(F.relu(self.conv5(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

device = 'cuda' if torch.cuda.is_available() else 'cpu'

custom_transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])

training_data = datasets.CIFAR10(root='./',train=True,download=True,transform=custom_transform)
testing_data = datasets.CIFAR10(root='./',train=False,transform=custom_transform)

training_load = DataLoader(training_data, batch_size=500, shuffle=True)
testing_load = DataLoader(testing_data, batch_size=500)

#hyperparams

model = AlexNet(10)
model = model.to(device)
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
n_iters = 10

#Training Loop
n_iters=10
model.train()
for epoch in range(n_iters):

  for i, (im,label) in enumerate(training_load):
    im , label = im.to(device), label.to(device)
    pred = model(im)
    l = loss(pred,label)
    l.backward()
    optimizer.step()
    optimizer.zero_grad()
    if i % 10 == 0:
      print(f"epoch: {epoch+1}, step: {i}, loss {l}")

def test_model(model, test_loader, device):
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    avg_loss = total_loss / total
    print(f"Test Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")
    return avg_loss, accuracy

print(test_model(model,testing_load,device))
