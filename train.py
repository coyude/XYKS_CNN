import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import SVHN
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


transform_test = transforms.Compose([
    transforms.ToTensor(),
])

transform_train = transforms.Compose([
    transforms.ToTensor(),
])


train_dataset = SVHN(root='./data', split='train', transform=transform_train, download=True)
test_dataset = SVHN(root='./data', split='test', transform=transform_test, download=True)


train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        
        self.dropout = nn.Dropout(0.5)  

        
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10) 

    def forward(self, x):
        # 前向传播
        x = self.pool(torch.relu(self.conv1(x))) 
        x = self.pool(torch.relu(self.conv2(x)))  
        x = self.pool(torch.relu(self.conv3(x)))  

        x = x.view(-1, 128 * 4 * 4)  
        x = torch.relu(self.fc1(x))  
        x = torch.relu(self.fc2(x))  
        x = self.fc3(x)  

        return x

model = CNN().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01,momentum=0.9)
criterion = nn.CrossEntropyLoss()

def train_model(model, criterion, optimizer, train_loader):
    for epoch in range(30):
        model.train()
        trainLoss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels % 10 
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            trainLoss += loss.item()
        avgTrainLoss = trainLoss / len(train_loader)
        print(f'Epoch {epoch + 1}, 训练损失: {avgTrainLoss}, 学习率: {optimizer.param_groups[0]["lr"]}')


train_model(model, criterion, optimizer, train_loader)

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels % 10  # 将标签10转换为0
            outputs = model(inputs)
            test_loss += F.nll_loss(outputs, labels, reduction='sum').item()
            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print(f'\n测试集: 平均损失: {test_loss:.4f}, 准确率: {accuracy:.2f}%\n')
                

test(model, device, test_loader)
torch.save(model.state_dict(), 'SVHN_cnn1.pth')
