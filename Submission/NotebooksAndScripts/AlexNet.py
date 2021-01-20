from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import numpy as np
import math
import matplotlib.pyplot as plt
class QDataset(Dataset):
    def __init__(self, x, y):
        # self.x=np.load(type+"_data.npy", allow_pickle=True)
        # self.y=np.load(type+'_label.npy',allow_pickle=True)
        self.x = (torch.from_numpy(x)/255).float()
        self.y = torch.from_numpy(y).float()

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.x.shape[0]


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.cnet=nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=11, stride=4,padding=2),  # (b x 64 x 15 x 15)
            nn.ReLU(inplace=True),  # section 3.3
            nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 96 x 27 x 27)
            nn.Conv2d(64, 192, 5, padding=2),  # (b x 256 x 27 x 27)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 256 x 13 x 13)
            nn.Conv2d(192, 384, 3, padding=1),  # (b x 384 x 13 x 13)
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, 3, padding=1),  # (b x 384 x 13 x 13)
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.nn=nn.Sequential(
            nn.Dropout(),
            nn.Linear(256,4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096,4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096,200),
        )

    def forward(self, x):
        x=self.cnet(x)
        x=torch.flatten(x,1)
        x=self.nn(x)
        return x


def train(log_interval, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.CrossEntropyLoss()
        target=torch.tensor(target.clone().detach(),dtype=torch.long, device=device)
        lo=loss(output, target)
        lo.backward()
        optimizer.step()
        '''if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), lo.item()))'''


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            target=torch.tensor(target.clone().detach(),dtype=torch.long, device=device)
            loss = nn.CrossEntropyLoss()
            lo=loss(output, target)
            test_loss +=lo.item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            data, target = data.to("cpu"), target.to("cpu")

    test_loss /= len(test_loader.dataset)

    '''print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))'''

    return test_loss,correct / len(test_loader.dataset)


def main():
    batch_size=128
    test_batch_size=1000
    epochs=125
    lr=0.1
    gamma=0.987
    no_cuda=False
    seed=1
    log_interval=100
    save_model=False

    use_cuda = not no_cuda and torch.cuda.is_available()

    torch.manual_seed(seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    x1=np.load('/home/cse/btech/cs1180349/scratch/train_x.npy')
    y1=np.load('/home/cse/btech/cs1180349/scratch/train_y.npy')
    train_loader = torch.utils.data.DataLoader(QDataset(x1, y1), batch_size=batch_size, shuffle=True, **kwargs)
    x2=np.load('/home/cse/btech/cs1180349/scratch/test_x.npy')
    y2=np.load('/home/cse/btech/cs1180349/scratch/test_y.npy')
    test_loader = torch.utils.data.DataLoader(QDataset(x2, y2), batch_size=test_batch_size, shuffle=True, **kwargs)

    model = AlexNet().to(device)
    if use_cuda:
        model=torch.nn.DataParallel(model)
        torch.backends.cudnn.benchmark=True
    optimizer = optim.SGD(model.parameters(), lr=lr)
    l1=[0]*epochs
    l2=[0]*epochs
    l3=[0]*epochs
    l4=[0]*epochs
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
    for epoch in range(1, epochs + 1):
        print(epoch)
        train(log_interval, model, device, train_loader, optimizer, epoch)
        l1[epoch-1],l2[epoch-1]=test(model, device, test_loader)
        l3[epoch-1],l4[epoch-1]=test(model, device, train_loader)
        scheduler.step()
    plot1 = plt.figure(1)
    plt.title("Train Accuracy vs epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Train Accuracy")
    plt.plot(l4)
    plt.savefig("/home/cse/btech/cs1180349/scratch/train_acc_alex.png")
    plot1 = plt.figure(2)
    plt.title("Train Loss vs epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.plot(l3)
    plt.savefig("/home/cse/btech/cs1180349/scratch/train_loss_alex.png")
    plot1 = plt.figure(3)
    plt.title("Test Accuracy vs epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")
    plt.plot(l2)
    plt.savefig("/home/cse/btech/cs1180349/scratch/test_acc_alex.png")
    plot1 = plt.figure(4)
    plt.title("Test Loss vs epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Test Loss")
    plt.plot(l1)
    plt.savefig("/home/cse/btech/cs1180349/scratch/test_loss_alex.png")
    if save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")

    return model

if __name__=='__main__':
    a=main()
