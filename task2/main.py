import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, SubsetRandomSampler

from sklearn.model_selection import KFold

from dataloader import SymbolDataset
from net import ConvNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

writer = SummaryWriter()

def image_transforms():
    return transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomPosterize(bits=2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])

criterion = nn.CrossEntropyLoss()
dataset = SymbolDataset(base_path='../charts', transforms=image_transforms())

num_epochs = 15
batch_size = 32

number_of_splits = 10
splits = KFold(n_splits=number_of_splits, shuffle=True, random_state=42)
foldperf = {}

for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(len(dataset)))):

    print(f'Fold {fold+1}')

    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(val_idx)
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = ConvNet()
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.002, weight_decay = 0.0001)

    history = {'train_loss': [], 'test_loss': [],'train_acc':[],'test_acc':[]}

    for epoch in range(num_epochs):
        train_loss,train_correct=0.0,0
        model.train()
        for images, labels in train_loader:

            images,labels = images.to(device),labels.to(device)
            optimizer.zero_grad()
            output = model(images)

            loss = criterion(output,labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            
            scores, predictions = torch.max(output.data, 1)
            train_correct += (predictions == labels).sum().item()

        train_loss = train_loss / len(train_loader.sampler)
        train_acc = train_correct / len(train_loader.sampler) * 100
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        
        test_loss, test_correct = 0.0, 0
        model.eval()
        
        for images, labels in test_loader:

            images,labels = images.to(device),labels.to(device)
            output = model(images)

            loss = criterion(output,labels)
            test_loss += loss.item()*images.size(0)
            scores, predictions = torch.max(output.data,1)
            test_correct += (predictions == labels).sum().item()

        test_loss = test_loss / len(test_loader.sampler)
        test_acc = test_correct / len(test_loader.sampler) * 100

        print("Epoch:{}/{} Average Training Loss:{:.3f} Average Test Loss:{:.3f} Average Training Acc {:.2f} % Average Test Acc {:.2f} %".format(epoch + 1, num_epochs, train_loss, test_loss, train_acc, test_acc))

        writer.add_scalar("Loss/val", test_loss, epoch)
        writer.add_scalar("Accuracy/val", test_acc, epoch)

        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)

    foldperf['fold{}'.format(fold+1)] = history  

testl_f,tl_f,testa_f,ta_f=[],[],[],[]

writer.flush(); writer.close();

for f in range(1, number_of_splits+1):
    tl_f.append(np.mean(foldperf['fold{}'.format(f)]['train_loss']))
    testl_f.append(np.mean(foldperf['fold{}'.format(f)]['test_loss']))

    ta_f.append(np.mean(foldperf['fold{}'.format(f)]['train_acc']))
    testa_f.append(np.mean(foldperf['fold{}'.format(f)]['test_acc']))

print('\n..TRAINING COMPLETE..\n')
print(f'Performance of {number_of_splits} fold cross validation\n')
print("Average Training Loss: {:.3f} \nAverage Test Loss: {:.3f} \nAverage Training Acc: {:.2f} \nAverage Test Acc: {:.2f}".format(np.mean(tl_f),np.mean(testl_f),np.mean(ta_f),np.mean(testa_f)))  

torch.save(model.state_dict(), 'ckpt.pth')
torch.save(model.state_dict(), 'flask_app/static/models/ckpt.pth')