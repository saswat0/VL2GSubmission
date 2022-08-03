import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.utils as utils
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from tensorboardX import SummaryWriter

import os

from net import VGGAttention
from utils import train_epoch, val_epoch, visualize_attn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),

    'val': transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

root_dir = '../../dataset'
train_set = torchvision.datasets.ImageFolder(
    root=os.path.join(root_dir, 'train'),
    transform=data_transforms['train']
)

test_set = torchvision.datasets.ImageFolder(
    root=os.path.join(root_dir, 'val'),
    transform=data_transforms['val']
)

lr = 0.0001
weight_decay = 0.0001
num_epochs = 15
CHECKPOINT_PATH = 'cnn_checkpoint.pth'

train_loader = DataLoader(train_set, batch_size = 16, shuffle=True, num_workers=2)
test_loader = DataLoader(test_set, batch_size = 16, shuffle=True, num_workers=2)

model = VGGAttention(sample_size=32, num_classes=100).to(device)

if torch.cuda.device_count() > 1:
    print("Using {} GPUs".format(torch.cuda.device_count()))
    model = nn.DataParallel(model)

writer = SummaryWriter()

best_acc = -float('inf')

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

for epoch in range(num_epochs):
    train_acc = train_epoch(model, criterion, optimizer, train_loader, device, epoch, 100, writer)
    val_acc = val_epoch(model, criterion, test_loader, device, epoch, writer)
    
    if val_acc > best_acc:
        best_acc = max(val_acc, best_acc)
        torch.save(model.state_dict(), CHECKPOINT_PATH)
        print("Saving Model of Epoch {}".format(epoch+1))

print(f'Best accuracy: {best_acc}')

model.load_state_dict(torch.load(CHECKPOINT_PATH))
model.eval()

with torch.no_grad():
    for batch_idx, (inputs, labels) in enumerate(test_loader):

        inputs = inputs.to(device)
        if batch_idx == 0:
            images = inputs[0:16,:,:,:]
            I = utils.make_grid(images, nrow=4, normalize=True, scale_each=True)
            writer.add_image('origin', I)
            _, c1, c2, c3 = model(images)

            attn1 = visualize_attn(I, c1)
            writer.add_image('attn1', attn1)
            attn2 = visualize_attn(I, c2)
            writer.add_image('attn2', attn2)
            attn3 = visualize_attn(I, c3)
            writer.add_image('attn3', attn3)