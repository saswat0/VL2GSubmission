import os, copy
import torch

import torchvision
import torch.nn as nn

import torch.utils.data as Data
import torchvision.transforms as transforms


from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomPosterize(bits=2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),

    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomPosterize(bits=2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])    
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

root_dir = '../dataset'
train_data = torchvision.datasets.ImageFolder(
    root=os.path.join(root_dir, 'train'),
    transform=data_transforms['train']
)

val_data = torchvision.datasets.ImageFolder(
    root=os.path.join(root_dir, 'val'),
    transform=data_transforms['val']
)

class_names = train_data.classes
dataset_sizes = {'train': len(train_data), 'val': len(val_data)}

dataloaders = {  
    'train': Data.DataLoader(
        dataset=train_data,
        batch_size=32,
        shuffle=True,
    ),

    'val': Data.DataLoader(
        dataset=val_data,
        batch_size=1,
        shuffle=True,
    ),    
}

class VGG(torch.nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.vgg = torchvision.models.vgg16(pretrained=True) 
        self.conv = nn.Sequential(
            self.vgg.features, 
            self.vgg.avgpool 
        )
        self.fc = nn.Linear(512, 5)

    def forward(self,x):    
        x = self.conv(x)
        x = x.view(-1, 512, 7*7).mean(2)
        x = self.fc(x)
        return x

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model(model, loss_fn, optimizer, scheduler, num_epochs=25):
    
    best_model_weights = copy.deepcopy(model.state_dict())
    best_accuracy = 0.0
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-'*10)
        
        step_loss = 0.0
        epoch_accuracy = 0.0
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train() 
            else:
                model.eval() 
            
            step_loss = 0.0
            step_corrects = 0
            
            for step, (images, labels) in enumerate(dataloaders[phase]):      
               
                images, labels = images.to(device), labels.to(device)
                outputs = model(images) 
                preds = torch.max(outputs, 1)[1]
                loss = loss_fn(outputs, labels)    
                
                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                step_loss += loss.item() * images.size(0) 
                step_corrects += torch.sum(preds == labels.data)
                        
            if phase == 'train':
                scheduler.step()
            
            epoch_loss = step_loss / dataset_sizes[phase]
            epoch_accuracy = step_corrects.double() / dataset_sizes[phase]
            print('{} Loss: {:.4f} Accuracy: {:.4f}'.format(
                phase, epoch_loss, epoch_accuracy))

            if phase == 'train':
                train_loss.append(epoch_loss)
                train_acc.append(epoch_accuracy)
            else:
                val_loss.append(epoch_loss)
                val_acc.append(epoch_accuracy)
        
            if phase == 'val' and epoch_accuracy > best_accuracy:
                best_accuracy = epoch_accuracy
                best_model_weights = copy.deepcopy(model.state_dict())
            
        print()
        
    print('Best Validation Accuracy: {:4f}'.format(best_accuracy))
    
    model.load_state_dict(best_model_weights)
    return model

model = VGG()
model = model.to(device)

trainable_parameters = []
for name, p in model.named_parameters():
    if "fc" in name:
        trainable_parameters.append(p)

loss_fn = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(trainable_parameters, lr=0.001, momentum=0.9)
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

model = train_model(model, loss_fn, optimizer, exp_lr_scheduler, num_epochs=10)