import os

import torchvision
import torch.utils.data as Data
import torchvision.transforms as transforms

class Loaders:

    def __init__(self, root_dir = '../dataset'):
        super(Loaders, self).__init__()

        self.root_dir = root_dir
        self.data_transforms = {
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

    def get_loaders(self):
        train_data = torchvision.datasets.ImageFolder(
            root=os.path.join(self.root_dir, 'train'),
            transform=self.data_transforms['train']
        )

        val_data = torchvision.datasets.ImageFolder(
            root=os.path.join(self.root_dir, 'val'),
            transform=self.data_transforms['val']
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

        return dataloaders, class_names, dataset_sizes