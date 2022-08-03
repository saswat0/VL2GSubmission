import os
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from net import PreTrainedModelFineTuning
from utils import adjust_learning_rate, accuracy, AverageMeter, save_checkpoint

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__"))

root_dir = '../../dataset'
print_interval = 5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

network = 'vgg16'
bs = 32
global_lr = 0.001

num_epochs = 15
start_epoch = 0

best_prec1 = 0

def main():
    global best_prec1

    traindir = os.path.join(root_dir, 'train')
    valdir = os.path.join(root_dir, 'val')

    num_classes = len([name for name in os.listdir(traindir)])
    print(f"Num classes'{num_classes}'")

    print(f"Network name: {network}")
    original_model = models.__dict__[network](pretrained=True)
    model = PreTrainedModelFineTuning(original_model, network, num_classes)

    if network.startswith('alexnet') or network.startswith('vgg'):
        model.features = torch.nn.DataParallel(model.features)
        model.to(device)
    else:
        model = torch.nn.DataParallel(model).to(device)

    cudnn.benchmark = True

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(traindir, transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=bs, shuffle=True,
        num_workers=2, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=bs, shuffle=False,
        num_workers=2, pin_memory=True)

    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                global_lr,
                                momentum=0.9,
                                weight_decay=0.0001)

    for epoch in range(start_epoch, num_epochs):
        adjust_learning_rate(optimizer, epoch, global_lr)

        train(train_loader, model, criterion, optimizer, epoch)

        prec1 = validate(val_loader, model, criterion)

        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': network,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best)


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        target = target.to(device)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        output = model(input_var)
        loss = criterion(output, target_var)

        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_interval == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.to(device)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        with torch.no_grad():
            output = model(input_var)
            loss = criterion(output, target_var)

        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_interval == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg

if __name__ == '__main__':
    main()