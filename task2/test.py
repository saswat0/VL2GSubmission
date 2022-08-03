import os

import torch
import torchvision.transforms as transforms

import numpy as np
from net import ConvNet
from PIL import Image, ImageOps

import matplotlib.pyplot as plt

checkpoint_path = 'ckpt.pth'
net = ConvNet()
net.load_state_dict(torch.load(checkpoint_path))
net.eval()

# test_image = Image.open(os.path.join('data/test', random.choice(os.listdir('data/test'))))
for f in os.listdir('data/test'):
    test_image = Image.open(os.path.join('data/test', f))
    plt.imshow(np.asarray(test_image))
    plt.show()

    label = ['vbar_categorical', 'hbar_categorical', 'line', 'pie', 'dot_line']
    data_transforms = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))])

    img = ImageOps.grayscale(test_image)
    input_tensor = data_transforms(img)

    net.eval()
    output = net(input_tensor.unsqueeze(0))

    print(label[torch.argmax(output)])