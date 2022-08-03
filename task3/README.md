## Task 3

The above code performs fine-tuning of pretrained models on the given dataset to get better prediction accuracies.

### Visualisation via jupyter
* The notebooks under `experimental` directory implement the AlexNet and VGG16 pretrained models.
* Apart from the network architecture, another difference between the two implementation is that AlexNet uses k-fold cross validation
* The accuracy and loss plots are obtained by running each of the notebooks.

### Testing across more models
* The scripts under `pretrained` directory implement 3 pretrained models (vgg, resnet and alexnet).
* Choose the desired network by changing the `network` variable inside `train.py`.
* This code tries to capture precision@k (top1 and top5) as the loss metric in each of the models.