## Task 4

The above code performs ablation studies on pretrained VGG model to understand the regions of interest developed in the CNN while training. 

It also implements an attention based model to create heatmaps across various spatial-attention layers.

</br>

### Ablation Study
The notebook named `Ablation studies.ipynb` implements VGG16 pretrained mdoel and then displays the Class Activation Maps (CAM) after training.

</br>

### Attention Model
The directory named `attention` houses the code for classification using self-attention. The intermediate logs and CAMs are logged in tensorboard.

Monitor the training progress live by using tensorboard. Execute the following command and open [localhost:6006](http://127.0.0.1:6006/) in the browser.
```bash
tensorboard --logdir runs
```

The CAMs are available under the `IMAGES` header of the tensorboard page.