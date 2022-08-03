## Task2

The above code creates a 2-layer CNN and trains on the given dataset.

### Usage
* Adjust the dataset path in main.py
* If needed, change the batch size, number of epochs and number of splits in the main.py file
* Run the training
    ```bash
    cd task2
    python main.py
    ```
* Monitor the training progress live by using tensorboard. Execute the following command and open [localhost:6006](http://127.0.0.1:6006/) in the browser
    ```bash
    tensorboard --logdir runs
    ```
* Trained model is saved in the same hierarchy under the name 'ckpt.pth'
* Use the above model to inference and compute the test accuracy
    ```bash
    cd task2
    python test.py
    ```

### Instructions on running the flask app
A flask application is developed to serve as an endpoint for effortless testing of the above model

- Run the application by issuing the following command
    ```bash
    cd task2/flask_app
    python app.py
    ```
- Visit [localhost:5000](http://127.0.0.1:5000/) to view the home page of the application. Upload the image to get the predictions.

### Visualising plots
Plotting inline in scripts is cumbersome for which, a jupyter notebook has been provided with the plots of accuracy and loss for each fold (in k-fold validation).