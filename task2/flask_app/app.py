import os

import torch
import torchvision.transforms as transforms

import numpy as np
from net import ConvNet
from PIL import Image, ImageOps

from flask import Flask, render_template, request, send_from_directory

app = Flask(__name__)

dir_path = os.path.dirname(os.path.realpath(__file__))
STATIC_FOLDER = "static"
UPLOAD_FOLDER = f"{STATIC_FOLDER}/uploads"

# Load model
checkpoint_path = f'{STATIC_FOLDER}/models/ckpt.pth'
net = ConvNet()
net.load_state_dict(torch.load(checkpoint_path))
net.eval()

data_transforms = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))])

# Predict & classify image
def classify(image_path):

    label = ['Vertical Bar', 'Horizontal Bar', 'Line Plot', 'Pie Chart', 'Dotted Line']

    img = ImageOps.grayscale(Image.open(image_path))
    input_tensor = data_transforms(img)

    output = net(input_tensor.unsqueeze(0))
    result = label[torch.argmax(output)]

    return result

# home page
@app.route("/")
def home():
    return render_template("home.html")


@app.route("/classify", methods=["POST", "GET"])
def upload_file():

    if request.method == "GET":
        return render_template("home.html")

    else:
        file = request.files["image"]
        upload_image_path = os.path.join(UPLOAD_FOLDER, file.filename)
        print(upload_image_path)
        file.save(upload_image_path)

        result = classify(image_path=upload_image_path)

    return render_template(
        "classify.html", image_file_name=file.filename, label=result, prob=100
    )


@app.route("/classify/<filename>")
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


if __name__ == "__main__":
    app.debug = True
    app.run(debug=True)
    app.debug = True