import torch
from PIL import Image
from torchvision import transforms

from model.mnist_model import MNISTModel

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


def predict_digit(image_path):
    """
    image_path: path to handwritten digit image
    returns: (predicted_digit, confidence)
    """
    image = Image.open(image_path)
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)
    model = MNISTModel()
    digit, confidence = model.predict(image_tensor)

    return digit, confidence
