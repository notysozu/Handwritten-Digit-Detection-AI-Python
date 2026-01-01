import torch
import torch.nn as nn
import torch.nn.functional as F


class MNISTCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class MNISTModel:
    def __init__(self):
        self.model = MNISTCNN()
        self.model.load_state_dict(
            torch.load("model/mnist_cnn.pth", map_location="cpu")
        )
        self.model.eval()

    def predict(self, image_tensor):
        with torch.no_grad():
            out = self.model(image_tensor)
            probs = F.softmax(out, dim=1)
            conf, pred = torch.max(probs, dim=1)
        return pred.item(), conf.item()
