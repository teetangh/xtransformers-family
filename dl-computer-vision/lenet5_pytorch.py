from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.nn.modules.flatten import Flatten
import torchvision

# It is necessary to have both the model, and the data on the same device, either CPU or GPU, for the model to process data.
# Data on CPU and model on GPU, or vice-versa, will result in a Runtime error

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LenetModel(nn.Module):

    def __init__(self, n_classes):
        super(LenetModel, self).__init__()

        self.feature_extractor = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1)),
            ("tanh1", nn.Tanh()),

            ("avgpool1", nn.AvgPool2d(kernel_size=2, stride=2))
            ("tanh2", nn.Tanh()),

            ("conv2", nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)),
            ("tanh3", nn.Tanh()),

            ("avgpool2", nn.AvgPool2d(kernel_size=+2, stride=2))
            ("tanh4", nn.Tanh()),

            ("conv2", nn.Conv2d(in_channels=16,
             out_channels=120, kernel_size=5, stride=1)),
            ("tanh5", nn.Tanh()),

        ]))

        self.classifier = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(in_channels=120, out_channels=84, bias=True)),
            ("tanh6", nn.Tanh()),
            ("linear2", nn.Linear(in_channels=84, out_channels=n_classes, bias=True))
        ]))

    def forward(self, input):
        x = self.feature_extractor(input)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        probs = torch.functional.softmax(logits, dim=0)

        return probs


##############################################


def train():
    pass


def validate():
    pass


def training_loop():
    pass


##############################################

def imshow(img):
    img = img/2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, axes=(1, 2, 0)))
    plt.show()


def show_class_description(train_set, train_loader, batch_size):

    classes = train_set.classes
    print("classes: ", classes)

    class_count = {}
    for _, index in train_set:
        # print("temp, index ", temp, index)
        label = classes[index]
        if label not in class_count:
            class_count[label] = 0
        class_count[label] += 1
    print("class_count: ", class_count)

    dataiter = iter(train_loader)
    images, labels = dataiter.next()

    imshow(torchvision.utils.make_grid(images))
    print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))

##############################################


def main():

    batch_size = 4
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(size=(32, 32)),
            torchvision.transforms.Normalize((0.5,), (0.5,)),
            torchvision.transforms.ToTensor()
        ]
    )

    train_set = torchvision.datasets.MNIST(
        root="./data/", train=True, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=batch_size,  shuffle=True, num_workers=2)

    val_set = torchvision.datasets.MNIST(
        root="./data/", train=False, transform=transform, download=True)
    val_loader = torch.utils.data.DataLoader(
        dataset=val_set, batch_size=batch_size,  shuffle=True, num_workers=2)

    # classes = ('plane', 'car', 'bird', 'cat',
    #            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # imshow(train_set[0])

    show_class_description(train_set, train_loader, batch_size)


if __name__ == "__main__":
    main()
