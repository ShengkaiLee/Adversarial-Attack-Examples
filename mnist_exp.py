import os

from absl import app, flags
from easydict import EasyDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from datasets import MNISTDataset
import matplotlib.pyplot as plt

from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import (
    projected_gradient_descent,
)

class CNN(torch.nn.Module):
    """Basic CNN architecture."""

    def __init__(self, in_channels=1):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, 64, 8, 1
        )  # (batch_size, 3, 28, 28) --> (batch_size, 64, 21, 21)
        self.conv2 = nn.Conv2d(
            64, 128, 6, 2
        )  # (batch_size, 64, 21, 21) --> (batch_size, 128, 8, 8)
        self.conv3 = nn.Conv2d(
            128, 128, 5, 1
        )  # (batch_size, 128, 8, 8) --> (batch_size, 128, 4, 4)
        self.fc1 = nn.Linear(
            128 * 4 * 4, 128
        )  # (batch_size, 128, 4, 4) --> (batch_size, 2048)
        self.fc2 = nn.Linear(128, 10)  # (batch_size, 128) --> (batch_size, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 128 * 4 * 4)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

def ld_mnist():
    """Load training and test data."""
    train_transforms = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor()]
    )
    test_transforms = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor()]
    )

    # Load MNIST dataset
    train_dataset = MNISTDataset(root="/tmp/data", transform=train_transforms)
    test_dataset = MNISTDataset(
        root="/tmp/data", train=False, transform=test_transforms
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=True, num_workers=2
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=128, shuffle=False, num_workers=2
    )
    return EasyDict(train=train_loader, test=test_loader)


def save_image(images, labels, name):
    num_row = 2
    num_col = 5
    # plot images

    fig, axes = plt.subplots(
        num_row, num_col, figsize=(1.5*num_col, 2*num_row))
    for i in range(10):
        ax = axes[i//num_col, i % num_col]
        ax.imshow(images[i][0], cmap='gray')
        ax.set_title('Label: {}'.format(labels[i]))
    plt.title(name)
    plt.tight_layout()
    plt.savefig(name + '.png')

def main():
    # Load training and test data
    data = ld_mnist()

    net = CNN(in_channels=1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        net = net.cuda()
    loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    net.load_state_dict(torch.load("mnist.pt", map_location=device))
    # Evaluate on clean and adversarial data
    net.eval()
    report = EasyDict(nb_test=0, correct=0, correct_fgm=0, correct_pgd=0,
                      correct_fgm_inf=0, correct_fgm_2=0, correct_pgd_inf=0, correct_pgd_2=0)
    images, labels = [], []
    images_fgm_inf, labels_fgm_inf = [], []
    images_fgm_2, labels_fgm_2 = [], []
    images_pgd_2, labels_pgd_2 = [], []
    images_pgd_inf, labels_pgd_inf = [], []
    for x, y in data.test:
        images, labels = x, y
        x, y = x.to(device), y.to(device)
        x_fgm_inf = fast_gradient_method(net, x, 0.2, np.inf)
        x_fgm_2 = fast_gradient_method(net, x, 2, np.inf)
        x_pgd_inf = projected_gradient_descent(net, x, 0.2, 0.01, 40, np.inf)
        x_pgd_2 = projected_gradient_descent(net, x, 2, 0.01, 40, np.inf)
        _, y_pred = net(x).max(1)
        _, y_pred_fgm_inf = net(x_fgm_inf).max(1)
        _, y_pred_fgm_2 = net(x_fgm_2).max(1)
        _, y_pred_pgd_inf = net(x_pgd_inf).max(1)
        _, y_pred_pgd_2 = net(x_pgd_2).max(1)

        images_fgm_inf, labels_fgm_inf = x_fgm_inf, y_pred_fgm_inf
        images_fgm_2, labels_fgm_2 = x_fgm_2, y_pred_fgm_2
        images_pgd_2, labels_pgd_2 = x_pgd_2, y_pred_pgd_2
        images_pgd_inf, labels_pgd_inf = x_pgd_inf, y_pred_pgd_inf

        report.nb_test += y.size(0)
        report.correct += y_pred.eq(y).sum().item()
        report.correct_fgm_inf += y_pred_fgm_inf.eq(y).sum().item()
        report.correct_fgm_2 += y_pred_fgm_2.eq(y).sum().item()
        report.correct_pgd_inf += y_pred_pgd_inf.eq(y).sum().item()
        report.correct_pgd_2 += y_pred_pgd_2.eq(y).sum().item()
    # print(x_fgm_2.shape)
    # print(x_pgd_2.shape)
    save_image(images, labels, 'cifar10_clean')
    save_image(images_fgm_inf, labels_fgm_inf, 'cifar10_fgm_inf')
    save_image(images_fgm_2, labels_fgm_2, 'cifar10_fgm_2')
    save_image(images_pgd_2, labels_pgd_2, 'cifar10_pgd_2')
    save_image(images_pgd_inf, labels_pgd_inf, 'cifar10_pgd_inf')

    print(
        "test acc on clean examples (%): {:.3f}".format(
            report.correct / report.nb_test * 100.0
        )
    )

    print(
        "test acc on FGM_inf adversarial examples (%): {:.3f}".format(
            report.correct_fgm_inf / report.nb_test * 100.0
        )
    )
    print(
        "test acc on FGM_2 adversarial examples (%): {:.3f}".format(
            report.correct_fgm_2 / report.nb_test * 100.0
        )
    )
    print(
        "test acc on PGD_inf adversarial examples (%): {:.3f}".format(
            report.correct_pgd_inf / report.nb_test * 100.0
        )
    )
    print(
        "test acc on PGD_2 adversarial examples (%): {:.3f}".format(
            report.correct_pgd_2 / report.nb_test * 100.0
        )
    )


if __name__ == "__main__":

    main()
