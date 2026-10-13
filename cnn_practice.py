import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = torch.nn.Conv2d(
            1, 10, kernel_size=5
        )
        self.conv2 = torch.nn.Conv2d(
            10, 20, kernel_size=5
        )
        self.pooling = torch.nn.MaxPool2d(2)
        self.fc = torch.nn.Linear(320, 10)

    def forward(self, x):
        batch_size = x.size(0)

        x = F.relu(self.pooling(self.conv1(x)))
        x = F.relu(self.pooling(self.conv2(x)))

        x = x.view(batch_size, -1)
        x = self.fc(x)

        return x


def train(epoch):
    running_loss = 0.0

    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        inputs = inputs.to(device)
        target = target.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, target)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if batch_idx % 300 == 299:
            print(
                '[%d, %5d] loss: %.3f'
                % (
                    epoch + 1,
                    batch_idx + 1,
                    running_loss / 300
                )
            )
            running_loss = 0.0


def test():
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, target in test_loader:
            inputs = inputs.to(device)
            target = target.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, dim=1)

            total += target.size(0)
            correct += (predicted == target).sum().item()

    print(
        'Accuracy on test set: %.2f %% [%d/%d]'
        % (
            100 * correct / total,
            correct,
            total
        )
    )


if __name__ == '__main__':
    batch_size = 64

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(
        root='./dataset/mnist',
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root='./dataset/mnist',
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    device = torch.device(
        'cuda:0' if torch.cuda.is_available() else 'cpu'
    )

    model = Net().to(device)

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = optim.SGD(
        model.parameters(),
        lr=0.01,
        momentum=0.5
    )

    for epoch in range(10):
        train(epoch)
        test()