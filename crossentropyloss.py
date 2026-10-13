import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# 1. 准备数据
batch_size = 64

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(
    root="./dataset/mnist",
    train=True,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0
)

test_dataset = datasets.MNIST(
    root="./dataset/mnist",
    train=False,
    download=True,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)


# 2. 定义模型
class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.l1 = torch.nn.Linear(784, 512)
        self.l2 = torch.nn.Linear(512, 256)
        self.l3 = torch.nn.Linear(256, 128)
        self.l4 = torch.nn.Linear(128, 64)
        self.l5 = torch.nn.Linear(64, 10)

    def forward(self, x):
        # 每张图片展开成包含 784 个数的一行
        x = x.view(-1, 784)

        # 前四个线性层后接 ReLU
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))

        # 最后一层直接输出 10 个类别分数
        return self.l5(x)


model = Net()


# 3. 定义损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()

optimizer = optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.5
)


# 4. 训练函数
def train(epoch):
    model.train()
    running_loss = 0.0

    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data

        # 清除旧梯度
        optimizer.zero_grad()

        # 预测并计算损失
        outputs = model(inputs)
        loss = criterion(outputs, target)

        # 计算梯度并更新参数
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # 每处理 300 批数据，打印一次平均损失
        if batch_idx % 300 == 299:
            print(
                f"[{epoch + 1}, {batch_idx + 1:5d}] "
                f"loss: {running_loss / 300:.3f}"
            )
            running_loss = 0.0


# 5. 测试函数
def test():
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data in test_loader:
            images, labels = data

            outputs = model(images)

            # 每行分数最高的位置就是预测类别
            _, predicted = torch.max(outputs, dim=1)

            # 累计图片总数和预测正确的数量
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(
        f"Accuracy on test set: "
        f"{100 * correct / total:.2f}%"
    )


# 6. 执行训练和测试
if __name__ == "__main__":
    for epoch in range(10):
        train(epoch)
        test()