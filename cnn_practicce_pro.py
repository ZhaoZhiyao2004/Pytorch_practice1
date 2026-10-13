import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # 第一层卷积：输入1个通道，输出10个通道，卷积核大小为5×5
        self.conv1 = torch.nn.Conv2d(
            1, 10, kernel_size=5
        )

        # 第二层卷积：接收上一层的10个通道，输出20个通道
        self.conv2 = torch.nn.Conv2d(
            10, 20, kernel_size=5
        )

        # 最大池化：每个2×2区域取最大值，默认步长为2
        self.pooling = torch.nn.MaxPool2d(2)

        # 两次卷积和池化后，每张图剩下20×4×4＝320个特征
        # 将320个特征转换为数字0～9的10个类别分数
        self.fc = torch.nn.Linear(320, 10)

    def forward(self, x):
        # 输入形状：(N, 1, 28, 28)
        # N是当前这一批的图片数量
        batch_size = x.size(0)

        # 从里向外执行：卷积 → 池化 → ReLU
        # (N, 1, 28, 28) → (N, 10, 24, 24) → (N, 10, 12, 12)
        # ReLU将负值变成0，引入非线性，不改变形状
        x = F.relu(self.pooling(self.conv1(x)))

        # (N, 10, 12, 12) → (N, 20, 8, 8) → (N, 20, 4, 4)
        x = F.relu(self.pooling(self.conv2(x)))

        # 保留图片数量N，把每张图的特征展开
        # -1自动计算为320，形状变成(N, 320)
        x = x.view(batch_size, -1)

        # 输出形状：(N, 10)，每张图片有10个类别分数
        # 后面使用CrossEntropyLoss，因此这里不添加Softmax
        x = self.fc(x)

        return x


def train(epoch):
    # 累计一段时间内的损失，用于打印观察训练情况
    running_loss = 0.0

    # 每次取出一批数据，并获得从0开始的批次编号
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data

        # 图片和标签都必须与模型处于同一设备
        inputs = inputs.to(device)
        target = target.to(device)

        # 1. 清除上一批的梯度，防止梯度累加
        optimizer.zero_grad()

        # 2. 前向传播，得到每张图片的10个类别分数
        outputs = model(inputs)

        # 3. 比较预测分数与真实标签，计算损失
        loss = criterion(outputs, target)

        # 4. 反向传播：计算损失对模型参数的梯度
        loss.backward()

        # 5. 根据梯度更新模型参数
        optimizer.step()

        # item()将只有一个元素的损失张量转换为普通数值
        running_loss += loss.item()

        # 编号从0开始，299表示已经处理了300批
        # 每300批打印一次这300批的平均损失
        if batch_idx % 300 == 299:
            print(
                '[%d, %5d] loss: %.3f'
                % (
                    epoch + 1,
                    batch_idx + 1,
                    running_loss / 300
                )
            )

            # 重新开始统计下一个300批的损失
            running_loss = 0.0


def test():
    correct = 0  # 累计预测正确的图片数量
    total = 0    # 累计测试过的图片数量

    # 测试不需要反向传播，因此关闭梯度记录
    with torch.no_grad():
        for inputs, target in test_loader:
            inputs = inputs.to(device)
            target = target.to(device)

            # 输出形状：(N, 10)
            outputs = model(inputs)

            # dim=1：在每张图片的10个类别分数中寻找最大值
            # 返回最大分数和对应的类别编号
            # _忽略最大分数，predicted保存预测类别，形状为(N,)
            _, predicted = torch.max(outputs, dim=1)

            # target形状为(N,)，size(0)就是本批图片数量
            total += target.size(0)

            # 逐个比较预测类别和真实标签
            # sum()统计正确数量，item()将结果转换为普通数值
            correct += (predicted == target).sum().item()

    # 正确率＝预测正确的数量÷测试总数量×100%
    print(
        'Accuracy on test set: %.2f %% [%d/%d]'
        % (
            100 * correct / total,
            correct,
            total
        )
    )


# 直接运行本文件时，才执行下面的数据准备和训练
if __name__ == '__main__':
    # 每批最多处理64张图片
    batch_size = 64

    # Compose让图片按顺序经过以下两种处理
    transform = transforms.Compose([
        # 转成张量，并将像素值从0～255缩放到0～1
        # 单张图片的形状变为(1, 28, 28)
        transforms.ToTensor(),

        # 标准化：将像素值减去0.1307，再除以0.3081
        # 只有一个灰度通道，因此均值和标准差各提供一个数
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 准备训练集：train=True选择训练数据
    # download=True：本地没有数据时自动下载
    train_dataset = datasets.MNIST(
        root='./dataset/mnist',
        train=True,
        download=True,
        transform=transform
    )

    # 准备测试集：用于检查模型识别效果
    test_dataset = datasets.MNIST(
        root='./dataset/mnist',
        train=False,
        download=True,
        transform=transform
    )

    # 将训练数据分批，每次遍历时打乱样本顺序
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    # 测试数据也分批，但不需要打乱顺序
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    # CUDA可用时选择第一个可见的CUDA设备，否则使用CPU
    device = torch.device(
        'cuda:0' if torch.cuda.is_available() else 'cpu'
    )

    # 创建CNN模型，并把模型参数放到选定的设备
    model = Net().to(device)

    # 多分类交叉熵损失
    # 接收(N, 10)的类别分数和(N,)的真实类别标签
    criterion = torch.nn.CrossEntropyLoss()

    # SGD优化器：负责更新model中的可学习参数
    # lr是学习率，控制更新步幅；momentum是动量系数
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.01,
        momentum=0.5
    )

    # 训练10轮，每轮遍历一次训练集，然后测试一次
    for epoch in range(10):
        train(epoch)
        test()