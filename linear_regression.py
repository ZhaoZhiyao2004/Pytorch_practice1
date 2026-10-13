import torch

# 1. 准备数据：3 个样本，每个样本有 1 个输入特征
x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[2.0], [4.0], [6.0]])


# 2. 定义模型
class LinearModel(torch.nn.Module):
    def __init__(self):
        # 初始化父类，让模型具备 PyTorch 的模块管理功能
        super(LinearModel, self).__init__()

        # 创建线性层：每个样本输入 1 个数，输出 1 个数
        # 层内部保存需要学习的参数 w 和 b
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        # 使用当前的 w、b 计算预测值
        y_pred = self.linear(x)
        return y_pred


# 创建一个模型对象
model = LinearModel()

# 3. 设置损失函数和优化器
criterion = torch.nn.MSELoss(reduction="sum")
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 4. 训练 1000 轮
for epoch in range(1000):
    y_pred = model(x_data)          # 计算预测值
    loss = criterion(y_pred, y_data)  # 衡量预测误差
    print(epoch, loss.item())

    optimizer.zero_grad()          # 清除上一轮的梯度
    loss.backward()                # 计算这一轮的梯度
    optimizer.step()               # 根据梯度更新 w 和 b

# 查看学到的参数
print("w =", model.linear.weight.item())
print("b =", model.linear.bias.item())

# 使用训练后的模型预测
x_test = torch.Tensor([[4.0]])
y_test = model(x_test)
print("y_pred =", y_test.item())