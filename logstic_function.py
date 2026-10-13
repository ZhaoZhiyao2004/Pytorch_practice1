import torch
import numpy as np
import matplotlib.pyplot as plt

# 1. 准备训练数据
x_data = torch.tensor([[1.0], [2.0], [3.0]])
y_data = torch.tensor([[0.0], [0.0], [1.0]])


# 2. 定义模型
class LogisticRegressionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


model = LogisticRegressionModel()

# 3. 定义损失函数和优化器
criterion = torch.nn.BCELoss(reduction="sum")
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 4. 训练模型
for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 5. 测试：输入 4.0
x_test = torch.tensor([[4.0]])

with torch.no_grad():
    y_test = model(x_test)

print("学习4小时的通过概率：", y_test.item())

# 6. 画图：预测 0～10 小时对应的通过概率
x = np.linspace(0, 10, 200)
x_t = torch.tensor(x, dtype=torch.float32).view(200, 1)

with torch.no_grad():
    y_t = model(x_t)

y = y_t.cpu().numpy()

plt.plot(x, y)
plt.axhline(y=0.5, color="r")
plt.xlabel("Hours")
plt.ylabel("Probability of Pass")
plt.grid()
plt.show()