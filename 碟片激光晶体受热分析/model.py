import deepxde as dde
import numpy as np

# 定义 PDE
def PDE(inputs, outputs):
    T_xx = dde.grad.hessian(outputs, inputs, component=2, i=0, j=0)
    T_yy = dde.grad.hessian(outputs, inputs, component=2, i=1, j=1)
    T_x = dde.grad.jacobian(outputs, inputs, i=2, j=0)
    T_y = dde.grad.jacobian(outputs, inputs, i=2, j=1)
    T_t = dde.grad.jacobian(outputs, inputs, i=2, j=2)

    u_xx = dde.grad.hessian(outputs, inputs, component=0, i=0, j=0)
    u_yy = dde.grad.hessian(outputs, inputs, component=0, i=1, j=1)
    u_xy = dde.grad.hessian(outputs, inputs, component=0, i=0, j=1)
    u_x = dde.grad.jacobian(outputs, inputs, i=0, j=0)

    v_xx = dde.grad.hessian(outputs, inputs, component=1, i=0, j=0)
    v_yy = dde.grad.hessian(outputs, inputs, component=1, i=1, j=1)
    v_xy = dde.grad.hessian(outputs, inputs, component=1, i=0, j=1)
    v_y = dde.grad.jacobian(outputs, inputs, i=1, j=1)

    l_x = u_xx + v_xy
    l_y = u_xy + v_yy

    # 热传导
    loss_heat = T_xx + T_yy + Q * L**2 / lam / T_bc - rho * cp / lam * L**2 * T_t

    # u, v 热位移
    loss_u = u_xx + u_yy + (1 + mu) / (1 - mu) * l_x - 2 * (1 + mu) / (1 - mu) * a * T_bc * L / muc * T_x
    loss_v = v_xx + v_yy + (1 + mu) / (1 - mu) * l_y - 2 * (1 + mu) / (1 - mu) * a * T_bc * L / muc * T_y

    return loss_heat, loss_u, loss_v


# 定义参数
lam = 10
cp = 0.59
E = 3.1e10
rho = 4.56
mu = 0.3
a = 7.8e-6
n = 0.58
h = 2
r = 20
P = 100
Q = n * P / np.pi / r**2 / h

u_bc = 1e-8
v_bc = 1e-8
T_bc = 22
muc = n * P * a / 4 / np.pi / lam
L = 20

# 定义几何和时间域
geom = dde.geometry.geometry_2d.Disk([0, 0], 1)
time = dde.geometry.TimeDomain(0, 5)
geomtime = dde.geometry.GeometryXTime(geom, time)

# 定义边界条件

bc_u = dde.DirichletBC(geomtime, lambda x: u_bc, lambda x, on_boundary: on_boundary, component=0)
bc_v = dde.DirichletBC(geomtime, lambda x: v_bc, lambda x, on_boundary: on_boundary, component=1)
bc_T = dde.DirichletBC(geomtime, lambda x: T_bc, lambda x, on_boundary: on_boundary, component=2)

# 定义数据类
data = dde.data.TimePDE(
    geomtime,
    PDE,
    [bc_u, bc_v, bc_T],
    num_domain=1000,
    num_boundary=500
)

# 定义网络结构
activations = ["tanh"] * 6 + [None]  # 最后一层无激活
net = dde.maps.FNN([3] + [20] * 6 + [3], activations, "Glorot normal")

# 定义模型并编译
model = dde.Model(data, net)
model.compile("adam", lr=1e-3)

# 训练模型
losshistory, train_state = model.train(iterations=100, display_every=1)

dde.saveplot(losshistory, train_state, issave=True, isplot=False)


#画loss曲线
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# 定义科学计数法格式化函数
def sci_format(x, pos):
    return f"${1e-6 * x:.1e}$"

# 绘制损失曲线
plt.figure(figsize=(10, 6))

# 绘制总损失
plt.plot(losshistory.steps, losshistory.loss_train, label="Total Training Loss", color="blue", linestyle="-")
plt.plot(losshistory.steps, losshistory.loss_test, label="Total Test Loss", color="blue", linestyle="--")

# 分别绘制每个方程的损失
# 注意：losshistory.loss_train 和 losshistory.loss_test 是二维数组，每一列对应一个方程的损失
# 根据你的代码，PDE 返回了三个损失值：loss_heat, loss_u, loss_v
colors = ["red", "green", "purple"]  # 为每个方程分配颜色
labels = ["Heat Loss", "u Displacement Loss", "v Displacement Loss"]  # 每个损失的名称

for i in range(3):  # PDE 返回了三个损失值
    plt.plot(losshistory.steps, [loss[i] for loss in losshistory.loss_train], label=f"Train {labels[i]}", color=colors[i], linestyle="-")
    plt.plot(losshistory.steps, [loss[i] for loss in losshistory.loss_test], label=f"Test {labels[i]}", color=colors[i], linestyle="--")

# 添加图例和标签
plt.xlabel("Iteration")
plt.ylabel("Loss")

# 设置纵坐标为科学计数法
plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
plt.gca().yaxis.get_major_formatter().set_scientific(True)
plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))  # 强制使用科学计数法

plt.title("Loss Convergence")
plt.legend()
plt.grid(True)
plt.show()


#画不同时刻的预测值在空间分布
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

# 定义绘制热力图的函数
def plot_heat_map(model, geom, time_point):
    # 在几何域上生成网格点
    points = geom.random_points(5000)
    # 将时间点添加到网格点中
    points_with_time = np.hstack((points, np.full((points.shape[0], 1), time_point)))
    # 使用模型预测温度分布
    pre = model.predict(points_with_time)
    # 提取温度值
    T_values = pre[:, 2]

    # 创建三角剖分
    x = points[:, 0]
    y = points[:, 1]
    triangulation = Triangulation(x, y)

    # 绘制热力图
    plt.figure(figsize=(6, 6))
    plt.tricontourf(triangulation, T_values, levels=50, cmap='viridis')
    plt.colorbar(label='Temperature')
    plt.title(f'Temperature Distribution at Time {time_point}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

# 绘制不同时间点的热力图
time_points = [0, 5, 10]  # 定义要绘制的时间点
for time in time_points:
    plot_heat_map(model, geom, time)
