import numpy as np
import deepxde as dde
import torch
dde.config.set_default_float("float64")

def PDE(inputs, outputs):

    u = outputs[:,0:1]
    u_t = dde.grad.jacobian(outputs, inputs, i=0, j=1)
    u_x = dde.grad.jacobian(outputs, inputs, i=0, j=0)
    u_xx = dde.grad.hessian(outputs, inputs, component=0, i=0, j=0)
   
    loss_pde = u_t+u*u_x-0.01/np.pi*u_xx
    return loss_pde


def output_transform(inputs, outputs):
    return -torch.sin(np.pi*inputs[:,0:1]) + (1-inputs[:,0:1]**2)*inputs[:,1:2]*outputs[:, 0:1]

geom = dde.geometry.Interval(-1,1)
time = dde.geometry.TimeDomain(0,1)
geomtime = dde.geometry.GeometryXTime(geom,time)
data_PINN = dde.data.TimePDE(geomtime,
                        PDE, 
                        [],
                        num_domain=500, 
                        num_test=200,
                        train_distribution="uniform")

net = dde.maps.FNN([2]+[64]*3+[1], "tanh", "Glorot normal")
net.apply_output_transform(output_transform)

model=dde.Model(data_PINN, net)
model.compile("adam", lr=1e-3)

model.train(iterations=1000,display_every=100)

# -------------------
# 混合采样 + 训练（目的是保持全局泛化）
# -------------------
k = 2
c = 1

for i in range(10):
    # 1. uniform 随机采样 150 个点
    X_uniform = geomtime.random_points(150)

    # 2. 从 1000 个候选点中挑出残差大的点（残差加权采样 50 个）
    X_candidate = geomtime.random_points(1000)
    Y_residual = np.abs(model.predict(X_candidate, operator=PDE)).astype(np.float64)
    weights = np.power(Y_residual, k) / np.power(Y_residual, k).mean() + c
    prob_dist = (weights / np.sum(weights))[:, 0]  # 转成概率分布
    selected_ids = np.random.choice(len(X_candidate), size=50, replace=False, p=prob_dist)
    X_highres = X_candidate[selected_ids]

    # 3. 混合两个集合，替换原训练点
    X_combined = np.vstack([X_uniform, X_highres])
    data_PINN.replace_with_anchors(X_combined)

    # 继续训练模型（不重新 compile）
    model.train(iterations=1000, display_every=100)
