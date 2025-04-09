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

#一次性取1000点
data_PINN = dde.data.TimePDE(geomtime,
                        PDE, 
                        [],
                        num_domain=1000, 
                        num_test=200,
                        train_distribution="uniform")

net = dde.maps.FNN([2]+[64]*3+[1], "tanh", "Glorot normal")
net.apply_output_transform(output_transform)

model=dde.Model(data_PINN, net)
model.compile("adam", lr=1e-3)

model.train(iterations=1000,display_every=50)


#取点画图
data = np.load("Burgers.npz")
t, x, exact = data["t"], data["x"], data["usol"].T
xx, tt = np.meshgrid(x,t)
X = np.vstack((np.ravel(xx), np.ravel(tt))).T
y = exact.flatten()[:, None]

y_pred = model.predict(X)

import matplotlib.pyplot as plt

plt.figure(figsize=(6, 6))
plt.scatter(y, y_pred, s=1, alpha=0.5, label="Prediction")
plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--", label="Ideal: y=x")
plt.xlabel("True u(x, t)")
plt.ylabel("Predicted u(x, t)")
plt.title("Predicted vs True Values")
plt.legend()
plt.grid(True)

#输出准确率
from sklearn.metrics import r2_score
r2 = r2_score(y, y_pred)
print(f"R² score: {r2:.4f}")
plt.tight_layout()
plt.show()
