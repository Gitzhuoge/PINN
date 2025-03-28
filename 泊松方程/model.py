import tensorflow.compat.v1
import numpy as np
import deepxde as dde 
import matplotlib.pyplot as plt

#定义精确解
def u(inputs, outputs, X):
    
    x = inputs[:,:1]
    y = inputs[:,1:]
    
    loss_u = 1/3 * (x**3 + y**3)-outputs#输出预测结果与边界要求的残差
    return loss_u

def exact_solution(x):
    return (x[:, 0:1]**3 + x[:, 1:2]**3) / 3

#定义拉PDE函数
def PDE(inputs, outputs):
    #定义源方程
    def f(x, y):
        return 2*x + 2*y
    
    u_xx = dde.grad.hessian(outputs, inputs,component=0, i=0, j=0)
    u_yy = dde.grad.hessian(outputs, inputs,component=0, i=1, j=1)
    
    x = inputs[:,:1]
    y = inputs[:,1:]
    
    loss_PDE = u_xx+u_yy-f(x,y)#输出预测结果与PDE要求的残差
    return [loss_PDE]

geom = dde.geometry.Rectangle([0,0], [1,1])#定义矩形

dx = 0.1#定义取点间距（均匀取点）

#定义网络
net = dde.maps.PFNN([2]+[40]*2+[1],"tanh","Glorot normal")
BC_loss = [dde.OperatorBC(geom, u, lambda x, _: geom.on_boundary(x))]#得到u中计算的损失
#BC_loss = dde.DirichletBC(geom, exact_solution, lambda x, on_boundary: not on_boundary)#会计算实际解和精确解的差

#定义处理对象
data = dde.data.PDE(geom,
                    PDE,
                    BC_loss,
                    num_domain=int(geom.area/ dx**2),
                    num_boundary=int(geom.perimeter/dx),
                    num_test=10000,
                    train_distribution="pseudo")
#定义网络
model = dde.Model(data, net)
loss = ["MSE","MSE"]#这里第一个MSE是PDE方程里定义的loss平方，第二个是u里定义的loss的平方

model.compile("adam", lr = 0.001, loss=loss,loss_weights=[1,2],metrics=["l2 relative error"])

losshistory, train_state = model.train(epochs=20000)
