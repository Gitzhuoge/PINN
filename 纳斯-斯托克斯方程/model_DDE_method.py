import torch
import numpy as np
import deepxde as dde 
dde.backend.set_default_backend("pytorch")
import matplotlib.pyplot as plt
import scipy.io
from sklearn.model_selection import train_test_split

#提取有限元仿真数据
data = scipy.io.loadmat('cylinder_nektar_wake.mat')
           
U_star = data['U_star'] # N x 2 x T
P_star = data['p_star'] # N x T
t_star = data['t'] # T x 1
X_star = data['X_star'] # N x 2

N = X_star.shape[0]
T = t_star.shape[0]

# Rearrange Data 
XX = np.tile(X_star[:,0:1], (1,T)) # N x T
YY = np.tile(X_star[:,1:2], (1,T)) # N x T
TT = np.tile(t_star, (1,N)).T # N x T

UU = U_star[:,0,:] # N x T
VV = U_star[:,1,:] # N x T
PP = P_star # N x T

x = XX.flatten()[:,None] # NT x 1
y = YY.flatten()[:,None] # NT x 1
t = TT.flatten()[:,None] # NT x 1

u = UU.flatten()[:,None] # NT x 1
v = VV.flatten()[:,None] # NT x 1
p = PP.flatten()[:,None] # NT x 1


#最终的总数据
data1 = np.concatenate([x,y,t,u,v,p],1)
data2 = data1[:, :][data1[:, 2] <= 7]
data3 = data2[:, :][data2[:, 0] >= 1]
data4 = data3[:, :][data3[:, 0] <= 8]
data5 = data4[:, :][data4[:, 1] >= -2]
data_total = data5[:, :][data5[:, 1] <= 2]

#从里面随机抽取2000个数,idx是索引
idx = np.random.choice(data_total.shape[0], 2000, replace=False)

#分割训练集和测试集
data = data_total[idx,:]
train_data, test_data = train_test_split(data, test_size=0.2, random_state=369)

train_input, train_output = train_data[:,:3], train_data[:, 3:]
test_input, test_output = test_data[:,:3], test_data[:, 3:]

# 定义PDE函数，PDE要调用NN中的lambda参数

lambda_1 = dde.Variable(0.0)
lambda_2 = dde.Variable(0.0)

def loss_PDE(inputs, outputs):
    
    u = outputs[:,0:1]
    v = outputs[:,1:2]
    
    
    #u导数
    u_x = dde.grad.jacobian(outputs, inputs, i=0, j=0)
    u_y = dde.grad.jacobian(outputs, inputs, i=0, j=1)
    u_t = dde.grad.jacobian(outputs, inputs, i=0, j=2)
    u_xx = dde.grad.hessian(outputs, inputs, component=0, i=0, j=0)
    u_yy = dde.grad.hessian(outputs, inputs, component=0, i=1, j=1)
    
    
    #v导数
    v_x = dde.grad.jacobian(outputs, inputs, i=1, j=0)
    v_y = dde.grad.jacobian(outputs, inputs, i=1, j=1)
    v_t = dde.grad.jacobian(outputs, inputs, i=1, j=2)
    v_xx = dde.grad.hessian(outputs, inputs, component=1, i=0, j=0)
    v_yy = dde.grad.hessian(outputs, inputs, component=1, i=1, j=1)

    
    #p导数
    p_x = dde.grad.jacobian(outputs, inputs, i=2, j=0)
    p_y = dde.grad.jacobian(outputs, inputs, i=2, j=1)
    
    
    #PDE损失
    loss_x = u_t+lambda_1*(u*u_x+v*u_y)+p_x-lambda_2*(u_xx+u_yy)
    loss_y = v_t+lambda_1*(u*v_x+v*v_y)+p_y-lambda_2*(v_xx+v_yy)
    loss_persistent = u_x+v_y
    
    return loss_x, loss_y, loss_persistent
    


#定义离散点损失

bc_u = dde.icbc.PointSetBC(train_input, train_output[:,0:1], component=0)
bc_v = dde.icbc.PointSetBC(train_input, train_output[:,1:2], component=1)
bc_p = dde.icbc.PointSetBC(train_input, train_output[:,2:3], component=2)

# 定义空间域
geom = dde.geometry.Rectangle([1.0,-2.0], [8.0,2.0])

#定义时间域
time = dde.geometry.TimeDomain(0, 7)

#时空域
geomtime = dde.geometry.GeometryXTime(geom, time)



#组建模型
layers = [3]+[50]*6+[3]

NN_net = dde.nn.FNN(layers, "tanh", "Glorot normal")

dde_data = dde.data.TimePDE(
        geomtime, 
        loss_PDE, 
        [bc_u,bc_v,bc_p], 
        num_domain=700,
        num_boundary=200, 
        num_initial=100,
        anchors = train_input
)

model = dde.Model(dde_data, NN_net)


#储存和检测lambda1，lambda2
fnamevar = "variables.txt"
variable = dde.callbacks.VariableValue([lambda_1, lambda_2], period=10, filename=fnamevar)


model.compile("adam", lr = 0.001, external_trainable_variables=[lambda_1, lambda_2])

losshistory, train_state = model.train(iterations=500,callbacks=[variable], display_every=10,disregard_previous_best=True)

# 使用测试集输入数据获取模型的预测值
test_pred = model.predict(test_input)

# 提取预测值和真实值
u_pred = test_pred[:, 0]
v_pred = test_pred[:, 1]
p_pred = test_pred[:, 2]

u_true = test_output[:, 0]
v_true = test_output[:, 1]
p_true = test_output[:, 2]

# 绘制预测值与真实值的相关曲线
plt.figure(figsize=(12, 4))

# 绘制u分量的预测值与真实值
plt.subplot(1, 3, 1)
plt.scatter(u_true, u_pred, alpha=0.5)
plt.plot([u_true.min(), u_true.max()], [u_true.min(), u_true.max()], 'r--')
plt.xlabel('True u')
plt.ylabel('Predicted u')
plt.title('u Prediction vs True')


# 绘制v分量的预测值与真实值
plt.subplot(1, 3, 2)
plt.scatter(v_true, v_pred, alpha=0.5)
plt.plot([v_true.min(), v_true.max()], [v_true.min(), v_true.max()], 'r--')
plt.xlabel('True v')
plt.ylabel('Predicted v')
plt.title('v Prediction vs True')

# 绘制压力p的预测值与真实值
plt.subplot(1, 3, 3)
plt.scatter(p_true, p_pred, alpha=0.5)
plt.plot([p_true.min(), p_true.max()], [p_true.min(), p_true.max()], 'r--')
plt.xlabel('True p')
plt.ylabel('Predicted p')
plt.title('p Prediction vs True')

plt.tight_layout()
plt.show()
