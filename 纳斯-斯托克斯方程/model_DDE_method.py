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
data_total = np.concatenate([x,y,t,u,v,p],1)


#从里面随机抽取2000个数,idx是索引
idx = np.random.choice(data_total.shape[0], 2000, replace=False)
data = data_total[idx,:]


#分割训练集和测试集
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
bc_u = dde.PointSetBC(train_input, train_output[:,0], component=0)
bc_v = dde.PointSetBC(train_input, train_output[:,1], component=1)
bc_p = dde.PointSetBC(train_input, train_output[:,2], component=2)

#定义空间域
geom = dde.geometry.Rectangle([np.min(x), np.min(y)], [np.max(x), np.max(y)])


#定义时间域
time = dde.geometry.TimeDomain(np.min(t), np.max(t))

#时空域
geomtime = dde.geometry.GeometryXTime(geom, time)



#组建模型
layers = [3]+[20]*6+[3]

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

train_pred = model.predict(train_input)

# 分别取出预测值和真实值
u_pred, v_pred, p_pred = train_pred[:, 0], train_pred[:, 1], train_pred[:, 2]
u_true, v_true, p_true = train_output[:, 0], train_output[:, 1], train_output[:, 2]


#画图
def plot_comparison_at_time(t_selected):
    """在特定时间 t_selected 处绘制 u, v, p 的真实值、预测值和误差"""
    mask = np.isclose(train_input[:, 2], t_selected)  # 找到 t 接近 t_selected 的点
    x_selected = train_input[mask, 0]
    y_selected = train_input[mask, 1]

    u_pred_selected = u_pred[mask]
    u_true_selected = u_true[mask]

    v_pred_selected = v_pred[mask]
    v_true_selected = v_true[mask]

    p_pred_selected = p_pred[mask]
    p_true_selected = p_true[mask]

    # 计算误差
    u_error = np.abs(u_pred_selected - u_true_selected)
    v_error = np.abs(v_pred_selected - v_true_selected)
    p_error = np.abs(p_pred_selected - p_true_selected)

    fig, axes = plt.subplots(3, 3, figsize=(15, 9))

    def plot_subplot(ax, x, y, values, title):
        sc = ax.scatter(x, y, c=values, cmap="jet", alpha=0.7)
        plt.colorbar(sc, ax=ax)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(title)

    # u 真实值 vs 预测值 vs 误差
    plot_subplot(axes[0, 0], x_selected, y_selected, u_true_selected, f"True u at t={t_selected}")
    plot_subplot(axes[0, 1], x_selected, y_selected, u_pred_selected, f"Predicted u at t={t_selected}")
    plot_subplot(axes[0, 2], x_selected, y_selected, u_error, f"Error |u_pred - u_true|")

    # v 真实值 vs 预测值 vs 误差
    plot_subplot(axes[1, 0], x_selected, y_selected, v_true_selected, f"True v at t={t_selected}")
    plot_subplot(axes[1, 1], x_selected, y_selected, v_pred_selected, f"Predicted v at t={t_selected}")
    plot_subplot(axes[1, 2], x_selected, y_selected, v_error, f"Error |v_pred - v_true|")

    # p 真实值 vs 预测值 vs 误差
    plot_subplot(axes[2, 0], x_selected, y_selected, p_true_selected, f"True p at t={t_selected}")
    plot_subplot(axes[2, 1], x_selected, y_selected, p_pred_selected, f"Predicted p at t={t_selected}")
    plot_subplot(axes[2, 2], x_selected, y_selected, p_error, f"Error |p_pred - p_true|")

    plt.tight_layout()
    plt.show()

# 选择几个时间步
for t_plot in [0.0, 1.0, 2.0]:
    plot_comparison_at_time(t_plot)
