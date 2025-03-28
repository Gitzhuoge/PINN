import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 

#偏微分方程条件
def f(x, y): 
    return (2*x+2*y)

#精确解
def exact(x,y):
    return 1/3*(x**3+y**3)

#取点
def points(n): 
    x=tf.constant(np.random.uniform(0,1,(n,1)),dtype=tf.float32)
    y=tf.constant(np.random.uniform(0,1,(n,1)),dtype=tf.float32)
    return x,y

# 定义PINN网络

class PINN(tf.keras.Model): 
    def __init__(self):
        super(PINN,self).__init__()
        self.inputs = tf.keras.layers.Input(shape=(2,))
        self.dense1 = tf.keras.layers.Dense(40,activation="tanh")
        self.dense2 = tf.keras.layers.Dense(40,activation="tanh")
        self.dense3 = tf.keras.layers.Dense(1,activation=None)
    
    def call(self, inputs): 
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.dense3(x)
    
    def laplacian(self, x, y):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)  # 确保二阶梯度计算时 x 仍然在计算图中
            tape.watch(y)
            with tf.GradientTape(persistent=True) as tape2:
                tape2.watch(x)  # 确保二阶梯度计算时 x 仍然在计算图中
                tape2.watch(y)
                inputs = tf.concat([x, y], axis=1)
                u = self.call(inputs)
            u_x = tape2.gradient(u, x)
            u_y = tape2.gradient(u, y)
    
        u_xx = tape.gradient(u_x, x)
        u_yy = tape.gradient(u_y, y)
    
        if u_xx is None or u_yy is None:
            raise ValueError("二阶导数计算失败，请检查梯度追踪")
    
        laplacian_u = u_xx + u_yy
        return laplacian_u


# 定义损失

@tf.function#会对程序进行优化减少训练速度
def loss(model, x, y, pde_coff=1.0, boundary_coeff=5.0):
    
    with tf.GradientTape(persistent=True) as tape: 
        tape.watch(x)
        tape.watch(y)
        inputs = tf.concat([x,y], axis=1)
        predicted_solution = model.call(inputs)
        pde = model.laplacian(x,y)-f(x,y)#定义偏微分差
    
    
    bc = predicted_solution-exact(x,y)#定义边界差
    
    mse_pde = tf.reduce_mean(tf.square(pde))
    mse_bc = tf.reduce_mean(tf.square(bc))

    loss_pde = pde_coff*mse_pde
    loss_bc = boundary_coeff*mse_bc
    loss = loss_bc+loss_pde

    return loss, loss_pde, loss_bc

np.random.seed(369)
n = 2000

x_train = tf.constant(np.random.uniform(0,1,(n,1)),dtype=tf.float32)
y_train = tf.constant(np.random.uniform(0,1,(n,1)),dtype=tf.float32)

model = PINN()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
iteration = 2001

# 迭代求解

for i in range(iteration): 
    
    with tf.GradientTape() as tape:
        loss_v = loss(model,x_train,y_train)
    grads = tape.gradient(loss_v[0],model.trainable_variables)
   
    if i%100 == 0: #每100次增加200个点，增强泛化
        print(f"iteration:{i}, losspde:{loss_v[1]},lossbc:{loss_v[2]}")
        addition_points = 200
        x_rand,y_rand = points(addition_points)
        x_train,y_train = tf.concat([x_train,x_rand],axis=0),tf.concat([y_train,y_rand],axis=0)

    optimizer.apply_gradients(zip(grads, model.trainable_variables))   

#画图预测值
num = 1
x = np.linspace(0, num, 100)  # 100 个点，范围 [0,1]
y = np.linspace(0, num, 100)
X,Y = np.meshgrid(x, y)  # 生成网格
points = np.vstack([X.ravel(), Y.ravel()]).T  # 变成 (N, 2) 形状的数据点
u_pred = model.predict(points).reshape(X.shape)
plt.figure(figsize=(6, 5))
plt.contourf(X, Y, u_pred, levels=100, cmap="jet")  # 使用等高线填充绘制热图
plt.colorbar(label="Predicted u(x, y)")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Heatmap of Predicted Solution")
plt.show()

#画图真实值
def l(inputs):
    x = inputs[:,:1]
    y = inputs[:,1:]
    return 1/3 * (x**3 + y**3)
points = np.vstack([X.ravel(), Y.ravel()]).T  # 变成 (N, 2) 形状的数据点
u_pred = l(points).reshape(X.shape)
plt.figure(figsize=(6, 5))
plt.contourf(X, Y, u_pred, levels=100, cmap="jet")  # 使用等高线填充绘制热图
plt.colorbar(label=" u(x, y)")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Heatmap of correct Solution")
plt.show()
