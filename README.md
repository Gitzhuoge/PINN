# PINN
偏微分方程
## 涉及的偏微分方程和边界条件

### 涉及的偏微分方程

#### 动量方程

- **x 方向动量方程**：
  
  $$
  \frac{\partial u}{\partial t} + \lambda_1 \left( u \frac{\partial u}{\partial x} + v \frac{\partial u}{\partial y} \right) + \frac{\partial p}{\partial x} - \lambda_2 \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} \right) = 0
  $$

- **y 方向动量方程**：
  
  $$
  \frac{\partial v}{\partial t} + \lambda_1 \left( u \frac{\partial v}{\partial x} + v \frac{\partial v}{\partial y} \right) + \frac{\partial p}{\partial y} - \lambda_2 \left( \frac{\partial^2 v}{\partial x^2} + \frac{\partial^2 v}{\partial y^2} \right) = 0
  $$

#### 连续性方程（不可压缩条件）

$$
\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = 0
$$

### 与有限元仿真数据残差比较

$$
\mathcal{L}_{\text{data}} = \sum_{i} \left( (u_i - \hat{u}_i)^2 + (v_i - \hat{v}_i)^2 \right)
$$

### 物理约束的边界条件

- **物理方程的边界约束**：
  
  $$
  \mathcal{L}_{\text{pde}} = \sum_{i} \left( f_{u,i}^2 + f_{v,i}^2 \right)
  $$
  
  其中
  
  $$
  f_u = \frac{\partial u}{\partial t} + \lambda_1 \left( u \frac{\partial u}{\partial x} + v \frac{\partial u}{\partial y} \right) + \frac{\partial p}{\partial x} - \lambda_2 \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} \right)
  $$
  
  $$
  f_v = \frac{\partial v}{\partial t} + \lambda_1 \left( u \frac{\partial v}{\partial x} + v \frac{\partial v}{\partial y} \right) + \frac{\partial p}{\partial y} - \lambda_2 \left( \frac{\partial^2 v}{\partial x^2} + \frac{\partial^2 v}{\partial y^2} \right)
  $$

## 总的损失函数

代码中的总损失函数是数据损失和物理方程残差损失的组合：

$$
\mathcal{L} = \mathcal{L}_{\text{data}} + \mathcal{L}_{\text{pde}}
$$
