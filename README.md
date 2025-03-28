# 涉及的偏微分方程

## 1、纳斯-斯托克斯方程部分

### 涉及的偏微分方程

#### 动量方程

- **x 方向动量方程**：
  
  ![公式](https://latex.codecogs.com/svg.latex?%5Cfrac%7B%5Cpartial%20u%7D%7B%5Cpartial%20t%7D%20&plus;%20%5Clambda_1%20%5Cleft(%20u%20%5Cfrac%7B%5Cpartial%20u%7D%7B%5Cpartial%20x%7D%20&plus;%20v%20%5Cfrac%7B%5Cpartial%20u%7D%7B%5Cpartial%20y%7D%20%5Cright)%20&plus;%20%5Cfrac%7B%5Cpartial%20p%7D%7B%5Cpartial%20x%7D%20-%20%5Clambda_2%20%5Cleft(%20%5Cfrac%7B%5Cpartial%5E2%20u%7D%7B%5Cpartial%20x%5E2%7D%20&plus;%20%5Cfrac%7B%5Cpartial%5E2%20u%7D%7B%5Cpartial%20y%5E2%7D%20%5Cright)%20%3D%200)

- **y 方向动量方程**：
  
  ![公式](https://latex.codecogs.com/svg.latex?%5Cfrac%7B%5Cpartial%20v%7D%7B%5Cpartial%20t%7D%20&plus;%20%5Clambda_1%20%5Cleft(%20u%20%5Cfrac%7B%5Cpartial%20v%7D%7B%5Cpartial%20x%7D%20&plus;%20v%20%5Cfrac%7B%5Cpartial%20v%7D%7B%5Cpartial%20y%7D%20%5Cright)%20&plus;%20%5Cfrac%7B%5Cpartial%20p%7D%7B%5Cpartial%20y%7D%20-%20%5Clambda_2%20%5Cleft(%20%5Cfrac%7B%5Cpartial%5E2%20v%7D%7B%5Cpartial%20x%5E2%7D%20&plus;%20%5Cfrac%7B%5Cpartial%5E2%20v%7D%7B%5Cpartial%20y%5E2%7D%20%5Cright)%20%3D%200)

#### 连续性方程（不可压缩条件）

![公式](https://latex.codecogs.com/svg.latex?%5Cfrac%7B%5Cpartial%20u%7D%7B%5Cpartial%20x%7D%20&plus;%20%5Cfrac%7B%5Cpartial%20v%7D%7B%5Cpartial%20y%7D%20%3D%200)

### 与有限元仿真数据残差比较

![公式](https://latex.codecogs.com/svg.latex?%5Cmathcal%7BL%7D_%7B%5Ctext%7Bdata%7D%7D%20%3D%20%5Csum_%7Bi%7D%20%5Cleft(%20(u_i%20-%20%5Chat%7Bu%7D_i)%5E2%20&plus;%20(v_i%20-%20%5Chat%7Bv%7D_i)%5E2%20%5Cright))

### 物理约束的边界条件

- **物理方程的边界约束**：
  
  ![公式](https://latex.codecogs.com/svg.latex?%5Cmathcal%7BL%7D_%7B%5Ctext%7Bpde%7D%7D%20%3D%20%5Csum_%7Bi%7D%20%5Cleft(%20f_%7Bu,i%7D%5E2%20&plus;%20f_%7Bv,i%7D%5E2%20%5Cright))
  
  其中
  
  ![公式](https://latex.codecogs.com/svg.latex?f_u%20%3D%20%5Cfrac%7B%5Cpartial%20u%7D%7B%5Cpartial%20t%7D%20&plus;%20%5Clambda_1%20%5Cleft(%20u%20%5Cfrac%7B%5Cpartial%20u%7D%7B%5Cpartial%20x%7D%20&plus;%20v%20%5Cfrac%7B%5Cpartial%20u%7D%7B%5Cpartial%20y%7D%20%5Cright)%20&plus;%20%5Cfrac%7B%5Cpartial%20p%7D%7B%5Cpartial%20x%7D%20-%20%5Clambda_2%20%5Cleft(%20%5Cfrac%7B%5Cpartial%5E2%20u%7D%7B%5Cpartial%20x%5E2%7D%20&plus;%20%5Cfrac%7B%5Cpartial%5E2%20u%7D%7B%5Cpartial%20y%5E2%7D%20%5Cright))
  
  ![公式](https://latex.codecogs.com/svg.latex?f_v%20%3D%20%5Cfrac%7B%5Cpartial%20v%7D%7B%5Cpartial%20t%7D%20&plus;%20%5Clambda_1%20%5Cleft(%20u%20%5Cfrac%7B%5Cpartial%20v%7D%7B%5Cpartial%20x%7D%20&plus;%20v%20%5Cfrac%7B%5Cpartial%20v%7D%7B%5Cpartial%20y%7D%20%5Cright)%20&plus;%20%5Cfrac%7B%5Cpartial%20p%7D%7B%5Cpartial%20y%7D%20-%20%5Clambda_2%20%5Cleft(%20%5Cfrac%7B%5Cpartial%5E2%20v%7D%7B%5Cpartial%20x%5E2%7D%20&plus;%20%5Cfrac%7B%5Cpartial%5E2%20v%7D%7B%5Cpartial%20y%5E2%7D%20%5Cright))

### 总的损失函数

代码中的总损失函数是数据损失和物理方程残差损失的组合：

![公式](https://latex.codecogs.com/svg.latex?%5Cmathcal%7BL%7D%20%3D%20%5Cmathcal%7BL%7D_%7B%5Ctext%7Bdata%7D%7D%20&plus;%20%5Cmathcal%7BL%7D_%7B%5Ctext%7Bpde%7D%7D)

## 涉及的偏微分方程和边界条件

### 涉及的偏微分方程

#### 泊松方程

- **泊松方程**：
  
  ```math
  \nabla^2 u = f(x, y)
  ```

  其中：
  
  - `\nabla^2 u` 是 Laplace 算子作用于 `u`，在二维情况下表示为：
    
    ```math
    \nabla^2 u = \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}
    ```
  - `f(x, y)` 是源项函数，在代码中定义为 `f(x, y) = 2x + 2y`

### 设置的边界条件

- **Dirichlet 边界条件**：
  
  ```math
  u(x, y) = \frac{x^3 + y^3}{3} \quad \text{当} \ (x, y) \ \text{位于边界上时}
  ```

### 各个损失的具体表达式

#### PDE 残差损失

- **PDE 残差损失**：
  
  ```math
  \mathcal{L}_{\text{pde}} = \sum_{i} \left( \nabla^2 u_{net} - f(x_i, y_i) \right)^2
  ```

  其中：
  
  - `u_{net}` 是神经网络预测的解
  - `\nabla^2 u_{net}` 是通过自动微分计算得到的 Laplace 算子作用于 `u_{net}` 的结果
  - `f(x_i, y_i)` 是源项函数在点 `(x_i, y_i)` 处的值

#### 边界条件损失

- **边界条件损失**：
  
  ```math
  \mathcal{L}_{\text{bc}} = \sum_{i} \left( u_{net}(x_i, y_i) - \frac{x_i^3 + y_i^3}{3} \right)^2
  ```

  其中：
  
  - `u_{net}(x_i, y_i)` 是神经网络预测的解在边界点 `(x_i, y_i)` 处的值
  - `\frac{x_i^3 + y_i^3}{3}` 是边界条件中给定的精确解

### 总的损失函数

代码中的总损失函数是 PDE 残差损失和边界条件损失的加权和：

```math
\mathcal{L} = \mathcal{L}_{\text{pde}} + \lambda \cdot \mathcal{L}_{\text{bc}}
```

在代码中，通过 `loss_weights` 参数设置了权重系数，`loss_weights=[1, 2]` 表示 PDE 残差损失的权重为 1，边界条件损失的权重为 2。
