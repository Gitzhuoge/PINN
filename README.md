# 涉及的偏微分方程

## 1、纳斯-斯托克斯方程

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

## 2、泊松方程

### 偏微分方程

- 二维泊松方程：

![泊松方程](https://latex.codecogs.com/svg.latex?%5Cnabla%5E2%20u%20%3D%202x%20&plus;%202y%20%5Cquad%20%5Ctext%7Bin%7D%20%5B0,1%5D%5E2)

- 精确解：
  
![精确解](https://latex.codecogs.com/svg.latex?u(x,y)%20%3D%20%5Cfrac%7B1%7D%7B3%7D(x%5E3%20&plus;%20y%5E3))

### 边界条件

- Dirichlet边界条件：
  
![边界条件](https://latex.codecogs.com/svg.latex?u%7C_%7B%5Cpartial%5COmega%7D%20%3D%20%5Cfrac%7B1%7D%7B3%7D(x%5E3%20&plus;%20y%5E3))

### 损失函数

1. **PDE残差**  
   ![PDE损失](https://latex.codecogs.com/svg.latex?%5Cmathcal%7BL%7D_%7Bpde%7D%20%3D%20%5Cmathbb%7BE%7D%5Cleft%5B%20(u_%7Bxx%7D%20&plus;%20u_%7Byy%7D%20-%202x%20-%202y)%5E2%20%5Cright%5D)

2. **边界残差**  
   ![BC损失](https://latex.codecogs.com/svg.latex?%5Cmathcal%7BL%7D_%7Bbc%7D%20%3D%20%5Cmathbb%7BE%7D%5Cleft%5B%20(u%20-%20%5Cfrac%7B1%7D%7B3%7D(x%5E3%20&plus;%20y%5E3))%5E2%20%5Cright%5D)


- 总损失：

![总损失](https://latex.codecogs.com/svg.latex?%5Cmathcal%7BL%7D%20%3D%20%5Cmathcal%7BL%7D_%7Bpde%7D%20&plus;%202%5Cmathcal%7BL%7D_%7Bbc%7D)

## Burger方程  
![Burger方程](https://latex.codecogs.com/svg.latex?\frac{\partial%20y}{\partial%20t}%20+%20y%20\frac{\partial%20y}{\partial%20x}%20=%20\frac{0.01}{\pi}%20\frac{\partial^2%20y}{\partial%20x^2})

求解带粘性项的 Burgers 方程，其中粘性系数为：

![粘性系数](https://latex.codecogs.com/svg.latex?\nu%20=%20\frac{0.01}{\pi})

**初始条件**：  
![初始条件](https://latex.codecogs.com/svg.latex?y(x,%200)%20=%20-\sin(\pi%20x))

**边界条件**：  
![边界条件](https://latex.codecogs.com/svg.latex?y(-1,%20t)%20=%20y(1,%20t)%20=%200)

可以直接构造输出的函数为：  
![输出函数](https://latex.codecogs.com/svg.latex?u(x,%20t)%20=%20-\sin(\pi%20x)%20+%20(1%20-%20x^2)\cdot%20t\cdot%20y(x,%20t))

这样函数本身满足边界和初始条件，只需要训练这整个函数（当然主要是训练 y(x,t)）使得其满足 burger 方程，最终整个方程即为满足需求的方程解。

**另外求解过程中，可以先在全局定义域里训练模型，再抽取定义域里模型预测精度比较低的点，再混合部分全局抽取的随机点，放入模型中训练，即对残差比较大区域自适应地加密训练**

