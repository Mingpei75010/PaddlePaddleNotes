# Boston房价预测服务

类似于机器学习领域的“Hello World”。数据集统计了**13种**可能影响房价的因素和该类型房屋的均价，尝试使用线性回归模型解决这个问题。

## 线性回归模型

$ y = x_i * w_i + b $

其中，$w_i$ 和 $b$ 分别表示线性模型的权重和偏置。损失函数采用均方误差（Mean Square Error）来衡量预测值和真实值的差异，采用均方误差可以保证Loss值处处可微。


## 数据处理

数据处理包括五部分：数据导入、数据形状变换、数据集划分、数据归一化处理和封装load data函数。

### 读入数据

使用```np.fromfile()```读取数据，读入的原始数据是一维的，所有数据连在一起，所以需要对数据的形状进行变换，形成一个二维矩阵，每行为一个数据样本（14个值）。

```
# 导入需要用到的package
import numpy as np
import json
# 读入训练数据
datafile = './work/housing.data'
data = np.fromfile(datafile, sep=' ')

# 读入之后的数据被转化成1维array，其中array的第0-13项是第一条数据，第14-27项是第二条数据，以此类推.... 
# 这里对原始数据做reshape，变成N x 14的形式
feature_names = [ 'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE','DIS', 
                 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV' ]
feature_num = len(feature_names)
data = data.reshape([data.shape[0] // feature_num, feature_num])
```

### 数据集划分

将数据集划分成训练集和测试集，其中训练集用于确定模型的参数，测试集用于评判模型的效果。我们期望模型学习的是任务的本质规律，而不是训练数据本身，模型训练未使用的数据，才能更真实的评估模型的效果。

```
ratio = 0.8
offset = int(data.shape[0] * ratio)
training_data = data[:offset]
training_data.shape
```

### 数据归一化处理

对每个特征进行归一化处理，使得每个特征的取值缩放到0~1之间。这样做有两个好处：一是模型训练更高效；二是特征前的权重大小可以代表该变量对预测结果的贡献度（因为每个特征值本身的范围相同）。

```
# 计算train数据集的最大值，最小值，平均值
maximums, minimums, avgs = \
                     training_data.max(axis=0), \
                     training_data.min(axis=0), \
     training_data.sum(axis=0) / training_data.shape[0]
# 对数据进行归一化处理
for i in range(feature_num):
    #print(maximums[i], minimums[i], avgs[i])
    data[:, i] = (data[:, i] - minimums[i]) / (maximums[i] - minimums[i])
```

### 封装成load_data函数

将上述几个数据处理操作封装成load_data函数，以便下一步模型的调用。


## 模型设计

模型设计是深度学习模型关键要素之一，也称为**网络结构设计**，相当于模型的假设空间，即实现模型“前向计算”（从输入到输出）的过程。

采用向量表示输入特征与输出预测值，则权重也是13*1的向量。采用任意数字赋值权重做初始化：

```
w = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, -0.1, -0.2, -0.3, -0.4, 0.0]
w = np.array(w).reshape([13, 1])
```
此外，还需要初始化偏置b，则线性回归模型的完整输出是 $y = w*x + b$，这个从特征和参数计算输出值的过程称为“**前向计算**”，可以通过一个forward函数完成上述计算过程。

```
class Network(object):
    def __init__(self, num_of_weights):
        # 随机产生w的初始值
        # 为了保持程序每次运行结果的一致性，
        # 此处设置固定的随机数种子
        np.random.seed(0)
        self.w = np.random.randn(num_of_weights, 1)
        self.b = 0.
        
    def forward(self, x):
        z = np.dot(x, self.w) + self.b
        return z
```


## 训练配置

模型设计完成后，需要通过训练配置寻找模型的最优值，即**通过损失函数来衡量模型的好坏**。训练配置也是深度学习模型关键要素之一。对于回归问题，最常采用的衡量方法是使用均方误差作为评价模型好坏的指标，具体定义如下：

$Loss = (y - z)^2$

在Network类中添加损失函数的计算过程：
```
    def loss(self, z, y):
        error = z - y
        cost = error * error
        cost = np.mean(cost)
        return cost
```

在均方误差下使用梯度下降法，相比于绝对值误差有两个好处：

- 曲线是可导的

- 越接近最低点，曲线斜率（的绝对值）越小，有助于根据梯度的变化来判断与最低点的接近程度，并适时减小部长以免错过最低点。


## 训练过程

接下来介绍如何求解参数 w 和 b 的数值，这个过程也称为模型训练过程。训练过程是深度学习模型的关键要素之一，其目标是**让定义的损失函数Loss尽可能的小**，也就是说找到一个参数解 w 和 b 使得损失函数取得极小值。

### 梯度下降法

求解Loss函数最小值可以这样实现：从当前的参数取值，一步步的按照下坡的方向下降，直到走到最低点。

对每一个样本，都可以根据公式计算出每个 $w_i$。公式是通用的，因此可以使用一个for循环计算出该样本对应的所有权重的梯度。或使用numpy的广播功能代替for循环。

在404个样本，13个权重的情况下，计算得到的梯度是404*13的矩阵，**总梯度是这404个样本对梯度贡献的平均值**。计算梯度的代码如下所示：

```
z = net.forward(x)
gradient_w = (z - y) * x
gradient_w = np.mean(gradient_w, axis=0)
gradient_w = gradient_w[:, np.newaxis]
```
其中，gradient_w 的形状是 (13,)，而 w 的形状是 (13,1)，因此需要更改一下 gradient_w 的形状。写成 Network 类的 gradient 函数：

```
    def gradient(self, x, y):
        z = self.forward(x)
        gradient_w = (z-y)*x
        gradient_w = np.mean(gradient_w, axis=0)
        gradient_w = gradient_w[:, np.newaxis]
        gradient_b = (z - y)
        gradient_b = np.mean(gradient_b)
        
        return gradient_w, gradient_b
```

### 确定损失函数 

只考虑 w_5 和 w_9 两个参数的情况下，使用以下代码确定下一步的点：
```
# 在[w5, w9]平面上，沿着梯度的反方向移动到下一个点P1
# 定义移动步长 eta
eta = 0.1
# 更新参数w5和w9
net.w[5] = net.w[5] - eta * gradient_w5
net.w[9] = net.w[9] - eta * gradient_w9
```
- 相减：参数需要向梯度的反方向移动。
- eta：控制每次参数值沿着梯度反方向变动的大小，即每次移动的步长，又称为学习率。

以上步骤封装为update函数：
```
    def update(self, graident_w5, gradient_w9, eta=0.01):
        net.w[5] = net.w[5] - eta * gradient_w5
        net.w[9] = net.w[9] - eta * gradient_w9
```

### 训练扩展到全部参数

由于不再限定参与计算的参数（上一步只有 w_5 和 w_9），修改之后的代码反而更加简洁。实现逻辑：“**前向计算输出、根据输出和真实值计算Loss、基于Loss和输入计算梯度、根据梯度更新参数值**”四个部分反复执行，直到到达参数最优点。

```
    def update(self, gradient_w, gradient_b, eta = 0.01):
        self.w = self.w - eta * gradient_w
        self.b = self.b - eta * gradient_b

    def train(self, x, y, iterations=100, eta=0.01):
        losses = []
        for i in range(iterations):
            z = self.forward(x)
            L = self.loss(z, y)
            gradient_w, gradient_b = self.gradient(x, y)
            self.update(gradient_w, gradient_b, eta)
            losses.append(L)
            if (i+1) % 10 == 0:
                print('iter {}, loss {}'.format(i, L))
        return losses
```

### 随机梯度下降法（Stochastic Gradient Descent, SGD）

在上述程序中，每次损失函数和梯度计算都是基于数据集中的全量数据，但在实际问题中，数据集往往非常大，如果每次都使用全量数据进行计算，效率非常低。

由于参数每次只沿着梯度反方向更新一点点，因此方向并不需要那么精确。一个合理的解决方案是**每次从总的数据集中随机抽取出小部分数据来代表整体**，基于这部分数据计算梯度和损失来更新参数，这种方法被称作随机梯度下降法，核心概念如下：

- min-batch：每次迭代时抽取出来的一批数据被称为一个min-batch。
- batch_size：一个mini-batch所包含的样本数目称为batch_size。
- epoch：当程序迭代的时候，按mini-batch逐渐抽取出样本，当把整个数据集都遍历到了的时候，则完成了一轮训练，也叫一个epoch。

此外，为了实现随机抽样的效果，我们先将train_data里面的样本顺序随机打乱，然后再抽取mini_batch。随机打乱样本顺序，需要用到np.random.shuffle函数。对于二维数组来说，数组的元素在第0维被随机打乱，但第1维的顺序保持不变。将这部分集成到Network类的train函数当中：

```
# 获取数据
train_data, test_data = load_data()

# 打乱样本顺序
np.random.shuffle(train_data)

# 将train_data分成多个mini_batch
batch_size = 10
n = len(train_data)
mini_batches = [train_data[k:k+batch_size] for k in range(0, n, batch_size)]

# 创建网络
net = Network(13)

# 依次使用每个mini_batch的数据
for mini_batch in mini_batches:
    x = mini_batch[:, :-1]
    y = mini_batch[:, -1:]
    loss = net.train(x, y, iterations=1)
```

将每个随机抽取的mini-batch数据输入到模型中用于参数训练。训练过程的核心是两层循环：
1. 第一层循环，代表样本集合要被训练遍历几次，称为“epoch”

1. 第二层循环，代表每次遍历时，样本集合被拆分成的多个批次，需要全部执行训练，称为“iter (iteration)”

集成到train函数中，最终Network类如下：

```
class Network(object):
    def __init__(self, num_of_weights):
        # 随机产生w的初始值
        # 为了保持程序每次运行结果的一致性，此处设置固定的随机数种子
        #np.random.seed(0)
        self.w = np.random.randn(num_of_weights, 1)
        self.b = 0.
        
    def forward(self, x):
        z = np.dot(x, self.w) + self.b
        return z
    
    def loss(self, z, y):
        error = z - y
        num_samples = error.shape[0]
        cost = error * error
        cost = np.sum(cost) / num_samples
        return cost
    
    def gradient(self, x, y):
        z = self.forward(x)
        N = x.shape[0]
        gradient_w = 1. / N * np.sum((z-y) * x, axis=0)
        gradient_w = gradient_w[:, np.newaxis]
        gradient_b = 1. / N * np.sum(z-y)
        return gradient_w, gradient_b
    
    def update(self, gradient_w, gradient_b, eta = 0.01):
        self.w = self.w - eta * gradient_w
        self.b = self.b - eta * gradient_b
            
                
    def train(self, training_data, num_epoches, batch_size=10, eta=0.01):
        n = len(training_data)
        losses = []
        for epoch_id in range(num_epoches):
            # 在每轮迭代开始之前，将训练数据的顺序随机打乱
            # 然后再按每次取batch_size条数据的方式取出
            np.random.shuffle(training_data)
            # 将训练数据进行拆分，每个mini_batch包含batch_size条的数据
            mini_batches = [training_data[k:k+batch_size] for k in range(0, n, batch_size)]
            for iter_id, mini_batch in enumerate(mini_batches):
                #print(self.w.shape)
                #print(self.b)
                x = mini_batch[:, :-1]
                y = mini_batch[:, -1:]
                a = self.forward(x)
                loss = self.loss(a, y)
                gradient_w, gradient_b = self.gradient(x, y)
                self.update(gradient_w, gradient_b, eta)
                losses.append(loss)
                print('Epoch {:3d} / iter {:3d}, loss = {:.4f}'.
                                 format(epoch_id, iter_id, loss))
        
        return losses
```


##  总结

使用神经网络建模房价预测有三个要点：

- 构建网络，初始化参数w和b，定义预测和损失函数的计算方法。

- 随机选择初始点，建立梯度的计算方法和参数更新方式。

- 从总的数据集中抽取部分数据作为一个mini_batch，计算梯度并更新参数，不断迭代直到损失函数几乎不再下降。
