# 手写文字识别 MNIST

数据处理与上个笔记中的保持一致，略。

## 逐环节优化 - 神经网络

经典的全连接神经网络来包含四层网络：输入层、两个隐含层和输出层。

- 输入层：将数据输入给神经网络。在该任务中，输入层的尺度为28×28的像素值。
- 隐含层：增加网络深度和复杂度，隐含层的节点数是可以调整的，节点数越多，神经网络表示能力越强，参数量也会增加。在该任务中，中间的两个隐含层为10×10的结构，通常隐含层会比输入层的尺寸小，以便对关键信息做抽象，激活函数使用常见的sigmoid函数。引入非线性激活函数sigmoid是为了**增加神经网络的非线性能力**。
- 输出层：输出网络计算结果，输出层的节点数是固定的。如果是回归问题，节点数量为需要回归的数字数量；如果是分类问题，则是分类标签的数量。在该任务中，模型的输出是回归一个数字，输出层的尺寸为1。

下述代码为经典全连接神经网络的实现。完成网络结构定义后，即可训练神经网络。

```
# 多层全连接神经网络实现
class MNIST(fluid.dygraph.Layer):
    def __init__(self):
        super(MNIST, self).__init__()
        # 定义两层全连接隐含层，输出维度是10，激活函数为sigmoid
        self.fc1 = Linear(input_dim=784, output_dim=10, act='sigmoid') # 隐含层节点为10，可根据任务调整
        self.fc2 = Linear(input_dim=10, output_dim=10, act='sigmoid')
        # 定义一层全连接输出层，输出维度是1，不使用激活函数
        self.fc3 = Linear(input_dim=10, output_dim=1, act=None)
    
    # 定义网络的前向计算
    def forward(self, inputs, label=None):
        inputs = fluid.layers.reshape(inputs, [inputs.shape[0], 784])
        outputs1 = self.fc1(inputs)
        outputs2 = self.fc2(outputs1)
        outputs_final = self.fc3(outputs2)
        return outputs_final
```

```
#网络结构部分之后的代码，保持不变
with fluid.dygraph.guard():
    model = MNIST()
    model.train()
    #调用加载数据的函数，获得MNIST训练数据集
    train_loader = load_data('train')
    # 使用SGD优化器，learning_rate设置为0.01
    optimizer = fluid.optimizer.SGDOptimizer(learning_rate=0.01, parameter_list=model.parameters())
    # 训练5轮
    EPOCH_NUM = 5
    for epoch_id in range(EPOCH_NUM):
        for batch_id, data in enumerate(train_loader()):
            #准备数据
            image_data, label_data = data
            image = fluid.dygraph.to_variable(image_data)
            label = fluid.dygraph.to_variable(label_data)
            
            #前向计算的过程
            predict = model(image)
            
            #计算损失，取一个批次样本损失的平均值
            loss = fluid.layers.square_error_cost(predict, label)
            avg_loss = fluid.layers.mean(loss)
            
            #每训练了200批次的数据，打印下当前Loss的情况
            if batch_id % 200 == 0:
                print("epoch: {}, batch: {}, loss is: {}".format(epoch_id, batch_id, avg_loss.numpy()))
            
            #后向传播，更新参数的过程
            avg_loss.backward()
            optimizer.minimize(avg_loss)
            model.clear_gradients()

    #保存模型参数
    fluid.save_dygraph(model.state_dict(), 'mnist')
```

### 卷积神经网络

对于计算机视觉问题，效果最好的模型仍然是卷积神经网络。卷积神经网络针**对视觉问题的特点进行了网络结构优化**，更适合处理视觉问题。卷积神经网络由多个**卷积层**和**池化层**组成，卷积层负责对输入进行扫描以**生成更抽象的特征表示**，池化层对这些特征表示进行过滤，**保留最关键的特征信息**。

两层卷积和池化的卷积神经网络实现如下所示。

```
# 多层卷积神经网络实现
class MNIST(fluid.dygraph.Layer):
     def __init__(self):
         super(MNIST, self).__init__()
         
         # 定义卷积层，输出特征通道num_filters设置为20，卷积核的大小filter_size为5，卷积步长stride=1，padding=2
         # 激活函数使用relu
         self.conv1 = Conv2D(num_channels=1, num_filters=20, filter_size=5, stride=1, padding=2, act='relu')
         # 定义池化层，池化核pool_size=2，池化步长为2，选择最大池化方式
         self.pool1 = Pool2D(pool_size=2, pool_stride=2, pool_type='max')
         # 定义卷积层，输出特征通道num_filters设置为20，卷积核的大小filter_size为5，卷积步长stride=1，padding=2
         self.conv2 = Conv2D(num_channels=20, num_filters=20, filter_size=5, stride=1, padding=2, act='relu')
         # 定义池化层，池化核pool_size=2，池化步长为2，选择最大池化方式
         self.pool2 = Pool2D(pool_size=2, pool_stride=2, pool_type='max')
         # 定义一层全连接层，输出维度是1，不使用激活函数
         self.fc = Linear(input_dim=980, output_dim=1, act=None)
         
    # 定义网络前向计算过程，卷积后紧接着使用池化层，最后使用全连接层计算最终输出
     def forward(self, inputs):
         x = self.conv1(inputs)
         x = self.pool1(x)
         x = self.conv2(x)
         x = self.pool2(x)
         x = fluid.layers.reshape(x, [x.shape[0], -1])
         x = self.fc(x)
         return x
 ```

## 分类任务的损失函数

若继续使用房价预测中的均方误差：手写识别的预测结果是标签值，并非数字，使用均方误差在量纲、逻辑上都说不过去。

### Softmax函数

将每一个值作自然指数(e^x)，除以所有值的自然指数的和。这样，每个输出的范围均在0~1之间，且所有输出之和等于1，可以被看作概率。对于二分类问题，使用两个输出接入softmax作为输出层，等价于使用单一输出接入Sigmoid函数。

### 交叉熵

在模型输出为分类标签的概率时，直接以标签和概率做比较也不够合理，人们更习惯使用**交叉熵误差**作为分类问题的损失衡量。

使得上述概率最大等价于最小化交叉熵，得到交叉熵的损失函数。

#### 交叉熵的代码实现

在手写数字识别任务中，仅改动三行代码，就可以将在现有模型的损失函数替换成交叉熵（cross_entropy）。

- 在读取数据部分，将标签的类型设置成``int``，体现它是一个标签而不是实数值（飞桨默认将标签处理成“int64”）。
- 在网络定义部分，将输出层改成“输出十个标签的概率”的模式。
- 在训练过程部分，将损失函数从均方误差换成交叉熵。

在数据处理部分，需要修改标签变量Label的格式，代码如下所示。

- 从：``label = np.reshape(labels[i], [1]).astype('float32')``
- 到：``label = np.reshape(labels[i], [1]).astype('int64')``

在网络定义部分，需要修改输出层结构，代码如下所示。

- 从：``self.fc = Linear(input_dim=980, output_dim=1, act=None)``
- 到：``self.fc = Linear(input_dim=980, output_dim=10, act='softmax')``

修改计算损失的函数，从均方误差（常用于回归问题）到交叉熵误差（常用于分类问题），代码如下所示。

- 从：``loss = fluid.layers.square_error_cost(predict, label)``
- 到：``loss = fluid.layers.cross_entropy(predict, label)``

由于我们修改了模型的输出格式，因此使用模型做预测时的代码也需要做相应的调整。从模型输出10个标签的概率中选择最大的，将其标签编号输出。

```
    lab = np.argsort(results.numpy())
    print("本次预测的数字是: ", lab[0][-1])
```

## 优化算法

### 设置学习率

在深度学习神经网络模型中，通常使用标准的随机梯度下降算法更新参数，学习率代表参数更新幅度的大小，即步长。学习率和深度学习任务类型有关，合适的学习率往往**需要大量的实验和调参经验**。探索学习率最优值时需要注意如下两点：

- 学习率不是越小越好。学习率越小，损失函数的变化速度越慢，意味着我们需要花费更长的时间进行收敛。
- 学习率不是越大越好。只根据总样本集中的一个批次计算梯度，抽样误差会导致计算出的梯度不是全局最优的方向，且存在波动。在接近最优解时，过大的学习率会导致参数在最优解附近震荡，损失难以收敛。

在训练时可以尝试调小或调大，通过观察Loss下降的情况判断合理的学习率。

### 学习率的主流优化算法

经过研究员的不断的实验，当前已经形成了四种比较成熟的优化算法：SGD、Momentum、AdaGrad和Adam。

- **SGD**： 随机梯度下降算法，每次训练少量数据，抽样偏差导致参数收敛过程中震荡。

- **Momentum**： 引入物理“动量”的概念，累积速度，减少震荡，使参数更新的方向更稳定。

- **AdaGrad**： 根据不同参数距离最优解的远近，动态调整学习率。学习率逐渐下降，依据各参数变化大小调整学习率。

- **Adam**： 由于动量和自适应学习率两个优化思路是正交的，因此可以将两个思路结合起来，这就是当前广泛应用的算法。































