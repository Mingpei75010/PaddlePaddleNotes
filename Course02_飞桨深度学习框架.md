# 深度学习框架

## 优势

1. **节省编写大量底层代码的精力**：屏蔽底层实现，用户只需关注模型的逻辑结构。

1. **省去了部署和适配环境的烦恼**：具备灵活的移植性，可将代码部署到CPU/GPU/移动端上，选择具有分布式性能的深度学习工具会使模型训练更高效。

## 设计思路

深度学习框架的本质是框架**自动实现建模过程中相对通用的模块**，建模者只实现模型个性化的部分。

- **通用部分**：网络模块（layer、Variable）、Loss函数、优化算法等。

- **个性化部分**：设计网络结构、指定Loss函数、指定优化算法等。

## PaddlePaddle深度学习平台

- **开发便捷的深度学习框架**：支持声明式、命令式编程，兼具开发灵活、高性能；网络结构自动设计，模型效果超越人类专家。

- **超大规模深度学习模型训练技术**：千亿特征、万亿参数、数百节点的开源大规模训练平台；万亿规模参数模型实时更新。

- **多端多平台部署的高性能推理引擎**：兼容多种开源框架训练的模型，不同架构的平台设备轻松部署推理速度全面领先。

- **产业级开源模型库**：开源100+算法和200+训练模型，包括国际竞赛冠军模型；快速助力产业应用。

## 联系方式

- 官方网站： [https://www.paddlepaddle.org.cn/](https://www.paddlepaddle.org.cn/)
- GitHub： [https://github.com/paddlepaddle](https://github.com/paddlepaddle)
- 微信公众号： 飞桨PaddlePaddle
- 官方QQ群： 703252161


# 使用飞桨构建Boston房价预测模型

## 加载相关库

```
import paddle
import paddle.fluid as fluid
import paddle.fluid.dygraph as dygraph
from paddle.fluid.dygraph import Linear
import numpy as np
import os
import random
```
- paddle/fluid：飞桨的主库，目前大部分的**实用函数**均在paddle.fluid包内。
- dygraph：**动态图**的类库。
- Linear：神经网络的**全连接层函数**，即包含所有输入权重相加和激活函数的基本神经元结构。在房价预测任务中，使用只有一层的神经网络（全连接层）来实现线性回归模型。

两种深度学习建模编写方式的区别：
- **静态图模式**（声明式编程范式，类比C++）：先编译后执行的方式，**性能更好并便于部署**。
- **动态图模式**（命令式编程范式，类比Python）：解析式的执行方式，每写一行网络代码即可同时获得计算结果，**更方便调试**。


## 数据处理

与使用Python构建时的代码相同。
```
def load_data():
    # 从文件导入数据
    datafile = './work/housing.data'
    data = np.fromfile(datafile, sep=' ')

    # 每条数据包括14项，其中前面13项是影响因素，第14项是相应的房屋价格中位数
    feature_names = [ 'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', \
                      'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV' ]
    feature_num = len(feature_names)

    # 将原始数据进行Reshape，变成[N, 14]这样的形状
    data = data.reshape([data.shape[0] // feature_num, feature_num])

    # 将原数据集拆分成训练集和测试集
    # 这里使用80%的数据做训练，20%的数据做测试
    # 测试集和训练集必须是没有交集的
    ratio = 0.8
    offset = int(data.shape[0] * ratio)
    training_data = data[:offset]

    # 计算train数据集的最大值，最小值，平均值
    maximums, minimums, avgs = training_data.max(axis=0), training_data.min(axis=0), \
                                 training_data.sum(axis=0) / training_data.shape[0]
    
    # 记录数据的归一化参数，在预测时对数据做归一化
    global max_values
    global min_values
    global avg_values
    max_values = maximums
    min_values = minimums
    avg_values = avgs

    # 对数据进行归一化处理
    for i in range(feature_num):
        #print(maximums[i], minimums[i], avgs[i])
        data[:, i] = (data[:, i] - avgs[i]) / (maximums[i] - minimums[i])

    # 训练集和测试集的划分比例
    #ratio = 0.8
    #offset = int(data.shape[0] * ratio)
    training_data = data[:offset]
    test_data = data[offset:]
    return training_data, test_data
```

## 模型设计

模型定义的实质是定义线性回归的网络结构。通过创建Python类的方式完成模型网络的定义，即定义``init``函数和``forward``函数。

- **定义init函数**：在类的初始化函数中声明每一层网络的实现函数。在房价预测模型中，只需要定义一层全连接层。
- **定义forward函数**：构建神经网络结构，实现前向计算过程，并返回预测结果，在本任务中返回的是房价预测结果。

```
class Regressor(fluid.dygraph.Layer):
    def __init__(self):
        super(Regressor, self).__init__()
        
        # 定义一层全连接层，输出维度是1，激活函数为None，即不使用激活函数
        self.fc = Linear(input_dim=13, output_dim=1, act=None)
    
    # 网络的前向计算函数
    def forward(self, inputs):
        x = self.fc(inputs)
        return x
```

## 训练配置

训练配置包括四步：指定运行训练的机器资源、声明模型实例、加载训练数据和测试数据、设置优化算法和学习率。

```
# 定义飞桨动态图的工作环境
with fluid.dygraph.guard():
    # 声明定义好的线性回归模型
    model = Regressor()
    # 开启模型训练模式
    model.train()
    # 加载数据
    training_data, test_data = load_data()
    # 定义优化算法，这里使用随机梯度下降-SGD
    # 学习率设置为0.01
    opt = fluid.optimizer.SGD(learning_rate=0.01, parameter_list=model.parameters())
```

说明：
1. 以``guard``函数**指定运行训练的机器资源**，表明在``with``作用域下的程序均执行在本机的CPU资源上。``dygraph.guard``表示在``with``作用域下的程序会以飞桨**动态图的模式**执行（实时执行）。
1. 声明定义好的回归模型Regressor实例，并将模型的状态设置为**训练**``model.train()``。此外，模型实例还有预测状态``model.eval()``，此状态下只需要执行正向计算，没有反向传播梯度的过程。
1. ``fluid.optimizer.SGD()``设置**优化算法**和**学习率**，优化算法采用随机梯度下降SGD，学习率设置为0.01。
1. 上述代码局长奶``with``创建的``fluid.dygraph.guard()``上下文环境中进行，可以理解为``with fluid.dygraph.guard()``创建了飞桨动态图的工作环境，在该环境下完成模型声明、数据转换及模型训练等操作。


## 训练过程

两层循环嵌套方式
- **内层循环：负责整个数据集的一次遍历**，采用分批次方式（batch）。假设数据集样本数量为1000，一个批次有10个样本，则遍历一次数据集的批次数量是1000/10=100，即内层循环需要执行100次。batch的取值会影响模型训练效果：batch过大，会增大内存消耗和计算时间，且效果并不会明显提升；batch过小，每个batch的样本数据将没有统计意义。由于房价预测模型的训练数据集较小，我们将batch为设置10。
- **外层循环：定义遍历数据集的次数**，通过参数EPOCH_NUM设置。

每个内层循环都要执行以下四个步骤：
1. **数据准备**：将一个批次的数据转变成np.array和内置格式。
1. **前向计算**：将一个批次的样本数据灌入网络中，计算输出结果。
1. **计算损失函数**：以前向计算结果和真实房价作为输入，通过损失函数``square_error_cost``计算出损失函数值（Loss）。
1. **反向传播**：执行梯度反向传播``backward``函数，即从后到前逐层计算每一层的梯度，并根据设置的优化算法更新参数``opt.minimize``。

```
with dygraph.guard(fluid.CPUPlace()):
    EPOCH_NUM = 1000   # 设置外层循环次数
    BATCH_SIZE = 10  # 设置batch大小
    
    # 定义外层循环
    for epoch_id in range(EPOCH_NUM):
        # 在每轮迭代开始之前，将训练数据的顺序随机的打乱
        np.random.shuffle(training_data)
        # 将训练数据进行拆分，每个batch包含10条数据
        mini_batches = [training_data[k:k+BATCH_SIZE] for k in range(0, len(training_data), BATCH_SIZE)]
        # 定义内层循环
        for iter_id, mini_batch in enumerate(mini_batches):
            x = np.array(mini_batch[:, :-1]).astype('float32') # 获得当前批次训练数据
            y = np.array(mini_batch[:, -1:]).astype('float32') # 获得当前批次训练标签（真实房价）
            # 将numpy数据转为飞桨动态图variable形式
            house_features = dygraph.to_variable(x)
            prices = dygraph.to_variable(y)
            
            # 前向计算
            predicts = model(house_features)
            
            # 计算损失
            loss = fluid.layers.square_error_cost(predicts, label=prices)
            avg_loss = fluid.layers.mean(loss)
            if iter_id%20==0:
                print("epoch: {}, iter: {}, loss is: {}".format(epoch_id, iter_id, avg_loss.numpy()))
            
            # 反向传播
            avg_loss.backward()
            # 最小化loss,更新参数
            opt.minimize(avg_loss)
            # 清除梯度
            model.clear_gradients()
```

## 保存并测试模型
### 保存模型
将模型当前的参数数据``model.state_dict()``保存到文件中（通过参数指定保存的文件名 LR_model），以备预测或校验的程序调用，代码如下所示。

```
# 定义飞桨动态图工作环境
with fluid.dygraph.guard():
    # 保存模型参数，文件名为LR_model
    fluid.save_dygraph(model.state_dict(), 'LR_model')
```

### 测试模型
测试过程和在应用场景中使用模型的过程一致，主要可分成如下三个步骤：
1. **配置模型预测的机器资源**。本案例默认使用本机，因此无需写代码指定。
1. **将训练好的模型参数加载到模型实例中**。由两个语句完成，第一句是从文件中读取模型参数；第二句是将参数内容加载到模型。加载完毕后，需要将模型的状态调整为``eval()``。
1. **将待预测的样本特征输入到模型中**，打印输出的预测结果。

通过``load_one_example``函数实现从数据集中抽一条样本作为测试样本。

```
def load_one_example(data_dir):
    f = open(data_dir, 'r')
    datas = f.readlines()
    # 选择倒数第10条数据用于测试
    tmp = datas[-10]
    tmp = tmp.strip().split()
    one_data = [float(v) for v in tmp]

    # 对数据进行归一化处理
    for i in range(len(one_data)-1):
        one_data[i] = (one_data[i] - avg_values[i]) / (max_values[i] - min_values[i])

    data = np.reshape(np.array(one_data[:-1]), [1, -1]).astype(np.float32)
    label = one_data[-1]
    return data, label
    
with dygraph.guard():
    # 参数为保存模型参数的文件地址
    model_dict, _ = fluid.load_dygraph('LR_model')
    model.load_dict(model_dict)
    model.eval()

    # 参数为数据集的文件地址
    test_data, label = load_one_example('./work/housing.data')
    # 将数据转为动态图的variable格式
    test_data = dygraph.to_variable(test_data)
    results = model(test_data)

    # 对结果做反归一化处理
    results = results * (max_values[-1] - min_values[-1]) + avg_values[-1]
    print("Inference result is {}, the corresponding label is {}".format(results.numpy(), label))
```


