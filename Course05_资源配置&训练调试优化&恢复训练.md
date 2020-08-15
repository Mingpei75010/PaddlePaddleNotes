
## 前提条件

需要进行数据处理、设计神经网络结构，代码与上一节保持一致。略。


## 资源配置

可以通过资源配置的优化，提升模型训练效率。

### 单GPU训练

飞桨动态图通过``fluid.dygraph.guard(place=None)``里的``place``参数，设置在GPU上训练还是CPU上训练。

```
with fluid.dygraph.guard(place=fluid.CPUPlace())　#设置使用CPU资源训神经网络。
with fluid.dygraph.guard(place=fluid.CUDAPlace(0))　#设置使用GPU资源训神经网络，默认使用服务器的第一个GPU卡。"0"是GPU卡的编号，比如一台服务器有的四个GPU卡，编号分别为０、１、２、３。
```

### 分布式训练

在机器资源充沛的情况下，建议采用分布式训练，大部分模型的训练时间可压缩到小时级别。分布式训练有两种实现模式：**模型并行**和**数据并行**。

#### 模型并行

模型并行是将一个网络模型拆分为多份，拆分后的模型分到多个设备上（GPU）训练，每个设备的训练数据是相同的。模型并行的实现模式可以节省内存，但是应用较为受限，一般适用于如下两个场景：

1. **模型架构过大**： 完整的模型无法放入单个GPU。如2012年ImageNet大赛的冠军模型AlexNet是模型并行的典型案例，由于当时GPU内存较小，单个GPU不足以承担AlexNet，因此研究者将AlexNet拆分为两部分放到两个GPU上并行训练。

2. **网络模型的结构设计相对独立**： 当网络模型的设计结构可以并行化时，采用模型并行的方式。如在计算机视觉目标检测任务中，一些模型（如YOLO9000）的边界框回归和类别预测是独立的，可以将独立的部分放到不同的设备节点上完成分布式训练。

#### 数据并行

数据并行每次读取多份数据，读取到的数据输入给多个设备（GPU）上的模型，每个设备上的模型是完全相同的，*飞桨采用的就是这种方式*。当前GPU硬件技术快速发展，深度学习使用的主流GPU的内存已经足以满足大多数的网络模型需求，所以大多数情况下使用数据并行的方式。

值得注意的是，每个设备的模型是完全相同的，但是输入数据不同，因此**每个设备的模型计算出的梯度是不同的**。如果每个设备的梯度只更新当前设备的模型，就会导致下次训练时，每个模型的参数都不相同。因此我们还需要一个**梯度同步机制**，保证每个设备的梯度是完全相同的。

梯度同步有两种方式：PRC通信方式和NCCL2通信方式。

- **PRC通信方式**：PRC通信方式通常用于CPU分布式训练，它有两个节点：参数服务器Parameter server和训练节点Trainer。
  - Parameter server**收集**来自每个设备的梯度更新信息，并**计算出一个全局的梯度更新**。
  - Trainer用于**训练**，每个Trainer上的程序相同，但数据不同。当Parameter server收到来自Trainer的梯度更新请求时，统一更新模型的梯度。
- **NCCL2通信方式**：相比PRC通信方式，使用NCCL2（Collective通信方式）进行分布式训练，不需要启动Parameter server进程，每个Trainer进程保存一份完整的模型参数，在完成梯度计算之后通过**Trainer之间的相互通信**，Reduce梯度数据到所有节点的所有设备，然后每个节点再各自完成参数更新。



在启动训练前，需要配置如下参数：

- 从环境变量获取设备的ID，并指定给CUDAPlace。
```
  device_id = fluid.dygraph.parallel.Env().dev_id
  place = fluid.CUDAPlace(device_id)
```
- 对定义的网络做预处理，设置为并行模式。
```
  strategy = fluid.dygraph.parallel.prepare_context() ## 新增
  model = MNIST()
  model = fluid.dygraph.parallel.DataParallel(model, strategy)  ## 新增
```
- 定义多GPU训练的reader，不同ID的GPU加载不同的数据集。
```
  valid_loader = paddle.batch(paddle.dataset.mnist.test(), batch_size=16, drop_last=true)
  valid_loader = fluid.contrib.reader.distributed_batch_reader(valid_loader)
```
- 收集每批次训练数据的loss，并聚合参数的梯度。
```
  avg_loss = model.scale_loss(avg_loss)  ## 新增
  avg_loss.backward()
  mnist.apply_collective_grads()         ## 新增
```

启动多GPU的训练，还需要在命令行中设置一些参数变量。打开终端，运行如下命令：
```
$ python -m paddle.distributed.launch --selected_gpus=0,1,2,3 --log_dir ./mylog train_multi_gpu.py
```
- paddle.distributed.launch：启动分布式运行。
- selected_gpus：设置使用的GPU的序号（需要是多GPU卡的机器，通过命令watch nvidia-smi查看GPU的序号）。
- log_dir：存放训练的log，若不设置，每个GPU上的训练信息都会打印到屏幕。
- train_multi_gpu.py：多GPU训练的程序，包含修改过的train_multi_gpu()函数。


## 训练调试与优化

训练过程优化思路主要有如下五个关键环节：

1. **计算分类准确率，观测模型训练效果。**交叉熵损失函数只能作为优化目标，无法直接准确衡量模型的训练效果。准确率可以直接衡量训练效果，但由于其离散性质，不适合做为损失函数优化神经网络。

2. **检查模型训练过程，识别潜在问题。**如果模型的损失或者评估指标表现异常，通常需要打印模型每一层的输入和输出来定位问题，分析每一层的内容来获取错误的原因。

3. **加入校验或测试，更好评价模型效果。**理想的模型训练结果是在训练集和验证集上均有较高的准确率，如果训练集上的准确率高于验证集，说明网络训练程度不够；如果验证集的准确率高于训练集，可能是发生了**过拟合现象**。通过在优化目标中加入正则化项的办法，解决过拟合的问题。

4. **加入正则化项，避免模型过拟合。**飞桨框架支持为整体参数加入正则化项，这是通常的做法。此外，飞桨框架也支持为某一层或某一部分的网络单独加入正则化项，以达到精细调整参数训练的效果。

5. **可视化分析。**用户不仅可以通过打印或使用matplotlib库作图，飞桨还提供了更专业的**可视化分析工具VisualDL**，提供便捷的可视化分析方法。

### 计算模型的分类准确率

准确率是一个直观衡量分类模型效果的指标，由于这个指标是离散的，因此不适合作为损失函数来优化。

飞桨提供了计算分类准确率的API，使用``fluid.layers.accuracy``可以直接计算准确率，该API的输入参数input为预测的分类结果predict，输入参数label为数据真实的label。

在下述代码中，我们在模型前向计算过程forward函数中计算分类准确率，并在训练时打印每个批次样本的分类准确率。

```
# 定义模型结构
class MNIST(fluid.dygraph.Layer):
     def __init__(self):
         super(MNIST, self).__init__()
         
         # 定义一个卷积层，使用relu激活函数
         self.conv1 = Conv2D(num_channels=1, num_filters=20, filter_size=5, stride=1, padding=2, act='relu')
         # 定义一个池化层，池化核为2，步长为2，使用最大池化方式
         self.pool1 = Pool2D(pool_size=2, pool_stride=2, pool_type='max')
         # 定义一个卷积层，使用relu激活函数
         self.conv2 = Conv2D(num_channels=20, num_filters=20, filter_size=5, stride=1, padding=2, act='relu')
         # 定义一个池化层，池化核为2，步长为2，使用最大池化方式
         self.pool2 = Pool2D(pool_size=2, pool_stride=2, pool_type='max')
         # 定义一个全连接层，输出节点数为10 
         self.fc = Linear(input_dim=980, output_dim=10, act='softmax')
    # 定义网络的前向计算过程
     def forward(self, inputs, label):
         x = self.conv1(inputs)
         x = self.pool1(x)
         x = self.conv2(x)
         x = self.pool2(x)
         x = fluid.layers.reshape(x, [x.shape[0], 980])
         x = self.fc(x)
         if label is not None:
             acc = fluid.layers.accuracy(input=x, label=label)
             return x, acc
         else:
             return x
 ```

### 检查模型训练过程，识别潜在训练问题

在网络定义的``Forward``函数中，可以打印每一层输入输出的尺寸，以及每层网络的参数。通过查看这些信息，不仅可以更好地理解训练的执行过程，还可以发现潜在问题，或者启发继续优化的思路。

在下述程序中，使用``check_shape``变量控制是否打印“尺寸”，验证网络结构是否正确。使用``check_content``变量控制是否打印“内容值”，验证数据分布是否合理。假如在训练中发现中间层的部分输出持续为0，说明该部分的网络结构设计存在问题，没有充分利用。

```
# 加入对每一层输入和输出的尺寸和数据内容的打印，根据check参数决策是否打印每层的参数和输出尺寸
     def forward(self, inputs, label=None, check_shape=False, check_content=False):
         # 给不同层的输出不同命名，方便调试
         outputs1 = self.conv1(inputs)
         outputs2 = self.pool1(outputs1)
         outputs3 = self.conv2(outputs2)
         outputs4 = self.pool2(outputs3)
         _outputs4 = fluid.layers.reshape(outputs4, [outputs4.shape[0], -1])
         outputs5 = self.fc(_outputs4)
         
         # 选择是否打印神经网络每层的参数尺寸和输出尺寸，验证网络结构是否设置正确
         if check_shape:
             # 打印每层网络设置的超参数-卷积核尺寸，卷积步长，卷积padding，池化核尺寸
             print("\n########## print network layer's superparams ##############")
             print("conv1-- kernel_size:{}, padding:{}, stride:{}".format(self.conv1.weight.shape, self.conv1._padding, self.conv1._stride))
             print("conv2-- kernel_size:{}, padding:{}, stride:{}".format(self.conv2.weight.shape, self.conv2._padding, self.conv2._stride))
             print("pool1-- pool_type:{}, pool_size:{}, pool_stride:{}".format(self.pool1._pool_type, self.pool1._pool_size, self.pool1._pool_stride))
             print("pool2-- pool_type:{}, poo2_size:{}, pool_stride:{}".format(self.pool2._pool_type, self.pool2._pool_size, self.pool2._pool_stride))
             print("fc-- weight_size:{}, bias_size_{}, activation:{}".format(self.fc.weight.shape, self.fc.bias.shape, self.fc._act))
             
             # 打印每层的输出尺寸
             print("\n########## print shape of features of every layer ###############")
             print("inputs_shape: {}".format(inputs.shape))
             print("outputs1_shape: {}".format(outputs1.shape))
             print("outputs2_shape: {}".format(outputs2.shape))
             print("outputs3_shape: {}".format(outputs3.shape))
             print("outputs4_shape: {}".format(outputs4.shape))
             print("outputs5_shape: {}".format(outputs5.shape))
             
         # 选择是否打印训练过程中的参数和输出内容，可用于训练过程中的调试
         if check_content:
            # 打印卷积层的参数-卷积核权重，权重参数较多，此处只打印部分参数
             print("\n########## print convolution layer's kernel ###############")
             print("conv1 params -- kernel weights:", self.conv1.weight[0][0])
             print("conv2 params -- kernel weights:", self.conv2.weight[0][0])

             # 创建随机数，随机打印某一个通道的输出值
             idx1 = np.random.randint(0, outputs1.shape[1])
             idx2 = np.random.randint(0, outputs3.shape[1])
             # 打印卷积-池化后的结果，仅打印batch中第一个图像对应的特征
             print("\nThe {}th channel of conv1 layer: ".format(idx1), outputs1[0][idx1])
             print("The {}th channel of conv2 layer: ".format(idx2), outputs3[0][idx2])
             print("The output of last layer:", outputs5[0], '\n')
            
        # 如果label不是None，则计算分类精度并返回
         if label is not None:
             acc = fluid.layers.accuracy(input=outputs5, label=label)
             return outputs5, acc
         else:
             return outputs5
 ```

### 加入校验或测试，更好评价模型效果

为了验证模型的有效性，通常将样本集合分成三份，训练集、校验集和测试集。

- **训练集** ：用于训练模型的参数，即训练过程中主要完成的工作。
- **校验集** ：用于对模型超参数的选择，比如网络结构的调整、正则化项权重的选择等。
- **测试集** ：用于模拟模型在应用后的真实效果。因为测试集没有参与任何模型优化或参数训练的工作，所以它对模型来说是完全未知的样本。在不以校验数据优化网络结构或模型超参数时，校验数据和测试数据的效果是类似的，均更真实的反映模型效果。

### 加入正则化项，避免模型过拟合

#### 过拟合现象

在训练集上的损失小，在验证集或测试集上的损失较大，称为**过拟合**，过拟合表示模型过于敏感，学习到了训练数据中的一些误差，而这些误差并不是真实的泛化规律（可推广到测试集上的规律）。反之，如果模型在训练集和测试集上均损失较大，则称为**欠拟合**，欠拟合表示模型还不够强大，还没有很好的拟合已知的训练样本，更别提测试样本了。

#### 过拟合的成因与防控

造成过拟合的原因是模型过于敏感，而**训练数据量太少**或其中的**噪音太多**。

对于情况1，我们或者限制模型表示能力，或者收集更多的训练数据。对于情况2，我们使用数据清洗和修正来解决。

#### 正则化项

为了防止模型过拟合，在没有扩充样本量的可能下，只能**降低模型的复杂度**，可以通过限制参数的数量或可能取值（参数值尽量小）实现。

具体来说，在模型的优化目标（损失）中人为加入**对参数规模的惩罚项**。当参数越多或取值越大时，该惩罚项就越大。通过调整惩罚项的权重系数，可以使模型在“尽量减少训练损失”和“保持模型的泛化能力”之间取得平衡。泛化能力表示模型在没有见过的样本上依然有效。正则化项的存在，增加了模型在训练集上的损失。

飞桨支持为所有参数加上统一的正则化项，也支持为特定的参数添加正则化项。前者的实现如下代码所示，仅在优化器中设置``regularization``参数即可实现。使用参数``regularization_coeff``调节正则化项的权重，权重越大时，对模型复杂度的惩罚越高。

```
    optimizer = fluid.optimizer.AdamOptimizer(learning_rate=0.01, regularization=fluid.regularizer.L2Decay(regularization_coeff=0.1),parameter_list=model.parameters())
```

### 可视化分析

训练模型时，经常需要观察模型的评价指标，分析模型的优化过程，以确保训练是有效的。可选用这两种工具：Matplotlib库和VisualDL。

- **Matplotlib库**：Matplotlib库是Python中使用的最多的2D图形绘图库，它有一套完全仿照MATLAB的函数形式的绘图接口，使用轻量级的PLT库（Matplotlib）作图是非常简单的。
- **VisualDL**：如果期望使用更加专业的作图工具，可以尝试VisualDL，飞桨可视化分析工具。VisualDL能够有效地展示飞桨在运行过程中的计算图、各种指标变化趋势和数据信息。

#### 使用Matplotlib库绘制损失随训练下降的曲线图

将训练的批次编号作为X轴坐标，该批次的训练损失作为Y轴坐标。

1. 训练开始前，声明两个列表变量存储对应的批次编号(iters=[])和训练损失(losses=[])。
2. 随着训练的进行，将iter和losses两个列表填满。
3. 训练结束后，将两份数据以参数形式导入PLT的横纵坐标。
4. 最后，调用plt.plot()函数即可完成作图。

详细代码如下：
```
#引入matplotlib库
import matplotlib.pyplot as plt

with fluid.dygraph.guard(place):
    model = MNIST()
    model.train() 
    
    optimizer = fluid.optimizer.SGDOptimizer(learning_rate=0.01, parameter_list=model.parameters())
    
    EPOCH_NUM = 10
    iter=0
    iters=[]
    losses=[]
    for epoch_id in range(EPOCH_NUM):
        for batch_id, data in enumerate(train_loader()):
            #准备数据，变得更加简洁
            image_data, label_data = data
            image = fluid.dygraph.to_variable(image_data)
            label = fluid.dygraph.to_variable(label_data)
            
            #前向计算的过程，同时拿到模型输出值和分类准确率
            predict, acc = model(image, label)

            #计算损失，取一个批次样本损失的平均值
            loss = fluid.layers.cross_entropy(predict, label)
            avg_loss = fluid.layers.mean(loss)
            
            #每训练了100批次的数据，打印下当前Loss的情况
            if batch_id % 100 == 0:
                print("epoch: {}, batch: {}, loss is: {}, acc is {}".format(epoch_id, batch_id, avg_loss.numpy(), acc.numpy()))
                iters.append(iter)
                losses.append(avg_loss.numpy())
                iter = iter + 100

            #后向传播，更新参数的过程
            avg_loss.backward()
            optimizer.minimize(avg_loss)
            model.clear_gradients()

    #保存模型参数
    fluid.save_dygraph(model.state_dict(), 'mnist')
```

#### 使用VisualDL可视化分析

飞桨可视化分析工具，以丰富的图表呈现训练参数变化趋势、模型结构、数据样本、高维数据分布等。帮助用户清晰直观地理解深度学习模型训练过程及模型结构，进而实现高效的模型调优，具体代码实现如下。

```
# 安装VisualDL
!pip install --upgrade --pre visualdl
```

1. 引入VisualDL库，定义作图数据存储位置（供第3步使用），本案例的路径是“log”。
```
from visualdl import LogWriter
log_writer = LogWriter("./log")
```
2. 在训练过程中插入作图语句。当每100个batch训练完成后，将当前损失作为一个新增的数据点(iter和acc的映射对)存储到第一步设置的文件中。使用变量iter记录下已经训练的批次数，作为作图的X轴坐标。
```
with fluid.dygraph.guard():
    model = MNIST()
    model.train() 
    
    optimizer = fluid.optimizer.SGDOptimizer(learning_rate=0.01, parameter_list=model.parameters())
    
    EPOCH_NUM = 10
    iter = 0
    for epoch_id in range(EPOCH_NUM):
        for batch_id, data in enumerate(train_loader()):
            #准备数据，变得更加简洁
            image_data, label_data = data
            image = fluid.dygraph.to_variable(image_data)
            label = fluid.dygraph.to_variable(label_data)
            
            #前向计算的过程，同时拿到模型输出值和分类准确率
            predict, avg_acc = model(image, label)

            #计算损失，取一个批次样本损失的平均值
            loss = fluid.layers.cross_entropy(predict, label)
            avg_loss = fluid.layers.mean(loss)
            
            #每训练了100批次的数据，打印下当前Loss的情况
            if batch_id % 100 == 0:
                print("epoch: {}, batch: {}, loss is: {}, acc is {}".format(epoch_id, batch_id, avg_loss.numpy(), avg_acc.numpy()))
                log_writer.add_scalar(tag = 'acc', step = iter, value = avg_acc.numpy())
                log_writer.add_scalar(tag = 'loss', step = iter, value = avg_loss.numpy())
                iter = iter + 100

            #后向传播，更新参数的过程
            avg_loss.backward()
            optimizer.minimize(avg_loss)
            model.clear_gradients()

    #保存模型参数
    fluid.save_dygraph(model.state_dict(), 'mnist')
```

3. 命令行启动VisualDL。使用“visualdl --logdir [数据文件所在文件夹路径] 的命令启动VisualDL。在VisualDL启动后，命令行会打印出可用浏览器查阅图形结果的网址。
```
$ visualdl --logdir ./log --port 8080
```
4. 打开浏览器，查看作图结果。查阅的网址在第三步的启动命令后会打印出来（如[http://127.0.0.1:8080/](http://127.0.0.1:8080/)），将该网址输入浏览器地址栏刷新页面的效果如下图所示。除了右侧对数据点的作图外，左侧还有一个控制板，可以调整诸多作图的细节。

## 模型加载及恢复训练

在日常训练工作中，我们会遇到一些突发情况，导致训练过程主动或被动的中断。如果训练一个模型需要花费几天的时间，中断后从初始状态重新训练是不可接受的。

飞桨支持从上一次保存状态开始继续训练，只要我们随时保存训练过程中的模型状态，就不用从初始状态重新训练。下面介绍恢复训练的实现方法，依然使用手写数字识别的案例，网络定义的部分保持不变。

```
import os
import random
import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, Linear
import numpy as np
from PIL import Image

import gzip
import json

# 定义数据集读取器
def load_data(mode='train'):

    # 数据文件
    datafile = './work/mnist.json.gz'
    print('loading mnist dataset from {} ......'.format(datafile))
    data = json.load(gzip.open(datafile))
    train_set, val_set, eval_set = data

    # 数据集相关参数，图片高度IMG_ROWS, 图片宽度IMG_COLS
    IMG_ROWS = 28
    IMG_COLS = 28

    if mode == 'train':
        imgs = train_set[0]
        labels = train_set[1]
    elif mode == 'valid':
        imgs = val_set[0]
        labels = val_set[1]
    elif mode == 'eval':
        imgs = eval_set[0]
        labels = eval_set[1]

    imgs_length = len(imgs)

    assert len(imgs) == len(labels), \
          "length of train_imgs({}) should be the same as train_labels({})".format(
                  len(imgs), len(labels))
                  
    index_list = list(range(imgs_length))

    # 读入数据时用到的batchsize
    BATCHSIZE = 100

    # 定义数据生成器
    def data_generator():
        if mode == 'train':
            random.shuffle(index_list)
        imgs_list = []
        labels_list = []
        for i in index_list:
            img = np.reshape(imgs[i], [1, IMG_ROWS, IMG_COLS]).astype('float32')
            label = np.reshape(labels[i], [1]).astype('int64')
            imgs_list.append(img) 
            labels_list.append(label)
            if len(imgs_list) == BATCHSIZE:
                yield np.array(imgs_list), np.array(labels_list)
                imgs_list = []
                labels_list = []

        # 如果剩余数据的数目小于BATCHSIZE，
        # 则剩余数据一起构成一个大小为len(imgs_list)的mini-batch
        if len(imgs_list) > 0:
            yield np.array(imgs_list), np.array(labels_list)

    return data_generator

#调用加载数据的函数
train_loader = load_data('train')

# 定义模型结构
class MNIST(fluid.dygraph.Layer):
     def __init__(self):
         super(MNIST, self).__init__()
         # 定义一个卷积层，使用relu激活函数
         self.conv1 = Conv2D(num_channels=1, num_filters=20, filter_size=5, stride=1, padding=2, act='relu')
         # 定义一个池化层，池化核为2，步长为2，使用最大池化方式
         self.pool1 = Pool2D(pool_size=2, pool_stride=2, pool_type='max')
         # 定义一个卷积层，使用relu激活函数
         self.conv2 = Conv2D(num_channels=20, num_filters=20, filter_size=5, stride=1, padding=2, act='relu')
         # 定义一个池化层，池化核为2，步长为2，使用最大池化方式
         self.pool2 = Pool2D(pool_size=2, pool_stride=2, pool_type='max')
         # 定义一个全连接层，输出节点数为10 
         self.fc = Linear(input_dim=980, output_dim=10, act='softmax')
    # 定义网络的前向计算过程
     def forward(self, inputs, label):
         x = self.conv1(inputs)
         x = self.pool1(x)
         x = self.conv2(x)
         x = self.pool2(x)
         x = fluid.layers.reshape(x, [x.shape[0], 980])
         x = self.fc(x)
         if label is not None:
             acc = fluid.layers.accuracy(input=x, label=label)
             return x, acc
         else:
             return x
```


注意进行恢复训练的程序不仅要保存模型参数，还要保存优化器参数。这是因为某些优化器含有一些随着训练过程变换的参数，例如Adam, AdaGrad等优化器采用可变学习率的策略，随着训练进行会逐渐减少学习率。这些优化器的参数对于恢复训练至关重要。为了演示这个特性,下面训练程序使用Adam优化器，学习率以多项式曲线从0.01衰减到0.001（polynomial decay）。

```
#在使用GPU机器时，可以将use_gpu变量设置成True
use_gpu = False
place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()

with fluid.dygraph.guard(place):
    model = MNIST()
    model.train() 
    
    EPOCH_NUM = 5
    BATCH_SIZE = 100
    # 定义学习率，并加载优化器参数到模型中
    total_steps = (int(60000//BATCH_SIZE) + 1) * EPOCH_NUM
    lr = fluid.dygraph.PolynomialDecay(0.01, total_steps, 0.001)
    
    # 使用Adam优化器
    optimizer = fluid.optimizer.AdamOptimizer(learning_rate=lr, parameter_list=model.parameters())
    
    for epoch_id in range(EPOCH_NUM):
        for batch_id, data in enumerate(train_loader()):
            #准备数据，变得更加简洁
            image_data, label_data = data
            image = fluid.dygraph.to_variable(image_data)
            label = fluid.dygraph.to_variable(label_data)
            
            #前向计算的过程，同时拿到模型输出值和分类准确率
            predict, acc = model(image, label)
            avg_acc = fluid.layers.mean(acc)
            
            #计算损失，取一个批次样本损失的平均值
            loss = fluid.layers.cross_entropy(predict, label)
            avg_loss = fluid.layers.mean(loss)
            
            #每训练了200批次的数据，打印下当前Loss的情况
            if batch_id % 200 == 0:
                print("epoch: {}, batch: {}, loss is: {}, acc is {}".format(epoch_id, batch_id, avg_loss.numpy(),avg_acc.numpy()))
            
            #后向传播，更新参数的过程
            avg_loss.backward()
            optimizer.minimize(avg_loss)
            model.clear_gradients()
            
        # 保存模型参数和优化器的参数
        fluid.save_dygraph(model.state_dict(), './checkpoint/mnist_epoch{}'.format(epoch_id))
        fluid.save_dygraph(optimizer.state_dict(), './checkpoint/mnist_epoch{}'.format(epoch_id))
```


### 恢复训练

在上述训练代码中，我们训练了五轮（epoch）。在每轮结束时，均保存了模型参数和优化器相关的参数。

- 使用model.state_dict()获取模型参数。
- 使用optimizer.state_dict()获取优化器和学习率相关的参数。
- 调用fluid.save_dygraph()将参数保存到本地。

当加载模型时，如果模型参数文件和优化器参数文件是相同的，我们可以使用load_dygraph同时加载这两个文件，如下代码所示。
```
params_dict, opt_dict = fluid.load_dygraph(params_path)
```
如果模型参数文件和优化器参数文件的名字不同，需要调用两次load_dygraph分别获得模型参数和优化器参数。


恢复训练有如下两个要点：

- 保存模型时同时保存模型参数和优化器参数。
- 恢复参数时同时恢复模型参数和优化器参数。



如何判断模型是否准确的恢复训练呢？*校验其后训练的损失变化是否和不中断时的训练完全一致。*


下面的代码将展示恢复训练的过程，并验证恢复训练是否成功。其中，我们重新定义一个train_again()训练函数，加载模型参数并从第一个epoch开始训练，以便读者可以校验恢复训练后的损失变化。

```
params_path = "./checkpoint/mnist_epoch0"        
#在使用GPU机器时，可以将use_gpu变量设置成True
use_gpu = False
place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()

with fluid.dygraph.guard(place):
    # 加载模型参数到模型中
    params_dict, opt_dict = fluid.load_dygraph(params_path)
    model = MNIST()
    model.load_dict(params_dict)
    
    EPOCH_NUM = 5
    BATCH_SIZE = 100
    # 定义学习率，并加载优化器参数到模型中
    total_steps = (int(60000//BATCH_SIZE) + 1) * EPOCH_NUM
    lr = fluid.dygraph.PolynomialDecay(0.01, total_steps, 0.001)
    
    # 使用Adam优化器
    optimizer = fluid.optimizer.AdamOptimizer(learning_rate=lr, parameter_list=model.parameters())
    optimizer.set_dict(opt_dict)

    for epoch_id in range(1, EPOCH_NUM):
        for batch_id, data in enumerate(train_loader()):
            #准备数据，变得更加简洁
            image_data, label_data = data
            image = fluid.dygraph.to_variable(image_data)
            label = fluid.dygraph.to_variable(label_data)
            
            #前向计算的过程，同时拿到模型输出值和分类准确率
            predict, acc = model(image, label)
            avg_acc = fluid.layers.mean(acc)
            
            #计算损失，取一个批次样本损失的平均值
            loss = fluid.layers.cross_entropy(predict, label)
            avg_loss = fluid.layers.mean(loss)
            
            #每训练了200批次的数据，打印下当前Loss的情况
            if batch_id % 200 == 0:
                print("epoch: {}, batch: {}, loss is: {}, acc is {}".format(epoch_id, batch_id, avg_loss.numpy(),avg_acc.numpy()))
            
            #后向传播，更新参数的过程
            avg_loss.backward()
            optimizer.minimize(avg_loss)
            model.clear_gradients()
```




